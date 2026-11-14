#!/usr/bin/env python3
"""ESPARGOS measured CSI A/B: v0.1 covariance vs v0.2 spectral fusion.

No tuning on walk/test data. Uses only pre-onset, separately held-out static
segments for calibration. Faults are injected mathematically into measured CSI
(not observed hardware faults). Saves source hashes, detailed trials, plots.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from sanad_adaptive import AdaptiveCore, DetectorConfig
from sanad_adaptive.spectral import SpectralPhaseMonitor, SpectralConfig, transform_recording
from run_espargos_adaptive import URLS, NAMES, PROFILES, download, sha256


def read_csi(path: Path, max_records=None):
    p=Path(path)
    if p.suffix == '.npz':
        with np.load(p,allow_pickle=False) as f:
            h=np.asarray(f['csi'])
        if h.ndim!=4 or not np.iscomplexobj(h):raise ValueError('NPZ must contain complex CSI [T,N,M,K]')
        return h[:max_records] if max_records else h
    import tensorflow as tf
    out=[]
    for i,rec in enumerate(tf.data.TFRecordDataset(str(p))):
        if max_records is not None and i>=max_records:break
        parsed=tf.io.parse_single_example(rec,{'csi':tf.io.FixedLenFeature([],tf.string)})
        h=tf.io.parse_tensor(parsed['csi'],out_type=tf.complex64).numpy()
        if h.shape!=(4,2,4,117):raise ValueError(f'Unexpected ESPARGOS-0001 shape: {h.shape}')
        out.append(h.reshape(4,8,117))
    if not out:raise ValueError('No CSI samples')
    return np.stack(out)


def covariance_windows(csi: np.ndarray,window: int) -> np.ndarray:
    if csi.ndim!=4:raise ValueError('Expected CSI [T,N,M,K]')
    n=len(csi)//window
    if n<80:raise ValueError('Need >=80 windows')
    h=csi[:n*window].reshape(n,window,*csi.shape[1:])
    # Element covariance: average over OFDM tones and records. No
    # cross-array covariance or assumed cross-array phase synchronization.
    return np.einsum('wtamk,wtank->wamn',h,h.conj(),optimize=True) / (window*csi.shape[-1])


def inject(mat: np.ndarray, element:int, gain_db:float,phase_deg:float)->np.ndarray:
    m=mat.shape[0]
    diag=np.ones(m,complex)
    diag[element]=10**(gain_db/20.)*np.exp(1j*np.deg2rad(phase_deg))
    if mat.ndim==2:
        return diag[:,None]*mat*diag[None,:].conj()
    if mat.ndim==3:
        return diag[:,None,None]*mat*diag[None,:,None].conj()
    raise ValueError('Must be covariance or per-tone spectral coherence')


def _step(core, spec, cov, spectral, names):
    x=core.evaluate({name:cov[i] for i,name in enumerate(names)})['decisions']
    y=spec.evaluate({name:spectral[i] for i,name in enumerate(names)})['decisions']
    old=np.stack([x[name]['alerts'] for name in names])
    ph=np.stack([y[name]['phase_alerts'] for name in names])
    old_shift=np.asarray([x[name]['channel_shift'] for name in names])
    spec_shift=np.asarray([y[name]['channel_shift'] for name in names])
    # Research fusion: do NOT coerce channel ambiguity to healthy.
    review=old_shift|spec_shift
    alerts=(old|ph)&~review[:,None]
    return old,ph,alerts,review,spec_shift


def evaluate_sequence(core,spec,cov,spectral,names):
    core.reset_history();spec.reset_history()
    old=[];ph=[];fused=[];review=[];shift=[]
    for t in range(len(cov)):
        v=_step(core,spec,cov[t],spectral[t],names)
        for row,arr in zip((old,ph,fused,review,shift),v):row.append(arr)
    return tuple(map(np.stack,(old,ph,fused,review,shift)))


def _summary(result):
    old,ph,alerts,review,spectral_shift=result
    return {
        'array_windows':int(np.prod(alerts.shape[:2])),
        'v01_array_alarm_count':int(old.any(-1).sum()),
        'spectral_only_array_alarm_count':int(ph.any(-1).sum()),
        'v02_fused_array_alarm_count':int(alerts.any(-1).sum()),
        'v02_fused_array_alarm_fraction':float(alerts.any(-1).mean()),
        'v02_review_array_windows':int(review.sum()),
        'v02_review_fraction':float(review.mean()),
        'spectral_channel_review_windows':int(spectral_shift.sum()),
    }


def run(static_csi,walking_csi,out_dir,*,window=16,source_hashes=None,
        measured_source=True,profiles=PROFILES):
    if static_csi.shape[1:]!=walking_csi.shape[1:]:
        raise ValueError('Source hardware schemas differ')
    if static_csi.shape[1] < 1 or static_csi.shape[2] < 3:
        raise ValueError('At least one array with >=3 elements required for localization')
    if not np.iscomplexobj(static_csi) or not np.iscomplexobj(walking_csi):
        raise ValueError('Complex CSI required')
    ids=NAMES if static_csi.shape[1]==len(NAMES) else tuple(f'array_{i}' for i in range(static_csi.shape[1]))
    s_cov=covariance_windows(static_csi,window)
    w_cov=covariance_windows(walking_csi,window)
    s_spec=transform_recording(static_csi,window)
    w_spec=transform_recording(walking_csi,window)
    n=len(s_cov)
    ref_end=int(.30*n);cal_end=int(.60*n)
    if min(ref_end,cal_end-ref_end,n-cal_end)<24:
        raise ValueError('Requires >=24 independent-window slots in reference, calibration and test')
    core=AdaptiveCore(DetectorConfig(min_reference_frames=24,min_calibration_frames=16))
    phase=SpectralPhaseMonitor(SpectralConfig(min_calibration_windows=24))
    core.register_many({name:s_cov[:ref_end,j] for j,name in enumerate(ids)},
                       {name:s_cov[ref_end:cal_end,j] for j,name in enumerate(ids)})
    phase.register_many({name:s_spec[:ref_end,j] for j,name in enumerate(ids)},
                        {name:s_spec[ref_end:cal_end,j] for j,name in enumerate(ids)})
    held_cov=s_cov[cal_end:];held_spec=s_spec[cal_end:]
    static=evaluate_sequence(core,phase,held_cov,held_spec,ids)
    walking=evaluate_sequence(core,phase,w_cov,w_spec,ids)
    onset=max(7,len(held_cov)//3)
    baseline_v01=static[0];baseline_fused=static[2]
    trials=[]
    for a,name in enumerate(ids):
        for ant in range(held_cov.shape[2]):
            for gd,pd in profiles:
                core.reset_history();phase.reset_history()
                out_old=[];out_fused=[];out_phase=[];out_review=[]
                for t in range(len(held_cov)):
                    cs=inject(held_cov[t,a],ant,gd,pd) if t>=onset else held_cov[t,a]
                    ss=inject(held_spec[t,a],ant,gd,pd) if t>=onset else held_spec[t,a]
                    # Physically distinct arrays are calibrated independently.
                    # For injected array, no other array can change under this test.
                    old,ph,fused,review,_=_step(core,phase,cs[None],ss[None],(name,))
                    out_old.append(old[0]);out_phase.append(ph[0]);out_fused.append(fused[0]);out_review.append(review[0])
                old=np.stack(out_old)[onset:];ph=np.stack(out_phase)[onset:];fu=np.stack(out_fused)[onset:]
                ref_old=baseline_v01[onset:,a];ref_fused=baseline_fused[onset:,a]
                new_old=old[:,ant]&~ref_old[:,ant]
                new_ph=ph[:,ant] # phase-only component, no v0.1 path
                new_fused=fu[:,ant]&~ref_fused[:,ant]
                other=np.ones(held_cov.shape[2],bool);other[ant]=False
                trials.append({
                    'array':name,'element':int(ant),'gain_db':float(gd),'phase_deg':float(pd),
                    'v01_incremental_detected_any':bool(new_old.any()),
                    'v02_spectral_phase_detected_any':bool(new_ph.any()),
                    'v02_fused_incremental_detected_any':bool(new_fused.any()),
                    'v02_new_neighbor_alarms':int(np.sum(fu[:,other] & ~ref_fused[:,other])),
                    'v02_review_windows':int(np.sum(out_review[onset:])),
                    'first_v02_window_from_onset':int(np.flatnonzero(new_fused)[0]) if new_fused.any() else None})
    profiles_out=[]
    for gd,pd in profiles:
        sel=[x for x in trials if (x['gain_db'],x['phase_deg'])==(gd,pd)]
        profiles_out.append({'gain_db':float(gd),'phase_deg':float(pd),'trials':len(sel),
                             'v01_incremental_detected_fraction':float(np.mean([x['v01_incremental_detected_any'] for x in sel])),
                             'v02_spectral_detected_fraction':float(np.mean([x['v02_spectral_phase_detected_any'] for x in sel])),
                             'v02_fused_incremental_detected_fraction':float(np.mean([x['v02_fused_incremental_detected_any'] for x in sel])),
                             'new_neighbor_alarms':int(sum(x['v02_new_neighbor_alarms'] for x in sel))})
    result={'status':'EXPERIMENTAL_SPECTRAL_V02_NOT_RC3_NOT_FIELD_QUALIFIED',
      'measured_source':bool(measured_source),'sources_sha256':source_hashes or {},
      'protocol':{'arrays':len(ids),'antennas_per_array':int(s_cov.shape[2]),
       'subcarriers':int(static_csi.shape[-1]),'window_records':window,
       'reference_windows':[0,ref_end],'calibration_windows':[ref_end,cal_end],
       'static_heldout_windows':[cal_end,n],'onset_window_heldout':onset,
       'calibration_uses_walking_or_fault_labels':False},
      'static':_summary(static),'walking':_summary(walking),
      'injections':{'total_trials':len(trials),'profiles':profiles_out},
      'phase_thresholds_deg':{key:phase.baselines[key]['phase_threshold_deg'].tolist() for key in ids},
      'channel_thresholds_spectral':{key:phase.baselines[key]['channel_threshold'] for key in ids},
      'limitations':['Spectral v0.2 is a research prototype, not frozen RC3.',
         'No verified RF fault labels; mathematically injected D at measured CSI is not a physical fault.',
         'Static and walking windows from the same sessions are correlated; not independent prospective field validation.',
         'Phase and channel change can be non-identifiable without stable pilots/external reference.',
         'All thresholds selected using static calibration only; model is not trained across devices.',
         'Reported detection is any incremental alert after onset, not per-window classification accuracy.',
         'Channel-review periods are abstentions; do not count them as correct negatives.',
         'Requires within-array per-tone complex CSI with stable antenna indexing.'],
      'research_only':True}
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'adaptive_v02_report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/'adaptive_v02_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=trials[0].keys());w.writeheader();w.writerows(trials)
    try:
        import matplotlib.pyplot as plt
        labels=['v0.1 (covariance)','v0.2 (spectral + covariance)']
        x=np.arange(len(profiles_out))
        fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
        axs[0].bar(x-.2,[x['v01_incremental_detected_fraction'] for x in profiles_out],.4,label=labels[0])
        axs[0].bar(x+.2,[x['v02_fused_incremental_detected_fraction'] for x in profiles_out],.4,label=labels[1])
        axs[0].set_ylim(0,1);axs[0].set_ylabel('Incremental trial-level detection fraction')
        axs[0].set_xticks(x,[f"{p['gain_db']} dB + {p['phase_deg']}°" for p in profiles_out],rotation=15)
        axs[0].legend()
        axs[1].bar(['Static v0.2','Walking v0.1','Walking v0.2'],[
            result['static']['v02_fused_array_alarm_fraction'],
            float(walking[0].any(-1).mean()),result['walking']['v02_fused_array_alarm_fraction']])
        axs[1].set_ylabel('Array-window alarm fraction (descriptive)')
        fig.savefig(out/'adaptive_v02_comparison.png',dpi=140)
        plt.close(fig)
    except ImportError:pass
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data')
    p.add_argument('--output',default='results_v02')
    p.add_argument('--static',default=None)
    p.add_argument('--walking',default=None)
    p.add_argument('--download',action='store_true')
    p.add_argument('--window',type=int,default=16)
    p.add_argument('--max-records',type=int,default=None)
    args=p.parse_args()
    root=Path(args.data_dir)
    src1=Path(args.static) if args.static else root/'static_middle.tfrecord'
    src2=Path(args.walking) if args.walking else root/'walking_nearby.tfrecord'
    if args.download:
        download(URLS['static_middle'],src1);download(URLS['walking_nearby'],src2)
    if not src1.exists() or not src2.exists():
        raise FileNotFoundError('Provide TFRecord files or use --download')
    r=run(read_csi(src1,args.max_records),read_csi(src2,args.max_records),args.output,
          window=args.window, source_hashes={'static':sha256(src1),'walking':sha256(src2)},
          measured_source=src1.suffix!='.npz' and src2.suffix!='.npz')
    print(json.dumps({x:r[x] for x in ('status','static','walking','injections')},indent=2))

if __name__=='__main__':main()
