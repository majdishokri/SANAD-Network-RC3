#!/usr/bin/env python3
"""Experimental SANAD Adaptive test on real ESPARGOS TFRecords.

Unknown real-hardware calibration faults are NOT labeled. Controlled D R D^H
injections are simulated faults applied to MEASURED covariance, not field faults.
Does NOT run the frozen 7x8 RC3 classifier or inherit its 95.14% Sionna score.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from sanad_adaptive import AdaptiveCore, DetectorConfig, covariance_from_csi

DATA_PAGE = 'https://espargos.net/datasets/data/espargos-0001/'
URLS = {
    'static_middle': 'https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi%3A10.18419%2Fdarus-4352%2F10',
    'walking_nearby': 'https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi%3A10.18419%2Fdarus-4352%2F11',
}
NAMES=('North','South','West','East')
PROFILES=((0.3,3.),(0.5,5.),(0.8,8.),(0.5,0.),(0.,5.))


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def download(url,destination):
    import requests
    destination=Path(destination)
    if destination.exists() and destination.stat().st_size>1_000_000:
        return destination
    tmp=destination.with_suffix(destination.suffix+'.part')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with requests.get(url,headers={'User-Agent':'SANAD-Adaptive-Research/0.1'},
                      timeout=(30,180),stream=True) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for c in r.iter_content(1048576):
                if c:f.write(c)
    if tmp.stat().st_size<1_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError('Dataset download unexpectedly short')
    tmp.replace(destination)
    return destination


def read_cov(path,max_records=None):
    path=Path(path)
    if path.suffix=='.npz':
        # ONLY for developer fixtures or precomputed covariances. Record truth
        # requires metadata; do not label this as measured CSI automatically.
        with np.load(path,allow_pickle=False) as f:
            out=np.array(f['cov'])
        if max_records is not None:out=out[:max_records]
        if out.ndim!=4 or out.shape[-1]!=out.shape[-2]:
            raise ValueError('NPZ must contain [T,N,M,M] covariance')
        return out
    import tensorflow as tf
    fields={'csi':tf.io.FixedLenFeature([],tf.string)}
    rows=[]
    for i,record in enumerate(tf.data.TFRecordDataset(str(path))):
        if max_records is not None and i>=max_records:break
        example=tf.io.parse_single_example(record,fields)
        h=tf.io.parse_tensor(example['csi'],out_type=tf.complex64).numpy()
        if h.shape!=(4,2,4,117):
            raise ValueError(f'Unexpected published ESPARGOS CSI shape {h.shape}')
        # Convert each of the four phase-synchronized groups independently.
        hh=h.reshape(4,8,117)
        R=(hh@np.swapaxes(hh.conj(),-1,-2))/117.0
        rows.append(R.astype(np.complex128))
    if not rows:
        raise ValueError('No TFRecords')
    return np.stack(rows)


def windowed(cov,window):
    t,n,m,k=cov.shape
    if m!=k or n==0:
        raise ValueError('Invalid covariance array')
    bins=t//window
    if bins<80:
        raise ValueError('Need >=80 covariance windows for reference, calibration, holdout')
    return np.stack([cov[i*window:(i+1)*window].mean(axis=0) for i in range(bins)])


def injection(R,j,gain_db,phase_deg):
    m=R.shape[0]
    d=np.ones(m,complex)
    d[j]=10**(gain_db/20)*np.exp(1j*np.deg2rad(phase_deg))
    return d[:,None]*R*d[None,:].conj()


def evaluate_sequence(engine,covs, names=NAMES, *, reset=True):
    if reset:engine.reset_history()
    t,n,m,_=covs.shape
    if len(names)!=n:raise ValueError('Array names do not match')
    score=np.zeros((t,n,m),float)
    alarms=np.zeros((t,n,m),bool)
    drift=np.zeros((t,n),bool)
    for i in range(t):
        obj=engine.evaluate({name:covs[i,j] for j,name in enumerate(names)})
        for j,name in enumerate(names):
            dec=obj['decisions'][name]
            score[i,j]=dec['score']
            alarms[i,j]=dec['alerts']
            drift[i,j]=dec['channel_shift']
    return score,alarms,drift


def run(static_cov, walking_cov, report_dir, *, window=16,
        source_static='unknown', source_walk='unknown', measured_source=True):
    if static_cov.shape[1:]!=(4,8,8) or walking_cov.shape[1:]!=(4,8,8):
        raise ValueError('ESPARGOS-0001 requires [T,4,8,8]. AdaptiveCore itself is shape-general.')
    static=windowed(static_cov,window)
    walking=windowed(walking_cov,window)
    n=len(static)
    ref_end=int(n*.30);cal_end=int(n*.60)
    if min(ref_end,cal_end-ref_end,n-cal_end)<16:
        raise ValueError('Not enough windows for separated fit, calibration and holdout')
    core=AdaptiveCore(DetectorConfig(min_reference_frames=24,min_calibration_frames=16))
    ids=NAMES
    core.register_many({name:static[:ref_end,j] for j,name in enumerate(ids)},
                       {name:static[ref_end:cal_end,j] for j,name in enumerate(ids)})
    heldout=static[cal_end:]
    held_scores,held_alarms,held_drifts=evaluate_sequence(core,heldout)
    walk_scores,walk_alarms,walk_drifts=evaluate_sequence(core,walking)
    target_base=held_alarms
    onset=max(7,len(heldout)//3)
    trials=[]
    for a,name in enumerate(ids):
        for ant in range(8):
            for gd,pd in PROFILES:
                core.reset_history()
                trial_alerts=np.zeros((len(heldout),8),bool)
                trial_drift=np.zeros(len(heldout),bool)
                for i in range(len(heldout)):
                    mat=injection(heldout[i,a],ant,gd,pd) if i>=onset else heldout[i,a]
                    dec=core.evaluate({name:mat})['decisions'][name]
                    trial_alerts[i]=dec['alerts']
                    trial_drift[i]=dec['channel_shift']
                after=trial_alerts[onset:]
                reference=target_base[onset:,a]
                incremental=after[:,ant]&~reference[:,ant]
                other=np.ones(8,bool);other[ant]=False
                trials.append({'array':name,'antenna_index':ant,'gain_db':gd,'phase_deg':pd,
                               'detected_any':bool(after[:,ant].any()),
                               'incremental_detected_any':bool(incremental.any()),
                               'channel_shift_windows':int(trial_drift[onset:].sum()),
                               'new_neighbor_alarms':int(np.sum(after[:,other] & ~reference[:,other])),
                               'first_incremental_window':int(np.flatnonzero(incremental)[0]) if incremental.any() else None})
    profiles=[]
    for gd,pd in PROFILES:
        subset=[t for t in trials if (t['gain_db'],t['phase_deg'])==(gd,pd)]
        profiles.append({'gain_db':gd,'phase_deg':pd,'trials':len(subset),
                         'incremental_detection_fraction':float(np.mean([t['incremental_detected_any'] for t in subset])),
                         'new_neighbor_alarms':int(sum(t['new_neighbor_alarms'] for t in subset)),
                         'channel_shift_windows':int(sum(t['channel_shift_windows'] for t in subset))})
    def summary(alarms,drifts):
        return {'array_windows':int(np.prod(alarms.shape[:2])),
                'array_window_alarm_count':int(alarms.any(-1).sum()),
                'array_window_alarm_fraction':float(alarms.any(-1).mean()),
                'element_window_alarm_count':int(alarms.sum()),
                'channel_review_array_windows':int(drifts.sum()),
                'channel_review_array_window_fraction':float(drifts.mean())}
    report={
        'status':'EXPERIMENTAL_DIMENSION_ADAPTIVE_CORE_NOT_FROZEN_RC3',
        'measured_source':bool(measured_source),
        'source': DATA_PAGE,'source_sha256':{'static':source_static,'walking':source_walk},
        'input_schema': '[T,4,8,8] measured covariance for ESPARGOS-0001',
        'core_supports_variable_shapes': True,
        'window_records':window,'reference_windows':[0,ref_end],
        'calibration_windows':[ref_end,cal_end],
        'heldout_static_windows':[cal_end,n],
        'thresholds':{name:{'gain_db':core.baselines[name].gain_threshold.tolist(),
                            'phase_deg':core.baselines[name].phase_threshold.tolist(),
                            'channel':core.baselines[name].channel_threshold} for name in ids},
        'static':summary(held_alarms,held_drifts),
        'walking':summary(walk_alarms,walk_drifts),
        'injections':{'onset_window_in_heldout':onset,'total_trials':len(trials),'profiles':profiles},
        'limitations':['This is NOT the 7x8 frozen RC3 learned detector.',
            'Digital injections in measured covariance are not observed physical hardware faults.',
            'Normal operation of real antenna hardware is assumed, not independently established.',
            'Record windows are short CSI intervals, not 15-minute network operation.',
            'Calibration only uses the beginning of static_middle; correlated windows are not independent trials.',
            'Channel-shift abstention avoids a forced classification, not a fault-free guarantee.'],
        'research_only':True,
    }
    output=Path(report_dir);output.mkdir(parents=True,exist_ok=True)
    (output/'adaptive_espargos_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (output/'adaptive_espargos_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=trials[0].keys());w.writeheader();w.writerows(trials)
    try:
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,1,figsize=(13,7),layout='constrained')
        for ax,data,drift,title in zip(axs,[held_scores,walk_scores],[held_drifts,walk_drifts],
                                       ['Static held-out', 'Walking nearby']):
            for j,name in enumerate(ids):ax.plot(data[:,j].max(-1),label=name,linewidth=1)
            ax.axhline(1,color='black',linestyle='--',label='candidate score threshold')
            ax.set_title(title+' (research core; channel-shift windows may abstain)')
            ax.set_ylabel('Normalized candidate score')
            ax.legend(ncol=5,fontsize=9)
        axs[-1].set_xlabel(f'{window}-CSI-record window index')
        fig.savefig(output/'adaptive_espargos_scores.png',dpi=145)
        plt.close(fig)
    except ImportError:pass
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='./data')
    p.add_argument('--output',default='./results')
    p.add_argument('--static',default=None)
    p.add_argument('--walking',default=None)
    p.add_argument('--download',action='store_true')
    p.add_argument('--window',type=int,default=16)
    p.add_argument('--max-records',type=int,default=None)
    args=p.parse_args()
    folder=Path(args.data_dir)
    static=Path(args.static) if args.static else folder/'static_middle.tfrecord'
    walk=Path(args.walking) if args.walking else folder/'walking_nearby.tfrecord'
    if args.download:
        download(URLS['static_middle'],static)
        download(URLS['walking_nearby'],walk)
    if not static.exists() or not walk.exists():
        raise FileNotFoundError('Provide ESPARGOS TFRecords or use --download')
    result=run(read_cov(static,args.max_records), read_cov(walk,args.max_records),
               args.output,window=args.window,source_static=sha256(static),
               source_walk=sha256(walk),measured_source=static.suffix!='.npz' and walk.suffix!='.npz')
    print(json.dumps({k:result[k] for k in ['status','static','walking','injections']},indent=2))

if __name__=='__main__':main()
