"""SANAD RF qualification v31: frozen v30, new Sionna seeds, independent-geometry controls.

Research only. Zero RF actuation. No training, selection, threshold tuning or rescoring
on prospective results. Each seed creates seven independent array objects, though
arrays within a channel seed are correlated and not seven channel realizations.
"""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
from statistics import median
import numpy as np
from geometry_config import N_ARRAYS, N_ELEMENTS

SEEDS = tuple(range(600, 744))  # 144 channel seeds => 1008 healthy array-cases
PERIODS = 300
LAST40 = 260
METHODS = ('v11', 'v29', 'v30')
HEALTHY_STREAM = 'healthy_only'
INJECTED_STREAM = 'injected_two_faults'
PROTOCOL = 'V31_FROZEN_V30_PROSPECTIVE_1008_HEALTHY_144_CHANNEL_SEEDS'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                prefix=path.stem+'.', suffix='.tmp', dir=path.parent, delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp=f.name
    os.replace(tmp,path)


def validate_covariance(cov):
    cov=np.asarray(cov)
    if cov.shape != (PERIODS,N_ARRAYS,N_ELEMENTS,N_ELEMENTS) or not np.iscomplexobj(cov):
        raise ValueError(f'Covariance must be complex [300,7,8,8]; got {cov.shape} {cov.dtype}')
    if not np.isfinite(cov).all(): raise ValueError('Nonfinite covariance values')
    if not np.allclose(cov, np.conjugate(np.swapaxes(cov,-1,-2)), rtol=1e-4,atol=1e-7):
        raise ValueError('Covariance is not Hermitian')
    return cov


def _eval_stream(vectors, faults, mode, *, cal_dir):
    from sanad_v30 import SANADNetworkV30
    detector=SANADNetworkV30(cal_dir=cal_dir)   # resets EACH independent draw
    flags={m:np.zeros((PERIODS,7),bool) for m in METHODS}
    named={m:np.full((PERIODS,7),-1,int) for m in ('v11','v30')}
    reviews=np.zeros((PERIODS,7),bool)
    for t in range(PERIODS):
        batch={i:vectors[i][t] for i in range(7) if vectors[i][t] is not None}
        d=detector.update(batch)
        for i,x in enumerate(d):
            if x.action != 'NO_HARDWARE_ACTUATION':raise RuntimeError('Unexpected RF actuation')
            flags['v11'][t,i]=x.advisory_v11
            flags['v29'][t,i]=x.advisory_v11 or x.v29_candidate
            flags['v30'][t,i]=x.advisory_v11 or x.v30_candidate
            reviews[t,i]=x.independent_review
            if x.antenna_1based is not None:
                named['v30'][t,i]=int(x.antenna_1based)
            # v30 preserves v11 antenna when v11 advisory; only store in that case
            if x.advisory_v11 and x.antenna_1based is not None:
                named['v11'][t,i]=int(x.antenna_1based)
    rows=[]
    for a in range(7):
        fault=faults.get(a)
        onset=int(fault['start']) if fault else None
        expected=int(fault['ant'])+1 if fault else None
        last40={m:bool(flags[m][LAST40:,a].any()) for m in METHODS}
        post_onset={}
        delays={}
        for m in METHODS:
            if fault:
                hit=np.flatnonzero(flags[m][onset:,a])
                post_onset[m]=bool(len(hit))
                delays[m]=int(hit[0]) if len(hit) else None
            else:
                post_onset[m]=None;delays[m]=None
        rows.append(dict(mode=mode,array=a,faulty=bool(fault),
            fault_onset_report=onset,expected_antenna_1based=expected,
            last40_alarm=last40,post_onset_detected=post_onset,
            delay_reports=delays,
            review_last40=bool(reviews[LAST40:,a].any()),
            any_alarm_after_learning={m:bool(flags[m][120:,a].any()) for m in METHODS},
            v11_terminal_antenna=int(named['v11'][-1,a]) if named['v11'][-1,a]>0 else None,
            v30_terminal_antenna=int(named['v30'][-1,a]) if named['v30'][-1,a]>0 else None,
            v11_terminal_correct=bool(fault and named['v11'][-1,a]==expected),
            v30_terminal_correct=bool(fault and named['v30'][-1,a]==expected),
        ))
    return rows


def evaluate_file(npz_path, cal_dir, seed, *, frozen_hash=None):
    """Primary 7 healthy arrays, plus two faults and five unaffected arrays per seed.

    Healthy and injected streams use the SAME physical covariance for matched
    analysis, but different missingness/glitch randomness. The data are simulated.
    """
    import sys
    cal_dir=Path(cal_dir).resolve()
    if str(cal_dir) not in sys.path:sys.path.insert(0,str(cal_dir))
    from glr_compare import pipeline
    npz_path=Path(npz_path)
    with np.load(npz_path,allow_pickle=False) as z:
        cov=validate_covariance(z[f's{seed}'])
    v_healthy, healthy_faults=pipeline(cov,900000+seed,with_faults=False)
    assert len(healthy_faults)==0
    healthy=_eval_stream(v_healthy, {}, HEALTHY_STREAM, cal_dir=cal_dir)
    v_faulty, faulty_faults=pipeline(cov,1000000+seed,with_faults=True)
    assert len(faulty_faults)==2
    injected=_eval_stream(v_faulty,faulty_faults,INJECTED_STREAM,cal_dir=cal_dir)
    return dict(protocol=PROTOCOL,seed=seed,covariance_sha256=sha256(npz_path),
                frozen_source_sha256=frozen_hash,healthy=healthy,injected=injected,
                simulation_not_field=True)


def wilson(k,n,z=1.959963984540054):
    if n<=0:return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    r=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [round(max(0,c-r),6),round(min(1,c+r),6)]


def _method_metrics(rows, method, flag):
    n=len(rows);k=sum(bool(r[flag][method]) for r in rows)
    return dict(k=k,n=n,pct=round(100*k/n,3) if n else None,ci95_wilson=wilson(k,n))


def aggregate_run_files(result_dir, required_seeds=SEEDS, *, expected_frozen_hash=None,
                        bootstrap_repetitions=1000):
    result_dir=Path(result_dir)
    by_seed={}
    for sd in required_seeds:
        path=result_dir/f's{sd}.json'
        if not path.exists():continue
        d=json.loads(path.read_text())
        if d.get('protocol')!=PROTOCOL or d.get('seed')!=sd:
            raise ValueError(f'Protocol or seed mismatch: {path}')
        if expected_frozen_hash and d.get('frozen_source_sha256')!=expected_frozen_hash:
            raise ValueError(f'Frozen detector code changed: {path}')
        if len(d['healthy'])!=7 or len(d['injected'])!=7 or sum(r['faulty'] for r in d['injected'])!=2:
            raise ValueError(f'Invalid case count: {path}')
        by_seed[sd]=d
    control=[r for d in by_seed.values() for r in d['healthy']]
    injected=[r for d in by_seed.values() for r in d['injected']]
    faulty=[r for r in injected if r['faulty']]
    unaffected=[r for r in injected if not r['faulty']]
    result={}
    for method in METHODS:
        result[method]={
          'primary_healthy_false_alarm':_method_metrics(control,method,'last40_alarm'),
          'primary_healthy_any_after_learning':_method_metrics(control,method,'any_alarm_after_learning'),
          'fault_recall_last40':_method_metrics(faulty,method,'last40_alarm'),
          'fault_recall_any_post_onset':_method_metrics(faulty,method,'post_onset_detected'),
          'unaffected_neighbor_false_alarm':_method_metrics(unaffected,method,'last40_alarm'),
          'delay_hours_median_any_post_onset':(
             round(float(median([r['delay_reports'][method] for r in faulty
                if r['delay_reports'][method] is not None]))*.25,3)
             if any(r['delay_reports'][method] is not None for r in faulty) else None),
        }
    for method in ('v11','v30'):
        result[method]['terminal_correct_antenna_all_faults']={
            'k':sum(r[f'{method}_terminal_correct'] for r in faulty),'n':len(faulty)}
        al=[r for r in faulty if r['last40_alarm'][method]]
        result[method]['terminal_correct_antenna_when_alerted']={
            'k':sum(r[f'{method}_terminal_correct'] for r in al),'n':len(al)}

    # Seed-cluster bootstrap: resample channel seeds, not arrays/draws.
    # CI reflects ONLY between-seed variability in this simulation distribution.
    if len(by_seed)>=2 and bootstrap_repetitions:
        ids=list(by_seed)
        rng=np.random.default_rng(44171)
        for method in METHODS:
            c=np.asarray([sum(r['last40_alarm'][method] for r in by_seed[s]['healthy']) for s in ids])
            f=np.asarray([sum(r['last40_alarm'][method] for r in by_seed[s]['injected'] if r['faulty']) for s in ids])
            pick=rng.integers(0,len(ids),size=(bootstrap_repetitions,len(ids)))
            rate=(c[pick].sum(axis=1)/(7*len(ids)))
            recall=(f[pick].sum(axis=1)/(2*len(ids)))
            result[method]['cluster_bootstrap_95_interval']={
                'primary_false_alarm_rate':np.quantile(rate,[.025,.975]).round(6).tolist(),
                'fault_recall':np.quantile(recall,[.025,.975]).round(6).tolist()}

    n=len(by_seed)
    # Pre-registered prospective acceptance. A point estimate alone is insufficient.
    v30=result['v30']
    v11=result['v11']
    if n == len(required_seeds):
        upper_wilson=v30['primary_healthy_false_alarm']['ci95_wilson'][1]
        upper_cluster=v30['cluster_bootstrap_95_interval']['primary_false_alarm_rate'][1] if \
            'cluster_bootstrap_95_interval' in v30 else None
        lower_delta=None
        if bootstrap_repetitions and len(by_seed)>=2:
            ids=list(by_seed)
            arr30=np.asarray([sum(r['last40_alarm']['v30'] for r in by_seed[s]['injected'] if r['faulty']) for s in ids])
            arr11=np.asarray([sum(r['last40_alarm']['v11'] for r in by_seed[s]['injected'] if r['faulty']) for s in ids])
            rng=np.random.default_rng(44172)
            picks=rng.integers(0,len(ids),size=(bootstrap_repetitions,len(ids)))
            lower_delta=float(np.quantile((arr30-arr11)[picks].sum(axis=1)/(2*len(ids)),.025))
        gate=bool(upper_wilson<.02 and upper_cluster is not None and upper_cluster<.02
                  and lower_delta is not None and lower_delta>0)
        acceptance=dict(gate_pass=gate,
          rules='primary healthy false-alert Wilson AND seed-cluster bootstrap upper bound <2%; paired seed-bootstrap lower recall gain over v11 > 0',
          wilson_upper=upper_wilson,cluster_upper=upper_cluster,
          paired_recall_gain_lower_95=round(lower_delta,6) if lower_delta is not None else None)
    else:
        acceptance=dict(gate_pass=None,reason='Incomplete: no prospective claims until all 144 seeds complete')
    output=dict(status='COMPLETE' if n==len(required_seeds) else 'INCOMPLETE',
                protocol=PROTOCOL,completed_seed_count=n,required_seed_count=len(required_seeds),
                completed_seeds=sorted(by_seed),
                dataset=dict(healthy_only_array_cases=len(control),faulty_array_cases=len(faulty),
                             unaffected_arrays_under_injection=len(unaffected),
                             independent_channel_seeds=n),
                acceptance=acceptance,
                frozen_parameters=dict(v30_score=0.7687096042171566,
                    dominance_ratio=1.5,same_antenna_persistence='8 of 12'),
                methods=result,
                limitations=['Sionna simulation, not equipment/field observations',
                  'Array-cases are correlated within each channel seed',
                  'Wilson CI ignores clustering; cluster bootstrap by channel seed also provided',
                  'New-seed data must not be used to tune frozen thresholds',
                  'Historical scoring counts any alarm in reports 260..299; onset-delay separate',
                  'v30 additional candidates remain research review, never an RF hardware command'])
    return output


def write_summary(result_dir, *, frozen_hash=None):
    result_dir=Path(result_dir)
    s=aggregate_run_files(result_dir,expected_frozen_hash=frozen_hash)
    atomic_json(result_dir/'summary.json',s)
    return s
