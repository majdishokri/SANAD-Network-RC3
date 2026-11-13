"""Causal v35 extraction and replay; no injected truth reaches inference."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from glr_compare import pipeline
from qualification import validate_covariance
from v32_replay import evidence_stream
from temporal_v35 import extract_delta
from qa_integrity import validate_trace
from sanad_v33 import ElementEvidence
from sanad_v35 import EvidenceFusionV35,FusionConfigV35


def extract_seed_v35(npz_path,cal_dir,sd,result_path):
    with np.load(npz_path,allow_pickle=False) as f:
        cov=validate_covariance(f[f's{sd}'])
    collect={}
    for stream,subseed,faulty in [('healthy',900000+sd,False),('injected',1000000+sd,True)]:
        vectors,faults=pipeline(cov,subseed,with_faults=faulty)
        if len(faults)!=(2 if faulty else 0):raise ValueError('Wrong synthetic fault count')
        raw,named,feats=evidence_stream(vectors,cal_dir=cal_dir)
        delta=extract_delta(vectors)
        onset=np.full(7,-1,np.int16);expected=np.full(7,-1,np.int8)
        for a,f in faults.items():onset[a]=int(f['start']);expected[a]=int(f['ant'])+1
        collect.update({f'{stream}_v11':raw,f'{stream}_v11_named':named,
                        f'{stream}_feats':feats,f'{stream}_delta':delta,
                        f'{stream}_onset':onset,f'{stream}_expected':expected})
    result_path=Path(result_path);result_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=result_path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**collect)
    os.replace(tmp,result_path)
    return collect


def validate_v35_trace(trace,stream):
    validate_trace(trace,stream)
    key=f'{stream}_delta'
    if key not in trace:raise ValueError(f'Missing temporal field {key}')
    a=np.asarray(trace[key])
    if a.shape!=(300,7,3) or not np.issubdtype(a.dtype,np.number) or np.isinf(a).any():
        raise ValueError('Bad temporal feature schema')
    active=np.isfinite(a)
    if np.any(active.any(-1)&~active.all(-1)):
        raise ValueError('Partially missing temporal triplet')
    mask=active.all(-1)
    score,dom,name=a[:,:,0][mask],a[:,:,1][mask],a[:,:,2][mask]
    if (np.any(score<0) or np.any(dom<0) or np.any(name<1) or np.any(name>8)
        or np.any(name!=np.floor(name)) or mask[:120].any()):
        raise ValueError('Invalid temporal evidence or noncausal baseline')


def replay_trace_v35(trace,stream,config:FusionConfigV35):
    validate_v35_trace(trace,stream)
    raw=np.asarray(trace[f'{stream}_v11']);named=np.asarray(trace[f'{stream}_v11_named'])
    feats=np.asarray(trace[f'{stream}_feats']);delta=np.asarray(trace[f'{stream}_delta'])
    eng=EvidenceFusionV35(config)
    flags={x:np.zeros((300,7),dtype=bool) for x in ('final','review','review_only','delta_persistent','delta_solo','delta_supported','cohort')}
    names=np.zeros((300,7),dtype=np.int8)
    def mk(v):
        if np.isfinite(v).all():return ElementEvidence(float(v[0]),float(v[1]),int(v[2]))
        return None
    for t in range(300):
        ef=[[mk(feats[w,t,a]) for a in range(7)] for w in (0,1)]
        ed=[mk(delta[t,a]) for a in range(7)]
        valid=[bool(np.isfinite(feats[:,t,a,0]).any() or np.isfinite(delta[t,a,0])) for a in range(7)]
        out=eng.step(raw[t].tolist(),named[t].tolist(),ef[0],ef[1],ed,valid)
        for a,x in enumerate(out):
            for key,val in [('final',x.final_alert),('review',x.review_candidate),('review_only',x.review_only),
                            ('delta_persistent',x.delta_persistent),('delta_solo',x.delta_solo),
                            ('delta_supported',x.delta_supported),('cohort',x.cohort_ambiguous)]:
                flags[key][t,a]=val
            names[t,a]=x.antenna_1based or 0
    return {**flags,'antenna':names}
