"""Reproduce v32 tuning with no future measurements and no test-label leakage.

Acquires raw per-report evidence from the frozen v11/SANAD v29 pipeline once
and performs rapid deterministic causal replays of the outer fusion gate.
"""
from __future__ import annotations
import json, sys, os, hashlib, argparse
from pathlib import Path
import numpy as np
from dataclasses import asdict
from sanad_v29 import SANADNetworkV29
from v29_research import UNIT, split_smooth_sparse
from sanad_v32 import FusionConfig
from qualification import validate_covariance, wilson

T=300; N=7; CUT=260

def evidence_stream(vectors, *, cal_dir):
    model=SANADNetworkV29(cal_dir=cal_dir,recent=15)
    raw_v11=np.zeros((T,N),np.uint8)
    raw_v11_named=np.zeros((T,N),np.int8)
    features=np.full((2,T,N,3),np.nan,np.float32) # windows 12/15, [score,dominance,antenna 1based]
    for t in range(T):
        batch={a:vectors[a][t] for a in range(N) if vectors[a][t] is not None}
        desc=model.update(batch)
        for a,x in enumerate(desc):
            raw_v11[t,a]=int(x.confirmed_v11)
            raw_v11_named[t,a]=x.antenna_1based or 0
            for ix,w in enumerate((12,15)):
                window=list(model.window[a])[-w:]
                valid=[v for v in window if v is not None]
                if len(window)<8 or len(valid)<max(8,w//2) or not x.valid:
                    continue
                r=np.median(np.asarray(valid).reshape(-1,2,8),axis=0).T/UNIT
                _,s=split_smooth_sparse(r)
                strengths=np.abs(s).sum(axis=1)
                ant=int(strengths.argmax())
                top=float(strengths[ant]);second=float(np.partition(strengths,-2)[-2])
                features[ix,t,a]=(top,top/max(second,1e-5),ant+1)
    return raw_v11,raw_v11_named,features


def extract_seed(npz_path:str|Path, cal_dir:str|Path, sd:int, result_path: str|Path):
    from glr_compare import pipeline
    with np.load(npz_path,allow_pickle=False) as f:
        z=validate_covariance(f[f's{sd}'])
    collected={}
    for stream,seed,with_faults in [('healthy',900000+sd,False),('injected',1000000+sd,True)]:
        vectors,faults=pipeline(z,seed,with_faults=with_faults)
        assert len(faults)==(2 if with_faults else 0)
        v11,named,feats=evidence_stream(vectors,cal_dir=cal_dir)
        onsets=np.full(N,-1,np.int16);expected=np.full(N,-1,np.int8)
        for a,f in faults.items():
            onsets[a]=int(f['start']);expected[a]=int(f['ant'])+1
        collected[f'{stream}_v11']=v11
        collected[f'{stream}_v11_named']=named
        collected[f'{stream}_feats']=feats
        collected[f'{stream}_onset']=onsets
        collected[f'{stream}_expected']=expected
    result_path=Path(result_path);result_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=result_path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**collected)
    os.replace(tmp,result_path)
    return collected


def _roll(x,period):
    y=np.zeros_like(x,dtype=np.int16)
    for t in range(x.shape[0]):
        y[t]=np.sum(x[max(0,t-period+1):t+1],axis=0)
    return y


def gate_evidence(raw,feat, cfg:FusionConfig, *, raw_named=None):
    widx=0 if cfg.recent==12 else 1
    feat=feat[widx]
    t,n=raw.shape
    hits=np.zeros((t,n,8),np.uint8)
    valid=np.isfinite(feat[:,:,0])
    active=valid & (feat[:,:,0]>cfg.score_threshold) & (feat[:,:,1]>=cfg.dominance_ratio)
    ti,ai=np.where(active)
    ant=feat[ti,ai,2].astype(int)-1
    hits[ti,ai,ant]=1
    candidate=np.max(_roll(hits,cfg.recent),axis=2)>=cfg.persistence
    confirmed=_roll(raw,cfg.v11_confirm_window)>=cfg.v11_confirm_min
    if cfg.v11_identity_min:
        if raw_named is None:raise ValueError("raw_named required when v11_identity_min>0")
        valid_names=np.where(raw>0,raw_named,0)
        identity_onehot=(valid_names[...,None]==np.arange(1,9)[None,None,:]).astype(np.uint8)
        identities=_roll(identity_onehot,cfg.v11_confirm_window)
        confirmed=confirmed & (identities.max(axis=2)>=cfg.v11_identity_min)
    alert=candidate|confirmed
    return alert,confirmed,candidate


def evaluate_trace(sd, trace, cfg):
    result=dict(seed=sd,healthy_false=0,fault_true=0,neighbor_false=0,delays=[],names_good=0,names_total=0,baseline_healthy_false=0,baseline_fault_true=0)
    for name in ('healthy','injected'):
        raw=trace[f'{name}_v11'];feat=trace[f'{name}_feats'];onset=trace[f'{name}_onset'];expected=trace[f'{name}_expected']
        alarms,confirmed,candidate=gate_evidence(raw,feat,cfg,raw_named=trace[f'{name}_v11_named'])
        for a in range(N):
            pred=bool(alarms[CUT:,a].any());base=bool(raw[CUT:,a].any())
            if name=='healthy':
                result['healthy_false']+=pred
                result['baseline_healthy_false']+=base
            elif onset[a]>=0:
                result['fault_true']+=pred
                result['baseline_fault_true']+=base
                after=np.flatnonzero(alarms[onset[a]:,a]);
                if len(after):result['delays'].append(int(after[0]))
            else:
                result['neighbor_false']+=pred
    return result


def summarize(files,configs,seed_range,*,tune=False):
    traces={s:dict(np.load(Path(files)/f's{s}.npz',allow_pickle=False)) for s in seed_range}
    rows=[]
    for cfg in configs:
        arr=[evaluate_trace(s,d,cfg) for s,d in traces.items()]
        tp=sum(r['fault_true'] for r in arr);fp=sum(r['healthy_false'] for r in arr)
        btp=sum(r['baseline_fault_true'] for r in arr);bfp=sum(r['baseline_healthy_false'] for r in arr)
        n=len(arr)
        rows.append({**asdict(cfg),'seeds':n,'tp':tp,'fault_n':2*n,'recall':tp/(2*n),
                     'fp':fp,'healthy_n':7*n,'fpr':fp/(7*n),
                     'neighbor_fp':sum(r['neighbor_false'] for r in arr),
                     'base_tp':btp,'base_fp':bfp,
                     'median_delay_reports':float(np.median([delay for r in arr for delay in r['delays']])) if any(r['delays'] for r in arr) else None,
                     'wilson_false_upper':wilson(fp,7*n)[1]})
    if tune:
        # Pre-specified lexicographic search: prioritize acceptable FPR, then recall,
        # then fewer false alarms; infeasible candidates reported as infeasible.
        ok=[r for r in rows if r['fpr']<=.02 and r['recall']>=.90]
        chosen=max(ok,key=lambda r:(r['recall'],-r['fpr'])) if ok else min(rows,key=lambda r:(max(0,r['fpr']-.02)*10+max(0,.90-r['recall'])*5,-r['recall']))
        return rows,chosen,bool(ok)
    return rows


def configs():
    from itertools import product
    # Fixed score threshold avoids tune-on-test shortcuts. Temporal options
    # include the requested 10/15 and stricter 12/15.
    for dom,(k,w),v11 in product((1.5,1.8,2.0),((8,12),(10,15),(12,15)),((3,5),(5,8),(7,10),(9,12))):
        yield FusionConfig(dominance_ratio=dom,persistence=k,recent=w,
             v11_confirm_min=v11[0],v11_confirm_window=v11[1],
             v11_identity_min=3)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--npz-dir',default='/mnt/data')
    p.add_argument('--cache-dir',default='cache_dev')
    p.add_argument('--seeds',default='500-505')
    p.add_argument('--skip-extract',action='store_true')
    p.add_argument('--cal-dir',default='.')
    p.add_argument('--out',default='v32_dev_tuning.json')
    a=p.parse_args();lo,hi=map(int,a.seeds.split('-')); seeds=list(range(lo,hi+1))
    if any(600<=s<=743 for s in seeds):
        raise ValueError('Frozen v31 seeds 600..743 are FORBIDDEN for model selection')
    Path(a.cache_dir).mkdir(parents=True,exist_ok=True)
    if not a.skip_extract:
        for sd in seeds:
            out=Path(a.cache_dir)/f's{sd}.npz'
            if not out.exists():
                extract_seed(Path(a.npz_dir)/f's{sd}.npz',a.cal_dir,sd,out)
                print(f'Extracted seed {sd}',flush=True)
    rows,best,feasible=summarize(a.cache_dir,list(configs()),seeds,tune=True)
    payload=dict(seeds=seeds,development_only=True,feasible=feasible,selected=best,
                 ranking=sorted(rows,key=lambda r:(r['fpr'],-r['recall']))[:25],grid=rows)
    Path(a.out).write_text(json.dumps(payload,indent=2))
    print('TUNING:',json.dumps({k:payload[k] for k in ('development_only','feasible','selected')},indent=2),flush=True)

if __name__=='__main__':main()
