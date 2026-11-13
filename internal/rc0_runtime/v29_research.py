"""SANAD Network v29 research: frozen v11 plus geometry-aware residual evidence.

No RF actuation. Signal is original glr_compare covariance proxy, not raw antenna
calibration telemetry. Thresholds MUST be frozen on development seeds only.
No simulator labels used inside inference.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

UNIT = np.array([0.5,5.0],float)
X = np.column_stack([np.ones(8),np.linspace(-1,1,8)])
PINV=np.linalg.pinv(X)


def soft_threshold(x, lam):
    return np.sign(x)*np.maximum(np.abs(x)-lam,0.0)


def split_smooth_sparse(y, lam=0.12, steps=8, beta=0.55):
    """Fit smooth spatial drift + sparse element fault in normalized physical units.

    Alternating projection onto linear spatial drift and soft-threshold of the
    residual. Damped (Ishikawa/Mann-type averaging) for numerical stability;
    no convergence theorem claimed for the time-varying RF process.
    """
    y=np.asarray(y,float)
    drift=X@(PINV@y)
    sparse=np.zeros_like(y)
    for _ in range(steps):
        new_drift=X@(PINV@(y-sparse))
        drift=(1-beta)*drift+beta*new_drift
        new_sparse=soft_threshold(y-drift,lam)
        sparse=(1-beta)*sparse+beta*new_sparse
    return drift,sparse


def chronological_features(aligned, windows=(12,24,40)):
    """[300,7,16] residuals; no future data, no labels, no adapt-after-alarm."""
    arr=np.asarray(aligned,float)
    T,N,C=arr.shape
    if N!=7 or C!=16:raise ValueError(f'expected [T,7,16]; got {arr.shape}')
    outs={f'{mode}_{w}':np.full((T,N),np.nan) for w in windows for mode in ('raw','sparse','contrast')}
    names={f'{mode}_{w}':np.full((T,N),-1,int) for w in windows for mode in ('raw','sparse','contrast')}
    for w in windows:
        for t in range(120,T):
            st=max(120,t-w+1)
            hist=arr[st:t+1]
            valid=np.isfinite(hist).all(axis=2)
            for i in range(N):
                subset=hist[valid[:,i],i]
                if len(subset)<max(8,w//2):continue
                r=np.median(subset.reshape(-1,2,8),axis=0).T / UNIT # [8,2] (gain8,phase8)
                d,s=split_smooth_sparse(r)
                raw=np.abs(r).sum(axis=1)
                sparse=np.abs(s).sum(axis=1)
                # singleton localized effect vs distributed spatial change
                top=np.sort(raw)
                scores={'raw':raw,'sparse':sparse,'contrast':np.maximum(0,raw-top[-2])}
                for key,z in scores.items():
                    nm=f'{key}_{w}'
                    outs[nm][t,i]=float(z.max())
                    names[nm][t,i]=int(z.argmax())
    return outs,names


def persistent_events(score, threshold, *, k=8, recent=12, cutoff=260):
    """Return earliest alarm index t within historical event window."""
    score=np.asarray(score,float)
    t,n=score.shape
    starts=np.full(n,-1,int)
    for j in range(n):
        for i in range(cutoff,t):
            s=score[max(120,i-recent+1):i+1,j]
            if np.sum(np.isfinite(s)&(s>threshold))>=k:
                starts[j]=i
                break
    return starts


def one_run_features(path):
    with np.load(path,allow_pickle=False) as z:
        aligned=z['aligned']; faulty=z['faulty']; v11any=z['v11any']; expected=z['expected']; start=z['start']; v11local=z['v11local']; v11named=z['v11named']
    feats,names=chronological_features(aligned)
    return dict(faulty=faulty,v11any=v11any,v11local=v11local,v11named=v11named,expected=expected,start=start,feats=feats,names=names)


def evaluate(runs, feature, th, *,combine=True,k=8):
    tp=fp=positive=negative=0; loc_good=loc_n=0; delays=[]
    for name,d in runs.items():
        events=persistent_events(d['feats'][feature],th,k=k)
        j=260
        # v11 ANY signal is already confirmed at update() time
        pre=np.where(d['v11any'][j:,:],np.arange(j, len(d['v11any']))[:,None],9999).min(axis=0)
        for i in range(7):
            present=events[i]>=0
            e=int(events[i]) if present else 9999
            v=int(pre[i])
            final=min(e,v) if combine else e
            detected=final<9999
            if d['faulty'][i]:
                positive+=1
                if detected:
                    tp+=1
                    delays.append(final-int(d['start'][i]))
                    if final==e:
                        loc=int(d['names'][feature][final,i])
                    else:
                        loc=int(d['v11named'][final,i])
                    loc_n+=1;loc_good+=int(loc==d['expected'][i])
            else:
                negative+=1
                fp+=int(detected)
    return dict(tp=tp,positives=positive,recall=tp/positive,fp=fp,negatives=negative,fpr=fp/negative,loc=loc_good,loc_n=loc_n,delay_median=float(np.median(delays)) if delays else None)
