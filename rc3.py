"""SANAD RC3 frozen read-only inference (threshold 0.50).

Requires only the bundled RC3 model, and the tightly scoped legacy detector
runtime on which the RC3 spatial model was originally trained. No training.
Never load untrusted joblib files. This is a research implementation, not
hardware actuation or real-world RF field qualification.
"""
from __future__ import annotations
from pathlib import Path
import sys, hashlib, warnings, json, gzip, io
import numpy as np
import joblib

HERE=Path(__file__).resolve().parent
ENGINE=HERE/'internal'/'rc0_runtime'
MODEL=HERE/'model'/'rc3_candidate.joblib.gz'
COMPRESSED_SHA='2cd6661607876462bd468e0cfdd720d1b893bc0b7c7f9679c209843b5ffbf12a'
EXPECTED_SHA='0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4'
RC0_SHA='032d2d600600b182b689a14682c8260e9217873dd8a65ab015d15765c98dd96d'
THRESHOLD=0.50

def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if _sha(MODEL)!=COMPRESSED_SHA:
    raise RuntimeError('Frozen RC3 compressed artifact SHA-256 mismatch')
MODEL_BYTES=gzip.decompress(MODEL.read_bytes())
if hashlib.sha256(MODEL_BYTES).hexdigest()!=EXPECTED_SHA:
    raise RuntimeError('Frozen RC3 unpacked model SHA-256 mismatch')
if _sha(ENGINE/'frozen_candidate'/'selected_model.joblib')!=RC0_SHA:
    raise RuntimeError('Frozen internal detector model SHA-256 mismatch')
if str(ENGINE) not in sys.path:sys.path.insert(0,str(ENGINE))
from inference_v35_gate import FrozenV35LearnedGate
def stats(x):
    x=np.asarray(x,dtype=np.float64).ravel()
    x=x[np.isfinite(x)]
    if not len(x):return [0.0]*9
    q=np.quantile(x,[.1,.25,.5,.75,.9])
    return [float(x.size),float(np.mean(x)),float(np.std(x)),*map(float,q),float(np.max(x))]

def make_features(z,prefix):
    f=np.asarray(z[prefix+'_feats'],dtype=float)
    d=np.asarray(z[prefix+'_delta'],dtype=float)
    v=np.asarray(z[prefix+'_v11'],dtype=float)
    if f.shape!=(2,300,7,3) or d.shape!=(300,7,3) or v.shape!=(300,7):
        raise ValueError('Unexpected source trace schema')
    raw=np.concatenate([f.transpose(1,2,0,3).reshape(300,7,6),d],axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        baseline=np.nanmedian(raw,axis=1,keepdims=True)
    rel=raw-baseline
    rows=[]
    for j in range(7):
        feats=[]
        for a in (raw[:,j,:],rel[:,j,:]):
            for c in range(a.shape[-1]):
                trace=a[:,c]
                for l,r in ((180,220),(220,260),(260,300),(280,300)):
                    feats.extend(stats(trace[l:r]))
                recent=np.asarray(trace[260:300],dtype=float)
                former=np.asarray(trace[220:260],dtype=float)
                recent=recent[np.isfinite(recent)]
                former=former[np.isfinite(former)]
                feats.append(float(np.median(recent)-np.median(former)) if len(recent) and len(former) else 0.)
        for l,r in ((180,220),(220,260),(260,300)):
            feats.extend(stats(v[l:r,j]))
        rows.append(feats)
    a=np.asarray(rows,dtype=np.float32)
    a=np.nan_to_num(a,nan=0,posinf=1e6,neginf=-1e6)
    assert a.shape==(7,693) and np.isfinite(a).all()
    return a

def meta_features(scores,flags):
    scores=np.asarray(scores,float);flags=np.asarray(flags,dtype=bool)
    assert scores.shape==flags.shape==(7,) and np.isfinite(scores).all()
    assert scores.min()>=0 and scores.max()<=1
    a=scores.reshape(-1,7);b=flags.reshape(-1,7)
    rank=(np.argsort(np.argsort(-a,axis=1),axis=1)+1).reshape(-1)
    descending=np.sort(a,axis=1)[:,::-1]
    top1=np.repeat(descending[:,0],7)
    top2=np.repeat(descending[:,1],7)
    top3=np.repeat(descending[:,2],7)
    mean=np.repeat(a.mean(axis=1),7)
    std=np.repeat(a.std(axis=1),7)
    rc0count=np.repeat(b.sum(axis=1),7)
    count90=np.repeat((a>.9).sum(axis=1),7)
    count40=np.repeat((a>.4).sum(axis=1),7)
    eps=1e-5
    v=np.column_stack([scores,np.log((scores+eps)/(1-scores+eps)),
      flags.astype(float),rank,top1,top2,top3,mean,std,top1-scores,
      scores/(top1+eps),scores/(top2+eps),count90,count40,rc0count])
    assert v.shape==(7,15)
    return np.asarray(v,dtype=np.float32)

class FrozenRC3:
    def __init__(self):
        bundle=joblib.load(io.BytesIO(MODEL_BYTES))
        assert bundle['model_kind']=='spatial_meta_two_stage'
        assert bundle['train_seeds']==[2500,2643]
        assert bundle['feature_count']==693 and bundle['meta_feature_count']==15
        self.base_models=bundle['rc1_models_full']
        self.meta_model=bundle['rc3_meta_model']
        self.rc0=FrozenV35LearnedGate(ENGINE/'frozen_candidate'/'selected_model.joblib')
        self.threshold=THRESHOLD

    def predict_trace(self, trace, stream):
        if stream not in ('healthy','injected'):
            raise ValueError('Expected an evidence stream name: healthy/injected')
        x=make_features(trace,stream)
        flags=np.asarray([d.learned_confirmed_alert for d in self.rc0.predict_trace(trace,stream)],dtype=bool)
        scores=.5*(self.base_models[0].predict_proba(x)[:,1]+self.base_models[1].predict_proba(x)[:,1])
        spatial=meta_features(scores,flags)
        proba=np.asarray(self.meta_model.predict_proba(spatial)[:,1],dtype=float)
        assert proba.shape==(7,) and np.isfinite(proba).all()
        return {'alerts':(proba>=self.threshold).tolist(),
                'probabilities':proba.tolist(),'threshold':self.threshold}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('evidence',help='Sionna NPZ evidence with seven-array traces')
    ap.add_argument('--stream',choices=('healthy','injected'),default='injected')
    args=ap.parse_args()
    with np.load(args.evidence,allow_pickle=False) as z:
        evidence={k:z[k] for k in z.files}
    print(json.dumps(FrozenRC3().predict_trace(evidence,args.stream),indent=2))
