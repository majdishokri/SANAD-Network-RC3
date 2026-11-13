"""Read-only, causal-feature extraction for frozen SANAD v35 evidence traces.

Model rows are per antenna ARRAY at the end of report 299. No fault labels,
stream identity, seed number, or future values are supplied to features.
Training labels are created separately by train_v35_gate.py.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sanad_v33 import FusionConfigV33
from sanad_v33 import ElementEvidence, EvidenceFusionV33
from sanad_v35 import EvidenceFusionV35
from qa_integrity import validate_trace
from sanad_hybrid_v33_v35 import load_frozen_v35_config

FROZEN_V35=load_frozen_v35_config(Path(__file__).resolve().parent)
N_ARRAYS=7
WINDOW=40
END=300


def _fill(x):
    return np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)

def _modal_strength(names, mask):
    if not np.any(mask): return 0,0,0
    vals=names[mask].astype(int)
    vals=vals[(vals>=1)&(vals<=8)]
    if not len(vals):return 0,0,0
    counts=np.bincount(vals,minlength=9)[1:]
    winner=int(np.argmax(counts)+1)
    return winner,int(counts.max()), int(len(vals))

def _replay_without_truth(trace, stream):
    """Frozen v33/v35 step() implementations, fed ONLY observation arrays.

    No fields named *_expected or *_onset are read, even for validation.
    """
    raw=np.asarray(trace[f'{stream}_v11'])
    named=np.asarray(trace[f'{stream}_v11_named'])
    feats=np.asarray(trace[f'{stream}_feats'])
    delta=np.asarray(trace[f'{stream}_delta'])
    eng35=EvidenceFusionV35(FROZEN_V35)
    eng33=EvidenceFusionV33(FusionConfigV33())
    k35=['final','review','delta_persistent','delta_supported','delta_solo','cohort']
    k33=['final','strict','corroborated','confirmed_v11']
    a35={k:np.zeros((300,7),bool) for k in k35}
    a33={k:np.zeros((300,7),bool) for k in k33}
    def mk(v):
        if np.isfinite(v).all():return ElementEvidence(float(v[0]),float(v[1]),int(v[2]))
        return None
    for t in range(300):
        fast=[mk(feats[0,t,a]) for a in range(7)]
        strict=[mk(feats[1,t,a]) for a in range(7)]
        temporal=[mk(delta[t,a]) for a in range(7)]
        valid35=[bool(np.isfinite(feats[:,t,a,0]).any() or np.isfinite(delta[t,a,0])) for a in range(7)]
        valid33=[bool(np.isfinite(feats[:,t,a,0]).any()) for a in range(7)]
        o35=eng35.step(raw[t].tolist(),named[t].tolist(),fast,strict,temporal,valid35)
        o33=eng33.step(raw[t].tolist(),named[t].tolist(),fast,strict,valid33)
        for a in range(7):
            x35=o35[a];x33=o33[a]
            for k,v in [('final',x35.final_alert),('review',x35.review_candidate),
                        ('delta_persistent',x35.delta_persistent),('delta_supported',x35.delta_supported),
                        ('delta_solo',x35.delta_solo),('cohort',x35.cohort_ambiguous)]:a35[k][t,a]=v
            for k,v in [('final',x33.final_alert),('strict',x33.strict_v32),
                        ('corroborated',x33.spatial_corroborated),('confirmed_v11',x33.confirmed_v11)]:a33[k][t,a]=v
    return a35,a33


def _validate_observations_only(trace,stream):
    validate_trace(trace,stream,require_truth=False)
    delta=np.asarray(trace[f'{stream}_delta'])
    if delta.shape!=(300,7,3) or np.isinf(delta).any():raise ValueError('Bad temporal features')
    finite=np.isfinite(delta)
    if np.any(finite.any(-1)&~finite.all(-1)):raise ValueError('Missing temporal triplet')
    d=delta[finite.all(-1)]
    if len(d) and (np.any(d[:,0]<0) or np.any(d[:,1]<0) or np.any(d[:,2]<1)
                    or np.any(d[:,2]>8) or np.any(d[:,2]!=np.floor(d[:,2]))):
        raise ValueError('Bad temporal feature values')


def features_per_array(trace, stream, *, include_replays=True):
    """Return (7, features) at endpoint 299; inference does not consult labels.

    Labels remain accessible in trace but are never read by this function.
    """
    if stream not in ('healthy','injected'):raise ValueError('invalid stream')
    _validate_observations_only(trace,stream)
    raw=np.asarray(trace[f'{stream}_v11'],bool)
    names=np.asarray(trace[f'{stream}_v11_named'],int)
    feats=np.asarray(trace[f'{stream}_feats'],float)
    delta=np.asarray(trace[f'{stream}_delta'],float)
    out35,out33=_replay_without_truth(trace,stream)
    rows=[];columns=None
    for a in range(7):
        d={}
        raw40=raw[260:300,a]
        n40=names[260:300,a]
        d['v11_count40']=int(raw40.sum())
        d['v11_count20']=int(raw[280:300,a].sum())
        d['v11_count12']=int(raw[288:300,a].sum())
        nm,nc,nt=_modal_strength(n40,raw40)
        d['v11_modal_count40']=nc
        d['v11_modal_frac40']=nc/max(nt,1)
        d['v11_unique_name_count']=len(np.unique(n40[(n40>0)&raw40]))
        d['v11_last10_count']=int(raw[290:300,a].sum())
        eviews=[]
        for key,idx in [('fast',0),('strict',1),('delta',None)]:
            ev=delta[260:300,a,:] if idx is None else feats[idx,260:300,a,:]
            score=ev[:,0]
            dominance=ev[:,1]
            ename=ev[:,2]
            valid=np.isfinite(score)&np.isfinite(dominance)&np.isfinite(ename)
            s=_fill(score);dom=_fill(dominance)
            d[f'{key}_valid_fraction']=float(valid.mean())
            for metric,x in [('score',s),('dom',dom)]:
                d[f'{key}_{metric}_median']=float(np.median(x))
                d[f'{key}_{metric}_max']=float(np.max(x))
                d[f'{key}_{metric}_p90']=float(np.quantile(x,.9))
                d[f'{key}_{metric}_last10_mean']=float(x[-10:].mean())
            d[f'{key}_score_early20']=float(s[:20].mean())
            d[f'{key}_score_late20']=float(s[20:].mean())
            if key=='fast': thr,domthr=.7687096042171566,1.5
            elif key=='strict':thr,domthr=.7687096042171566,1.8
            else:thr,domthr=4.0,1.45
            hit=valid&(score>=thr)&(dominance>=domthr)
            top,num,ntrials=_modal_strength(ename,hit)
            d[f'{key}_strong_count40']=int(hit.sum())
            d[f'{key}_modal_count40']=int(num)
            d[f'{key}_modal_fraction']=num/max(ntrials,1)
            d[f'{key}_last10_strong']=int(hit[-10:].sum())
            d[f'{key}_peak_coherent_score']=float(np.max(np.where(hit,s,0)))
            d[f'{key}_modal_id']=int(top) if key=='delta' else 0   # only delta ID for cross-view consistency
            eviews.append((top,hit))
        # Cross-view agreement of modal candidates (no ground truth)
        fast_id,fast_hits=eviews[0];strict_id,strict_hits=eviews[1];delta_id,delta_hits=eviews[2]
        d['fast_strict_same_element']=int(fast_id>0 and fast_id==strict_id)
        d['delta_strict_same_element']=int(delta_id>0 and delta_id==strict_id)
        d['delta_fast_same_element']=int(delta_id>0 and delta_id==fast_id)
        d['delta_v11_same_element']=int(delta_id>0 and delta_id==nm)
        d['strict_v11_same_element']=int(strict_id>0 and strict_id==nm)
        d['delta_coincidence_with_raw']=int(np.sum(delta_hits & raw40))
        d['strict_coincidence_with_raw']=int(np.sum(strict_hits & raw40))
        d['v35_alert_count40']=int(out35['final'][260:300,a].sum())
        d['v33_alert_count40']=int(out33['final'][260:300,a].sum())
        d['v35_review_count40']=int(out35['review'][260:300,a].sum())
        d['v35_temporal_persistent_count40']=int(out35['delta_persistent'][260:300,a].sum())
        d['v35_temporal_supported_count40']=int(out35['delta_supported'][260:300,a].sum())
        d['v35_temporal_solo_count40']=int(out35['delta_solo'][260:300,a].sum())
        d['v35_cohort_count40']=int(out35['cohort'][260:300,a].sum())
        d['v33_strict_count40']=int(out33['strict'][260:300,a].sum())
        d['v33_corroborated_count40']=int(out33['corroborated'][260:300,a].sum())
        d['v33_v11_count40']=int(out33['confirmed_v11'][260:300,a].sum())
        d['v35_fired']=int(np.any(out35['final'][260:300,a]))
        d['v33_fired']=int(np.any(out33['final'][260:300,a]))
        # Rates of simultaneous array alarms, measured from observables only.
        active=out35['final'][260:300].sum(axis=1)
        d['v35_group_max_active']=int(active.max())
        d['v35_group_mean_active']=float(active.mean())
        if columns is None: columns=list(d)
        assert columns==list(d)
        rows.append([d[k] for k in columns])
    matrix=np.asarray(rows,dtype=np.float64)
    assert matrix.shape==(7,len(columns)) and np.isfinite(matrix).all()
    return matrix,columns


def extract_dataset(files, *, validate=True):
    X=[];y=[];groups=[];origin=[];array=[];case_sha=[]
    import hashlib
    cols=None
    for file in files:
        file=Path(file)
        seed=int(file.stem[1:])
        with np.load(file,allow_pickle=False) as z:
            t={k:z[k] for k in z.files}
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        for stream in ('healthy','injected'):
            mx,names=features_per_array(t,stream)
            if cols is None:cols=names
            assert cols==names
            truth=(t[f'{stream}_expected']>0).astype(int)
            for a in range(7):
                X.append(mx[a]);y.append(int(truth[a]));groups.append(seed)
                origin.append(stream);array.append(a);case_sha.append(digest)
    return dict(X=np.array(X),y=np.array(y),seed=np.array(groups),
                stream=np.array(origin),array=np.array(array),
                case_sha=np.array(case_sha),columns=cols)
