"""SANAD Network v29 — read-only RF calibration assurance research.

Requires 7 independent 8-element arrays, 16-vector per array (gain dB followed
by phase degrees) from the SAME preprocessing contract as original v11.
Do not substitute raw RF, RSSI, or covariance diagonal in place of calibration
residual vectors. Never commands RF hardware.

Confirmed channel: frozen v11.any (loc/disp/Wasserstein). Additional v29 spatial
sparse-temporal evidence is strictly a review channel, not a confirmed fault.

Source: `full.pdf` Shukri 2016 section 5.1.5 motivates damped iteration;
this use of damping does not inherit convergence results without hypotheses.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Mapping
import numpy as np

from v29_research import UNIT, split_smooth_sparse
from geometry_config import N_ARRAYS, N_ELEMENTS

@dataclass(frozen=True)
class Decision:
    array_id: int
    status: str
    confirmed_v11: bool
    research_candidate: bool
    independent_review: bool
    antenna_1based: int | None
    spatial_sparse_score: float | None
    reference_kind: str
    valid: bool
    action: str = 'NO_HARDWARE_ACTUATION'

class SANADNetworkV29:
    def __init__(self, *, threshold:float=0.7687096042171566,
                 warmup:int=60, learn:int=60, k:int=8,
                 recent:int=12, cal_dir: str | Path | None=None):
        if k<1 or k>recent:raise ValueError('invalid persistence')
        if cal_dir is not None:
            p=str(Path(cal_dir).resolve())
            if p not in sys.path:sys.path.insert(0,p)
        from detect_v11 import default_detector,choose_reference_inliers,ABS_LIMIT,MIN_GROUP
        self._choose_reference=choose_reference_inliers
        self._abs_limit=ABS_LIMIT; self._min_group=MIN_GROUP
        self.baseline=default_detector(N_ARRAYS,warmup=warmup,learn=learn)
        self.window=[deque(maxlen=recent) for _ in range(N_ARRAYS)]
        self.binary=[deque(maxlen=recent) for _ in range(N_ARRAYS)]
        self.k=int(k);self.recent=int(recent);self.threshold=float(threshold)
        self.period=0
        self.t_last=[-1]*N_ARRAYS

    def update(self, vecs: Mapping[int,np.ndarray]):
        if not isinstance(vecs,dict):vecs=dict(vecs)
        accepted={}
        for i,v in vecs.items():
            if type(i)!=int or not 0<=i<N_ARRAYS:raise ValueError('array id must be integer 0..6')
            x=np.asarray(v,float)
            if x.shape!=(2*N_ELEMENTS,) or not np.isfinite(x).all():
                raise ValueError('calibration vector must be finite 16-vector: gains8,phases8')
            accepted[i]=x
        d11=self.baseline.update(accepted)
        z={i:self.baseline.baseline[i].z(x) for i,x in accepted.items()
           if self.baseline.baseline[i] is not None}
        if len(z)>=5:
            reference=self._choose_reference(np.array(list(z.values())),self._abs_limit,self._min_group)
            aligned={i:x-reference.vector for i,x in z.items()}
            kind=reference.kind
        else:
            aligned={};kind='insufficient'
        out=[]
        for i in range(N_ARRAYS):
            v=aligned.get(i)
            self.window[i].append(v)
            valid=np.array([a for a in self.window[i] if a is not None])
            score=None; ant=None
            if len(self.window[i])>=8 and len(valid)>=max(8,self.recent//2):
                m=np.median(valid.reshape(-1,2,N_ELEMENTS),axis=0).T/UNIT
                _,s=split_smooth_sparse(m)
                unit=np.abs(s).sum(axis=1)
                score=float(unit.max());ant=int(unit.argmax())+1
            over=bool(score is not None and score>self.threshold)
            self.binary[i].append(over)
            research=bool(sum(self.binary[i])>=self.k)
            v11_any=bool(d11['alarm'][i] or d11['w_alarm'][i] or d11['distributed_alarm'][i])
            additional=bool(research and not v11_any)
            if v11_any:
                status='ADVISORY_V11_EVIDENCE'
                suspect=d11['suspect'][i]
                selected=int(suspect)+1 if suspect is not None else None
            elif additional:
                status='RESEARCH_REVIEW_ONLY'
                selected=ant
            elif self.baseline.baseline[i] is None:
                status='LEARNING'
                selected=None
            elif i not in aligned:
                status='DATA_INSUFFICIENT'
                selected=None
            else:
                status='MONITORING'
                selected=None
            out.append(Decision(i,status,v11_any,research,additional,
                                selected,score,kind,i in aligned))
        self.period+=1
        return out
