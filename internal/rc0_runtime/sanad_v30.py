"""SANAD Network v30 - conservative RF research evidence gate.

Frozen v11 ADVISORY path, plus a temporally stable, spatially DOMINANT
single-element sparse-residual review path. Research-only; NO hardware commands.

This is a candidate for validation, not a certified RF calibration instrument.
Input: original `glr_compare.pipeline` 16D gain/phase proxy per array.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np
from sanad_v29 import SANADNetworkV29
from v29_research import UNIT,split_smooth_sparse

@dataclass(frozen=True)
class V30Decision:
    array_id: int
    status: str
    advisory_v11: bool
    v29_candidate: bool
    v30_candidate: bool
    independent_review: bool
    antenna_1based: int|None
    score: float|None
    dominance_ratio: float|None
    instrument_valid: bool
    action: str = 'NO_HARDWARE_ACTUATION'

class SANADNetworkV30:
    def __init__(self, *, cal_dir:str|Path|None=None,
                 score_threshold:float=0.7687096042171566,
                 dominance_ratio:float=1.50,
                 persistence:int=8, recent:int=12):
        if not 1<=persistence<=recent:raise ValueError('invalid persistence')
        if dominance_ratio<1:raise ValueError('dominance_ratio must be >=1')
        if not np.isfinite(score_threshold) or score_threshold<0:raise ValueError('invalid threshold')
        self.v29=SANADNetworkV29(cal_dir=cal_dir)
        self.persistence=int(persistence)
        self.recent=int(recent)
        self.threshold=float(score_threshold)
        self.ratio=float(dominance_ratio)
        self.history=[deque(maxlen=recent) for _ in range(7)]

    def update(self,vecs:Mapping[int,np.ndarray]):
        d29=self.v29.update(vecs)
        out=[]
        for i,d in enumerate(d29):
            score=None;dom=None; cand_ant=None
            window=self.v29.window[i]
            valid=[x for x in window if x is not None]
            allowed=(len(window)>=8 and len(valid)>=max(8,self.recent//2) and d.valid)
            active=np.zeros(8,dtype=bool)
            if allowed:
                r=np.median(np.asarray(valid).reshape(-1,2,8),axis=0).T/UNIT
                _,s=split_smooth_sparse(r)
                v=np.abs(s).sum(axis=1)
                a=int(v.argmax())
                top=float(v[a]);second=float(np.partition(v,-2)[-2])
                score=top
                dom=top/max(second,1e-5)
                if top>self.threshold and dom>=self.ratio:
                    active[a]=True
            self.history[i].append(active)
            count=np.sum(np.asarray(self.history[i],dtype=int),axis=0)
            candidate=bool(np.max(count)>=self.persistence)
            if candidate:
                cand_ant=int(np.argmax(count))+1
            if d.confirmed_v11:
                status='ADVISORY_V11_EVIDENCE'
                ant=d.antenna_1based
            elif candidate:
                status='RESEARCH_REVIEW_ONLY'
                ant=cand_ant
            else:
                status=d.status if d.status not in ('RESEARCH_REVIEW_ONLY',) else 'MONITORING'
                ant=None
            out.append(V30Decision(array_id=i,status=status,advisory_v11=d.confirmed_v11,
                   v29_candidate=d.research_candidate,v30_candidate=candidate,
                   independent_review=bool(candidate and not d.confirmed_v11),
                   antenna_1based=ant,score=score,dominance_ratio=dom,
                   instrument_valid=d.valid))
        return out
