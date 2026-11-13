"""SANAD Network v32: causal evidence-fusion RESEARCH candidate.

Preserves original v11 engine, adds vetoable advisory promotion and a distinct
sparse-spatial review channel. Configuration is selected on DEVELOPMENT data;
no calibration using frozen v31 seeds 600..743 or future test labels.

Input: seven 16-dimensional gain/phase proxy vectors, NEVER raw RF power.
No hardware command path, no automatic calibration or safety claim.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np
from sanad_v29 import SANADNetworkV29
from v29_research import UNIT, split_smooth_sparse


@dataclass(frozen=True)
class FusionConfig:
    score_threshold: float = 0.7687096042171566
    dominance_ratio: float = 1.8
    persistence: int = 10
    recent: int = 15
    v11_confirm_min: int = 5
    v11_confirm_window: int = 8
    v11_identity_min: int = 3  # repeated identification of SAME antenna
    # Boolean research review and gated v11 advisory deliberately not conflated.
    def __post_init__(self):
        if not 1 <= self.persistence <= self.recent <= 60:
            raise ValueError('persistence/recent must be between 1 and 60')
        if not 1 <= self.v11_confirm_min <= self.v11_confirm_window <= 60:
            raise ValueError('invalid v11 temporal gate')
        if not 0<=self.v11_identity_min<=self.v11_confirm_window:
            raise ValueError('invalid v11 same-antenna confirmation')
        if self.dominance_ratio < 1 or not np.isfinite(self.dominance_ratio):
            raise ValueError('invalid dominance ratio')
        if not np.isfinite(self.score_threshold) or self.score_threshold < 0:
            raise ValueError('invalid score threshold')


@dataclass(frozen=True)
class FusionDecision:
    array_id: int
    status: str
    advisory_raw_v11: bool
    advisory_temporally_confirmed: bool
    sparse_candidate: bool
    final_alert: bool
    antenna_1based: int | None
    sparse_score: float | None
    dominance_ratio: float | None
    instrument_valid: bool
    action: str = 'NO_HARDWARE_ACTUATION'


class SANADNetworkV32:
    def __init__(self, config: FusionConfig = FusionConfig(), *, cal_dir:str|Path|None=None):
        self.config=config
        self.v29=SANADNetworkV29(cal_dir=cal_dir, recent=config.recent)
        self.sparse_history=[deque(maxlen=config.recent) for _ in range(7)]
        self.v11_history=[deque(maxlen=config.v11_confirm_window) for _ in range(7)]
        self.v11_named_history=[deque(maxlen=config.v11_confirm_window) for _ in range(7)]

    def update(self,vecs:Mapping[int,np.ndarray]):
        out=[]
        source=self.v29.update(vecs)
        for i,d in enumerate(source):
            valid=[v for v in self.v29.window[i] if v is not None]
            allowed=len(self.v29.window[i])>=8 and len(valid)>=max(8,self.config.recent//2) and d.valid
            hits=np.zeros(8,dtype=bool)
            score=dom=None
            if allowed:
                m=np.median(np.asarray(valid).reshape(-1,2,8),axis=0).T/UNIT
                _,s=split_smooth_sparse(m)
                strengths=np.abs(s).sum(axis=1)
                ant=int(strengths.argmax())
                score=float(strengths[ant])
                dom=score/max(float(np.partition(strengths,-2)[-2]),1e-5)
                if score>self.config.score_threshold and dom>=self.config.dominance_ratio:
                    hits[ant]=True
            self.sparse_history[i].append(hits)
            h=np.asarray(self.sparse_history[i],dtype=int)
            c=h.sum(axis=0)
            spatial=bool(c.max()>=self.config.persistence)
            spatial_ant=int(c.argmax())+1 if spatial else None
            self.v11_history[i].append(bool(d.confirmed_v11))
            self.v11_named_history[i].append(int(d.antenna_1based or 0) if d.confirmed_v11 else 0)
            names=np.asarray(self.v11_named_history[i],dtype=int)
            valid_names=names[names>0]
            identity_support=(np.bincount(valid_names,minlength=9)[1:].max()
                              if len(valid_names) else 0)
            confirmed=(sum(self.v11_history[i])>=self.config.v11_confirm_min
                       and identity_support>=self.config.v11_identity_min)
            # V11-only alerts need sustained temporal corroboration: otherwise
            # expose the raw signal in audit but do NOT promote to final alert.
            alert=bool(confirmed or spatial)
            if confirmed:
                status='GATED_V11_ADVISORY'
                name=d.antenna_1based if d.confirmed_v11 and d.antenna_1based else spatial_ant
            elif spatial:
                status='SPARSE_RESEARCH_REVIEW'
                name=spatial_ant
            elif d.confirmed_v11:
                status='RAW_V11_PENDING_CONFIRMATION'
                name=None
            else:
                status='MONITORING' if d.valid else 'INSUFFICIENT_MEASUREMENTS'
                name=None
            out.append(FusionDecision(i,status,bool(d.confirmed_v11),bool(confirmed),
                bool(spatial),alert,name,score,dom,d.valid))
        return out
