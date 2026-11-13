"""SANAD Network v33 — research-only multi-path evidence fusion.

Inputs are seven calibrated, aligned *proxy* 16-vectors per report (8 gains +
8 phases), NOT raw RSSI/covariance diagonals or instrument telemetry.

Shared parent: frozen SANADNetworkV29.  Path A retains v11 raw advisory;
path B reuses the sensitivity rule of v30 (8/12, 1.5 dominance);
path C reuses the strict spatial identity rule of v32 (10/15, 1.8).
All three share data and are NOT independent detector votes.

No transmit, control, beam steering, or hardware retuning commands.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping
import numpy as np
from sanad_v29 import SANADNetworkV29
from v29_research import UNIT, split_smooth_sparse

N_ARRAYS=7
N_ANTENNAS=8
SCORE_THRESHOLD=0.7687096042171566

@dataclass(frozen=True)
class FusionConfigV33:
    score_threshold: float = SCORE_THRESHOLD
    fast_dominance: float = 1.5
    fast_hits: int = 8
    fast_window: int = 12
    strict_dominance: float = 1.8
    strict_hits: int = 10
    strict_window: int = 15
    v11_hits: int = 5
    v11_window: int = 8
    v11_identity_hits: int = 3
    # Promotion of spatial evidence requires at least this many raw v11
    # indications in its recent window, AND agreement on the antenna identity.
    corroboration_hits: int = 2
    corroboration_window: int = 12
    identity_agreement: bool = True

    def __post_init__(self):
        for k,w,label in [(self.fast_hits,self.fast_window,'fast'),
                          (self.strict_hits,self.strict_window,'strict'),
                          (self.v11_hits,self.v11_window,'v11')]:
            if not (1<=k<=w<=60):raise ValueError(f'Invalid {label} window')
        if not 1<=self.corroboration_window<=60:raise ValueError('corroboration_window')
        if not 0<=self.corroboration_hits<=self.corroboration_window:raise ValueError('corroboration_hits')
        if not 0<=self.v11_identity_hits<=self.v11_window:raise ValueError('v11_identity_hits')
        if not (np.isfinite(self.score_threshold) and self.score_threshold>=0):raise ValueError('score_threshold')
        if not (np.isfinite(self.fast_dominance) and self.fast_dominance>=1):raise ValueError('fast_dominance')
        if not (np.isfinite(self.strict_dominance) and self.strict_dominance>=self.fast_dominance):raise ValueError('strict_dominance')

@dataclass(frozen=True)
class ElementEvidence:
    """Score/dominance and candidate element from one causal window."""
    score: float | None
    dominance: float | None
    element: int | None  # 1-based, not a certified physical fault

@dataclass(frozen=True)
class FusionDecisionV33:
    array_id: int
    status: str
    raw_v11: bool
    confirmed_v11: bool
    fast_v30: bool
    strict_v32: bool
    spatial_corroborated: bool
    review_candidate: bool
    review_only: bool
    final_alert: bool
    antenna_1based: int | None
    antenna_source: str | None
    evidence_valid: bool
    action: str = 'NO_HARDWARE_ACTUATION'


def _normal_evidence(x:ElementEvidence|tuple|None):
    if x is None:return None
    if not isinstance(x,ElementEvidence):x=ElementEvidence(*x)
    if x.score is None or x.dominance is None or x.element is None:return None
    if not (np.isfinite(x.score) and np.isfinite(x.dominance)):
        return None
    if not (1<=int(x.element)<=N_ANTENNAS) or x.element!=int(x.element):
        return None
    return x


def _modal_nonzero(values):
    counts=np.bincount(np.asarray([v for v in values if 1<=v<=N_ANTENNAS],dtype=int),minlength=9)
    if not counts[1:].any():return None,0
    return int(np.argmax(counts[1:])+1),int(counts[1:].max())


class EvidenceFusionV33:
    """Pure, causal decision layer; online and replay call exactly this class."""
    def __init__(self, config:FusionConfigV33|None=None):
        self.config=config or FusionConfigV33()
        maxw=max(self.config.fast_window,self.config.strict_window,
                 self.config.v11_window,self.config.corroboration_window)
        self.raw=[deque(maxlen=maxw) for _ in range(N_ARRAYS)]
        self.names=[deque(maxlen=maxw) for _ in range(N_ARRAYS)]
        self.fast=[deque(maxlen=self.config.fast_window) for _ in range(N_ARRAYS)]
        self.strict=[deque(maxlen=self.config.strict_window) for _ in range(N_ARRAYS)]
        self.period=0

    def step(self, raw_v11, raw_named, fast_evidence, strict_evidence, valid=None):
        if len(raw_v11)!=N_ARRAYS or len(raw_named)!=N_ARRAYS or len(fast_evidence)!=N_ARRAYS or len(strict_evidence)!=N_ARRAYS:
            raise ValueError('Expected seven arrays per period')
        if valid is None:valid=[True]*N_ARRAYS
        if len(valid)!=N_ARRAYS:raise ValueError('Expected seven validity flags')
        cfg=self.config
        out=[]
        for a in range(N_ARRAYS):
            raw=bool(raw_v11[a]);named=int(raw_named[a] or 0)
            if named<0 or named>N_ANTENNAS:raise ValueError('v11 antenna id must be 0..8')
            self.raw[a].append(raw)
            self.names[a].append(named if raw else 0)
            fe=_normal_evidence(fast_evidence[a])
            se=_normal_evidence(strict_evidence[a])
            hitfast=0 if fe is None or fe.score<=cfg.score_threshold or fe.dominance<cfg.fast_dominance else int(fe.element)
            hitstrict=0 if se is None or se.score<=cfg.score_threshold or se.dominance<cfg.strict_dominance else int(se.element)
            self.fast[a].append(hitfast)
            self.strict[a].append(hitstrict)
            raw_hist=list(self.raw[a]); name_hist=list(self.names[a]);
            v11_count=sum(raw_hist[-cfg.v11_window:])
            v11_name,v11_support=_modal_nonzero(name_hist[-cfg.v11_window:])
            v11_confirmed=(v11_count>=cfg.v11_hits and v11_support>=cfg.v11_identity_hits)
            fast_name,fast_count=_modal_nonzero(self.fast[a])
            strict_name,strict_count=_modal_nonzero(self.strict[a])
            v30=bool(fast_count>=cfg.fast_hits)
            v32=bool(strict_count>=cfg.strict_hits)
            v11_partial=sum(raw_hist[-cfg.corroboration_window:])>=cfg.corroboration_hits
            raw_name,_=_modal_nonzero(name_hist[-cfg.corroboration_window:])
            matched=(not cfg.identity_agreement or (raw_name is not None and raw_name==strict_name))
            # v30 and v32 reuse the SAME spatial evidence and therefore cannot
            # be counted as two independent witnesses. Require independent-style
            # v11 residual evidence for promotion of this channel.
            corroborated=bool(v30 and v32 and v11_partial and matched)
            final=bool(v11_confirmed or corroborated)
            review=bool(v30 or v32 or raw)
            if final:
                if v11_confirmed:
                    status='CONFIRMED_RESEARCH_ALERT_V11';name=v11_name;origin='v11_repeated_identity'
                else:
                    status='CONFIRMED_RESEARCH_ALERT_CORROBORATED';name=strict_name;origin='spatial_with_v11_support'
            elif review:
                status='REVIEW_ONLY';name=strict_name if v32 else fast_name if v30 else None
                origin='strict_spatial' if v32 else 'fast_spatial' if v30 else None
            else:
                status='MONITORING' if valid[a] else 'INSUFFICIENT_MEASUREMENTS'
                name=None;origin=None
            out.append(FusionDecisionV33(a,status,raw,v11_confirmed,v30,v32,corroborated,
                                         review,bool(review and not final),final,name,origin,bool(valid[a])))
        self.period+=1
        return out


def _spatial_feature(window, *, score_threshold=0.0):
    """Same sparse residual calculation used in v30/v32, applied causally."""
    valid=[v for v in window if v is not None]
    if len(window)<8 or len(valid)<max(8,len(window)//2):return None
    m=np.median(np.asarray(valid).reshape(-1,2,8),axis=0).T/UNIT
    _,sp=split_smooth_sparse(m)
    strengths=np.abs(sp).sum(axis=1)
    a=int(strengths.argmax())
    val=float(strengths[a]);runner_up=float(np.partition(strengths,-2)[-2])
    return ElementEvidence(val,val/max(runner_up,1e-5),a+1)


class SANADNetworkV33:
    """Use frozen v29/v11 parent once; derive parallel fast and strict evidence."""
    def __init__(self, config:FusionConfigV33|None=None, *, cal_dir:str|Path|None=None):
        self.config=config or FusionConfigV33()
        self.source=SANADNetworkV29(cal_dir=cal_dir, recent=max(self.config.strict_window,self.config.fast_window))
        self.fusion=EvidenceFusionV33(self.config)
    def update(self,vecs:Mapping[int,np.ndarray]):
        parents=self.source.update(vecs)
        cfg=self.config
        fast=[];strict=[]
        for i,d in enumerate(parents):
            w=list(self.source.window[i])
            if not d.valid:
                fast.append(None);strict.append(None);continue
            fast.append(_spatial_feature(w[-cfg.fast_window:]))
            strict.append(_spatial_feature(w[-cfg.strict_window:]))
        return self.fusion.step([d.confirmed_v11 for d in parents],
                                [d.antenna_1based or 0 for d in parents],fast,strict,
                                [d.valid for d in parents])
