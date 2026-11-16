"""SANAD v35: frozen-baseline temporal path + preserved v34 primary, research only.

All evidence paths share covariance-derived gain/phase proxies. Their agreement
is correlated and must not be represented as independent statistical witnesses.
No control, transmission, steering, or physical actuation is implemented.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Mapping
from pathlib import Path
import numpy as np
from sanad_v33 import ElementEvidence,_normal_evidence,_spatial_feature,N_ARRAYS
from sanad_v34 import FusionConfigV34,EvidenceFusionV34,_modal
from sanad_v29 import SANADNetworkV29
from temporal_v35 import FrozenTemporalEvidenceV35

@dataclass(frozen=True)
class FusionConfigV35(FusionConfigV34):
    # Keep the v34-selected pre-existing path unchanged.
    quality_score_multiple: float=1.2
    quality_dominance: float=2.0
    # Frozen temporal anomaly significance in empirical baseline-noise units.
    delta_min_score: float=4.0
    delta_min_dominance: float=1.45
    delta_hits: int=9
    delta_window: int=14
    delta_spatial_hits: int=5  # same-element fast/strict corroboration
    delta_solo_score: float=6.0
    delta_solo_dominance: float=2.15
    delta_solo_hits: int=11
    delta_cohort_review_at: int=4
    base_policy: str='corroborated'  # v34 evidence guarded, not automatically a primary alert

    def __post_init__(self):
        super().__post_init__()
        if not (1<=self.delta_spatial_hits<=self.delta_hits<=self.delta_solo_hits<=self.delta_window<=45):
            raise ValueError('Invalid temporal persistence counts')
        if not 2<=self.delta_cohort_review_at<=N_ARRAYS+1:
            raise ValueError('Invalid cohort control')
        if self.base_policy not in ('corroborated','legacy_union','temporal_only'):
            raise ValueError('Invalid inherited decision policy')
        for k in ('delta_min_score','delta_solo_score','delta_min_dominance',
                  'delta_solo_dominance'):
            x=getattr(self,k)
            if not np.isfinite(x) or x<=0:raise ValueError('Invalid temporal threshold')
        if self.delta_solo_score<self.delta_min_score or self.delta_solo_dominance<self.delta_min_dominance:
            raise ValueError('Solo evidence must be stricter')

@dataclass(frozen=True)
class DecisionV35:
    array_id:int
    final_alert:bool
    review_candidate:bool
    review_only:bool
    antenna_1based:int|None
    status:str
    source:str|None
    delta_persistent:bool
    delta_solo:bool
    delta_supported:bool
    cohort_ambiguous:bool
    evidence_valid:bool
    action:str='NO_HARDWARE_ACTUATION'

class EvidenceFusionV35:
    """Use v34 unchanged; append a guarded temporal rescue channel."""
    def __init__(self,config:FusionConfigV35|None=None):
        self.config=config or FusionConfigV35()
        self.base=EvidenceFusionV34(self.config)
        c=self.config
        self.delta=[deque(maxlen=c.delta_window) for _ in range(N_ARRAYS)]
        self.delta_strong=[deque(maxlen=c.delta_window) for _ in range(N_ARRAYS)]

    def step(self,raw_v11,raw_named,fast_evidence,strict_evidence,delta_evidence,valid=None):
        if len(delta_evidence)!=N_ARRAYS:raise ValueError('Expected 7 temporal evidence elements')
        if valid is None:valid=[True]*N_ARRAYS
        if len(valid)!=N_ARRAYS:raise ValueError('Expected 7 validity flags')
        base=self.base.step(raw_v11,raw_named,fast_evidence,strict_evidence,valid)
        c=self.config;pending=[]
        for a in range(N_ARRAYS):
            d=_normal_evidence(delta_evidence[a]) if bool(valid[a]) else None
            hit=(int(d.element) if d is not None and d.score>=c.delta_min_score
                 and d.dominance>=c.delta_min_dominance else 0)
            strong=(int(d.element) if d is not None and d.score>=c.delta_solo_score
                    and d.dominance>=c.delta_solo_dominance else 0)
            self.delta[a].append(hit)
            self.delta_strong[a].append(strong)
            name,count=_modal(self.delta[a]);solo_count=sum(x==name for x in self.delta_strong[a]) if name else 0
            supported=False
            if name:
                # Secondary fast and strict paths share raw RF proxy; require same ID.
                fast_support=sum(int(x==name) for x in self.base.fast[a])
                strict_support=sum(int(x==name) for x in self.base.strict[a])
                supported=bool(max(fast_support,strict_support)>=c.delta_spatial_hits)
            pending.append(dict(name=name,ready=bool(count>=c.delta_hits),
                                solo=bool(solo_count>=c.delta_solo_hits),supported=supported,
                                base=base[a],valid=bool(valid[a])))
        simultaneous=sum(int(p['ready']) for p in pending if p['valid'])
        out=[]
        for a,p in enumerate(pending):
            b=p['base'];ambiguous=bool(simultaneous>=c.delta_cohort_review_at and p['ready'])
            rescue=bool(p['valid'] and p['ready'] and (p['supported'] or p['solo']) and not ambiguous)
            # Avoid blindly unioning v34 alarms: v34 already exceeded the FAR budget.
            # Retain v34 evidence; only confirm it under predeclared cross-path constraints.
            strict_name,_=_modal(self.base.strict[a])
            v11_strict_agree=bool(b.v11_persistent and b.v32_strict and
                                  strict_name is not None and strict_name==b.antenna_1based)
            temporal_same=bool(p['ready'] and p['name'] is not None and
                               b.antenna_1based==p['name'])
            if c.base_policy=='legacy_union':
                base_confirm=bool(b.final_alert)
            elif c.base_policy=='temporal_only':
                base_confirm=False
            else:
                base_confirm=bool(b.final_alert and (v11_strict_agree or
                                   (temporal_same and not ambiguous)))
            final=bool(base_confirm or rescue)
            review=bool(b.review_candidate or b.final_alert or p['ready'])
            if base_confirm:
                status=b.status;name=b.antenna_1based;source='v34_evidence_with_v35_gate'
            elif rescue:
                status='RESEARCH_ALERT_TEMPORAL';name=p['name'];source=('temporal_spatial_support' if p['supported'] else 'temporal_strong_solo')
            elif review:
                status='REVIEW_ONLY';name=p['name'] if p['ready'] else b.antenna_1based;source='temporal_cohort_ambiguity' if ambiguous else 'unconfirmed_evidence'
            else:
                status='MONITORING' if p['valid'] else 'INSUFFICIENT_MEASUREMENTS';name=None;source=None
            out.append(DecisionV35(a,final,review,bool(review and not final),name,status,source,
                                   p['ready'],p['solo'],p['supported'],ambiguous,p['valid']))
        return out

class SANADNetworkV35:
    def __init__(self,config:FusionConfigV35|None=None,*,cal_dir:Path|str|None=None):
        self.config=config or FusionConfigV35()
        self.source=SANADNetworkV29(cal_dir=cal_dir,recent=max(self.config.strict_window,self.config.fast_window))
        self.temporal=FrozenTemporalEvidenceV35()
        self.fusion=EvidenceFusionV35(self.config)
    def update(self,vecs:Mapping[int,np.ndarray]):
        incoming=dict(vecs)
        parent=self.source.update(incoming)
        delta=self.temporal.step(incoming)
        c=self.config
        fast=[];strict=[]
        for a,d in enumerate(parent):
            if not d.valid:fast.append(None);strict.append(None)
            else:
                w=list(self.source.window[a])
                fast.append(_spatial_feature(w[-c.fast_window:]))
                strict.append(_spatial_feature(w[-c.strict_window:]))
        valid=[bool(parent[a].valid or delta[a] is not None) for a in range(N_ARRAYS)]
        return self.fusion.step([x.confirmed_v11 for x in parent],
                                [x.antenna_1based or 0 for x in parent],fast,strict,delta,valid)
