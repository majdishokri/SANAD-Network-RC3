"""SANAD Network v34 -- modular research-only RF proxy assurance candidate.

Engineering hypothesis (NOT performance-validated): preserve frozen v11 raw
anomaly evidence, v30 short-window sensitive spatial evidence, and v32
long-window persistent antenna localization, while decoupling the confirmation
rules that prevented v33 from promoting most spatial events.

All paths use correlated representations of the *same* original 16-dimensional
covariance-derived proxy (eight gain + eight phase coordinates), so path
agreement must never be described as independent corroboration. This module
cannot infer actual calibrated hardware RF errors or control radio equipment.

Inference never receives injected-fault labels. A separate frozen protocol
assesses the research alert channel using new NVIDIA Sionna seeds.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from sanad_v29 import SANADNetworkV29
from sanad_v33 import ElementEvidence, _normal_evidence, _spatial_feature, N_ARRAYS, N_ANTENNAS, SCORE_THRESHOLD


@dataclass(frozen=True)
class FusionConfigV34:
    # Frozen historical component definitions. These are shared, NOT votes.
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

    # v34 policy: permit high-quality spatial path without requiring raw v11;
    # v11-supported evidence provides a *separate* route. No independence claim.
    quality_score_multiple: float = 1.40
    quality_dominance: float = 2.10
    quality_hits: int = 6
    partial_v11_hits: int = 2
    partial_v11_window: int = 12
    fast_rescue: bool = False
    rescue_quality_hits: int = 8
    # A large simultaneous array-wide event is ambiguous; request review.
    cohort_review_at: int = 4

    def __post_init__(self):
        for hits, win, name in (
            (self.fast_hits,self.fast_window,'fast'),
            (self.strict_hits,self.strict_window,'strict'),
            (self.v11_hits,self.v11_window,'v11'),
            (self.quality_hits,self.strict_window,'quality'),
            (self.partial_v11_hits,self.partial_v11_window,'partial v11'),
            (self.rescue_quality_hits,self.strict_window,'rescue quality'),
        ):
            if not (1<=hits<=win<=60):
                raise ValueError(f'Invalid {name} count/window: {hits}/{win}')
        if not (0<=self.v11_identity_hits<=self.v11_window):
            raise ValueError('v11_identity_hits')
        if not (2<=self.cohort_review_at<=N_ARRAYS+1):
            raise ValueError('cohort_review_at')
        for k in ('score_threshold','fast_dominance','strict_dominance',
                  'quality_score_multiple','quality_dominance'):
            x=getattr(self,k)
            if not np.isfinite(x):raise ValueError(f'{k} must be finite')
        if self.score_threshold<=0 or self.fast_dominance<1 or self.strict_dominance<self.fast_dominance:
            raise ValueError('Bad basic score/dominance')
        if self.quality_score_multiple<1 or self.quality_dominance<self.strict_dominance:
            raise ValueError('Quality criteria must be at least strict spatial criteria')


def _modal(xs:Sequence[int]):
    counts=np.bincount(np.asarray([int(x) for x in xs if 1<=int(x)<=8],dtype=int),minlength=9)
    if not np.any(counts[1:]):return None,0
    return int(np.argmax(counts[1:])+1),int(np.max(counts[1:]))


@dataclass(frozen=True)
class FusionDecisionV34:
    array_id: int
    status: str
    final_alert: bool
    review_candidate: bool
    review_only: bool
    raw_v11: bool
    v11_persistent: bool
    v30_fast: bool
    v32_strict: bool
    spatial_quality: bool
    spatial_correlated_agreement: bool
    supported_by_v11: bool
    cohort_ambiguous: bool
    antenna_1based: int | None
    antenna_source: str | None
    evidence_valid: bool
    action: str = 'NO_HARDWARE_ACTUATION'


class EvidenceFusionV34:
    """Causal three-module LEGO adapter and explicit arbitration policy.

    v11 = conservative independent *algorithm* (not independent RF data)
    v30 = fast candidate on short window
    v32 = persistent identity on long window
    v34 = arbitration of quality, identity, raw evidence and array-wide ambiguity
    """
    def __init__(self,config:FusionConfigV34|None=None):
        self.config=config or FusionConfigV34()
        c=self.config
        maxw=max(c.v11_window,c.partial_v11_window)
        self.raw=[deque(maxlen=maxw) for _ in range(N_ARRAYS)]
        self.raw_names=[deque(maxlen=maxw) for _ in range(N_ARRAYS)]
        self.fast=[deque(maxlen=c.fast_window) for _ in range(N_ARRAYS)]
        self.strict=[deque(maxlen=c.strict_window) for _ in range(N_ARRAYS)]
        self.quality=[deque(maxlen=c.strict_window) for _ in range(N_ARRAYS)]
        self.period=0

    def step(self,raw_v11,raw_named,fast_evidence,strict_evidence,valid=None):
        for k,x in [('raw_v11',raw_v11),('raw_named',raw_named),('fast',fast_evidence),('strict',strict_evidence)]:
            if len(x)!=N_ARRAYS:raise ValueError(f'{k} needs {N_ARRAYS} arrays')
        if valid is None:valid=[True]*N_ARRAYS
        if len(valid)!=N_ARRAYS:raise ValueError('valid needs 7 arrays')
        # Fail closed on malformed flags instead of letting bool(np.nan) turn into True.
        for a in range(N_ARRAYS):
            if valid[a] not in (True,False,0,1):
                raise ValueError(f'Invalid sample-valid flag for array {a}')
            if raw_v11[a] not in (True,False,0,1):
                raise ValueError(f'Invalid v11 alert flag for array {a}')
        c=self.config
        states=[]
        for a in range(N_ARRAYS):
            ok=bool(valid[a]);raw=bool(raw_v11[a]) if ok else False
            named=int(raw_named[a] or 0)
            if not 0<=named<=N_ANTENNAS or named!=raw_named[a] and raw_named[a] is not None:
                raise ValueError('Raw named element ID must be integer 0..8')
            fe=_normal_evidence(fast_evidence[a]) if ok else None
            se=_normal_evidence(strict_evidence[a]) if ok else None
            fast=(int(fe.element) if (fe is not None and fe.score>c.score_threshold
                                    and fe.dominance>=c.fast_dominance) else 0)
            strict=(int(se.element) if (se is not None and se.score>c.score_threshold
                                        and se.dominance>=c.strict_dominance) else 0)
            quality=(strict if (se is not None and se.score>c.score_threshold*c.quality_score_multiple
                                and se.dominance>=c.quality_dominance) else 0)
            self.raw[a].append(raw)
            self.raw_names[a].append(named if raw else 0)
            self.fast[a].append(fast)
            self.strict[a].append(strict)
            self.quality[a].append(quality)
            rn,support=_modal(list(self.raw_names[a])[-c.v11_window:])
            stable_v11=(sum(list(self.raw[a])[-c.v11_window:])>=c.v11_hits
                        and support>=c.v11_identity_hits and rn is not None)
            fn,fc=_modal(self.fast[a]);sn,sc=_modal(self.strict[a])
            _,qc=_modal(self.quality[a])
            vn,vn_support=_modal(list(self.raw_names[a])[-c.partial_v11_window:])
            # v11 evidence may support only the *same* physical identity.
            partial=(sum(list(self.raw[a])[-c.partial_v11_window:])>=c.partial_v11_hits
                     and vn is not None)
            v30=fc>=c.fast_hits
            v32=sc>=c.strict_hits
            aligned=bool(v30 and v32 and fn==sn and sn is not None)
            qs=sum(int(x==sn) for x in self.quality[a]) if sn else 0
            quality_ok=bool(aligned and qs>=c.quality_hits)
            v11_help=bool(aligned and partial and vn==sn)
            rescue=bool(c.fast_rescue and v30 and fn is not None and partial and vn==fn
                        and sum(int(x==fn) for x in self.quality[a])>=c.rescue_quality_hits)
            states.append(dict(ok=ok,raw=raw,rn=rn,stable_v11=stable_v11,
                               fn=fn,sn=sn,v30=v30,v32=v32,quality=quality_ok,
                               agreement=aligned,v11_help=v11_help,rescue=rescue,
                               anyhit=bool(v30 or v32 or raw)))

        # Correlated network-wide drift can create many simultaneous spatial
        # hits. Do not assert element-specific RF failure from that pattern.
        num_spatial=sum(s['agreement'] for s in states if s['ok'])
        out=[]
        for a,s in enumerate(states):
            cohort=bool(num_spatial>=c.cohort_review_at and s['agreement'])
            spatial=bool((s['quality'] or s['v11_help'] or s['rescue']) and not cohort)
            primary=bool(s['ok'] and (s['stable_v11'] or spatial))
            review=bool(s['anyhit'] or s['agreement'] or cohort)
            if not s['ok']:
                status='INSUFFICIENT_MEASUREMENTS';name=None;origin=None
                primary=False
                review=False
            elif s['stable_v11']:
                status='CONFIRMED_RESEARCH_ALERT_V11';name=s['rn'];origin='v11_repeated_identity'
            elif spatial:
                status='CONFIRMED_RESEARCH_ALERT_SPATIAL';name=s['sn'] if s['agreement'] else s['fn']
                origin=('spatial_with_v11_support' if s['v11_help'] else
                        'spatial_quality_correlated' if s['quality'] else 'fast_rescue')
            elif review:
                status='REVIEW_ONLY';name=s['sn'] if s['v32'] else s['fn'] if s['v30'] else s['rn']
                origin='cohort_ambiguity' if cohort else 'spatial_candidate_or_raw_v11'
            else:
                status='MONITORING';name=None;origin=None
            out.append(FusionDecisionV34(
                array_id=a,status=status,final_alert=primary,
                review_candidate=review,review_only=bool(review and not primary),
                raw_v11=s['raw'],v11_persistent=s['stable_v11'],
                v30_fast=s['v30'],v32_strict=s['v32'],spatial_quality=s['quality'],
                spatial_correlated_agreement=s['agreement'],supported_by_v11=s['v11_help'],
                cohort_ambiguous=cohort,antenna_1based=name,antenna_source=origin,
                evidence_valid=s['ok']))
        self.period+=1
        return out


class SANADNetworkV34:
    """Online wrapper; exact same fusion.step function is used by replay."""
    def __init__(self,config:FusionConfigV34|None=None,*,cal_dir:str|Path|None=None):
        self.config=config or FusionConfigV34()
        self.source=SANADNetworkV29(cal_dir=cal_dir,recent=max(self.config.strict_window,self.config.fast_window))
        self.fusion=EvidenceFusionV34(self.config)

    def update(self,vecs:Mapping[int,np.ndarray]):
        parent=self.source.update(vecs)
        c=self.config
        feats_fast=[];feats_strict=[]
        for a,d in enumerate(parent):
            if not d.valid:
                feats_fast.append(None);feats_strict.append(None)
            else:
                hist=list(self.source.window[a])
                feats_fast.append(_spatial_feature(hist[-c.fast_window:]))
                feats_strict.append(_spatial_feature(hist[-c.strict_window:]))
        return self.fusion.step(
            [x.confirmed_v11 for x in parent],
            [x.antenna_1based or 0 for x in parent],
            feats_fast,feats_strict,[x.valid for x in parent])
