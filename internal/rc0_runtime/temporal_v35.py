"""Causal per-element frozen-baseline RF-proxy drift evidence (research only).

Each array learns its own 16-d gain/phase proxy under an assumed healthy
first-120-report commissioning window. No future reports, labels or fault
injection metadata enter inference. This is not calibrated hardware telemetry.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from sanad_v33 import ElementEvidence

N_ARRAYS=7
N_ANT=8
N_DIM=16
BASELINE_END=120
# Robust channel-wide magnitude and gradient removal, per gain/phase channel.
POSITIONS=np.arange(8,dtype=float)
X=np.column_stack((np.ones(8),POSITIONS-POSITIONS.mean()))
PROJECTOR=np.eye(8)-X@np.linalg.pinv(X)
UNIT=np.array([0.5,5.0],dtype=float)


@dataclass(frozen=True)
class TemporalConfigV35:
    baseline_end: int=BASELINE_END
    min_baseline: int=24
    window: int=12
    min_window_valid: int=8
    noise_floor: float=0.16

    def __post_init__(self):
        if not (60<=self.baseline_end<=120 and 8<=self.min_baseline<=self.baseline_end-60):
            raise ValueError('Invalid frozen healthy baseline window')
        if not (4<=self.min_window_valid<=self.window<=45):
            raise ValueError('Invalid causal time window')
        if not (np.isfinite(self.noise_floor) and .02<=self.noise_floor<=3):
            raise ValueError('Invalid noise floor')


class FrozenTemporalEvidenceV35:
    """Identical stateful implementation for online and offline replay."""
    def __init__(self,config:TemporalConfigV35|None=None):
        self.config=config or TemporalConfigV35()
        self.period=0
        self.learn=[[] for _ in range(N_ARRAYS)]
        self.ref=[None]*N_ARRAYS
        self.scale=[None]*N_ARRAYS
        self.window=[deque(maxlen=self.config.window) for _ in range(N_ARRAYS)]

    @staticmethod
    def _residual(raw):
        a=np.asarray(raw,dtype=float)
        if a.shape!=(16,) or not np.isfinite(a).all():
            raise ValueError('Expected finite calibrated 16-vector')
        scaled=a.reshape(2,8)/UNIT[:,None]
        return (scaled@PROJECTOR.T).T  # [element, gain/phase], detrended

    def step(self,raw_vectors:dict[int,np.ndarray]|None):
        vecs={} if raw_vectors is None else dict(raw_vectors)
        if any(type(k)!=int or not 0<=k<N_ARRAYS for k in vecs):
            raise ValueError('Invalid array index')
        clean={i:self._residual(v) for i,v in vecs.items()}
        c=self.config
        out=[]
        for a in range(N_ARRAYS):
            x=clean.get(a)
            if self.period<c.baseline_end:
                if self.period>=60 and x is not None:
                    self.learn[a].append(x)
                out.append(None)
                continue
            if self.ref[a] is None and self.period==c.baseline_end:
                if len(self.learn[a])>=c.min_baseline:
                    arr=np.array(self.learn[a],dtype=float)
                    self.ref[a]=np.median(arr,axis=0)
                    mad=np.median(np.abs(arr-self.ref[a]),axis=0)*1.4826
                    self.scale[a]=np.maximum(mad,c.noise_floor)
                self.learn[a].clear()
            if x is None or self.ref[a] is None:
                self.window[a].append(None)
                out.append(None)
                continue
            self.window[a].append(x)
            valid=[v for v in self.window[a] if v is not None]
            if len(valid)<c.min_window_valid:
                out.append(None)
                continue
            # Robust time median of residuals. No adaptive learning past report 119.
            delta=np.median(valid,axis=0)-self.ref[a]
            normalized=delta/self.scale[a]
            mag=np.sqrt(np.sum(normalized**2,axis=1))
            top=int(np.argmax(mag));second=float(np.partition(mag,-2)[-2]);score=float(mag[top])
            dom=score/max(second,.10)
            out.append(ElementEvidence(score,dom,top+1))
        self.period+=1
        return out


def extract_delta(vectors):
    """Deterministic 300-report trace, no ground-truth arguments."""
    if set(vectors)!=set(range(N_ARRAYS)) or any(len(vectors[a])!=300 for a in range(N_ARRAYS)):
        raise ValueError('Expected 7 arrays x 300 reports')
    engine=FrozenTemporalEvidenceV35()
    feats=np.full((300,N_ARRAYS,3),np.nan,dtype=np.float32)
    for t in range(300):
        out=engine.step({a:vectors[a][t] for a in range(N_ARRAYS) if vectors[a][t] is not None})
        for a,v in enumerate(out):
            if v is not None:
                feats[t,a]=(v.score,v.dominance,v.element)
    return feats
