"""SANAD Network Adaptive v0.1: research-only variable-topology RF integrity prototype.

This is NOT the frozen RC3 trained detector. No threshold or accuracy claim is
transferred from RC3. No RF hardware actuation. Clinical-grade / production-grade
assurance is not provided.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping
import warnings
import numpy as np
from .geometry import decompose


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def _circular_median(a: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    x = _wrap_deg(x[np.isfinite(x)])
    if not len(x):
        return float('nan')
    loss = np.abs(_wrap_deg(x[:, None] - x[None, :])).sum(axis=1)
    return float(x[np.argmin(loss)])


@dataclass(frozen=True)
class ArraySpec:
    array_id: str
    positions: np.ndarray
    neighbors: tuple[str, ...] = ()

    def __post_init__(self):
        p = np.asarray(self.positions, dtype=np.float64)
        if not isinstance(self.array_id, str) or not self.array_id.strip():
            raise ValueError("array_id must be a non-empty string")
        if p.ndim != 2 or not 1 <= p.shape[1] <= 3 or p.shape[0] < 1 or not np.isfinite(p).all():
            raise ValueError("positions must be finite [elements, 1..3]")
        if len(np.unique(p, axis=0)) != len(p):
            raise ValueError("duplicate element coordinates")
        if self.array_id in self.neighbors:
            raise ValueError("self is not a valid neighbor")
        object.__setattr__(self, 'positions', p.copy())
        object.__setattr__(self, 'neighbors', tuple(self.neighbors))

    @property
    def elements(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class RFMeasurement:
    gain_db: np.ndarray
    phase_deg: np.ndarray | None = None
    valid: np.ndarray | None = None

    def as_channels(self, elements: int) -> np.ndarray:
        gain = np.asarray(self.gain_db, dtype=float)
        if gain.shape != (elements,):
            raise ValueError(f"gain must have shape ({elements},)")
        ph = np.full(elements, np.nan) if self.phase_deg is None else np.asarray(self.phase_deg, dtype=float)
        if ph.shape != (elements,):
            raise ValueError(f"phase must have shape ({elements},)")
        valid = np.ones(elements, dtype=bool) if self.valid is None else np.asarray(self.valid, dtype=bool)
        if valid.shape != (elements,):
            raise ValueError(f"valid mask must have shape ({elements},)")
        if np.any(valid & ~np.isfinite(gain)) or np.isinf(ph).any():
            raise ValueError("invalid gain or phase")
        out = np.column_stack([gain, ph])
        out[~valid, :] = np.nan
        return out


@dataclass(frozen=True)
class AdaptiveConfig:
    baseline_frames: int = 24
    min_baseline_samples: int = 16
    window: int = 8
    min_window_samples: int = 4
    gain_unit_db: float = 0.5
    phase_unit_deg: float = 5.0
    sparse_lambda: float = 0.18
    solver_damping: float = 0.65
    solver_max_iter: int = 120
    # Pure engineering DEMO thresholds, NOT calibrated RC3 thresholds.
    local_score_threshold: float = 0.55
    common_mode_review_threshold: float = 1.25
    min_local_hits: int = 3
    hit_window: int = 5

    def __post_init__(self):
        if not (2 <= self.min_baseline_samples <= self.baseline_frames):
            raise ValueError('baseline settings invalid')
        if not (1 <= self.min_window_samples <= self.window):
            raise ValueError('window settings invalid')
        if not (1 <= self.min_local_hits <= self.hit_window):
            raise ValueError('hit settings invalid')
        for name in ('gain_unit_db','phase_unit_deg','sparse_lambda','solver_damping',
                     'local_score_threshold','common_mode_review_threshold'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be finite positive')
        if self.solver_damping > 1 or self.solver_max_iter < 1:
            raise ValueError('solver settings invalid')


@dataclass(frozen=True)
class ArrayDecision:
    array_id: str
    element_count: int
    status: str
    element_scores: tuple[float | None, ...]
    candidate_element: int | None   # 0-based; same stable ordering as specification
    score: float | None
    dominance: float | None
    common_mode_score: float | None
    phase_observed: tuple[bool, ...]
    valid_elements: int
    spatial_observable: bool
    solver_converged: bool | None
    solver_iterations: int | None
    neighbor_score_contrast: float | None = None
    actuation: str = 'NONE_RESEARCH_ONLY'


@dataclass
class _State:
    frames_seen: int = 0
    commissioning: list[np.ndarray] = field(default_factory=list)
    reference: np.ndarray | None = None
    commissioning_failed: bool = False
    recent: deque = field(default_factory=deque)
    hits: deque = field(default_factory=deque)


class SANADAdaptiveNetwork:
    """Stateful per-report detector for heterogeneous RF arrays.

    Methods accept calibrated per-element gain and phase proxy vectors. A stable
    reference is learned only from the initial commissioning window, then frozen.
    One-sided, untrusted or missing reports cannot update the reference.
    Array graph context is advisory only until validated prospectively.
    """
    def __init__(self, specs: list[ArraySpec], config: AdaptiveConfig | None = None):
        self.config = config or AdaptiveConfig()
        self.specs: dict[str, ArraySpec] = {}
        self.states: dict[str, _State] = {}
        self.period = 0
        for spec in specs:
            self.register(spec)

    def register(self, spec: ArraySpec) -> None:
        """Allow a new array to enter commissioning without resetting existing arrays."""
        if spec.array_id in self.specs:
            raise ValueError('duplicate array id')
        self.specs[spec.array_id] = spec
        self.states[spec.array_id] = _State(
            recent=deque(maxlen=self.config.window),
            hits=deque(maxlen=self.config.hit_window),
        )

    def _commission(self, st: _State, raw: np.ndarray, elements: int) -> str:
        c = self.config
        st.commissioning.append(raw)
        if st.frames_seen < c.baseline_frames:
            return 'LEARNING'
        h = np.asarray(st.commissioning)
        valid_counts = np.isfinite(h).sum(axis=0)
        baseline = np.full((elements, 2), np.nan)
        for i in range(elements):
            for k in range(2):
                if valid_counts[i,k] >= c.min_baseline_samples:
                    vals = h[:,i,k]
                    baseline[i,k] = float(np.nanmedian(vals)) if k == 0 else _circular_median(vals)
        ready = int(np.isfinite(baseline[:,0]).sum()) >= 1
        st.reference = baseline if ready else None
        st.commissioning_failed = not ready
        st.commissioning.clear()
        return 'BASELINE_FROZEN' if ready else 'INSUFFICIENT_COMMISSIONING'

    def update(self, measurements: Mapping[str, RFMeasurement]) -> dict[str, ArrayDecision]:
        """Advance one report. Unknown arrays raise; missing registered arrays are masked."""
        if not set(measurements).issubset(self.specs):
            raise ValueError('unknown array id: call register first')
        c = self.config
        results: dict[str, ArrayDecision] = {}
        for id, spec in self.specs.items():
            st = self.states[id]
            n = spec.elements
            raw = (measurements[id].as_channels(n) if id in measurements
                   else np.full((n,2), np.nan))
            st.frames_seen += 1
            if st.reference is None:
                # A failed commissioning window does not get relabelled healthy:
                # freezing a later contaminated period could hide persistent drift.
                if st.commissioning_failed:
                    status = 'INSUFFICIENT_COMMISSIONING'
                elif st.frames_seen <= c.baseline_frames:
                    status = self._commission(st, raw, n)
                else:
                    status = 'INSUFFICIENT_COMMISSIONING'
                results[id] = ArrayDecision(id,n,status,(None,)*n,None,None,None,None,
                                            (False,)*n,int(np.isfinite(raw[:,0]).sum()),
                                            False,None,None)
                continue
            delta = raw - st.reference
            delta[:,1] = _wrap_deg(delta[:,1])
            delta[:,0] /= c.gain_unit_db
            delta[:,1] /= c.phase_unit_deg
            st.recent.append(delta)
            hist = np.asarray(st.recent)
            observed = np.sum(np.isfinite(hist), axis=0) >= c.min_window_samples
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                avg = np.nanmedian(hist, axis=0)
            avg[~observed] = np.nan
            valid_gain = np.isfinite(avg[:,0])
            valid_phase = np.isfinite(avg[:,1]) & valid_gain
            if valid_gain.sum() < 3:
                st.hits.append(False)
                results[id] = ArrayDecision(id,n,'INSUFFICIENT_DATA',(None,)*n,None,None,None,None,
                                            tuple(bool(v) for v in valid_phase),int(valid_gain.sum()),
                                            False,None,None)
                continue
            solver = decompose(avg,spec.positions,valid=np.isfinite(avg),
                               lam=c.sparse_lambda,damping=c.solver_damping,
                               max_iter=c.solver_max_iter)
            scores=np.nansum(np.abs(solver.sparse),axis=1)
            scores[~valid_gain] = np.nan
            local_observable=bool(solver.observable[0])
            if local_observable:
                candidate=int(np.nanargmax(scores))
                strength=float(scores[candidate])
                runner=float(np.nanmax(np.delete(scores,candidate))) if n>1 and np.isfinite(np.delete(scores,candidate)).any() else 0.0
                dominance=strength/max(runner,0.10)
                has_local=strength>=c.local_score_threshold
                st.hits.append(bool(has_local))
            else:
                candidate=None;strength=None;dominance=None
                st.hits.append(False)
            cm=float(np.nanmedian(np.abs(solver.smooth[valid_gain,0])))
            if not local_observable:
                status='UNOBSERVABLE_LOCALIZATION'
            elif not solver.converged:
                status='SOLVER_NOT_CONVERGED'
            elif sum(st.hits)>=c.min_local_hits and has_local:
                status='LOCAL_RESEARCH_ALERT'
            elif has_local:
                status='LOCAL_REVIEW'
            elif cm>=c.common_mode_review_threshold:
                status='COMMON_MODE_REVIEW'
            else:
                status='MONITORING'
            results[id] = ArrayDecision(
                id,n,status,tuple(float(v) if np.isfinite(v) else None for v in scores),
                candidate if has_local and local_observable else None,
                strength,dominance,cm,
                tuple(bool(v) for v in valid_phase),int(valid_gain.sum()),
                local_observable,solver.converged,solver.iterations)
        # Graph context: advisory, never overrides local decisions.
        final={}
        for id,d in results.items():
            ns=[results[x].score for x in self.specs[id].neighbors
                if x in results and results[x].score is not None]
            contrast=None if d.score is None or not ns else d.score-float(np.median(ns))
            final[id]=ArrayDecision(**{**d.__dict__,'neighbor_score_contrast':contrast})
        self.period += 1
        return final
