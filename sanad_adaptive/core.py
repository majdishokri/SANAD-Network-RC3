"""Dimension-adaptive covariance integrity research core.

Research prototype: NOT the trained/frozen SANAD Network RC3 detector.
No transmitter or RF hardware commands. Requires a known-healthy baseline.
The same code supports variable number of arrays and elements per array,
but a shared statistical model must be trained and validated separately.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence
import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class ArrayGeometry:
    """Optional measured element positions; never infer layout from matrix shape.

    positions is M x D and may be specified in metres. None means unknown geometry.
    """
    positions: np.ndarray | None = None
    frequency_hz: float | None = None
    topology: str = "unknown"

    def validate(self, elements: int) -> None:
        if self.positions is not None:
            pos = np.asarray(self.positions, dtype=float)
            if pos.ndim != 2 or pos.shape[0] != elements or pos.shape[1] not in (1, 2, 3):
                raise ValueError(f"positions must have shape ({elements}, 1|2|3)")
            if not np.isfinite(pos).all():
                raise ValueError("non-finite antenna positions")
            if len(np.unique(pos, axis=0)) != elements:
                raise ValueError("antenna positions must be unique")
        if self.frequency_hz is not None and not (float(self.frequency_hz) > 0):
            raise ValueError("frequency_hz must be positive")


@dataclass(frozen=True)
class DetectorConfig:
    gain_floor_db: float = 0.25
    phase_floor_deg: float = 2.5
    coherence_min: float = 0.12
    reference_quantile: float = 0.99
    calibration_multiplier: float = 1.5
    channel_floor: float = 0.04
    persistence_hits: int = 3
    persistence_window: int = 5
    min_reference_frames: int = 24
    min_calibration_frames: int = 16

    def __post_init__(self):
        if not (0 < self.coherence_min < 1 and 0.5 < self.reference_quantile < 1):
            raise ValueError("Invalid coherence_min or reference_quantile")
        if min(self.gain_floor_db, self.phase_floor_deg, self.calibration_multiplier,
               self.channel_floor) <= 0:
            raise ValueError("Floors and multiplier must be positive")
        if not 1 <= self.persistence_hits <= self.persistence_window:
            raise ValueError("Invalid persistence")
        if min(self.min_reference_frames, self.min_calibration_frames) < 4:
            raise ValueError("Insufficient minimum calibration frames")


def covariance_from_csi(csi: np.ndarray) -> np.ndarray:
    """Input [T,M,K] complex CSI, output [T,M,M] per-array covariance.

    Subcarriers are used to form a covariance estimate; they are NOT treated
    as K independent temporal channel realizations.
    """
    h = np.asarray(csi)
    if h.ndim != 3 or h.shape[1] < 1 or h.shape[2] < 1 or not np.iscomplexobj(h):
        raise ValueError("CSI must be complex [time,elements,subcarriers]")
    if not np.isfinite(h).all():
        raise ValueError("CSI contains nonfinite samples")
    return (h @ np.swapaxes(h.conj(), -1, -2)) / h.shape[2]


def validate_covariances(cov: np.ndarray, *, check_psd: bool = True) -> np.ndarray:
    c = np.asarray(cov)
    if c.ndim != 3 or c.shape[1] < 1 or c.shape[1] != c.shape[2]:
        raise ValueError("Covariance must have shape [time,M,M] with M>=1")
    if not np.iscomplexobj(c) or not np.isfinite(c).all():
        raise ValueError("Covariance must be complex and finite")
    if np.max(np.abs(c - np.swapaxes(c.conj(), -1, -2))) > 1e-6 * max(1, np.max(abs(c))):
        raise ValueError("Covariance is not Hermitian")
    d = np.diagonal(c, axis1=-2, axis2=-1).real
    if np.any(d <= 0):
        raise ValueError("Positive antenna power is required")
    if check_psd:
        eig = np.linalg.eigvalsh(c)
        if np.any(eig[:, 0] < -1e-6 * np.maximum(1., np.max(abs(eig), axis=1))):
            raise ValueError("Covariance is not positive semidefinite")
    return c.astype(np.complex128, copy=False)


def discover_arrays(data: Mapping[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
    """Discover N,M automatically; dictionary permits ragged element counts.

    [T,N,M,M] is accepted for equal M; {id: [T,M_i,M_i]} for unequal M_i.
    """
    if isinstance(data, Mapping):
        if not data:
            raise ValueError("No antenna arrays")
        return {str(key): validate_covariances(val) for key, val in data.items()}
    a = np.asarray(data)
    if a.ndim != 4 or a.shape[-1] != a.shape[-2]:
        raise ValueError("Use [T,N,M,M] or a mapping of individual arrays")
    if a.shape[1] < 1:
        raise ValueError("No antenna arrays")
    return {f"array_{j}": validate_covariances(a[:, j]) for j in range(a.shape[1])}


def _unit_coherence(c: np.ndarray) -> np.ndarray:
    diagonal = np.diag(c).real
    denom = np.sqrt(np.maximum(diagonal[:, None] * diagonal[None, :], EPS))
    return c / denom


def _edge_weights(ref: np.ndarray, cur: np.ndarray, conf: DetectorConfig,
                  geom: ArrayGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = ref.shape[0]
    ii, jj = np.triu_indices(m, 1)
    strength = np.sqrt(np.abs(ref[ii, jj] * cur[ii, jj]))
    valid = (np.abs(ref[ii, jj]) >= conf.coherence_min) & (np.abs(cur[ii, jj]) >= conf.coherence_min)
    if geom.positions is not None and len(ii):
        positions = np.asarray(geom.positions, dtype=float)
        d = np.linalg.norm(positions[ii] - positions[jj], axis=1)
        # Only use geometry as a smooth reliability weight: no ULA assumption.
        positive = d[d > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        strength = strength * np.exp(-d / max(scale * 3., EPS))
    return ii[valid], jj[valid], strength[valid], np.arange(len(ii))[valid]


def _connected(m: int, ii: np.ndarray, jj: np.ndarray) -> bool:
    if m < 2 or len(ii) < m - 1:
        return False
    seen = {0}
    changed = True
    while changed:
        changed = False
        for a, b in zip(ii, jj):
            if a in seen and b not in seen:
                seen.add(int(b)); changed = True
            if b in seen and a not in seen:
                seen.add(int(a)); changed = True
    return len(seen) == m


def _phase_fit(ref: np.ndarray, cur: np.ndarray,
               conf: DetectorConfig, geom: ArrayGeometry) -> tuple[np.ndarray, bool]:
    m = ref.shape[0]
    ii, jj, weight, _ = _edge_weights(ref, cur, conf, geom)
    if not _connected(m, ii, jj):
        return np.full(m, np.nan), False
    delta = np.angle(cur[ii, jj] * np.conj(ref[ii, jj]))
    base_weight = np.maximum(weight, 1e-5)
    robust_weight = base_weight.copy()
    phase = np.zeros(m, float)
    for _ in range(4):
        # Weighted graph-Laplacian solve: delta_ij approximately theta_i-theta_j.
        lap = np.zeros((m, m), float)
        rhs = np.zeros(m, float)
        np.add.at(lap, (ii, ii), robust_weight)
        np.add.at(lap, (jj, jj), robust_weight)
        np.add.at(lap, (ii, jj), -robust_weight)
        np.add.at(lap, (jj, ii), -robust_weight)
        np.add.at(rhs, ii, robust_weight * delta)
        np.add.at(rhs, jj, -robust_weight * delta)
        # Fix otherwise unobservable global phase gauge.
        phase = np.linalg.solve(lap + np.ones((m, m)) / m + np.eye(m) * 1e-9, rhs)
        err = np.angle(np.exp(1j * (phase[ii] - phase[jj] - delta)))
        robust_weight = base_weight * np.minimum(1., 0.30 / np.maximum(abs(err), 1e-8))
    phase = np.rad2deg(np.angle(np.exp(1j * (phase - np.median(phase)))))
    return phase, True


def _analyze(c: np.ndarray, power_ref: np.ndarray, coherence_ref: np.ndarray,
             conf: DetectorConfig, geom: ArrayGeometry) -> tuple[np.ndarray, np.ndarray, float, bool]:
    m = c.shape[0]
    gain_db = 10 * np.log10(np.diag(c).real) - power_ref
    # Absolute/common gain is not identifiable without an external reference.
    gain_db -= np.median(gain_db)
    cur = _unit_coherence(c)
    phase_deg, observable = _phase_fit(coherence_ref, cur, conf, geom)
    if observable:
        phi = np.deg2rad(phase_deg)
        corrected_ref = coherence_ref * np.exp(1j * (phi[:, None] - phi[None, :]))
    else:
        corrected_ref = coherence_ref
    pairs = np.triu_indices(m, 1)
    channel_residual = (float(np.median(np.abs(cur[pairs] - corrected_ref[pairs])))
                        if len(pairs[0]) else float("nan"))
    return gain_db, phase_deg, channel_residual, observable


@dataclass
class ArrayBaseline:
    name: str
    elements: int
    geometry: ArrayGeometry
    power_db: np.ndarray
    coherence: np.ndarray
    gain_threshold: np.ndarray
    phase_threshold: np.ndarray
    channel_threshold: float
    calibration_frames: int


@dataclass
class ArrayDecision:
    array_id: str
    elements: int
    candidate: np.ndarray
    alert: np.ndarray
    score: np.ndarray
    gain_db: np.ndarray
    phase_deg: np.ndarray
    channel_residual: float
    channel_shift: bool
    phase_observable: bool
    status: str
    quality: str

    def as_dict(self) -> dict:
        return {
            "array_id": self.array_id, "elements": self.elements,
            "candidates": self.candidate.tolist(), "alerts": self.alert.tolist(),
            "score": self.score.tolist(), "gain_db": self.gain_db.tolist(),
            "phase_deg": [float(v) if np.isfinite(v) else None for v in self.phase_deg],
            "channel_residual": float(self.channel_residual) if np.isfinite(self.channel_residual) else None,
            "channel_shift": self.channel_shift, "phase_observable": self.phase_observable,
            "status": self.status, "quality": self.quality,
        }


class AdaptiveCore:
    """Read-only candidate detector with per-geometry baseline and fail-closed drift gate.

    No trained universal RC3 weights are loaded. This is a baseline math engine,
    not qualified for field deployment. Calling fit() assumes known-healthy data.
    """
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.baselines: dict[str, ArrayBaseline] = {}
        self._hist: dict[str, list[deque[bool]]] = {}

    def register(self, name: str, reference: np.ndarray, calibration: np.ndarray,
                 geometry: ArrayGeometry | None = None):
        name = str(name)
        ref = validate_covariances(reference)
        cal = validate_covariances(calibration)
        if name in self.baselines:
            raise ValueError("Array already registered; do not silently replace a baseline")
        if len(ref) < self.config.min_reference_frames or len(cal) < self.config.min_calibration_frames:
            raise ValueError("Insufficient known-healthy baseline/calibration data")
        if ref.shape[1] != cal.shape[1]:
            raise ValueError("Antenna element count changed between baseline and calibration")
        geom = geometry or ArrayGeometry()
        geom.validate(ref.shape[1])
        db = np.median(10*np.log10(np.real(np.diagonal(ref, axis1=-2, axis2=-1))), axis=0)
        norm = np.stack([_unit_coherence(c) for c in ref])
        coh = np.median(norm.real, axis=0) + 1j*np.median(norm.imag, axis=0)
        coh = (coh + coh.conj().T)/2.
        np.fill_diagonal(coh, 1.0)
        analyzed = [_analyze(c, db, coh, self.config, geom) for c in cal]
        gain = np.abs(np.stack([v[0] for v in analyzed]))
        phase = np.abs(np.stack([v[1] for v in analyzed]))
        drifts = np.asarray([v[2] for v in analyzed])
        gf = max(self.config.gain_floor_db, EPS)
        pf = max(self.config.phase_floor_deg, EPS)
        mult = self.config.calibration_multiplier
        gain_thr = np.maximum(gf, np.quantile(gain, self.config.reference_quantile, axis=0)*mult)
        with np.errstate(all="ignore"):
            # Unobservable phases receive infinite threshold; score falls back to gain only.
            ph_thr = []
            for j in range(ref.shape[1]):
                finite = phase[:, j][np.isfinite(phase[:, j])]
                ph_thr.append(max(pf, float(np.quantile(finite, self.config.reference_quantile)*mult))
                              if len(finite) else float("inf"))
        channel = drifts[np.isfinite(drifts)]
        channel_thr = max(self.config.channel_floor,
                          float(np.quantile(channel, self.config.reference_quantile)*mult)) if len(channel) else float("inf")
        self.baselines[name] = ArrayBaseline(name, ref.shape[1], geom, db, coh,
                                            gain_thr, np.array(ph_thr), channel_thr, len(cal))
        self._hist[name] = [deque(maxlen=self.config.persistence_window) for _ in range(ref.shape[1])]

    def register_many(self, reference: Mapping[str, np.ndarray] | np.ndarray,
                      calibration: Mapping[str, np.ndarray] | np.ndarray,
                      geometry: Mapping[str, ArrayGeometry] | None = None):
        ref, cal = discover_arrays(reference), discover_arrays(calibration)
        if set(ref) != set(cal):
            raise ValueError("Reference and calibration array IDs do not match")
        for name in ref:
            self.register(name, ref[name], cal[name], (geometry or {}).get(name))

    def reset_history(self) -> None:
        """Start a fresh independent evaluation sequence; preserve baselines."""
        for arrays in self._hist.values():
            for q in arrays:
                q.clear()

    def evaluate(self, snapshot: Mapping[str, np.ndarray] | np.ndarray, *, update_history=True) -> dict:
        """One covariance snapshot per array. New arrays require a baseline first.

        A changed element count for an existing ID raises instead of reshaping
        prior weights/baselines. Missing arrays do not change other array states.
        """
        if isinstance(snapshot, Mapping):
            frames = {str(k): np.asarray(v) for k, v in snapshot.items()}
        else:
            x = np.asarray(snapshot)
            if x.ndim != 3 or x.shape[-1] != x.shape[-2]:
                raise ValueError("Snapshot must be [N,M,M] or {id:[M_i,M_i]}")
            frames = {f"array_{i}": x[i] for i in range(x.shape[0])}
        if not frames:
            raise ValueError("No arrays present")
        decisions = {}
        for name, frame in frames.items():
            if name not in self.baselines:
                raise ValueError(f"Unregistered array {name}: fit a known-healthy baseline first")
            c = validate_covariances(np.asarray(frame)[None, ...])[0]
            base = self.baselines[name]
            if c.shape[0] != base.elements:
                raise ValueError(f"Array {name} changed from M={base.elements} to {c.shape[0]}; recalibrate")
            gain, phase, drift, obs = _analyze(c, base.power_db, base.coherence, self.config, base.geometry)
            gscore = np.abs(gain)/base.gain_threshold
            pscore = np.nan_to_num(np.abs(phase)/base.phase_threshold, nan=0., posinf=0.)
            score = np.maximum(gscore, pscore)
            channel_shift = bool(np.isfinite(drift) and drift > base.channel_threshold)
            # Channel mismatch is ambiguous, never silently conclude "healthy".
            candidate = (score >= 1) & (not channel_shift)
            alerts = np.zeros(base.elements, dtype=bool)
            if update_history:
                for j, cand in enumerate(candidate):
                    self._hist[name][j].append(bool(cand))
                    alerts[j] = (len(self._hist[name][j]) >= self.config.persistence_window and
                                 sum(self._hist[name][j]) >= self.config.persistence_hits and
                                 not channel_shift)
            status = ("UNOBSERVABLE_SINGLE_ELEMENT" if base.elements == 1 else
                      "CHANNEL_SHIFT_REVIEW" if channel_shift else
                      "AMBIGUOUS_TWO_ELEMENT_REFERENCE" if base.elements == 2 and candidate.any() else
                      "CANDIDATE" if candidate.any() else "MONITORING")
            if base.elements < 3:
                # Relative diagonal calibration cannot unambiguously isolate one
                # faulty antenna when only 1 or 2 elements are available.
                candidate[:] = False
                alerts[:] = False
            quality = ("relative_calibration_underidentified" if base.elements < 3 else
                       "phase_and_gain" if obs else "gain_only_phase_unobservable")
            decisions[name] = ArrayDecision(name, base.elements, candidate, alerts, score,
                                            gain, phase, drift, channel_shift, obs, status, quality)
        count = len(decisions)
        return {
            "engine": "SANAD_ADAPTIVE_MATH_V0_RESEARCH_ONLY",
            "arrays_observed": count,
            "elements_per_array": {name: v.elements for name, v in decisions.items()},
            "channel_shift_fraction": sum(d.channel_shift for d in decisions.values())/count,
            "decisions": {name: d.as_dict() for name, d in decisions.items()},
            "research_only": True,
            "qualified_frozen_rc3": False,
        }
