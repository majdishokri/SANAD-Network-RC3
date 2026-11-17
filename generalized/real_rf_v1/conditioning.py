"""Time-honest measured-RF adapter. No RC3 inference, retraining or actuation.

300 x 900 s = 75 HOURS: consecutive sub-second CSI frames are not 300 RC3 reports.
Only time-stamped, phase-coherent acquisition within each physical receiver is accepted.
OFDM tones are frequency samples, NOT independent 15-minute temporal reports.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional
import math
import numpy as np

@dataclass(frozen=True)
class AlignmentPolicy:
    report_seconds: int = 900
    slots_per_report: int = 60
    min_occupied_slots: int = 48
    min_acquisitions: int = 60
    max_gap_seconds: float = 60.
    required_reports: int = 300
    healthy_baseline_reports: int = 120
    minimum_reference_sites: int = 5

    def __post_init__(self):
        if (self.report_seconds != 900 or self.required_reports != 300
            or self.healthy_baseline_reports != 120):
            raise ValueError("Frozen RC3 timing is 300 consecutive reports x 900s; warm/learn=120")
        if not (1 <= self.min_occupied_slots <= self.slots_per_report <= self.report_seconds):
            raise ValueError("Invalid occupied-slot policy")
        if self.min_acquisitions < self.min_occupied_slots or self.max_gap_seconds <= 0:
            raise ValueError("Invalid coverage policy")

@dataclass(frozen=True)
class Observation:
    timestamp_s: float                   # UTC Unix seconds, from hardware timestamp
    site_id: str                         # logical site ID
    physical_station_id: str             # independent receiver/base-station identity
    session_id: str
    csi: Optional[np.ndarray] = None     # complex array [elements, OFDM tones]
    covariance: Optional[np.ndarray] = None
    phase_coherent_within_array: bool = True

def checked_covariance(obs: Observation):
    if (obs.csi is None) == (obs.covariance is None):
        raise ValueError("Supply exactly one of measured CSI or measured covariance")
    if not obs.phase_coherent_within_array:
        raise ValueError("No within-array phase coherence: do not infer phase")
    if obs.csi is not None:
        h = np.asarray(obs.csi)
        if h.ndim != 2 or min(h.shape) < 2 or h.shape[0] < 4:
            raise ValueError("Expected CSI [M,K] with M>=4, K>=2")
        if not np.iscomplexobj(h) or not np.isfinite(h).all():
            raise ValueError("CSI must be finite complex values")
        r = h @ h.conj().T / h.shape[1]
    else:
        r = np.asarray(obs.covariance)
        if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] < 4:
            raise ValueError("Expected covariance [M,M], M>=4")
        if not np.iscomplexobj(r) or not np.isfinite(r).all():
            raise ValueError("Covariance must be finite complex values")
    if not np.allclose(r, r.conj().T, rtol=1e-7, atol=1e-9):
        raise ValueError("Non-Hermitian covariance")
    if np.min(np.linalg.eigvalsh(r)) < -1e-6 * max(1., np.max(np.abs(r))):
        raise ValueError("Non-positive-semidefinite covariance")
    if np.any(np.real(np.diag(r)) <= 0):
        raise ValueError("Zero or negative receive element power")
    return np.asarray((r+r.conj().T)/2, dtype=np.complex128)

@dataclass(frozen=True)
class WindowQuality:
    site_id: str
    physical_station_id: str
    session_id: str
    start_utc_s: int
    accepted: bool
    n_acquisitions: int
    occupied_slots: int
    max_gap_s: float
    reason: str
    covariance: Optional[np.ndarray] = None

class _Bucket:
    def __init__(self, obs, start, r, policy):
        self.start = start
        self.site = obs.site_id
        self.physical = obs.physical_station_id
        self.session = obs.session_id
        self.n = 0
        self.sum = np.zeros_like(r, dtype=np.complex128)
        self.slots = set()
        self.times = []
        self.policy = policy
        self.add(obs, r)

    def add(self, obs, r):
        if (obs.physical_station_id != self.physical or obs.session_id != self.session
            or r.shape != self.sum.shape):
            raise ValueError("Identity, acquisition session or antenna count changed mid-window")
        self.n += 1
        self.sum += r
        self.times.append(float(obs.timestamp_s))
        k = int((obs.timestamp_s-self.start) * self.policy.slots_per_report //
                self.policy.report_seconds)
        self.slots.add(min(self.policy.slots_per_report - 1, k))

    def finish(self):
        p = self.policy
        times = [float(self.start)] + self.times + [float(self.start+p.report_seconds)]
        gap = float(max(np.diff(times)))
        cause = []
        if self.n < p.min_acquisitions: cause.append("insufficient_acquisitions")
        if len(self.slots) < p.min_occupied_slots: cause.append("insufficient_temporal_coverage")
        if gap > p.max_gap_seconds: cause.append("long_acquisition_gap")
        good = not cause
        return WindowQuality(self.site, self.physical, self.session, self.start, good,
                             self.n, len(self.slots), gap,
                             ",".join(cause) if cause else "OK",
                             self.sum/self.n if good else None)

class WindowAligner:
    """Streaming non-interpolating UTC time alignment. No synthetic report expansion."""
    def __init__(self, policy=AlignmentPolicy()):
        self.policy = policy
        self.buckets = {}
        self.last_t = {}

    def push(self, obs: Observation):
        if (not isinstance(obs.site_id, str) or not obs.site_id or
            not isinstance(obs.physical_station_id, str) or not obs.physical_station_id or
            not isinstance(obs.session_id, str) or not obs.session_id):
            raise ValueError("Site, physical station and session identifiers required")
        if not np.isfinite(obs.timestamp_s):
            raise ValueError("Timestamp must be a finite hardware UTC epoch")
        t = float(obs.timestamp_s)
        if t <= self.last_t.get(obs.site_id, -math.inf):
            raise ValueError("Non-monotone or repeated timestamp for site")
        r = checked_covariance(obs)
        start = int(t // self.policy.report_seconds) * self.policy.report_seconds
        old = self.buckets.get(obs.site_id)
        emitted = []
        if old is None:
            self.buckets[obs.site_id] = _Bucket(obs, start, r, self.policy)
        elif start == old.start:
            old.add(obs, r)
        elif start > old.start:
            emitted.append(old.finish())
            # Deliberately fail rather than silently padding hours of missing evidence.
            if start != old.start + self.policy.report_seconds:
                del self.buckets[obs.site_id]
                raise ValueError("One or more entire 15-minute windows missing; cannot create fake reports")
            self.buckets[obs.site_id] = _Bucket(obs, start, r, self.policy)
        else:
            raise ValueError("Out-of-order report timestamp")
        self.last_t[obs.site_id] = t
        return emitted

    def flush(self, watermark_utc_s: float):
        """Finalize only windows whose real 900-second end precedes a trustworthy watermark."""
        if not np.isfinite(watermark_utc_s):
            raise ValueError("Invalid watermark")
        complete = []
        for sid, b in list(self.buckets.items()):
            if watermark_utc_s >= b.start + self.policy.report_seconds:
                complete.append(b.finish())
                del self.buckets[sid]
        return complete

def validate_run(reports: list[WindowQuality], *, policy=AlignmentPolicy()):
    """Enforce 75 h x physical sites, no pseudo-sites or mixed M in legacy model."""
    if not reports: raise ValueError("No reports")
    by_site = {}
    for item in reports:
        if not item.accepted or item.covariance is None:
            raise ValueError("Rejected short or incomplete window cannot be fed to RC3")
        by_site.setdefault(item.site_id, []).append(item)
    if len(by_site) < policy.minimum_reference_sites:
        raise ValueError("Original RC3 needs at least 5 independent reporting sites")
    physical = [v[0].physical_station_id for v in by_site.values()]
    if len(set(physical)) != len(physical):
        raise ValueError("Multiple rows of one physical receiver are NOT independent sites")
    shapes = set()
    starts = set()
    for site, rows in by_site.items():
        rows.sort(key=lambda r:r.start_utc_s)
        if len(rows) != policy.required_reports:
            raise ValueError(f"{site}: require 300 verified reports (75 real elapsed hours)")
        for i, r in enumerate(rows):
            if r.start_utc_s != rows[0].start_utc_s + i*policy.report_seconds:
                raise ValueError("Temporal gap or duplicate window")
            if r.physical_station_id != rows[0].physical_station_id:
                raise ValueError("Physical site identity changed")
            if r.session_id != rows[0].session_id:
                raise ValueError("Mixed acquisition sessions not qualified")
            shapes.add(r.covariance.shape)
        starts.add(rows[0].start_utc_s)
    if len(starts) != 1:
        raise ValueError("Sites did not collect aligned 75-hour UTC windows")
    if len(shapes) != 1:
        raise ValueError("Current generalized RC3 needs uniform per-site M; do not pad mixed arrays")
    m = next(iter(shapes))[0]
    n = len(by_site)
    return {"status":"MEASUREMENT_TIME_AND_GEOMETRY_COMPATIBLE_ONLY",
            "qualification":"NOT_FIELD_QUALIFIED",
            "sites":n, "elements":m, "reports":policy.required_reports,
            "duration_hours":policy.required_reports*policy.report_seconds/3600,
            "legacy_7x8_geometry":(n,m)==(7,8),
            "source_to_rc3_evidence_converter":"NOT_YET_VALIDATED",
            "start_utc_s":next(iter(starts))}

class BaselineNormalizer:
    """Healthy-confirmed per-element power reference; completely frozen during scoring.

    A positive diagonal congruence preserves Hermitian PSD. This conditioning
    changes RC3's feature distribution: treat it as a NEW experimental adapter,
    not a verified parity-preserving change to the frozen model.
    """
    def __init__(self):
        self.scale = {}
        self.healthy_session = {}

    def fit(self, site_id, confirmed_healthy_covariances, *, session_id,
            externally_verified_healthy=False, policy=AlignmentPolicy()):
        if not externally_verified_healthy:
            raise ValueError("External healthy-session confirmation required; no self-labeling")
        rs = [np.asarray(r) for r in confirmed_healthy_covariances]
        if len(rs) < policy.healthy_baseline_reports:
            raise ValueError("Need 120 confirmed-healthy 15-minute baseline reports")
        if site_id in self.scale:
            raise ValueError("Baseline locked; create a new version for recalibration")
        p = np.array([np.real(np.diag(r)) for r in rs])
        if p.ndim != 2 or not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError("Nonphysical calibration power")
        med = np.median(p, axis=0)
        self.scale[site_id] = 1/np.sqrt(med)
        self.healthy_session[site_id] = str(session_id)

    def apply(self, site_id, r):
        if site_id not in self.scale: raise ValueError("Missing verified baseline")
        r = np.asarray(r)
        v = self.scale[site_id]
        if r.shape != (len(v),len(v)) or not np.iscomplexobj(r):
            raise ValueError("Baseline covariance shape/type mismatch")
        out = (v[:,None] * r) * v[None,:]
        if not np.isfinite(out).all(): raise ValueError("Bad normalized result")
        return (out+out.conj().T)/2

    def relative_gain_db(self, site_id, r):
        x = np.real(np.diag(self.apply(site_id,r)))
        if np.any(x <= 0): raise ValueError("Invalid reference-normalized power")
        delta = 10*np.log10(x)
        return delta - np.median(delta)   # common gain unobservable

class ObservationFilter:
    """Optional causal measurement companion; DOES NOT change frozen RC3 alerts."""
    def __init__(self, threshold=0.5, persistence=3, median_window=3):
        if threshold <= 0 or persistence < 2 or median_window < 1:
            raise ValueError("Invalid filter")
        self.threshold = float(threshold)
        self.persistence = int(persistence)
        self.history = deque(maxlen=int(median_window))
        self.hits = None

    def step(self, relative_change_scores):
        scores = np.asarray(relative_change_scores, dtype=float)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("Expected finite per-element change scores")
        if self.hits is None: self.hits = np.zeros(len(scores),dtype=int)
        if len(scores)!=len(self.hits): raise ValueError("Antenna set changed")
        self.history.append(scores.copy())
        smooth = np.median(np.asarray(self.history),axis=0)
        candidate = np.abs(smooth) >= self.threshold
        self.hits = np.where(candidate,np.minimum(self.hits+1,self.persistence),0)
        confirmed = self.hits>=self.persistence
        transient = (np.abs(scores)>=self.threshold)&~confirmed  # review until persistent
        return {"smoothed_change":smooth.copy(),
                "persistent_change":confirmed.copy(),
                "transient_review":transient.copy(),
                "not_hardware_fault_labels":True}
