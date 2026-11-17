"""Auditable measured covariance -> ORIGINAL RC3 v11/v29/v33/v35 evidence.

NOT a field-qualified detector. Requires timestamps/coverage checked by
WindowAligner + validate_run; 5+ verified physically distinct ULA receivers,
uniform M and verified port sequence. The exact existing RC3 extraction math
is reused, but measured data have different domain statistics from Sionna.

Importantly there are NO probabilistic fault/glitch/missingness injections here.
The "healthy" stream name is ONLY a frozen API namespace, not a claim of health.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import hashlib
import re
import sys
import numpy as np

from .conditioning import AlignmentPolicy, WindowQuality, validate_run

GENERALIZED = Path(__file__).resolve().parents[1]
ENGINE = GENERALIZED / "runtime"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from geometry_config import N_ARRAYS, N_ELEMENTS
from monitor_v8 import calib_vector, calibration_estimate
from v32_replay import evidence_stream
from temporal_v35 import extract_delta
from qa_integrity import validate_trace

_WARM = 60
_EMA = .05
_MIN_COHERENCE = .10
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

@dataclass(frozen=True)
class PhysicalArray:
    """Attestation supplied by acquisition operator, not independently certified by code."""
    site_id: str
    physical_station_id: str
    element_ids: tuple[str, ...]
    source_dataset: str
    acquisition_manifest_sha256: str
    utc_clock_verified: bool
    per_array_phase_coherence_verified: bool
    element_order_verified: bool
    stationary_ula_verified: bool
    half_wavelength_geometry_verified: bool
    verified_healthy_first_120: bool
    is_measured_rf_capture: bool = False

    def validate(self, m):
        if (not self.site_id or not self.physical_station_id or
            not self.source_dataset or not _SHA256.fullmatch(self.acquisition_manifest_sha256)):
            raise ValueError("Missing or invalid physical-source provenance SHA256")
        if len(self.element_ids)!=m or len(set(self.element_ids))!=m or not all(self.element_ids):
            raise ValueError("Unverified or duplicated physical element IDs")
        checks = {
            "UTC clock":self.utc_clock_verified,
            "phase-coherent CSI":self.per_array_phase_coherence_verified,
            "physical antenna order":self.element_order_verified,
            "stationary ULA geometry":self.stationary_ula_verified,
            "half-wavelength physical spacing":self.half_wavelength_geometry_verified,
            "verified-healthy commissioning reports 0..119":self.verified_healthy_first_120,
        }
        failed=[k for k,v in checks.items() if v is not True]
        if failed:raise ValueError("Missing operator-verified prerequisite: " + "; ".join(failed))

def calibration_vectors_from_covariance(covariances):
    """Replicate glr_compare.pipeline's ORIGINAL observed-signal math exactly.

    This is the same trace-normalisation, .05 causal covariance EMA,
    4-iteration Toeplitz estimator and coherence-qualified phase EMA, but NO
    RNG dropouts, no injected hardware faults, no simulated glitches.
    """
    cov = np.asarray(covariances)
    if cov.ndim != 4 or cov.shape[1:]!=(N_ARRAYS,N_ELEMENTS,N_ELEMENTS):
        raise ValueError("Expected [T, N, M, M] observed covariance under configured geometry")
    if cov.shape[0]<120 or not np.iscomplexobj(cov) or not np.isfinite(cov).all():
        raise ValueError("Insufficient time or noncomplex/nonfinite covariances")
    arith = [None]*N_ARRAYS
    pema = [None]*N_ARRAYS
    vectors={a:[] for a in range(N_ARRAYS)}
    for t in range(len(cov)):
        for a in range(N_ARRAYS):
            R = np.asarray(cov[t,a],dtype=np.complex128)
            if (not np.allclose(R,R.conj().T,rtol=1e-7,atol=1e-9)
                or np.linalg.eigvalsh(R).min()< -1e-6*max(1.,np.abs(R).max())
                or np.real(np.trace(R))<=0):
                raise ValueError(f"Nonphysical measured spatial covariance at report {t}, site {a}")
            # Preserve original front-end equations, including the same noise ridge.
            R = R + 1e-3 * np.real(np.trace(R))/N_ELEMENTS * np.eye(N_ELEMENTS)
            Rn = R / (np.real(np.trace(R))/N_ELEMENTS)
            arith[a] = Rn.copy() if arith[a] is None else (1-_EMA)*arith[a]+_EMA*Rn
            _,phase,coh = calib_vector(Rn,iters=4)
            if coh>=_MIN_COHERENCE:
                pema[a] = phase.copy() if pema[a] is None else (1-_EMA)*pema[a]+_EMA*phase
            if t < _WARM:
                vectors[a].append(None)
                continue
            gain,phase,_ = calibration_estimate(arith[a],iters=4)
            if pema[a] is not None:
                phase=pema[a]
            vectors[a].append(np.r_[gain,np.nan_to_num(phase)])
    return vectors

def generate_rc3_evidence(
    reports: list[WindowQuality],
    arrays: Mapping[str,PhysicalArray],
    *,
    policy: AlignmentPolicy=AlignmentPolicy(),
    allow_synthetic_test_fixture: bool = False,
):
    """Turn 300 GOOD real 15-min windows/site into the original *observation-only* schema.

    Returns an in-memory observation-only trace with the legacy "healthy"
    namespace. NO synthetic fault truth, injection, labels or output formatting
    is added. Never silently interpolate reports, reassign rows as towers,
    expand physical arrays, or assume the operator-supplied attestation is fact.
    """
    contract=validate_run(reports,policy=policy)
    if (contract["sites"],contract["elements"]) != (N_ARRAYS,N_ELEMENTS):
        raise ValueError("Configured SANAD_STATIONS/ELEMENTS disagree with real RF geometry")
    by_site={}
    for r in reports:by_site.setdefault(r.site_id,[]).append(r)
    if set(arrays)!=set(by_site):
        raise ValueError("Operator attestation must include precisely the measured sites")
    sites=sorted(by_site)
    m=contract["elements"]
    for name in sites:
        meta=arrays[name]
        meta.validate(m)
        if not meta.is_measured_rf_capture and not allow_synthetic_test_fixture:
            raise ValueError("Synthetic/code-fixture covariance cannot be represented as measured RF")
        if meta.site_id!=name or meta.physical_station_id!=by_site[name][0].physical_station_id:
            raise ValueError("Station identity disagrees with acquisition manifest")
    cov=np.empty((300,len(sites),m,m),dtype=np.complex128)
    for a,name in enumerate(sites):
        rows=sorted(by_site[name],key=lambda r:r.start_utc_s)
        for t,r in enumerate(rows):
            if r.covariance is None:
                raise ValueError("Incomplete measured covariance")
            cov[t,a]=r.covariance
    vectors=calibration_vectors_from_covariance(cov)
    raw,named,features=evidence_stream(vectors,cal_dir=ENGINE)
    delta=extract_delta(vectors)
    prefix="healthy"   # Legacy API namespace ONLY, not ground truth.
    observation={
        f"{prefix}_v11":raw,
        f"{prefix}_v11_named":named,
        f"{prefix}_feats":features,
        f"{prefix}_delta":delta,
    }
    validate_trace(observation,prefix,require_truth=False)
    if not np.isfinite(cov).all():
        raise ValueError("Nonfinite covariance encountered")
    evidence_sha = hashlib.sha256(b"".join(
        np.ascontiguousarray(observation[k]).tobytes() for k in sorted(observation)
    )).hexdigest()
    metadata={
        **contract,
        "source_kind":("MEASURED_RF_OPERATOR_ATTESTED_NOT_INDEPENDENTLY_CERTIFIED"
            if all(arrays[name].is_measured_rf_capture for name in sites)
            else "SYNTHETIC_CODE_FIXTURE_ONLY_NOT_MEASURED"),
        "feature_extractor":"LEGACY_RC3_v11_v29_v33_v35_OBSERVATION_ONLY",
        "original_rc3_weights_modified":False,
        "original_threshold":0.50,
        "measurement_ingest_injected_faults":False,
        "measurement_ingest_simulated_glitches":False,
        "measurement_ingest_random_missingness":False,
        "quality_fail_closed":True,
        "evidence_sha256":evidence_sha,
        "site_order":sites,
        "source_manifest_sha256_by_site":
            {name:arrays[name].acquisition_manifest_sha256 for name in sites},
        "field_recall_far_localization":"NOT_ESTIMABLE_NO_VERIFIED_HARDWARE_FAULT_LABELS",
    }
    return observation,metadata

def infer_read_only(observation, metadata):
    """Evaluate unchanged frozen RC3; outputs observational candidates, not field FP/TP."""
    if metadata.get("feature_extractor")!="LEGACY_RC3_v11_v29_v33_v35_OBSERVATION_ONLY":
        raise ValueError("Unverified evidence contract")
    from generalized.rc3 import FrozenRC3
    out=FrozenRC3().predict_trace(observation,"healthy")
    return {
        "model":"FROZEN_RC3_READ_ONLY",
        "probabilities":out["probabilities"],
        "alerts":out["alerts"],
        "threshold":out["threshold"],
        "element_localization":None,
        "not_verified_hardware_fault_labels":True,
        "field_qualified":False,
        "evidence_sha256":metadata["evidence_sha256"],
    }
