"""Regression and fail-closed measured-covariance -> frozen RC3 evidence tests.

Synthetic covariances here are CODE FIXTURES ONLY, not new RF validation.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pytest
from generalized.real_rf_v1.conditioning import WindowQuality, validate_run
from generalized.real_rf_v1.rc3_bridge import (
    PhysicalArray, calibration_vectors_from_covariance,
    generate_rc3_evidence, infer_read_only,
)
from generalized.real_rf_v1.rc3_bridge import ENGINE
import glr_compare

def measured_covariance_fixture(t=300,n=7,m=8):
    """Stable PSD covariance test fixtures with time-varying element factors."""
    i,j=np.indices((m,m))
    base=(0.65**np.abs(i-j))*np.exp(1j*0.18*(i-j))
    data=np.empty((t,n,m,m),dtype=np.complex128)
    for a in range(n):
        for k in range(t):
            gains=np.exp(0.015*np.sin(k/20.+a+np.arange(m)*.8))
            phases=0.04*np.sin(k/32.+a/4.+np.arange(m)*.5)
            d=gains*np.exp(1j*phases)
            data[k,a]=(d[:,None]*base)*d[None,:].conj()
    return data

def make_report_fixture(cov):
    n=cov.shape[1];m=cov.shape[2]
    reports=[];metadata={}
    for a in range(n):
        name=f"site-{a:02d}"
        phys=f"hardware-{a:02d}"
        metadata[name]=PhysicalArray(
            site_id=name,
            physical_station_id=phys,
            element_ids=tuple(f"port-{a}-{p}" for p in range(m)),
            source_dataset="DETERMINISTIC_SYNTHETIC_TEST_FIXTURE_NOT_MEASURED",
            acquisition_manifest_sha256=hashlib.sha256(name.encode()).hexdigest(),
            utc_clock_verified=True,
            per_array_phase_coherence_verified=True,
            element_order_verified=True,
            stationary_ula_verified=True,
            half_wavelength_geometry_verified=True,
            verified_healthy_first_120=True,
        )
        for t in range(len(cov)):
            reports.append(WindowQuality(
                name,phys,"session-fixture",t*900,True,60,60,15.,"OK",cov[t,a]))
    return reports,metadata

@pytest.fixture(scope="module")
def covariance():
    return measured_covariance_fixture()

def test_original_calibration_vectors_exact_parity_without_random_corruptions(covariance,monkeypatch):
    """Only remove SYNTHETIC dropout/glitch injection, not model/detector math."""
    monkeypatch.setattr(glr_compare,"P_MISSING",-1.0)
    monkeypatch.setattr(glr_compare,"P_GLITCH",-1.0)
    ref,faults=glr_compare.pipeline(covariance,109,with_faults=False)
    assert not faults
    ours=calibration_vectors_from_covariance(covariance)
    for a in range(7):
        for t in range(300):
            if t<60:
                assert ours[a][t] is None and ref[a][t] is None
            else:
                np.testing.assert_array_equal(ours[a][t],ref[a][t])

def test_75_hour_fixture_converts_to_exact_legacy_schema_and_read_only_inference(covariance):
    reports,meta=make_report_fixture(covariance)
    data,manifest=generate_rc3_evidence(reports,meta,allow_synthetic_test_fixture=True)
    assert data["healthy_feats"].shape==(2,300,7,3)
    assert data["healthy_delta"].shape==(300,7,3)
    assert data["healthy_v11"].shape==(300,7)
    assert data["healthy_v11_named"].shape==(300,7)
    assert len(data)==4 and all("expected" not in k and "onset" not in k for k in data)
    assert manifest["duration_hours"]==75
    assert manifest["source_kind"]=="SYNTHETIC_CODE_FIXTURE_ONLY_NOT_MEASURED"
    assert manifest["legacy_7x8_geometry"] is True
    assert manifest["qualification"] == "NOT_FIELD_QUALIFIED"
    assert manifest["field_recall_far_localization"].startswith("NOT_ESTIMABLE")
    predictions=infer_read_only(data,manifest)
    assert len(predictions["probabilities"])==7
    assert predictions["field_qualified"] is False
    assert predictions["threshold"]==0.50
    assert predictions["not_verified_hardware_fault_labels"] is True

def test_operator_attestation_must_be_complete(covariance):
    reports,meta=make_report_fixture(covariance)
    bad=dict(meta)
    first=next(iter(bad))
    from dataclasses import replace
    bad[first]=replace(meta[first],stationary_ula_verified=False)
    with pytest.raises(ValueError,match="stationary ULA"):
        generate_rc3_evidence(reports,bad)

def test_physical_duplicate_sites_must_fail(covariance):
    reports,meta=make_report_fixture(covariance)
    from dataclasses import replace
    wrong=[replace(x,physical_station_id="hardware-00") if x.site_id=="site-01" else x
           for x in reports]
    with pytest.raises(ValueError,match="NOT independent"):
        validate_run(wrong)

def test_missing_15_minute_report_fails(covariance):
    reports,meta=make_report_fixture(covariance)
    missing=[r for r in reports if not (r.site_id=="site-01" and r.start_utc_s==100*900)]
    with pytest.raises(ValueError,match="300 verified"):
        generate_rc3_evidence(missing,meta)

def test_no_7x8_claim_when_4_site_data_masquerade_as_7(covariance):
    reports,meta=make_report_fixture(covariance)
    four=[r for r in reports if r.site_id in {"site-00","site-01","site-02","site-03"}]
    four_meta={k:v for k,v in meta.items() if k in {"site-00","site-01","site-02","site-03"}}
    with pytest.raises(ValueError,match="at least 5"):
        generate_rc3_evidence(four,four_meta)

def test_synthetic_fixture_not_accepted_as_real_measurement(covariance):
    reports,meta=make_report_fixture(covariance)
    with pytest.raises(ValueError,match="cannot be represented as measured RF"):
        generate_rc3_evidence(reports,meta)
