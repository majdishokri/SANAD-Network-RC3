"""Developer fixtures only, NOT simulated/real-field RF performance evidence."""
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import generalized.real_rf_v1.quality_gated_bridge as guarded
from generalized.real_rf_v1.raw_iq_quality import RFInputNotQualified
from generalized.real_rf_v1.rc3_bridge import PhysicalArray

def make_array():
    sites={}
    data={}
    valid=np.tile(np.arange(-128,128,dtype=np.int16),40).reshape(20,512)
    for site in range(7):
        name=f"s{site}"
        ports=tuple(f"el{p}" for p in range(8))
        sites[name]=PhysicalArray(
            site_id=name, physical_station_id=f"hardware{site}",
            element_ids=ports, source_dataset="FIXTURE_NOT_REAL_MEASUREMENT",
            acquisition_manifest_sha256=hashlib.sha256(name.encode()).hexdigest(),
            utc_clock_verified=True,per_array_phase_coherence_verified=True,
            element_order_verified=True,stationary_ula_verified=True,
            half_wavelength_geometry_verified=True,
            verified_healthy_first_120=True,is_measured_rf_capture=False
        )
        for p in ports:
            data[(name,p)]=valid
    return sites,data

def test_reject_unattested_raw_antenna_session_binding():
    arr,data=make_array()
    with pytest.raises(RFInputNotQualified,match="binding not verified"):
        guarded.generate_rc3_evidence_with_raw_iq_preflight(
            [],arr,data,same_acquisition_binding_verified=False,
            allow_synthetic_test_fixture=True
        )

def test_reject_partial_or_extra_raw_physical_port_coverage():
    arr,data=make_array()
    del data[("s1","el4")]
    with pytest.raises(RFInputNotQualified,match="incorrect per-element"):
        guarded.generate_rc3_evidence_with_raw_iq_preflight(
            [],arr,data,same_acquisition_binding_verified=True,
            allow_synthetic_test_fixture=True
        )
    data[("s1","el4")]=next(iter(data.values()))
    data[("not_real","el4")]=next(iter(data.values()))
    with pytest.raises(RFInputNotQualified,match="incorrect per-element"):
        guarded.generate_rc3_evidence_with_raw_iq_preflight(
            [],arr,data,same_acquisition_binding_verified=True,
            allow_synthetic_test_fixture=True
        )

def test_bad_raw_receiver_abstains_before_any_classifier_inference(monkeypatch):
    arr,data=make_array()
    v=np.full((20,512),-16,dtype=np.int16)
    v.reshape(-1)[::1800]=0
    data[("s3","el6")]=v
    def do_not_call(*args,**kwargs):
        raise AssertionError("Immutable RC3 evidence must not run on bad RF input")
    monkeypatch.setattr(guarded,"generate_rc3_evidence",do_not_call)
    with pytest.raises(RFInputNotQualified,match="s3/el6:ABSTAIN_NEAR_CONSTANT_QUANTIZED_RAW_RF"):
        guarded.generate_rc3_evidence_with_raw_iq_preflight(
            [],arr,data,same_acquisition_binding_verified=True,
            allow_synthetic_test_fixture=True
        )

def test_full_valid_source_coverage_forwards_unchanged_original_evidence(monkeypatch):
    arr,data=make_array()
    sentinel=object()
    def unchanged(reports,arrays,**kwargs):
        assert arrays is arr
        assert kwargs["allow_synthetic_test_fixture"] is True
        return sentinel,{"legacy_field_label":"NOT_HARDWARE_TRUTH"}
    monkeypatch.setattr(guarded,"generate_rc3_evidence",unchanged)
    result,metadata=guarded.generate_rc3_evidence_with_raw_iq_preflight(
        [],arr,data,same_acquisition_binding_verified=True,
        allow_synthetic_test_fixture=True
    )
    assert result is sentinel
    assert len(metadata["raw_iq_preflight_source_sha256"])==56
    assert metadata["legacy_field_label"]=="NOT_HARDWARE_TRUTH"
    assert metadata["raw_iq_preflight"]=="PASS_THIS_SINGLE_RAW_RF_QUALITY_GATE_ONLY"
    assert metadata["raw_iq_preflight_does_not_certify_healthy_hardware"]
    assert metadata["raw_iq_preflight_original_rc3_weights_modified"] is False

def test_declared_site_count_does_not_create_independent_hardware_identity():
    # This wrapper cannot turn two measured hubs into 7 independent sites;
    # original generate_rc3_evidence still checks the physical-site contract.
    arr,data=make_array()
    from generalized.real_rf_v1.rc3_bridge import generate_rc3_evidence
    with pytest.raises(ValueError):
        guarded.generate_rc3_evidence_with_raw_iq_preflight(
            [],arr,data,same_acquisition_binding_verified=True,
            allow_synthetic_test_fixture=True
        )
