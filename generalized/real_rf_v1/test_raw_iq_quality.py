"""Code-only fixtures (NOT a real RF benchmark)."""
from __future__ import annotations
import numpy as np
import pytest
from generalized.real_rf_v1.raw_iq_quality import (
    RFInputNotQualified,analyze_receiver_iq,require_qualified_raw_iq
)

def test_constant_two_value_raw_iq_abstains_without_hardware_claim():
    p=np.full((6,2,2240),-16,dtype=np.int16)
    p.reshape(-1)[::2000]=0
    r=analyze_receiver_iq(p)
    assert r.distinct_integer_levels==2
    assert r.modal_integer_level==-16
    assert r.modal_fraction>0.995
    assert r.status=="ABSTAIN_NEAR_CONSTANT_QUANTIZED_RAW_RF"
    assert not r.physical_hardware_fault_verified
    with pytest.raises(RFInputNotQualified,match="no calibration inference"):
        require_qualified_raw_iq(p)

def test_realistic_multilevel_code_fixture_does_not_trip_one_specific_quality_gate():
    p=np.tile(np.arange(-128,128,dtype=np.int16),100).reshape(10,2,1280)
    r=require_qualified_raw_iq(p)
    assert r.distinct_integer_levels==256
    assert r.status=="NOT_FLAGGED_BY_THIS_ONE_GATE"
    assert not r.physical_hardware_fault_verified

def test_reject_float_as_unverified_native_iq():
    with pytest.raises(RFInputNotQualified,match="signed native int16/int32"):
        analyze_receiver_iq(np.zeros((10,3000),dtype=float))

def test_reject_unpaired_iq_stream():
    with pytest.raises(RFInputNotQualified,match="interleaved"):
        analyze_receiver_iq(np.zeros((5,5001),dtype=np.int16))

def test_reject_too_few_actual_samples():
    with pytest.raises(RFInputNotQualified,match="Insufficient"):
        analyze_receiver_iq(np.zeros((5,20),dtype=np.int16))

def test_sha_provenance_and_repeated_quality_result():
    p=np.full((10,4000),-16,dtype=np.int16)
    a=analyze_receiver_iq(p);b=analyze_receiver_iq(p.copy())
    assert a.source_slice_sha256==b.source_slice_sha256
    assert a.modal_fraction==1.0
    assert a.as_json()["decision_kind"]=="INPUT_QUALITY_GATE_NOT_HARDWARE_FAULT_CLASSIFIER"

def test_signed_int32_source_storage_supported_without_dtype_coercion():
    p=np.full((8,4000),-16,dtype=np.int32)
    p.reshape(-1)[::1800]=0
    quality=analyze_receiver_iq(p)
    assert quality.status.startswith("ABSTAIN_")
    assert quality.source_integer_storage_dtype=="int32"
    assert quality.modal_integer_level==-16
