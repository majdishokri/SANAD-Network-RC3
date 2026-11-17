"""Offline logic checks only; never count generated fixtures as measured RF."""
import tempfile
from pathlib import Path
import numpy as np
import pytest
import scipy.io
from generalized.tests.lensfd_real_csi_github import (
    expected_cases, KNOWN, examine_native_mat, relative_rms_db,
    per_port_rms, summarize,
)

def _mock_case(family):
    return next(x for x in expected_cases() if x["family"]==family)

def test_exact_publisher_case_inventory():
    cases=expected_cases()
    assert len(cases)==192
    assert sum(x["family"]=="a2c-static" for x in cases)==112
    assert sum(x["family"]=="fd" for x in cases)==80
    assert len(KNOWN)==10
    assert set(KNOWN).issubset({x["path"] for x in cases})

def test_per_port_rms_uses_complex_amplitude():
    a=np.ones((2,80,52),dtype=complex)
    a[:,67,:]=0
    x=per_port_rms(a,1)
    assert len(x)==80 and x[67]==0 and x[0]==1
    assert relative_rms_db(x,67) is None
    assert np.isclose(relative_rms_db(x,0),0)

def test_native_measured_shape_guard_succeeds_on_local_fixture():
    a=np.ones((2,80,52),dtype=np.complex128)
    a[:,67,:]=0
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/"sample.mat"
        scipy.io.savemat(f,{"chan_full":a,"chan_sci":a[:,:,0]})
        row=examine_native_mat(f,_mock_case("a2c-static"))
    assert row["port68_exact_zero"]
    assert not row["port67_exact_zero"]
    assert row["quietest_port_1based"]==68
    assert row["rc3_field_inference"] is False

def test_native_measured_fd_columns_are_not_physical_tx_rx_claims():
    a=np.ones((32,32,52),dtype=np.complex128)
    a[:,19,:]=0
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/"fd.mat"
        scipy.io.savemat(f,{"chan_full":a,"chan_sci":a[:,:,0]})
        row=examine_native_mat(f,_mock_case("fd"))
    assert row["local_axis1_index19_below20db"] is True
    assert row["local_axis1_index19_exact_zero"] is True
    assert row["local_axis0_index19_exact_zero"] is False

def test_reject_real_only_or_wrong_shapes():
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/"invalid.mat"
        scipy.io.savemat(f,{"chan_full":np.ones((2,80,52)), "chan_sci":np.ones((2,80))})
        with pytest.raises(ValueError,match="shape/types"):
            examine_native_mat(f,_mock_case("a2c-static"))
        scipy.io.savemat(f,{"chan_full":np.ones((2,80,53),dtype=complex),
                              "chan_sci":np.ones((2,80),dtype=complex)})
        with pytest.raises(ValueError,match="shape/types"):
            examine_native_mat(f,_mock_case("a2c-static"))

def test_no_fake_labels_or_rc3_field_predictions():
    r=summarize([],0,0)
    assert r["frozen_rc3_real_field_inferences"]==0
    assert r["rc3_hardware_fault_recall"] is None
    assert r["rc3_hardware_fault_far"] is None
    assert r["physical_port_fault_truth_verified"] is False
    assert r["rc3_qualification"].startswith("OUT_OF_SCOPE")
