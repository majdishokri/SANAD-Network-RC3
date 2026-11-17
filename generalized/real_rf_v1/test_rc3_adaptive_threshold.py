"""Unit tests on code fixtures ONLY. They are not RF detection trials."""
import copy
import numpy as np
import pytest
from generalized.real_rf_v1.rc3_adaptive_threshold import (
    UnqualifiedDomainCalibration, calibrate_reference_only,evaluate_holdout
)

CAL=[0.6656196451118326,0.643674890647547,0.8888268849206349,
     0.7580592476100287,0.3916341145833335,0.8123511904761903,
     0.8949466765873015]

def fixtures():
    clean=[{"window":i,"no_injection_probs":CAL if i==0 else [0.2]*7,
            "real_native_start_time_s":float(i*100),"real_native_end_time_s":float(i*100+20),
            "real_native_elapsed_seconds":20.,}
           for i in range(3)]
    trials=[]
    for win in range(3):
        for scenario in ("small_ramp","large_ramp","port_loss"):
            for group in range(7):
                trials.append({
                   "window":win,
                   "synthetic_fault_family":scenario,
                   "digital_target_group_zero_based":group,
                   "digital_target_port_one_based":(group+2)%8+1,
                   "paired_clean_target_probability":float(clean[win]["no_injection_probs"][group]),
                   "frozen_rc3_target_probability":0.92 if scenario=="port_loss" else 0.25
                })
    return {"clean_windows":clean,"injected_trials":trials,
            "original_rc3_detection_threshold":.50,"real_physical_arrays":4,
            "numerical_subarray_groups":7,
            "subarrays_are_independent_physical_sites":False}

def test_calibrates_from_original_unmodified_reference_values_only():
    x=calibrate_reference_only([CAL])
    assert x.operational_shadow_threshold==.90
    assert x.original_threshold==.50
    assert x.n_calibration_probabilities==7
    assert not x.original_rc3_model_modified
    assert not x.target_false_alarm_rate_certified
    assert not x.real_hardware_health_certified

def test_raised_gate_suppresses_controls_preserves_very_large_port_loss():
    x=calibrate_reference_only([CAL])
    assert sum(x.legacy_alerts(CAL))==6
    assert sum(x.shadow_alerts(CAL))==0
    assert sum(x.shadow_alerts([.917]*7))==7
    assert sum(x.shadow_alerts([.68]*7))==0

def test_no_calibration_leak_from_future_clean_or_injected_trials():
    d=fixtures()
    a=evaluate_holdout(d)
    assert a["calibration_policy"]["operational_shadow_threshold"]==.90
    assert a["heldout_digital_test_count"]==42
    assert a["heldout_control_total_alerts_adapted"]==0
    assert a["heldout_digital_by_scenario"]["port_loss"]["adapted_target_alerts"]==14
    assert not a["field_protocol_eligible"] and a["field_recall"] is None
    mutated=copy.deepcopy(d)
    mutated["clean_windows"][2]["no_injection_probs"]=[.97]*7
    for x in mutated["injected_trials"]:
        if x["window"]==2:
            x["paired_clean_target_probability"]=.97
            x["frozen_rc3_target_probability"]=.99
    b=evaluate_holdout(mutated)
    assert a["calibration_policy"]==b["calibration_policy"]
    assert b["heldout_control_total_alerts_adapted"]==7
    assert a["heldout_control_total_alerts_adapted"]==0

def test_reject_missing_time_separation_or_too_few_holdout_cases():
    d=fixtures()
    d["clean_windows"][1]["real_native_start_time_s"]=10.
    with pytest.raises(UnqualifiedDomainCalibration,match="overlap"):
        evaluate_holdout(d)
    d=fixtures();d["injected_trials"].pop()
    with pytest.raises(UnqualifiedDomainCalibration,match="Missing prespecified"):
        evaluate_holdout(d)

def test_fail_closed_if_reference_alarms_extreme():
    with pytest.raises(UnqualifiedDomainCalibration,match="ABSTAIN"):
        calibrate_reference_only([[.999]*7])

def test_bad_source_probs_and_incomplete_reference_disallowed():
    with pytest.raises(UnqualifiedDomainCalibration):
        calibrate_reference_only([])
    with pytest.raises(UnqualifiedDomainCalibration):
        calibrate_reference_only([[.1,.2]])
    with pytest.raises(UnqualifiedDomainCalibration):
        calibrate_reference_only([[float("nan")]*7])
    with pytest.raises(UnqualifiedDomainCalibration):
        calibrate_reference_only([[1.2]*7])

def test_legacy_0p50_is_unchanged_and_adaptation_only_changes_separate_gate():
    c=calibrate_reference_only([CAL])
    assert c.legacy_alerts(CAL)==[True,True,True,True,False,True,True]
    assert c.as_json()["training_labels_used"] is False
    assert c.as_json()["output_kind"]=="RESEARCH_OOD_SHADOW_ONLY"
