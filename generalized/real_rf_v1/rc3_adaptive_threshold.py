"""Separate SANAD-RF adaptive sensitivity gate; original frozen RC3 untouched.

Input is the exact original FrozenRC3.predict_trace() probability vector.
Calibrate from SOURCE-NATIVE, NO-INJECTION reference observations only; never
use fault labels, test windows, or future model predictions for threshold fit.

RESEARCH LIMITATION: 7 numeric 8-channel DICHASUS groups originate from FOUR
physical 16-element arrays, and 300 real frames span seconds, not 75 h.
A no-injection source window is NOT certified hardware-healthy.

This module changes ONLY the separate downstream alert threshold and does not
retrain the RC3 model, change the pretrained 0.50 baseline, estimator or scores.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json,math
import numpy as np

FROZEN_ORIGINAL_THRESHOLD=0.50
MIN_CALIBRATION_GROUPS=7

class UnqualifiedDomainCalibration(ValueError):
    """Reject missing/invalid calibration, never silently claim healthy RF."""

@dataclass(frozen=True)
class AdaptiveRC3Threshold:
    original_threshold: float
    operational_shadow_threshold: float
    calibration_source_scores_sha256: str
    n_calibration_probabilities: int
    max_calibration_probability: float
    calibration_margin_probability: float
    threshold_grid_step: float
    training_labels_used: bool=False
    original_rc3_model_modified: bool=False
    target_false_alarm_rate_certified: bool=False
    real_hardware_health_certified: bool=False
    original_five_site_seventyfive_hour_protocol_met: bool=False
    output_kind: str="RESEARCH_OOD_SHADOW_ONLY"

    def as_json(self):
        return asdict(self)

    def shadow_alerts(self,probabilities):
        p=_validated_scores(probabilities)
        return (p>=self.operational_shadow_threshold).tolist()

    def legacy_alerts(self,probabilities):
        p=_validated_scores(probabilities)
        return (p>=self.original_threshold).tolist()

def _validated_scores(probs):
    a=np.asarray(probs,dtype=np.float64)
    if a.shape!=(7,) or not np.all(np.isfinite(a)) or np.any(a<0) or np.any(a>1):
        raise UnqualifiedDomainCalibration(
            "Expected precisely 7 finite original frozen-RC3 group probabilities in [0,1]"
        )
    return a

def calibrate_reference_only(
    reference_windows,
    *,
    margin=0.005,
    step=0.01,
    min_threshold=FROZEN_ORIGINAL_THRESHOLD,
    max_threshold=0.99,
):
    """Predefined conservative max-plus-margin policy, not learned-fault tuning.

    Sorted-score maximum acts only as a reference coverage heuristic; it has
    NO distribution-free false-alert-rate guarantee from 7 correlated groups.
    If required threshold exceeds safety policy, fail closed. A numeric test
    case must supply >=1 source-native reference window, never injected truth.
    """
    windows=list(reference_windows)
    if len(windows)<1:
        raise UnqualifiedDomainCalibration("Require at least one real source calibration window")
    scores=np.concatenate([_validated_scores(w) for w in windows])
    if not (0<margin<0.1 and 0<step<=0.05 and
            0<min_threshold<max_threshold<=1):
        raise ValueError("Invalid prespecified research sensitivity policy")
    max_ref=float(np.max(scores))
    candidate=max(float(min_threshold),
                  float(math.ceil((max_ref+margin-1e-12)/step)*step))
    candidate=round(candidate,10)
    if candidate>max_threshold:
        raise UnqualifiedDomainCalibration(
            "ABSTAIN: source-domain reference alerts too high to safely calibrate "
            f"a threshold <= {max_threshold:.2f} without model retraining"
        )
    canonical=np.asarray(scores,dtype="<f8").tobytes()
    digest=hashlib.sha256(canonical).hexdigest()
    return AdaptiveRC3Threshold(
        original_threshold=FROZEN_ORIGINAL_THRESHOLD,
        operational_shadow_threshold=candidate,
        calibration_source_scores_sha256=digest,
        n_calibration_probabilities=int(scores.size),
        max_calibration_probability=max_ref,
        calibration_margin_probability=float(margin),
        threshold_grid_step=float(step),
    )

def evaluate_holdout(original_shadow,*,calibration_window=0):
    """Recalibration is fitted ONLY from the first NO-INJECTION calibration row.

    Heldout metrics are purely descriptive alert rates on ORIGINAL unlabeled
    RF and known DIGITAL interventions. Never represent FP/FAR/TPR on hardware.
    Does not look at holdout labels/probabilities when deriving threshold.
    """
    cleans=original_shadow["clean_windows"]
    trials=original_shadow["injected_trials"]
    if not (isinstance(cleans,list) and len(cleans)>=3 and
            isinstance(trials,list) and len(trials)>=21*3):
        raise UnqualifiedDomainCalibration(
            "Need >=3 separate native 300-frame source windows and digital test trials"
        )
    if calibration_window!=0:
        raise UnqualifiedDomainCalibration("Pre-registered first-window-only calibration")
    if original_shadow.get("original_rc3_detection_threshold")!=.50:
        raise UnqualifiedDomainCalibration("Original 0.50 baseline must be preserved")
    if original_shadow.get("real_physical_arrays")!=4 or original_shadow.get("numerical_subarray_groups")!=7:
        raise UnqualifiedDomainCalibration("Expected four actual physical arrays, seven numeric rows")
    if original_shadow.get("subarrays_are_independent_physical_sites") is not False:
        raise UnqualifiedDomainCalibration("Never assert seven independent physical sites")
    windows={int(x["window"]):x for x in cleans}
    if set(windows)<={0} or set(windows)!={0,1,2}:
        raise UnqualifiedDomainCalibration("Require native nonoverlapping 300-frame windows 0,1,2")
    ordered=sorted(windows)
    for a,b in zip(ordered,ordered[1:]):
        if windows[a]["real_native_end_time_s"]>=windows[b]["real_native_start_time_s"]:
            raise UnqualifiedDomainCalibration("Calibration/test window overlap or reverse time")
    calibration=calibrate_reference_only([windows[0]["no_injection_probs"]])
    heldout_rows=[]
    for w in (1,2):
        clean=windows[w]
        probs=_validated_scores(clean["no_injection_probs"])
        heldout_rows.append({
            "window":w,
            "source_start_time_seconds":float(clean["real_native_start_time_s"]),
            "source_end_time_seconds":float(clean["real_native_end_time_s"]),
            "source_span_seconds":float(clean["real_native_elapsed_seconds"]),
            "unmodified_frozen_original_0p50_alerts":int(sum(calibration.legacy_alerts(probs))),
            "unmodified_source_adapted_alerts":int(sum(calibration.shadow_alerts(probs))),
            "groups_in_unmodified_source":7,
            "actual_hardware_healthy_labels":None,
        })
    cases=[]
    for x in trials:
        w=int(x["window"])
        if w not in (1,2):continue
        group=int(x["digital_target_group_zero_based"])
        if not 0<=group<7:raise ValueError("Digital fault target missing")
        clean=windows[w]
        original_clean=float(clean["no_injection_probs"][group])
        if abs(float(x["paired_clean_target_probability"])-original_clean)>1e-9:
            raise ValueError("Mismatched digital trial and original no-injection source")
        p=float(x["frozen_rc3_target_probability"])
        new_clean=p>=calibration.operational_shadow_threshold
        base_clean=original_clean>=calibration.operational_shadow_threshold
        cases.append({
          "window":w,
          "synthetic_fault_family":x["synthetic_fault_family"],
          "numeric_target_group":group,
          "source_antenna_1based":x["digital_target_port_one_based"],
          "frozen_original_0p50_target_alert":bool(p>=.50),
          "frozen_original_0p50_clean_target_alert":bool(original_clean>=.50),
          "adapted_target_alert":bool(new_clean),
          "adapted_matched_clean_target_alert":bool(base_clean),
          "adapted_new_target_alert":bool(new_clean and not base_clean),
          "paired_probability_delta":p-original_clean,
          "real_hardware_fault":False,
          "digital_target_label_only":True,
        })
    scenarios=sorted(set(x["synthetic_fault_family"] for x in cases))
    expected=len(scenarios)*2*7
    if len(cases)!=expected or set(scenarios)!={"small_ramp","large_ramp","port_loss"}:
        raise UnqualifiedDomainCalibration(
            f"Missing prespecified full factorial heldout digital trials: found {len(cases)}"
        )
    by_scenario={}
    for name in scenarios:
        t=[x for x in cases if x["synthetic_fault_family"]==name]
        by_scenario[name]={
            "digital_test_trials":len(t),
            "frozen_0p50_target_alerts":int(sum(x["frozen_original_0p50_target_alert"] for x in t)),
            "adapted_target_alerts":int(sum(x["adapted_target_alert"] for x in t)),
            "adapted_target_alert_rate_shadow":float(np.mean([x["adapted_target_alert"] for x in t])),
            "adapted_new_alerts_vs_matched_clean":int(sum(x["adapted_new_target_alert"] for x in t)),
            "mean_paired_target_probability_delta":float(np.mean([x["paired_probability_delta"] for x in t])),
            "real_hardware_fault_recall":None
        }
    output={
      "source":"DICHASUS 2024 ad25 real RF with digital fault interventions",
      "independent_physical_array_count":4,
      "numeric_frozen_RC3_7x8_groups_from_four_arrays":7,
      "calibration_window_native_300frames":0,
      "heldout_windows_native_300frames":[1,2],
      "calibration_digital_fault_labels_read":False,
      "calibration_policy":calibration.as_json(),
      "frozen_original_RC3_inference_artifact_modified":False,
      "adaptive_threshold_is_separate_postprocessing":True,
      "healthy_hardware_certification":None,
      "field_protocol_eligible":False,
      "field_false_alarm_rate":None,
      "field_recall":None,
      "field_time_to_detect":None,
      "heldout_no_injection":heldout_rows,
      "heldout_control_total_alerts_frozen":sum(x["unmodified_frozen_original_0p50_alerts"] for x in heldout_rows),
      "heldout_control_total_alerts_adapted":sum(x["unmodified_source_adapted_alerts"] for x in heldout_rows),
      "heldout_control_total_groups":14,
      "heldout_digital_test_count":len(cases),
      "heldout_digital_by_scenario":by_scenario,
      "heldout_digital_trial_rows":cases,
      "qualifier":"ONE_SESSION_TEMPORAL_HOLDOUT_NOT_INDEPENDENT_HARDWARE_VALIDATION",
    }
    return output
