"""Run actual original frozen RC3 on original measured RF, calibrate no-injection
first 300 native source frames only, evaluate later independent contiguous
source frames on the same physical session. Not multi-site 75-hour qualification.
"""
from __future__ import annotations
import csv,json
from pathlib import Path
from generalized.tests.dichasus_frozen_rc3_measured_shadow import run
from generalized.real_rf_v1.rc3_adaptive_threshold import evaluate_holdout

OUT=Path("out/adaptive_rc3_heldout")
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=run(OUT/"original_model_predictions",blocks=3,
            scenarios=("small_ramp","large_ramp","port_loss"))
    results=evaluate_holdout(raw)
    (OUT/"rc3_adapted_sensitivity_later_window_metrics.json").write_text(
        json.dumps(results,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    trials=results["heldout_digital_trial_rows"]
    with (OUT/"rc3_adapted_heldout_trials.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(trials[0]))
        w.writeheader();w.writerows(trials)
    if results["field_protocol_eligible"] or results["field_recall"] is not None:
        raise RuntimeError("Cannot silently misclassify DICHASUS shadow as field qualified")
    print("SANAD_ADAPTIVE_HOLDOUT_ACTUAL_REAL_CSI="+json.dumps(
        {k:v for k,v in results.items() if k!="heldout_digital_trial_rows"},
        sort_keys=True,allow_nan=False),flush=True)
    return results

if __name__=="__main__":
    main()
