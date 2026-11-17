#!/usr/bin/env python3
"""Verify byte-level freeze and published RC3 accelerated parity report."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "CONFIRMED_FREEZE.json").read_text())
for filename, record in manifest["artifacts"].items():
    path = HERE / filename
    if not path.is_file():
        raise SystemExit(f"Missing frozen artifact: {filename}")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != record["sha256"] or len(payload) != record["bytes"]:
        raise SystemExit(f"Integrity failure: {filename}")
    print(f"PASS {filename} sha256={actual}")

with zipfile.ZipFile(HERE / "SANAD_RC3_Accelerated_Stateful_CPU_V1.zip") as z:
    model = z.read("frozen/model/rc3_candidate.joblib.gz")
    actual = hashlib.sha256(model).hexdigest()
    assert actual == manifest["model_sha256"]["compressed"], "Frozen model changed"
    lock = json.loads(z.read("frozen/PORTABLE_RELEASE_LOCK.json"))
    assert lock["uncompressed_model_sha256"] == manifest["model_sha256"]["uncompressed"]
    assert lock["frozen_commit"] == manifest["original_frozen_commit"]
    print("PASS frozen RC3 model and portable lock")

summary = json.loads((HERE / "fast_latency_and_parity_summary.json").read_text())
assert summary["status"] == "FULL_PARITY_PASS"
assert summary["completed_seeds"] == 144
assert summary["maximum_prediction_probability_abs_error"] <= 1e-12
assert summary["unchanged_model_sha256"] == manifest["model_sha256"]["uncompressed"]
assert summary["totals"] == {"tp":274,"fault_n":288,"healthy_fp":0,"healthy_n":1008,"neighbor_fp":13,"neighbor_n":720}
assert summary["online_first_alarm_ttd_hours"] is None
print("PASS published parity metrics (report integrity; not a new benchmark run)")
