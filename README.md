# SANAD Network RC3 — Frozen Research Release

**RC3 only.** Frozen trained two-stage RF calibration/anomaly detector: 693 causal features + 15 spatial-meta features, with fixed decision threshold **0.50**. No retraining and no automatic RF hardware actuation.

## Contents

- `rc3.py`: sole public inference entry point. Seven-antenna trace required; reads observations only.
- `model/rc3_candidate.joblib.gz`: deterministic gzip of the original trained joblib. Uncompressed SHA-256 `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4`; compressed SHA-256 `2cd6661607876462bd468e0cfdd720d1b893bc0b7c7f9679c209843b5ffbf12a`. Only load from the trusted release.
- `internal/rc0_runtime`: **internal, frozen compatibility dependencies** required to reproduce RC3's upstream RC0 signal. They are not additional released products or independently maintained APIs.
- `prospective/`: frozen study and quota-safe v3 GPU Sionna runner used for the completed seeds 3000–3143 study; model and threshold were unchanged. Results are in `qualification/`.
- `reports/`: previously consumed development Nested GroupKFold evaluation.
- `FREEZE_MANIFEST.json`, `verify_freeze.py`: SHA-256 integrity and source lineage; `qualification/verify_qualification.py` verifies the prospective metrics and acceptance gates.

## Independent prospective Sionna qualification — PASS (2026-11-14)

With the **pre-committed frozen threshold 0.50** and unchanged model, new Sionna seeds **3000–3143** were evaluated: **274/288 TP (95.14% recall)**; **0/1008 healthy-only FP (0% observed FAR)**; 95% Wilson upper healthy FAR **0.38%**. All three pre-registered acceptance gates passed, including seed-cluster bootstrap FAR upper <2%. Compared with frozen RC0: **253/288 TP (87.85% recall)** and **7/1008 healthy FP**. RC3 had **13/720 neighbour false positives** in fault-injected traces (RC0: 6/720); this is a separate residual risk. Seed-cluster 95% bootstrap recall interval: **92.36%–97.57%**, so point recall >95% does not establish a >95% population lower confidence bound.

This is **independent simulated Sionna validation**, **not real-world RF deployment/field qualification**. Raw evidence and 144 per-seed checkpoints remain on Google Drive; `qualification/` stores the exact frozen summary and pre-registered study manifest and the evidence provenance. See `qualification/QUALIFICATION_REPORT.md`.

Earlier **development-only** nested grouped CV (seeds 2500–2643) was 281/288 TP (97.57%) and 4/1008 healthy FP. Never confuse these two studies.

## Install and infer

```bash
python -m pip install -r requirements.txt
python verify_freeze.py
python qualification/verify_qualification.py
python rc3.py /path/to/s2643.npz --stream injected
```

Use Python 3.11+ with compatible binary wheels. Inference does not require GPU, Sionna, Drive or a training environment. The Google Colab prospective study *does* require GPU, Sionna 2.2.0 and access to its Google Drive source/output locations. The bundled legacy runtime provides an internal detector signal, not a separate version release.

The model, feature engineering, decision threshold and prospective seed range remain frozen. The independent Sionna study is now complete; never retrain or tune RC3 on its labels. Any follow-on detector changes require a new model/release and a separate holdout.
