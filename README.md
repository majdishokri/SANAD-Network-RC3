# SANAD Network RC3 — Frozen Research Release

**RC3 only.** Frozen trained two-stage RF calibration/anomaly detector: 693 causal features + 15 spatial-meta features, with fixed decision threshold **0.50**. No retraining and no automatic RF hardware actuation.

## Contents

- `rc3.py`: sole public inference entry point. Seven-antenna trace required; reads observations only.
- `model/rc3_candidate.joblib.gz`: deterministic gzip of the original trained joblib. Uncompressed SHA-256 `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4`; compressed SHA-256 `2cd6661607876462bd468e0cfdd720d1b893bc0b7c7f9679c209843b5ffbf12a`. Only load from the trusted release.
- `internal/rc0_runtime`: **internal, frozen compatibility dependencies** required to reproduce RC3's upstream RC0 signal. They are not additional released products or independently maintained APIs.
- `prospective/`: corrected v2, fixed-threshold GPU Sionna study on fresh seeds 3000–3143; Drive-authenticated Colab workflow. No completed independent results are claimed here.
- `reports/`: previously consumed development Nested GroupKFold evaluation.
- `FREEZE_MANIFEST.json`, `verify_freeze.py`: SHA-256 integrity and source lineage.

## Performance status

RC3 **development** nested grouped cross-validation, seeds 2500–2643: **281/288 TP (97.57% Recall)**, **4/1008 healthy FP (0.397% FAR)**, 95% Wilson FAR upper 1.016%. These are **not independent prospective results**. The evaluation on unused seeds 3000–3143 must be completed to establish independent performance. Research-only, no field deployment qualification.

## Install and infer

```bash
python -m pip install -r requirements.txt
python verify_freeze.py
python rc3.py /path/to/s2643.npz --stream injected
```

Use Python 3.11+ with compatible binary wheels. Inference does not require GPU, Sionna, Drive or a training environment. The Google Colab prospective study *does* require GPU, Sionna 2.2.0 and access to its Google Drive source/output locations. The bundled legacy runtime provides an internal detector signal, not a separate version release.

Do not modify the model, feature engineering, fixed decision threshold, or the prospective seed range after opening the new-seed outcomes. Any follow-on changes require a different candidate/release.
