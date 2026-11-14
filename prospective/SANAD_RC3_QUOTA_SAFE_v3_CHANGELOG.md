# SANAD RC3 — prospective GPU runner v3 (Google Drive quota handling)

## Incident

The corrected v2 Colab runner was interrupted with HTTP 403 `rateLimitExceeded` while looking up `s3048.npz`. The error was a Google Drive requests-per-minute limit, not a training, inference, Sionna, or model-integrity failure.

The already-created prospective folder remains the study destination:
https://drive.google.com/drive/folders/1GWZJPx5k8nL0Uk_GgKpcsIMBdHWYepmS

On inspection of Drive on 2026-11-14, the `evidence_npz` and `seed_checkpoints` subfolders each held **48 files**, for seeds **3000–3047**. `s3048.npz` had not been uploaded at that point.

## What v3 changes

- Inventory each Drive evidence/checkpoint folder once, instead of running a `files.list` per individual seed/checkpoint.
- Throttle Drive API calls using a conservative minimum interval.
- Retry *only* Google HTTP 403 `rateLimitExceeded` / `userRateLimitExceeded` and 429 quota responses using bounded exponential backoff and jitter. Permanent 403 permissions errors still fail immediately.
- Hash-check existing checkpoints and uploads; never silently overwrite or duplicate immutable evidence.
- Reuse the same named prospective study/folders and seed range; recover pre-existing checkpoints.

## Scientific freeze unchanged

- RC3 source ZIP SHA-256 `121ce9aba417fb27b07d7f7f6e52633a8ef29a2c99aefddc12a10428d02ddafb`.
- RC3 model SHA-256 `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4`.
- Decision threshold **0.50**.
- Prospective seeds **3000–3143**, no retraining/tuning on prospective results.
- Exact Python AST of `stats`, `make_features`, `meta_features`, `predict_unlabeled`, `score_one`, `summarize`, and `wilson_upper` verified unchanged versus v2.
- The frozen GitHub `main` and `frozen/rc3-2026-11-14` commit remains unchanged. v3 is a **transport maintenance notebook**, not a new detector release.

## Resume

1. Use a fresh GPU Colab runtime.
2. Run `SANAD_RC3_Prospective_3000_3143_v3_QUOTA_SAFE.ipynb`. Do not execute v2 concurrently.
3. It reuses the 48 completed checkpoints, then continues with seed 3048.
4. If Drive quota still persists after bounded retries, rerun v3 later; no model/threshold changes are needed.

This document does not claim that the 144-seed independent study has completed, or that its acceptance gates passed.
