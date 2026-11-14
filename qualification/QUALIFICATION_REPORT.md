# SANAD Network RC3 — Frozen Independent Prospective Qualification

**Status: PASSED the three pre-registered simulation-study acceptance conditions** on **2026-11-14**. This is *research-only* validation in NVIDIA Sionna (3GPP TR 38.901 UMa); not a claim about real deployed RF hardware, certified antenna calibration, or fault-free behavior.

## Protocol and freeze

- Study: `SANAD_RC3_SPATIAL_META_FROZEN_0p50_GPU_PROSPECTIVE_3000_3143`.
- Untouched evaluation seeds: `3000..3143` inclusive (**144**); development seeds `2500..2643`.
- Frozen detector model SHA-256: `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4`.
- Frozen source ZIP SHA-256: `121ce9aba417fb27b07d7f7f6e52633a8ef29a2c99aefddc12a10428d02ddafb`.
- Decision threshold was locked at **0.50** before new-seed labels; no prospective retraining or threshold search.
- 7 healthy-only arrays per seed: 1,008 healthy-only tests; 2 fault-injected arrays per seed: 288 fault tests; 5 non-faulty neighbours in injected trace per seed: 720 neighbour tests.
- `prospective/…v3_QUOTA_SAFE` retries rate-limited Google Drive requests; no scientific code changes to the frozen v2 protocol.

## Independent simulation outcomes

| Metric | RC0 baseline | Frozen RC3 |
|---|---:|---:|
| True-positive fault detections | 253 / 288 | **274 / 288** |
| Point recall | 87.85% | **95.14%** |
| False positives in healthy-only traces | 7 / 1,008 | **0 / 1,008** |
| Observed healthy-only FAR | 0.694% | **0.000%** |
| 95% Wilson upper on healthy-only FAR | 1.426% | **0.380%** |
| Neighbour false positives in fault-injected traces | 6 / 720 | **13 / 720** |

Paired recall improvement: **21 additional faults**, **+7.29 percentage points**. Seed-cluster bootstrap 95% interval for the paired recall gain: **+4.51 to +10.42 percentage points**.

Pre-registered criteria: (1) point recall **strictly greater than 95%**; (2) 95% Wilson upper bound for healthy-only FAR **strictly below 2%**; (3) seed-cluster bootstrap upper healthy-only FAR **strictly below 2%**. **All three passed.**

The seed-cluster bootstrap 95% recall interval is **92.36%–97.57%**. Consequently, the study passes the **specified point-estimate criterion**; it does **not** prove population recall above 95% at a 95% lower confidence bound. A zero *observed* healthy-only false-alarm rate likewise is not proof of zero future false alarms. Neighbour false positives (13 vs 6 for RC0) are a distinct trade-off, not counted in the specified healthy-only FAR gate.

## Source artifacts and custody

- Exact frozen results: `qualification/SUMMARY_FROZEN.json`; SHA-256 `cda17210aeb4450b1d0a67eded5c2959fc5e4ed371bb170cf44791f3201cdabd`.
- Pre-registered study declaration: `qualification/MANIFEST_FROZEN.json`; SHA-256 `c22c6a62151292d120e58df1806bf289fa9750a7594cf83137d7754431b75140`.
- Original output folder: https://drive.google.com/drive/folders/1GWZJPx5k8nL0Uk_GgKpcsIMBdHWYepmS
- Raw per-seed evidence: https://drive.google.com/drive/folders/1yoKqnablDwjOLadUz4eAcY9QDPtemW8x
- Immutable per-seed checkpoints: https://drive.google.com/drive/folders/1N9XK2W_Vh-Cr3o6HoO1T_1aP3ewf7MmV
- Folder listings checked on 2026-11-14: **144/144 expected evidence filenames** and **144/144 expected checkpoint filenames**, each `s3000` through `s3143`.

The raw NPZ evidence and all per-seed checkpoint JSON files reside in Drive rather than being duplicated in the repository. The repo freezes the **exact reported results and provenance**, not every underlying multi-megabyte simulation artifact. Verify SHA-256 using `python verify_freeze.py` and study assertions using `python qualification/verify_qualification.py`.

**No model, trained weights, RC3 feature code, or threshold was changed after the prospective outcome.** Future detector improvement requires a distinct, independently evaluated candidate.
