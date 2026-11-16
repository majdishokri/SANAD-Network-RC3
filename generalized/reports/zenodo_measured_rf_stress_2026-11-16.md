# SANAD Network RC3 — measurement-derived RF stress-test report

**Date:** 2026-11-16 · **Research-only / NOT field-qualified**

**Execution evidence:** [Successful GitHub Actions run](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/43865370879)
· [RF stress test code](https://github.com/majdishokri/SANAD-Network-RC3/blob/feature/rc3-variable-geometry/generalized/tests/zenodo_mpc_rf_stress.py)

### Measurement source and protocol

- Source: Calist & Kim, Niigata University, **Sub-6 GHz MIMO Channel Sounding Dataset**, Zenodo DOI [10.5281/zenodo.21213882](https://doi.org/10.5281/zenodo.21213882), urban Yokohama 4.85 GHz.
- Download: `4.85GHz Dataset.zip`; verified MD5 `a19578d10f2edab687136bee42274bf4`.
- The `_all_Param.mat` multipath records provide measured path angles and complex weights. Only measurement-derived **input covariance** is reconstructed from receive AoA (MPC column 3) with a half-wavelength ULA spatial-power sum and an identity noise term. This input adapter is not directly measured covariance or a new detector.
- For N stations, N *disjoint 300-valid-snapshot drive-route segments* are used as **pseudo-stations**. These are not simultaneously measured or actual independent base stations. The measured instrument had 8×8 channels; M = 16, 32, 64 are modeled apertures, not physically observed M-element arrays.
- The unchanged original `glr_compare.pipeline` injects **two faulted stations per trial** using 0.3–0.8 dB gain and 3–8° phase shifts, an 80-period ramp, 5% missing data and 3% glitches.
- Exact original RC3 feature extraction (693+15), frozen model weights, temporal/spatial arbitration and threshold **0.50** were retained. No retraining, tuning, or RF actuation.
- Each dimension is evaluated on the same measured propagation profiles for several independent **synthetic fault randomizations**. Repeated draws are not independent channel realizations.

### Results — receiver arrival-angle input (MPC column 3)

| Stations × elements | Injected faults detected | Healthy pseudo-site alerts | Nonfaulty neighbors alerted | Synthetic draws |
|---|---:|---:|---:|---:|
| 4×8 | 0/8 | 0/16 | 0/8 | 4 |
| 7×4 | 4/8 | 0/28 | 0/20 | 4 |
| 7×8 | 16/16 | 0/56 | 0/40 | 8 |
| 8×8 | 8/8 | 0/32 | 0/24 | 4 |
| 7×16 | 8/8 | 0/28 | 0/20 | 4 |
| 7×32 | 8/8 | 0/28 | 0/20 | 4 |
| 7×64 | 4/4 | 0/14 | 0/10 | 2 |

**Interpretation:** Four stations produced **0/8 fault detections**, consistent with RC3's unchanged requirement for at least five reporting stations to form its legacy reference. Seven stations with only four antenna elements produced **4/8** detections. The other tested geometries detected all faults in these *small repeated-fault trials*, with no observed healthy alerts.

These are **stress-test counts, NOT independently qualified recall/FAR estimates**. The dataset is measurement-derived but the covariances and injected faults are synthetic; sampling intervals, site geometry, and channel covariance provenance differ from the frozen Sionna study. Zero observed false positives in these trials is not evidence of zero operational FAR.

The historical **95.14% recall (274/288) and 0/1008 healthy-only false alarms** were obtained from a **different frozen 7×8 Sionna simulation** and must not be represented as the performance of the new geometries.

### Required next qualification

To qualify a geometry: collect genuine time-aligned `(T, N, M, M)` covariances from real M-element arrays at N simultaneous RF sites, or generate a new precommitted independent Sionna campaign for each geometry; use independent fault and healthy traces, prespecified metrics, seed-cluster uncertainty, and no tuning on the holdout. Without changing the **five-station** rule, four-station configurations cannot exercise the original reference stage.

**Original frozen RC3 and its historical results remain unchanged.**
