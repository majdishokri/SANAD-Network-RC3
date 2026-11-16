# SANAD RC3 — Direct measured DICHASUS CSI benchmark

**Date:** 2026-11-16  
**Execution:** [Successful GitHub Actions run](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/43888520888)  
**Source code:** [generalized/tests/dichasus_csi_benchmark.py](https://github.com/majdishokri/SANAD-Network-RC3/blob/feature/rc3-variable-geometry/generalized/tests/dichasus_csi_benchmark.py)  
**Status:** exploratory stress test on genuinely *measured* CSI with *synthetically injected* hardware faults. **NOT field-qualified**.

## Source and provenance

DICHASUS University of Stuttgart; [DaRUS DOI 10.18419/DARUS-4062, V3](https://doi.org/10.18419/DARUS-4062), 64-antenna measured, phase-coherent RF channels, 3.440 GHz carrier, 50.056 MHz and 1024 OFDM subcarriers.

- Recording `dichasus-ad25.tfrecords`; DaRUS data-file ID `664466`, complete-file MD5 `3e907ad72ae6aab1aa44ef53ba60a3ef`; first 180,000,000-byte range SHA-256 `e1f6cfd47af0d73c4d494fd73d676d7517401a8493d5ce5c2aac83abaa1e230a`.
- Recording `dichasus-ad21.tfrecords`; DaRUS data-file ID `664462`, complete-file MD5 `c89c49a4a01e8b344b3e33920ee5b04f`; first 180,000,000-byte range SHA-256 `1c35fe576de3b604d45fb65a91a6942d6918cbdd60242cea01f65847fbf1fb60`.
- **Partial-file SHA-256** checksums identify exactly the byte ranges processed, **not** the full source files. Full-file MD5 values are source metadata and were not independently recomputed.
- Native TFRecord CSI float32 `[64,1024,2]` is decoded directly. Spatial complex covariance for each physical ULA row is `H @ H.conj().T / 1024` using measured channel coefficients; **not** a synthetic channel reconstruction.
- Official antenna channel mapping preserves **four physical distributed sites**, each with **two 8-element horizontal ULA rows**. For 4×8, only the first row of each of four distinct sites is used. For 7×8 and 8×8, physically measured 8-element rows are used, but the extra rows share sites: they are **not seven/eight independent base stations**.
- Each trace uses the first 300 measured CSI samples, ~14.4 seconds at the source's 48 ms reporting interval. The original RC3 300 reports span 75 hours (15 minutes/report). Fault ramp duration therefore differs greatly from training, and **results cannot be interpreted as 10-hour operational alarm rates**.

## Protocol

1. RC3 original weights, frozen threshold 0.50, source features, and mathematical decisions remain unchanged.
2. Original `glr_compare.pipeline` synthesizes two subtle gain/phase faults per trial (0.3–0.8 dB, 3–8°, 80-report ramp), ~5% missing records and ~3% glitches. Seeds `510000` and `510001` reuse each recording's same physical CSI, varying only fault draws.
3. Both control and injected traces pass the original feature pipeline. No real hardware-fault labels are present.
4. Paired conventional controls **GLR-1**, **EWMA**, **CUSUM** receive the *same deterministic RC3 front-end calibration vectors and identical injected faults, glitches and missing data*. Their thresholds were pre-fixed, **not tuned** for matched false-alarm rates. They use calibration-vector observations 60–119 only; all use 8-of-12 alarm persistence.
5. Every metric is a number of *array-case alarms anywhere in reports 260–299*, not a time-aligned measurement of operational alarm rates. A control case is **unlabeled for genuine hardware health**; its alarm is a control alert and **not a verified false positive**.

## Results

| Measured recording | Layout | Detector | Injected faults detected | Unlabeled control alerts | Noninjected neighbor alerts |
|---|---|---|---:|---:|---:|
| ad25 | 4 sites × 8 | RC3 | 0/4 | 0/8 | 0/4 |
| ad25 | 4 sites × 8 | GLR-1 | 4/4 | 8/8 | 4/4 |
| ad25 | 4 sites × 8 | EWMA | 4/4 | 8/8 | 4/4 |
| ad25 | 4 sites × 8 | CUSUM | 4/4 | 8/8 | 4/4 |
| ad25 | 7 rows × 8 | RC3 | 3/4 | 12/14 | 9/10 |
| ad25 | 7 rows × 8 | GLR-1 | 4/4 | 14/14 | 10/10 |
| ad25 | 7 rows × 8 | EWMA | 4/4 | 14/14 | 10/10 |
| ad25 | 7 rows × 8 | CUSUM | 4/4 | 14/14 | 10/10 |
| ad25 | 8 rows × 8 | RC3 | 4/4 | 14/16 | 11/12 |
| ad25 | 8 rows × 8 | GLR-1 | 4/4 | 16/16 | 12/12 |
| ad25 | 8 rows × 8 | EWMA | 4/4 | 16/16 | 12/12 |
| ad25 | 8 rows × 8 | CUSUM | 4/4 | 16/16 | 12/12 |
| ad21 | 4 sites × 8 | RC3 | 0/4 | 0/8 | 0/4 |
| ad21 | 4 sites × 8 | GLR-1 | 4/4 | 8/8 | 4/4 |
| ad21 | 4 sites × 8 | EWMA | 4/4 | 8/8 | 4/4 |
| ad21 | 4 sites × 8 | CUSUM | 4/4 | 8/8 | 4/4 |
| ad21 | 7 rows × 8 | RC3 | 4/4 | 14/14 | 10/10 |
| ad21 | 7 rows × 8 | GLR-1 | 4/4 | 14/14 | 10/10 |
| ad21 | 7 rows × 8 | EWMA | 4/4 | 14/14 | 10/10 |
| ad21 | 7 rows × 8 | CUSUM | 4/4 | 14/14 | 10/10 |

## Interpretation

- **4 physical sites:** RC3 did not detect any injected faults. This is consistent with the unchanged five-site reference minimum, so 4×8 is not currently deployable on the frozen logic for this task.
- **7 and 8 physical ULA rows:** RC3 detected 7/8 injected faults across two different recordings for 7×8, but also flagged 26/28 unlabeled control cases. 8×8 on one recording detected 4/4 with 14/16 control alerts. These figures **are not validated recall and false-alarm rates**.
- **Classical GLR, EWMA and CUSUM:** Pre-specified thresholds yielded complete or near-complete alarm saturation on the measured streams. Their 4/4 injected detections are not useful discrimination in the presence of 100% control alerting. RC3 has not demonstrated real-RF superiority over these baselines.
- Every method sees time-compressed, highly dynamic moving-transmitter channels and physical array signatures that were absent or differently modeled in the frozen Sionna qualification.
- The historic RC3 result **95.14% recall, zero of 1008 healthy-only alerts** applies **only to the frozen 7×8 Sionna simulation**, not DICHASUS or new geometries.

## Product evaluation implications

**Do not deploy or market a 95% measured-hardware detection claim from these tests.** Next benchmark should use measured, timestamp-preserving, **multi-hour/multi-day** covariance sequences from stationary hardware with verified healthy-reference segments and independently recorded fault-induction onset and ground truth. Separate environment/channel shift from hardware calibration changes.

The [Royal Military Academy four-channel USRP N310 calibration measurements](https://zenodo.org/records/23084004) can supplement *per-element calibration diagnosis*, but its 4-element hardware is not a seven-site RC3 detection benchmark. Rice RENEW 64-antenna repeat-position captures could provide a second receiver-hardware distribution if available under its access terms.

**Version policy:** The original root RC3, model hashes, frozen rules and decision threshold remain untouched. Everything here is an evaluation extension on `feature/rc3-variable-geometry`.
