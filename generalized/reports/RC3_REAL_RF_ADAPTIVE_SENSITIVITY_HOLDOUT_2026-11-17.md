# SANAD RC3 → SANAD-RF experimental sensitivity adaptation on genuine DICHASUS CSI

**Date:** 2026-11-17. **Repository branch:** \`feature/rc3-adaptive-real-rf-sensitivity-v1\`.

**Decision:** An externally applied 0.90 research sensitivity threshold (selected using the earliest unmodified source window) decreases the **unmodified-source** alert rate from **12/14 to 4/14** across two chronological holdout windows and preserves **14/14 digitally imposed −20 dB port-attenuation endpoint alerts**, but fails to discriminate subtle gain/phase drift. This is a genuinely executed model-data stress test and a useful *port-loss-only adaptation*, NOT evidence of field hardware-fault detection performance.

## Frozen original model versus separate sensitivity gate

* SANAD RC3 original 7×8 inference, frozen model weights and original cutoff **0.50 remain unmodified**. Original feature/meta/prediction bit-for-bit parity reverified in the new branch.
* Separate module [\`rc3_adaptive_threshold.py\`](../real_rf_v1/rc3_adaptive_threshold.py) provides \`AdaptiveRC3Threshold\`. It uses only the **no-injection source probabilities of window 0** to calculate a conservative ceiling: \`ceil((max_source_probability + 0.005) / 0.01) × 0.01\` with original 0.50 floor, 0.99 maximum, and abstention on impossible reference calibration.
* The original no-injection window 0 maximum model probability was **0.8949466765873015** (seven numeric groups). Thus the initial experimental threshold is exactly **0.90**.
* **Scientific disclosure:** The sensitivity scheme was developed *after examining the preceding exploratory ad25 original and digitally faulted results*. This is a development/test-design step, not a prospectively preregistered blind experiment. Later ad25 windows were not used in the **implemented threshold calculation**; independent ad00 is a distinct recording, but not an independently maintained hardware testbed.
* There is no trained parameter update, no digital fault-label feature leakage to RC3, no source time resampling, and **no use of actual test-window model scores when computing the 0.90 cutoff**.

## Dataset and unavoidable qualification limits

Publisher original University of Stuttgart DICHASUS adxx, DOI [10.18419/DARUS-4062](https://doi.org/10.18419/DARUS-4062); \`dichasus-ad25.tfrecords\` original measured CSI, 3.44 GHz, real 64-channel ×1024-subcarrier complex arrays.

- DICHASUS has **FOUR independent physical 2×8 planar arrays, 16 elements per site**, with noncontiguous source antenna index mappings. The trained frozen RC3 architecture assumes **SEVEN independent 8-element ULA receivers**. For this experiment seven disjoint 8-element rows were presented to the unchanged frozen 7×8 model: **seven numeric groups are NOT seven physical sites**.
- Source frames are separated by approximately 48 ms, with each 300-frame window spanning only **15–18 real seconds**, rather than 300 genuine 15-minute reports (75 hours). Native timestamps retained. This is not RC3's training temporal dynamics.
- Clean experimental control means **untouched original source CSI**, not independently verified failure-free hardware.
- Digital injection represents software perturbations to real complex measured covariance, **not actual RF hardware failure**.
- The original ad25 file 596,950,014 bytes passed publisher MD5 \`3e907ad72ae6aab1aa44ef53ba60a3ef\` and full SHA-256 \`bbd4ff14fc30993bd865cdd37dc53f652c4f4a1beb54e4a5ab0c31d1d083cc74\`, with native TFRecord CRC32C checks. No copyrighted raw RF redistributed.
- Real-field hardware-FAR, real hardware-fault recall, verified physical antenna localization and true time-to-detect are **NOT ESTIMABLE**.

## A. Chronologically heldout ad25: 3 source-native windows × 300 real CSI snapshots

[**Completed full source-fed GitHub Actions 44048189256 — SUCCESS**](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44048189256).

Window 0 first 300 original measured frames: threshold reference only. Window 1: next 300 source frames. Window 2: next 300 source frames. No overlap and strictly increasing authentic timestamps. The initial calibration window was never used as a heldout outcome.

| Actual original ad25 later window | Frozen original 0.50 no-injection group alerts | Separate adapted 0.90 no-injection group alerts |
|---|---:|---:|
| Window 1 | 5 / 7 | **2 / 7** |
| Window 2 | 7 / 7 | **2 / 7** |
| **Both windows** | **12 / 14** | **4 / 14** |

The adapted external cutoff reduced the unmodified-source alert count by \`(12−4)/12 = 66.7%\`. These are **source alert proportions, not certified hardware false-alarm rates**. Four unmodified-source alerts still remain across 14 highly correlated numerical endpoint decisions.

### Prespecified digital fault transformations tested on heldout measured CSI

Original real complex covariance transformed in exactly one numeric 8-channel group per scenario. An injection at native frame 160; 80-frame ramp (when applicable). Digital faults alter measured channel responses but not underlying model weights.

| Prespecified digital fault type | Original 0.50 targeted alerts | Adapted 0.90 targeted alerts | NEW adapted alerts vs matched unmodified source | Mean original probability delta |
|---|---:|---:|---:|---:|
| **−0.5 dB gain, +5° phase, gradual ramp** | 11/14 | **3/14** | **0/14** | **−0.06531** |
| **−2.0 dB gain, +12° phase, gradual ramp** | 13/14 | **4/14** | **1/14** | **−0.00507** |
| **−20 dB gain, instantaneous single-port loss** | 14/14 | **14/14** | **10/14** | **+0.18610** |

This demonstrates that the adapted threshold is useful for coarse port loss but **is not a successful calibration-drift detector**. On the small ramp, none of the already unmodified-silent target groups generated a new alarm. Moderate gain/phase ramp created only one new adapted alarm across 14 trials.

**Confidence limitations:** the fourteen source group decisions are from four real physical arrays and two short adjacent source intervals, not independent receiver stations or hardware-fault episodes. A binomial confidence interval pretending 14 independent RF failures would be invalid.

The original benchmark 0.50 had an alarming 12/14 unmodified-source alerts on later windows. These are not necessarily false positives; cross-site and time-scale semantics in this 7 numeric grouping experiment are invalid for RC3's trained domain. Simply increasing threshold cannot fix geometry or spatiotemporal data distribution.

## B. Independent recording/configuration test: ad00

[**Original source-fed independent ad00 experiment 44048649201 — SUCCESS**](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44048649201).

Different publisher original file: \`dichasus-ad00.tfrecords\`, a recording with **no reflector wall**, **64 dB receiver gain** (vs ad25 wall, 67 dB), original 64×1024 complex RF. The benchmark downloaded the first 160 MiB of original source, verified the first 48 MiB SHA-256 \`fbfb6041804d5b2af41b03cd8dee82d8b0e93595f2db93c9aecad1b5d2a0a549\`, and CRC32C of every consumed original TFRecord. This does **not** claim whole-file MD5 of the 6.6 GB ad00 file.

The independently recorded ad00 dataset was NOT used to select the 0.90 threshold.

| ad00 outcome, original source first 300 real frames | Frozen 0.50 | Adapted 0.90 (selected from ad25) |
|---|---:|---:|
| Unmodified source group alerts | 6 / 7 | **2 / 7** |
| −0.5 dB +5° digital ramp target alerts | 6 / 7 | **3 / 7** (1 newly triggered) |
| −2 dB +12° digital ramp target alerts | 6 / 7 | **3 / 7** (2 newly triggered) |
| −20 dB digital port loss target alerts | 7 / 7 | **7 / 7** (5 newly triggered) |

Source ad00 total native time span **16.6080 seconds**; this is an **independent recording configuration**, not independent physically faulted hardware. The adapted threshold still makes 2/7 pre-existing source alerts and underdetects subtle ramps.

## Tests and assets

- [7/7 source-independent adaptive gate unit tests and unchanged original RC3 bit-for-bit parity, Actions 44048189256](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44048189256).
- Source-native adapter: [\`generalized/real_rf_v1/rc3_adaptive_threshold.py\`](../real_rf_v1/rc3_adaptive_threshold.py).
- Full frozen model measured CSI replays: [\`generalized/tests/dichasus_frozen_rc3_measured_shadow.py\`](../tests/dichasus_frozen_rc3_measured_shadow.py).
- First-window fit, later-window verification: [\`generalized/tests/rc3_adaptive_threshold_runner.py\`](../tests/rc3_adaptive_threshold_runner.py).
- Independent recording replay, ad00: [\`generalized/tests/dichasus_frozen_rc3_independent_ad00.py\`](../tests/dichasus_frozen_rc3_independent_ad00.py).
- Downloadable derived numerical JSON/CSV GitHub artifact: \`sanad-rc3-real-rf-adaptive-3window-heldout-results\`, Actions run **44048189256**, artifact ID **14312009463**. No source CSI redistribution.
- Companion Google Colab notebook: \`SANAD_RC3_Adaptive_Sensitivity_Heldout.ipynb\` runs the same exact source SHA-verified pipeline and reports both thresholds and heldout tables.

## Development recommendation — no unearned field accuracy claims

**Working engineering improvement:** the separate **0.90** gate reduced spurious-looking unmodified source alarms on both ad25 heldout and independent ad00, while preserving all −20 dB digitally imposed severe-port-loss endpoint alerts. It can be an experimental **gross port-loss safeguard**, not the final gain/phase calibration monitor.

**Not solved:** subtle 0.3–0.8 dB / 3–8° antenna calibration drift, correct native four-array 2×8 planar spatial reference, real 15-minute report semantics, genuine healthy commissioning labels, and independent physically annotated failed hardware. With no physical fault labels, no hardware FPR or Recall can be reported.

For a defensible product, build a separate **native 4×16 phase-coherent planar-array SANAD-RF calibration engine** with per-array reference and genuine native-time baselines, train and tune only on development recordings, then use independent hardware configurations and physically intervened faults as blind tests. Preserve frozen RC3 as the simulation reference. **Do not deploy 0.90 as a field-qualified universal threshold**.
