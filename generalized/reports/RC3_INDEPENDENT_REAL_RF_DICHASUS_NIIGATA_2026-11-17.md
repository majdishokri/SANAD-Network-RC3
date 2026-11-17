# SANAD Network RC3 — INDEPENDENT real RF measured data OOD evidence

**Execution date:** 2026-11-17  
**Branch:** \`feature/rc3-independent-measured-rf-ood-v1\`  
**Status:** All original-source acquisitions and announced test workflows succeeded.  
**Frozen RC3:** no edits to trained model weights, detection logic, threshold 0.50 or original RC3 inference code.  
**CRITICAL:** these datasets do NOT provide the five-plus physically independent sites, synchronized genuine 75-hour / 300×15-minute histories or confirmed physical fault interventions needed to calculate full-RC3 field Recall/FAR.  
**Evidence category:** (1) unmodified **real measured healthy/unlabeled CSI shadow-domain validation**; (2) **digitally imposed gain/phase transformations on real measured CSI** as a separate numerical check; (3) **processed real propagation / OOD** validation.

## 1. Source A: University of Stuttgart DICHASUS adxx genuine measured CSI

* Published primary DOI: https://doi.org/10.18419/DARUS-4062
* Source: https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-adxx/
* Spec downloaded from original DaRUS datafile ID **293194**, SHA-256 \`ddc951c7d9f65ae22206eccdce556be1a76877e176853e62a4b620892d4fea50\`.
* Physics: 64 real measured receiver antenna channels, 1024 OFDM subcarriers, 3.44 GHz, **four physically distributed 2×8 planar arrays (16 coherent elements each)**, with **noncontiguous acquisition indices** given by \`spec.json > antennas > assignments\`. All 64 original positions were used exactly once.
* \`csi\` is the original TensorFlow float32 tensor \`[64,1024,2]\` (I/Q channel real/imag components), decoded without materializing an alternate synthetic channel.
* Original TFRecord native CRC32C is checked for **every** processed record. File-source metadata is retrieved directly from publisher API, not Google Drive.
* Original source data stay on temporary GitHub Actions runners and are **not redistributed**.

### 1.1 Complete original ad25 recording — all source bytes, full publisher checksum

[**Executed GitHub Actions run 44038529291 — SUCCESS**](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44038529291)

| Field | Measured/source-verified result |
|---|---|
| Original source | \`dichasus-ad25.tfrecords\`, DaRUS file ID **664466** |
| Original bytes downloaded and processed | **596,950,014** |
| Whole-file published MD5 | \`3e907ad72ae6aab1aa44ef53ba60a3ef\` — **VERIFIED** |
| Whole-file SHA-256 | \`bbd4ff14fc30993bd865cdd37dc53f652c4f4a1beb54e4a5ab0c31d1d083cc74\` |
| Authentic measured RF snapshots | **1,137** |
| Complex RF CSI coefficients scanned | **74,514,432** (1,137×64×1,024) |
| Genuine source time span | **61.578125 seconds**, median adjacent timestamp interval **0.046875 seconds** |
| 16×16 physical-array covariance matrices | **4,548/4,548 source-valid Hermitian, PSD** |
| Nonfinite original measured CSI elements | **0** |
| Worst normalized minimum eigenvalue (H H* / 1,024) | **+0.0002315296** (all positive) |
| Synthetic source modifications | **None** in measured CSI validation |
| Frozen RC3 field predictions / FAR / recall | **0 / NOT ESTIMABLE / NOT ESTIMABLE** |

The covariance uses exactly \`R = H Hᴴ / 1024\` **per measured 16-element planar array**; it does not transform these physical rectangles into fictitious independent ULA sites. The automatic PSD check is a numerical ingestion check, not proof that SANAD detected failures.

**Complete native ad25 per-array descriptive metrics:**

| Actual physical antenna array, as identified by \`spec.json\` | Median absolute offdiagonal correlation | Median within-array max–min port power spread (dB) | Median measured source SNR |
|---|---:|---:|---:|
| Close to roll-up door | **0.64856** | **3.7135** | **13.8201** |
| Close to pillar 16-A | **0.40765** | **4.0318** | **5.2013** |
| Close to pillar 14-A | **0.52670** | **3.9328** | **3.6093** |
| On top of container | **0.35631** | **3.6133** | **7.0226** |

All measurements are real, original RF, but spatial channel differences here are primarily an environment/channel phenomenon and cannot be labeled electronic hardware faults without external truth.

### 1.2 Independent original ad00/ad25 prefix comparison

[GitHub Actions run 44038100844 — SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44038100844)

* Source files \`dichasus-ad00.tfrecords\` (no reflector wall, 64 dB receiver gain) and \`dichasus-ad25.tfrecords\` (reflector wall, 67 dB gain).
* Downloaded exactly **50,331,648 source bytes from each file** using publisher HTTP 206 byte-range requests; **prefix SHA-256** computed. **Full-file MD5 was NOT claimed** for these two partial downloads.
* 95 contiguous original records from each; **190 records**, **12,451,840 complex coefficients**, **760 actual covariance matrices** checked; all CRC32C valid, all original CSI finite and all covariance matrices PSD.
* Genuine native spans: **5.856018 s** and **6.234375 s**. Unused gaps were not invented or backfilled.
* "Close to roll-up door" median inter-element absolute correlation **0.73480** (ad00) vs **0.64965** (ad25) in the early-file sample. **This is not a causal fault-effect estimate**: wall presence, receiver gain, position, date/time and recording content vary.

### 1.3 Digital gain and phase transformation controls on actual measured CSI

In the COMPLETE real ad25 file, a separate *synthetic transformation code test* applied a nominal **−3 dB** amplitude step and **+5°** phase rotation to one antenna's actually measured complex channel within each array and snapshot, and numerically checked the source-preserving ratio.

* **4,548 arithmetic transformation checks**, exactly one per native array–snapshot pair.
* Mean measured gain change **−2.999999989835 dB**, mean relative phase rotation **+5.0°**.
* \`fault_origin: SYNTHETIC_DIGITAL_TRANSFORMATION_NOT_PHYSICAL_HARDWARE\`.
* This is NOT an independent fault-detection experiment or frozen RC3 Recall result; it verifies source decoding and that a controlled complex transformation retains expected mathematical magnitude/phase. It has **no hardware fault labels and no RC3 classifier inference**.
* No trained parameter, original RC3 weight or 0.50 threshold was changed.

## 2. Source B: Niigata University / Yokohama 4.85 GHz real urban RF

**Primary publisher:** https://doi.org/10.5281/zenodo.21213882

[Measured source inventory run 44037381257 — SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44037381257) downloaded the **actual publisher \`4.85GHz Dataset.zip\`**, **37,825,230 original bytes**. Publisher MD5 \`a19578d10f2edab687136bee42274bf4\` was **VERIFIED**; complete ZIP SHA-256 \`51e0eabb4a54564db2e0387a6846aa150230bca452cfe87644796b52ed10fc3d\`.

**Important discovered source-format limitation:** the 35 ZIP members contain real processed multipath-parameter MATLAB data (\`MPC\`, \`PowerdB\`, \`RouteInfo\`, \`SelInd\`) and route-CSV measurements. The published archive **does not expose original source \`[T,8,8]\` complex CSI** directly. This cannot be promoted to frozen RC3 antenna gain/phase traces without synthesis and changed assumptions. Although publisher webpage describes 3,386 valid samples over R3–R6, **this exact ZIP contains original available data for R3, R4, R6 only**; R5 data are **absent**.

[**Original real measured propagation analysis run 44037821836 — SUCCESS**](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44037821836)

| Measured route, real environment | Original valid CSV rows | LoS | NLoS | Median source field-measured path loss (dB) |
|---|---:|---:|---:|---:|
| R3, Yuraku urban macro | **915** | 69 | 846 | **113.16348** |
| R4, NTTCom urban macro | **634** | 182 | 452 | **118.71699** |
| R6, World Porters urban micro | **1,735** | 827 | 908 | **94.20086** |
| **Actual available original total** | **3,284** | 1,078 | 2,206 | Not pooled across distinct routes |

* CSV count and measurement index set of all three routes **match** their corresponding original \`SelInd_*.mat\` MATLAB validity selections (and LoS/NLoS category counts); each source includes 3,000 \`MPC\` slots, many not valid per route selection.
* Path loss median R4−R6 = **24.51614 dB**, describing distinct environments, antenna locations and distance distributions. This is NOT RF receiver hardware gain drift and not model alarms.
* No credible healthy-only false-alert rate can be calculated because original frozen RC3 was not run and no attested healthy 75-hour site stream exists.

## 3. Original RC3 frozen regression

[**Parity run 44038358011 — SUCCESS**](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44038358011) verified:

* \`PASS: bit-for-bit 7x8 feature/meta/prediction equivalence\` with untouched original frozen RC3.
* Variable-geometry code-fixture compatibility for \`4×4\`, \`8×16\`, \`16×32\`, \`32×64\`, \`64×8\` — NOT additional field qualifications.
* **12/12 standalone fail-closed raw-IQ and quality-wrapper code-fixture tests passed**.

## 4. Scientific outcome

| Evidence tier | This benchmark | Permissible claim |
|---|---|---|
| REAL_RF / intact CSI | DICHASUS full ad25 (1,137 snapshots, four authentic 2×8 arrays) | Original real RF complex-CSI importer and array covariance reconstruction validated numerically |
| REAL_RF / partial shadow | DICHASUS ad00/ad25 first 95 frames each | Independently recorded configurations exhibit differing natural array covariance/coherence statistics |
| REAL_CHANNEL_SYNTHETIC_FAULT | −3 dB, +5° on actual DICHASUS CSI | Controlled complex channel transformations preserve nominal mathematical phase/gain (not actual detector) |
| REAL_RF / processed propagation | Niigata R3/R4/R6, 3,284 points | Real urban macro/micro propagation OOD statistics established |
| REAL_HARDWARE_FAULT with independently certified per-port labels | NOT PROVIDED | **NO field hardware-failure Recall** |
| ≥5 physical sites, genuinely synchronized 300×15min, ≥75h | NOT PROVIDED | **NO full field RC3 FAR/Recall/TTD** |

Do not turn four physical DICHASUS 2×8 planar arrays into invented seven 8-element ULA sites, do not turn 48 ms sounder observations into synthetic 15-minute timestamps, do not infer true hardware-fault labels from CSI outliers, and do not imply mathematically verified injected phase/gain shifts are fault *detections*.

**Main conclusion:** On a new publisher/source, SANAD's independent source path correctly validates genuine complex RF CSI, native per-record provenance, actual time and original physical antenna mapping, and it can reject unqualified field claims while preserving the frozen baseline. **A field detection-rate estimate is still unavailable.** The original Sionna 95.14% recall remains a simulation benchmark, separate from these measured-RF shadow validations.
