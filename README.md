# SANAD Adaptive Core — v0.1 (EXPERIMENTAL)

**One codebase for variable numbers of arrays and antenna elements.** This is a new research prototype, **not** the independently Sionna-qualified, fixed 7×8 SANAD Network RC3 classifier. RC3 model weights and its 0.50 decision threshold remain frozen and unchanged.

## What works today

- Accepts `{array_id: covariance[T,M_i,M_i]}` where **each array can have a different number of elements**, or a uniform tensor `[T,N,M,M]`. `N` and `M_i` are discovered from the input.
- Accepts CSI `[T,M,K]` to form per-array covariance without assuming that OFDM subcarriers are independent temporal observations.
- Estimates *relative* per-element gain/phase from a **separately fitted known-healthy baseline**, without imposing ULA/Toeplitz structure. Optional real element positions affect coherence-edge reliability weights; physical geometry is **not inferable reliably from covariance dimensions alone**.
- Measures a **channel-shift residual** after best-fit relative phase correction. When the current channel differs too much from the known-healthy channel, issues `CHANNEL_SHIFT_REVIEW` and abstains from confirming a fault rather than silently predicting healthy.
- Uses per-array empirical thresholds and configurable temporal persistence (`3/5` by default). These thresholds are **not validated field-calibration thresholds**.
- Offers an invariant 12-feature per-element representation and a **trainable, initially UNTRAINED** common classifier head for future cross-hardware supervised learning.
- Fail-closed: a newly introduced array must obtain a known-healthy baseline before scoring; changed element count under the same array ID requires recalibration; 1–2-element arrays are flagged **underidentified** rather than incorrectly claiming localization.

## Mathematical approach

For array `i` with `M_i` elements and a covariance `R_i(t)`:

- Relative gain: `g_j = 10 log10(R_jj(t)) - baseline_j`, with the array median removed (absolute common gain is not identifiable from this relative measurement alone).
- Coherence: `C_jk = R_jk / sqrt(R_jj R_kk)`. Relative phase differences are estimated from the graph of sufficiently coherent element pairs by robust weighted least squares of `arg(C_jk(t) conj(C_jk^base)) ≈ φ_j - φ_k`. A disconnected graph results in **phase unavailable**.
- Channel drift: median magnitude of coherence mismatch after applying the fitted phase correction. High drift triggers review; it is not counted as a confirmed hardware fault.
- Per-element candidate score: `max(|g_j|/calibrated_gain_threshold_j, |φ_j|/calibrated_phase_threshold_j)`. A persistent candidate with acceptable channel quality is a **research alarm**, not a certified diagnosis.

Unlike RC3, the mathematical core is *shape-adaptive*; the separate learned head still needs **training and hold-out qualification on mixed real hardware**. Different propagation conditions may be impossible to distinguish perfectly from calibration changes using covariance alone. A calibrated external reference can improve identifiability.

## Quick start

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q tests
```

```python
from sanad_adaptive import AdaptiveCore
core = AdaptiveCore()
# known_healthy_reference and known_healthy_calibration:
# both dicts {"siteA": [T,4,4], "siteB": [T,8,8], "siteC": [T,16,16]}
core.register_many(known_healthy_reference, known_healthy_calibration)
result = core.evaluate({"siteA": current4x4, "siteB": current8x8})
```

Array names in the two calibration mappings must match. Timing/warm-up and antenna positions must be measured, configured, and validated on each deployment.

## Real ESPARGOS compatibility experiment

Use `SANAD_Adaptive_ESPARGOS_Colab.ipynb` in the accompanying Google Drive folder, or:

```bash
pip install -r requirements-espargos.txt
PYTHONPATH=. python experiments/run_espargos_adaptive.py \
    --download --data-dir ./data --output ./results
```

The pilot downloads the **real** `static_middle` and `walking_nearby` ESPARGOS-0001 CSI TFRecords using the published DaRUS persistent dataset endpoints. It calibrates on *only* the beginning of static_middle; compares held-out static CSI to walking CSI; and injects `D R Dᴴ` faults into separate measured covariance windows (160 challenges). It outputs JSON, CSV and PNG. All results are experimental and **must not be compared numerically with RC3's 95.14% Sionna result**. The real measurements have no independently certified hardware-fault labels. The Colab notebook saves outputs directly to Google Drive.

## Current validation

- **15 automated checks passed** (synthetic input and a synthetic full-pipeline fixture), testing dynamic element counts, mixed N/M, covariance validity, phase/gain injection, channel-shift abstention and a common 12-feature head.
- **ESPARGOS physical recordings have NOT been run in this environment**; the included Colab notebook is the execution path for real data.
- **No universal supervised model trained or accuracy qualified**. A shared head is an engineering interface, not evidence that the fixed RC3 model generalizes.

## Science and safety restrictions

No RF control/actuation. No destructive fault generation. Do not retrain or overwrite the frozen SANAD RC3 weights, `main`, or `frozen/rc3-sionna-qualified-2026-11-14`. Any deployment claim requires a prospective, device-stratified hold-out with real hardware metadata and properly labeled faults, independent calibration periods, and confidence intervals clustered by recording session/site.
