# SANAD Network Adaptive Core v0.1.0 (research prototype)

**Development is intentionally isolated from the frozen SANAD Network RC3 release.**
Do **not** modify `rc3.py`, trained `joblib` artifacts, RC3 thresholds, model hashes,
legacy runtime or its Sionna qualification manifests. This package is an **untrained,
non-probabilistic research baseline**, not RC3, not a compatible drop-in model, and
not a claim of 95.14% recall or 0% field false alarms.

## Implemented

- Any number of registered arrays, with *different* element counts and 1D/2D/3D
  physical coordinates. New arrays may be registered later without resetting prior
  healthy baselines. Each element retains its stable index; array IDs are strings.
- Causal streaming, commissioning-only frozen baselines, missing report / missing
  phase masks, circular differences for phase, time-window robust medians.
- A geometry-aware smooth/sparse decomposition per array with an explicit convex
  objective, damped alternating minimization, stopping tolerance, and solver status.
  The spatial smooth subspace uses an intercept and eligible physical coordinate
  gradients. It is **not** an arbitrary-geometry RF channel model.
- Per-element heuristic localization scores (normalized gain + phase residual),
  persistence, and separate common-mode **review**, rather than declaring an
  ambiguous full-array movement to be a localized fault.
- Optional *reference-covariance* adapter for a verified stable spatial reference;
  relative gain extraction and coherent-component phase synchronization. Missing
  spatial coherence produces NaN phases, not silently fabricated zeros.
- Advisory neighboring-array score contrast, which never changes alert decisions.
- No injected ground-truth labels in inference; no automatic antenna corrections.

## Mathematical model

For each array with physical element locations `P in R^(M x d)`, calibrated
proxy features are converted to normalized, baseline-relative residuals
`Y in R^(M x 2)`. Gain is expressed in units of 0.5 dB and phase in units of
5 degrees by default. The geometry subspace `H(P)` includes an intercept and,
when identifiable, low-order gradients. We solve separately by channel:

```text
min_(b,s)  0.5 ||Mask * (Y - H(P)b - s)||_2^2
            + lambda ||Mask * s||_1 + 0.5 * ridge * ||b||_2^2
```

The nuisance term `H(P)b` represents smooth spatial movement; sparse `s`
is *evidence* for a localized element deviation. This is inspired by the
smooth/sparse separation in historical SANAD v29 (`v29_research.py`) and damped
iterations in the cited fixed-point thesis; **no theorem in the thesis is claimed
to prove convergence of the learned RC3 cascade or real-world RF correctness**.

## Quick start

```bash
cd adaptive
python -m pip install -e .
python -m unittest discover -s tests -v
python -m examples.demo
```

Example:

```python
import numpy as np
from sanad_adaptive import ArraySpec, RFMeasurement, SANADAdaptiveNetwork

network = SANADAdaptiveNetwork([
    ArraySpec('sector-4', np.arange(4)[:, None].astype(float)),
    ArraySpec('sector-16', np.arange(16)[:, None].astype(float), ('sector-4',)),
])
for report in reports:  # one time-aligned report; report maps array IDs to raw proxies
    decisions = network.update({
        key: RFMeasurement(gain_db=reading['gain_db'], phase_deg=reading['phase_deg'])
        for key, reading in report.items()
    })
```

The initial 24 report periods are *assumed genuinely healthy* (minimum 16 valid
samples per channel). They become a frozen commissioning baseline. A missing
commissioning period is not silently relearned from later possibly faulty data.

`ArrayDecision.candidate_element` is **zero-based**; `status` is a research
heuristic, not field-certified alarm; `element_scores` are **not probabilities**.
For arrays with insufficient independent element information, localization is
withheld. `COMMON_MODE_REVIEW` explicitly does **not** identify the cause of a
uniform movement. An external trusted reference is needed to distinguish
shared gain/phase fault from changing RF environment.

## Input contract and limitations

1. The input is already-aligned, calibrated per-element **gain (dB)** and
   **phase (degrees)** proxies. It is **not** raw RSSI, power, or the diagonal of
   an arbitrary covariance. The frozen RC3 v11/v35 preprocessing does not
   directly accept the new package's outputs.
2. `covariance_to_proxy(current, reference)` is valid only when the trusted
   reference covariance is stable enough that `current ~ D reference D^H`.
   Changing traffic/path profiles can break that assumption. The adapter removes
   the unobservable global gain and phase gauges. It is not a full UMa/UMi/RMa
   estimator, and has no field validation.
3. Thresholds here are **synthetic demonstration defaults**. RC3 threshold
   `0.50` and RC3's trained probability scores cannot be transferred.
4. Array neighbors are metadata for context, not a learned graph neural network.
   Topology generalization and real cross-array inference are **future work**.
5. The algorithm does not yet handle automatic RF actuation, online reference
   re-qualification, drift in per-element coordinate geometry, mixed hardware
   frequencies, deployed inference latency, or learned probability calibration.
6. Simulation unit tests are API and algebra sanity checks; no claim of recall,
   false-alarm rate, confidence interval or deployment readiness follows.

## Development acceptance gates

- **G0 (implemented):** Python unit tests for 4/8/16/32 elements, 101 arrays,
  missing masks, commissioning, gain/phase localization, phase wraps,
  permutation equivariance, common mode and reference observability.
- **G1 (pending):** reproduce original `rc3.py` on frozen 7x8 NPZ evidence and
  prove bit-for-bit regression parity of the **unchanged** baseline.
- **G2 (pending):** a new Sionna data generator covering 1x4, 2x8, 7x8, 5x16,
  100+ arrays, irregular coordinates, user load variability, missing data, and
  independent healthy/fault seeds. Separate size and topology holdouts.
- **G3 (pending):** train/calibrate a new adaptive classifier with matched
  fault-level labels and a new, untouched prospective holdout. Track array-level
  recall, **element localization**, healthy FAR and injected-neighbor FAR,
  detection delay, confidence intervals and computational cost by topology.
- **G4 (pending):** UMi/RMa and real RF hardware validation, including
  unobservable multi-element/common-mode faults and reference quality checks.

The frozen RC3 Sionna result (274/288 recalled and 0/1008 healthy-only false
positives on seeds 3000..3143) remains **separate, unchanged, and incomparable**
to the present synthetic unit tests. Do not tune a future model on RC3's
prospective holdout labels.
