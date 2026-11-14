# SANAD Network — RC3 Frozen Standalone Engine

**Exact frozen inference, extracted into a self-contained software package.** No Sionna runtime, notebooks, training scripts, GPU, cloud account, network access, RF device control, or new calibration hardware are needed for **inference on compatible precomputed evidence**.

This is a *repackaging* of the frozen RC3 code and learned artifacts, **not a newly trained version**. The original implementation and models have identical SHA-256 digests to commit `182e77b485cceb770ee6570ba332205c73823c91` of `majdishokri/SANAD-Network-RC3`.

## Quick start

```bash
python -m pip install -r requirements.txt
python verify_engine.py
python run_engine.py tests/fixtures/s2643.npz --stream injected
python run_engine.py tests/fixtures/s2643.npz --stream healthy
python -m pip install -r requirements-test.txt
python -m pytest -q tests/
```

Programmatic API:

```python
from sanad_rc3_engine import SANADRC3Engine
engine = SANADRC3Engine()  # loads the bundled frozen, trusted models
result = engine.predict_npz("tests/fixtures/s2643.npz", "injected")
print(result)  # {'alerts': [bool]*7, 'probabilities': [float]*7, 'threshold': 0.5}
```

Or import `FrozenRC3` from unmodified `rc3.py` and call `predict_trace(trace, stream)` directly.

## Input contract — essential limitation

The input is **not raw 5G CSI or arbitrary hardware telemetry**. RC3 expects exactly **7 arrays × 300 reports**, with **8 elements per array in the original preprocessing contract**, for each evidence stream `healthy` or `injected`:

| Name under prefix | Shape | Meaning |
|---|---:|---|
| `<stream>_v11` | `[300,7]` | Frozen v11 candidate flags |
| `<stream>_v11_named` | `[300,7]` | Candidate element IDs (0–8) |
| `<stream>_feats` | `[2,300,7,3]` | Fast/strict score, dominance, element ID |
| `<stream>_delta` | `[300,7,3]` | Temporal score, dominance, element ID |

`*_expected` and `*_onset` may be included in development evidence but are **not read during inference**. `healthy/injected` names select the input namespace; they are not labels used by the classifier.

The upstream conversion **from antenna observations to those features** must obey the original frozen v11/v29/v33/v35 preprocessing contract; the executable engine does not recreate CSI or estimate unknown signal geometry on its own. It does **not** automatically adapt to other hardware counts, unlike the unrelated experimental Adaptive branch.

## What's inside

- `rc3.py`: **original byte-for-byte frozen public inference implementation**.
- `model/rc3_candidate.joblib.gz`: frozen RC3 ExtraTrees/HistGradientBoosting/ExtraTrees model bundle.
- `internal/rc0_runtime/`: **only 10 frozen Python dependency files**, the pinned v35 config, and the pinned internal RC0 learned-gate model. These are internal compatibility dependencies, not separately qualified products.
- `sanad_rc3_engine.py`: thin wrapper, no decision changes.
- `run_engine.py`: command-line wrapper.
- `MATHEMATICS.md` / `MATHEMATICS_AR.md`: equations and code provenance.
- `SOURCE_SHA256.json` + `verify_engine.py`: SHA integrity check of all original source/model files.
- `tests/`: exact development fixture + parity, no-label-leakage, shape and integrity tests.

**The engine is read-only.** It does not send, tune, or reconfigure radio hardware.

## Mathematical pipeline, briefly

1. **Original RF covariance preprocessing (upstream, not in this standalone engine):** array covariance and relative gain/phase proxies; historical ULA structural fitting assumes Toeplitz covariance. See `MATHEMATICS.md` for precise source boundaries.
2. **Smooth + sparse decomposition:** standardized 8-element gain/phase vectors separated into spatially smooth drift and sparse localized disturbance with soft-thresholded, damped updates.
3. **Temporal evidence:** robust baseline/MAD, stable element identity, 8/12 and 10/15 fast/strict persistence, 9/14 temporal persistence, group ambiguity handling.
4. **RC0 learned confirmation:** frozen ensemble of logistic and random forest models, threshold selected in its upstream bundle (0.18); RC0 is an auxiliary meta-feature for RC3.
5. **RC3 learned classification:** 693 endpoint causal features → ExtraTrees + HistGradientBoosting probabilities averaged 50/50 → 15 spatial cohort features including RC0 flags → ExtraTrees meta-classifier → **locked decision threshold 0.50**.

**Important:** The serialized RC3 bundle stores a development `decision_threshold` value different from 0.50. Frozen `rc3.py` explicitly overrides it with **0.50**, as pre-registered for the independent validation. Never replace the locked threshold with the serialized metadata.

## Verification and research status

- Local extraction tests: see `VALIDATION_LOCAL.json` (8 automated checks).
- Local development evidence `s2643.npz`: injected stream gives alerts on array indices **0 and 5**; healthy stream gives none. These are **smoke tests, not new independent validation**.
- Frozen RC3 independent Sionna test, seeds **3000–3143**: **274/288** fault TP (95.14% observed recall), **0/1008** healthy-only FP, and **13/720** neighbour FP in injected traces. Three registered simulation gates passed. The 95% seed-bootstrap recall interval was 92.36%–97.57%.
- **Not field-qualified** and not shown to detect actual malfunctioning radio hardware. An untouched frozen Sionna model cannot be declared universally deployable on arbitrary antennas.

Security: `joblib`/pickle model files can run executable Python during loading. Never replace either pinned model with an untrusted file. Run `verify_engine.py` first.

The source repository's frozen `main` and `frozen/rc3-sionna-qualified-2026-11-14` branches have not been edited by this extraction.
