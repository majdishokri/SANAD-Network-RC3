# SANAD Network RC3

**A research prototype for calibration-integrity monitoring in multi-antenna RF networks.** SANAD RC3 combines causal per-array calibration evidence with a frozen two-stage spatial classifier. The **recommended reference release** is **RC3 Accelerated Stateful CPU V1** with the unchanged decision threshold **0.50**.

[Accelerated release](accelerated_v1/) · [Simulation qualification](qualification/QUALIFICATION_REPORT.md) · [Reproduce](#verify-the-release) · [Evidence limitations](docs/RESEARCH_BOUNDARIES.md) · [Security](SECURITY.md)

> **Research-only—not field certified.** The fault detection metrics below come from held-out **NVIDIA Sionna simulations**, not actual deployed receiver faults. The model produces a terminal decision after 300 reports; faster CPU computation does not mean earlier fault detection. No RF hardware actuation.

## Verified results

| Measure | Result | Scientific scope |
|---|---:|---|
| Fault recall | **274/288 = 95.14%** | Held-out Sionna seeds 3000–3143 |
| Observed healthy-only false alarms | **0/1,008** | Simulation; not a zero-FAR warranty |
| Faulted-trace healthy-neighbour false alarms | **13/720** | Separate neighbour risk |
| Frozen operating threshold | **0.50** | No retraining or retuning |
| Accelerated full-window CPU p95 | **453.84 ms**, down from **842.40 ms** | Replayed 300-report simulation evidence |
| Ordinary stateful report update p95 | **1.96 ms** | CPU only; excludes acquisition and CSI |
| Terminal report scoring p95 | **11.63 ms** | CPU only |
| Prediction parity | **144/144** seeds; max error **1.11×10⁻¹⁶** | Same model and decisions as frozen RC3 |

The original simulation's seed-cluster 95% recall interval is **92.36%–97.57%**. The point estimate passed the preregistered simulated recall gate; the lower confidence bound does **not** establish true recall over 95%. All 3 original simulation acceptance gates passed. See [qualification evidence](qualification/QUALIFICATION_REPORT.md).

## Architecture and supported input

The original frozen RC3 qualification protocol assumes **7 independent receiver arrays × 8 coherent antenna elements**, with **300 reports collected at 15-minute intervals (75 real hours)**. The model scores preprocessed calibration evidence and generates a **terminal per-array probability**. It does not accept arbitrary raw IQ/CSI files or unsynchronized arrays as equivalent input.

The CPU benchmark excludes physical radio acquisition, covariance extraction, hardware synchronization, network delivery and end-to-end processing. These must be validated independently before any physical deployment.

## Repository structure

| Path | Purpose |
|---|---|
| [accelerated_v1/](accelerated_v1/) | **Preferred release:** complete CPU package ZIP, Colab notebook, parity/timing data and integrity verifier |
| [rc3.py](rc3.py) and [model/](model/) | Original frozen RC3 inference and hash-verified weights |
| [internal/rc0_runtime/](internal/rc0_runtime/) | Required frozen upstream compatibility runtime, not separate supported products |
| [qualification/](qualification/) | Held-out simulated benchmark and preregistered result checks |
| [prospective/](prospective/) | Original Sionna GPU experiment notebooks/scripts |
| [reports/](reports/) | Development-only research evidence |
| [docs/](docs/) | Research limitations, exact freeze policy and public release handling |

Other branches contain **experimental research**, not new certified releases. The external **DICHASUS** stress test used four actual physical planar arrays, artificially regrouped into seven numerical inputs, and only seconds of real CSI. It exhibited high unmodified-data alerting at 0.50; a separate experimental 0.90 threshold helped large digital port loss but not subtle drift. **Neither result qualifies RC3 on real hardware.** [Read the OOD report](https://github.com/majdishokri/SANAD-Network-RC3/blob/feature/rc3-adaptive-real-rf-sensitivity-v1/generalized/reports/RC3_REAL_RF_ADAPTIVE_SENSITIVITY_HOLDOUT_2026-11-17.md).

## Verify the release

Clone the repository and run:

\`\`\`bash
git clone https://github.com/majdishokri/SANAD-Network-RC3.git
cd SANAD-Network-RC3
python -m pip install -r requirements.txt
python verify_freeze.py
python qualification/verify_qualification.py
python accelerated_v1/verify_frozen_release.py
\`\`\`

These commands check unchanged original model/runtime hashes, the frozen simulation results and the accelerated artifact's recorded parity. The accelerator verifier **does not rerun 144 Sionna seeds**. The original raw per-seed simulator evidence is referenced by provenance in the qualification report, not included here.

For local use of the accelerated code, see its [installation and extraction steps](accelerated_v1/README.md). To invoke the original model on a compatible 300-report Sionna evidence trace:

\`\`\`bash
python rc3.py /path/to/compatible_sionna_evidence.npz --stream injected
\`\`\`

Python joblib models must **never** be loaded from untrusted sources: the distributed trained model is hash-verified. Generalizing array sizes or testing new RF data does **not** itself qualify them for the original 7×8 performance claim.

## Development and legal status

[Research boundaries](docs/RESEARCH_BOUNDARIES.md) · [Contribution policy](CONTRIBUTING.md) · [Security reporting](SECURITY.md) · [Public repository guide](docs/PUBLIC_RELEASE_GUIDE.md)

**License:** No explicit software license has been granted in this repository. Public source availability does **not** itself authorize redistribution, modification or commercial use. Dataset licenses are separate; maintainers should select a software license before advertising the repository as open source.

The original 2026-11-14 frozen study and the 2026-11-17 accelerated CPU release remain preserved by exact historical commit/branch. Public README and maintenance documentation may change without modifying the pinned research implementation.
