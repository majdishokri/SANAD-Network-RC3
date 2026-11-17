# SANAD RC3 Accelerated Stateful CPU V1 — confirmed research freeze

**Status:** Confirmed **simulation** baseline only; not independently field-validated.  
**Frozen on:** 2026-11-17  
**Original RC3 frozen commit:** `182e77b485cceb770ee6570ba332205c73823c91`  
**Implementation:** CPU-only incremental ingestion plus terminal scoring. Underlying RC3 model unchanged.

This folder is a complete recorded snapshot of the accelerated engine and the **actual Colab parity output**. All runtime source, frozen model, and local tests are preserved inside `SANAD_RC3_Accelerated_Stateful_CPU_V1.zip`. The notebook is included separately.

## Confirmed evidence

- Exact parity on all 144 Sionna seeds (3000–3143), maximum absolute probability delta `1.1102230246251565e-16`.
- Reused simulated benchmark: **274/288** faults detected (**95.14% recall**), **0/1008** healthy-only false alarms, **13/720** neighbor false alarms.
- CPU full-window p95 **842.404 → 453.839 ms** (46.13% reduction).
- Ordinary stateful report p95 **1.956 ms**.
- Terminal scoring report p95 **11.632 ms**.
- Cold startup including JIT **761.0 ms**.
- **No first-alarm time-to-detect claim**: RC3 emits terminal classification at report 300.
- RF acquisition and front-end covariance construction were not included in the CPU timings.

## Reproduce and verify

```bash
python accelerated_v1/verify_frozen_release.py
unzip accelerated_v1/SANAD_RC3_Accelerated_Stateful_CPU_V1.zip -d rc3_accelerated_frozen
cd rc3_accelerated_frozen
python tests.py
```

The helper checks SHA-256 digests and validates the published parity report. It does **not** rerun the 144-seed benchmark; that requires the original Sionna evidence files.

## Change-control

This is a reference release, not a development branch. **Do not overwrite, cherry-pick into, or force-push this freeze.** Future iterative, PhaseGraph, or RF research remains separate. Cite the release commit SHA, not the mutable branch tip.

*Scientific scope: research-only simulated parity and detection metrics. Not an independent clinical, industrial, hardware, or field qualification.*
