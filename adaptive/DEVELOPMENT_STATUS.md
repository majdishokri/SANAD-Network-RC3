# SANAD Network Adaptive v0.1 — development branch

This branch is isolated from frozen RC3. The main branch, trained RC3 model, SHA-pinned artifacts, fixed feature cascade and 0.50 threshold must remain unchanged.

## Initial package (not yet extracted into GitHub)

The tested initial source bundle is attached to the development conversation as SANAD_Adaptive_Core_v0.1_release.zip. It contains:

- sanad_adaptive/geometry.py — geometry-aware masked smooth/sparse decomposition and damped solver.
- sanad_adaptive/engine.py — dynamic registry, frozen commissioning baselines, heterogeneous arrays, missing-report masks, circular phase, element scoring and localization.
- sanad_adaptive/covariance.py — stable-reference covariance to gain/phase adapter with coherence masks.
- 13 deterministic unittest cases and executable examples.

The full source must still be extracted from the supplied bundle into this branch before code review or merge. This file is a development marker only.

## Qualification gates

G0 complete locally: 13/13 deterministic tests, including a 101-array synthetic smoke test; NOT RF qualification.
G1 pending: RC3 7x8 regression parity on frozen evidence.
G2 pending: variable-topology Sionna generation and size/topology holdouts.
G3 pending: separate calibrated adaptive classifier and untouched prospective evaluation.
G4 pending: real RF hardware validation.

The original RC3 95.14% point recall and 0/1008 healthy false positives in frozen Sionna are NOT performance claims for Adaptive v0.1. No hardware actuation.

Engineering invariants: do not pad/resize arrays to 7x8; preserve coordinates, antenna IDs and missing phase masks; distinguish common-mode ambiguity; do not leak labels into inference.
