# Release status / immutability rules

2026-11-14. `SANAD Adaptive Core v0.1` is **EXPERIMENTAL**. This is a *new* implementation, not a configurable form of RC3's frozen two-stage 693+15-feature model. The generic 12-column feature interface is untrained. RC3 remains unchanged at model SHA-256 `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4` and decision threshold `0.50`.

Validations performed here: synthetic multi-shape unit tests and end-to-end synthetic covariance fixture. No field CSI measurement outcome claimed. A covariance-only detector cannot in general distinguish calibration drift from arbitrary channel changes without additional assumptions or a known reference. The abstention status is intentional.

This branch is for experimental research. Train supervised heads only using independent, labeled and appropriately split multi-hardware datasets; never refit the frozen RC3 weights to the Sionna 3000–3143 holdout.
