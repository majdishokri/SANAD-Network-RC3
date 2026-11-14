# SANAD Adaptive v0.2 — EXPERIMENTAL (2026-11-14)

This branch adds a **CSI spectral phase-evidence monitor** to the v0.1 covariance-based, dimension-adaptive core. The spectral module accepts any supported number of arrays, elements and subcarriers, provided the receiver elements are **phase synchronized within each array** and recorded against stable element identities. It does not assume 7 arrays, 8 elements, or 117 tones.

## What was tested

- Automated unit and synthetic end-to-end tests, including 5° persistent phase-only injection, common packet-phase invariance, gain/phase injection, nonmatching configurations, and frequency-selective channel change abstention.
- The real ESPARGOS-0001 data **have not yet been executed through v0.2** in this environment. Colab notebook runs the real files and automatically saves outputs to Google Drive.
- No pretrained universal multi-hardware classifier weights; the optional shared head is still untrained.

## Scientific and validation boundaries

- This is not frozen SANAD Network RC3. That exact detector, fixed threshold 0.50, its Sionna result and GitHub main/frozen branches remain untouched.
- The 5° synthetic unit test is only a software check on simulated coherent CSI, not an empirical success rate on ESPARGOS.
- v0.1 ESPARGOS results already examined in development: 0/208 static array-window alarms, 56/664 walking array-window alarms, and 1/32 phase-only trials. They are **development diagnostics**, not a fresh held-out data source for v0.2 qualification.
- Changes in channel that resemble diagonal antenna phase changes can be mathematically indistinguishable without pilots, an independent calibrated reference or assumptions on channel motion. The spectral gate abstains on large spectral mismatch but does not guarantee zero false alarms or correct calibration localization.
- A gain fault is evaluated in the v0.1 covariance component; a phase-only fault in the v0.2 spectral component. Fused alerts require both covariance and spectral channel-shift gates to allow a decision.
- Injections are `D R Dᴴ` and `D S Dᴴ` applied to measured data. They are NOT observed physical hardware faults.
- No field certification, deployment eligibility or guaranteed detection threshold is claimed. Evaluation recordings are correlated; report descriptive frequencies, not population CIs.
