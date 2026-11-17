# Rice LensFD — raw-IQ two-level stream anomaly and SANAD fail-closed response

**Executed:** 2026-11-17 on public publisher-original measured RF with source SHA verification.
**Authoritative native source:** Rice RENEW LensFD, Chen et al., *LensFD: Using Lenses for Improved Sub-6 GHz Massive MIMO Full-Duplex*, IEEE TVT (2023), DOI 10.1109/TVT.2023.3240558.
**GitHub working branch:** \`feature/lensfd-failure-origin-audit\`.
**Frozen SANAD RC3:** source, weights and threshold 0.50 **unchanged**. No valid LensFD field inference; field Recall/FAR **not estimable**.
**Scientific status:** confirms near-constant quantized recorded I/Q in certain array ports; this is an **input-quality failure** and a **receive-path investigation target**, not certified broken physical RF hardware.

## Executed publisher raw source test

Run [44032734241 — original measured I/Q port-provenance benchmark SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44032734241) fetched two distinct Rice raw HDF5 recordings and scanned all genuine original 401 source frames.

| Capture and port | Native HDF5 SHA-256 | Scalar I/Q values | Distinct levels | Modal level and fraction | Full-stream raw RMS (single I/Q scalar units) |
|---|---|---:|---:|---|---:|
| **No lens, indoor spot1, port 68** (zero-based 67) | \`a793135d1e653ff9bc301e79febbaebf9fc3eb794bf10d561730a5967a73dd43\` | 1,796,480 | **2: -16, 0** | **-16, 99.72535%** | 15.9780 |
| **Small lens, indoor spot1, port 67** (zero-based 66) | \`c00b7691ae6055aca4ef5342d1558800d49009629652fe4a45f5bfef6422dddf\` | 1,796,480 | **2: -16, 0** | **-16, 99.73092%** | 15.97846 |
| No lens, port 67 (adjacent stronger channel) | same no-lens HDF5 | 1,796,480 | **321** | 0, 1.62740% | 509.8624 |
| Small lens, port 68 (adjacent stronger channel) | same small-lens HDF5 | 1,796,480 | **412** | 0, 1.77241% | 514.6096 |

Actual raw I/Q source storage uses signed **big-endian int16 \`>i2\`**, preserved by the new quality detector; these are **not** coerced to generic float observations before source SHA verification.

No-lens port 68 and small-lens port 67 I/Q arrays have **99.9944336% identical scalar values**, correlation **0.9897867**, and raw integer-scalar RMS difference **0.11937**. Their full SHA-256 digests differ, so they are **not identical files** or exact byte-for-byte duplicated port streams. A full 80×80 original-port SHA cross-check found **no exactly identical raw port streams** between the two captures.

**Interpretation:** a 401-frame RF receiver stream restricted to \`-16\` and \`0\` is not a usable continuously varying complex RF sample sequence. Treat its CSI as untrustworthy for per-element gain/phase drift inference, even if the publisher's processed MAT reports exactly zero. The source may be a muted/inactive ADC path, failed radio, unpopulated channel, acquisition placeholder or receiver mapping error. **The source records alone cannot distinguish these physical/software causes.** In particular, \`-16\` cannot be claimed as a documented hardware failure code without hardware/firmware evidence.

## Pilot detection confirmation on true measured source

Run [44032366490 — original Rice publisher CSI source pipeline, SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44032366490) reprocessed **300 actual original RF frames**, separately from the 401-frame raw sample investigation.

| Capture | Suspect port | Published MAT CSI RMS | Recomputed unmodified Rice CSI RMS | Raw matched-filter timing clamped to sample 0 |
|---|---:|---:|---:|---:|
| No-lens indoor spot1 | 68 | exactly 0 | **6.3567828669e-7** | **599/600** |
| Small-lens indoor spot1 | 67 | exactly 0 | **6.3567828669e-7** | **599/600** |

These are source-frame/client combinations, not independent hardware failures. The matched-filter peak median for both suspect ports was **0.05252376** (about 6.2–6.4 dB below the 80-port median); strong adjacent ports have higher peaks. The original release's code cannot reproduce the exact published \`.mat\` (overall complex RMSE \`0.4454\` for no-lens, \`0.5306\` for small-lens using default first 300 source frames). This may reflect different processing/replay settings, a software mask, or acquisition state; no single cause is yet confirmed.

## Practical SANAD engine response — implemented and measured

Module: [\`generalized/real_rf_v1/raw_iq_quality.py\`](../real_rf_v1/raw_iq_quality.py).
Regression: [\`generalized/real_rf_v1/test_raw_iq_quality.py\`](../real_rf_v1/test_raw_iq_quality.py).
Live measured RF acceptance test: [run 44033267151 — SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44033267151).

The deterministic preflight validates original signed int16/int32 I/Q storage and recording length. It emits \`ABSTAIN_NEAR_CONSTANT_QUANTIZED_RAW_RF\` if the capture has >=10,000 real scalar I/Q values, <=2 distinct integer sample levels and >=99.5% occupancy by one level.

| Measured original source | Raw-IQ gate outcome |
|---|---|
| No lens, port 68 | **ABSTAIN_NEAR_CONSTANT_QUANTIZED_RAW_RF** |
| Small lens, port 67 | **ABSTAIN_NEAR_CONSTANT_QUANTIZED_RAW_RF** |
| No lens, port 67 | NOT_FLAGGED_BY_THIS_ONE_GATE |
| Small lens, port 68 | NOT_FLAGGED_BY_THIS_ONE_GATE |

The live-source GitHub job also passed **7/7 standalone quality-code regression tests**, while original packet/source SHA remained unchanged and raw Rice datasets were never uploaded to the repository. Passing the single gate **does not prove a healthy radio or a valid CSI history**, and the fixed trigger has not undergone an external false-rejection study.

## Root-cause confidence tiers

- **Confirmed:** original Rice HDF5 carries nearly static \`{-16, 0}\` receiver I/Q values on identified suspect port *indices*; pilot correlations and averaged CSI are unusable there; same failure pattern switches between neighboring port indices under two recorded lens configurations; a trustworthy calibration detector must abstain on this evidence.
- **Strong technical suspicion:** acquisition/device channel muted, inactive, incorrectly routed or impaired. Same-chain pair and the dual-polarized radio concept suggest physical radio 34 only *under an unverified consecutive-port wiring assumption*. The anomalous CSI in the 192 MAT-file scan was associated with Chain 5.
- **Not established:** purpose of \`-16\` quantization code; hardware failure vs intentionally disabled or unconfigured receiver; firmware or cable identity; when the impairment began; original MAT generation settings.
- **Not a valid claim:** measured LensFD RC3 sensitivity/false-alarm rate, 95% operational recall, or 32 independent hardware failures from 32 repeated chain-5 recordings.

## What would settle the physical cause

Require Rice/Skylark hardware records and frame configuration for the relevant source captures:
1. Chain 5's original eight radio serial numbers, antenna-port permutations and A/B polarization mapping, plus whether ports 67/68 were swapped.
2. The meaning of constant \`-16\`/0 signed I/Q from the Argos V3 SDR/FPGA/driver; whether it is an unmapped or disabled stream, receive-chain failure, ADC floor or software placeholder.
3. Original HDF5-to-MAT processing revision, frame range/offset, source calibration state and the reason published MAT contains exact zeros while the public code derives tiny nonzero CSI.
4. Verified failed-radio/port history (Fig. 7), repairs and acquisition timestamps.

Until then the scientifically correct operational label is:

\`RECORDED_RF_CHANNEL_INVALID_OR_MUTED; HARDWARE_CAUSE_UNVERIFIED; CALIBRATION_INFERENCE_ABSTAINED\`.

Raw source copyright: Rice University permits commercial/research use with attribution but disallows redistribution without written permission.
