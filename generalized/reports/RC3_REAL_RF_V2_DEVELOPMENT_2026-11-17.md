# SANAD Network RC3 — Real-RF Development V2: strict ingestion bridge

**Date:** 2026-11-17
**Branch:** \`feature/rc3-real-rf-bridge-v2\`
**Frozen baseline:** 7 sites x 8 elements, RC3 threshold = 0.50, model original SHA-256 \`0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4\`.
**Status:** DEVELOPMENT / NOT FIELD QUALIFIED. No production model change.

## Implemented and code-tested

- Original RC3 measured-covariance feature-construction path: \`real_rf_v1/rc3_bridge.py\`, using **the unchanged** \`monitor_v8.calib_vector\`, \`monitor_v8.calibration_estimate\`, \`v32_replay.evidence_stream\`, \`temporal_v35.extract_delta\` and original \`FrozenRC3\` scoring. No fault injection, random missing reports, RNG noise/glitches or synthetic timestamps in the new bridge.
- Time validity: \`WindowAligner\` aggregates covariance from actual time-stamped complex CSI within 900-second UTC bins. Default 60 occupied slots, require >=48 occupied, >=60 acquisitions and <=60s max gap. A short recording cannot be multiplied or resampled into a 75-hour campaign.
- Whole-run acceptance requires 300 consecutive actual 15-minute windows = 75 hours; 120 confirmed-healthy commissioning reports; >=5 physically distinct simultaneous stations; common UTC start; unchanged per-site M and physical element mapping. 4-site legacy v29 reference cannot run with the original five-site rule.
- Read-only \`generate_rc3_evidence\` returns only \`healthy_v11\`, \`healthy_v11_named\`, \`healthy_feats\`, \`healthy_delta\`. The \`healthy\` prefix is a **legacy API name**, not a hardware-health label. No \`*_expected\` or \`*_onset\` are fabricated.
- Operator attestations require independently known physical port identity, phase coherence, station ULA geometry, half-wavelength element spacing, source checksum, verified UTC clocks and healthy baseline. These booleans are **claims supplied by the operator**, not physical proofs made by the code.
- Synthetic code fixtures are explicitly prohibited by default in the measured bridge. The unit tests must pass \`allow_synthetic_test_fixture=True\` and receive a clearly labeled \`SYNTHETIC_CODE_FIXTURE_ONLY_NOT_MEASURED\` provenance.
- New \`OneClassThreshold\` and \`ShadowHead\` remain separate exploratory models; no RC3 final-layer retraining, no auto-modification of 0.50, no qualified field accuracy.

## Existing measured datasets — independent evidence inventory

| Source | Actual measured data | What prior tests established | Why NOT RC3 field qualification |
|---|---|---|---|
| DICHASUS ad25/ad21 (DaRUS 10.18419/DARUS-4062) | Native measured 64-port CSI; four real distributed sites; ~48ms CSI reporting | RC3 with 7 selected ULA rows detected 7/8 digitally injected faults, but 26/28 unlabeled controls also alerted. **These are separate rows of only four physical sites.** | actual first-300-file-record spans confirmed by new source timestamp audit: ad21 14.3984375s, ad25 17.8515625s, not 75h; no independently certified physical fault labels; 7 selected rows ≠ 7 sites |
| Niigata/Yokohama Zenodo 10.5281/zenodo.21213882 | Measured multipath angles/complex weights on drive routes | Legacy RC3 detected 16/16 digital faults on modeled 7x8 covariance, 0/56 model controls, 8 synthetic draws | Antenna covariance is modeled from measured MPCs; pseudo-stations are disjoint routes, not simultaneous physical base stations |
| Rice LensFD (Chen et al. 2023) | 392 source files: 192 processed complex-CSI MAT + 200 raw IQ HDF5 | Measured 192 source CSI descriptors analyzed; native configuration-shift AUC 0.556 / 0.630; suspected zero/suppressed channels explored | 0/392 valid frozen RC3 300-report field inferences; missing verified per-port fault truth/port map and timing |
| Rice RENEW cluster2/3 | Measured CSI from same platform, separate acquisition clusters | **PhaseGraph**, not RC3, scored 0/8 digitally injected faults and 0/56 healthy control alerts | Different experimental algorithm; frames are not 15-min reports; no verified hardware failures |
| ESPARGOS-0001 | Published four-array real indoor CSI (~16-21s for selected captures) | Separate adaptive prototypes explored measured-domain covariance and digital perturbations | Four arrays, insufficient for unmodified five-site RC3 rule; no 75-hour station history |

**DICHASUS source:** https://doi.org/10.18419/DARUS-4062
**Prior directly measured DICHASUS benchmark:** [Report](./dichasus_direct_measured_csi_benchmark_2026-11-16.md).
**Niigata/Zendodo report:** [Report](./zenodo_measured_rf_stress_2026-11-16.md).
**Rice LensFD and PhaseGraph experimental reports:** Stored on the user's connected Google Drive, not in frozen GitHub release.

## Experiment protocol for the first genuine field holdout

1. **Acquisition prerequisite**: at least five *physically distinct and stationary* base stations (ideally seven for strict 7x8 comparability), each with independently verified 8-element ULA, known port/cabling map and time-synchronized hardware clock. At least 75 consecutive hours of actual CSI/covariance capture, with no fabricated reports. Better duration and varied days are recommended for robust external validation.
2. **Physical reference**: verify normal commissioning in reports 0..119 before collecting/reading held-out fault labels. Preserve separate session/device/site splits for development, threshold calibration, and untouched assessment.
3. **Source integrity**: raw waveform/CSI hashes; sample timestamps; 15-minute bucket coverage; dropout gap; physical RX element mapping; channel coherence; source manifest. Reject incomplete or malformed windows.
4. **Detector comparison**: unchanged frozen RC3 versus (a) time aggregation alone, (b) optional fixed prelearned baseline normalization, (c) causal smoothing, (d) independent shadow head. No threshold/grid search on the untouched holdout. Record ablations and the failure/review rate.
5. **Truth**: hardware-confirmed physical gain/phase drift or controlled calibration hardware intervention with a calibration reference, plus genuinely healthy comparison sessions. *Digital injection into measured CSI remains a measurement-derived synthetic fault test*, not a measured hardware failure.
6. **Output**: fault detection TP/FN, healthy alarm fraction, neighbor alarm fraction, correct physical port, time-to-first-alert measured in actual minutes/hours, site/session-cluster confidence intervals, and failed/abstained data counts. Compare only identical observation windows and site topology.

**Current blocker:** no existing accessible dataset has yet been shown to satisfy *simultaneously* 75-hour timestamps, >=5 independently measured stationary ULA sites, aligned 15-min covariance windows and independently verified per-port hardware fault labels. The new code can test a properly captured source, but cannot produce an honest field accuracy figure from incompatible short recordings.

## Test commands

\`\`\`sh
python -m pip install -r requirements.txt pytest
python -m pytest -q generalized/real_rf_v1/test_conditioning.py generalized/real_rf_v1/test_rc3_bridge.py
python generalized/tests/test_dimensions.py
\`\`\`

The bridge test compares its 300-report 7x8 calibration vectors **element-for-element** to the original \`glr_compare.pipeline\` with only synthetic faults/dropouts/glitches disabled. The other tests exercise rejection of missing windows, physically repeated towers, invalid port/array attestations, synthetic-as-real mislabeling, and the unchanged 7x8 inference.

**Not a test on independently measured hardware faults.**


## Native timestamp follow-up: completed after this plan

2026-11-17, source-verified \`ad21\` and \`ad25\` 180 MB TFRecord prefixes were read with the genuine embedded \`time\` field and matching SHA-256. Both passed checksum/decoder audits. The actual first-300-record spans were **14.3984375 seconds** and **17.8515625 seconds**, respectively. In these prefixes, the measured times were monotonically ordered; the original dataset does not generally guarantee file-order monotonicity. Thus no compatible 75-hour 300-report campaign exists **within those audited prefixes**. See [native time audit](./DICHASUS_NATIVE_TIME_AUDIT_2026-11-17.md) and [successful CI run](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44022619055). This source-audit conclusion does not assert that every capture in the complete DICHASUS collection was tested.
