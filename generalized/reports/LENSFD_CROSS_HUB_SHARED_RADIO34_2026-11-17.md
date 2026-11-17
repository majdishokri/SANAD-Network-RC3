# SANAD RC3 × Rice LensFD — cross-hub shared SDR-34 identity and invalid pilot receiver

**Executed:** November 17, 2026.
**Experimental scope:** publisher-original **raw interleaved I/Q in eight independent HDF5 capture files** (not eight independent base stations), HDF5 SDR metadata, original raw-pilot processing and separate frozen RC3 fidelity tests.
**Working branch:** \`feature/lensfd-rx-chain-metadata-v3\`.
**Frozen RC3:** model source, trained weights, original detection threshold 0.50 and mathematics **unchanged**. No original RC3 inference on LensFD because it lacks an eligible 75-hour synchronized multi-site evidence stream.

## Data, provenance and licensing

Original measured source: Rice RENEW LensFD, Chen et al., *LensFD: Using Lenses for Improved Sub-6 GHz Massive MIMO Full-Duplex*, IEEE TVT 72(6):7517–7529 (2023), DOI [10.1109/TVT.2023.3240558](https://doi.org/10.1109/TVT.2023.3240558).

Rice dataset: https://renew.rice.edu/dataset-lens.html
Publisher HDF5 source root: https://renew.rice.edu/datasets/ashu/traces/LensFD/data_LensFD/
Source recorder metadata code: https://github.com/renew-wireless/RENEWLab/blob/master/CC/Sounder/recorder_worker.cc (attributes \`BS_SDR_ID\`, \`BS_HUB_ID\`, \`BS_CH_PER_RADIO\`, \`ANT_OFFSET\`, \`ANT_NUM\`).

Raw IQ measurements are used under Rice's research/commercial/educational license, **with redistribution prohibited without Rice's written permission**. All raw samples and complete hardware files remain on disposable GitHub Actions workers; GitHub commits and job reports store only source digests and descriptive results.

## 1. Eight original raw receiver captures, covering both environments and all lens states

[GitHub Actions run 44034116465, SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44034116465): downloaded and processed **8/8 original SHA-256 fingerprinted source HDF5**, **3,522,428,016 bytes** total. Seven source-level raw-IQ analyzer unit tests passed. The audit checked all 80 receiver channels per recording on 24 actual source frames, then analyzed the identified near-constant channels and adjacent ports over **all native capture frames**. The input-quality cutoff was previously designed from exploratory measurements; this is neither a blind detector evaluation nor hardware-label classification.

| Original Rice capture | Raw invalid channel index, 1-based | Values on affected raw signed I/Q stream | Exact interpretation |
|---|---:|---|---|
| Indoor, no lens, static location 1 | **68** | \`-16\`, \`0\` | Near-constant digitally sampled port |
| Indoor, small lens, location 1 | **67** | \`-16\`, \`0\` | Index switches by config; physical cause not proven |
| Indoor, medium lens, location 1 | **68** | \`-16\`, \`0\` | Same source-level invalid input class |
| Indoor, large lens, location 1 | **68** | \`-16\`, \`0\` | Same source-level invalid input class |
| Stadium, no lens, location 1 | **68** | \`-16\`, \`0\` | Repeats in separate hub dataset |
| Stadium, small lens, location 1 | **68** | \`-16\`, \`0\` | Repeats |
| Stadium, medium lens, location 1 | **68** | \`-16\`, \`0\` | Repeats |
| Stadium, large lens, location 1 | **68** | \`-16\`, \`0\` | Repeats |

All eight recordings identified exactly one near-constant raw port among 80 in the first-24-frame screen. **This is input-quality discrimination, not ground-truth Recall/FAR**. A nonflagged port is not certified healthy by this single test.

## 2. Two different hub identities, but the same 16 SDR boards in the last 16 positions

The raw HDF5 \`Data\` hardware metadata has:
- Indoor \`BS_HUB_ID = FH4B000019\`.
- Stadium \`BS_HUB_ID = FH4B000011\`.
- Both: \`ANT_OFFSET = 0\`, \`ANT_NUM = 80\`, \`BS_CH_PER_RADIO = 2\`, **40 recorded SDR board IDs**.
- \`BS_SDR_ID\` inventories **differ in their first 24 serials**, but share **16/40 serials at exactly the same list indices 25–40**. No shared serial is at a changed list index.
- Under the *recorder's nominal contiguous two-ports-per-SDR ordering*, one-based raw ports **67 and 68** correspond to nominal SDR-board position **34**.
- **The 34th reported radio serial is \`RF3E000446\` in both the indoor and stadium recordings.** The same metadata-reported device could have accompanied the deployment across hubs.

[Independent SHA-pinned cross-hub full 40-SDR inventory audit, SUCCESS, job 44034823641](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44034823641) verified two published original HDF5 and all 40 source-board IDs in each.

**Important qualification:** Source HDF5 board IDs prove **recorded hardware identity under the source's mapping**, not successful independent physical cabling verification, nor that board \`RF3E000446\` malfunctioned. The manufacturer/author must certify the board-to-port acquisition arrangement and whether \`-16/0\` is a muted/empty stream, disabled RF channel, driver filler, or genuine RF hardware impairment.

**This supersedes the earlier inference** that different hub IDs implied a completely different radio inventory. The two lists are *partially overlapping*, and crucially share the nominal problematic radio at slot 34. They are also not seven synchronized and physically independent RC3 receivers.

## 3. Pilot synchronization independently supports invalid receiver input

[GitHub Actions source-Rice original 300-frame pilot pipeline, SUCCESS run 44032366490](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44032366490) used the publisher's own unchanged CSI extractor on two matched original raw HDF5:

- Indoor no-lens, port 68 and indoor small-lens, port 67 each had peak-frame alignment clipped to position 0 in **599/600 native frame/client combinations**.
- Both reconstructed CSI channel RMS are approximately **6.35678e-7**; both published processed MAT mark that channel exactly zero. The raw IQ streams are almost entirely \`-16\` (99.725%/99.731%), not normally variable complex RF signals.
- Comparing Rice's published processed MAT with replayed 300-frame CSI still shows a global mismatch under default processing settings. There is **no proof** that a particular hardware fault or code line caused the exact zeros.
- Across full 401-frame raw recordings, the two bad-stream integer-value arrays are **99.9944% value-identical**, but not exactly byte-identical. This is consistent with a common absent/muted capture pattern, not proof of a manufacturer-documented fault sentinel.

## 4. Real input qualification added *around* original frozen RC3, without mathematical edits

New standalone module: [\`raw_iq_quality.py\`](../real_rf_v1/raw_iq_quality.py)
New **optional, fail-closed preflight wrapper**: [\`quality_gated_bridge.py\`](../real_rf_v1/quality_gated_bridge.py).

For any future eligible ≥5-independent-physical-site measured RC3 campaign, the optional wrapper:
1. Requires an operator-verified original I/Q channel-to-acquisition binding for **every** physical site/antenna element.
2. Requires exact coverage of those source channels; missing or extra keys are rejected.
3. Checks each original signed native int16/int32 I/Q port for near-constant two-level recordings; rejects any bad channel **before** constructing calibration evidence.
4. On full pass only, calls the **unchanged existing** \`rc3_bridge.generate_rc3_evidence\`. It does **not** substitute ports, invent healthy labels, modify RC3 training/weights/threshold or alter the calibrated estimator.
5. Outputs per-channel source SHA-256 provenance and an explicit note that a single preflight pass does not certify healthy hardware or physical-geometry validity.

[GitHub job 44034823641, SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44034823641) also passed **5/5 code-fixture preflight wrapper regression tests**. [Eight-original-RF job 44034116465](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44034116465) passed **7/7 standalone raw-IQ gate regression tests**. None of the code-fixture tests count as additional measured RF trials. The source original-simulation 7×8 parity remains a separate gate.

## Research conclusions and next decision

**Evidence-backed:** original measured RF streams are near-constant in port 68 for 7/8 tested capture configurations, and port 67 in 1/8. The faulty-*looking* port belongs to the same nominal 34th SDR position and **same recorded radio serial** across two different HDF5 hub IDs. Its original pilot alignment is almost entirely unusable. This is a concrete hardware-investigation target that a trustworthy calibration monitor should quarantine.

**Not proven:** \`RF3E000446\` is physically broken, \`-16\` is an RF firmware-documented fault signature, the 67/68 port-to-physical-polarization map is the actual cabling, or that the input-quality gate has validated field false-rejection rates.

**No RC3 field qualification:** no 75-hour synchronized five-or-more independent array stream, calibrated physical-fault ground truth, or controlled prospective hardware holdout. The historical Sionna simulated 95.14% recall is not a real-hardware result.

**Next essential external evidence:** Obtain the **physical SDR \`RF3E000446\` board status, physical A/B antenna wiring, manufacturer mode of \`{-16,0}\` output, and per-capture failed-radio / calibration / acquisition logs** from Rice or their hardware team. A documented device status is required to upgrade from **INVALID_RECEIVER_SOURCE / SUSPECT_NOMINAL_SDR34** to **CONFIRMED_HARDWARE_FAILURE**.
