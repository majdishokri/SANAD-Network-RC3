# Rice LensFD — which physical port is faulty? Source-level forensic audit

**Date:** 2026-11-17
**Branch:** \`feature/lensfd-failure-origin-audit\`
**Outcome:** severe **measured coherent-CSI dropout localized to a Chain-5-associated adjacent port pair**; **physical radio failure NOT CERTIFIED**; published exact-zero CSI not exactly reproducible with original public code at default settings.
**Frozen RC3:** unchanged, field inferences 0, field Recall/FAR not estimable.
**Publisher:** Rice RENEW LensFD, Chen et al., IEEE TVT (2023), DOI 10.1109/TVT.2023.3240558. Source: https://renew.rice.edu/dataset-lens.html.
**License:** original Rice data may not be redistributed without permission. All tests use ephemeral GitHub Actions memory/disk and publish only derived numeric findings.

## Source-code mapping evidence

Official processing code was downloaded at:
\`https://renew.rice.edu/datasets/ashu/traces/LensFD/code_LensFD.zip\`
SHA-256: \`3aaf2b65024753fa680b096999a104ca249234cec8a427e4488e7ea11ca1e164\`.

The original author's \`hdf5process.py\` defines \`chan_full[client/transmitter, BS_receiver_antenna, subcarrier]\`, with zero-based axis 1 the BS receive-antenna axis. The \`hdf5analysis.py\` \`cal_trace()\` routine saves \`chan_full\` and \`chan_sci\` *before* \`--bad-TX\` and \`--bad-RX\` filtering used only for analysis_power plotting. Those options cannot explain the zero entries in the **saved** published MAT as currently implemented. The \`csi2chan()\` function initializes an output matrix with zeros, then fills all antennas via phase normalization of the pilot-based CSI and complex averaging. The pilot extractor uses correlation peak alignment and FFT; a missing **coherent pilot** may yield very low CSI even when raw IQ noise is nonzero. Inspection alone does not prove where publisher MAT zeros originated.

Source-code ZIP audit: [GitHub workflow run 44026254067](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44026254067); detailed pilot extraction [run 44026531651](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44026531651).

## Descriptive measured CSI indicators (192 original Rice MAT files)

Previously repeated directly from publisher original complex MAT recordings in successful [GitHub run 44025368152](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44025368152):

- 112 static A2C: exact zero at one-based port 68 in **91/112**; exact zero at one-based port 67 in **20/112**; neither in 1.
- The suspect index **switches by lens configuration**: indoor small lens has port 67 zero in **14/14**; indoor no lens, medium and large have port 68 zero in **42/42**. This cannot straightforwardly be interpreted as one permanently failed fixed *port* without corroborated rewiring, acquisition changes or different channel processing.
- 80 SI MAT files: 32 pairs include chain 5 and **32/32** have local zero-based receiver-axis index 19 more than 20 dB below their per-capture nonzero receiver-port median; the same criterion is **0/48** among pairs not involving chain 5. **19/32** chain-5 files have an exact zero at that coordinate. Criterion -20 dB was chosen after exploratory inspection, not a prospectively preregistered independent threshold.
- Chain-5 association is strong descriptive evidence, not 32 independent hardware fault trials. The 40 radios and dual-polarized 80-port UPA are one physical base station, not five distinct stations.

## Actual native RF raw IQ against processed CSI (two *original raw HDF5* recordings)

Source HDF5 raw \`Data/Pilot_Samples\` complex IQ data:
\`[frames, cells, clients, 80 receiver antennas, interleaved IQ]\`.

[Successful raw HDF5 GitHub test](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44026390621) fetched these **two original source HDF5 files** (total 880,633,104 bytes), verified source SHA-256 and measured first-24-frame receiver RMS directly from packed I/Q without generating artificial data:

| Recording | Actual raw HDF5 SHA-256 | Port 67 raw RMS vs 80-port median | Port 68 raw RMS vs median | CSI exact-zero in published MAT |
|---|---|---:|---:|---|
| \`no-indoor-spot-1.hdf5\` | \`a793135d1e653ff9bc301e79febbaebf9fc3eb794bf10d561730a5967a73dd43\` | **+23.693 dB** | -1.139 dB | Port 68 |
| \`sm-indoor-spot-1.hdf5\` | \`c00b7691ae6055aca4ef5342d1558800d49009629652fe4a45f5bfef6422dddf\` | -4.534 dB | **+25.651 dB** | Port 67 |

**Both suspect channels contain nonzero raw IQ samples in both captures.** However, raw IQ RMS includes thermal noise, pilot-independent energy and interference, so this is NOT proof that the suspect channel received a correct coherent pilot.

## Reconstruction by executing Rice's ORIGINAL unmodified code

A separate job downloaded the exact official source-code ZIP above, the publisher's *matching raw HDF5* \`no-indoor-spot-1.hdf5\`, and its *published processed MAT* \`no-indoor-spot-1.mat\` (SHA-256 \`d60f4726e334bb7dd284cf686fd71330ea545b66a0db1752056e8605679e024c\`), and executed \`hdf5_lib.hdf5_lib()\` + \`hdf5process.process_hdf5()\` without modifying publisher mathematical code.

| Test | Recomputed CSI RMS port 67 | Recomputed CSI RMS port 68 | Published MAT port 67 RMS | Published MAT port 68 RMS |
|---|---:|---:|---:|---:|
| 24 raw frames, same code | 0.08398667 | 0.00000794598 | 0.00663632 | **exact zero** |
| 300 raw frames, same code | **0.14315080** | **0.000000635678** | 0.00663632 | **exact zero** |

*24-frame replay:* [GitHub Actions run 44026723512](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44026723512).

*Full 300-frame replay:* [GitHub Actions run 44026901573](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44026901573); every workflow step finished successfully. The published and recomputed full 2×80×52 complex matrices have an **RMSE of 0.44540945**; thus the public defaults for processing the archived matching raw file **do not reproduce** the distributed MAT. The significant *global* mismatch precludes attributing the exact-zero difference to any specific line of code or deliberate mask; likely explanations include different acquisition frame selection, unrecorded calibration/preprocessing settings, different source association, or an additional processing step. This is a set of hypotheses, NOT a demonstrated cause.

The reconstructed port 68 is about **107 dB lower in coherent CSI RMS than port 67** with all 300 frames, despite both raw IQ channels having nonzero energy. This strongly suggests absence of a decodable coherent pilot on one channel or a different defect in that receive path. Neither data nor processing code by itself proves which physical RF component is faulty.

## Which physical radio is the leading suspect?

**Candidate: receiver channel associated with Chain 5, local index 19 zero-based, plus adjacent global port labels 67/68.** If—and *only if*—physical ports are numbered sequentially across the five rows (16 ports each), and successive pairs of antenna ports belong to each two-polarization radio, this corresponds to a nominal *radio 34* with polarization ports 67 and 68. **This mapping is NOT independently certified by publisher wiring or fault logs**. Never present Radio 34 as publisher-confirmed failed hardware.

## What would settle the question

Request from Rice:
1. Published **Figure 7 failed-radio legend** and full physical chain-to-port wiring/permutation with polarization mapping.
2. Per-file \`badTXnodes\`/\`badRXnodes\` lists, including whether exported MAT zeros are software masks, missing coherent pilot, or actual physical device disablement.
3. Exact recording-to-MAT processing revision, frame offsets/subsamples, reference/normalization conventions. Why does full 300-frame replay of the matching raw HDF5 produce a nonzero port 68 and a global 0.445 complex RMSE relative to the distributed MAT?
4. Dated radio repair/failure records and independent injected-calibration/hardware truth.

## Scientific decision

- **Confirmed by source:** one chain-associated receive CSI anomaly, measured over 192 real processed CSI source captures; genuine receiver IQ exists but coherent CSI can be extremely weak.
- **Not confirmed:** an actual failed radio/port, physical port index, exact hardware fault class, onset timestamp, or software masking origin of published exact zeros.
- **SANAD frozen RC3 field inference:** 0. Existing Sionna 95.14% recall and simulated 0/1008 healthy-only alerts remain separate, not established on LensFD.

Until physical port labels/clocked measurements exist, report **REPEATABLE_REAL_MEASURED_CSI_DROPOUT / HARDWARE_FAULT_GROUND_TRUTH_UNVERIFIED**, not \`HARDWARE_FAULT_CONFIRMED\`.
