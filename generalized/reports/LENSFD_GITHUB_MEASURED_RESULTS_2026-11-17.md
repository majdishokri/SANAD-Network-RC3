# SANAD RC3 × Rice LensFD — real measured publisher-CSI execution

**Date:** 2026-11-17
**Result:** EXECUTED SUCCESSFULLY on real measured complex CSI.
**Source:** Rice University RENEW LensFD, Chen et al., IEEE Trans. Vehicular Technology (2023), DOI 10.1109/TVT.2023.3240558.
**Primary execution evidence:** [GitHub Actions run 44025368152](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44025368152) — all workflow steps passed, including source download, source shape checks, diagnostic replication and frozen 7×8 RC3 regression.
**Action head SHA:** \`c717705b6bd7833dfa0dda6d906f0de8059fd181\`.
**Actions artifact:** \`rice-lensfd-measured-192-native-csi-research\` (GitHub artifact ID \`14302299544\`, SHA-256 \`c56514dbdce65082c000375e0a93f79cb572cfd88ffa88d9d45ed572fcffcd3d\`), contains derived JSON diagnostics and per-file source hash manifest, NOT native MAT/CSI.

## Real publisher source ingestion

- Fetched individual native processed \`.mat\` files **directly from the official Rice server**, not cached Google Drive-derived descriptions.
- **192 / 192** successfully re-read, native shape/type validated.
- **112 static array-to-client** captures: complex \`chan_full[2,80,52]\` + \`chan_sci[2,80]\`.
- **80 base-station self-interference** captures: complex \`chan_full[32,32,52]\` + \`chan_sci[32,32]\`.
- Total processed complex-CSI numeric values scanned including \`chan_full\` and \`chan_sci\`: **5,291,520**.
- Bytes fetched in full 192-file pass: **84,719,616 bytes** (84.72 MB decimal), with a separate ten-file smoke pass of **4,290,368 bytes**.
- Ten designated original files were verified against independently recorded SHA-256 digests; SHA-256 of all 192 publisher downloads was also **computed and recorded** in the execution artifact. Recording a new hash for the other 182 does NOT constitute an independent known-hash comparison.
- Original published .mat data were deleted from each ephemeral GitHub runner work file as each file was processed; no original CSI, IQ or raw RF arrays committed or uploaded.
- Some real self-interference source matrices contain NaN/Inf coefficients. These were **counted and excluded from valid-channel RMS denominators**, never silently replaced by valid zero-power observations.

## Real measured, descriptive diagnostic outcomes

| Test | N source captures | Findings | Interpretation |
|---|---:|---:|---|
| Static A2C: port 68 exactly zero in processed CSI | 112 | **91/112** | Nonzero covariance data exist elsewhere; cannot infer physical port failure |
| Static A2C: port 67 exactly zero | 112 | **20/112** | Configuration-specific index shift; source physical port map remains unverified |
| Static A2C neither exact-zero port 67 nor 68 | 112 | **1/112** | Confirms not every file fits an assumed single failed port |
| SI pair includes chain 5: axis-1 local index 19 below -20dB | 32 | **32/32** | Strong repeated hardware-chain-associated signal indicator, possibly acquisition masking |
| SI pair does not include chain 5: same local index 19 below -20dB | 48 | **0/48** | Descriptive source-domain comparison, not a randomized independent control |
| SI pair includes chain 5: same axis-1 index exactly zero | 32 | **19/32** | Actual exported complex CSI is zero at this coordinate |

The -20 dB cutoff was **selected during earlier exploratory inspection**. Its reappearance is a valid technical reproduction but **not** a new prospectively registered sensitivity/specificity test. Captures share physical radios and are correlated; \`32/32\` does not mean 32 independent equipment faults. CSI matrix axes 0 and 1 are not labeled physical TX/RX by this experiment.

## Frozen RC3 result and qualification

**Frozen RC3 field predictions on LensFD: ZERO.**

- No 300-report × real 15-minute consecutive histories over 75 hours.
- No five independent physical receiver sites; LensFD is one rectangular UPA with 80 antenna ports and multiple collection chains.
- No verified data-to-physical-port routing and exact fault onset/repair labels.
- No valid way to infer either fault-recall or false alarm rate from these captures without changing the evaluation target.

Therefore **RC3 field Recall = NOT ESTIMABLE** and **RC3 field FAR = NOT ESTIMABLE**. The original RC3 frozen 7×8 bit-for-bit feature/meta/prediction comparison passed in the same GitHub Actions job, and the variable-array interface regression passed for 4×4, 8×16, 16×32, 32×64 and 64×8 shapes; none are new field accuracy qualifications.

## Test status and reproducibility

- Python 3.11; NumPy 2.3.5, SciPy 1.17.1, pytest.
- Offline parsing/claim-boundary tests: **6 passed**.
- Publisher-source ten-file SHA-pinned test: **passed**.
- All-original 192-file measured-CSI reanalysis: **passed**.
- Historical descriptive-count revalidation: **passed**.
- Frozen 7×8 parity and variable-geometry shape tests: **passed**.
- GitHub Actions workflow conclusion: **success**.

Source code: [\`generalized/tests/lensfd_real_csi_github.py\`](../tests/lensfd_real_csi_github.py)
Test code: [\`generalized/tests/test_lensfd_real_csi.py\`](../tests/test_lensfd_real_csi.py)
Workflow: [\`.github/workflows/rc3-lensfd-measured-csi.yml\`](../../.github/workflows/rc3-lensfd-measured-csi.yml)

The source authors specify Rice data must **not** be redistributed without permission. Cite Rice LensFD in any publication or product reporting.

**Next gate:** independently confirm with Rice the physical 80-port mapping, source CSI export zeros vs masked channels, which radios actually failed, and their acquisition/failure timestamps; only then can a separately calibrated, source-compatible hardware-fault benchmark be constructed. Do not overwrite RC3, invent time reports, split UPA into fake physical stations or report 95% real-hardware fault recall.
