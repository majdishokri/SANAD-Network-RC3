# SANAD RC3 × Rice LensFD — actual publisher-native CSI on GitHub Actions

**Status:** Measured-data source reanalysis; frozen RC3 itself remains unchanged. 
**Branch:** \`feature/rc3-lensfd-real-measured-v1\`.
**Executed GitHub workflow:** [RC3 LensFD measured CSI workflow](https://github.com/majdishokri/SANAD-Network-RC3/actions/workflows/rc3-lensfd-measured-csi.yml).

## Source and copyright

* Rice University RENEW LensFD, **real measured**, processed complex CSI MAT files.
* Official source: https://renew.rice.edu/dataset-lens.html
* Original individual files: https://renew.rice.edu/datasets/ashu/traces/LensFD/data_LensFD/
* Citation: Zhican Chen, C. Nicolas Barati, Jon Veihl, Clayton Shepard and Ashutosh Sabharwal, **LensFD: Using Lenses for Improved Sub-6 GHz Massive MIMO Full-Duplex**, IEEE Transactions on Vehicular Technology, 2023, DOI 10.1109/TVT.2023.3240558.
* Rice grants use for research, commercial, and educational purposes, **but does not permit redistribution of the original data without permission**. The GitHub workflow streams individual source files into its ephemeral runner, deletes each MAT after measurement, and uploads **metadata / derived summaries only**.

## What the GitHub test actually does

1. SHA-256-pin ten original publisher MAT files against independently verified prior Rice downloads, before any measurement. Reject publisher substitutions or changed content.
2. Process the complete 192 original measured CSI MAT files, **112 static 2×80×52** and **80 self-interference 32×32×52**, directly from Rice, not just existing Google Drive descriptors.
3. Confirm complex array types, exact native shapes and source structure. Extract per-port RMS by averaging **finite** complex CSI magnitudes over native frequency/partner axes, never claiming these axes represent 15-minute reports or actual hardware gain coefficients. Record missing/nonfinite coefficients separately; exclude them from the RMS denominators rather than pretending they are zeros.
4. Describe a previously observed unexplained zero/low-energy port. For static A2C, compare native one-based port indices **67 and 68** across the 112 recording configurations. For SI, compare **local matrix axis-1 index 19** in captures with chain 5 vs without it. The \`-20 dB\` criterion was chosen in prior exploratory research, NOT preregistered as an independent fault threshold.
5. Recheck publisher-origin data against previously observed descriptive counts, while explicitly reporting independent hardware-fault labels **unverified**. The first ten SHA-256 controls are the same data source, so data reprocessing is an independent *implementation*, not a statistically independent RF campaign.
6. Preserve the SHA checksums and derived, labeled reports under Actions artifacts; **never** upload raw MAT, HDF5, IQ or measured CSI arrays.
7. Separately rerun the unchanged 7×8 RC3 regression test. A passed smoke/regression test does not mean a measured-CSI RC3 inference occurred.

## Why this is not a frozen RC3 field qualification

* LensFD comprises **one physical 80-port uniform planar array** with multiple radio chains; not five or seven physically independent ULA-equipped sites.
* MAT tensors have 52 OFDM subcarrier bins and 2 or 32 partner-port dimensions; these are **not 300 consecutive 15-minute reports** and cannot be resampled into fake 75-hour data.
* Rice says some radios had hardware failures, but has not provided a verified per-file port/failure-time label table, cabling/permutation map, or demonstration that an exact zero in exported CSI is a hardware defect rather than an acquisition mask.
* Therefore **frozen RC3 real LensFD inferences = 0, hardware fault Recall/FAR = NOT ESTIMABLE**. This is a physically truthful native-signal diagnostic, not an artificial time-domain RC3 test.
* The existing RC3 simulator 274/288 = 95.14% recall result remains separate and unchanged.

## Run the test

The Actions workflow checks out the branch, installs Python 3.11, NumPy/SciPy/Pytest, executes six local code-fixture guards, then executes:

\`\`\`sh
python -m generalized.tests.lensfd_real_csi_github --mode smoke --out results/lensfd_smoke
python -m generalized.tests.lensfd_real_csi_github --mode full --out results/lensfd_full
python generalized/tests/test_dimensions.py
\`\`\`

The official MAT files are downloaded directly over HTTPS with publisher-domain enforcement, bounded file sizes and retry handling. Per-file SHA-256s are produced after measurement. This source-dependent job can fail if the publisher is unavailable or changes a file.

**Next scientific gate:** obtain independently verified physical port-fault logs, element mapping, and actual timestamps; then evaluate a separately designed hardware-fault detector on untouched data and plan a prospective, at-least-five-site 75-hour RC3 qualification. Public data alone cannot bypass these requirements.
