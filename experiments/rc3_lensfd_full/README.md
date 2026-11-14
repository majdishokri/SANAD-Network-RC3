# SANAD RC3 - Rice LensFD Full Dataset Colab

**Coverage:** all regular members of the 18-GB `data_lensFD.tgz` official dataset; all numeric entries in every HDF5 dataset. No sampling or row limit. Input files are hashed to SHA-256 and a resumable JSONL audit is stored in private Google Drive. This is a measured-RF data intake and RC3 input-compatibility study, not a scientifically unfounded RC3 field-recall claim.

## Run
1. Open `SANAD_RC3_LensFD_FULL_DATA_Colab.ipynb` in Google Colab.
2. Mount your Google Drive and ensure >20 GB free in Drive, sufficient local extraction space, and available runtime duration.
3. Run all cells. Dataset archive is downloaded in full with Range resume to Drive, SHA-256 hashed, streamed from the `.tgz` into one temporary member at a time, full numeric HDF5 data scanned, and every file hashed; checkpoints let a repeated run resume.
4. Inspect `FULL_AUDIT_SUMMARY.json`, `members.jsonl`, and `RC3_COMPATIBILITY.json` in Drive.

## Scientific input gate
**Do not reshape LensFD data to match RC3**. Frozen RC3 requires a 300-report, 7-array Sionna-derived evidence format (`*_feats`, `*_delta`, `*_v11`, `*_v11_named`), with its own upstream signal pipeline, and LensFD radio chains were recorded separately in up to 32-port runs. Without measured-domain fidelity and a real time alignment/feature adapter, scoring these data with RC3 would yield invented evidence. Notebook tests the code integrity but reports `NO_VALIDATED_LENSFD_TO_RC3_FEATURE_ADAPTER` for LensFD instead of fabricated metrics. This is a deliberate gate; not a data subset or a hidden sample test.

## Source / license
Rice University LensFD: https://renew.rice.edu/dataset-lens.html ; published 2023 by Chen, Barati, Veihl, Shepard and Sabharwal. Rice grants a non-exclusive, non-transferable use license for commercial/educational/research purposes and prohibits redistribution without permission. Keep the 18-GB archive and source recordings private. Do not upload them into public GitHub repositories.
