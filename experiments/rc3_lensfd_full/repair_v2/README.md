# SANAD RC3 LensFD FULL DATA — Repair v2

Run `SANAD_RC3_LensFD_FULL_REPAIR_v2_Colab.ipynb` from the top in Colab. It uses your existing `MyDrive/SANAD_Network/RC3_LensFD_FULL` folder and `members.jsonl`, never samples records, and does not alter frozen RC3.

## Root cause diagnosis
At archive member 113 the extracted `no-s2n.hdf5` is 21,589,436,536 bytes (20.11 GiB). The prior per-member `ERROR` and `tarfile.ReadError: unexpected end of data` may indicate: (a) insufficient `/content` disk for full extraction, (b) incomplete/corrupted archive or truncation of the last gzip/tar segment. Check last 5 rows of `results_full/members.jsonl`. The new notebook checks disk and the *full* TAR/GZIP integrity before running the expensive scan.

## Recovery design
- Read complete TAR/GZIP to EOF with GNU tar; refuses incomplete archive (no trust in transfer byte count alone).
- When existing public dataset archive is invalid, records the evidence in `ARCHIVE_REPAIR_EVENT.json`, removes only the invalid archive to free Drive storage, and resumes/re-downloads with HTTP Range as available. `members.jsonl` stays untouched.
- New archive with a new SHA: every old `OK` member is rehashed against the new archive before its HDF5 numerical summary is reused. Prior successes are never silently trusted on different source bytes.
- Missing `/content` disk aborts the affected member with a precise diagnostic. To process the 20.11-GiB uncompressed HDF5, select a Colab runtime with at least ~21 GiB free *local scratch* (plus margin). No data is dropped or sampled.
- All errors and partial progress are saved in `FULL_AUDIT_SUMMARY.json` and `members.jsonl`.

**Source:** Rice RENEW LensFD, https://renew.rice.edu/dataset-lens.html. Keep raw data private according to the publisher licence.

**RC3:** integrity and qualified Sionna results are unchanged. The LensFD audit does not fabricate native RC3 evidence inputs or claim RC3 LensFD fault recall.
