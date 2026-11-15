# SANAD RC3 — LensFD Direct Full-Corpus Recovery v3

## Why this version exists

The official `data_lensFD.tgz` reports 18,898,485,248 bytes after repeated downloads but fails GNU tar/GZIP integrity with `gzip: unexpected end of file` / `tar: Unexpected EOF`. A matching advertised byte count does **not** establish a complete archive. **Do not re-download the same invalid 18-GB tarball again.**

The official publisher's top-level directory index also advertises `data_LensFD/` alongside `data_lensFD.tgz` (https://renew.rice.edu/datasets/ashu/traces/LensFD/). This notebook attempts to enumerate individual published files from that directory and download the complete files independently. The publisher's nested directory listing and individual file access are **not yet verified in Colab**; v3 tests them before beginning any transfer.

## Run

1. Open the `SANAD_RC3_LensFD_FULL_DIRECT_v3_Colab.ipynb` in Google Colab, mount Drive.
2. Run the **discovery** cell first. It fails explicitly if the directory index cannot provide a complete file manifest.
3. The full corpus loop reuses previously completed 112+ source archive members recorded in `MyDrive/SANAD_Network/RC3_LensFD_FULL/results_full/members.jsonl`, clearly marking that provenance; otherwise downloads files individually. All HDF5 numeric values are scanned with bounded-memory chunks. No sampling or silent omission.
4. New outputs are written under `MyDrive/SANAD_Network/RC3_LensFD_FULL/results_direct/`. Never delete the old `members.jsonl`.
5. If the publisher doesn't permit directory listing/direct-file downloads, stop and request a fixed archive or per-file links/checksums from the RENEW team. The notebook will *not* falsely claim full coverage.

## Practical constraints

- The known `no-s2n.hdf5` is **21,589,436,536 bytes (20.11 GiB)** uncompressed. Ensure at least 21 GiB of local scratch space plus safety margin before its direct download.
- Processed files are deleted from local scratch only after verification and numerical scanning; the previous journal remains intact.
- Prior archive prefix files may be reused without re-downloading. The report marks these as `reused_verified_archive_prefix=true` with `direct_file_sha256_revalidated=false`; set `REVALIDATE_OLD_FILES=True` for strict direct-source byte equality (additional bandwidth).
- License: Rice LensFD recordings must remain private; do not upload raw recordings to public GitHub.
- This package is a **complete real-measurement intake**, not a magic adapter to the frozen RC3 features. RC3's original model and threshold remain unchanged. No predictive scores are fabricated.

All five local tests of file enumeration, complete numeric scanning, resume, and mismatch handling passed on a controlled HTTP fixture; full publisher connectivity is checked only when the Colab runs.
