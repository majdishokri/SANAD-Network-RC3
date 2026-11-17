# SANAD RC3: independently rechecked authentic DICHASUS RF timestamps

**Executed:** 2026-11-17
**CI:** [GitHub Actions run 44022619055](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44022619055): both original data jobs **SUCCESS**, 4/4 timestamp parser tests in each.
**Source:** DICHASUS ad21 and ad25 measured CSI TFRecord, DaRUS DOI [10.18419/DARUS-4062](https://doi.org/10.18419/DARUS-4062).

## What was actually done

Two source-recording prefixes of **180,000,000 bytes each** (360,000,000 total) were downloaded from the original provider over strict HTTP Range. The downloaded slices were verified with their PREEXISTING prefix-specific SHA-256 digests. This verified the *downloaded 180 MB prefix*, not the complete 596.95 MB or 1.553 GB source file.

A new direct protobuf decoder extracted the *actual recorded TFRecord \`time\` float32 field* for the first 300 CSI observations from each source. This is source-provided elapsed time in seconds since campaign start, **not 48 ms times invented by indexing samples**, not wall-clock UTC epochs, and not a 15-minute RC3 report schedule.

| Source | File ID | Prefix SHA-256 | 300 measured time records | Real elapsed span | Median interrecord | Maximum sorted gap | Backward time steps |
|---|---:|---|---:|---:|---:|---:|---:|
| ad21 | 664462 | \`1c35fe576de3b604d45fb65a91a6942d6918cbdd60242cea01f65847fbf1fb60\` | 300/300, all unique | **14.3984375 s** | 0.046875 s | 0.09375 s | 0 |
| ad25 | 664466 | \`e1f6cfd47af0d73c4d494fd73d676d7517401a8493d5ce5c2aac83abaa1e230a\` | 300/300, all unique | **17.8515625 s** | 0.046875 s | 0.53125 s | 0 |

These are the first 300 *file records*, not a random or full-campaign estimate. The original publisher warns file ordering is not generally guaranteed, although both checked prefixes happened to be monotonic. We must sort by the source \`time\` property before downstream temporal analysis when a broader set is ingested.

## Pre-registered frozen RC3 compatibility decision

**BLOCKED on real RF qualification**, with exactly four reasons:

1. Minimum needed duration: 300 accepted 900-second reports = **270,000 seconds / 75 actual hours**. Observed first-300-source spans: 14.4 and 17.85 seconds.
2. Original cross-site reference requires >=5 distinct PHYSICAL stations; DICHASUS adxx has **4 distinct receiver array sites** with two eight-element ULA rows per site. Seven selected rows cannot be relabeled as seven physical base stations.
3. No independent, verified actual hardware-port gain/phase failure truth on these source records.
4. No 300 consecutive, valid 15-min covariance windows with known, verified healthy first-120-report commissioning periods.

Neither prefix's source RF content was modified or redistributed to GitHub. Only the timestamp summary JSON files were uploaded as CI artifacts.

## Consequences for older benchmark claims

An earlier **DICHASUS measured-CSI RC3 stress run** reported 7/8 digitally injected faults detected at 7x8 and 26/28 alerts on unlabeled controls. Those are valid *research experiment counts*, **not** 75-hour operational Recall / FAR. The actual source timestamps further show its first 300 nominal reports represent only seconds; representing these as 15-minute windows would constitute time-scale distortion.

The frozen Sionna qualification result (274/288 = 95.14% fault detection; 0/1008 healthy-only alerts) is not invalidated; it is a separate simulated-domain result. No claim of equal accuracy in measured RF is established by this audit.

## Reproduce

\`\`\`bash
python -m pip install numpy==2.3.5 pytest
python -m pytest -q generalized/real_rf_v1/test_dichasus_real_timestamp_audit.py
python -m generalized.real_rf_v1.dichasus_real_timestamp_audit \
    --dataset ad21 --out /tmp/ad21_source_time.json
python -m generalized.real_rf_v1.dichasus_real_timestamp_audit \
    --dataset ad25 --out /tmp/ad25_source_time.json
\`\`\`

\`dichasus_real_timestamp_audit.py\` rejects a source download with a wrong byte range or checksum, a truncated first-300-record set, or a source missing its true \`time\` float32 field. No synthetic time conversion or fault injection exists in this measurement audit.

**Scientific status: real measured-source timestamp audit passed, RC3 real-hardware fault qualification not performed.**
