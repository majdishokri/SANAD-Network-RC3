# SANAD RC3 — Real-RF Conditioning V1 (experimental)

**Status: executable engineering prototype; not an RF field-qualified detector.**

This is a *new* development branch based on \`feature/rc3-variable-geometry\`.
The canonical freeze \`d19d9b8d598601e512757d8ec36468e95f1dc9f1\`,
trained RC3 model SHA-256 \`0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4\`,
threshold 0.50, existing feature formulas and its reported 95.14% Sionna result are **unchanged**.
None of the new code is imported into frozen RC3 inference.

## What is implemented

1. **Real 15-minute time alignment** — \`WindowAligner\` accepts hardware UTC timestamps,
   site/device/session IDs and measured complex CSI \`[M,K]\` or measured Hermitian PSD
   covariance \`[M,M]\`. The covariance from CSI is \`H H^H/K\` *within an observation*:
   OFDM tones are not treated as elapsed time or statistically independent channel draws.
   Output requires 60+ actual acquisitions spread across >=48 of 60 temporal slots and
   <=60 s worst acquisition gap by default. Incomplete windows are rejected; missing
   entire windows raise errors rather than being fabricated.

2. **Synchronized 75-hour series and geometry check** — \`validate_run\` requires
   **300 consecutive 900-second reports = 75 elapsed hours**, the same UTC-window starts
   across sites, >=5 *distinct physical stations*, coherent physical element identities,
   and uniform M among stations. N=5,7,8,... and M=4,8,16,32,64 may pass the **shape/timing
   compatibility check**, but this does NOT validate model sensitivity, and **4 sites
   are explicitly rejected** because the unchanged RC3 reference rule needs five.
   Mixed M_i remains unsupported by the generalized RC3 classifier; no padding/relabeling.

3. **Healthy-only per-station normalization** — \`BaselineNormalizer\` uses >=120
   explicitly verified healthy reports (30 h) from a designated session to learn
   median per-element power. Its diagonal power congruence preserves covariance PSD.
   The baseline remains locked during scoring; no auto-contamination by silent
   presumed-healthy periods. Common-mode gain is removed from the relative-power
   diagnostic. This is **new input conditioning**, so its effect on frozen RC3
   feature distribution and accuracy is **unvalidated**, not 7x8 parity-preserving.

4. **Causal temporal companion** — \`ObservationFilter\` takes per-element relative
   change scores, uses trailing median + repeated report persistence, reports
   \`transient_review\` and \`persistent_change\` separately. It does not suppress or
   overwrite the unchanged frozen RC3 probability/alert output. Physical environment
   shifts and persistent electronic faults are *not necessarily identifiable* from
   covariance alone.

5. **Sim-to-real SHADOW adaptation** — \`ShadowHead\` trains a *separate* logistic
   candidate from precomputed valid real-RF features, with digitally injected fault
   labels and **disjoint** healthy-only calibration sessions; \`OneClassThreshold\`
   calibrates a held-out healthy novelty cutoff. It **does not fine-tune the frozen
   RC3 weights** or overwrite 0.50. No new trained artifact is shipped because
   no validated 75-hour measured time series with independent sessions was supplied.
   A digitally injected fault is not a certified physical hardware failure.

## Why past measured datasets cannot simply be replayed as 15-minute training data

- **DICHASUS ad25/ad21:** first 300 CSI samples represent only ~14.4 seconds
  (48 ms/sample), NOT 75 hours. Seven/eight ULA rows also share only four physical
  sites, so treating rows as independent towers breaks the original reference model.
- **Rice LensFD:** static/mismatched acquisition configurations, not the contiguous
  7-independent-site x 300-report evidence contract; independent hardware fault
  labels and verified port mapping are absent.
- **Yokohama/Niigata Zenodo MPC:** real multipath inputs, but modeled ULA spatial
  covariance and pseudo-sites were reconstructed; not simultaneously measured
  independent 15-minute base-station covariance streams.
- **Rice/RENEW and ESPARGOS:** measured CSI is useful for per-array exploratory
  adaptation; it does not establish a 75-hour synchronized 7-site stream.
  A 4-site ESPARGOS recording cannot run the frozen RC3 five-site reference rule.

Thus there is **no legitimate new RC3 field Recall/FAR/TTD** from this module.

## Run automated tests

\`\`\`bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q generalized/real_rf_v1/test_conditioning.py
\`\`\`

Usage sketch (real 15-minute source time is mandatory):

\`\`\`python
from generalized.real_rf_v1 import WindowAligner, Observation
aligner = WindowAligner()
# hardware_csi: complex [M, K]; hardware_utc: actual capture timestamp in UTC epoch seconds
completed = aligner.push(Observation(
    timestamp_s=hardware_utc, site_id="tower-01",
    physical_station_id="verified-device-01",
    session_id="2026-11-acquisition-01", csi=hardware_csi))
# Only accepted completed WindowQuality reports can enter validate_run().
\`\`\`

## Next qualification gates

- Obtain physical device IDs, calibrated per-port map and true UTC timestamps.
- Collect >=75 hours with verified healthy reference sessions and held-out site/session
  splits, preferably from >=7 **independent** simultaneous receiver stations;
  acquire longer repeated seasons/environments for drift and transient controls.
- Externally validate the now-implemented *measured covariance → original RC3
  v11/v29/v33/v35 evidence* bridge on an independently eligible real 75-hour
  multi-site dataset. **Code-fixture parity is tested; field performance is not qualified.**
- Freeze adapters/normalization thresholds before evaluation; report TP/FP,
  neighbour FP, device-cluster 95% intervals, true first-alert time,
  localisation, data dropout and latency at the same reporting horizon.
- Compare original RC3 baseline, baseline-only conditioning, smoothing-only
  postprocessing and a separately trained shadow head on exactly matched held-out
  real streams; do not retrofit 95.14% simulation performance.

**No hardware actuation. No model weights modified. No false time interpolation.**


## Raw I/Q source-integrity preflight (Rice LensFD discovery, November 17, 2026)

\`raw_iq_quality.py\` adds a standalone **ABSTAIN** check for invalid or
digitally muted raw receiver streams. It is not part of the frozen RC3
classifier and does not decide whether an actual antenna failed.

This check was motivated by **genuinely measured Rice LensFD HDF5** records:
- No-lens indoor static spot 1: raw port 68 (one-based), all 401 captured
  frames, **1,796,480 native I/Q integer values**, only \`-16\` and \`0\`,
  modal fraction 99.725%; exported CSI port 68 is exactly zero.
- Small-lens indoor static spot 1: raw port 67, all 401 frames, same
  two integer levels, modal fraction 99.731%; exported CSI port 67 zero.
- Across recordings, the suspect raw I/Q values match in 99.9944% of
  positions, although the full raw streams are **not bitwise identical**.
- Adjacent measured strong-signal ports have 321 and 412 distinct
  integer values, respectively. Thus neither near-constant stream is
  adequate for calibrating complex RF phase/gain drift.
- With the publisher's unchanged 300-frame pilot extractor, both
  near-constant ports had correlation alignment clipped to start index
  zero for 599/600 client-frame combinations, confirming unusable
  pilot-timing evidence.
- This establishes near-static **recorded digital IQ**, not whether the
  source is a failed physical radio, disabled ADC, lost samples, or
  acquisition placeholder.

Application:
\`\`\`python
from generalized.real_rf_v1.raw_iq_quality import (
    analyze_receiver_iq, require_qualified_raw_iq, RFInputNotQualified
)
# raw_one_port: native interleaved I,Q signed int16/int32
quality = analyze_receiver_iq(raw_one_port)
if quality.status.startswith("ABSTAIN_"):
    # Quarantine source and physical-port map; do not feed fake covariance.
    print(quality.as_json())
else:
    # Passing this ONE gate does not certify that CSI/clock/geometry is valid.
    ...
# Alternatively, require_qualified_raw_iq(raw_one_port) fails closed.
\`\`\`

The conservative trigger is: >=10,000 original I/Q values; at most two
distinct signed integer levels; largest level occupies >=99.5% of
observed values. Signed **int16 or int32 HDF5 storage** is preserved
exactly and per-port SHA-256 supplied. Threshold is an **input-quality
policy** and its false-rejection properties have not been field-calibrated.
Failing this gate means **do not infer RC3 calibration faults from this
raw stream**, not \`HARDWARE_FAULT_CONFIRMED\`.

Source-level GitHub Actions experiments:
- https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44032366490
- https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44032734241

When original raw IQ is unavailable, this gate cannot certify or deny
source integrity. Do not claim a healthy source from its absence.
