# SANAD frozen RC3 — Actual model predictions on real measured DICHASUS CSI

**Executed** 2026-11-17, reproducible GitHub Actions:
- [Full 21-case original-source real CSI inference, SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44043499351).
- [Independent 7-case full-source port-loss replication, SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44043374655).
- [Fast 160 MiB native-prefix 7-case port-loss confirmation, SUCCESS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44043819770).
- Frozen original 7×8 code-fixture bit-for-bit parity independently [PASS](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44038358011).

## Dataset and unchanged frozen model

Source: **DICHASUS ad25**, University of Stuttgart (DaRUS DOI 10.18419/DARUS-4062), genuine 3.44 GHz measured complex CSI (64 × 1024 subcarriers) with original source timestamps. The full source file **596,950,014 bytes** was verified with publisher MD5 \`3e907ad72ae6aab1aa44ef53ba60a3ef\` and whole-file SHA256 \`bbd4ff14fc30993bd865cdd37dc53f652c4f4a1beb54e4a5ab0c31d1d083cc74\`. Every original TFRecord's CRC32C was validated. No source CSI redistributed.

Frozen SANAD RC3 uses the **original trained model**, original legacy evidence pipeline (calibration vector → v11, v29, v35) and original threshold **0.50**. No changes, tuning or retraining. Double-precision covariance accumulation is solely an input-side numerical precision adjustment for strict physical covariance checks, not modified detector math.

**Critical OOD limitation:** DICHASUS has FOUR physically distinct 2×8 planar arrays, not seven independent physical receiver stations. To execute the **unchanged trained 7×8 artifact**, seven disjoint authentic 8-channel rows were taken from the four original arrays. This is a **numerical grouping only**; it is NOT a valid cross-site RC3 reference geometry. Furthermore, the 300 source snapshots span **17.8515625 seconds**, not 300×15 real minutes or 75 hours. Healthy hardware is NOT independently certified; the source is an unlabeled no-injection control.

## Confirmed model outputs — 21 source-real digital-fault tests

Full run uses **one original 300-snapshot source window**, baseline source unchanged and seven per-scenario digital single-port transformations. Injection begins at real CSI frame 160; gain/phase ramp spans 80 real snapshots. Each scenario has one target port in each of the seven numeric groups. Labels are known injected *digital* alterations, NOT hardware fault labels. All endpoint inferences executed the frozen RC3 original artifact.

| Scenario injected digitally into measured complex covariance | Target group alerted after injection | New target alerts absent from matched no-injection controls | Mean target probability change vs matched control | Cumulative untargeted-group alerts in seven trials |
|---|---:|---:|---:|---:|
| **−0.5 dB gain / +5° phase, 80 snapshot ramp** | 6/7 (85.7%) | **0/7 (0%)** | **+0.011092** | 36 |
| **−2.0 dB gain / +12° phase, 80 snapshot ramp** | 7/7 (100%) | **1/7 (14.3%)** | **+0.005523** | 34 |
| **−20 dB gain attenuation, instant digital port loss** | 7/7 (100%) | **1/7 (14.3%)** | **+0.192543** | 36 |

**No-injection control already generated 6/7 endpoint group alerts (85.7%)**. Those alerts are **not independently certified false positives**, because source hardware fault/health ground truth is not available. The apparent 85–100% target hit rates are heavily contaminated by these pre-existing control alerts. The correct diagnostic is the **paired new-alert rate**: 0/7, 1/7, 1/7. Six of seven groups were already above threshold, leaving one previously nonalerting group as the only opportunity for new alerts.

The full 21-case trial matrix and exact per-site paired predictions (including some *negative* probability deltas) are in the source-derived GitHub Actions artifact named \`sanad-rc3-independent-dichasus-digital-shadow\`, run [44043499351](https://github.com/majdishokri/SANAD-Network-RC3/actions/runs/44043499351), artifact ID **14310103424**.

The independent fast original-source-prefix benchmark downloaded **167,772,160 original bytes**; verified SHA256 of first 48MiB as \`2514ca7d55751e08ce2362e93fdec2b7f214a4a9b596b81aade06c88b2be4041\`, and validated every consumed original native TFRecord CRC32C. It did NOT claim full-file MD5; reproduced the exact same port-loss **7/7 after injection, 1/7 new target alerts**, mean paired delta **+0.192543**.

## Scientific decision

**Executed:** actual original frozen RC3 inference, after deriving calibrated 7×8 per-step evidence from authentic DICHASUS measured complex CSI. Controlled digital fault labels held out from inference and used only after prediction. Input integrity and covariance PSD checks passed.

**Negative OOD result:** Frozen RC3 demonstrates poor baseline selectivity under this invalid-for-RC3 domain/geometry/time mapping. The pre-existing 6/7 controls invalidate treating 7/7 target alerts as a useful field detection rate. Even digitally strong faults yield only 1/7 new alerts; small near-trained fault yields 0/7. Do not tune the frozen model on these source-real test cases then re-label them as independent validation.

**Unavailable and must NOT be fabricated:** verified hardware fault Recall, genuine hardware False Alarm Rate, real time-to-detect, hardware physical port localization, clustered confidence intervals, or 75h × ≥5 distinct physically synchronized site qualification. The prior 95.14% Sionna simulation recall remains separate, not disconfirmed nor validated in field RF by these tests.

## Reproduction

Full source/21 cases:
\`\`\`bash
python -m pip install numpy==2.3.5 scipy==1.17.1 scikit-learn==1.8.0 joblib==1.5.3 requests google-crc32c tfrecord protobuf tensorboard
python -m generalized.tests.dichasus_frozen_rc3_measured_shadow --blocks 1 --scenarios small_ramp large_ramp port_loss --out-dir out/dichasus_measured
\`\`\`
Quick source prefix/7 port-loss cases:
\`\`\`bash
python -m generalized.tests.dichasus_frozen_rc3_fast_original_prefix --scenarios port_loss --out-dir out/dichasus_quick
\`\`\`

The *separate* Google Colab notebook can run both modes. Full source and model hashes are checked in the code. Source raw CSI remains ephemeral; only numerical derived trial matrix and scope notes are exported. **Never report a DICHASUS 4-array shadow as RC3 7-site qualified performance.**
