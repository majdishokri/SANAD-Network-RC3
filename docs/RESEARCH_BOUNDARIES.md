# Scientific validation and claim boundaries

## What the frozen RC3 actually demonstrates

The original 2026-11-14 SANAD RC3 study used held-out NVIDIA Sionna simulator seeds 3000–3143, 7 arrays × 8 antenna elements, a 300-report structure and the pretrained model with threshold 0.50. It measured **274/288 faults detected**, **0/1008 healthy-only false alerts**, and **13/720 neighbour false alerts in injected-fault traces**. This is independent *simulation* evaluation, not on-site receiver hardware qualification.

The 2026-11-17 Accelerated Stateful CPU V1 replays that SAME simulated evidence with the SAME model and probabilities to numerical precision. It reduces computational latency but does not create a new independent test set or a first-alert time-to-detect metric. Full-window p95 CPU scoring 453.84 ms versus 842.40 ms original; radio source acquisition and CSI feature extraction are excluded from these runtime figures.

The 95% seed-cluster simulation recall interval of 92.36%–97.57% does not establish a >95% population lower bound. Zero observed simulated healthy alerts is not a promise of zero field false alarms.

## External measured-RF experiments are separate

- **Rice LensFD:** Original 80-port raw RF and processed CSI show repeated near-constant digital input in one channel and a common nominal SDR serial across two recording environments. Whether its board suffered a physically verified failure, disablement or acquisition placeholder is unresolved. Missing 75 hours and five independent synchronized sites.
- **DICHASUS adxx:** Original 64-element complex measured CSI, **four physical 16-element planar arrays**, with native milliseconds-separated timestamps. Correct source indexing, checksums, covariance reconstruction and true source time were validated on real measurements. Those tests validate the ingest path, not the frozen 7×8 calibrated field detector.
- **Frozen RC3 on regrouped DICHASUS CSI:** Seven nonoverlapping eight-element numerical groups drawn from four physical arrays, with a 300-frame interval spanning seconds, violated the model's original seven-site, 15-minute protocol. A 0.50 baseline often alerted on unmodified measured RF. Digitally injected port loss yielded additional signals but the result is **nonqualified, out-of-distribution model stress evidence**.
- **0.90 experimental research sensitivity:** An externally applied threshold selected on one earlier ad25 no-injection window reduced later unmodified-source alerts from 12/14 to 4/14 while preserving 14/14 −20 dB digitally imposed port-loss target alerts. Subtle 0.5 dB / 5° gain-phase drifts remained poorly detected. The 0.90 threshold is **not** part of the frozen RC3 release.
- **Niigata 4.85 GHz:** Published measured propagation parameters for 3,284 urban route records; the available archive lacked directly usable complex per-antenna CSI and cannot generate RC3 fault Recall/FAR.

For original source-processing and branch results see:
- [DICHASUS + Niigata measured RF](https://github.com/majdishokri/SANAD-Network-RC3/blob/feature/rc3-independent-measured-rf-ood-v1/generalized/reports/RC3_INDEPENDENT_REAL_RF_DICHASUS_NIIGATA_2026-11-17.md)
- [True model predictions, digital interventions](https://github.com/majdishokri/SANAD-Network-RC3/blob/feature/rc3-independent-measured-rf-ood-v1/generalized/reports/RC3_ACTUAL_FROZEN_PERFORMANCE_DICHASUS_DIGITAL_FAULTS_2026-11-17.md)
- [Research sensitivity chronological and separate-recording tests](https://github.com/majdishokri/SANAD-Network-RC3/blob/feature/rc3-adaptive-real-rf-sensitivity-v1/generalized/reports/RC3_REAL_RF_ADAPTIVE_SENSITIVITY_HOLDOUT_2026-11-17.md)

## What is required before real-hardware qualification

At least five *actually independent* synchronized coherent receiver arrays, verified per-element index/geometry, authentic elapsed-time recording, known healthy commissioning, precommitted fault-intervention timings and hardware truth (gain, phase and port status), and untouched prospective test episodes. Do not pad a short trace into 75 hours, relabel numerical subarrays as distinct sites, or label synthetic perturbation as real hardware failure.

**No released data establish real-hardware sensitivity, false-alarm rate, physical faulty-element localization accuracy or first-alarm time-to-detect.** No model should automatically adjust safety-critical radio hardware.
