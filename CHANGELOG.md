# Changelog

## 2026-11-17 — Public release presentation

- Featured the unchanged **RC3 Accelerated Stateful CPU V1** as the recommended research reference.
- Published verification, security, contribution and validation-scope guidance; added public integrity checks.
- Preserved historical frozen RC3 source, original threshold **0.50**, trained model, simulation evidence and accelerated ZIP without alteration.
- Stated the independent Sionna results and separate measured-RF limitations without marketing real-hardware qualification.

## 2026-11-17 — Frozen accelerated reference

- 144-seed exact output parity to original frozen RC3; CPU full-stream p95 improved from **842.404 ms** to **453.839 ms** on replayed simulated evidence.
- Original simulated recall **274/288**, healthy false alerts **0/1008**, neighbour alerts **13/720**.
- CPU timing excludes RF acquisition and does not establish earlier first-alert time.

## 2026-11-14 — Original frozen RC3 qualification

- Preserved pretrained two-stage classifier and decision threshold 0.50.
- Passed three preregistered acceptance gates on 144 new Sionna simulation seeds.
- No real hardware calibration-fault qualification asserted.
