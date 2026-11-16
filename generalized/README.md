# SANAD RC3 variable-geometry copy (unqualified)

This is the original RC3 algorithm with dimensional constants replaced by
`SANAD_STATIONS` (N base stations) and `SANAD_ELEMENTS` (M antenna
elements per base station). Defaults are 7 stations × 8 elements.

No changes to the 693-feature definitions, 15 spatial-meta features, detector
rules, fitted weights or decision threshold (0.50). The existing trained
model is reused. It is **not retrained or validated for alternative dimensions**.
The 95.14% recall and 0/1008 healthy-only false positives apply exclusively
to the frozen 7×8 Sionna study.

```bash
pip install -r requirements.txt
SANAD_STATIONS=4 SANAD_ELEMENTS=16 python generalized/rc3.py evidence_4x16.npz --stream injected
SANAD_STATIONS=16 SANAD_ELEMENTS=32 python generalized/rc3.py evidence_16x32.npz --stream injected
```

Inputs require 300 periods; `*_feats` = (2,300,N,3),
`*_delta` = (300,N,3), `*_v11` = (300,N),
`*_v11_named` = (300,N). Covariance source: (300,N,M,M).
Changing environment settings does not convert old evidence into new evidence.

`runtime/sionna_cuda_v34.py` preserves the original Sionna UMa channel model,
extending the original hexagonal station layout with more rings. That changes
the geography for more than seven stations, not the detector equations.

The original v29 reference requires five available base stations; for a
four-station setup that path cannot form a reference, so sensitivity may
degrade severely. Preserving this rule is intentional: changing it would
change the algorithm. Other geometries require retraining on new data using
the same model family, and an independent prospective qualification study
before performance claims. No RF actuation or field qualification is provided.

The root `rc3.py`, original runtime, models and freeze manifest are untouched.
