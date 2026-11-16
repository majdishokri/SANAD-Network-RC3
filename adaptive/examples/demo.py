"""Deterministic demonstration, NOT a detector qualification."""
import json
from dataclasses import asdict
import numpy as np
from sanad_adaptive import ArraySpec, RFMeasurement, SANADAdaptiveNetwork

rng=np.random.default_rng(77)
specs=[ArraySpec('A4',np.arange(4)[:,None].astype(float),('B8',)),
       ArraySpec('B8',np.arange(8)[:,None].astype(float),('A4','C16')),
       ArraySpec('C16',np.arange(16)[:,None].astype(float),('B8',))]
det=SANADAdaptiveNetwork(specs)
for t in range(44):
    batch={}
    for spec in specs:
        gain=rng.normal(0,0.008,spec.elements)
        phase=rng.normal(0,0.09,spec.elements)
        if t>=30 and spec.array_id=='B8':
            gain[3]+=0.85
            phase[3]+=7.0
        batch[spec.array_id]=RFMeasurement(gain,phase)
    out=det.update(batch)
for key,d in out.items():
    print(json.dumps({'array':key,'elements':d.element_count,'status':d.status,
          'candidate_1based':None if d.candidate_element is None else d.candidate_element+1,
          'score':d.score,'neighbor_contrast':d.neighbor_score_contrast},indent=2))
