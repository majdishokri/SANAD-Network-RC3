"""SANAD v35 learned-gate research inference. No RF actuation.

The model is trained on already-reported simulator seeds (1264..1407).
Model cutoff selected using retrospective, seed-grouped OOF folds.
It is not field-qualified and must not command antenna corrections.

Decision for each array uses ONLY Sionna-derived time-series evidence from
reports 0..299 and the frozen v35/v33 detectors; neither injection truth nor
stream type nor seed ID enters model features.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import joblib
from feature_engineering import features_per_array
from geometry_config import N_ARRAYS

@dataclass(frozen=True)
class LearnedGateDecision:
    array_id: int
    v35_candidate: bool
    v33_conservative_alert: bool
    learned_confirmed_alert: bool
    probability: float
    confirmation_threshold: float
    status: str
    action: str = 'NO_HARDWARE_ACTUATION'


class FrozenV35LearnedGate:
    def __init__(self, model_path: str | Path | None = None):
        p = Path(model_path) if model_path is not None else Path(__file__).resolve().parent/'frozen_candidate'/'selected_model.joblib'
        self.bundle=joblib.load(p)  # only load the shipped trusted model; joblib is not safe for untrusted files
        if self.bundle.get('frozen_v35_config')!='legacy_union':raise ValueError('Unexpected source policy')
        self.threshold=float(self.bundle['selected_threshold'])
        if not 0. < self.threshold <=1.:raise ValueError('Unexpected confirmation threshold')

    def predict_features(self, x:np.ndarray, columns:list[str]):
        """One endpoint per array (7 rows). Feature extraction is separate and causal."""
        x=np.asarray(x,dtype=float)
        if x.shape != (N_ARRAYS,len(columns)) or not np.isfinite(x).all():
            raise ValueError('Expected 7 finite array feature vectors')
        colmap={k:i for i,k in enumerate(columns)}
        if len(colmap)!=len(columns):raise ValueError('Duplicate feature names')
        names=self.bundle['names']
        if any(k not in colmap for k in names+['v35_fired','v33_fired']):
            raise ValueError('Required features missing')
        X=x[:,[colmap[k] for k in names]]
        p=np.zeros(N_ARRAYS,float)
        for name,weight in self.bundle['weight'].items():
            p+=float(weight)*self.bundle['trained_models'][name].predict_proba(X)[:,1]
        candidate=x[:,colmap['v35_fired']]>0
        confirmed=candidate&(p>=self.threshold)
        conservative=x[:,colmap['v33_fired']]>0
        output=[]
        for i in range(N_ARRAYS):
            status=('LEARNED_RESEARCH_ALERT' if confirmed[i] else
                    'REVIEW_ONLY' if candidate[i] or conservative[i] else 'MONITORING')
            output.append(LearnedGateDecision(i,bool(candidate[i]),bool(conservative[i]),
                            bool(confirmed[i]),float(p[i]),self.threshold,status))
        return output

    def predict_trace(self, trace:dict, stream:str):
        x,names=features_per_array(trace,stream)
        return self.predict_features(x,names)

    def score_trace(self, trace:dict, stream:str):
        """Research only. Scoring reads labels AFTER inference; never during inference."""
        decisions=self.predict_trace(trace,stream)
        expected=np.asarray(trace[f'{stream}_expected'])>0
        active=np.array([d.learned_confirmed_alert for d in decisions],bool)
        if stream=='healthy':
            return {'healthy_fp':int(active.sum()),'healthy_n':N_ARRAYS,'fault_tp':0,'fault_n':0,'neighbor_fp':0}
        if stream!='injected':raise ValueError('Unknown stream')
        return {'healthy_fp':0,'healthy_n':0,'fault_tp':int((active&expected).sum()),
                'fault_n':int(expected.sum()),'neighbor_fp':int((active&~expected).sum())}

if __name__=='__main__':
    with np.load('/mnt/data/s1264.npz',allow_pickle=False) as z: trace={k:z[k] for k in z.files}
    model=FrozenV35LearnedGate()
    for stream in ('healthy','injected'):
        print(stream,model.score_trace(trace,stream))
