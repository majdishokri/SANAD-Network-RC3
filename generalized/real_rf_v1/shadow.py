"""Opt-in sim-to-real SHADOW head. Never edits frozen RC3 weights or threshold.

Inputs are already constructed, *time-valid* 300-report RC3-compatible
features and/or derived scores, NOT raw CSI frames treated as 15-min reports.
Only explicit train/calibration session separation is accepted.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

FROZEN_MODEL_SHA256 = "0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4"

@dataclass(frozen=True)
class OneClassThreshold:
    threshold: float
    alpha: float
    calibration_n: int
    qualification: str = "EXPERIMENTAL_HEALTHY_ONLY_CALIBRATION"

    @classmethod
    def fit(cls, verified_healthy_scores, *, alpha=0.02):
        """Finite-sample rank cutoff. Correlated site/session samples are NOT IID."""
        s = np.asarray(verified_healthy_scores,dtype=float)
        if s.ndim != 1 or not np.isfinite(s).all() or not 0<alpha<1:
            raise ValueError("Need finite 1D healthy scores and 0<alpha<1")
        if len(s)<100:
            raise ValueError("Need >=100 healthy calibration scores for this exploratory gate")
        rank = int(math.ceil((len(s)+1)*(1-alpha)))
        cutoff = float(np.sort(s)[rank-1]) if rank<=len(s) else float("inf")
        return cls(cutoff,float(alpha),int(len(s)))

    def candidate(self, scores):
        s = np.asarray(scores,dtype=float)
        if not np.isfinite(s).all(): raise ValueError("Nonfinite candidate scores")
        return s > self.threshold  # conservative tie-handling

class ShadowHead:
    """New independent logistic head trained with digitally injected anomalies.

    This is NOT an in-place fine-tune of RC3's meta-model and cannot inherit
    RC3's simulation performance. No field claim without certified fault labels.
    """
    def __init__(self, *, frozen_source_sha=FROZEN_MODEL_SHA256):
        if frozen_source_sha != FROZEN_MODEL_SHA256:
            raise ValueError("Unapproved model provenance")
        self.frozen_source_sha = frozen_source_sha
        self.coef = None
        self.intercept = None
        self.center = None
        self.scale = None
        self.calibration = None
        self.training_groups = ()
        self.calibration_groups = ()

    @staticmethod
    def _check_x(X):
        x = np.asarray(X,dtype=float)
        if x.ndim!=2 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError("Expected finite [cases,features] matrix")
        return x

    def fit(self, X, y, groups, *, calibration_X, calibration_y,
            calibration_groups, source_kind, source_time_verified,
            labels_provenance="digital_fault_injection", alpha=0.02):
        """Separate TRAIN and HEALTHY CALIBRATION sessions, never new-data eval."""
        if source_kind != "measured_RF" or not source_time_verified:
            raise ValueError("Require physically measured RF plus verified 15-min report times")
        if labels_provenance != "digital_fault_injection":
            raise ValueError("Pilot supports only explicit digitally injected fault labels")
        x=self._check_x(X); cx=self._check_x(calibration_X)
        labels=np.asarray(y,dtype=int); cy=np.asarray(calibration_y,dtype=int)
        g=np.asarray(groups,dtype=str); cg=np.asarray(calibration_groups,dtype=str)
        if (len(x)!=len(labels) or len(x)!=len(g) or len(cx)!=len(cy)
            or len(cx)!=len(cg) or cx.shape[1]!=x.shape[1]):
            raise ValueError("Sample/feature shape mismatch")
        if set(g).intersection(set(cg)):
            raise ValueError("Train/calibration acquisition sessions overlap")
        if len(set(g))<2 or len(set(cg))<2:
            raise ValueError("Require >=2 independent acquisition sessions in each split")
        if set(labels)!={0,1} or set(cy)!={0}:
            raise ValueError("Training needs digital injected positives and negatives; calibration strictly healthy")
        if len(cx)<100 or min(np.bincount(labels))<10:
            raise ValueError("Insufficient train or calibration cases")
        from sklearn.linear_model import LogisticRegression
        self.center = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0),1e-8)
        normalized = (x-self.center)/self.scale
        model = LogisticRegression(C=0.25,max_iter=2000,random_state=20261117)
        model.fit(normalized,labels)
        self.coef=np.asarray(model.coef_[0],dtype=float)
        self.intercept=float(model.intercept_[0])
        p=self.predict_proba(cx)
        self.calibration=OneClassThreshold.fit(p,alpha=alpha)
        self.training_groups=tuple(sorted(set(g)))
        self.calibration_groups=tuple(sorted(set(cg)))
        return self

    def predict_proba(self,X):
        if self.coef is None: raise ValueError("Shadow head not trained")
        x=self._check_x(X)
        if x.shape[1]!=len(self.coef): raise ValueError("Wrong feature count")
        z=((x-self.center)/self.scale)@self.coef+self.intercept
        return 1/(1+np.exp(-np.clip(z,-40,40)))

    def predict_experimental(self,X):
        p=self.predict_proba(X)
        return {"probabilities":p.tolist(),
                "research_candidates":self.calibration.candidate(p).tolist(),
                "qualification":"DIGITAL_FAULT_STRESS_ONLY_NO_FIELD_RECALL_OR_FAR",
                "original_rc3_threshold_unchanged":0.50}

    def export_manifest(self):
        if self.coef is None: raise ValueError("Not trained")
        return {"kind":"SANAD_REAL_RF_SHADOW_HEAD_V1",
                "frozen_rc3_sha256":self.frozen_source_sha,
                "source_kind":"measured_RF",
                "labels_provenance":"digitally_injected_faults_not_physical_faults",
                "calibration":{"n":self.calibration.calibration_n,"alpha":self.calibration.alpha,
                               "cutoff":self.calibration.threshold},
                "coefficient":self.coef.tolist(),"intercept":self.intercept,
                "center":self.center.tolist(),"scale":self.scale.tolist(),
                "training_groups":list(self.training_groups),
                "calibration_groups":list(self.calibration_groups),
                "field_qualified":False}
