"""Fixed-width, dimension-invariant features for a FUTURE supervised shared head.

A single trained head could operate over heterogeneous array sizes because its
per-element input width is constant. This module ships NO model weights and
makes NO accuracy claim. Train on labeled, multi-hardware data before use.
"""
from __future__ import annotations
from typing import Mapping
import numpy as np

FEATURE_NAMES = (
    'log_abs_gain_db', 'log_abs_phase_deg', 'phase_observable',
    'log_raw_anomaly_score', 'element_rank_percentile',
    'array_max_score', 'array_median_score', 'log_array_elements',
    'log_network_arrays', 'log_channel_residual',
    'array_channel_shift', 'network_channel_shift_fraction',
)

def per_element_features(result: Mapping) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Always [sum_i M_i,12], independent of N, M_i, and input ordering.

    Returns stable (array_id, element_index) row identities. No labels used.
    """
    decisions = result['decisions']
    n = len(decisions)
    if not n:
        raise ValueError('No arrays')
    global_drift = float(result['channel_shift_fraction'])
    arrays = []
    identities = []
    for name, dec in sorted(decisions.items()):
        m = int(dec['elements'])
        s = np.asarray(dec['score'], dtype=float)
        g = np.asarray(dec['gain_db'], dtype=float)
        p = np.asarray([v if v is not None else 0. for v in dec['phase_deg']], dtype=float)
        if s.shape != (m,) or g.shape != (m,) or p.shape != (m,):
            raise ValueError('Malformed per-antenna output')
        ranks = np.argsort(np.argsort(s)) / max(m-1, 1)
        features = np.column_stack([
            np.log1p(np.abs(g)), np.log1p(np.abs(p)),
            np.full(m, float(dec['phase_observable'])), np.log1p(s), ranks,
            np.full(m, np.max(s)), np.full(m, np.median(s)),
            np.full(m, np.log1p(m)), np.full(m, np.log1p(n)),
            np.full(m, np.log1p(dec['channel_residual'] or 0.)),
            np.full(m, float(dec['channel_shift'])), np.full(m, global_drift),
        ])
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite shared-head inputs')
        arrays.append(features.astype(np.float32))
        identities.extend((name, i) for i in range(m))
    return np.concatenate(arrays), identities


class ExperimentalSharedHead:
    """Optional supervised head. No pretrained weights, no RC3 model reuse.

    The caller must supply training labels and hold-out validation grouped by
    site, session and hardware configuration. This head should never overrule a
    CHANNEL_SHIFT_REVIEW abstention without independent evidence.
    """
    def __init__(self):
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f'Expected {len(FEATURE_NAMES)}-feature matrix')
        if len(X) != len(y) or len(np.unique(y)) != 2:
            raise ValueError('Both labeled classes are required')
        self._model = HistGradientBoostingClassifier(max_iter=90, max_leaf_nodes=15,
                                                       l2_regularization=2., random_state=3101).fit(X,y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('Shared head is UNTRAINED. Supply labels and fit first.')
        return self._model.predict_proba(X)[:, 1]
