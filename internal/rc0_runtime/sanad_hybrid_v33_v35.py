"""SANAD Network — v33/v35 hybrid research fusion (NEW / NOT QUALIFIED).

Run from a directory containing the UNMODIFIED frozen v35 source modules.
Inputs are 7 arrays of aligned 16-dimensional covariance-derived RF proxies
(gain x8 + phase x8), not calibrated hardware sensor measurements.

Three disjoint output flags:
  sensitive_alert: original frozen v35 (legacy_union) final alert.
  conservative_alert: original frozen v33 confirmation final alert.
  hybrid_alert: conservative alert OR qualified temporal-only rescue.

The historical 95.14% recall (v35) and 1.69% FPR (v33) were obtained from
DIFFERENT prospective cohorts. No hybrid accuracy/FPR claim is made.
No RF actuation, radio control or automatic calibration is implemented.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from sanad_v29 import SANADNetworkV29
from sanad_v33 import (
    ElementEvidence, EvidenceFusionV33, FusionConfigV33,
    N_ARRAYS, _modal_nonzero, _spatial_feature,
)
from sanad_v35 import EvidenceFusionV35, FusionConfigV35
from temporal_v35 import FrozenTemporalEvidenceV35


@dataclass(frozen=True)
class HybridConfig:
    # New, unqualified proposal. These are NOT historical v33 or v35 thresholds.
    temporal_raw_hits: int = 1
    require_strict_spatial: bool = True
    require_matching_raw_identity: bool = True

    def __post_init__(self):
        if not 1 <= self.temporal_raw_hits <= 12:
            raise ValueError('temporal_raw_hits must be between 1 and 12')


@dataclass(frozen=True)
class HybridDecision:
    array_id: int
    sensitive_alert: bool     # v35 frozen decision, used for research review
    conservative_alert: bool  # v33 frozen decision
    hybrid_alert: bool        # NEW, not performance-validated
    temporal_promoted: bool
    review_required: bool
    antenna_1based: int | None
    status: str
    evidence_valid: bool
    action: str = 'NO_HARDWARE_ACTUATION'


def load_frozen_v35_config(source_dir: Path | str | None = None) -> FusionConfigV35:
    folder = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    path = folder / 'config_v35.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen v35 configuration: {path}')
    values = json.loads(path.read_text(encoding='utf-8'))
    if values.get('base_policy') != 'legacy_union':
        raise ValueError('This hybrid requires the actual 95.14%-recall v35 legacy_union configuration')
    return FusionConfigV35(**values)


class HybridFusionV33V35:
    """One causal update for both untouched frozen fusion models plus new gate.

    Pass original v11 alerts, v11 antenna names, 12-report fast spatial features,
    15-report strict spatial features, and v35 frozen-temporal features.
    All are per-report lists of length 7; ElementEvidence = score, dominance, id.
    """

    def __init__(self, config: HybridConfig | None = None,
                 *, frozen_source_dir: Path | str | None = None):
        self.config = config or HybridConfig()
        self.v33 = EvidenceFusionV33(FusionConfigV33())
        self.v35 = EvidenceFusionV35(load_frozen_v35_config(frozen_source_dir))

    def step(self, raw_v11, raw_named, fast_evidence, strict_evidence,
             temporal_evidence, valid=None) -> list[HybridDecision]:
        values = (raw_v11, raw_named, fast_evidence, strict_evidence, temporal_evidence)
        if any(len(v) != N_ARRAYS for v in values):
            raise ValueError('Every input must contain 7 arrays')
        if valid is None:
            valid = [True] * N_ARRAYS
        if len(valid) != N_ARRAYS or any(v not in (True, False, 0, 1) for v in valid):
            raise ValueError('valid must contain 7 boolean indicators')

        # Malformed/invalid arrays are never promoted. Feed identical, scrubbed
        # evidence to both ORIGINAL frozen models to prevent stale observations.
        flags = [bool(v) for v in valid]
        raw = [bool(raw_v11[a]) if flags[a] else False for a in range(N_ARRAYS)]
        named = [int(raw_named[a] or 0) if flags[a] else 0 for a in range(N_ARRAYS)]
        fast = [fast_evidence[a] if flags[a] else None for a in range(N_ARRAYS)]
        strict = [strict_evidence[a] if flags[a] else None for a in range(N_ARRAYS)]
        temporal = [temporal_evidence[a] if flags[a] else None for a in range(N_ARRAYS)]

        conservative = self.v33.step(raw, named, fast, strict, flags)
        sensitive = self.v35.step(raw, named, fast, strict, temporal, flags)
        decisions = []
        for a, (d33, d35) in enumerate(zip(conservative, sensitive)):
            # Only new v35 TEMPORAL rescue can be promoted. v34-only alerts
            # remain advisory unless independently confirmed by v33.
            temporal_candidate = bool(
                d35.final_alert and d35.delta_persistent and
                d35.antenna_1based is not None and
                d35.source in ('temporal_spatial_support', 'temporal_strong_solo') and
                not d35.cohort_ambiguous
            )
            raw_hist = list(self.v33.raw[a])[-12:]
            raw_names = list(self.v33.names[a])[-12:]
            raw_count = sum(raw_hist)
            raw_element, _ = _modal_nonzero(raw_names)
            strict_element, _ = _modal_nonzero(self.v33.strict[a])
            same_strict = bool(
                d33.strict_v32 and strict_element == d35.antenna_1based
            )
            same_raw = bool(raw_element == d35.antenna_1based)
            temporal_promoted = bool(
                flags[a] and temporal_candidate and
                raw_count >= self.config.temporal_raw_hits and
                (not self.config.require_strict_spatial or same_strict) and
                (not self.config.require_matching_raw_identity or same_raw)
            )
            conservative_alert = bool(flags[a] and d33.final_alert)
            sensitive_alert = bool(flags[a] and d35.final_alert)
            hybrid_alert = bool(conservative_alert or temporal_promoted)
            review = bool(flags[a] and (d33.review_candidate or d35.review_candidate or
                                        sensitive_alert) and not hybrid_alert)
            if conservative_alert:
                status, element = 'V33_CONFIRMED_RESEARCH_ALERT', d33.antenna_1based
            elif temporal_promoted:
                status, element = 'HYBRID_TEMPORAL_RESEARCH_ALERT', d35.antenna_1based
            elif review:
                status = 'REVIEW_ONLY'
                element = d35.antenna_1based or d33.antenna_1based
            else:
                status = 'MONITORING' if flags[a] else 'INSUFFICIENT_MEASUREMENTS'
                element = None
            decisions.append(HybridDecision(
                array_id=a, sensitive_alert=sensitive_alert,
                conservative_alert=conservative_alert, hybrid_alert=hybrid_alert,
                temporal_promoted=temporal_promoted, review_required=review,
                antenna_1based=element, status=status, evidence_valid=flags[a]
            ))
        return decisions


class SANADHybridNetwork:
    """End-to-end research adapter for the original calibrated 16-vector input.

    Reuses SANADNetworkV29 (frozen v11+v29 processing) ONCE, then feeds the
    two frozen decision models and the frozen v35 temporal evidence path.
    """

    def __init__(self, *, cal_dir: str | Path | None = None,
                 hybrid_config: HybridConfig | None = None,
                 frozen_source_dir: str | Path | None = None):
        self.fusion = HybridFusionV33V35(hybrid_config,
                                        frozen_source_dir=frozen_source_dir)
        c = self.fusion.v35.config
        self.source = SANADNetworkV29(cal_dir=cal_dir,
                                      recent=max(c.fast_window, c.strict_window))
        self.temporal = FrozenTemporalEvidenceV35()

    def update(self, vectors: Mapping[int, np.ndarray]) -> list[HybridDecision]:
        vectors = dict(vectors)
        for a, v in vectors.items():
            x = np.asarray(v, dtype=float)
            if type(a) is not int or not 0 <= a < N_ARRAYS or x.shape != (16,) or not np.isfinite(x).all():
                raise ValueError('Expected 0..6 array IDs and finite 16-vector gain/phase proxies')
        parent = self.source.update(vectors)
        delta = self.temporal.step(vectors)
        c = self.fusion.v35.config
        fast, strict = [], []
        for a, d in enumerate(parent):
            if not d.valid:
                fast.append(None)
                strict.append(None)
            else:
                w = list(self.source.window[a])
                fast.append(_spatial_feature(w[-c.fast_window:]))
                strict.append(_spatial_feature(w[-c.strict_window:]))
        valid = [bool(parent[a].valid or delta[a] is not None) for a in range(N_ARRAYS)]
        return self.fusion.step([d.confirmed_v11 for d in parent],
                                [d.antenna_1based or 0 for d in parent],
                                fast, strict, delta, valid)


def replay_cached_trace(path: str | Path, *, stream='healthy') -> dict:
    """Retrospective only: replay frozen v35 .npz evidence, never tune on it.

    Generates candidate/confirmed/hybrid outcomes on the SAME cached cases.
    This function does not read fault labels during inference, only scoring.
    """
    if stream not in ('healthy', 'injected'):
        raise ValueError('stream must be healthy or injected')
    from v35_replay import validate_v35_trace
    with np.load(path, allow_pickle=False) as z:
        trace = {key: z[key] for key in z.files}
    validate_v35_trace(trace, stream)
    raw = trace[f'{stream}_v11']
    named = trace[f'{stream}_v11_named']
    feats = trace[f'{stream}_feats']
    delta = trace[f'{stream}_delta']
    engine = HybridFusionV33V35()
    flags = {key: np.zeros((300, N_ARRAYS), dtype=bool) for key in
             ('sensitive_alert', 'conservative_alert', 'hybrid_alert', 'temporal_promoted')}

    def evidence(values):
        return (ElementEvidence(float(values[0]), float(values[1]), int(values[2]))
                if np.isfinite(values).all() else None)

    for t in range(300):
        fast = [evidence(feats[0, t, a]) for a in range(N_ARRAYS)]
        strict = [evidence(feats[1, t, a]) for a in range(N_ARRAYS)]
        temporal = [evidence(delta[t, a]) for a in range(N_ARRAYS)]
        valid = [bool(np.isfinite(feats[:, t, a, 0]).any() or np.isfinite(delta[t, a, 0]))
                 for a in range(N_ARRAYS)]
        row = engine.step(raw[t].tolist(), named[t].tolist(), fast, strict, temporal, valid)
        for a, decision in enumerate(row):
            for name in flags:
                flags[name][t, a] = getattr(decision, name)

    # Evaluation labels are used ONLY after inference and outside the engine.
    expected = trace[f'{stream}_expected']
    metrics = {}
    for key, values in flags.items():
        fired = values[260:300].any(axis=0)
        if stream == 'healthy':
            metrics[key] = {'false_alarms': int(fired.sum()), 'healthy_arrays': N_ARRAYS}
        else:
            fault = np.asarray(expected) > 0
            metrics[key] = {'true_positives': int(np.sum(fired & fault)),
                            'faulty_arrays': int(fault.sum()),
                            'unaffected_alerts': int(np.sum(fired & ~fault))}
    return {'stream': stream, 'metrics': metrics, 'status': 'RETROSPECTIVE_ONLY_NOT_QUALIFIED'}


def smoke_test():
    """Algorithmic fixture, NOT an RF calibration performance benchmark."""
    c = HybridFusionV33V35()
    none = [None] * N_ARRAYS
    strong = ElementEvidence(10.0, 3.0, 4)
    for _ in range(16):
        row = c.step([False]*7, [0]*7, none, none, [strong] + [None]*6)
    assert row[0].sensitive_alert and not row[0].conservative_alert
    assert not row[0].hybrid_alert and row[0].review_required
    assert all(not d.hybrid_alert for d in row[1:])

    c = HybridFusionV33V35()
    for _ in range(12):
        row = c.step([True]+[False]*6, [3]+[0]*6, none, none, none)
    assert row[0].conservative_alert and row[0].hybrid_alert
    assert not any(d.hybrid_alert for d in row[1:])
    invalid = c.step([True]*7, [3]*7, none, none, none, [False]*7)
    assert not any(d.hybrid_alert for d in invalid)
    return 'PASS (synthetic decision-rule smoke, not independent RF qualification)'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--trace', type=Path, help='v35 cached evidence .npz file')
    ap.add_argument('--stream', choices=['healthy', 'injected'], default='healthy')
    args = ap.parse_args()
    if args.trace is not None:
        print(json.dumps(replay_cached_trace(args.trace, stream=args.stream), indent=2))
    else:
        print(smoke_test())
