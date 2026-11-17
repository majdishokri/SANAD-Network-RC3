# Traceability: RC3 and Shukri 2016 thesis -> Adaptive v0.1

| Frozen RC3 source (read-only) | Exact mechanism retained as a concept | Adaptive implementation | Not yet equivalent |
|---|---|---|---|
| `internal/rc0_runtime/glr_compare.py` | calibrated 8-gain / 8-phase array proxy | `RFMeasurement` accepts M gains and optionally M phases | original Sionna covariance-to-proxy reconstruction |
| `internal/rc0_runtime/v29_research.py` | smooth spatial movement vs localized sparse deviation | `geometry.decompose()` with position-dependent basis and masks | historic thresholds / v29 feature definition |
| `internal/rc0_runtime/temporal_v35.py` | frozen healthy commissioning and temporal drift | `SANADAdaptiveNetwork` baseline and causal windows | 120-period / 300-period v35 contract |
| `internal/rc0_runtime/detect_v11.py` | compare suspicious element to rest; distinguish common movement | explicit LOCAL vs COMMON_MODE_REVIEW | anchored multi-array external reference / Wasserstein |
| `rc3.py` | RC0, 693+15 learned feature cascade | remains frozen on main, not imported or modified | model accuracy / 0.50 threshold |
| `monitor_v8.py` | matrix-based calibration, fixed-point numerical ideas, coherence observability | optional `covariance_to_proxy()` with coherence mask | dynamic UMa propagation / Toeplitz SPD fit |

## Thesis relation (motivation, not blanket theorem transfer)

- Sec. 2.3: convex metric spaces and distinction between metric and physical
  RF assumptions. This prototype currently optimizes in Euclidean engineering
  coordinates, **not** an assumed Poincare manifold.
- Sec. 3.2.2: G-monotone nonexpansive maps; not invoked as a guarantee because
  an RF neighbor graph is **not automatically** the order graph required by the
  theorem. `neighbors` are advisory context only.
- Sec. 3.3 and 4.2: best-proximity and projection motivate explicitly measuring
  inconsistency, rather than forcing incompatible evidence into a healthy state.
- Sec. 5.1: damped Mann / Ishikawa-style updates motivate numerical stabilization;
  the solver objective and mask handling here must be justified independently.

## Key engineering invariants

1. No ground-truth fields in `update` or `covariance_to_proxy`.
2. Never replace an unobserved physical phase with a valid zero.
3. The spatial basis uses coordinates, not element *row order*.
4. No deterministic claim is made from heuristic threshold without holdout.
5. No training on the RC3 untouched prospective seed labels 3000..3143.
6. No automatic hardware action, gain change, beam steering or transmit command.
7. Never execute downloaded `joblib`/pickle models from untrusted sources.
