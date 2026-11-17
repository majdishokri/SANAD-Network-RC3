"""Geometry-aware sparse fault decomposition (research prototype).

The objective for each physical channel is
  min_{b,s} 1/2 ||M (y - H b - s)||_2^2 + lam ||M s||_1 + ridge/2 ||b||_2^2
where H is a spatially smooth nuisance subspace built from physical coordinates.
No convergence guarantee from Shukri's fixed-point theorems is assumed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Decomposition:
    smooth: np.ndarray          # [M, C] in normalized engineering units
    sparse: np.ndarray          # [M, C] in normalized engineering units
    valid: np.ndarray           # [M, C] valid measurement mask
    observable: np.ndarray      # [C] enough degrees of freedom to localize
    objective: float
    iterations: int
    converged: bool
    smooth_rank: int


def _basis(positions: np.ndarray) -> np.ndarray:
    """Column space of [1, centered geometry] with >1 residual DOF where possible.

    Rows are elements; ordering is irrelevant. For M <= 3, fit only intercept to
    avoid removing all spatial signal. A smooth linear gradient is allowed when
    at least four elements are present.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] not in (1, 2, 3):
        raise ValueError("positions must have shape [M,1..3]")
    if not np.isfinite(p).all():
        raise ValueError("positions must be finite")
    n = len(p)
    q = p - p.mean(axis=0, keepdims=True)
    scale = np.maximum(np.std(q, axis=0), 1e-9)
    q = q / scale
    # SVD identifies effective geometric axes; do not depend on antenna order.
    if n >= 4:
        u, singular, _ = np.linalg.svd(q, full_matrices=False)
        rank = int(np.sum(singular > (singular[0] if len(singular) else 0) * 1e-8))
        extra = min(rank, n - 3)
        basis = u[:, :extra] if extra else np.zeros((n, 0))
    else:
        basis = np.zeros((n, 0))
    return np.column_stack([np.ones(n), basis])


def _soft(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def decompose(
    values: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    lam: float = 0.18,
    damping: float = 0.65,
    ridge: float = 1e-5,
    max_iter: int = 120,
    tolerance: float = 1e-8,
) -> Decomposition:
    """Return smooth environmental movement and sparse antenna-element evidence.

    ``values`` is [M,C] normalized gain/phase DIFFERENCE from trusted baseline.
    Missing values are represented using ``valid`` rather than replaced by zero.
    Sparse evidence is not automatically a calibrated probability or field alarm.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 2 or y.shape[0] < 1:
        raise ValueError("values must be [M,C]")
    n, c = y.shape
    if not np.asarray(positions).shape[0] == n:
        raise ValueError("position count differs from element count")
    m = np.isfinite(y) if valid is None else np.asarray(valid, dtype=bool)
    if m.shape != y.shape or np.any(m & ~np.isfinite(y)):
        raise ValueError("bad mask or nonfinite valid values")
    if not (lam > 0 and 0 < damping <= 1 and ridge > 0 and max_iter > 0 and tolerance > 0):
        raise ValueError("invalid solver settings")
    h = _basis(np.asarray(positions))
    smooth = np.full_like(y, np.nan)
    sparse = np.full_like(y, np.nan)
    observable = np.zeros(c, dtype=bool)
    obj = 0.0
    maximum_iters = 0
    all_converged = True
    for j in range(c):
        indices = np.flatnonzero(m[:, j]); nn = len(indices)
        if not nn:
            continue
        hj = h[indices]
        yj = y[indices, j]
        rank = int(np.linalg.matrix_rank(hj))
        # At least two degrees of freedom beyond the nuisance model.
        observable[j] = nn >= rank + 2
        gram = hj.T @ hj + ridge * np.eye(hj.shape[1])
        s = np.zeros(nn, dtype=float)
        b = np.linalg.solve(gram, hj.T @ yj)
        converged = False
        count = 0
        for k in range(max_iter):
            b_new = np.linalg.solve(gram, hj.T @ (yj - s))
            target_s = _soft(yj - hj @ b_new, lam)
            s_new = (1.0 - damping) * s + damping * target_s
            count = k + 1
            if max(np.max(np.abs(s_new - s)), np.max(np.abs(b_new - b))) < tolerance:
                s = s_new
                b = b_new
                converged = True
                break
            s, b = s_new, b_new
        # Ensure decomposition corresponds to final sparse coefficients.
        b = np.linalg.solve(gram, hj.T @ (yj - s))
        sv = h @ b
        smooth[:, j] = sv
        sparse[indices, j] = s
        res = yj - hj @ b - s
        obj += float(0.5 * (res @ res) + lam * np.abs(s).sum() + 0.5 * ridge * (b @ b))
        maximum_iters = max(maximum_iters, count)
        all_converged = all_converged and converged
    return Decomposition(smooth=smooth, sparse=sparse, valid=m,
                         observable=observable, objective=obj,
                         iterations=maximum_iters, converged=all_converged,
                         smooth_rank=h.shape[1])
