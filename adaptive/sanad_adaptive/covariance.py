"""Optional reference-covariance -> calibrated gain/phase proxy adapter.

This adapter assumes the channel covariance *reference itself* is stable enough
that C_new approximately equals diag(d) C_reference diag(d)^H. It MUST NOT be
presented as generic raw-RF extraction under arbitrary changing propagation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CovarianceProxy:
    gain_db: np.ndarray  # relative to median; common gain is unobservable
    phase_deg: np.ndarray  # relative phase on largest coherent connected component
    phase_observed: np.ndarray
    coherence_edges: int
    reference_reconstruction_error: float


def _check_cov(c: np.ndarray, name: str) -> np.ndarray:
    c = np.asarray(c, dtype=np.complex128)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix >=2")
    if not np.isfinite(c).all() or not np.allclose(c, c.conj().T, rtol=1e-6, atol=1e-8):
        raise ValueError(f"{name} must be finite Hermitian")
    diag = c.diagonal().real
    if np.any(diag <= 0):
        raise ValueError(f"{name} must have positive diagonal")
    # PSD tolerance scaled by the largest eigenvalue.
    eig = np.linalg.eigvalsh(c)
    if eig[0] < -1e-7 * max(eig[-1], 1.0):
        raise ValueError(f"{name} is not positive semidefinite")
    return c


def covariance_to_proxy(covariance: np.ndarray, reference: np.ndarray,
                        coherence_min: float = 0.10) -> CovarianceProxy:
    """Estimate differential element gain and phase from a trustworthy reference.

    Eigenvector angular synchronization recovers phase modulo one common phase
    for the largest connected coherence component. All other phases remain NaN.
    """
    c = _check_cov(covariance, "covariance")
    ref = _check_cov(reference, "reference")
    if c.shape != ref.shape or not (0 < coherence_min < 1):
        raise ValueError("shape/coherence mismatch")
    n = len(c)
    dc, dr = np.real(np.diag(c)), np.real(np.diag(ref))
    g = 10.0 * np.log10(dc / dr)
    g -= np.median(g)  # gauge-fixed. Common increase can be traffic, not a fault.
    coh_ref = np.abs(ref) / np.sqrt(np.outer(dr, dr))
    coh_now = np.abs(c) / np.sqrt(np.outer(dc, dc))
    adj = (np.minimum(coh_ref, coh_now) >= coherence_min)
    np.fill_diagonal(adj, False)
    components: list[list[int]] = []
    seen: set[int] = set()
    for node in range(n):
        if node in seen:
            continue
        stack = [node]; group = []
        seen.add(node)
        while stack:
            k = stack.pop(); group.append(k)
            for v in np.flatnonzero(adj[k]):
                if int(v) not in seen:
                    seen.add(int(v)); stack.append(int(v))
        components.append(group)
    largest = max(components, key=len)
    phases = np.full(n, np.nan, dtype=float)
    observed = np.zeros(n, dtype=bool)
    if len(largest) >= 2:
        ids = np.asarray(sorted(largest), dtype=int)
        weights = np.minimum(coh_ref[np.ix_(ids, ids)], coh_now[np.ix_(ids, ids)])
        mask = adj[np.ix_(ids, ids)]
        angles = np.angle(c[np.ix_(ids, ids)] * ref[np.ix_(ids, ids)].conj())
        h = weights * mask * np.exp(1j * angles)
        h = 0.5 * (h + h.conj().T)
        _, vecs = np.linalg.eigh(h)
        phi = np.angle(vecs[:, -1])
        # Define a gauge-invariant relative phase; global phase is not identifiable.
        center = np.angle(np.sum(np.exp(1j * phi)))
        phases[ids] = np.rad2deg(np.angle(np.exp(1j * (phi - center))))
        observed[ids] = True
    # A diagnostic only: changing propagation breaks reference-cov assumption.
    dj = 10 ** (g / 20.0) * np.exp(1j * np.deg2rad(np.nan_to_num(phases)))
    approximate = np.outer(dj, dj.conj()) * ref
    off = observed[:, None] & observed[None, :]
    err = np.linalg.norm((c - approximate)[off]) / max(np.linalg.norm(c[off]), 1e-12) if off.any() else float("nan")
    return CovarianceProxy(g, phases, observed, int(np.triu(adj, 1).sum()), float(err))
