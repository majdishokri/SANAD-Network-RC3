"""
DETECT v11 - has an antenna array left its normal state, and in what way?
=======================================================================

Stand-alone (numpy only). Input: one calibration vector per array per period - 8 gains in dB then
8 phases in degrees - as produced by locate.py. Nothing upstream is touched.

How it decides
--------------
  1. OWN BASELINE.  Each array learns a frozen absolute baseline over a fault-free learning window
     (median after rejecting outlier rows).            z_abs = vector - baseline
  2. REFERENCE.     The cluster's common movement is removed, but the reference is ANCHORED:
     it is the median of the arrays that are still inside their own absolute acceptance region
     (when at least `min_group` of them are); otherwise the plain median.
     v9.1 used the plain median, which assumes that fewer than half of the arrays are faulty:
     with z_abs = [0,0,0,D,D,D,D] the median is D, the faulty arrays vanish and the three healthy
     ones are flagged. Anchoring keeps the reference on the arrays that did not move.
                                                         z_rel = z_abs - reference
  3. SCORE.         Per antenna, |gain|/0.5 dB + |phase|/5 deg of z_rel. ON = the worst antenna,
     OFF = the second worst (what a one-antenna fault cannot explain).
  4. FOUR OUTPUTS, each confirmed on 8 of the last 12 scored periods:
       alarm              ON above its threshold and OFF inside its healthy range:
                          ONE antenna left its place - a localised fault. Robust to drift.
       distributed_alarm  ON above its threshold and OFF outside its healthy range:
                          the array moved as a whole (drift, or a fault on several antennas).
       w_alarm            Wasserstein layer: the distribution of the last 48 periods moved away
                          from the distribution of the learning window. Separate layer; it never
                          changes the two alarms above.
       majority_alarm     Majority layer: every split of the cluster into two groups is tested on
                          windows of 8 to 128 periods; reported when the group that moved holds at
                          least 4 arrays and the difference sits on one antenna. Helps at 4/7 and
                          5/7 (about 90%); at 6/7 it reached only 17-44% in validation. Separate
                          layer; it also fires when the whole cluster drifts unevenly.

Every threshold is a pooled 99% quantile of HEALTHY data (calibrate()); none is tuned on a run that
contains a fault. The fixed constants are physical: 0.5 dB, 5 deg, 8 of 12.

Measured (DETECTOR_V11_REPORT.md, FINAL_RESULTS.md)
---------------------------------------------------
Confirmation on 25 simulation seeds never used for any choice (150 faulty + 375 healthy arrays,
analysis registered before the data existed), 10-hour decision window:
                              detection   false alarms   false alarms under per-array drift
   v9.1 (plain median)          76.7%        0.8%              64.3%
   v11  alarm, no OFF limit     76.7%        0.8%              62.9%
   v11  alarm (localised)       70.0%        0.0%               3.5%
   v11  w_alarm (Wasserstein)   80.7%        0.0%              65.9%
Same fault shared by k of 7 arrays: v9.1 fails from 4/7. On 12 unseen seeds with the fault injected in the
raw reports (v14_redteam.log) `majority_alarm` detects 91.7% at 4/7, 90.0% at 5/7 and only 16.7% at 6/7
(44.4% when the arrays' fault sizes differ). The 100% first measured came from faults written directly
into the vectors and did not survive this test.

Known limits
------------
  * Slow faults (ramps of 400 periods and more) are not detected by any layer.
  * A fault shared by 6 of 7 arrays is NOT reliably detected (17-44% by `majority_alarm`); a fault
    shared by ALL arrays cannot be told from a healthy common-mode shift without an external reference.
  * `alarm` does not fire for a fault on several antennas of one array; `distributed_alarm` and
    `w_alarm` do, and those two also fire on drift.
  * The learning window must be fault-free; `learn_shift` flags a window that was already moving.
  * All numbers come from NVIDIA Sionna (3GPP TR 38.901) simulation, not from real towers.

Use
---
    cal = calibrate(healthy_runs, w_window=48, majority=True)   # healthy data only
    det = Detector11(n, "anchored", cal["alarm_threshold"], abs_limit=cal["abs_limit"], min_group=2,
                     off_limit=cal["off_limit"], w_window=48, w_limit=cal["w_limit"],
                     majority=True, maj_thresholds=cal["maj_thresholds"], maj_final=cal["maj_final"])
    out = det.update({array_index: vector})                 # every period
or  det = default_detector(n)                               # the values calibrated on the benchmark
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

WARMUP, LEARN, CM_MIN_TOWERS, CONFIRM = 60, 60, 5, (8, 12)
GAIN_UNIT_DB, PHASE_UNIT_DEG = 0.5, 5.0
ALARM_THRESHOLD = 1.55            # v9.1's pooled 99% value; re-fit with calibrate() on your fleet
ANCHOR_LIMIT = 0.60               # healthy common-mode envelope (diagnostic label only; measured 0.49-0.63)
ABS_LIMIT = 1.54                  # absolute acceptance region of one array (inlier test of the reference)
OFF_LIMIT = 1.03                  # healthy range of the second-worst antenna (localised vs distributed)
W_WINDOW, W_LIMIT = 48, 1.12      # Wasserstein layer: periods compared, and its threshold
MIN_GROUP = 2                     # fewest arrays still at their baseline needed to anchor the reference
MAJ_THRESHOLDS = {8: 1.91, 16: 1.81, 32: 1.66, 64: 1.56, 128: 1.33}   # majority layer, per window
MAJ_FINAL = 1.016                 # majority layer, on the best normalised split


def unit_distance(d: np.ndarray) -> np.ndarray:
    """Per-antenna distance in fixed physical units: |gain|/0.5 dB + |phase|/5 deg."""
    d = np.abs(np.asarray(d, float)); m = d.shape[-1] // 2
    return d[..., :m] / GAIN_UNIT_DB + d[..., m:] / PHASE_UNIT_DEG


def size(d: np.ndarray) -> float:
    """Worst-antenna distance of a 16-vector (the detector's score)."""
    return float(unit_distance(d).max())


# ----------------------------------------------------------------------- baseline
class Baseline:
    """Frozen absolute baseline of one array: the median of its raw vectors over the learning
    window after rejecting rows that are outliers by the modified z-score (MAD).

    Limits, stated plainly: row rejection removes isolated bad readings; it CANNOT remove a
    persistent fault that fills more than half of the window (the median then follows it).
    `learn_shift` (distance between the medians of the two halves of the window) is exposed so a
    window that was already moving can be recognised; it is a diagnostic, it never redefines
    "normal"."""

    def __init__(self, history: Sequence[np.ndarray]):
        H = np.asarray(history, float)
        crude = np.median(H, 0)
        mad = np.maximum(np.median(np.abs(H - crude), 0), 1e-6)
        keep = np.all(0.6745 * np.abs(H - crude) / mad < 3.5, axis=1)
        self.center = np.median(H[keep] if keep.sum() > 0.5 * len(H) else H, 0)
        h = len(H) // 2
        self.learn_shift = size(np.median(H[:h], 0) - np.median(H[h:], 0)) if h >= 5 else 0.0
        self.n_learn = len(H)

    def z(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v, float) - self.center


# ---------------------------------------------------------------------- reference
@dataclass
class Reference:
    vector: np.ndarray
    kind: str            # 'median' | 'anchored' | 'none'
    group_size: int
    common_score: float  # size of the reference vector (how far the cluster moved as a whole)


def choose_reference_inliers(Z: np.ndarray, abs_limit: float, min_group: int = 1) -> Reference:
    """v11 rule. Z: (n_reporting, 16) absolute residuals of the arrays reporting this period.

    Inliers = arrays still inside their OWN absolute acceptance region (size <= abs_limit, the
    pooled 99% quantile of healthy absolute sizes). The reference is the median of the inliers.
    Nothing moved -> every array is an inlier -> the plain median (v9.1 behaviour).  A fault
    shared by most arrays -> the faulty arrays fall outside, the healthy ones stay inside, and the
    reference stays at the healthy arrays instead of following the majority.  Every array moved
    together by more than a healthy array may (cluster drift, or a fault on all arrays - the same
    vector, so they cannot be told apart) -> fewer than min_group inliers -> the plain median."""
    sizes = np.array([size(z) for z in Z]); inl = np.flatnonzero(sizes <= abs_limit)
    if min_group <= len(inl) < len(Z):
        c = np.median(Z[inl], 0); return Reference(c, "anchored", len(inl), size(c))
    med = np.median(Z, 0); return Reference(med, "median", len(Z), size(med))


def choose_reference(Z: np.ndarray, anchor_limit: float, group_radius: float,
                     min_group: int = 1) -> Reference:
    """FIRST DRAFT, REJECTED (kept for the record, see DETECTOR_V11_REPORT.md): groups of
    mutually agreeing arrays whose centre sits inside the healthy common-mode envelope. It failed
    because a group of two or three arrays has a noisier centre than the median of seven, so the
    envelope calibrated on medians rejected genuinely healthy groups. Z: (n_reporting, 16)."""
    med = np.median(Z, 0); s_med = size(med)
    if s_med <= anchor_limit or len(Z) < 2:
        return Reference(med, "median", len(Z), s_med)
    # groups of mutually agreeing arrays, each seeded at one array; keep those whose centre is
    # inside the healthy envelope
    best: Optional[Tuple[int, float, np.ndarray]] = None
    for i in range(len(Z)):
        members = [j for j in range(len(Z)) if size(Z[j] - Z[i]) <= group_radius]
        centre = np.median(Z[members], 0); sc = size(centre)
        if sc <= anchor_limit and len(members) >= min_group:
            if best is None or (len(members), -sc) > (best[0], -best[1]):
                best = (len(members), sc, centre)
    if best is None:
        return Reference(med, "median", len(Z), s_med)
    return Reference(best[2], "anchored", best[0], best[1])


# ------------------------------------------------------------------ Wasserstein layer
class WassersteinLayer:
    """A SEPARATE layer: has the DISTRIBUTION of an array's recent relative residuals moved away from the
    distribution it had while learning?  Per antenna, the 1-Wasserstein distance between the two samples
    (mean absolute difference of their quantiles), gain in units of 0.5 dB plus phase in units of 5 deg;
    the layer's score is the worst antenna.  It compares `window` recent periods with the learning window,
    so it integrates evidence over time instead of judging one period. Own threshold, own confirmation,
    own alarm (`w_alarm`); it never changes `alarm` or `distributed_alarm`."""
    Q = np.linspace(0.025, 0.975, 20)

    def __init__(self, window: int = 48):
        self.window = window
        self.ref_q: Optional[np.ndarray] = None
        self.recent: deque = deque(maxlen=window)

    def set_reference(self, sample: np.ndarray) -> None:
        if len(sample) >= 20:
            self.ref_q = np.quantile(np.asarray(sample, float), self.Q, axis=0)

    def update(self, r: np.ndarray) -> Optional[float]:
        self.recent.append(np.asarray(r, float))
        if self.ref_q is None or len(self.recent) < max(2, self.window // 2):
            return None
        w1 = np.abs(np.quantile(np.array(self.recent), self.Q, axis=0) - self.ref_q).mean(0)
        m = len(w1) // 2
        return float((w1[:m] / GAIN_UNIT_DB + w1[m:] / PHASE_UNIT_DEG).max())


# -------------------------------------------------------------------- majority layer
_PARTITIONS: Dict[int, np.ndarray] = {}


def _partitions(m: int) -> np.ndarray:
    """All splits of m arrays into two non-empty groups, each once (2^(m-1) - 1 rows of booleans)."""
    if m not in _PARTITIONS:
        _PARTITIONS[m] = np.array([[(b >> j) & 1 for j in range(m - 1)] + [0]
                                   for b in range(1, 2 ** (m - 1))], bool)[:, ::-1]
    return _PARTITIONS[m]


class MajorityLayer:
    """A SEPARATE layer for the one case a reference cannot handle: MOST of the cluster moved together.

    Every split of the reporting arrays into two groups is tested on several time scales:
        delta   = mean of group A - mean of group B   (window means of the absolute residuals)
        stat    = worst-antenna size of delta x sqrt(nA nB / (nA + nB))
    The best split over all windows is kept. It is reported only when (a) it exceeds the healthy
    thresholds, (b) the group that moved away from its baselines holds at least `min_side` arrays, and
    (c) the difference is localised on one antenna (second-worst antenna of delta within off_limit) -
    a drift moves every antenna, a shared one-antenna fault does not. Its alarm (`majority_alarm`) is
    separate; it never changes `alarm`, `distributed_alarm` or `w_alarm`."""
    WINDOWS = (8, 16, 32, 64, 128)

    def __init__(self, n_arrays: int, thresholds: Optional[Dict[int, float]] = None,
                 final: Optional[float] = None, off_limit: Optional[float] = None, min_side: int = 4,
                 min_arrays: int = 5):
        self.th, self.final, self.off_limit = thresholds, final, off_limit
        self.min_side, self.min_arrays = min_side, min_arrays
        self.buf: List[deque] = [deque(maxlen=max(self.WINDOWS)) for _ in range(n_arrays)]

    def update(self, Z: Dict[int, np.ndarray]) -> dict:
        for i, z in Z.items():
            self.buf[i].append(np.asarray(z, float))
        per_window: Dict[int, float] = {}; best = (-1.0, None)
        for W in self.WINDOWS:
            ids = [i for i in Z if len(self.buf[i]) >= max(1, W // 2)]
            if len(ids) < self.min_arrays:
                continue
            X = np.array([np.mean(list(self.buf[i])[-W:], 0) for i in ids])
            P = _partitions(len(ids)); nA = P.sum(1); nB = len(ids) - nA
            mA = (P @ X) / nA[:, None]; mB = ((~P) @ X) / nB[:, None]
            ud = unit_distance(mA - mB); stat = ud.max(1) * np.sqrt(nA * nB / (nA + nB))
            k = int(np.argmax(stat)); per_window[W] = float(stat[k])
            if self.th is not None:
                v = stat[k] / self.th[W]
                if v > best[0]:
                    a_far = size(mA[k]) >= size(mB[k]); mask = P[k] if a_far else ~P[k]
                    local = self.off_limit is None or float(np.sort(ud[k])[-2]) <= self.off_limit
                    best = (float(v), [ids[j] for j in np.flatnonzero(mask)] if local else [])
        flagged: List[int] = []
        if self.final is not None and best[1] is not None and best[0] > self.final and len(best[1]) >= self.min_side:
            flagged = best[1]
        return dict(per_window=per_window, stat=best[0] if best[1] is not None else None, flagged=flagged)


# ---------------------------------------------------------------------- detector
class Detector11:
    """mode: 'anchored' (v11: reference = median of the arrays inside their absolute acceptance
    region), 'median' (v9.1/V10 relative path), 'absolute' (no reference),
    'anchored_group' (rejected first draft)."""

    def __init__(self, n_arrays: int, mode: str = "anchored",
                 alarm_threshold: Optional[float] = ALARM_THRESHOLD,
                 anchor_limit: float = ANCHOR_LIMIT, group_radius: Optional[float] = None,
                 min_group: int = 1, abs_limit: Optional[float] = None, warmup: int = WARMUP, learn: int = LEARN,
                 confirm: Tuple[int, int] = CONFIRM, cm_min_towers: int = CM_MIN_TOWERS,
                 window: int = 1, off_limit: Optional[float] = None,
                 w_window: int = 0, w_limit: Optional[float] = None,
                 majority: bool = False, maj_thresholds: Optional[Dict[int, float]] = None,
                 maj_final: Optional[float] = None):
        assert mode in ("anchored", "median", "absolute", "anchored_group")
        self.n, self.mode = n_arrays, mode
        self.alarm_threshold, self.anchor_limit, self.min_group = alarm_threshold, anchor_limit, min_group
        self.abs_limit = abs_limit if abs_limit is not None else (alarm_threshold or ALARM_THRESHOLD)
        self.group_radius = group_radius if group_radius is not None else (alarm_threshold or ALARM_THRESHOLD)
        self.learn, self.confirm, self.cm_min = learn, confirm, cm_min_towers
        # ON / OFF split (best proximity to the fault set "one antenna moved", Pythagorean property):
        # ON = worst antenna, OFF = second worst = what one antenna cannot explain. With off_limit set,
        # `alarm` is raised only for a LOCALISED deviation (OFF within its healthy range); a deviation
        # spread over the array (drift, or a fault on several antennas) raises `distributed_alarm`.
        self.off_limit = off_limit
        # Wasserstein layer (off when w_window == 0); w_limit None -> score only, no alarm
        self.w_window, self.w_limit = int(w_window), w_limit
        self._w: List[Optional[WassersteinLayer]] = [WassersteinLayer(w_window) if w_window else None
                                                      for _ in range(n_arrays)]
        self._over_w: List[List[bool]] = [[] for _ in range(n_arrays)]
        self._hist_t: List[List[int]] = [[] for _ in range(n_arrays)]
        # majority layer (off unless majority=True); thresholds None -> statistics only
        self._maj: Optional[MajorityLayer] = (MajorityLayer(n_arrays, maj_thresholds, maj_final, off_limit)
                                              if majority else None)
        self._over_m: List[List[bool]] = [[] for _ in range(n_arrays)]
        self._over_d: List[List[bool]] = [[] for _ in range(n_arrays)]
        self.window = max(1, int(window))   # score the MEAN of the last `window` relative residuals
        self._win: List[deque] = [deque(maxlen=self.window) for _ in range(n_arrays)]
        self.period = 0
        self.baseline: List[Optional[Baseline]] = [None] * n_arrays
        self._hist: List[List[np.ndarray]] = [[] for _ in range(n_arrays)]
        self._over: List[List[bool]] = [[] for _ in range(n_arrays)]
        self._from = [warmup] * n_arrays

    # ---- one period
    def update(self, vecs: Dict[int, Sequence[float]]) -> dict:
        """Call every period (empty dict if nothing reported). Returns, per array (None when not
        scored): score, suspect (antenna), over, alarm; plus absolute_score, relative_score,
        and cluster-level reference_type, common_score, affected_fraction, cluster_state."""
        n = self.n
        out = dict(score=[None] * n, suspect=[None] * n, over=[False] * n, alarm=[False] * n,
                   absolute_score=[None] * n, relative_score=[None] * n,
                   off_score=[None] * n, kind=[None] * n, distributed_alarm=[False] * n,
                   w_score=[None] * n, w_alarm=[False] * n,
                   majority_alarm=[False] * n, majority_stat=None, majority_windows={},
                   confirmation=[0] * n, reference_type="none", common_score=None,
                   affected_fraction=None, cluster_state="not_scored",
                   scored=False, common_mode_removed=False)
        Z: Dict[int, np.ndarray] = {}
        for i, v in vecs.items():
            a = self._from[i]
            if self.period < a:
                continue
            if self.period < a + self.learn:
                self._hist[i].append(np.asarray(v, float)); self._hist_t[i].append(self.period); continue
            if self.baseline[i] is None:
                if len(self._hist[i]) < 20:
                    continue
                self.baseline[i] = Baseline(self._hist[i])
            Z[i] = self.baseline[i].z(v)
        need_ref = self.mode != "absolute"
        if Z and (not need_ref or len(Z) >= self.cm_min):
            ids = list(Z); M_ = np.array([Z[i] for i in ids])
            if need_ref:
                if self.mode == "anchored":
                    ref = choose_reference_inliers(M_, self.abs_limit, self.min_group)
                elif self.mode == "anchored_group":
                    ref = choose_reference(M_, self.anchor_limit, self.group_radius, self.min_group)
                else:
                    med = np.median(M_, 0); ref = Reference(med, "median", len(ids), size(med))
                c = ref.vector
                out.update(reference_type=ref.kind, common_score=ref.common_score,
                           common_mode_removed=True)
            else:
                c = np.zeros(M_.shape[1])
            R = M_ - c
            if self._maj is not None:
                mj = self._maj.update({i: M_[row] for row, i in enumerate(ids)})
                out["majority_stat"], out["majority_windows"] = mj["stat"], mj["per_window"]
                for i in ids:
                    bm = self._over_m[i]; bm.append(i in mj["flagged"]); del bm[:-self.confirm[1]]
                    out["majority_alarm"][i] = sum(bm) >= self.confirm[0]
            if self.w_window:
                self._set_w_references(ids)
            affected = 0
            for row, i in enumerate(ids):
                if self.window > 1:
                    self._win[i].append(R[row])
                    if len(self._win[i]) < max(1, self.window // 2):
                        continue                      # not enough recent periods yet: not scored
                    r_eff = np.mean(self._win[i], 0)
                else:
                    r_eff = R[row]
                t = unit_distance(r_eff); k = int(np.argmax(t)); s = float(t[k])
                ta = unit_distance(M_[row]); sa = float(ta.max())
                over = self.alarm_threshold is not None and s > self.alarm_threshold
                off = float(np.sort(t)[-2])
                local = self.off_limit is None or off <= self.off_limit
                affected += over
                out["score"][i], out["suspect"][i], out["over"][i] = s, k, over
                out["relative_score"][i], out["absolute_score"][i] = s, sa
                out["off_score"][i] = off
                out["kind"][i] = None if not over else ("localised" if local else "distributed")
                out["alarm"][i] = self._confirmed(i, over and local)
                bd = self._over_d[i]; bd.append(bool(over and not local)); del bd[:-self.confirm[1]]
                out["distributed_alarm"][i] = sum(bd) >= self.confirm[0]
                if self.w_window:
                    ws = self._w[i].update(R[row])
                    if ws is not None:
                        out["w_score"][i] = ws
                        bw = self._over_w[i]; bw.append(self.w_limit is not None and ws > self.w_limit)
                        del bw[:-self.confirm[1]]
                        out["w_alarm"][i] = sum(bw) >= self.confirm[0]
                out["confirmation"][i] = sum(self._over[i])
            out["affected_fraction"] = affected / len(ids)
            out["scored"] = True
            out["cluster_state"] = self._state(out, ids)
        self.period += 1
        return out

    def _set_w_references(self, ids: List[int]) -> None:
        """Reference sample of the Wasserstein layer: each array's relative residuals over ITS learning
        window (own baseline removed, then the cluster median of the arrays learning in the same period)."""
        todo = [i for i in ids if self._w[i].ref_q is None and self.baseline[i] is not None]
        if not todo:
            return
        known = [j for j in range(self.n) if self.baseline[j] is not None]
        at = {j: dict(zip(self._hist_t[j], self._hist[j])) for j in known}
        for i in todo:
            sample = []
            for t in self._hist_t[i]:
                zs = {j: self.baseline[j].z(at[j][t]) for j in known if t in at[j]}
                if len(zs) >= self.cm_min:
                    sample.append(zs[i] - np.median(np.array(list(zs.values())), 0))
            self._w[i].set_reference(np.array(sample)) if sample else None

    @staticmethod
    def _state(out: dict, ids: List[int]) -> str:
        cs, ref = out["common_score"], out["reference_type"]
        if ref == "anchored":
            return "majority_shift_reference_anchored"
        if cs is not None and cs > ANCHOR_LIMIT and out["affected_fraction"] == 0:
            return "common_mode_shift"
        return "normal" if out["affected_fraction"] == 0 else "arrays_deviate"

    def _confirmed(self, i: int, over: bool) -> bool:
        k, m = self.confirm
        b = self._over[i]; b.append(bool(over)); del b[:-m]
        return sum(b) >= k

    def relearn(self, i: int, settle: int = 20) -> None:
        self.baseline[i] = None; self._hist[i] = []; self._over[i] = []; self._over_d[i] = []; self._win[i].clear()
        self._hist_t[i] = []; self._over_w[i] = []; self._over_m[i] = []
        if self._maj is not None: self._maj.buf[i].clear()
        if self.w_window: self._w[i] = WassersteinLayer(self.w_window)
        self._from[i] = self.period + settle


# ------------------------------------------------------------------- calibration
def calibrate(healthy_runs: Sequence[Dict[int, List[Optional[np.ndarray]]]], level: float = 0.99,
              decide_from: int = WARMUP + LEARN, min_n: int = 200, window: int = 1,
              w_window: int = 0, majority: bool = False) -> dict:
    """Thresholds from HEALTHY runs only (never from a run containing a fault).

    healthy_runs : list of {array: [vector or None per period]}.
    Returns alarm_threshold (pooled `level` quantile of the median-reference score, the same
    definition as v9.1), anchor_limit (`level` quantile of the common-mode size), learn_shift_limit."""
    sc, cm, ls, ab, of, ww, mw = [], [], [], [], [], [], []
    for V in healthy_runs:
        n, T = len(V), len(V[0]); d = Detector11(n, mode="median", alarm_threshold=None, window=window, w_window=w_window,
                                               majority=majority)
        for t in range(T):
            o = d.update({i: np.asarray(V[i][t], float) for i in V if V[i][t] is not None})
            if t >= decide_from and o["scored"]:
                sc += [s for s in o["score"] if s is not None]; cm.append(o["common_score"])
                ab += [s for s in o["absolute_score"] if s is not None]
                of += [s for s in o["off_score"] if s is not None]
            if o["scored"]:
                ww += [s for s in o["w_score"] if s is not None]
                if majority and t >= decide_from and o["majority_windows"]:
                    mw.append(o["majority_windows"])
        ls += [b.learn_shift for b in d.baseline if b is not None]
    if len(sc) < min_n:
        raise ValueError(f"{len(sc)} healthy scores; need >= {min_n}")
    maj_th = maj_final = None
    if majority and mw:
        maj_th = {W: float(np.percentile([m[W] for m in mw if W in m], 100 * level)) for W in MajorityLayer.WINDOWS}
        maj_final = float(np.percentile([max(m[W] / maj_th[W] for W in m) for m in mw], 100 * level))
    return dict(maj_thresholds=maj_th, maj_final=maj_final,
                alarm_threshold=float(np.percentile(sc, 100 * level)),
                abs_limit=float(np.percentile(ab, 100 * level)),
                off_limit=float(np.percentile(of, 100 * level)),
                w_limit=float(np.percentile(ww, 100 * level)) if ww else None,
                anchor_limit=float(np.percentile(cm, 100 * level)),
                learn_shift_limit=float(np.percentile(ls, 100 * level)) if ls else None)


# -------------------------------------------- optional: standardized slow trend (audited)
def standardized_trend(times: np.ndarray, Y: np.ndarray, rho_cap: float = 0.97) -> Tuple[float, np.ndarray]:
    """Slope t-statistic per component, using the TRUE period numbers as the time axis and an
    AR(1) variance correction for the residual autocorrelation EMA filtering creates:

        Var(beta_hat) = sigma^2 / S_tt * (1 + r1) / (1 - r1)      (r1 = lag-1 residual autocorr)

    Without the factor the t-statistic is too large by sqrt((1+r)/(1-r)) (6.2x at r = 0.95), so
    sum(t^2) is not chi-square(16) and a chi-square threshold is meaningless. Returns
    (mean of t^2 over components, t vector). Empirical null check: v11_trend_null.py."""
    times = np.asarray(times, float); Y = np.asarray(Y, float); n = len(times)
    if n < 10:
        return 0.0, np.zeros(Y.shape[1])
    tc = times - times.mean(); stt = float((tc ** 2).sum())
    if stt == 0:
        return 0.0, np.zeros(Y.shape[1])
    Yc = Y - Y.mean(0); b = tc @ Yc / stt; res = Yc - np.outer(tc, b)
    ss = np.maximum((res ** 2).sum(0), 1e-12)
    r1 = np.clip((res[1:] * res[:-1]).sum(0) / ss, 0.0, rho_cap)
    se = np.sqrt(np.maximum(ss / max(n - 2, 1) / stt * (1 + r1) / (1 - r1), 1e-18))
    t = b / se
    return float((t ** 2).mean()), t


def default_detector(n_arrays: int, **kw) -> Detector11:
    """Detector11 with all four outputs on, using the values calibrated on the benchmark's healthy
    training runs. Re-fit them on your own healthy fleet with calibrate() before relying on them."""
    args = dict(mode="anchored", alarm_threshold=ALARM_THRESHOLD, abs_limit=ABS_LIMIT, min_group=MIN_GROUP,
                off_limit=OFF_LIMIT, w_window=W_WINDOW, w_limit=W_LIMIT,
                majority=True, maj_thresholds=dict(MAJ_THRESHOLDS), maj_final=MAJ_FINAL)
    args.update(kw)
    return Detector11(n_arrays, **args)
