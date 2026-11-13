"""
CALIBRATION MONITOR: detect and diagnose antenna calibration faults on cell towers
=================================================================================

Input : every reporting period (e.g. 15 min) each tower sends its received spatial
        covariance estimate (an M x M Hermitian positive-definite matrix, uniform
        linear array). Reports may be missing.

Version 3 (after the 3GPP TR 38.901 / NVIDIA Sionna test), the default pipeline:
  1. STRUCTURE. A healthy uniform linear array has a Toeplitz covariance whatever the
     radio field, and a calibration error D (one gain/phase per antenna) turns it into
     D T D^H. So each tower is checked against ITSELF, not against neighbours, which see
     different fields in reality.
  2. BASELINE. Each tower keeps an arithmetic running mean of its trace-normalised
     reports (trace normalisation removes gain blips). It is arithmetic because the
     Toeplitz matrices form a linear subspace, and geodesic averaging would leave it.
  3. CALIBRATION ESTIMATE. Every period a structure fit B ~ D T D^H (alternating:
     Toeplitz projection, then a weighted fixed-point update of D) gives each antenna's
     gain (dB) and phase (deg).
  4. OWN NORMAL. After a warm-up, each tower learns, per antenna, the center of its
     estimates and how far they normally stray (the asymptotic center and radius idea
     of thesis ch. 5.2).
  5. DETECTION. The residual v - center is whitened with the tower's own full covariance
     and the score is the largest per-antenna squared Mahalanobis length. That is the
     generalized-likelihood-ratio statistic for "one antenna has moved", with the covariance
     learned instead of assumed. Alarm when score > alarm_threshold.
  6. DIAGNOSIS. diagnose(i) returns per-antenna gain/phase from the structure fit.
     Phase is withheld (NaN) when the field has too little spatial coherence to
     reveal it.

Sionna TR 38.901 UMa results (36 faulty / 90 healthy towers; faults 0.3-0.8 dB, 3-8 deg):
  detection sensitivity 64% (95% CI 48-78), specificity 82% (73-89) at 0.5% false
  alarms per tower-period; right antenna named in 94% (82-98) of faulty towers.
  See sionna_eval.log.

Legacy (kept, not the default): cross-tower consensus = geometric median of the towers'
reports in the SPD cone (thesis ch. 4), clipped-Mann geodesic baselines (ch. 5.1), and the
certificate / error_bound (ch. 5.3). It works only if all towers see the same field. In
3GPP channels it detected 0 of 36 faults. Its outputs are still returned under the
names instant_score / baseline_score / consensus / certificate / error_bound.
"""

import numpy as np


# ---- Hermitian positive-definite matrix functions ---------------------------
def herm_fun(A, f):
    w, V = np.linalg.eigh((A + A.conj().T) / 2)
    return (V * f(w)) @ V.conj().T

def sqrtm(A):    return herm_fun(A, np.sqrt)
def invsqrtm(A): return herm_fun(A, lambda w: 1 / np.sqrt(w))
def logm(A):     return herm_fun(A, np.log)
def expm(A):     return herm_fun(A, np.exp)


def ai_dist(A, B):
    """Affine-invariant distance on the SPD cone."""
    Ais = invsqrtm(A)
    return float(np.linalg.norm(logm(Ais @ B @ Ais), "fro"))


def geodesic(A, B, t):
    """Point at fraction t along the geodesic from A to B."""
    As, Ais = sqrtm(A), invsqrtm(A)
    return As @ herm_fun(Ais @ B @ Ais, lambda w: w ** t) @ As


def riemann_median(mats, iters=40, tol=1e-6):
    """
    Geometric median in the SPD cone (Riemannian Weiszfeld).
    Returns (median, residuals) where residuals[k] is the geodesic length of
    step k, i.e. the fixed-point residual d(x_k, S x_k). A small final residual
    is the convergence certificate.
    """
    # start from the log-Euclidean mean (cheap, inside the cone)
    X = expm(sum(logm(A) for A in mats) / len(mats))
    residuals = []
    for _ in range(iters):
        Xs, Xis = sqrtm(X), invsqrtm(X)
        logs = [logm(Xis @ A @ Xis) for A in mats]
        w = [1.0 / max(np.linalg.norm(L, "fro"), 1e-9) for L in logs]
        T = sum(wi * L for wi, L in zip(w, logs)) / sum(w)
        step = float(np.linalg.norm(T, "fro"))
        residuals.append(step)
        X = Xs @ expm(T) @ Xs
        if step < tol:
            break
    return X, residuals


def error_bound(X, residuals, mats, rho_default=0.9, rho_max=0.9):
    """
    Accuracy guarantee for the returned median X (validated empirically; the proof is
    Sami's conjecture B1). Near the answer the Weiszfeld map contracts at rate rho, so
        d(X, x*) <= r_last / (1 - rho).
    rho is estimated as the worst of the last three residual ratios.
    Returns (bound, flag). The bound is NOT claimed (inf) when:
      "near-report": X sits on one input matrix (Weiszfeld's false fixed point), or
      "not-settled": residuals are not shrinking fast (rho > rho_max), e.g. escaping
                     from a report or stopped in the early phase.
    """
    dists = [ai_dist(X, A) for A in mats]
    if min(dists) < 1e-3 * float(np.median(dists)):
        return float("inf"), "near-report"
    r = residuals
    if len(r) >= 4:
        ratios = [r[i + 1] / r[i] for i in range(len(r) - 4, len(r) - 1) if r[i] > 0]
        rho = max(ratios) if ratios else rho_default
    else:
        rho = rho_default
    if rho > rho_max:
        return float("inf"), "not-settled"
    return r[-1] / (1 - rho), "ok"


# ---- structure-based calibration estimate (version 3) --------------------------
def toeplitz_proj(A):
    """Nearest Hermitian Toeplitz matrix in Frobenius norm (average each diagonal), kept PD."""
    M = A.shape[0]
    t = np.array([np.mean(np.diagonal(A, k)) for k in range(M)])
    j, k = np.indices((M, M))
    T = np.where(k >= j, t[np.abs(k - j)], np.conj(t[np.abs(k - j)]))
    T = (T + T.conj().T) / 2
    w = np.linalg.eigvalsh(T)
    if w[0] <= 1e-9 * abs(w[-1]):
        T = T + (abs(w[0]) + 1e-6 * np.real(np.trace(T)) / M) * np.eye(M)
    return T


def structure_fit(B, iters=8, inner=50):
    """
    Fit B ~ D T D^H with T Hermitian Toeplitz (unknown field) and D = diag(d) (calibration).
    Alternates a T-step (Toeplitz projection of D^-1 B D^-H) with a D-step, the weighted
    least-squares fixed-point update  d_j <- sum_k B_jk conj(T_jk) d_k / sum_k |T_jk|^2 |d_k|^2
    (no division by entries of T). Returns (d, T), with the common phase removed.
    """
    M = B.shape[0]
    d = np.ones(M, complex)
    T = toeplitz_proj(B)
    for _ in range(iters):
        Dinv = np.diag(1 / d)
        T = toeplitz_proj(Dinv @ B @ Dinv.conj().T)
        BTc, W = B * T.conj(), np.abs(T) ** 2
        for _ in range(inner):
            d = (BTc @ d) / (W @ (np.abs(d) ** 2))
        d = d / np.exp(1j * np.angle(np.sum(d)))
    return d, T


def calibration_estimate(B, iters=8, coherence_min=0.10):
    """
    Per-antenna (gain_dB, phase_deg, phase_observable) relative to the array median.
    Phase is observable only if the field has spatial coherence across antennas, i.e. some
    off-diagonal lag of T is at least coherence_min of its diagonal; otherwise NaN.
    """
    d, T = structure_fit(B, iters=iters)
    t = T[0]
    coh = float(np.max(np.abs(t[1:])) / np.real(t[0])) if len(t) > 1 else 0.0
    mag = np.abs(d)
    gain = 20 * np.log10(mag / np.median(mag))
    ph = np.angle(d, deg=True)
    ph = ph - np.median(ph)
    ok = coh >= coherence_min
    if not ok:
        # no spatial coherence: the fit is ill-posed, but gains remain exactly identifiable
        # from the diagonal, since B_jj = |d_j|^2 t_0 for ANY Toeplitz T
        dg = np.real(np.diag(B))
        gain = 10 * np.log10(dg / np.median(dg))
        ph = np.full_like(ph, np.nan)
    return gain, ph, ok


def calib_vector(B, iters=8, coherence_min=0.10):
    """(gain_dB, phase_deg, coherence) from ONE covariance matrix, gauge-fixed."""
    d, T = structure_fit(B, iters=iters)
    t = T[0]
    coh = float(np.max(np.abs(t[1:])) / np.real(t[0])) if len(t) > 1 else 0.0
    mag = np.abs(d)
    gain = 20 * np.log10(mag / np.median(mag))
    ph = np.angle(d, deg=True)
    return gain, ph - np.median(ph), coh


def _whitener(H, center, radius):
    """Cholesky factor of the inverse sample covariance, ridged by (radius/10)^2."""
    D = H - center
    S = np.cov(D.T) + np.diag((radius / 10.0) ** 2)
    return np.linalg.cholesky(np.linalg.pinv(S) + 1e-12 * np.eye(S.shape[0]))


# Version 7: confirmation over time. One reading cannot tell a real fault from a random
# spike - a bad sample, a thin user drop, a burst of interference all lift the score for one
# period. A real calibration drift does not go away, so requiring the score to clear the
# threshold k times out of the last m periods filters the spikes and leaves the drifts. The
# bar on each single reading can then be LOWER for the same false-alarm budget, which is
# where the gain comes from. Measured held out on 72 faulty and 180 healthy towers
# (cm_test.log is the common-mode study; confirm_test.log is this one):
#
#   at 95% specificity   1 reading: 44.4% sensitivity, 80.0% precision, threshold 851
#                        8 of 12 :  55.6% sensitivity, 87.0% precision, threshold 513
#
# 8 of the last 12 periods is 3 hours of confirmation. It was the best of the five rules
# tried at every operating point, and the ordering was monotone in the amount of
# confirmation, so it is a trend rather than one lucky rule.
# Version 8: a second opinion from information the first test throws away.
#
# The own-normal score compares an array with a FIXED reference, its learned normal. So it
# cannot separate a slow environmental drift from a fault: both are simply "far from
# normal". The information it discards is how the reference itself moves. Comparing the
# array now with the array two days ago cancels any drift that was present at both ends and
# leaves the step that a fault makes.
#
# The first attempt at a second stage used which antenna is worst, how concentrated the
# deviation is and whether it grows. All three are computed from the SAME residual, so the
# two stages measured one thing twice and the gain was nil (51.4% -> 58.3%, precision flat).
#
# The step test is genuinely independent, and it is the only change measured in this project
# that lifted sensitivity AND precision together. Held out, per array, 95% specificity
# target (confirm_shift_test.log):
#
#   faults ramping over 20 h   81.9% / 75.6%  ->  90.3% / 85.5%
#   faults drifting over 45 h  70.8% / 77.3%  ->  81.9% / 84.3%
#
# Two rules about it, both measured:
#
# 1. The lookback must be AT LEAST ABOUT THE FAULT'S DRIFT TIME. Sweeping it against a
#    180-period drift: at 0.13 of the drift time the step test actively HURTS (30.6% / 66.7%
#    against 72.2% / 75.4% for the score alone), at 0.53 it breaks even, at 0.8-1.1 it is
#    the best configuration measured. Taking the maximum over several lookbacks does not
#    fix a short one - it unions their false alarms and the gain disappears.
# 2. The step test is an ALTERNATIVE ROUTE, never a gate. As a gate it would blind the
#    monitor to a fault drifting far slower than the lookback, which shows only a small step
#    inside the window - and those are exactly the faults worth catching early. So a large
#    deviation alarms on its own, and the step only lets a MODERATE deviation alarm too.
# The step uses only TWO readings, now and one lookback ago; everything in between is
# discarded. The DERIVATIVE of the curve uses them all. For a steady drift the two carry the
# same signal (step = slope x window), but the slope averages W readings instead of 2, and
# that is what makes it safe when the window does not match the fault's drift time - which
# in the field is never known. Measured (confirm_shift_test.log):
#
#   window 192 (matched)   fast faults  step 90.3/81.2   slope 86.1/83.8   a wash
#                          slow faults  step 75.0/83.1   slope 80.6/82.9   slope wins
#   window  48 (too short) fast faults  step 59.7/79.6   slope 77.8/68.3
#                          slow faults  step 52.8/76.0   slope 63.9/74.2
#
# The slope is shipped because real faults drift over weeks and the drift time cannot be
# known in advance, so robustness to a mismatched window is worth more than the last point
# of sensitivity on a fault that happens to match. "step" stays available.
SHIFT_METHOD = "slope"         # "slope" (least-squares trend) or "step" (two-point)
SHIFT_LAG = 192                # periods of window; 192 x 15 min = 2 days
SHIFT_THRESHOLD = 0.5          # on the step statistic
SHIFT_RATIO = 4.0              # the moderate path sits at alarm_threshold / this
SHIFT_CONFIRM = (4, 6)         # the moderate path needs this much persistence too
DEFAULT_CONFIRM = (8, 12)
DEFAULT_ALARM_THRESHOLD = 694.8   # v8 with SHIFT_METHOD="slope": 95% specificity per
                                  # array on the fault-free runs (step would be 752.9)
SINGLE_READING_THRESHOLD = 441.8  # confirm=None, 0.5% false alarms per tower-PERIOD
# Version 6. The combined statistic gets tighter as the shared drift grows, so one fixed
# number does not work: calibrated for the worst case it throws away sensitivity at the
# high end (measured end to end: 0.00% false alarms and 4.4% sensitivity at 0.8 dB). The
# threshold follows the one observable that tracks the drift without saturating - the
# movement of the cluster median during learning, in dB. Fitted on fault-free calibration
# runs only (cm_test.py / cm_test.log, seeds 100-102), predicting the measured 99.5th
# percentile within 4% across 0-0.6 dB.
CM_THRESHOLD_A, CM_THRESHOLD_B = 111.4, 2.85


def cm_threshold(shared_db):
    """0.5% false alarms per tower-period on the combined statistic."""
    return CM_THRESHOLD_A / (1.0 + CM_THRESHOLD_B * max(shared_db, 0.0))


class CalibrationMonitor:
    def __init__(self, n_towers, alpha=0.05, clip=2.0, warmup=60, learn=60,
                 struct_alpha=0.05, alarm_threshold=None, cm_alarm_threshold=None,
                 cm_gate=0.65, cm_min_towers=5, confirm=DEFAULT_CONFIRM,
                 shift_lag=SHIFT_LAG, shift_threshold=SHIFT_THRESHOLD,
                 shift_ratio=SHIFT_RATIO, shift_method=SHIFT_METHOD):
        """
        warmup         : periods before learning starts (lets the running baseline settle)
        learn          : periods used to learn each tower's own normal (must be fault-free;
                         in practice: the first day after installation or after a repair)
        struct_alpha   : step of the arithmetic running baseline (0.05 = ~20-period memory)
        alarm_threshold: detection threshold on the own-normal score (default: calibrated
                         with the confirmation rule below for 95% specificity per tower)
        confirm        : (k, m) - raise the alarm only when the score clears the threshold
                         on k of the last m periods. None reverts to version 5 behaviour,
                         one reading, for which the threshold is SINGLE_READING_THRESHOLD.
        shift_lag      : periods in the window for the trend test (version 8). Should be at
                         least about as long as the drift time of the faults being hunted.
                         0 turns the trend test off.
        shift_method   : "slope" (least-squares trend over the window, the default, safe
                         when the window is mismatched) or "step" (two-point difference).
        shift_threshold: how big a step counts. Raise it for near-zero false alarms at the
                         cost of sensitivity (measured: 100% precision at 68% sensitivity).
        shift_ratio    : the step route admits deviations above alarm_threshold/this.
        cm_gate        : turn the common-mode rule on when the cluster's shared wobble
                         reaches this fraction of a tower's own wobble (measured, no truth
                         needed). 0.65 sits between the clean case (0.49 measured) and
                         every contaminated case (0.78-1.00) on the benchmark.
        cm_min_towers  : fewest towers in the cluster for a cluster median to be meaningful
        cm_alarm_threshold: leave None to have it set from the measured shared drift
        alpha: Mann step size for the temporal baseline (how fast it learns).
        clip : the most one report may move a baseline, in distance units;
               this is what makes one glitch harmless.
        """
        self.n = n_towers
        self.alpha = alpha
        self.clip = clip
        self.baselines = [None] * n_towers
        self.last = {}
        # version-3 state
        self.warmup, self.learn, self.struct_alpha = warmup, learn, struct_alpha
        self.confirm = confirm
        if alarm_threshold is not None:
            self.alarm_threshold = alarm_threshold
        else:
            self.alarm_threshold = (DEFAULT_ALARM_THRESHOLD if confirm
                                    else SINGLE_READING_THRESHOLD)
        self._over = [[] for _ in range(n_towers)]   # recent over-threshold flags
        # version 8
        self.shift_lag = shift_lag
        self.shift_threshold = shift_threshold
        self.shift_ratio = shift_ratio
        self.shift_method = shift_method
        self._vhist = [[] for _ in range(n_towers)]   # calibration vectors, for the step test
        self._over_lo = [[] for _ in range(n_towers)]
        self.period = 0
        self.arith = [None] * n_towers        # covariance EMA  -> gain
        self.phase_ema = [None] * n_towers    # EMA of per-period phase estimates -> phase
        self._hist = [[] for _ in range(n_towers)]
        self.normal_center = [None] * n_towers
        self.normal_radius = [None] * n_towers
        self.normal_whiten = [None] * n_towers   # v5: full-covariance whitener
        self.calib = [None] * n_towers
        # version 6: common-mode defence. A real network moves as a whole: the same antenna
        # model at every site, the same temperature cycle, so the same per-antenna drift
        # pattern appears everywhere. A UNIFORM shift is already invisible here (trace
        # normalisation and gauge fixing remove it - measured: injecting 0.3 dB uniformly
        # leaves the shared/own ratio at 0.41, unchanged). A shared per-antenna PATTERN
        # survives the gauge, and it hurts: on the 3GPP benchmark the own-normal score falls
        # from 0.393 to 0.196 area under the detection curve as the pattern grows to 0.8 dB.
        # Subtracting the cluster MEDIAN of the calibration vectors each period (median, so
        # one faulty tower cannot drag the reference) removes it, and requiring BOTH views to
        # be abnormal holds the false-alarm rate. cm_gate decides when the rule turns on.
        self.cm_min_towers = cm_min_towers
        self.cm_gate = cm_gate
        self.cm_alarm_threshold = cm_alarm_threshold   # None = set from the measured drift
        self._hist_c = [[] for _ in range(n_towers)]
        self._med_hist = []
        self.cent_center = [None] * n_towers
        self.cent_whiten = [None] * n_towers
        self._loc_stats = [None] * n_towers      # (mean, sd) of the local score while learning
        self._cen_stats = [None] * n_towers      # (mean, sd) of the centered score
        self.common_mode_ratio = None            # observable: shared wobble / own wobble
        self.common_mode_db = None               # observable: shared wobble itself, in dB
        self.common_mode_active = False
        self._cm_threshold_given = cm_alarm_threshold is not None

    def update(self, reports):
        """reports: list of length n; each entry an SPD matrix or None (missing)."""
        present = [i for i, R in enumerate(reports) if R is not None]

        # (1) instant consensus + instant spatial scores
        inst_cons, inst_res = riemann_median([reports[i] for i in present])
        inst_score = [None] * self.n
        for i in present:
            inst_score[i] = ai_dist(inst_cons, reports[i])

        # (2) temporal baselines: clipped Mann step toward each new report
        for i in present:
            R = reports[i]
            if self.baselines[i] is None:
                self.baselines[i] = R.copy()
            else:
                d = ai_dist(self.baselines[i], R)
                t = self.alpha * min(1.0, self.clip / max(d, 1e-9))
                self.baselines[i] = geodesic(self.baselines[i], R, t)

        # (3) consensus of the baselines + baseline scores
        have = [i for i in range(self.n) if self.baselines[i] is not None]
        base_cons, base_res = riemann_median([self.baselines[i] for i in have])
        base_score = [None] * self.n
        for i in have:
            base_score[i] = ai_dist(base_cons, self.baselines[i])

        # (4) version 3: structure-based calibration estimate + own-normal detection
        self.centered_score = [None] * self.n
        self.combined_z = [None] * self.n
        self.shift = [None] * self.n
        self.shift_alarm = [False] * self.n
        struct_score, alarm = self._structure_step(reports)

        self.last = {
            "structure_score": struct_score,     # None while the tower is still learning
            "alarm": alarm,
            "calibration": list(self.calib),     # per tower (gain_dB, phase_deg) or None
            # version 6
            "centered_score": list(self.centered_score),
            "combined_z": list(self.combined_z),
            # version 8
            "shift": list(self.shift),
            "shift_alarm": list(self.shift_alarm),
            "common_mode_ratio": self.common_mode_ratio,
            "common_mode_db": self.common_mode_db,
            "common_mode_active": self.common_mode_active,
            "consensus": inst_cons,
            "baseline_consensus": base_cons,
            "instant_score": inst_score,
            "baseline_score": base_score,
            "certificate": inst_res[-1],          # fixed-point residual at exit
            "certificate_baseline": base_res[-1],
            # guaranteed-accuracy number: consensus is within this distance of the exact median
            "error_bound": error_bound(inst_cons, inst_res, [reports[i] for i in present])[0],
        }
        return self.last

    def _structure_step(self, reports):
        M = next(R.shape[0] for R in reports if R is not None) if any(R is not None for R in reports) else None
        score, alarm = [None] * self.n, [False] * self.n
        for i, R in enumerate(reports):
            if R is None:
                continue
            Rn = R / (np.real(np.trace(R)) / R.shape[0])
            a = self.struct_alpha
            self.arith[i] = Rn.copy() if self.arith[i] is None else (1 - a) * self.arith[i] + a * Rn
            # version 4: the phase must be read BEFORE averaging. Averaging covariances over
            # periods mixes different fields and washes out the off-diagonals, which are exactly
            # what carries phase (measured: coherence 0.77 -> 0.18, phase noise 3.7x worse).
            # So fit each period separately and average the ESTIMATES instead (thesis ch. 5.2
            # applied to the sequence of estimates; low-coherence periods are dropped as the
            # vanishing error terms of ch. 5.4).
            _, ph_t, coh_t = calib_vector(Rn, iters=4)
            if coh_t >= 0.10:
                self.phase_ema[i] = ph_t.copy() if self.phase_ema[i] is None \
                    else (1 - a) * self.phase_ema[i] + a * ph_t
        if self.period >= self.warmup:
            # pass 1: this period's calibration vector for every tower that has one
            vecs = {}
            for i in range(self.n):
                if self.arith[i] is None:
                    continue
                g, p, _ = calibration_estimate(self.arith[i], iters=4)
                if self.phase_ema[i] is not None and np.isfinite(p).all():
                    p = self.phase_ema[i]            # phase from the per-period average
                self.calib[i] = (g, p)
                vecs[i] = np.r_[g, np.nan_to_num(p)]

            # pass 2 (version 6): the cluster's common mode, as a median so that one bad
            # tower cannot define the reference
            med = (np.median(np.array([vecs[i] for i in vecs]), 0)
                   if len(vecs) >= self.cm_min_towers else None)
            learning = self.period < self.warmup + self.learn
            if learning and med is not None:
                self._med_hist.append(med)

            for i, v in vecs.items():
                vc = None if med is None else v - med
                if learning:
                    self._hist[i].append(v)
                    if vc is not None:
                        self._hist_c[i].append(vc)
                    continue
                if self.normal_center[i] is None and len(self._hist[i]) >= 20:
                    H = np.array(self._hist[i])
                    self.normal_center[i] = np.median(H, 0)
                    m = H.shape[1] // 2
                    r = np.maximum(
                        np.percentile(np.abs(H - self.normal_center[i]), 90, axis=0),
                        np.r_[np.full(m, 0.02), np.full(m, 0.2)])   # floors: 0.02 dB, 0.2 deg
                    self.normal_radius[i] = r
                    # v5: whiten with the FULL covariance, not per coordinate. The coordinates
                    # are correlated (gauge fixing subtracts a common value from all gains and
                    # all phases), and ignoring that loses detection power. Measured on the
                    # 3GPP benchmark: area under the detection curve 0.322 -> 0.393.
                    # This is the generalized-likelihood-ratio form used for antenna-array
                    # monitoring by Harrou & Sun (2016); here with a learned covariance
                    # instead of an assumed noise variance and a known reference pattern.
                    self.normal_whiten[i] = _whitener(H, self.normal_center[i], r)
                    L = np.array([self._glr(i, w) for w in H])
                    self._loc_stats[i] = (float(L.mean()), float(L.std()) + 1e-9)
                    # v6: the same construction on the common-mode-free vectors
                    if len(self._hist_c[i]) >= 20:
                        Hc = np.array(self._hist_c[i])
                        self.cent_center[i] = np.median(Hc, 0)
                        rc = np.maximum(
                            np.percentile(np.abs(Hc - self.cent_center[i]), 90, axis=0),
                            np.r_[np.full(m, 0.02), np.full(m, 0.2)])
                        self.cent_whiten[i] = _whitener(Hc, self.cent_center[i], rc)
                        C = np.array([self._glr_c(i, w) for w in Hc])
                        self._cen_stats[i] = (float(C.mean()), float(C.std()) + 1e-9)
                    if self.common_mode_ratio is None and len(self._med_hist) >= 20:
                        self._measure_common_mode()

                if self.normal_center[i] is not None:
                    score[i] = self._glr(i, v)
                    self._vhist[i].append(v)
                    if self.shift_lag:
                        del self._vhist[i][:-(self.shift_lag + 20)]
                    self.shift[i] = self.shift_score(i)
                    # version 6: when the cluster is measurably moving as a whole, a tower
                    # must look abnormal BOTH against its own history and against the
                    # cluster at this instant. Measured on the benchmark, sensitivity at a
                    # 0.5% false-alarm budget with a 0.3/0.5/0.8 dB shared pattern:
                    # 14.3%/11.8%/11.3% for the own-history test alone against
                    # 20.0%/18.6%/17.4% for the pair.
                    use_cm = (self.common_mode_active and vc is not None
                              and self.cent_whiten[i] is not None
                              and self._loc_stats[i] is not None
                              and self._cen_stats[i] is not None
                              and self.cm_alarm_threshold is not None)
                    if use_cm:
                        lm, ls = self._loc_stats[i]
                        cm_, cs_ = self._cen_stats[i]
                        cen = self._glr_c(i, vc)
                        combined = min((score[i] - lm) / ls, (cen - cm_) / cs_)
                        self.centered_score[i] = cen
                        self.combined_z[i] = combined
                        alarm[i] = self._confirmed(i, combined > self.cm_alarm_threshold)
                    else:
                        alarm[i] = self._confirmed(
                            i, self.alarm_threshold is not None
                               and score[i] > self.alarm_threshold)
                    # version 8: a moderate deviation may alarm too, but only with an
                    # independent step behind it. This is an extra route, never a gate.
                    if not alarm[i] and self.shift_lag and self.alarm_threshold is not None:
                        k, m = SHIFT_CONFIRM
                        buf = self._over_lo[i]
                        buf.append(score[i] > self.alarm_threshold / self.shift_ratio)
                        del buf[:-m]
                        if sum(buf) >= k and self.shift[i] > self.shift_threshold:
                            alarm[i] = True
                            self.shift_alarm[i] = True
        self.period += 1
        return score, alarm

    def shift_score(self, i):
        """How much array i has moved over the trend window, per antenna.

        "slope": the least-squares trend across the whole window, multiplied back by the
        window length so the number means "how much it moved", comparable with "step".
        "step": medians of 13 readings at each end, so one bad period cannot create a step.

        Returns 0.0 until there is enough history, which keeps the trend route silent
        rather than wrong while the monitor is still filling up.
        """
        h = self._vhist[i]
        W = self.shift_lag
        if not W or len(h) < W + 8:
            return 0.0
        if self.shift_method == "step":
            d = (np.median(np.array(h[-13:]), 0)
                 - np.median(np.array(h[-W - 13:-W]), 0))
        else:
            seg = np.array(h[-(W + 1):])
            x = np.arange(len(seg)) - (len(seg) - 1) / 2.0
            d = (x[:, None] * seg).sum(0) / (x * x).sum() * W
        m = len(d) // 2
        return float(np.max(d[:m] ** 2 / 0.25 + d[m:] ** 2 / 25.0))

    def _confirmed(self, i, over):
        """Version 7: k of the last m readings must be over the threshold."""
        if not self.confirm:
            return bool(over)
        k, m = self.confirm
        buf = self._over[i]
        buf.append(bool(over))
        del buf[:-m]
        return sum(buf) >= k

    def _glr(self, i, v):
        """Largest per-antenna squared Mahalanobis length against the tower's own normal."""
        z = self.normal_whiten[i].T @ (v - self.normal_center[i])
        m = len(v) // 2
        return float(np.max(z[:m] ** 2 + z[m:] ** 2))

    def _glr_c(self, i, vc):
        z = self.cent_whiten[i].T @ (vc - self.cent_center[i])
        m = len(vc) // 2
        return float(np.max(z[:m] ** 2 + z[m:] ** 2))

    def _measure_common_mode(self):
        """How much of a tower's wobble is shared by the whole cluster.

        Observable from the data alone: the movement of the cluster median during the
        learning window, divided by a single tower's own movement. Near 0.5 when the towers
        are independent, near 1 when a common mode dominates. No ground truth needed.
        """
        Med = np.array(self._med_hist)
        m = Med.shape[1] // 2
        shared = float(np.mean(np.std(Med, 0)[:m]))
        own = [float(np.mean(np.std(np.array(h), 0)[:m]))
               for h in self._hist if len(h) >= 20]
        if not own:
            return
        self.common_mode_db = shared
        self.common_mode_ratio = shared / (float(np.mean(own)) + 1e-12)
        self.common_mode_active = self.common_mode_ratio >= self.cm_gate
        if not self._cm_threshold_given:
            self.cm_alarm_threshold = cm_threshold(shared)

    def relearn(self, i):
        """Forget tower i's learned normal (call after a repair or recalibration)."""
        self._hist[i] = []
        self.phase_ema[i] = None
        self.normal_whiten[i] = None
        self.normal_center[i] = self.normal_radius[i] = None
        self._hist_c[i] = []
        self._over[i] = []
        self._over_lo[i] = []
        self._vhist[i] = []
        self.cent_center[i] = self.cent_whiten[i] = None
        self._loc_stats[i] = self._cen_stats[i] = None

    def diagnose(self, i, method="structure", return_quality=False, coherence_min=0.10):
        """
        Per-antenna (gain_dB, phase_deg) for tower i.
        method="structure" (default, version 3): structure fit on the tower's own running baseline;
                            phase is NaN when the field reveals no phase.
        method="consensus" (legacy): compare with the other towers' consensus (needs the
                            same-field assumption).
        """
        if method == "structure":
            g, p, ok = calibration_estimate(self.arith[i], coherence_min=coherence_min)
            if ok and self.phase_ema[i] is not None:
                p = self.phase_ema[i]
            if return_quality:
                return g, p, {"phase_ok": np.full(len(g), ok)}
            return g, p
        return self._diagnose_consensus(i, return_quality, coherence_min)

    def _diagnose_consensus(self, i, return_quality=False, coherence_min=0.10):
        """
        Layer 2 (the "MRI"): estimate per-antenna gain/phase errors of tower i.
        Model: faulty baseline  B ~ D R D^H  with D = diag(d), R the consensus of the
        OTHER towers' baselines, i.e.  B[j,k] ~ d_j conj(d_k) R[j,k].

        Fit: weighted least squares  min_d  sum_jk |B[j,k] - d_j conj(d_k) R[j,k]|^2.
        Its fixed-point update needs NO division by entries of R:
            d_j <- sum_k B[j,k] conj(R[j,k]) d_k  /  sum_k |R[j,k]|^2 |d_k|^2
        Entries of R near zero simply get near-zero weight. (The earlier version
        divided B by R entry by entry, which explodes when an entry of R is small,
        e.g. at a site with little interference.)

        Honesty rule: an antenna's PHASE can only be seen through its coupling to
        other antennas (off-diagonal entries). Coherence c_jk = |R_jk|/sqrt(R_jj R_kk).
        Antennas are linked when c_jk >= coherence_min; phase is reported only for
        antennas in the largest linked group, NaN ("cannot tell here") otherwise.
        GAIN is always identifiable from the diagonal (R_jj > 0: noise power).

        Returns (gain_dB, phase_deg) per antenna, relative to the array median;
        with return_quality=True also a dict: phase_ok mask, coherence, fit residual.
        """
        others = [self.baselines[k] for k in range(self.n)
                  if k != i and self.baselines[k] is not None]
        R, _ = riemann_median(others)
        B = self.baselines[i]
        M = R.shape[0]
        diagR = np.real(np.diag(R))
        diagB = np.real(np.diag(B))

        # weighted rank-one fit, started from the diagonal-only gain estimate
        d = np.sqrt(np.maximum(diagB, 0) / diagR).astype(complex)
        BRc = B * R.conj()
        W = np.abs(R) ** 2
        for _ in range(100):
            num = BRc @ d
            den = W @ (np.abs(d) ** 2)
            d_new = num / den
            if np.linalg.norm(d_new - d) < 1e-10 * np.linalg.norm(d):
                d = d_new
                break
            d = d_new
        fit = np.outer(d, d.conj()) * R
        residual = float(np.linalg.norm(B - fit) / np.linalg.norm(B))

        # which antennas' phases are observable: largest group linked by coherence
        coh = np.abs(R) / np.sqrt(np.outer(diagR, diagR))
        link = coh >= coherence_min
        np.fill_diagonal(link, True)
        unseen, best = set(range(M)), set()
        while unseen:
            stack, comp = [unseen.pop()], set()
            while stack:
                a = stack.pop()
                comp.add(a)
                for b in np.flatnonzero(link[a]):
                    if b in unseen:
                        unseen.discard(b)
                        stack.append(int(b))
            if len(comp) > len(best):
                best = comp
        phase_ok = np.array([j in best for j in range(M)]) if len(best) >= 2 else np.zeros(M, bool)

        mag = np.abs(d)
        gain_db = 20 * np.log10(mag / np.median(mag))
        phase = np.full(M, np.nan)
        if phase_ok.any():
            ok = np.flatnonzero(phase_ok)
            ref = np.exp(1j * np.angle(np.sum(d[ok] / np.abs(d[ok]))))   # common phase
            ph = np.angle(d[ok] / ref, deg=True)
            phase[ok] = ph - np.median(ph)
        if return_quality:
            return gain_db, phase, {"phase_ok": phase_ok, "coherence": coh, "fit_residual": residual}
        return gain_db, phase
