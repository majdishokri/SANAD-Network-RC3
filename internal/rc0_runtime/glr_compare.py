"""
Take the one transferable idea from Harrou & Sun (2016), "Statistical monitoring of linear
antenna arrays": test the residual with a GENERALIZED LIKELIHOOD RATIO instead of an
ad-hoc score, so the threshold comes from a distribution and not from tuning.

Their setting needs a known reference radiation pattern and a known noise variance, and
their test cannot localise the fault. Ours needs neither reference nor variance. What does
transfer is the statistic.

Our calibration vector is v_n in R^{2M} (M gains, M phases). Each tower learns its own
normal during a healthy window. Three statistics on the whitened residual z:
  current : per-coordinate scaling by the 90th-percentile radius, then max over antennas
  GLR-1   : same, but whitened by the FULL empirical covariance of v (the coordinates are
            correlated: gauge fixing subtracts a median from every phase)
  GLR-all : full-covariance whitening, then ||z||^2 over all 2M coordinates
            (the alternative "something is wrong" instead of "one antenna is wrong")
"""
import numpy as np, sys
sys.path.insert(0, '.')
from monitor_v8 import calibration_estimate, calib_vector

M, N_TOW, WARM, LEARN = 8, 7, 60, 60
P_GLITCH, P_MISSING, RAMP, ALPHA = 0.03, 0.05, 80, 0.05
FIXED_START = None   # set to a period to make every fault start there (slow-drift set)


def pipeline(healthy, seed, with_faults=True):
    """Replicates monitor v4 without the legacy consensus (much faster). Returns v_t per tower."""
    rng = np.random.default_rng(seed)
    T = healthy.shape[0]
    faults = {}
    if with_faults:
        for tw in rng.choice(N_TOW, 2, replace=False):
            faults[int(tw)] = dict(start=(FIXED_START if FIXED_START is not None
                                          else int(rng.integers(WARM + LEARN, T - RAMP - 20))),
                                   ant=int(rng.integers(M)),
                                   gain=float(rng.uniform(0.3, 0.8)) * rng.choice([-1, 1]),
                                   phase=float(rng.uniform(3, 8)) * rng.choice([-1, 1]))
    arith = [None] * N_TOW
    pema = [None] * N_TOW
    V = {i: [] for i in range(N_TOW)}
    for t in range(T):
        for i in range(N_TOW):
            if rng.random() < P_MISSING:
                V[i].append(None); continue
            R = healthy[t, i].copy()
            if i in faults and t >= faults[i]["start"]:
                f = faults[i]; fr = min(1.0, (t - f["start"]) / RAMP)
                d = np.ones(M, complex)
                d[f["ant"]] = 10 ** (fr * f["gain"] / 20) * np.exp(1j * np.radians(fr * f["phase"]))
                R = np.diag(d) @ R @ np.diag(d).conj().T
            elif rng.random() < P_GLITCH:
                R = R * 10 ** (rng.uniform(3, 8) * rng.choice([-1, 1]) / 10)
            R = R + 1e-3 * np.real(np.trace(R)) / M * np.eye(M)
            Rn = R / (np.real(np.trace(R)) / M)
            arith[i] = Rn.copy() if arith[i] is None else (1 - ALPHA) * arith[i] + ALPHA * Rn
            _, ph_t, coh = calib_vector(Rn, iters=4)
            if coh >= 0.10:
                pema[i] = ph_t.copy() if pema[i] is None else (1 - ALPHA) * pema[i] + ALPHA * ph_t
            if t < WARM:
                V[i].append(None); continue
            g, p, _ = calibration_estimate(arith[i], iters=4)
            if pema[i] is not None:
                p = pema[i]
            V[i].append(np.r_[g, np.nan_to_num(p)])
    return V, faults


def scores(V, faults):
    """Return per-(tower,period) scores for the three statistics, plus labels."""
    out = {k: [] for k in ("current", "GLR-1", "GLR-all")}
    labels = []
    for i in range(N_TOW):
        seq = V[i]
        train = np.array([v for t, v in enumerate(seq) if v is not None and WARM <= t < WARM + LEARN])
        if len(train) < 20:
            continue
        c = np.median(train, 0)
        r = np.maximum(np.percentile(np.abs(train - c), 90, axis=0),
                       np.r_[np.full(M, 0.02), np.full(M, 0.2)])
        S = np.cov((train - c).T) + np.diag((r / 10) ** 2)      # ridge for stability
        Si = np.linalg.pinv(S)
        L = np.linalg.cholesky(Si + 1e-12 * np.eye(2 * M))
        for t, v in enumerate(seq):
            if v is None or t < WARM + LEARN:
                continue
            z = (v - c) / r
            zw = L.T @ (v - c)
            out["current"].append(float(np.max(np.sqrt(z[:M] ** 2 + z[M:] ** 2))))
            out["GLR-1"].append(float(np.max(zw[:M] ** 2 + zw[M:] ** 2)))
            out["GLR-all"].append(float(zw @ zw))
            labels.append(i in faults and t >= faults[i]["start"])
    return out, np.array(labels)


if __name__ == "__main__":
    data = np.load('sionna_reports.npz')
    tests = [k for k in data.files if k.startswith('s2')]
    agg = {k: [] for k in ("current", "GLR-1", "GLR-all")}
    lab = []
    for n, k in enumerate(tests):
        for dr in range(2):
            V, f = pipeline(data[k], 2000 + 100 * n + dr)
            s, l = scores(V, f)
            for key in agg:
                agg[key] += s[key]
            lab += list(l)
        print(f"   {k} done", flush=True)
    lab = np.array(lab)
    print(f"\n{lab.sum()} faulty tower-periods, {(~lab).sum()} healthy\n")
    
    
    def roc(sc, lab):
        sc = np.array(sc)
        order = np.argsort(-sc)
        s, l = sc[order], lab[order]
        tp = np.cumsum(l) / l.sum()
        fp = np.cumsum(~l) / (~l).sum()
        return fp, tp
    
    
    print(f"{'false alarms/period':>20}" + "".join(f"{k:>14}" for k in agg))
    rows = {k: roc(agg[k], lab) for k in agg}
    for target in (0.002, 0.005, 0.01, 0.02, 0.05):
        line = f"{target*100:>19.1f}%"
        for k in agg:
            fp, tp = rows[k]
            line += f"{tp[np.searchsorted(fp, target)]*100:>13.1f}%"
        print(line)
    print("\nArea under the curve, 0-5% false alarms (higher is better)")
    for k in agg:
        fp, tp = rows[k]
        m = fp <= 0.05
        print(f"   {k:<10}{np.trapezoid(tp[m], fp[m]) / 0.05:.3f}")
