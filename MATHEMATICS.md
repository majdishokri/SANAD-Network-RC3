# SANAD Network RC3 — Verified Mathematical Model and Source Map

This document describes the mathematics **actually present in the frozen RC3 source and its inference dependencies**. It distinguishes the current trained classifier from earlier physical preprocessing or geometric experiments. Equations are intended to reflect code behavior, not to introduce a new algorithm.

## 1. Signal and source boundary

The inference entry point `rc3.py` consumes the derived sequence:

\[
\{v_{t,a},n_{t,a},f^{(\mathrm{fast})}_{t,a},f^{(\mathrm{strict})}_{t,a},\delta_{t,a}\},\quad
0\leq t<300,\ 0\leq a<7.
\]

Each evidence triplet consists of **anomaly score**, **dominance**, and **1-based antenna element ID**. The raw input contract of the preceding preprocessing, described in `internal/rc0_runtime/sanad_v29.py`, is a 16-vector per array:

\[
y_{t,a}=(g_1,\ldots,g_8,\phi_1,\ldots,\phi_8),
\]

where gains are in dB and phases are in degrees. **These are covariance-derived calibration proxies, not directly verified RF instrumentation values.** They are computed upstream of this standalone inferencer.

### Upstream structural physics (context; NOT run by `rc3.py`)

Historical `monitor_v8.py` fits the covariance model

\[
B\approx DTD^H,\qquad D=\operatorname{diag}(d_1,\ldots,d_8),\quad T=T^H\text{ Toeplitz},
\]

using alternating projections onto Hermitian Toeplitz structure and weighted fixed-point updates of the diagonal calibration vector. With a sufficiently coherent field, relative estimates use

\[
\hat g_j=20\log_{10}\frac{|d_j|}{\operatorname{median}_k |d_k|},\quad
\hat\phi_j=\angle d_j-\operatorname{median}_k \angle d_k.
\]

The original `monitor_v8.py` also implements affine-invariant SPD distance
\(d_{\mathrm{AI}}(A,B)=\|\log(A^{-1/2}BA^{-1/2})\|_F\), matrix geodesics, and a Riemannian median. **Its own module documentation labels the consensus and error-certificate route as legacy / not the default for 3GPP channels. Do not describe that route as an active RC3 learned-classifier layer.** The Toeplitz assumption is tied to the original regular linear-array model and must not be generalized to arbitrary antenna geometries without new validation.

## 2. Spatial smooth–sparse decomposition (v29/v33)

This computation is implemented in `internal/rc0_runtime/v29_research.py`, invoked by historical spatial evidence modules. Gains/phases are rescaled by physical units \((0.5\,\mathrm{dB},\ 5^\circ)\):

\[
Y\in\mathbb R^{8\times2},\quad Y_{j,1}=g_j/0.5,\quad Y_{j,2}=\phi_j/5.
\]

Let \(X=[\mathbf1,\operatorname{linspace}(-1,1,8)]\in\mathbb R^{8\times2}\) and \(P_X=XX^+\). A smooth drift \(D\) and sparse disturbance \(S\) are separated by **eight damped iterations**:

\[
D^{(k+1)}=(1-\beta)D^{(k)}+\beta P_X\big(Y-S^{(k)}\big),
\]
\[
S^{(k+1)}=(1-\beta)S^{(k)}+\beta\,\operatorname{soft}_{\lambda}\big(Y-D^{(k+1)}\big),
\]

where \(\beta=0.55\), \(\lambda=0.12\), and \(\operatorname{soft}_{\lambda}(u)=\operatorname{sgn}(u)\max(|u|-\lambda,0)\). Initial drift is \(P_XY\) and sparse component is zero. This is a **damped alternating smooth/sparse update**, discussed as Ishikawa/Mann-*type* averaging in the source. The code explicitly does **not** establish a hyperbolic fixed-point convergence theorem for its time-varying RF process.

An element score is

\[
s_j=|S_{j,1}|+|S_{j,2}|,\quad j^*=\arg\max_j s_j,\quad
\eta=\frac{s_{j^*}}{\max(s_{\mathrm{runner\ up}},10^{-5})}.
\]

The fast/strict v33 evidence paths use score threshold \(0.7687096042171566\), dominance \(\eta\geq1.5\) or \(\eta\geq1.8\), and persistence **8 of 12** or **10 of 15** reports respectively. The RC3 input trace already contains these evidence summaries; its RC0 replay uses the same frozen logic for meta-features.

## 3. Frozen temporal residual evidence (v35)

`temporal_v35.py` uses a second linear spatial projector for a normalized \(8\times2\) gain/phase array:

\[
X_t=[\mathbf1,\,(0,1,\ldots,7)-3.5],\quad
P_\perp=I-X_tX_t^+,\quad E_t=P_\perp Y_t.
\]

For each element, baseline \(b_j\) is the coordinate-wise median of commissioning reports 60–119; spread \(\sigma_j=\max(1.4826\operatorname{MAD}(E_j),0.16)\). With the recent up-to-12-report window, compute:

\[
\Delta_j=\operatorname{median}(E_{\mathrm{recent},j})-b_j,\qquad
q_j=\sqrt{\sum_{c=1}^{2}(\Delta_{j,c}/\sigma_{j,c})^2},
\]

and top-element dominance \(\eta=q_{j^*}/\max(q_{\mathrm{second}},0.10)\).

`sanad_v35.py` requires at least **9/14** threshold hits with \(q\ge4.0\) and \(\eta\ge1.45\); a stronger sole-evidence path requires **11/14** hits with \(q\ge6.0\) and \(\eta\ge2.15\). Evidence from at least four simultaneous ready arrays is flagged as potentially ambiguous, not automatically interpreted as four independent calibration failures. The v35 frozen config used by RC3 has `base_policy="legacy_union"`. All paths share correlated underlying observations.

## 4. Frozen RC0 learned gate (meta input to RC3)

The internal `inference_v35_gate.py` uses causal per-array replay, window counts, modal IDs, and persistence counts from `feature_engineering.py`. Let \(x_a^{(0)}\) be the RC0 feature vector:

\[
p_a^{(0)}=\tfrac12 p_{\mathrm{logisticL2}}(x_a^{(0)})+
\tfrac12 p_{\mathrm{randomforest}}(x_a^{(0)}),
\]

and \(c_a^{(35)}\) be the v35 candidate flag. RC0's source gate is:

\[
F_a^{(0)}=\mathbf1[c_a^{(35)}=1]\mathbf1[p_a^{(0)}\ge0.18].
\]

The RC0 flags enter RC3's 15-dimension spatial feature vector. The frozen RC0 candidate may be absent even when RC3 ultimately alerts; RC0 is **not** a mandatory final RC3 decision gate.

## 5. RC3's 693 causal endpoint features

The frozen `rc3.py` uses nine observed channels per array and their spatial relative residuals:

\[
r_{t,a,c}=z_{t,a,c}-\operatorname{median}_{a'=1}^{7}z_{t,a',c}.
\]

`z` consists of 6 fast/strict evidence coordinates and 3 temporal evidence coordinates. Therefore there are **18 streams** per array (9 raw + 9 relative). For every stream, RC3 computes **nine statistics** (finite count, mean, standard deviation, five quantiles at 10/25/50/75/90%, maximum) in four time windows, plus a median difference between 260–299 and 220–259:

\[
\underbrace{18(4\cdot9+1)}_{666}
+\underbrace{3(3\cdot9)}_{27\ \mathrm{v11\ summaries}}
=\boxed{693}\ \mathrm{features\ per\ array}.
\]

All windows end no later than report 299 and use no future observations. NaNs are handled explicitly; train labels and fault onset times are **not used in inference**.

## 6. Two base classifiers and 15 spatial meta-features

Let the two stage-1 classifiers be ExtraTrees and HistGradientBoosting. Their average score is

\[
p_a^{(1)}=\tfrac12\big(p_{\mathrm{ExtraTrees}}(x_a)+p_{\mathrm{HGB}}(x_a)\big).
\]

For each array \(a\), RC3 forms this exact 15-dimensional vector from the seven stage-1 probabilities and RC0 flags:

1. \(p_a^{(1)}\)
2. stabilized log-odds \(\log((p_a+10^{-5})/(1-p_a+10^{-5}))\)
3. RC0 confirmed flag \(F_a^{(0)}\)
4. descending cohort rank of \(p_a\)
5–7. largest, second-largest, third-largest cohort probabilities
8–9. cohort mean and population standard deviation
10. largest probability minus \(p_a\)
11–12. \(p_a/(p_{\max}+10^{-5})\), \(p_a/(p_{\mathrm{second}}+10^{-5})\)
13–14. number of arrays with \(p>0.9\) and with \(p>0.4\)
15. number of RC0-confirmed arrays.

The final **ExtraTrees meta-classifier** outputs \(p_a^{(3)}\) and frozen decision:

\[
\boxed{\mathrm{Alert}_a=\mathbf1\big[p_a^{(3)}\ge0.50\big]}.
\]

The saved bundle includes `decision_threshold=0.21120083844004756` as historical model metadata, but the **qualified `rc3.py` hard-codes `THRESHOLD=0.50`**, and that is the only threshold used for this frozen release. Changing it invalidates the prospective validation.

## 7. What the frozen engine DOES NOT implement

- No Poincaré/Lorentz hyperbolic recommender training; SANAD Engine/SegProx is a separate project.
- No generic self-configuration for arbitrary N/M. The frozen classifier is 7 arrays × 8 elements × 300 reports.
- No direct live CSI-to-evidence preprocessing; no actual RF gain/phase actuation.
- No per-element identity output in the **final** RC3 public JSON (only seven array-level alert booleans and scores); element identities occur in intermediate evidence.
- No independent proof of distinguishing arbitrary channel changes from physical calibration defects.
- No prospective field validation on faulty antenna hardware.

## 8. Validation provenance

Model digest (uncompressed) `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4`; frozen threshold 0.50; 144 prospective Sionna seeds 3000–3143. Observed detection 274/288 faults, healthy-only FP 0/1008, injected-neighbor FP 13/720. These are **simulation** results, not an RF field-product claim.

Source map: `rc3.py`, `internal/rc0_runtime/{v29_research,temporal_v35,sanad_v33,sanad_v35,feature_engineering,inference_v35_gate}.py`, plus source-repository `monitor_v8.py` and `detect_v11.py` (upstream context, intentionally not included in the runtime-only extraction).
