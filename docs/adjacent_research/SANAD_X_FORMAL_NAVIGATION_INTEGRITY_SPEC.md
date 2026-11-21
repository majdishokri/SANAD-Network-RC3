# SANAD-X — Physics-Constrained Navigation Integrity: Formal Mathematical Specification

**Research proposal only (21 November 2026).** This document proves conditional mathematical statements, not operational GNSS authentication, universal spoofing immunity, or aircraft safety. It is adjacent research, not a modification to the frozen RC3 algorithm, SANAD-X v10.26 stable baseline, or v10.36 executable research monitor. No tests, simulations, or flights were performed for this document.

## 1. Correct starting point: what SANAD-X actually measures

The project archive contains a v10.33 three-latent-clock research model, a v10.35 causal fusion-policy comparison, and a v10.36 causal monitor. The v10.36 inputs are three quantities: **RAW** \(G-A\), **REFERENCE** \(G-B\), and **CORRECTED** \(G-A-W\), where \(W\) is a nuisance witness. The results are an early warning from any channel, a cross-comparison warning from RAW/CORRECTED with REFERENCE, or abstention on a declared trust/provenance defect.

The separate SANAD Network RC3 system processes 300 reports at 15-minute intervals for seven receiver arrays of eight antennas each; it detects *calibration anomalies* in a frozen Sionna setting. It is not a GNSS spoofing receiver or a flight-critical 1 Hz/10 Hz/100 Hz monitor. The independent-reference CGPE and constrained MFR work concern *RF-chain estimation and beam recovery*, not satellite source authentication. No RC3, CGPE or MFR output may silently be promoted to a trusted UAV position or time witness.

## 2. Physics assumptions and state equations

Define a local navigation state \(x=(p,v,R,b_a,b_\omega,b_c,d_c)\), with position \(p\in\mathbb R^3\), velocity \(v\in\mathbb R^3\), attitude \(R\in SO(3)\), inertial biases \(b_a,b_\omega\), receiver clock bias \(b_c\) in seconds and clock drift \(d_c\). Strapdown kinematics in a properly corrected navigation frame are

$$
\dot p=v,\quad
\dot v=R(f_m-b_a-n_a)+g(p)+e_{\rm nav},\quad
\dot R=R[\omega_m-b_\omega-n_\omega]_\times,\quad
\dot b_c=d_c.
$$

Earth rotation, local gravity, lever arms, timing latency, temperature and vibration must be included in a complete physical model or covered by verified bounded errors. An IMU measures specific force and angular rate; it does not independently provide an absolute position.

A GNSS code pseudorange from satellite \(i\) obeys

$$
P_i=\|p-s_i(t_i^{\rm tx})\|+
c(b_c-\delta t_i)+I_i+T_i+S_i+H_i+M_i+\eta_i+a_i,
$$

where \(s_i\) is satellite ephemeris, \(I_i,T_i\) are ionosphere/troposphere, \(S_i\) is the appropriate propagation/relativity/Earth-rotation correction, \(H_i\) instrument group delay, \(M_i\) multipath, \(\eta_i\) noise, and \(a_i\) an unknown disturbance. Corrections must not be double-counted if already incorporated in \(s_i\), \(\delta t_i\), or another term. Doppler adds line-of-sight relative velocity and receiver-clock drift, not source authentication.

A nanosecond of timing error corresponds to

$$
c(10^{-9}\ {\rm s})=0.299792458\ {\rm m}.
$$

**A1 — Independent clean anchor.** At \(t_0\), independently established pose and velocity satisfy \(\|p_0-\hat p_0\|\le e_{p0}\), \(\|v_0-\hat v_0\|\le e_{v0}\). A GNSS position already controlled by an attacker is not independent.

**A2 — Bounded inertial uncertainty.** Within the stated horizon, effective accelerometer error is at most \(e_a\), gravity/model error at most \(e_g\), angular misalignment is at most \(e_{\theta0}+e_\omega T\) radians, and measured specific force magnitude is at most \(F_{\max}\), all as *deterministic physical bounds*, not covariance-only assumptions.

**A3 — Independent clock.** A trusted time source gives \(|b_c-\hat b_c|\le E_c(T)\) in seconds, independently of suspect GNSS. Re-disciplined GNSS-derived clocks do not satisfy this claim.

**A4 — Complete error envelopes.** Ephemeris, propagation, instrument, multipath, hardware, synchronization, sampling and latency errors have justified upper bounds, with unit consistency.

**A5 — Trust separation.** Data provenance, calibration validity and physical independence are actually established. An asserted Boolean flag, cryptographic hash or MAC does not establish sensor truth.

**A6 — Certified evidence horizon.** The model is used only while A1–A5 still hold. Else the integrity result must be **INSUFFICIENT_EVIDENCE**, not **SAFE**.

## 3. Theorem 1 — Common-mode impossibility of the original three comparisons

Let \(z=(G,A,B)^T\). The two raw timing differences are \(y=Hz\), with

$$
H=\begin{bmatrix}1&-1&0\\1&0&-1\end{bmatrix},\qquad
H\begin{bmatrix}1\\1\\1\end{bmatrix}=0.
$$

The rows are linearly independent, so \(\operatorname{rank}(H)=2\), and \(\ker H=\operatorname{span}\{(1,1,1)^T\}\).

**Proof.** For any time-varying scalar \(q(t)\),

$$
H(z+q(t)\mathbf 1)=Hz+q(t)H\mathbf1=Hz.
$$

Thus all three clocks can change together without affecting either comparison. The corrected channel \(G-A-W=(G-A)-W\) is derived from the first comparison and a witness. It does not independently measure the missing common-mode degree of freedom. Therefore neither the three comparison channels nor any voting rule based solely on them can detect every shared clock perturbation. **QED.**

**Implication:** ANY_CHANNEL and NEW_CROSS are useful *anomaly indicators*, not signal-origin proofs. Each can also be triggered by nonadversarial faults. Adding more algebraic transforms of the same clocks does not add missing information; a physically independent measurement with a different observation equation is required.

## 4. Theorem 2 — Orbital geometry and time of flight do not authenticate GNSS

For any arbitrary hypothetical position \(p^\star\) and receiver clock bias \(b^\star\), the apparent ranges

$$
P_i^\star=\|p^\star-s_i(t_i^{\rm tx})\|+cb^\star+d_i
$$

can satisfy ordinary GNSS navigation equations for admissible model corrections \(d_i\).

**Proof.** Substitution of \(p^\star,b^\star\) produces the same equations used by a navigation solver. Those equations constrain *apparent propagation delays and satellite ephemeris*, but do not on their own prove the RF waveform was actually radiated from each satellite. Therefore even perfectly consistent orbital-ellipse, arc, tangent or common-view pseudorange geometry cannot unconditionally authenticate origin. **QED.**

A real external trusted position/time anchor, independently measured RF direction, securely authenticated navigation data where available, or independent physical sensing can supply additional evidence, each subject to its own threat model. None can be inferred from the existing v10.36 timing channels alone.

## 5. Theorem 3 — Certified inertial position-error tube

Let \(T=t-t_0\ge0\). From \(\|R-\hat R\|_2=2\sin(\theta/2)\le\theta\), we have the conservative effective acceleration-error envelope

$$
\|\delta a(T)\|\le A_0+A_1T,\quad
A_0=e_a+e_g+F_{\max}e_{\theta0},\quad
A_1=F_{\max}e_\omega.
$$

Then define

$$
E_v(T)=e_{v0}+A_0T+\frac12 A_1T^2,
$$

$$
\boxed{E_p(T)=e_{p0}+e_{v0}T+\frac12A_0T^2+\frac16A_1T^3.}
$$

**Proof.** Apply the triangle inequality to \(\dot e_v=\delta a\): \(\|e_v(T)\|\le e_{v0}+\int_0^T(A_0+A_1u)\,du=E_v(T)\). Then integrate \(\dot e_p=e_v\) to obtain the displayed \(E_p(T)\). **QED.**

Consequently the conditional inertial position set is

$$
\mathcal X_I(T)=\{p:\|p-\hat p_I(T)\|_2\le E_p(T)\}.
$$

This does **not** guarantee that the inertial estimate remains accurate indefinitely. The tube expands quadratically or cubically when acceleration/orientation error terms are nonzero. Unverified calibration means the certified set is unknown, not a convenient small sphere.

## 6. Theorem 4 — Conditional, deterministic GNSS inconsistency detection

Compute a GNSS code observation predicted from the independently anchored inertial state and independent clock:

$$
\hat P_i=h_i(\hat p_I,\hat b_c).
$$

Geometric range is 1-Lipschitz in receiver position:

$$
\left|\|p-s_i\|-\|\hat p_I-s_i\|\right|\le\|p-\hat p_I\|\le E_p(T).
$$

Let \(U_i(T)\) be a physically justified bound in **metres** on all remaining benign errors, including \(cE_c(T)\), satellite ephemeris/time, propagation, multipath, radio hardware and synchronization errors. Set

$$
B_i(T)=E_p(T)+U_i(T),\qquad r_i(T)=P_i^{\rm obs}(T)-\hat P_i(T).
$$

**Soundness proof.** Under A1–A6 and a benign measurement, the triangle inequality gives \(|r_i|\le B_i\). A rule that flags \(|r_i|>B_i\) therefore cannot reject a benign measurement *as long as the verified bounds actually hold*. **QED.**

**Detectability proof.** Suppose an additive perturbation \(a_i\) alters a benign measurement, so \(P_i^{\rm obs}=P_i^{\rm benign}+a_i\). By the reverse triangle inequality,

$$
|r_i|\ge |a_i|-|P_i^{\rm benign}-\hat P_i|
\ge|a_i|-B_i.
$$

It follows that

$$
\boxed{|a_i(T)|>2B_i(T)\ \Longrightarrow\ |r_i(T)|>B_i(T).}
$$

Thus at a valid sample, a perturbation exceeding **twice the complete benign error envelope** on at least one covered channel is guaranteed to produce **PHYSICAL_INCONSISTENCY_UNATTRIBUTED**. **QED.**

This theorem proves neither that all spoofing is detectable nor that any detected discrepancy was an attack: hardware failure, model error, wrong input calibration and other disturbances remain alternative explanations. If \(a_i\) stays inside the envelope or corrupts the independent anchor, detection is not guaranteed.

A more powerful set-membership certificate defines

$$
\mathcal F(T)=\{(p,b):p\in\mathcal X_I(T),\ |b-\hat b_c|\le E_c(T),\
|P_i^{\rm obs}-h_i(p,b)|\le U_i^{\rm other}(T)\ \forall i\}.
$$

If \(\mathcal F(T)=\varnothing\), the sensors have **no jointly feasible physical state within the assumed bounds**. If nonempty, the conclusion is **CONSISTENT_UNDER_MODEL_NOT_AUTHENTICATED**. A solver's failure to converge is not proof of emptiness. For nonconvex \(h_i\), a globally valid infeasibility certificate or verified enclosure is required.

## 7. Theorem 5 — Legitimate use of SANAD fixed-point mathematics

To connect the HOK/MFR mathematical idea without falsely claiming GNSS authenticity, consider a *local convex surrogate* with constrained state \(z\in\mathcal C\), \(\mathcal C\) a nonempty closed convex set:

$$
J(z)=\tfrac12\|W(Az-y)\|_2^2+\tfrac{\lambda}{2}\|z-z_I\|_2^2,\qquad\lambda>0.
$$

Let \(Q=A^TW^TWA+\lambda I\) be symmetric positive definite with spectrum \(0<\mu\le\lambda_i(Q)\le L\). Define the projected-gradient map

$$
\mathcal T(z)=P_{\mathcal C}(z-\eta\nabla J(z)),\qquad0<\eta<2/L.
$$

Because Euclidean convex projection is nonexpansive,

$$
\|\mathcal T(x)-\mathcal T(y)\|_2
\le\|(I-\eta Q)(x-y)\|_2
\le q\|x-y\|_2,
$$

where \(q=\max_i|1-\eta\lambda_i(Q)|<1\).

**Proof.** The first inequality is nonexpansivity of projection. The second is the spectral norm of a symmetric matrix. The contraction mapping theorem yields a unique fixed point and geometric convergence to the unique constrained minimizer. **QED.**

**Boundary of the proof:** obtaining a unique numerical fixed point proves only numerical optimization, not signal truth. If nonlinear GNSS range equations are linearized, their Taylor remainder must be bounded and included in the physical envelope; otherwise this proof cannot be transferred to real measurements. CGPE/MFR's complex RF gain/beam constraints cannot substitute for a trusted GNSS position/time reference.

## 8. Theorem 6 — Finite certified horizon during GNSS loss or jamming

Let \(m>0\) be a justified safe **uncertainty margin** (after separately accounting for maps, obstacles, winds, vehicle dynamics and control errors). Only if

$$
E_p(T)\le m
$$

can the inertial position tube fit inside that margin. Define

$$
T_{\rm cert}=\sup\{T\ge0:E_p(T)\le m\}.
$$

When a nonzero error-growth coefficient exists, \(E_p(T)\) grows without bound, so \(T_{\rm cert}\) is finite for finite \(m\).

**Proof.** By Theorem 3, any point in the radius-\(E_p\) ball is consistent with the assumptions. Once that ball exceeds the predeclared margin, the information no longer proves all possible states lie in the certified region. Polynomial growth yields finite \(T_{\rm cert}\) for nonzero positive growth coefficient. **QED.**

This is a **loss-of-guarantee condition** only. It does not establish that a particular UAV maneuver, return route or landing procedure is safe, or authorize control of a vehicle.

## 9. Alert timing: the strongest valid conditional statement

Suppose a perturbation remains outside \(2B_i(T)\) from time \(T_\star\), accepted sensor reports arrive with gap at most \(\Delta\), and measurement/decision delay is bounded by \(L_d\). Then the first valid subsequent report must flag physical inconsistency no later than \(T_\star+\Delta+L_d\).

Without bounded sampling gaps, continued sensor availability, trustworthy clocks or finite error envelopes, **there is no guaranteed time-to-alert**. A sufficiently gradual perturbation can stay inside an expanding error tube indefinitely. Full common-mode movement is unobservable under the original clock-comparison model (Theorem 1).

## 10. Additional formal hazard: repeated values are not replay proof

The current v10.36 research monitor marks a witness as replayed after repeated identical numeric values. Equal consecutive readings can also arise honestly from finite quantization or a steady physical quantity. Thus identical samples alone cannot logically prove an adversarial replay. Verified sequence counters and timestamps bound to a trusted sensor identity are needed for replay **provenance**; even these do not prove that a sensor's underlying physics is uncompromised.

## 11. Proposed advisory-only mathematical interface

The formal extension should be a distinct **PHYSICS_CERTIFICATE**, not an extension of the existing alarm vote. It needs:

1. **Anchor certificate:** independent initial position/velocity source, acquisition time, confidence *bounds* and dependencies.
2. **Inertial envelope:** \(e_{p0},e_{v0},e_a,e_g,e_{\theta0},e_\omega,F_{\max}\), time validity, \(E_p(T)\).
3. **Clock envelope:** independent time source, expiry and \(E_c(T)\); explicitly detect GNSS-disciplined-clock dependency.
4. **GNSS envelope:** per-satellite synchronized code/rate observations and all corrected measurement-error bounds.
5. **Rank/dependency certificate:** expose that RAW and REFERENCE have one common GNSS clock, and CORRECTED is not an independent third anchor.
6. **Decision certificate:** **PHYSICAL_INCONSISTENCY_UNATTRIBUTED**, **CONSISTENT_UNDER_MODEL_NOT_AUTHENTICATED**, or **INSUFFICIENT_EVIDENCE**, with assumptions, timestamp and residual witness.
7. **Position uncertainty set and expiry:** finite only when every physical assumption is valid.
8. **Authority separation:** advisory-only, no automatic GNSS authentication, no actuator or navigation-control authority.

Precedence: a failed independent trust/clock/anchor condition produces **INSUFFICIENT_EVIDENCE** regardless of CUSUM votes. Retain existing ANY_CHANNEL as early warning and NEW_CROSS as a correlated comparison. Neither should promote a safety certificate.

## 12. What mathematics cannot replace

Theorems 1–2 are unconditional **limitations of the specified measurement models**. Theorems 3–6 are mathematically correct **only if their explicitly stated physical error, anchor, clock and model assumptions hold**. A mathematical derivation cannot by itself establish real sensor independence, true deterministic temperature/vibration bounds, correct antennas and timebase, worst-case environmental errors, aviation integrity risk, or safe flight-control outcomes. No deployment, field validation or flight safety claim is made.

**Status:** mathematical research specification complete; standalone advisory proposal. Frozen v10.26, v10.36 and RC3 inference paths remain unchanged.

## References

- Project-owned archives reviewed for design: SANAD-X v10.33 causal observer, v10.35 fusion policy, v10.36 causal monitor; SANAD Network RC3 README; SANAD CGPE/MFR mathematical model.
- European Space Agency, [GNSS Basic Observables](https://gssc.esa.int/navipedia/index.php/GNSS_Basic_Observables).
- European Space Agency, [GNSS Measurements Modelling](https://gssc.esa.int/navipedia/index.php?title=GNSS_Measurements_Modelling).
- Rothmaier et al., [Providing Continuity and Integrity in the Presence of GNSS Spoofing](https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Rothmaier_IONGNSS_2021_SpoofingIntegrity.pdf) — threat-model-specific prior work, not validation of SANAD-X.

**Repository rule:** Keep the note in an isolated research branch. Do not merge it into frozen RC3's qualified inference, or market it as verified UAV spoofing protection.