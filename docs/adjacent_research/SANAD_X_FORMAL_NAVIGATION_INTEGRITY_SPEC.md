# SANAD-X: Physics-Constrained Navigation Integrity — Formal Research Specification v1

> **Status: mathematical research proposal, not an operational release, an anti-spoofing certificate, or flight-control software.** This is an adjacent SANAD-X design note. It does **not** modify or qualify the frozen SANAD Network RC3 model, SANAD-X v10.26 stable baseline, or SANAD-X v10.36 research implementation. No tests, simulations, flight trials, or physical measurements are claimed here.

## 0. Scope and relation to existing SANAD components

The archived SANAD-X v10.36 implements three causal timing comparisons (`RAW=G-A`, `REFERENCE=G-B`, `CORRECTED=G-A-W`), online CUSUM scores and the `ANY_CHANNEL` / `NEW_CROSS` decision hierarchy. Its trust flags are external assertions, not evidence of physical independence. This is a useful **anomaly monitor**, not an authenticated GNSS receiver and not a certified independent navigation solution.

The original RC3 repository is a **7 arrays x 8 antennas / 300 reports / 15-minute-report calibration-integrity classifier** trained on Sionna evidence. The associated CGPE/MFR work performs independent-reference RF-chain estimation and constrained beam recovery. None of these capabilities directly authenticate satellite signal origin, grant usable real-time UAV position accuracy, or prove the vehicle safe to fly. This document defines a separately deployable, advisory-only **physical integrity certificate** with explicit assumptions.

**Claim structure:** deterministic implications conditional on calibrated sensor/error bounds; not claims of universal spoofing detection or attacker identification.

## 1. State, physics, and assumptions

For an Earth-fixed/local navigation model, use state
[
x(t)=(p(t),v(t),R(t),b_a(t),b_omega(t),b_c(t),d_c(t)),
]
where (p,vinmathbb R^3), (Rin SO(3)), (b_a,b_omega) are inertial biases, (b_c) is receiver-clock offset in seconds, and (d_c) is receiver-clock drift. Use properly modeled local gravity, Earth rotation, lever-arm correction and sensor timestamps.

The strapdown equations, before bounded errors, are
[
dot p=v,quad
dot v=R(f_m-b_a-n_a)+g(p),quad
dot R=R[omega_m-b_omega-n_omega]_	imes,quad
dot b_c=d_c.
]
These describe actual **proper acceleration and angular rate**, not absolute location. Without an independent absolute reference, the inertial position uncertainty grows.

For satellite (i) with ephemeris (s_i(t)), a corrected code pseudorange model is
[
ho_i=|p-s_i(t_i^{tx})|
+c(b_c-delta t_i)+I_i+T_i+S_i+H_i+M_i+eta_i+a_i.
]
(c) is light speed; (I,T,S,H,M) represent ionosphere, troposphere, Earth-rotation/Sagnac or relativistic modeling corrections as applicable, receiver/satellite hardware group delays, and multipath; (eta) is measurement error; (a_i) is an **unspecified disturbance or adversarial input**. Doppler/pseudorange rate has a corresponding relative line-of-sight velocity and clock-drift term. An RF transmitter's true location is **not** determined solely by the apparent satellite ephemeris.

Necessary assumptions for a deterministic certificate:

- **A1. Clean anchor:** at (t_0), an independently established position and velocity satisfy (|p_0-hat p_0|le e_{p0}), (|v_0-hat v_0|le e_{v0}). A potentially spoofed GNSS solution is not a clean anchor.
- **A2. Bounded inertial errors:** over a finite horizon, effective accelerometer/gravity residual (le e_a+e_g), orientation error (le e_{	heta0}+e_omega t), and specific-force magnitude (le F_{max}); all are **physical upper bounds**, not mere sample standard deviations.
- **A3. Clock independence:** a separately trusted monotonic oscillator or other timing anchor provides a finite bound (|b_c-hat b_c|le E_c(t)). A clock re-disciplined from the suspect GNSS stream fails this assumption.
- **A4. Measurement envelopes:** ephemeris, atmosphere, multipath, lever-arm, hardware, latency and sensing errors have declared finite bounds; stale/missing calibration voids the certificate.
- **A5. Trust/provenance:** timestamps, origin and integrity of each independent measurement are established by an actual trust root. A Boolean `trusted=True`, checksum, or MAC alone cannot prove physical sensor truth.
- **A6. Separation:** the proposed independent anchor is not secretly computed from the same untrusted GNSS inputs or shared clock. Common-mode dependencies must be modeled.

If any necessary assumption fails, the mathematical conclusion is **INSUFFICIENT_EVIDENCE / ABSTAIN**, not `SAFE`.

## 2. Theorem I — Why SANAD-X timing differences cannot prove satellite authenticity

Let latent clocks be (z=[G,A,B]^T). The two raw comparisons are
[
y=Hz,qquad H=egin{bmatrix}1&-1&0\\1&0&-1end{bmatrix}.
]
(H) has rank 2, and
[
H,[1,1,1]^T=0.
]

**Proof.** The two rows are linearly independent, so (operatorname{rank}H=2). Their common null vector is ((1,1,1)^T). Hence for any scalar time function (q(t)), (H(z+q(t)mathbf1)=Hz). Arbitrary common-mode variation of all three clocks is information-theoretically invisible to these comparisons. □

The archived corrected comparison is (y_C=(G-A)-W=y_R-W). It is algebraically derived from the raw comparison plus a nuisance witness; it is **not an additional independent GNSS-orbit observation**. A multichannel alarm can be stronger evidence of *inconsistency* under a chosen model, but a `NEW_CROSS` vote cannot make the nullspace disappear.

**Design consequence:** retain `ANY_CHANNEL` as advisory early warning, `NEW_CROSS` as correlated cross-check, and use **physically independent** absolute/time/motion evidence for a separate integrity certificate. A third channel helps only if its measurement model adds independent rank and its independence is established.

## 3. Theorem II — Satellite-orbit plausibility is not origin authentication

For any candidate position (p') and clock bias (b'), one may form mathematically self-consistent pseudoranges (ho_i'=|p'-s_i|+cb'+d_i), with admissible modeled corrections (d_i). A standard navigation solution can then satisfy the orbital equations even when the measurements were not emitted by those satellites.

**Proof.** Substitution of ((p',b')) into the pseudorange model yields the same set of algebraic constraints the receiver solves. The equations describe apparent delays, not a cryptographic or geometrically independently measured transmitter-origin witness. Therefore orbital ellipses, virtual tangent points, nominal time-of-flight or PRN consistency alone do not logically imply authentic satellite origin. □

Adding real independent angle-of-arrival, authenticated signal features, independent motion or trusted local time can shrink admissible alternatives; none is assumed available without hardware evidence. A GNSS-only mathematical consistency proof cannot provide an unconditional anti-spoofing proof.

**Dimensional check:** a (1 mathrm{ns}) time bias produces (c	imes10^{-9}approx0.299792458 mathrm m) apparent range bias. Timing error envelopes must therefore be specified before claiming metre-level integrity.

## 4. Theorem III — Bounded inertial dead-reckoning tube

Define elapsed time (T=t-t_0). Under A1–A2 and a bound on unmodeled inertial/gravity acceleration, the acceleration prediction error satisfies
[
|delta a(T)|le A_0+A_1 T,quad
A_0=e_a+e_g+F_{max}e_{	heta0},quad
A_1=F_{max}e_omega.
]
The rotation contribution follows from (|R-hat R|_2=2sin(	heta/2)le	heta), for relative rotation angle (	hetage0) in its principal range. Thus
[
E_v(T)=e_{v0}+A_0 T+	frac12 A_1T^2
]
and
[
oxed{E_p(T)=e_{p0}+e_{v0}T+	frac12 A_0T^2+	frac16 A_1T^3.}
]

**Proof.** With (dot e_v=delta a), the integral triangle inequality gives (|e_v(T)|le e_{v0}+int_0^T(A_0+A_1u),du=E_v(T)). Integrating (dot e_p=e_v) once more gives the displayed (E_p(T)). □

The **physically guaranteed** position set (only under A1–A2) is
[
mathcal X_I(T)={p:|p-hat p_I(T)|le E_p(T)}.
]
Because (E_p) grows quadratically/cubically, bounded IMU data do **not** justify unlimited GNSS-denied navigation. If any bound or anchor is untrusted, use (mathcal X_I=mathbb R^3), not an unjustified small tube.

## 5. Theorem IV — Physics-consistency gate with a deterministic detection condition

Let (hatho_i(T)) be the correctly modeled pseudorange predicted from independently propagated ((hat p_I,hat b_c)). Geometric range obeys the reverse triangle inequality:
[
ig||p-s_i|-|hat p_I-s_i|ig|le|p-hat p_I|le E_p(T).
]
Let (U_i(T)) upper-bound *all remaining* benign errors in metres, including independent clock (cE_c(T)), satellite/orbit, atmosphere, multipath, hardware and latency. Set
[
B_i(T)=E_p(T)+U_i(T),qquad r_i(T)=ho_i^{obs}(T)-hatho_i(T).
]
Define (	extsf{CONSISTENT}) only when every required measurement has valid provenance and (|r_i|le B_i).

**Soundness under assumptions.** If the observation is benign and A1–A6 and the (U_i) envelopes hold, then (|r_i|le B_i). Thus this deterministic gate cannot falsely reject a genuinely benign sample *within its complete verified model*.

**Guaranteed inconsistency condition.** Suppose (ho_i^{obs}=ho_i^{benign}+a_i) and the benign residual obeys (|ho_i^{benign}-hatho_i|le B_i). Then
[
|r_i|ge |a_i|-B_i.
]
Therefore
[
oxed{|a_i(T)|>2B_i(T) Longrightarrow |r_i(T)|>B_i(T),}
]
so the gate **must** report physical inconsistency for that channel at that accepted observation. □

The certificate is **not** a proof that every attack is caught: disturbances (|a_i|le2B_i), coordinated sensor compromise, common clock corruption, or violated envelopes may remain invisible. Nor does the contradiction identify whether the cause was spoofing, hardware fault, unknown multipath or an inaccurate bound.

An alternative stricter set-based formulation is
[
mathcal F(T)={(p,b):pinmathcal X_I(T),, |b-hat b_c|le E_c(T),,
|ho_i^{obs}-h_i(p,b)|le U_i^{nonclock} orall i}.
]
If (mathcal F(T)=arnothing), no **single** state satisfies all assumed physics; the verdict is `PHYSICAL_INCONSISTENCY_UNATTRIBUTED`. Nonempty (mathcal F) means only `CONSISTENT_UNDER_MODEL_NOT_AUTHENTICATED`. Do not replace emptiness proof with an unconverged nonlinear solver's `false`.

## 6. Theorem V — Exact role of fixed-point mathematics (MFR/HOK conceptual bridge)

For a **local convex surrogate**, not the original nonconvex GNSS measurement manifold, let
[
J(z)=rac12|W(Az-y)|_2^2+rac{lambda}{2}|z-z_I|_2^2,quad lambda>0,
]
on nonempty closed convex (mathcal C). Write (Q=A^TW^TWA+lambda I), whose eigenvalues satisfy (0<mulelambda_i(Q)le L). Define
[
mathcal T(z)=P_{mathcal C}(z-eta
abla J(z)),qquad 0<eta<2/L.
]
Then
[
|mathcal T(x)-mathcal T(y)|
lemax_i|1-etalambda_i(Q)|,|x-y|
=q|x-y|,quad q<1.
]

**Proof.** The metric projection onto a nonempty closed convex set is nonexpansive. The gradient difference equals (Q(x-y)), and the spectral norm of (I-eta Q) is (q<1) for the specified step. By the Banach fixed-point theorem, (mathcal T) has a unique fixed point and its iterations converge geometrically to the unique constrained minimizer. □

**Critical boundary:** solver convergence proves that a **mathematical optimization problem was solved**, not that the measurements represent genuine satellites. If the physical measurement map (h(z)) is linearized, the Taylor remainder must have an independently certified bound and be included in (U_i); otherwise the guarantee does not transfer to the actual nonlinear system. RC3 probabilities, CGPE gain estimates and MFR beam proposals must not be substituted as authenticated position/time witnesses.

## 7. Theorem VI — Finite guaranteed navigation-information horizon

Suppose a civil mission has a justified position uncertainty tolerance (m>0) (e.g., distance to a known and independently validated unsafe-region boundary, after accounting for obstacle-map and control uncertainties). If the state tube is used as evidence, a **necessary information condition** for continuing to claim the entire position set lies within that tolerance is
[
E_p(T)le m.
]
Define (T_{mathrm{cert}}=sup{Tge0:E_p(T)le m}); when (A_0>0), (A_1>0), or (e_{v0}>0), this horizon is finite for finite (m).

**Proof.** Every physically consistent true position may lie anywhere in (mathcal X_I(T)). If the whole tube no longer fits within the certified tolerance, the evidence does not imply safety. Growth of (E_p(T)) follows from Theorem III. □

This is a **loss-of-guarantee rule**, not a guarantee of a safe maneuver or landing. A robust reachability/safety supervisor needs independent maps, wind/disturbance bounds, vehicle constraints, health monitoring and separately certified control; SANAD-X itself should have no direct actuator authority.

## 8. Latency and the impossibility of universal protection

If trustworthy samples arrive with inter-arrival gap at most (Delta), a disturbance remains above (2B_i(T)) after a time (T_*), and decision computation/communication has bounded delay (L_d), the next accepted sample will report inconsistency by (T_*+Delta+L_d). This is conditional on continued sensing and bounded delays. No such bound exists after total RF/clock/provenance loss.

Conversely, a slowly evolving disturbance can remain inside the expanding inertial/clock tube; mathematical consistency alone cannot guarantee eventual detection. Likewise a full common-mode perturbation of (G,A,B) is invisible to their differences (Theorem I). **No universal “protected UAV against all spoofing” theorem is possible from the v10.36 inputs.**

## 9. Proposed SANAD-X advisory-only interface

Keep archived v10.36 timing logic intact. Add a *separate* `PHYSICS_CERTIFICATE` record containing:

1. `anchor_provenance`: time, independent position/velocity origin, physical trust root, expiry and dependency graph.
2. `inertial_envelope`: (e_{p0},e_{v0},e_a,e_g,e_{	heta0},e_omega,F_{max}), calibration validity, and (E_p(T)) with units.
3. `clock_envelope`: independent timebase identity, (E_c(T)), drift and tamper/staleness status.
4. `gnss_envelope`: per-satellite corrected observation, synchronized epoch, orbital/atmospheric/receiver uncertainty budget and (B_i(T)).
5. `observability`: measurement rank/dependency report; never count `RAW`, `CORRECTED`, `REFERENCE` as three independent anchors.
6. `integrity_verdict`: `PHYSICAL_INCONSISTENCY_UNATTRIBUTED`, `CONSISTENT_UNDER_MODEL_NOT_AUTHENTICATED` or `INSUFFICIENT_EVIDENCE`, with earliest causal timestamp and the exact assumptions supporting the decision.
7. `position_set` and `certificate_expiry`: finite set only under verified assumptions; otherwise `unknown`.
8. `advisory_only=true`, `navigation_authority=false`, and `actuator_authority=false`.

**Causal composition:** missing, stale, replayed, nonmonotonic or compromised provenance has precedence over comparison scores. `ANY_CHANNEL` and `NEW_CROSS` remain attributable only to the archived timing monitor. Physical certificate failures do not label an attacker; they indicate conditional infeasibility. Never silently reset, use future samples, or promote an advisory into an autonomous flight command.

## 10. Mathematical proof obligations not yet discharged by existing hardware

| Obligation | Why existing files do not prove it |
|---|---|
| Independent clean pose and time anchor | v10.36 receives timing differences and asserted trust flags only |
| Uniform finite IMU angular/acceleration error bounds | Must be certified for actual temperature, vibration, ageing, sensor saturation and fault modes |
| Full GNSS uncertainty budget including independent oscillator | Satellite geometry alone is not a trustworthy clock or RF origin check |
| Physically independent witness | Three algebraically related timing channels do not establish independent truth |
| Correlated failures and common-mode effects | Unobservable under the two-difference clock model |
| End-to-end latency and fail-safe civil-UAV behavior | Not established by a symbolic proof of a residual inequality |
| Real-world integrity-risk probability | Cannot be inferred from deterministic inequalities or past synthetic detection percentages |

**Conclusion.** The theorems *prove conditional bounded-error inconsistency detection and numerical fixed-point convergence*. They do **not** prove GNSS signal authenticity, adversary attribution, universal attack detection, continuity of accurate navigation after indefinite outage, or aircraft safety. Those claims require additional independent physical instruments and their separate qualification. No experiment or flight test has been run as part of this mathematical document.

## References and provenance

- Project baseline: archived SANAD-X v10.33 causal observer, v10.35 fusion policy and v10.36 causal research monitor (project Drive research registry, reviewed 21 Nov 2026).
- SANAD Network RC3 [README](../../README.md): RF calibration classifier and explicit research boundaries.
- SANAD MFR/CGPE: independent-reference complex gain/phase estimation and constrained projection; distinct from GNSS position integrity.
- ESA Navipedia: [GNSS Basic Observables](https://gssc.esa.int/navipedia/index.php/GNSS_Basic_Observables) and [GNSS Measurements Modelling](https://gssc.esa.int/navipedia/index.php?title=GNSS_Measurements_Modelling).
- Prior literature motivation, not an assertion of our hardware qualification: [Providing Continuity and Integrity in the Presence of GNSS Spoofing](https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Rothmaier_IONGNSS_2021_SpoofingIntegrity.pdf).

**Review rule:** keep the entire proposal in an isolated research branch; do not merge into the frozen calibrated RC3 inference path or claim a SANAD-X new released version.
