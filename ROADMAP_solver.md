# WaferKinetic solver roadmap

Self-contained: every equation this plan relies on is written out here.
Companion file: `PLAN_comsol_reproduction.md` holds the benchmark targets
and the COMSOL-setup comparison table.

**Product goal.** Many independent ICP simulations batched on a single
GPU (`vmap` over operating points on a fixed mesh), with large reaction
sets (many excited states), eventually Monte Carlo electrons — faster
than HPEM per converged case, to generate training data for AI models.
Architecturally the target is a differentiable, batched HPEM: segregated
modules coupled through re-entry thresholds, with a Newton–Krylov core
for the tightly coupled charged-particle/field block.

Two throughput metrics govern every decision below, in this order:

1. converged cases per GPU-hour across a batch (the product metric);
2. time-to-steady-state for one warm-started case (the sweep metric).

---

## 0. State of the solver (validated baseline)

### 0.1 What is implemented and verified

All-fluid 2D axisymmetric argon ICP on a structured finite-volume mesh:

* electron continuity and mean-energy equations, backward-Euler implicit
  in transport ("P3"), Gummel-coupled to a semi-implicit Poisson solve;
* one drift-diffusion ion, explicit with a donor-cell positivity limiter;
* frequency-domain electromagnetic solve (EMM) with power control;
* 7-reaction argon chemistry (ground, one lumped excited state, one ion);
* dielectric surface charge on plasma–dielectric faces;
* analytic sheath model (ASM) behind `sheath_model="asm"`;
* pseudo-transient outer iteration with per-module re-entry thresholds.

Verified against the resolved-sheath baseline and internal gates:
bit-reproducible 800-iteration regression trajectory; implicit path
consistent with the explicit reference; power ledger closing to <1%;
exact positivity; reverse-mode gradient matching finite differences;
ASM grid-convergence pair (0.278 mm vs 2.108 mm wafer cell) agreeing to
a few percent in every bulk and sheath quantity.

### 0.2 The model equations as currently solved

Species densities n (1/m^3), fluxes Gamma (1/m^2 s), Te in eV, SI
otherwise; e is the elementary charge.

Electron continuity and drift-diffusion flux:

    d(ne)/dt + div(Gamma_e) = S_e
    Gamma_e = -mu_e ne E - grad(D_e ne),      D_e = mu_e Te

Electron mean-energy density n_eps (eV/m^3), with Te = (2/3) n_eps/ne:

    d(n_eps)/dt + div(Gamma_eps) = -Gamma_e . E
                                   - sum_j dE_j r_j
                                   - 3 (me/M) nu_m ne (Te - Tg)
    Gamma_eps = (5/3) [ -mu_e n_eps E - grad(D_e n_eps) ]

The electrostatic Joule term is the work on the ACTUAL discrete flux,
-Gamma_e.E (restricted to interior faces). The pure-drift form
mu_e ne |E|^2 keeps only the positive-definite half of a nearly
cancelling drift/diffusion pair and overstates the heating by orders of
magnitude in an ambipolar plasma; this was measured, and the flux form
is the working configuration.

Ion continuity (drift-diffusion, constant mobility for now):

    d(ni)/dt + div(Gamma_i) = S_i
    Gamma_i = +mu_i ni E - grad(D_i ni)

Electrostatics with dielectric surface charge sigma_s on
plasma–dielectric faces:

    -div(eps grad Phi) = e (sum_k Z_k n_k - ne) + rho_s
    d(sigma_s)/dt = e (Gamma_i - Gamma_e) . n_hat

Semi-implicit Poisson ("P2"): the charge densities on the RHS are
predicted with drift fluxes evaluated at the future potential; because
every charged species' drift contributes with the same sign to the net
current, those terms reform a Laplacian and the operator becomes

    -div[ (eps + dt sigma_DC) grad Phi^{n+1} ] = predicted charge + ...
    sigma_DC = e (mu_e ne + mu_i ni)

implemented in delta form so that at a stationary point the augmentation
cancels exactly and Phi satisfies the UNAUGMENTED equation with the
converged charge. Time steps exceed the dielectric relaxation time
eps0/sigma_DC by factors of 1e4 or more.

Electromagnetics (frequency domain, azimuthal A):

    curl( mu0^-1 curl A ) + j omega sigma_p A = J_coil
    sigma_p = ne e^2 / [ me (nu_m + j omega) ]
    Q_ind = (1/2) Re(sigma_p) |E_theta|^2,   E_theta = -j omega A

with the coil current rescaled every outer iteration so the deposited
power integrates to the setpoint.

Analytic sheath model (ASM), the working wall closure — the mesh's last
plasma cell is the sheath edge and the entire sheath drop lives outside
the mesh:

* ion wall flux: the Bohm criterion, no thermal term, no drift term:

      Gamma_i . n_hat = ni u_B,        u_B = sqrt(e Te / M)

* electron wall flux: one-sided thermal efflux throttled by the
  Boltzmann factor of the MESH potential drop across the wall face
  (dynamic — this is the discharge's charge regulator; a frozen
  analytic multiplier removes the feedback and the state drifts
  charge-inconsistent until it detonates, which was observed):

      Gamma_e . n_hat = c_p vbar_e ne exp(-dPhi_face/Te)
      c_p = (1/2)(1 - r_e)/(1 + r_e),  vbar_e = sqrt(8 e Te / (pi me))

* the equilibrium of that throttle is pinned by a battery jump in the
  Poisson equation at every plasma-boundary face,

      -div[ eps grad(Phi + dPhi_b) ] = rho + rho_s
      dPhi_b = chi Te,   chi = ln( c_p vbar_e / u_B )
                             = ln( c_p sqrt(8 M / (pi me)) )

  (chi = 4.97 for argon at r_e = 0.2; both speeds scale as sqrt(Te), so
  chi is a constant of the gas and r_e alone). Implemented as an
  antisymmetric RHS term so the operator stays SPD. At zero net current
  the face drop relaxes to exactly chi Te.

* electron ENERGY wall flux carries the climb term: escaping electrons
  remove their sheath-edge kinetic energy, flux-averaged (2 + x) Te at
  barrier x = dPhi_face/Te, so the thermal energy flux
  (5/6)(1-r_e)/(1+r_e) vbar_e n_eps is multiplied by

      exp(-x) (1 + 2x/5)

  Without the climb term the battery does ~P_ion_wall of free work and
  the electron fluid runs hot (observed: +44% ne).

* the ion sheath+presheath energy is reported as a ledger channel:

      P_ion_wall = sum_faces A e Gamma_i (dPhi_face + Te/2)

* self-consistency: the Child–Langmuir thickness must fit inside the
  wall cell,

      s = (sqrt(2)/3) lambda_D (2 dPhi_b/Te)^{3/4} < dz_wall
      lambda_D = sqrt(eps0 Te / (e ne))

  Coarsening the wall cells IMPROVES this margin (s/dz 0.14 -> 0.02).

### 0.3 Measured numerical limits (what binds the clock today)

Two explicit stability limits remain, and which one binds depends on the
wall-cell size:

* ion Courant on the finest wall-normal cells,
  dt < CFL dz / (mu_i |E| + D_i/dz) — 1.5e-10 s on the 0.278 mm wafer
  row of the fine mesh, and that single row sets the whole clock (next
  binder 29x away);
* explicit surface-charge relaxation at DIELECTRIC faces. Linearising
  the face charge against its own potential response
  (dPhi/dsigma = A/g with g the face conductance, so the permittivity
  counts) gives the relaxation rate

      lambda = e Gamma_e A / (g Te),     stability lambda dt < 2

  This rate GROWS with cell size (a thicker capacitor gap is more volts
  per coulomb) — the opposite of ion Courant — so coarsening the wall
  trades one limit for the other. Measured: the coarse-wall mesh gains
  1.9x, not the 8x the ion-Courant headroom alone would give.

The Phi<->rho density prediction is NOT what binds (measured by cap
bisection); a short-history ion flux extrapolation would buy nothing and
is dropped from the plan.

Chemical stiffness caps (dt < safety/nu_reaction) do not bind for the
7-reaction argon set but WILL dominate for large sets — see Stage 2.

### 0.4 Standing constraints and guards (do not regress)

* Krylov warm-start breakdown guards: accept a solver's answer only if
  its residual is <= the warm start's (breakdowns return NaN OR finite
  garbage; isfinite alone certified 8e77 once).
* Physics-bounded divergence guard: species <= 10 n_g, n_eps <= 1e3 n_g,
  |Phi| <= 10 kV; return the last finite state.
* dt = 0 must mean "do not advance": a zero sink-cap (n_eps decayed to
  exactly zero) must become an INACTIVE cap, and the implicit solve's
  1/dt must go to 1e30 at dt = 0, not 1.
* Guarded divisions use the double-where idiom or reverse-mode AD forms
  0 * inf = NaN in the unselected branch.
* Hard clips (Te) have identically zero gradient — replace with soft
  saturation before any optimization use.
* cg_tol on the Poisson solve stays at 1e-8: Phi rides on ppm-level
  departures from quasineutrality while the RHS norm is dominated by the
  augmentation term, so loosening to 1e-5 destroys ignition (measured).
  im_tol 1e-6 and n_gummel 1 are safe (2.7x, verified full-trajectory).
* `uniform_dt` currently collapses only the CEILINGS; the in-loop clips
  re-introduce per-cell dt. Either honour it globally or rename it —
  local dt gradients are themselves destabilising (observed).
* Regression protocol: the pseudo-transient trajectory is bit-reproducible
  on CPU; regressions diff the full iteration trajectory, never a
  single snapshot.
* Explicit stepper body stays byte-identical; every new physics/numerics
  switch is one flag, default unchanged, threaded through stepper,
  clips, ledger, NumPy reference and field dump at once.

---

## Stage 1 — Benchmark anchor: the COMSOL argon GEC case

Cheap, first, and the credibility anchor for everything after. Targets
and the full setup comparison live in `PLAN_comsol_reproduction.md`
(peak ne 1.5e18 m^-3, Te 3.85 eV, Phi 19–20 V, Ar* 8.5e18 m^-3 at
20 mTorr, 1500 W coil / ~1200 W plasma).

1.1 Operating point: 20 mTorr, 1200 W deposited. Parameters only.

1.2 Constants: electron momentum-transfer rate k_mom = 4.40e-14 m^3/s
    (from COMSOL's reduced mobility mu_e N = 4e24 1/(m V s); ours is
    2.3x low) and metastable quenching k_q = 3.0e-21 m^3/s (1.43x low).

1.3 COMSOL-compatible wall mode, one flag: r_e = 0 (electron wall flux
    (1/4) vbar_e ne instead of (1/6)) and ion wall flux without the
    thermal term (migration-only plus surface reaction).

1.4 Convergence declared on the residual, not on the iterate change:
    the change-based criterion floors at solver jitter (~3e-4) and no
    run can honestly be called converged today. Define

        res_k = || R(u_k) || / || R(u_seed) ||

    with R the steady residual of Stage 6.1 (its assembly is useful long
    before Newton) and converge on res_k < tol.

1.5 Maxwellian rate tables from Phelps cross sections (LXCat):

        k_j(Te) = gamma Int_0^inf  eps sigma_j(eps) f_M(eps; Te) d(eps)
        gamma = sqrt(2 e / me)
        f_M(eps; Te) = 2 sqrt(eps/pi) Tbar^{-3/2} exp(-eps/Tbar),
        Tbar = (2/3)<eps> = Te

    tabulated on a log-Te grid at build time, evaluated by interpolation
    at run time. Same source data as COMSOL, so the remaining physics
    difference collapses to closure and discretisation. Also supplies
    the elastic momentum-transfer frequency properly (superseding 1.2's
    patch) and re-tests the "fluid tier is 2x high in ne" claim under
    matched inputs.

1.6 Comparison harness: one command reporting peaks, line cuts, power
    split, skin depth against stored targets. Grid-convergence
    (fine vs coarse wall) as a standing regression, not a one-off.

New diagnostics this stage (no new solver physics): coil circuit
quantities, since they are the cheaply-measurable validation numbers —

    R_coil = 2 P_coil / |I|^2,     L_coil = 2 W_mag / |I|^2

with and without plasma, and the coil/plasma power split vs time.
Comparing R(t) requires the time-accurate mode of Stage 7.4.

---

## Stage 2 — Chemistry restructure (and OUT of the future Newton block)

The scaling axis for large reaction sets is species-local, not spatial.
Chemistry must become (a) data instead of program, (b) implicit per
cell, (c) permanently outside the global Newton unknown vector. A
40-species set changes NOTHING in Stage 6 if this stage is done first.

2.1 Matrix-form chemistry. Reactions become arrays, not unrolled code.
    With reactant matrix nu'_ij, product matrix nu''_ij, and rate
    constants k_j(Te):

        r_j = k_j(Te) prod_i n_i^{nu'_ij}
            = k_j(Te) exp( sum_i nu'_ij ln n_i )
        S_i = sum_j (nu''_ij - nu'_ij) r_j
        S_eps = - sum_j dE_j r_j

    i.e. one matrix–vector product in log space and one against the net
    stoichiometry. Compile time and program size become independent of
    the reaction count; a 300-reaction set is a tensor shape, and the
    contraction is dense GPU work. Elastic collisions contribute only

        S_eps_el = -3 (me/M_k) nu_mk ne (Te - Tg_eV)   per partner k.

2.2 Frozen heavy densities as runtime arguments of the jitted steppers
    (today they are baked constants and every refresh retraces the whole
    stepper, ~5.5 s for 7 reactions; minutes for hundreds). Mechanism
    already validated: closure-name rebinding, one trace across changing
    inputs, bit-identical values.

2.3 Per-cell implicit chemistry integration. For stiff sets the explicit
    source treatment (dt < safety/nu) collapses; the remedy is a local
    backward-Euler (or Rosenbrock) solve per cell over the transport
    step dt:

        [ I/dt - J_chem(n*, Te) ] delta_n = S(n*, Te)
        J_chem = dS/dn   (N_s x N_s, dense, analytic from 2.1:
                          dr_j/dn_i = nu'_ij r_j / n_i)

    Batched N_s x N_s LU factorisations across cells and across sims are
    exactly what GPUs do best; no spatial coupling. Positivity follows
    from the M-matrix structure for nonnegative production or is
    enforced by a damped update. Coupling to transport by first-order
    (or Strang) splitting, with the steady-state consistency requirement
    that the SPLIT scheme's fixed point satisfies transport + chemistry
    simultaneously — enforced by evaluating the Stage 6 residual with
    both terms together.

2.4 Species registry: per-species charge, mass, transport data (mobility
    or LJ parameters), wall sticking coefficients and return species
    (e.g. Ars -> Ar with gamma = 1), lumped excited manifolds with their
    internal energies. The surface chemistry generalises the wall flux:

        Gamma_k,wall = gamma_k / (1 - gamma_k/2) (vbar_k/4) n_k

    (Chantry/Motz–Wise), with the returned mass credited to the return
    species.

Deliverables: identical results on the 7-reaction argon set (regression:
bit-level vs the loop implementation), then a demonstration set with
resolved Ar(4s)/Ar(4p) manifolds (~10 species, ~50 reactions) running at
unchanged per-iteration cost.

---

## Stage 3 — Neutral gas transport at high and low pressure

New physics. Today the background gas is a fixed-density, fixed-
temperature bath with one diffusing excited species; that is only valid
at low power and low pressure. Two regimes must be covered, selected by
the Knudsen number Kn = lambda_mfp / L.

3.1 Gas continuity + momentum (laminar compressible, steady or
    transient), for the continuum regime (Kn < 0.01, i.e. roughly
    p > 100 mTorr for L ~ cm, and any case with inflow/outflow):

        d(rho)/dt + div(rho u) = 0
        rho [ du/dt + (u.grad) u ] = -grad p + div(tau) + F_ion
        tau = mu_visc [ grad u + (grad u)^T - (2/3)(div u) I ]
        F_ion = e n_i E - (momentum returned by ion-neutral collisions)

    with ideal-gas closure p = n_g kB Tg. Inlet mass-flow / outlet
    pressure boundary conditions bring gas residence time and flow
    convection of species into the model (needed for any real recipe).

3.2 Gas energy (heating matters above a few hundred watts; rarefaction
    n_g = p/(kB Tg) then feeds back on E/N and every rate):

        rho c_p [ dTg/dt + u.grad Tg ] = div(kappa grad Tg) + Q_gas
        Q_gas = 3 (me/M) nu_m ne kB(Te - Tg)e-terms   (elastic transfer)
              + Gamma_i . E e                        (in-mesh ion Joule)
              + sum_j (Franck–Condon / quench fractions) dE_j r_j

    Wall boundary: fixed wall temperature with a thermal-slip jump in
    the transition regime (3.4).

3.3 Multi-species neutral transport, mixture-averaged:

        rho [ dw_k/dt + u.grad w_k ] = -div j_k + M_k S_k
        j_k = -rho D_k^m grad w_k - rho w_k D_k^m grad(ln Mbar)
        D_k^m = (1 - x_k) / sum_{j != k} (x_j / D_kj)

    with binary coefficients from Chapman–Enskog,

        D_kj = (3/16) sqrt(2 pi kB^3 Tg^3 / mu_kj) /
               ( p pi d_kj^2 Omega_D(T*) ),   mu_kj reduced mass,

    Lennard-Jones collision integrals Omega_D tabulated. One species is
    eliminated by the mass constraint sum w_k = 1 (the carrier gas).

3.4 Low-pressure / transition-regime corrections (0.01 < Kn < ~0.5,
    which includes the 10–20 mTorr ICP window for the smallest
    features):

    * flux-limited diffusion, already used and kept:
          D'_k = min(D_k, vbar_k Lambda),  Lambda the diffusion length,
      with mobility rescaled through Einstein so ambipolar fields stay
      consistent;
    * Maxwell velocity-slip and temperature-jump wall conditions:
          u_slip = ((2 - sigma_v)/sigma_v) lambda (du_t/dn)
          Tg_wall - T_wall = ((2 - sigma_T)/sigma_T)(2 gamma_h/(gamma_h+1))
                             (lambda / Pr) (dTg/dn)
    * free-molecular wall loss for species (the Chantry form in 2.4) —
      the correct limit the continuum flux must blend into;
    * validity flag: report Kn per region; above Kn ~ 0.5 the fluid gas
      model is out of scope (a DSMC neutral module is the eventual
      answer and is explicitly out of scope here).

3.5 Coupling back to the plasma: n_g(r, z) and Tg(r, z) feed E/N, all
    k_j(Te) collision partners, nu_m, and mu_i p-scaling. The gas module
    is slow physics — outside the Newton block, re-entered on a drift
    threshold exactly like the EMM.

Deliverables: (a) unchanged results for the fixed-bath configuration
(flag default), (b) gas-heating argon case showing central rarefaction
at >= 1 kW, (c) a flow case with inlet/outlet.

---

## Stage 4 — Sheath: analytic vs resolved, both as first-class options

The ASM is the production path and the physically correct closure at
low pressure: at 10–20 mTorr the ion mean free path (~mm) exceeds the
sheath thickness (~30–100 um) by ~100x, the sheath is collisionless,
and drift-diffusion inside it is the wrong equation. A resolved fluid
sheath there is an expensive way to compute a worse answer. The resolved
option exists for validation and for the parameter regimes where the
sheath IS collisional.

4.1 ASM (keep, default for production). Equations in 0.2. Remaining
    items: per-face zero-net-current is exact at floating dielectrics
    and an APPROXIMATION at grounded metal (suppresses DC wall-current
    circulation; measured effect small — wafer carries net ion current
    Ge/Gi ~ 0.09 that closes through ground). Document as a modelling
    choice.

4.2 Implicit surface charge — removes the binding sigma_s limit of 0.3
    without waiting for full JFNK. Treat the face charge at t+dt in
    both the Poisson RHS and the wall-flux linearisation:

        sigma^{n+1} = sigma^n + dt e [ Gamma_i - Gamma_e(Phi^{n+1}) ]
        Gamma_e(Phi^{n+1}) ~= Gamma_e(Phi^n)
                              - (Gamma_e/Te) (Phi^{n+1} - Phi^n)

    The linearised term adds a face conductance e Gamma_e/Te to the
    Poisson operator diagonal (same symmetric augmentation pattern as
    the drift terms), so the sigma_s mode becomes unconditionally
    stable and the coarse-wall mesh collects its full ~8x. NOTE: if
    Stage 6 puts sigma_s inside the Newton unknowns this is subsumed —
    decide there first, build once.

4.3 Resolved-sheath option (validation gate + collisional regimes).
    Prerequisites, in dependency order, all behind flags:
    * graded wall-normal mesh at lambda_D/3 (3–5 um at 1e18 m^-3);
    * field-dependent ion mobility (constant mu_i is 10–25x wrong at
      sheath fields), Frost form:
          mu_i(E/N) N = (mu N)_0 / sqrt( 1 + (E/N)/(E/N)_c )
      with Ar+/Ar parameters from swarm data; keep D_i thermal and
      document the deliberate Einstein-relation break;
    * implicit ions: backward-Euler ion continuity (same machinery as
      the electron P3 path) — the ion Courant limit at resolved-sheath
      fields is ~1e-12 s and cannot be marched explicitly;
    * the Stage 5 Poisson solver (conditioning grows as (L/dx_min)^2;
      Jacobi-CG goes from ~60 to ~1e4 iterations and stops being
      viable).
    Validation gate: on a sheath-resolving mesh, the resolved run and
    the ASM run must agree in bulk quantities to a few percent, and the
    resolved electron wall flux needs no Boltzmann throttle (the mesh
    holds the depletion itself). This turns the ASM from an assumption
    into a result.

4.4 RF-biased sheath (the IEAD path, later). The ASM barrier becomes a
    dynamic per-face variable driven by current continuity through the
    sheath capacitance:

        C_s d(dPhi_b)/dt = e (Gamma_i - Gamma_e(dPhi_b)) + J_displacement
        C_s ~= eps0 / s,   s from the Child–Langmuir relation above

    solved implicitly per face (it is stiff by construction), with the
    applied RF bias entering the metal-surface potential. Ion energy
    distributions at the wafer follow from dPhi_b(t) and the ion transit
    time — the quantity the process-control use case ultimately wants.

---

## Stage 5 — Poisson solver upgrades

5.1 Sparse direct Cholesky (CPU reference). The augmented operator is
    SPD; symbolic factorisation once, numeric per coefficient update.
    Role: exactness reference for 5.2 and the resolved-sheath gate. It
    is NOT the product path: host callbacks break `vmap` and GPU sparse
    direct is weak.

5.2 Geometric multigrid — the batched/GPU production path and the
    Stage 6 preconditioner. V-cycle with:
    * line (Thomas) relaxation along the wall-normal direction wherever
      cell aspect ratio or grading exceeds ~5 (point smoothers stall on
      anisotropy);
    * operator-dependent interpolation across the plasma/dielectric
      permittivity jump (plain bilinear loses mesh-independence there);
    * fixed cycle counts (2–3 V-cycles) rather than convergence loops,
      for `vmap` friendliness.
    Expected: mesh-independent ~8-iteration solves at any grading, all
    dense stencil work, batchable.

---

## Stage 6 — JFNK core (Jacobian-free Newton–Krylov with PTC)

Newton on the steady coupled residual of the FAST block only. Chemistry
(Stage 2), neutrals (Stage 3), and the EMM stay outside as segregated
modules — their outputs enter R as frozen inputs, refreshed on re-entry
thresholds, iterated to joint convergence by the outer Picard loop that
already exists.

6.1 Residual assembly. Unknowns u = (ne, n_eps, ni per charged species,
    Phi, sigma_s), residuals the steady forms of 0.2:

        R_ne   = div Gamma_e(ne, Phi, Te) - S_e(n; Te)
        R_neps = div Gamma_eps + Gamma_e.E + sum_j dE_j r_j
                 + 3 (me/M) nu_m ne (Te - Tg) - Q_ind/e
        R_ni   = div Gamma_i(ni, Phi) - S_i(n; Te)
        R_Phi  = -div(eps grad(Phi + dPhi_b)) - e(sum Z n - ne) - rho_s
        R_sig  = e (Gamma_i - Gamma_e) . n_hat        (per dielectric face)

    R_sig = 0 at the solution IS the zero-net-current condition, so
    sigma_s inside the Newton block subsumes 4.2. The ASM battery and
    wall closures are just terms of R — analytic vs resolved sheath is a
    residual-assembly flag, not a different solver.

    This residual is worth building EARLY (Stage 1.4 uses its norm as
    the convergence criterion long before Newton exists).

6.2 Jacobian-free Newton: solve J du = -R with J v obtained exactly by
    forward-mode AD,

        J v = d/d(eps) R(u + eps v) |_{eps=0}     (jax.jvp, no FD noise)

    inner solver GMRES(m) with FIXED restart length and iteration count
    (batchability), right-preconditioned. The Jacobian is non-symmetric
    (upwinding); the adjoint path needs an explicit transpose solve —
    a symmetric-solver shortcut silently corrupts gradients while
    leaving forward results correct (this trap already cost a debugging
    cycle once).

6.3 Preconditioner, block triangular:
    * Phi block: one multigrid V-cycle (5.2);
    * transport blocks: the existing Jacobi/line-smoother diagonals of
      the backward-Euler operators;
    * sigma_s rows: the face conductance e Gamma_e/Te from 4.2;
    * chemistry coupling: the per-cell J_chem (2.3) folded in as a
      block-diagonal correction if Picard chemistry lags convergence.

6.4 Globalisation: pseudo-transient continuation. Solve

        ( I/dtau - J ) du = R,      u <- u + du

    with switched evolution relaxation for the pseudo-step:

        dtau_{k+1} = dtau_k * ||R_{k-1}|| / ||R_k||

    At small dtau this IS the current implicit stepper (the migration is
    continuous, not a rewrite); as ||R|| -> 0, dtau -> inf and the
    iteration becomes pure Newton with quadratic convergence. Cold
    starts (ignition) run the PTC ramp; warm starts (parameter sweeps —
    the data-generation workload) take ~3–5 Newton steps per operating
    point. Line search / trust region only if SER proves insufficient.

6.5 Scaling: nondimensionalise the residual blocks (n by a reference
    density, Phi by a reference Te, n_eps by n_ref Te_ref) so GMRES sees
    O(1) blocks; without this the Phi rows (Coulombs) and density rows
    (1e18) differ by ~40 orders and Krylov convergence is meaningless.

6.6 Adjoint at the fixed point (free with 6.1): for an objective F(u, p),

        dF/dp = dF/dp|_explicit - lambda^T dR/dp,
        J^T lambda = (dF/du)^T

    — differentiate the converged state, not the iteration path. Not
    needed for data generation, valuable for calibration and
    physics-informed training later.

Expected effect: from ~1e5 preconditioned linear solves per converged
case (pseudo-time) to ~30–50 Newton steps x fixed GMRES work — the
algorithmic gap that separates marching codes from implicit codes, plus
the warm-start advantage that dominates sweep throughput.

---

## Stage 7 — GPU batching and the data factory

7.1 `vmap` the whole solve over operating points (power, pressure, gas
    mix) on a FIXED mesh and geometry (geometry changes retrace; a data
    campaign is per-reactor). Requirements already threaded through the
    plan: fixed-trip-count inner loops everywhere (GMRES restarts, MG
    cycles, substep counts), no host callbacks in the batched path, and
    per-member convergence masks — a converged batch member does
    freeze-and-carry (its du is masked to zero) until the batch retires;
    stragglers are regrouped into the next batch.

7.2 Throughput benchmark as a tracked regression: converged cases per
    GPU-hour vs batch size N. This is the product metric; single-case
    timing is only a diagnostic.

7.3 Precision strategy. The stack is float64; consumer GPUs run FP64 at
    1/64 rate. Either run data-center parts, or investigate FP32 inner
    solves with FP64 outer residuals (iterative refinement). Caution
    flag from measurement: the Poisson solve tolerates NO accuracy loss
    (loosening cg_tol detonated ignition), so mixed precision must be
    validated against the full-trajectory regression, not assumed.

7.4 Time-accurate transient mode (also needed by Stage 1's R(t)
    benchmark): one global dt, no local time stepping, no steady-state
    acceleration, BDF1/BDF2 in real time. The pseudo-transient
    trajectory has no physical meaning; transient validation and any
    future pulsed-plasma work need this mode.

7.5 The training-data harness: sweep definition -> batched solve ->
    per-case QA gates (residual norm, ledger closure, guard flags) ->
    labelled dataset with provenance (git hash, mesh, chemistry set,
    solver settings). Cases failing QA are excluded automatically —
    silent bad labels are worse than missing labels.

---

## Stage 8 — Monte Carlo electrons (far horizon, architecture only)

The EMCS replaces the fluid electron-energy closure: electron transport
and rate COEFFICIENTS (mobility, diffusion, k_j, mean energy sources)
are computed by particle trajectories in the frozen fields and fed to
the fluid/Newton solve through the SAME coefficient interface the
Phelps tables use (Stage 1.5). Consequences already accounted for:

* the module is non-differentiable and does not `vmap` — it lives
  outside the Newton block and outside the batched path, alternating
  with fluid solves exactly like the EMM does;
* Maxwellian-table results (1.5) come first and quantify how much of
  the fluid-tier error is the EEDF assumption vs the closure — that
  measurement decides how urgent this stage ever becomes.

---

## Suggested order and why

1. **Stage 1** (benchmark anchor) — cheap, and every later stage needs
   the harness and the residual-based convergence to prove itself.
2. **Stage 2** (chemistry restructure) — the large-set goal dies without
   it, it is independent of solver work, and 2.3's per-cell machinery is
   reused by 6.3.
3. **Stage 5.2** (multigrid) — the JFNK preconditioner and the batched
   Poisson path; 5.1 alongside as the cheap exactness reference.
4. **Stage 6** (JFNK + PTC), deciding sigma_s inside (subsumes 4.2).
   If a quick win is wanted earlier, 4.2 alone unlocks the coarse-wall
   mesh's 8x and is small.
5. **Stage 7** (batching + data factory) — the product.
6. **Stage 3** (neutral gas) in parallel with 5–7 by anyone not touching
   the solver core; it is segregated physics.
7. **Stage 4.3/4.4** (resolved sheath, RF bias) when validation or IEAD
   demands them.
8. **Stage 8** last, gated on the Stage 1.5 EEDF measurement.
