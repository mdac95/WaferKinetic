# Plan: reproduce the COMSOL argon GEC ICP benchmark

Source: `models.plasma.argon_gec_icp.pdf` (COMSOL 6.4, Application Library
`Plasma_Module/Inductively_Coupled_Plasmas/argon_gec_icp`). Written before
any code, per the standing preference.

---

## 0. Headline: do NOT resolve the sheath

The question was whether resolving the sheath needs implicit sigma_s plus
JFNK. Three separate answers:

1. **sigma_s is not the sheath.** sigma_s is the dielectric surface charge.
   It is stiff whether or not the sheath is resolved, and it is what binds
   the ASM path *today* (measured). Implicit sigma_s is worth doing on its
   own merits and has nothing to do with sheath resolution.

2. **Resolving the sheath needs far more than those two items.** At the
   COMSOL operating point lambda_D = 11.9 um (n_e = 1.5e18, T_e = 3.85 eV)
   and the Child-Langmuir sheath is s = 32 um. Resolving it means 3-5 um
   wall-normal cells against today's 278 um -- a 60-90x graded refinement.
   That cascades:
   - ion Courant and dielectric relaxation -> dt ~ 1e-12 s, so implicit
     ions (roadmap 2.5) stop being optional;
   - Poisson conditioning kappa ~ (L/dx_min)^2, so Jacobi-CG iterations go
     from ~58 to ~1e4 -- the current solver simply stops working, and
     section 4 (direct Cholesky and/or multigrid with line relaxation)
     becomes a prerequisite, not a nice-to-have;
   - mu_i(E/N) (roadmap 1.3) becomes mandatory: E/N inside a resolved
     sheath is where constant mobility is wrong by 10-25x.

3. **And it would compute a physically wrong sheath.** At 20 mTorr the ion
   mean free path is ~3 mm against a 32 um sheath -- the sheath is
   collisionless by a factor ~100. Drift-diffusion, and even a fluid
   momentum equation with a collision term, is the wrong closure there. A
   resolved fluid sheath is an expensive way to get a worse answer than the
   analytic collisionless sheath we already have. **The ASM is the correct
   closure at this pressure**, which is exactly why the doc says bulk
   quantities differ by only a few percent with and without it.

**Evidence that ASM is already the better match to COMSOL:**

| quantity | resolved-mesh attempt | ASM (ours) | COMSOL |
|---|---|---|---|
| dPhi/T_e | 6.57 wafer / 3.14 window | 5.53 / 5.49 | ~5.0 |

COMSOL does not Debye-resolve the sheath either: 5 boundary layers with
stretching 1.4 on an "extra fine" triangular mesh, 1 mm max element on the
wall edges. Their sheath is partially resolved and mesh-dependent; ours is
analytic. Chasing a resolved sheath would move us *away* from the target.

---

## 1. What COMSOL actually solves (from the PDF)

| item | COMSOL | ours today |
|---|---|---|
| pressure | **20 mTorr**, 300 K | 10 mTorr |
| power | **1500 W to the coil**, ~1200 W into the plasma | 500 W deposited |
| electron transport | drift-diffusion + mean-energy equation | same (P3 implicit) |
| reduced mobility | **mu_e N = 4e24 1/(m V s)** | implies 1.75e24 (see 2.1) |
| EEDF / rates | Phelps cross sections (LXCat), integrated | illustrative Lieberman fits |
| chemistry | 7 reactions, 4 species | **same set, same thresholds** |
| Penning | 3.734e8 m3/(mol s) = 6.20e-16 m3/s | **6.2e-16, exact match** |
| quenching | 1807 m3/(mol s) = 3.00e-21 m3/s | 2.1e-21 (**1.43x low**) |
| electron wall flux | (1/2) v_th n_e, i.e. **r_e = 0** | (1-r_e)/(1+r_e)/2, r_e = 0.2 -> (1/3) v_th |
| energy wall flux | (5/6) v_th n_eps | same form |
| ion wall flux | surface reaction + outward-gated migration, **no thermal term** | 0.25 v_th,i + gated drift (Eq. 37) |
| sheath | full Poisson, boundary-layer mesh (5 layers, 1.4) | analytic (ASM) |
| surface reactions | Ars=>Ar and Ar+=>Ar, sticking 1 | equivalent |
| secondary emission | none | none (matches) |
| time | transient BDF, log times to 1 ms | pseudo-transient to a fixed point |
| dielectric | eps_r = 4.2 | **4.2, matches** |

**COMSOL results to hit** (read off the figures at t = 1 ms):

| metric | COMSOL |
|---|---|
| n_e peak | 1.5e18 m^-3 |
| T_e peak | 3.85 eV (under the coil, at z = 40 mm) |
| plasma potential peak | 19-20 V |
| Ar* peak | 8.5e18 m^-3 (mass fraction ~0.013) |
| skin depth | ~1 cm |
| coil resistance | rises ~4x when the plasma ignites |
| power split | ~1400 W in the coil pre-ignition -> ~1200 W in the plasma |

---

## 2. The two concrete numerical mismatches found so far

### 2.1 Electron mobility is 2.3x too low

COMSOL: mu_e N = 4e24, so at n_g = 6.44e20 (20 mTorr, 300 K),
mu_e = 6214 m2/(V s).

Ours: nu_m = n_g k_mom with k_mom = 1e-13 gives nu_m = 6.44e7 s^-1 and
mu_e = q/(m_e nu_m) = 2732 m2/(V s).

Matching COMSOL requires **k_mom = 4.40e-14 m3/s**, i.e. our illustrative
value is 2.3x too large. This propagates into D_e = mu_e T_e, the ambipolar
field, the plasma conductivity (hence skin depth and power coupling), and
the ES heating term -- it is not a small correction. `HybridCase.k_mom`
already documents itself as illustrative; this is the number to replace.

### 2.2 Metastable quenching is 1.43x low

`K_QUENCH = 2.1e-21` (flagged "illustrative" in the source) against
COMSOL's 3.00e-21 m3/s. Direct substitution.

Everything else in the chemistry -- the reaction set, the stoichiometry,
the three thresholds, the Penning rate -- already matches exactly. The doc's
"Table 2" set *is* the COMSOL set. That is a good position to start from:
the remaining chemistry gap is only the k(T_e) curves themselves.

---

## 3. Recommended order

### Phase 0 -- parameters only, no new physics (hours)

The point is to find out how big the real gap is before building anything.

- **0.1 Match the operating point**: 20 mTorr, P_dep = 1200 W. Pure
  parameter change; rerun the ASM case and compare against the table above.
- **0.2 Apply 2.1 and 2.2** (k_mom, K_QUENCH). Also parameters.
- **0.3 COMSOL-compatible wall mode**, behind one flag: r_e = 0 and the ion
  wall flux without the thermal term. Both are cheap and both change the
  particle balance by O(1.5x) at the walls -- they must be A/B-able, not
  silently adopted.
- **0.4 Convergence on the residual, not `rel`** (roadmap 6.2). `rel` floors
  at ~3e-4 from solver jitter, so today no run can honestly be *declared*
  converged. This has to be fixed before we start quoting numbers against
  someone else's code.

Deliverable: a first honest side-by-side. My expectation is that we land
high on n_e -- our 10 mTorr / 500 W run already gives 1.33e18 against
COMSOL's 1.5e18 at twice the pressure and 2.4x the power -- and the
diagnosis then splits between the mobility fix (2.1) and the rate
coefficients (Phase 1).

### Phase 1 -- the accuracy floor (the real work)

- **1.1 Phelps cross sections -> k(T_e) tables** (roadmap 1.1). The
  `k(Te)` callable interface in `chemistry.py` was designed for exactly
  this. Phelps data is on LXCat, freely usable with citation; COMSOL uses
  the same source, so this removes the largest remaining physics
  difference. It also supplies the elastic momentum-transfer frequency
  properly, superseding the 2.1 patch.
- **1.2 Re-test the "fluid is 2x high" claim** at matched conditions
  (roadmap 1.2). With the same chemistry, the same closure and the same
  operating point, any residual gap against COMSOL is *our* discretisation
  or wall model, not the fluid tier -- that is the whole value of this
  benchmark.

### Phase 2 -- speed and architecture (after we can compare, not before)

- **2.1 Implicit sigma_s** (roadmap 2.1) -- now measured as the binding
  limit under ASM, not assumed. Removes both the timestep limit and the
  per-cell dt gradient it currently creates.
- **2.2 Direct Cholesky Poisson** (roadmap 4.1-4.3) as exact reference and
  CPU path; it is also the preconditioner block JFNK will need.
- **2.3 JFNK** (roadmap 5.1-5.4) with PTC globalisation. Decide up front
  whether sigma_s is inside the Newton block -- if it is, 2.1 is subsumed
  and must not be built twice.

### Phase 3 -- sheath resolution, only as a validation experiment

Do this only if Phase 1 shows the wall model is where we disagree, and then
do it as roadmap 4.4's gate -- a resolved run that must agree with the ASM
run -- never as the production path. Prerequisites: graded mesh (3.1),
implicit ions (2.5), the new Poisson solver (4.x), mu_i(E/N) (1.3).

---

## 4. Items to ADD to the roadmap for this target

1. **Time-accurate transient mode.** COMSOL's headline plots are functions
   of time: coil resistance R(t), power split P(t), ignition at ~1 us,
   quasi-steady by 1 ms. Our pseudo-time has no physical meaning (per-cell
   dt, Eq. 56 acceleration). Comparing transients needs a distinct mode:
   one global dt, no local time stepping, no acceleration, and a real
   implicit time integrator. Worth having anyway -- it is the only way to
   validate against any transient measurement.
2. **Coil circuit diagnostics**: coil resistance and inductance with the
   plasma on and off, copper conductivity 6e7 S/m, 5-turn series coil
   group, and the coil-vs-plasma power split. COMSOL singles these out as
   the quantities that are cheap to measure experimentally -- "without the
   need for expensive optical emission spectroscopy or Langmuir probes" --
   so they are the most valuable validation output we do not currently
   produce. Our EMM prescribes deposited power and never models coil loss.
3. **A COMSOL-comparison harness** (roadmap 6.4, extended to an external
   reference): one command that reports n_e peak, T_e peak, Phi peak, Ar*
   peak, deposited power, skin depth, coil R ratio and wafer ion flux
   against stored COMSOL targets, plus line cuts at fixed r and z. Without
   this the comparison will be re-done by hand every time.
4. **Geometry audit against their sequence.** From the figures: plasma
   z in [0, 40] mm, r to 57.5 mm with the step out to 63.5 mm, wafer
   z in [-2, 0] mm to r = 83.5 mm, chamber to r = 140 mm, z in [-25, 80] mm.
   Ours differs in the chamber floor (-40 vs -25 mm), outer radius (145 vs
   140 mm) and wafer thickness (3 vs 2 mm). Minor, but they change the
   diffusion length, so align them before blaming physics.
5. **Heavy-species diffusion coefficient check.** COMSOL solves mass
   fractions with mixture-averaged diffusion; we use a fixed flux-limited
   D for Ar*. The formulations need not match, but the coefficient must.
6. **Fix the `uniform_dt` inconsistency** found this session: with
   `uniform_dt=True` the in-loop clips still make dt per-cell, which
   defeats the flag's purpose and creates the dt gradients that are
   themselves destabilising. Either honour it globally or document it.
7. **Grid-convergence as a standing check.** The ASM fine-vs-coarse pair
   now passes SPEC section D; make it a regression rather than a one-off.

## 5. Items to DROP or defer for this target

Reproducing COMSOL is a *narrower* problem than the roadmap, which is good
news. Not required, and in some cases actively counterproductive:

- **gas heating / neutral temperature** -- COMSOL fixes T_g = 300 K;
- **ion temperature equation** -- COMSOL uses T_i = T_gas;
- **radiation trapping / Holstein** -- COMSOL lumps the excited states into
  one Ars with wall sticking 1 and no trapping;
- **ion momentum (Eq. 25)** -- COMSOL uses drift-diffusion. Adding momentum
  would make us differ from the target *by design*. Keep it as a
  post-benchmark fidelity item, not a reproduction item;
- **doc Eq. 35 (ion predictor)** -- measured this session: the Phi-rho
  density prediction is not what binds;
- **electronegative chemistry, RF bias, secondary emission** -- none are in
  this model.
