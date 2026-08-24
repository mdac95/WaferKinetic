# WaferKinetic — handoff to Claude Code

Read this first, then `ROADMAP_solver.md` and `SPEC_asm_and_ion_predictor.md`
(both should be in the repo). This file is the *state of the world* as of
the end of the chat-based session; those two are the plan.

---

## 0. Where things stand (one paragraph)

The implicit (P3) electron path is implemented, verified, and is the
production path: consistency vs explicit 2.6e-5 at equal clock, ledger
residual 3e-10 at 1e4x the explicit dt, exact positivity, reverse-mode
gradient matches finite differences to 2e-5. Three physics defects were
found and fixed behind flags (§2). The GEC argon ICP converges to a
physically credible state (§3). The binding numerical limit was measured
by bisection and is **the sheath-RC clip at ~1e-10 s per substep** — not
ion Courant, not Gummel convergence, not electron diffusion (§4). The
next step is doc P6 (analytic sheath), which is expected to remove that
limit *and* fix the remaining sheath-drop discrepancy at the same time.

## 1. Run the converged case (regression baseline)

```
JAX_PLATFORMS=cpu python demo_implicit_diag.py --converge --ramp 30 --ramp-gate 0.02 \
    --chem-thr 0.5 --uniform-dt --es-joule flux --no-eetm-slice --wall-flux sheath \
    --dt-cap 4e-10 --max-outer 800 --no-accel
```

~2.3 s/iteration on CPU, ~800 iterations, ~30 min. **CPU is ~3x faster
than GPU at this mesh size** (0.7 MB/field fits in cache; GPU is
launch-latency-bound at ~26 cells per FP64 core). Do not optimise for
GPU until the mesh is refined or `vmap` batching is in play. Also verify
FP64 hardware rate if a GPU is ever used — GeForce runs FP64 at 1/64.

Post-process:
```
python postprocess_fields.py outputs/<stem>_fields.npz --r 0.001 0.04
python plot_mesh.py
```

`outputs/` holds log, maps, transient figure, and a `.npz` field dump
(grid, ne, ni, n_eps, n_ars, Te, Phi, Er, Ez, sigma_s, face fluxes,
Q_ind). On a divergence the dump is the **last finite state**.

## 2. Flags whose non-default value is the one that works

These were all introduced as flags per the user's standing preference
("as flag please"). The working configuration is the command above.
**Roadmap item 0.x is to flip the defaults and amend the doc**, with the
evidence recorded here.

| flag | working value | why (evidence) | doc impact |
|---|---|---|---|
| `es_joule` | `flux` (−Γ_e·E_S) | drift form q μ_e n_e E² gives 335 W of ES heating from a 4.9 V ambipolar potential at 25 W input; true ambipolar J·E is ~1 W. Drift-only keeps the positive-definite half of a nearly cancelling pair. | **amend Eq. 12**; record departure from Kushner 2009 Eq. 5 |
| `wall_flux` | `sheath` (Boltzmann factor on the Eq. 36 thermal flux) | thermal form has no Φ dependence → σ_s cannot reach zero net current on a non-sheath-resolving mesh; d(σ_s) plateaus at 3e-13 with the factor. **This is not a departure** — it is a partial implementation of doc P6 / Kushner Option 6 (see SPEC). | none once P6 is complete |
| `eetm_slice` | `False` | redundant with the FKPM's internal sub-slicing; computed dt once with no in-loop re-clip (the original CFL-2.25 divergence); a second site for source terms to drift out of sync — which happened. | delete the slice |
| `implicit_electrons` | `True` | the whole point of the session | P3 is already in the doc |
| `uniform_dt` | `True` | per-cell local dt is **not conservative in transit** (dt_A≠dt_B across a face creates charge that enters Poisson); produced erratic Φ. The doc's §12.3 Courant bound is a global one. Use uniform while debugging physics. | doc already says global |
| `use_accel` | `False` | Eq. 56 extrapolates *slow* species; during ignition Ar* is the *fastest* variable in a stepwise-ionisation loop → diverges at any clamp size (observed at δ_max = 0.026). | none; re-enable only after ignition |
| `ion_wall` | `thermal` (keep) | `bohm` floor was a **no-op**: ion delivery at the wafer is drift-dominated at 5.07e5 m/s = 157 u_B (E/N ≈ 112,000 Td). Prediction refuted by measurement. Keep the flag, it is harmless. | — |
| `ion_field` | `static` (keep) | Richards E_eff is a **no-op at any DC fixed point** (∂E_eff/∂t = 0 ⇒ E_eff ≡ E_S) and injects a 145-iteration slow mode that couples to the solution's own slow mode → oscillation → divergence. Useful only for RF-resolved sheaths. Keep, document as RF-only. | — |

Default-flip checklist: `es_joule`, `wall_flux`→(superseded by P6),
`eetm_slice`, `implicit_electrons`, `uniform_dt`. Leave `use_accel`
default True but gate it on "ramp complete AND rel < threshold".

## 3. Converged state (10 mTorr, 500 W, 61×62 mesh)

| quantity | value | note |
|---|---|---|
| n_e peak | 6.56e17 m⁻³ = 6.6e11 cm⁻³ | ~3.3e11 kinetic-equivalent after the documented 2× fluid overestimate — in the GEC band |
| T_e | 4.47 eV bulk, flat | plausible |
| Φ_pk | 32.7 V | high; see sheath drop below |
| ledger | 0.4–0.5 % | P_in 500 + P_es −163 = P_wall 95 + P_inel 239 |
| σ_s | −3.12e-9 C, d/dit ~4e-14 | plateaued: the surface-charge balance **closes** |
| window (floating dielectric) ΔΦ/T_e | **2.95** | ASM predicts 4.97 → P6 will fix |
| wafer (grounded) ΔΦ/T_e | **6.5** | set by net ion current at a grounded surface; should stop depending on mesh after P6 |
| wafer ion delivery | 5.07e5 m/s per ion | drift-dominated, E/N ≈ 112,000 Td; constant μ_i invalid by ~10–25× there |

`rel` floors at ~1e-3 against `tol=1e-5` because the breakdown guards
(below) inject solver-tolerance jitter into Φ. The state *is* converged.
**Roadmap 6.2: convergence criterion on the residual, not on `rel`.**

## 4. Measured numerical limits

- Per-substep dt threshold (bisection with `--dt-cap`): stable at
  4e-11, marginal at 4e-11–1e-10, diverges at 1.7e-10.
- Ion Courant at Φ≈20 V: 8.9e-9 s — **50× larger, not binding**.
- Gummel sweeps: n_gummel=2 and 6 agree to four digits — converged,
  not binding.
- **Binding: the explicit surface-charge response (sheath-RC clip).**
  The code's clip estimates the rate from the *net* current
  q|Γ_e−Γ_i|/(ε₀E_s); the quantity that actually limits an explicit
  σ_s update is the *differential* response dΓ_e/dΦ = −Γ_e/T_e, which
  the net-current estimate understates by ~6×. The sheath flag relaxed
  it (Γ_e→Γ_i at balance); P6 should remove it (wall cell becomes
  quasineutral, presheath field ~225 V/m vs 3.6e4 V/m now, 161×).

## 5. Guards now in the code (do not remove)

- **Krylov breakdown guards** in `imsolve` (BiCGStab) and
  `make_jax_poisson_solver` (CG): accept the solver's answer only if
  ‖Ax−b‖ ≤ the warm start's residual. Both solvers warm-start from the
  previous solution; at convergence the initial residual is rounding
  noise and the recurrence forms 0/0 — **NaN or finite garbage (8e77
  was observed and passed an isfinite test)**. Failure probability
  *rises* as the outer iteration converges. Residual acceptance is
  monotone by construction.
- **Physics-bounded divergence guard** in `run_hybrid`: species ≤
  10·n_g, n_ε ≤ 1e3·n_g, |Φ| ≤ 10 kV; returns the last finite state
  with history. Finiteness alone let a frozen garbage state be
  *certified converged* (rel→0 on a frozen state).
- `check_module_versions()` in the demo: fails fast with the stale
  file named. Five files move together (poisson, hybrid, demo, test,
  postprocess). A mixed set produced two "you broke it" incidents.
- NaN-gradient trap fixed in `clip_dt` (divide inside a `where` still
  evaluates the unselected branch in reverse mode → 0·∞). The same
  double-`where` idiom is used in `limit_outflow`; apply it to any new
  guarded division.
- T_e is `jnp.clip`-ed → **zero gradient on the clamp**; this silently
  killed the first gradient test. Use a soft saturation (roadmap 1.8).

## 6. Things that were wrong in my own reasoning, so you don't repeat them

- Attributed the Richards effective field to Kushner — it's Richards,
  Thompson & Sawin APL 50, 492 (1987), cited in the Economou tutorial
  Eq. 6 and doc §5.1 option 2; Kushner doesn't use it (HPEM solves ion
  momentum). And it cannot act at a DC fixed point.
- Called `wall_flux=sheath` a "departure from the doc". It is an
  incomplete doc P6. The doc also says bulk parameters with/without the
  analytic sheath differ by only a few percent — i.e. **mesh-resolving
  the sheath is the expensive path to a few percent.**
- Mis-cited the Eremin benchmark: HC1 = full ion momentum (matched PIC);
  HC2 = drift-diffusion + field-dependent μ_i (still 2× high). Neither
  is the effective field. Lesson: μ_i(E/N) and momentum fix *different*
  defects; neither substitutes for the other.
- Predicted the Bohm floor would help; flux decomposition refuted it in
  one run. The decomposition printout (thermal vs Bohm vs drift delivery
  per ion at each wall face) is in `postprocess_fields.py` — use it
  before theorising about ion wall physics.
- `dt_sink` alone is not a bound (returned 1e42 s where L→0); it is now
  capped by `dt_chem`.

## 7. Stated direction from the user, mapped onto the roadmap

> speed up significantly; ion and neutral momentum; more accurate,
> generalised chemistry; keep Maxwellian electrons now, plan
> Boltzmann/EMCS later; Kushner-side ion improvements so the timestep
> is not so slow.

Ordered by dependency (not appeal), with the reasoning:

1. **P6 / ASM** (SPEC §A). Physics fix for the sheath drop *and* the
   largest single speed win available — it removes the binding clip.
   Resolve SPEC §A.6 (σ_s vs ΔΦ_b on dielectrics) before coding; the
   two must not both enforce zero current. Validation: floating ΔΦ/T_e
   = 4.97 to tolerance; coarse vs fine mesh agree to a few percent.
2. **Re-measure the binding clip** after P6 (`--dt-cap` bisection).
   Only then decide on Eq. 35 (SPEC §B) — it helps the Φ↔ρ prediction,
   not advection, and probably earns nothing until the RC clip is gone.
3. **Kill chemistry re-bake retraces** (roadmap 2.2): frozen heavy
   densities as runtime arguments. `chem_thr=0.5` reduced retraces to
   ~1%; making them zero is cheap.
4. **Profile** before any other speed work (roadmap 2.3). The 2.3 s/it
   has never been attributed.
5. **μ_i(E/N)** (roadmap 1.3, equations in chat: Frost form
   μN = (μN)₀/√(1+(E/N)/(E/N)_c), Ar⁺/Ar from Ellis et al. tables;
   keep D_i thermal, document the Einstein break; update the P2 ion
   augmentation with the local μ_i or predictor/corrector disagree).
   After P6 it is a physics refinement, not a numerical rescue.
6. **Ion momentum (doc Eq. 25 / Kushner Option 2).** This is what
   matched PIC in the Eremin benchmark and is the real answer to the
   ~2× density overestimate. **Cost is structural**: Kushner states
   "the direct semi-implicit solution for the ion fluxes with Φ is
   lost" — P2/P3 ion coupling goes away, and Eq. 35 or a Newton
   linearisation must replace it. Do P6 and μ_i(E/N) first so the
   momentum result is compared against a sheath-correct baseline.
7. **Neutral momentum** (doc Eq. 25 with viscous stress, plus gas
   heating roadmap 1.5). Needed above a few hundred W and for any flow
   case; independent of the charged-species solver, can proceed in
   parallel.
8. **Generalised chemistry.** The `k(Te)` callable interface in
   `chemistry.py` was designed for this. Order: (a) BOLSIG+ tables for
   argon to **re-test the 2× density claim** — some of it may be
   Maxwellian-averaged rate coefficients, which changes the EMCS
   urgency; (b) multi-species/electronegative sets; (c) Holstein
   trapping for Ar* (at 1.4e19 it matters). Clean-room IP hygiene
   applies.
9. **Poisson solver** (roadmap §4): sparse direct (Cholesky — the
   augmented operator is SPD; symbolic once, numeric per update via
   CHOLMOD/PyPardiso, *not* `splu`) as reference + CPU path; geometric
   multigrid with line relaxation as the GPU/batch path. Do this
   **before** any sheath-resolving refinement; with P6 in place the
   refinement may not be needed at all.
10. **JFNK** (roadmap §5) on the coupled (n_e, n_i, n_ε, Φ) residual;
    `jax.jvp` for exact JVPs; PTC globalisation so it is a continuous
    migration from the current solver. This is where the ~2000×
    algorithmic gap against COMSOL closes.
11. **EMCS / Boltzmann plan** (last, by the user's own ordering): keep
    the coefficient interface; BOLSIG+ tables first (cheap, reversible);
    EMCS as the non-differentiable GPU module behind the same
    interface. The fidelity notice (2×, missing on-axis T_e minimum, no
    stochastic heating) is a closure property — decide from (8a)
    whether the 2× is closure or chemistry before committing.

## 8. Process rules that earned their keep

- **Spec before code** ("describe equations and what you'd modify").
  Two of the last three features were refuted cheaply because the
  predictions were written down first.
- **One flag per physics change, default unchanged**, threaded through
  every site that must agree: stepper (both bodies), RC clip, ledger,
  NumPy reference, dump. Three bugs came from a site being missed.
- **Explicit path byte-identical.** There is a diff audit in the
  session history; keep it.
- **End-to-end trace on the 10×12 synthetic reactor before delivery.**
  CPU jax is cheap; `py_compile` is not a test. The recipe is in the
  last smoke script of the session (build_setup → seed → HybridDrivers
  → run_hybrid → save_fields, warnings-as-errors, both flag modes).
- **Instrument before theorising.** ε̄ unclipped, σ_s total, Φ/T_e,
  per-face flux decomposition, measured (not ceiling) slice advance —
  each of these ended a wrong theory in one run.
- No test execution during sessions was the old constraint; with Code
  access it no longer applies — run the smoke.

## 9. Files touched this session

`poisson.py`, `hybrid.py`, `electron_energy.py`, `demo_implicit_diag.py`
(new), `test_implicit_electrons.py` (new), `postprocess_fields.py`
(new), `plot_mesh.py` (new), `ROADMAP_solver.md` (new),
`SPEC_asm_and_ion_predictor.md` (new). `transport.py`, `chemistry.py`,
`inductive.py`, `neutrals.py`, `reactor_mesh.py`, `gec_case.py`
unchanged — except the recommended one-line mesh fix in `gec_case.py`
(replace `np.linspace(Z_STEP_BOT, Z_SLAB_BOT, 5)` with two
`geometric_grid` calls meeting at the midpoint; slab face 1.5 mm →
0.23 mm for +8 z-cells). That fix becomes less important once P6 takes
the sheath out of the mesh.
