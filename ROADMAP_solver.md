# WaferKinetic — solver roadmap

State as of this session: the implicit (P3) electron path is verified
(consistency 2.6e-5 vs explicit at equal clock, ledger residual 3e-10 at
1e4x dt, exact positivity, adjoint matches FD to 2e-5). Three physics
defects were found and fixed behind flags. The run now ignites, the
surface-charge balance closes, and `imb` falls to ~4%.

Everything below is what is left. Ordering is by dependency, not by
appeal.

---

## 0. Decisions blocking everything else

These are not code tasks. Each is currently a flag whose non-default
value is the one that works, which is an unstable place to sit.

| # | decision | evidence |
|---|---|---|
| 0.1 | `es_joule="flux"` becomes default; **amend doc Eq. 12** | drift form gives 335 W of ES heating from a 4.9 V ambipolar potential at 25 W input; true ambipolar J·E is ~1 W |
| 0.2 | `wall_flux="sheath"` becomes default; **amend doc Eq. 36** | thermal form has no Φ dependence, so σ_s cannot reach zero net current on a non-sheath-resolving mesh; with the Boltzmann factor d(σ_s) → 3e-13 and plateaus |
| 0.3 | `eetm_slice=False` becomes default (delete the slice) | redundant with the FKPM's internal sub-slicing, computes dt once with no in-loop re-clip (the original CFL-2.25 divergence), and is a second site for source terms to drift out of sync — which is exactly what happened |

Both doc amendments should record *why* they depart from Kushner 2009
(Eq. 5 and the Eq. 36 flux), so the deviation is a documented modelling
choice and not an undisclosed one. Note 0.2 reduces to the plain
thermal flux when λ_D is resolved — see 4.4 for the validation that
makes this claim testable rather than asserted.

---

## 1. Correctness, before any performance work

| # | task | notes |
|---|---|---|
| 1.1 | **BOLSIG+ rate tables** replacing the illustrative Lieberman fits in `chemistry.py` | largest single accuracy item, and cheap. Also feeds 1.2 |
| 1.2 | **Re-test the ~2x density claim** | the fidelity notice attributes it to the fluid closure. Some may be Maxwellian-averaged rate coefficients. If BOLSIG+ tables close much of it, the urgency of EMCS changes and the notice needs rewording |
| 1.3 | **Richards effective field** for ions (Kushner Eq. 6) | E/N in sheath cells is ~37,000 Td; constant μ_i is valid below ~100 Td. Ion drift saturates, so μ_i·E overstates the speed — this is both a physics error and a pessimistic Courant bound. Fixes both at once, and is cheaper than implicit ions |
| 1.4 | **Ion temperature equation** | currently a fixed parameter |
| 1.5 | **Gas heating / neutral temperature field** | absent entirely; matters above a few hundred W |
| 1.6 | **Radiation trapping (Holstein escape factors)** for Ar* | metastables are at ~1e18; resonance trapping directly changes that |
| 1.7 | **Secondary electron emission** at surfaces | |
| 1.8 | **Soft-clamp T_e instead of `jnp.clip`** | a hard clip has *identically zero* gradient in the saturated region. This silently killed the first gradient test. Use a smooth saturation so the adjoint survives even when the solution touches the ceiling |
| 1.9 | Electronegative chemistry; multiple ion species | next complexity tier, after the above are stable |
| 1.10 | RF bias / IEAD at the wafer | separable; needs 1.3 first |

---

## 2. Numerics — the currently binding limits

Measured: per-substep dt is 2.78e-10 s with the cap at 4e-10, so the
**in-loop clip binds**. Ion Courant is 4.8e-9 — twenty times larger, not
binding. The sheath-RC clip sets the step.

| # | task | expected |
|---|---|---|
| 2.1 | **Implicit σ_s**: add the wall-flux derivative −Γ_e/T_e to `p2_augmentation` on surface faces (the Kushner Eq. 8 term that evaluates surface fluxes at Φ(t+Δt)) | 10–30x step size. **Subsumed by 5.x if JFNK includes σ_s** — decide before building both |
| 2.2 | **Kill chemistry re-bake retraces**: pass frozen heavy densities as runtime arguments instead of baking them into the closure | 2–5x wall clock, cheapest item on the list |
| 2.3 | **Profile** before optimising anything else | the 6 s/iteration has never been attributed |
| 2.4 | **Verify FP64 hardware**: `JAX_PLATFORMS=cpu` control run | consumer cards run FP64 at 1/64 of FP32 peak; the whole stack is float64. If this is a GeForce card, every timing so far is misleading |
| 2.5 | **Implicit ions** | forced once the grid is refined (see 3.2). Do *after* 1.3, or you build an implicit solver around an ion velocity model that is wrong by a large factor exactly where it sets the timestep |

---

## 3. Grid refinement

| # | task | notes |
|---|---|---|
| 3.1 | **Graded mesh generator** + axisymmetric metrics for the graded case | λ_D ≈ 0.044 mm at 1e17, 0.020 mm at 5e17. Uniform at λ_D/3 is 8e7–4e8 cells (infeasible); graded with ~40 cells across the sheath is ~1e5 cells with a 200–400x cell-size span |
| 3.2 | Re-measure the ion Courant limit after refinement | on a resolved mesh dt_i ∝ λ_D²/(μ_i Φ) ∝ **1/n_e** — density hurts linearly. Expect ~1e-12 s at converged density, four decades below today. This is what forces 2.5 |
| 3.3 | Poisson solver must be replaced *before* refining (section 4) | otherwise a stalled run is ambiguous between physics and solver |

---

## 4. Sparse direct Poisson (LU / Cholesky with nested dissection)

**Key structural fact: the augmented Poisson operator is symmetric.**
The P2 augmentation adds face conductances symmetrically to the
Laplacian — which is why the current solver can legitimately pass
`symmetric=True` to CG. So this is **SPD → Cholesky**, not general LU:
roughly 2x cheaper, and it admits symbolic/numeric separation.

| # | task |
|---|---|
| 4.1 | New module `poisson_direct.py`. Assemble CSR from the existing `PoissonCoeffs` (`g_r`, `g_z`, `ga_r`, `ga_z`, `dcp_*`, masks) — all the pieces exist; this is assembly, not new discretisation |
| 4.2 | **Symbolic analysis once**, numeric factorisation per coefficient update. The sparsity structure is fixed by geometry and never changes; only the values change each Gummel sweep. Use CHOLMOD (`scikit-sparse`: `analyze()` once, `cholesky()` per update) or PyPardiso. **SciPy's `splu` does not expose symbolic reuse** — that matters, because refactorisation cost is what you pay, not the solve |
| 4.3 | Wrap in `jax.lax.custom_linear_solve` with `solve` and `transpose_solve`. For SPD, A = Aᵀ so `transpose_solve` is the same factorisation. Call through `jax.pure_callback` |
| 4.4 | **Validation gate**: at a sheath-resolving mesh, `wall_flux="thermal"` and `"sheath"` must converge to the same answer. If they don't, one is wrong. This is what turns 0.2 from an assertion into a result |

**Scope honestly.** `pure_callback` forces a host sync, does not
`vmap`, and sparse direct factorisation on GPU is weak and irregular.
So the direct path is:

- the **reference solver** (validates the iterative path exactly),
- the **small-mesh / CPU production** path,
- **not** the GPU or batch path.

At 1e5 unknowns in 2D, factorisation is fractions of a second and
back-substitution sub-millisecond, so it will likely beat the current
Jacobi-CG outright on a single case. It will not carry `vmap`, which is
where the actual differentiation-from-COMSOL lives — hence 4.5.

| 4.5 | **Geometric multigrid** as the GPU/batch path, behind the same `poisson_solver=` flag |

Jacobi-CG iterations scale as √κ with κ ~ (L/Δx_min)²: ~58 on the
current mesh, ~9700 on a sheath-graded one. Multigrid is ~8 at either —
mesh-independent, a different complexity class rather than a constant
factor. Two caveats: with 200–400x grading, **point smoothers fail on
the anisotropy** (needs line relaxation along the graded direction, or
semi-coarsening); and the plasma/dielectric permittivity jump needs
operator-dependent interpolation or multigrid loses its
mesh-independence.

---

## 5. JFNK — scope and staging

**Scope: the coupled steady residual, not a single PDE.** Take
u = (n_e, n_i, n_ε, Φ) and define R(u) = 0 as the simultaneous steady
form of electron continuity, ion continuity, electron energy and
Poisson. Newton on that; `jax.jvp` supplies exact Jacobian-vector
products — no hand-derived Jacobian, no finite differencing.

| # | task |
|---|---|
| 5.1 | Factor the steady residual `R(u)` out of the existing steppers as a standalone function (it is already implicit in what `step_imp` computes; this is refactoring, not new physics) |
| 5.2 | JVP via `jax.jvp`; Krylov via `jax.scipy.sparse.linalg.gmres` inside `custom_linear_solve` with an **explicit `transpose_solve`** — the Jacobian is non-symmetric (upwinding), so `symmetric=True` would corrupt gradients while leaving forward results correct. This exact trap already cost one debugging cycle |
| 5.3 | **Block preconditioner**: Poisson block from 4.2/4.5, transport blocks from the existing `op_diag` Jacobi or an ILU. JFNK does not remove the need for a good Poisson solver — it depends on it |
| 5.4 | **Globalisation by pseudo-transient continuation**: let Δt grow as ‖R‖ falls. (I/Δt + J)δu = −R *is* the Newton step as Δt → ∞, so this is a continuous migration from the current solver, not a rewrite. The ramp gate is already a crude version of this logic |

**Keep outside the Newton block:** the EMM (linear frequency-domain
A_φ, coupled only through σ(n_e)); the metastables (slow, and Eq. 56
acceleration already handles them). Full-system Newton including
chemistry is badly conditioned — segregated Newton on the fast coupled
set is the right target.

**Decide early whether σ_s is inside or outside.** If inside, JFNK
subsumes task 2.1 and that work should not be done twice.

Expected: this is where the ~2000x algorithmic gap against COMSOL
closes. Current cost is ~400 outer x 100 substeps x 3 solves ≈ 1.2e5
linear solves; a fully-coupled Newton is 20–50 solves.

---

## 6. Validation and infrastructure

| # | task |
|---|---|
| 6.1 | Re-run `test_implicit_electrons.py` now that the warm state is physical — the gradient test in particular was previously measuring a solution parked on a clamp |
| 6.2 | Convergence criterion on the **residual**, not on `rel`. `rel` floors at the ignition growth rate (~2%/iteration), which is why the ramp gate stalled |
| 6.3 | Compare converged n_e, T_e, Φ against the doc Sec. 13 GEC targets |
| 6.4 | Golden-value regression harness across the flag matrix (`es_joule`, `wall_flux`, `implicit_electrons`, `poisson_solver`) — there are now enough switches that silent divergence between them is likely |
| 6.5 | Batch throughput benchmark: `vmap` over N operating points, wall time vs N. **This is the number that differentiates the product**, not single-case speed |

---

## Suggested order

1. **0.1–0.3** (decisions + doc), **2.2**, **2.3**, **2.4** — cheap, unblock measurement
2. **1.1, 1.2** — accuracy floor; may reframe the EMCS timeline
3. **1.3** (Richards) — physics + relaxes the ion bound before you build around it
4. **4.1–4.4** (direct Poisson as reference and CPU path)
5. **5.1–5.4** (JFNK), deciding σ_s scope first
6. **3.1–3.2** (refinement) with **4.5** (multigrid) in place
7. **2.5** (implicit ions), now forced and now measurable
8. **1.4–1.7**, then **1.9–1.10**
9. Monte Carlo electrons (EMCS)

Sections 1.4–1.7 can proceed in parallel with 4–5 by anyone not
touching the solver core.
