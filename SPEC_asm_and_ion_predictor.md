# Design spec: doc P6 (analytic sheath) and Eq. 35 (ion predictor)

Written before any code, per the standing preference. Both items are
already **in the model document** — this is implementing options the
spec offers, not deviating from it, which is a different posture from
the `es_joule` / `wall_flux` flags.

**Ranking up front.** P6 is high value on *both* physics and speed and
should go first. Eq. 35 is worth doing but I do **not** expect it to
relax the limit that is actually binding us — see §B.4. If time is
short, do P6 alone.

---

## A. Doc P6 / Kushner Option 6 — analytic sheath module (ASM)

### A.1 What the doc specifies

> **P6. Sheath jump condition (ASM).** When the mesh does not resolve
> the sheath, an analytic potential jump is added at the plasma
> boundary,
> −∇·ε∇(Φ(t) + ΔΦ_b(t)) = Σ q_i N_i(t) + ρ_s(t),  (34)
> with ΔΦ_b from a collisionless Child–Langmuir sheath between the last
> mesh point and the surface: **positive ions pass unhindered; electron
> (and negative-ion) fluxes are reflected according to Boltzmann
> statistics.**

And, importantly for expectations:

> In high-density, low-pressure discharges the sheath is thin and
> collisionless, and bulk parameters (density, peak potential) computed
> with and without the analytic sheath differ by only a few percent,
> the principal difference being a flatter presheath; the necessity of
> resolving the sheath diminishes accordingly.

That last sentence is the argument for doing this **instead of** chasing
sheath-resolving mesh refinement.

### A.2 The closure

The mesh no longer tries to hold the sheath. The last plasma cell
becomes the **sheath edge**, quasineutral by construction, and the
entire drop lives in ΔΦ_b outside the mesh.

**Ion flux (replaces Eq. 37).** "Pass unhindered" = the Bohm criterion
at the sheath edge:

    n·Γ_i = n_se u_B,     u_B = sqrt(q T_e / M_i)

No thermal term, no drift term, no dependence on the (now absent)
wall-cell field. This is also what makes the ion flux independent of Φ
again, so the P2/P3 semi-implicit coupling is **preserved** — unlike
the momentum-equation route, where Kushner notes "the direct
semi-implicit solution for the ion fluxes with Φ is lost."

**Electron flux (Eq. 36 with the Boltzmann barrier).**

    n·Γ_e = c_p v_th,e n_se exp(−ΔΦ_b / T_e),
    c_p = (1−r_e)/(1+r_e) · 1/2,   v_th,e = sqrt(8 q T_e / π m_e)

**Energy flux (Eq. 36 second relation)** carries the same barrier
factor; electrons that surmount ΔΦ_b arrive with ~2 T_e each. Ions
arrive with ΔΦ_b + presheath energy — that term is currently missing
from the ledger entirely and should be added as `P_ion_wall` (it is
also the quantity the future IEAD work needs).

**Determining ΔΦ_b.**

*Floating surfaces (dielectric).* Zero net current gives ΔΦ_b
analytically:

    n_se u_B = c_p v_th,e n_se exp(−ΔΦ_b/T_e)
    ⇒ ΔΦ_b = T_e ln( c_p v_th,e / u_B )

Both speeds scale as sqrt(T_e), so **the ratio is a constant of the gas
and r_e alone**: for argon with r_e = 0.2, c_p v_th,e/u_B = 143.5 and

    ΔΦ_b = 4.97 T_e     (14.9 V at 3 eV, 22.3 V at 4.5 eV)

*Conducting surfaces (grounded metal, later RF-biased).* The surface
potential is imposed, so ΔΦ_b = Φ_se − Φ_surface is whatever the global
solution requires; metal may carry net current, and current continuity
closes through the circuit rather than face by face.

**Sheath thickness (diagnostic, and the consistency check).**
Child–Langmuir with the Bohm ion current:

    s = (sqrt2/3) λ_D (2 ΔΦ_b / T_e)^{3/4}

Not needed to advance the solution, but it is the number that tells you
whether the ASM assumption is self-consistent: **s must stay smaller
than the wall cell**, otherwise the sheath is partly inside the mesh and
you are double-counting.

### A.3 What I would modify

| site | change |
|---|---|
| `poisson.i_fluxes` | wall term → `n_se * u_B(Te)`; drop the thermal + gated-drift form on the ASM path. Interior fluxes unchanged. |
| `sheath_r` / `sheath_z` (jitted) | barrier becomes ΔΦ_b from the ASM closure, not the cell-to-cell Φ difference. For floating faces this is the analytic 4.97·T_e; for conducting faces it is Φ_cell − Φ_surface. |
| Poisson boundary | Eq. 34: conducting-surface Dirichlet value becomes Φ_surface + ΔΦ_b, i.e. the mesh sees the **sheath edge**, not the metal. This is the one structurally new piece — it touches `PoissonOperator`'s Dirichlet assembly, not just a flux. |
| electron energy wall flux | same barrier factor (already threaded through `wall_flux`). |
| `power_ledger` | same barrier; **add `P_ion_wall` = Σ A Γ_i q(ΔΦ_b + ~0.5 T_e)**, currently missing. Without it the ledger will not close once ions carry the sheath energy. |
| `_species_fluxes` (NumPy) | mirror all of the above so dumps match the run. |
| in-loop clips | the sheath-RC clip should be recomputed — with a quasineutral wall cell its premise largely disappears (§A.5). |
| flag | `sheath_model="resolved"|"asm"`, default `"resolved"`, threaded like the others. |

### A.4 Predictions (so this is falsifiable)

1. **Window (floating) ΔΦ/T_e: 2.9 → 4.97.** By construction. If it
   lands anywhere else, the implementation is wrong.
2. **Wafer (grounded) 6.5 → set by the global current balance**, no
   longer by an under-resolved cell. I would not predict a specific
   number; I would predict it stops depending on mesh spacing, which is
   testable by running coarse and fine.
3. **Φ_pk falls** toward T_e·(4.97 + presheath) ≈ 5–6 T_e ≈ 25 V.
4. **Bulk n_e and T_e change by only a few percent** — the doc says so
   explicitly, and if they move a lot something is wrong.

### A.5 The numerical bonus, which may dominate the value

With the sheath outside the mesh, the wall cell is quasineutral and the
in-mesh field at the boundary becomes **presheath scale**:

    E_presheath ~ T_e/L ≈ 225 V/m    vs    3.62e4 V/m measured now   (161×)

Consequences: the ion Courant bound (∝ 1/μ_i E) relaxes by up to that
factor, and the **sheath-RC clip — which our bisection identified as the
binding constraint at ~1e-10 s/substep — largely loses its premise**,
because there is no resolved charge separation left to relax. This
could be worth more than everything in the roadmap's numerics section,
and it comes free with the physics fix.

It also makes `--dt-cap` and the mesh-refinement cost estimate moot:
you would no longer be paying for cells that exist only to (badly) hold
a sheath.

### A.6 Risks and open questions

- **Presheath boundary condition.** With the sheath removed, what
  condition does Poisson get at the sheath edge on the *plasma* side?
  Eq. 34 as written adds the jump to the Dirichlet value, which is the
  simplest reading and what I would implement first. The alternative
  (zero normal field at the sheath edge) changes the presheath profile.
  Worth checking against the doc's "flatter presheath" remark.
- **Bohm criterion needs a sheath-edge density.** Using the wall-cell
  n_i is standard but the cell is not exactly at the Bohm point;
  expect a factor ~0.6 (the usual h_l edge-to-centre factor) of
  ambiguity. I would not fold in an h-factor initially — one change at
  a time.
- **σ_s dynamics.** The floating-surface ΔΦ_b already encodes zero net
  current, which is what σ_s was doing dynamically. These must not
  both act, or the surface charge and the analytic barrier will fight.
  My reading: σ_s continues to set Φ_surface for the dielectric via
  Gauss's law, and ΔΦ_b sits on top; but this is the part I am least
  sure of and would want to reason through with you before coding.
- Consistency of the barrier across all four sites (flux, energy flux,
  RC clip, ledger) — the same class of mismatch that produced the
  EETM-slice and σ_s-dump bugs. One helper, called everywhere.

---

## B. Doc Eq. 35 — short-history ion predictor

### B.1 What the doc specifies

> **Ion predictor.** Including ion momentum implicitly is burdensome
> with many ion species; stability at large Δt is instead recovered
> with a short-history predictor,
> N(t+Δt) = N(t) + Δt[ −∇·(Γ(t) + Δt dΓ/dt) + S(t) + Δt dS/dt ],  (35)
> with derivatives formed numerically from recorded flux and source
> histories.

### B.2 The scheme

With backward differences on the recorded history,

    dΓ/dt ≈ (Γⁿ − Γⁿ⁻¹)/Δt_prev,     dS/dt ≈ (Sⁿ − Sⁿ⁻¹)/Δt_prev

the bracket becomes a **linear extrapolation of flux and source to
t+Δt**:

    Γ* = Γⁿ + (Δt/Δt_prev)(Γⁿ − Γⁿ⁻¹),   S* likewise
    Nⁿ⁺¹ = Nⁿ + Δt(−∇·Γ* + S*)

For constant Δt this is Γ* = 2Γⁿ − Γⁿ⁻¹. Note this is *more* aggressive
than Adams–Bashforth 2 (which uses 1.5Γⁿ − 0.5Γⁿ⁻¹); AB2 is the safer
variant and I would implement the doc form with an option for the AB2
weights, since the two differ only in two coefficients.

### B.3 What I would modify

| site | change |
|---|---|
| `step_imp` carry | add `Fr_i_prev`, `Fz_i_prev`, `S_prev` (two face arrays, one cell array), zero-initialised; first substep falls back to plain Euler. |
| ion update | advance with Γ*, S* instead of Γⁿ, Sⁿ. |
| P2 ion predictor `ni_p` | must use the **same** Γ*, or the Poisson RHS and the ion update disagree. |
| σ_s ledger | must charge with the **same** Γ*, or the surface-charge balance leaks — this is the exact failure mode we hit three times already. |
| positivity limiter | applied to Γ*, not Γⁿ. |
| flag | `ion_predictor="euler"|"history"`, default `"euler"`. |

No new state on `HybridState` — the history lives inside the substep
loop, so it resets each outer iteration. That is consistent with the
doc's "recorded flux and source histories" being a short history.

### B.4 Honest assessment of what it buys

The doc places Eq. 35 in §5.3, among the **Poisson options** — its role
is to make the *density prediction fed to the semi-implicit Poisson
solve* accurate enough to stay stable at large Δt. It is not a fix for
the advective CFL condition: a two-level explicit extrapolation does
not enlarge the advection stability region (AB2's is in fact smaller
than forward Euler's on the imaginary axis).

Our measured binding constraints are the **sheath-RC clip** (~1e-10 s)
and the **ion Courant limit**, not the Φ↔ρ prediction. So I expect
Eq. 35 to buy little *at present*, and to become useful mainly after
P6 removes the first of those — at which point the field-charge
coupling could become binding again and this is the documented remedy.

That is why the ordering matters: **P6 first, then re-measure the
binding clip, then decide whether Eq. 35 earns its complexity.**

---

## C. Suggested order

1. **P6 / ASM** — the physics fix for the sheath drop *and*, probably,
   the largest single speed win available. Resolve the σ_s/ΔΦ_b
   interaction question (§A.6) before coding.
2. **Re-measure** the binding clip with the bisection method that found
   the current one (`--dt-cap` sweep), and re-run the coarse/fine mesh
   pair as a grid-convergence check — now cheap, because the answer
   should no longer depend on wall-cell size.
3. **Eq. 35** only if the Φ↔ρ coupling is what binds after that.
4. **μ_i(E/N)** as a physics refinement, no longer a numerical rescue.
5. Full ion momentum (Eq. 25) as a later tier, with its known cost:
   loss of the semi-implicit ion–Φ coupling.

## D. Validation for P6

- Floating-surface ΔΦ/T_e must equal 4.97 to solver tolerance.
- Coarse vs fine mesh: bulk n_e, T_e, Φ_pk within a few percent, per
  the doc's own claim. This is the test that the sheath is genuinely
  out of the mesh.
- Child–Langmuir s vs wall cell size must satisfy s < Δx everywhere, or
  the ASM assumption is violated where it fails.
- Ledger must close with the new `P_ion_wall` channel included.
- `test_implicit_electrons.py` re-run: consistency, ledger residual,
  positivity, gradient — all four should be unaffected, since none of
  this touches the implicit operator itself.
