"""
test_implicit_electrons.py
==========================
Validation of the P3 implicit electron path (poisson.py
`implicit_electrons` / `scheme="p2i"`; module docstring P3, Kushner
2009 option 3 disambiguated) against the explicit FKPM path.

None of the pre-existing tests exercise this code: `HybridNumerics`
defaults to `implicit_electrons=False`, so `test_hybrid.py` and
`demo_hybrid_gec.py` still run the explicit stepper bit-for-bit. This
file is the entry point for the implicit path.

What is checked, in increasing cost:

  1. consistency -- at the EXPLICIT clock (where forward Euler is
     stable and both schemes are accurate) implicit and explicit
     single substeps agree to O(dt * rate), i.e. to the Courant
     fraction. This is the "did I discretize the same operator"
     test: a wrong sign, a missing wall term or a transposed face
     index shows up here as a large discrepancy.
  2. ledger + positivity -- the NumPy reference solve returns the
     operator fluxes AT the solution; the discrete balance
     (n - n_old)/dt + div F(n) - S must vanish to solver tolerance,
     which is exactly the property that makes the sigma_s wall
     charging exact. Positivity is checked at a dt far beyond the
     explicit floor (backward Euler + donor cell = M-matrix).
  3. stiffness relief -- the implicit clock vs the explicit clock,
     cell by cell. This is the headline number: the electron
     diffusion bound is gone, so what remains is the ion Courant /
     sheath-RC / chemical caps. Expect ~2-3 decades, NOT more.
  4. reverse-mode gradient -- d/ds of a scalar functional of the
     stepper output against a central finite difference. This is the
     one test that exercises `transpose_solve`; a wrong transpose
     leaves forward results perfect and corrupts gradients silently,
     so it is the highest-value check in the file.
  5. steady state -- explicit and implicit converged GEC solutions
     must agree on peaks and on the power ledger. Skipped in FAST
     mode (it converges the case twice).

Usage:
  python test_implicit_electrons.py            # all, incl. steady state
  WAFERKINETIC_FAST=1 python test_implicit_electrons.py   # 1-4 only
  pytest -s test_implicit_electrons.py
"""

import os
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

try:  # package layout
    from waferkinetic.solvers import hybrid, poisson, transport
except ImportError:  # flat layout
    import hybrid
    import poisson
    import transport

from gec_case import gec_setup, TURNS

FAST = os.environ.get("WAFERKINETIC_FAST", "0") not in ("0", "", "false")

P_SET, P_TORR = 500.0, 0.010
#: outer iterations of the ignition power ramp. Slicing from a state
#: parked on the Te_max clamp measures nothing useful (and carries no
#: gradient), so the warm-up must complete the ramp before any test
#: reads the state.
N_RAMP = 12 if FAST else 30
#: outer iterations run on the explicit path to get a physically
#: non-degenerate state to slice from (the seed profile has Phi = 0 and
#: no sheath, which would make the comparison vacuous)
N_WARM = N_RAMP + (3 if FAST else 10)

_STATE_CACHE: dict = {}


# ---------------------------------------------------------------- fixtures --
def _warm_state():
    """A partially relaxed GEC state + its setup, from the EXPLICIT path
    (so nothing under test contaminates the initial condition)."""
    if "warm" in _STATE_CACHE:
        return _STATE_CACHE["warm"]
    mesh, mask = gec_setup()
    case = hybrid.HybridCase(P_set_W=P_SET, p_torr=P_TORR)
    setup = hybrid.build_setup(case, mesh, mask, TURNS)
    num = hybrid.HybridNumerics(P_ramp_iters=N_RAMP)
    drv = hybrid.HybridDrivers(setup, num)
    state = hybrid.seed_state(setup)
    print(f"--- warming {N_WARM} explicit outer iterations "
          f"({P_SET:.0f} W, {P_TORR * 1e3:.0f} mTorr) ---")
    for it in range(1, N_WARM + 1):
        state, diag = hybrid.outer_iteration(setup, num, drv, state, it)
        led = diag["ledger"]
        print(f"    [{it:3d}] P={diag['P_target']:6.1f} W  "
              f"ne_pk={diag['ne_pk']:.3e}  Te_pk={diag['Te_pk']:5.2f}  "
              f"Phi_pk={diag['Phi_pk']:6.2f} V  "
              f"P_wall={led['P_wall']:.3e} W  imb={led['imbalance']:.1%}")
    # Sanity gate on the BASELINE. Every test below slices from this
    # state, so a diverged warm-up would make them measure nothing --
    # and divergence in the energy variable is invisible in Te, which
    # the steppers clamp at Te_max: n_eps can be at 1e50 while Te_pk
    # still reads a healthy 15 eV. P_wall is the unclamped diagnostic
    # that exposes it.
    led = hybrid.power_ledger(setup, state, hybrid.heating_field(setup, state))
    assert led["P_wall"] < 50.0 * setup.case.P_set_W, (
        f"the EXPLICIT warm-up diverged: P_wall = {led['P_wall']:.3e} W "
        f"against {setup.case.P_set_W:.0f} W deposited. n_eps has run "
        f"away (the Te clamp hides it). Nothing downstream of this is "
        f"meaningful -- fix the explicit baseline first.")
    assert np.all(np.isfinite(state.n_eps)) and np.all(np.isfinite(state.ne))
    _STATE_CACHE["warm"] = (setup, state)
    return setup, state


def _slice_inputs(setup, state):
    """(Te, Er, Ez, S_ext) as `outer_iteration` would assemble them."""
    Q = hybrid.heating_field(setup, state)
    S_ext = setup.active_f * Q / hybrid.QE
    Er, Ez = setup.pop.efield(state.Phi)
    Te = hybrid.electron_energy.temperature(state.ne, state.n_eps,
                                            setup.eparams)
    return Te, Er, Ez, S_ext


def _drivers(setup, *, implicit, n_sub, n_gummel=2, state=None):
    num = hybrid.HybridNumerics(n_sub_fkpm=n_sub,
                                implicit_electrons=implicit,
                                n_gummel=n_gummel)
    drv = hybrid.HybridDrivers(setup, num)
    drv.refresh_chem(state.n_ars, force=True)   # builds the jitted FKPM
    return num, drv


def _rel(a, b, mask):
    scale = max(float(np.abs(a[mask]).max()), float(np.abs(b[mask]).max()),
                1.0e-30)
    return float(np.abs(a - b)[mask].max()) / scale


# ------------------------------------------------------------------ test 1 --
def test_implicit_matches_explicit_at_explicit_clock():
    """At the explicit stability floor both schemes resolve the same
    dynamics, so one substep must agree to the Courant fraction. Both
    steppers are fed the IDENTICAL dt arrays (the explicit ones), which
    is the only apples-to-apples comparison: the implicit path's own
    clock is deliberately much larger."""
    setup, state = _warm_state()
    Te, Er, Ez, S_ext = _slice_inputs(setup, state)
    act = setup.top.active

    num_e, drv_e = _drivers(setup, implicit=False, n_sub=1, state=state)
    dt_e, dt_i, dt_eps = hybrid._local_dts(setup, num_e, state, Te, Er, Ez)

    num_i, drv_i = _drivers(setup, implicit=True, n_sub=1, n_gummel=4,
                            state=state)

    outs_e = drv_e.fkpm_step(state, dt_e, dt_i, dt_eps, S_ext)
    outs_i = drv_i.fkpm_step(state, dt_e, dt_i, dt_eps, S_ext)
    names = ("ne", "ni", "n_eps", "ss_r", "ss_z", "Phi")

    print("\n  one substep at the explicit clock "
          f"(dt_e mean {dt_e[act].mean():.3e} s):")
    for nm, a, b in zip(names, outs_e, outs_i):
        m = act if a.shape == act.shape else np.ones(a.shape, bool)
        d = _rel(a, b, m)
        print(f"    {nm:>6s}  max rel diff {d:.3e}")
        assert np.all(np.isfinite(b)), f"{nm}: implicit output not finite"
        assert d < 5.0e-2, (f"{nm}: implicit and explicit disagree by "
                            f"{d:.2e} at the explicit clock -- the two "
                            f"paths are not discretizing the same operator")

    # the discrepancy must also SHRINK with dt (first-order consistency):
    # halving the clock should roughly halve the difference
    outs_e2 = drv_e.fkpm_step(state, 0.5 * dt_e, 0.5 * dt_i, 0.5 * dt_eps,
                              S_ext)
    outs_i2 = drv_i.fkpm_step(state, 0.5 * dt_e, 0.5 * dt_i, 0.5 * dt_eps,
                              S_ext)
    d1 = _rel(outs_e[0], outs_i[0], act)
    d2 = _rel(outs_e2[0], outs_i2[0], act)
    print(f"    ne diff at dt {d1:.3e} -> at dt/2 {d2:.3e} "
          f"(ratio {d2 / max(d1, 1e-300):.2f}, first order ~0.5)")
    assert d2 <= 0.8 * d1 + 1.0e-12, (
        "the implicit-explicit difference does not shrink with dt: it is "
        "a discretization inconsistency, not truncation error")


# ------------------------------------------------------------------ test 2 --
def test_reference_solve_ledger_and_positivity():
    """The NumPy reference `_implicit_electron_solve` returns the
    operator fluxes at the solution; the discrete balance must close
    (this is what makes the sigma_s wall ledger exact) and the density
    must stay nonnegative at a dt far past the explicit floor."""
    setup, state = _warm_state()
    Te, Er, Ez, _ = _slice_inputs(setup, state)
    top, p = setup.top, setup.fkpm_params
    act = top.active

    # the same volumetric electron source the stepper sees (Table 2
    # ionization + stepwise + Penning at frozen heavies), nonnegative
    n_Ar = setup.active_f * np.maximum(setup.case.ng - state.n_ars,
                                       setup.nparams.n_floor)
    src_e = hybrid.chemistry.make_jax_source_fn(
        setup.chem, species=hybrid.chemistry.IE)
    S = np.asarray(src_e((jnp.asarray(state.ne), jnp.asarray(n_Ar),
                          jnp.asarray(state.n_ars),
                          jnp.zeros_like(jnp.asarray(state.ne))),
                         jnp.asarray(Te)))
    S = np.maximum(S, 0.0)

    num = hybrid.HybridNumerics()
    dt_expl, _, _ = hybrid._local_dts(setup, num, state, Te, Er, Ez)
    dt_ref = float(np.max(dt_expl[act]))

    for factor in (1.0, 1.0e2, 1.0e4):
        dt = np.where(act, factor * dt_ref, 0.0)
        n_new, (Fr, Fz) = poisson._implicit_electron_solve(
            top, state.ne, Te, Er, Ez, S, p, dt, tol=1.0e-12)
        dtc = np.where(dt > 0.0, dt, 1.0)
        resid = (n_new - state.ne) / dtc + transport.divergence(top, Fr, Fz) \
            - S
        scale = np.maximum(np.abs(n_new - state.ne) / dtc, np.abs(S)).max()
        rel = float(np.abs(resid[act]).max()) / max(scale, 1.0e-30)
        neg = float(np.minimum(n_new[act], 0.0).min())
        print(f"\n  dt = {factor:8.0e} x explicit floor: "
              f"balance residual {rel:.3e}, most negative n_e {neg:.3e}")
        assert np.all(np.isfinite(n_new))
        assert rel < 1.0e-6, ("the returned fluxes are not the fluxes of "
                              "the solved density: the sigma_s ledger "
                              "would leak charge")
        assert neg > -1.0e-6 * float(n_new[act].max()), (
            "backward Euler + donor-cell upwinding must be positivity "
            "preserving for S >= 0; negativity means the operator is not "
            "an M-matrix (check the upwind switch / wall signs)")


# ------------------------------------------------------------------ test 3 --
def test_stiffness_relief():
    """How much clock the implicit path actually buys. The electron
    diffusion bound (~ps on 0.3 mm cells at 10 mTorr) is gone; what
    remains is the ion Courant limit, the sheath-RC charging limit and
    the chemical caps -- expect 2-3 decades, not more."""
    setup, state = _warm_state()
    Te, Er, Ez, S_ext = _slice_inputs(setup, state)
    act = setup.top.active

    num_e = hybrid.HybridNumerics()
    num_i = hybrid.HybridNumerics(implicit_electrons=True)
    dte_e, dti_e, dtp_e = hybrid._local_dts(setup, num_e, state, Te, Er, Ez)
    dte_i, dti_i, dtp_i = hybrid._local_dts(setup, num_i, state, Te, Er, Ez)

    ge = float(np.median(dte_i[act] / np.maximum(dte_e[act], 1e-300)))
    gp = float(np.median(dtp_i[act] / np.maximum(dtp_e[act], 1e-300)))
    print(f"\n  pre-loop clock, median over plasma cells:")
    print(f"    electron  {dte_e[act].mean():.3e} -> {dte_i[act].mean():.3e} s"
          f"   (x{ge:.3g})")
    print(f"    energy    {dtp_e[act].mean():.3e} -> {dtp_i[act].mean():.3e} s"
          f"   (x{gp:.3g})")
    print(f"    ion (unchanged, binds next) {dti_e[act].mean():.3e} s")
    assert ge > 10.0, ("the implicit path did not relax the electron "
                       "clock -- is `implicit_electrons` reaching "
                       "`_local_dts`?")
    assert np.all(dti_i[act] == dti_e[act]), \
        "the ion Courant bound must be untouched by the implicit path"

    # and it must actually RUN there: 20 substeps at the implicit clock
    num, drv = _drivers(setup, implicit=True, n_sub=20, state=state)
    out = drv.fkpm_step(state, dte_i, dti_i, dtp_i, S_ext)
    ne, ni, n_eps = out[0], out[1], out[2]
    Te_out = hybrid.electron_energy.temperature(ne, n_eps, setup.eparams)
    print(f"    20 substeps at the implicit clock: ne_pk {ne.max():.3e}, "
          f"Te_pk {Te_out[act].max():.2f} eV")
    for nm, a in zip(("ne", "ni", "n_eps", "ss_r", "ss_z", "Phi"), out):
        assert np.all(np.isfinite(a)), f"{nm} went non-finite"
    assert ne[act].min() >= -1e-6 * ne.max()
    assert n_eps[act].min() >= -1e-6 * n_eps.max()
    assert Te_out[act].max() <= setup.case.Te_max + 1e-9


# ------------------------------------------------------------------ test 4 --
def test_reverse_mode_gradient():
    """THE transpose test. `custom_linear_solve` is given an explicit
    `transpose_solve`; had it been declared symmetric instead, forward
    results would be identical and gradients silently wrong.

    The functional differentiates w.r.t. a scale on the INPUT electron
    density, so the derivative path runs straight through the implicit
    electron solve (and its transpose). Differentiating w.r.t. the
    deposited power instead is a trap: it reaches n_e only via
    Te = (2/3) n_eps / n_e, which is CLAMPED at Te_max, so wherever the
    plasma sits on the clamp the derivative is identically zero and the
    test passes while measuring nothing. Both are checked, and a
    vacuous (identically zero) derivative is a failure."""
    setup, state = _warm_state()
    Te, Er, Ez, S_ext = _slice_inputs(setup, state)

    num, drv = _drivers(setup, implicit=True, n_sub=3, state=state)
    dt_e, dt_i, dt_eps = hybrid._local_dts(setup, num, state, Te, Er, Ez)

    rest = [jnp.asarray(a) for a in (state.ni, state.n_eps, state.ss_r,
                                     state.ss_z, state.Phi,
                                     np.zeros_like(state.ss_r),   # Ei_r
                                     np.zeros_like(state.ss_z),   # Ei_z
                                     dt_e, dt_i, dt_eps)]
    ne_j, S_j = jnp.asarray(state.ne), jnp.asarray(S_ext)
    w = jnp.asarray(setup.top.volume * setup.active_f)

    def f_ne(scale):                    # through the implicit solve
        out = drv._fkpm(scale * ne_j, *rest, S_j)
        return jnp.sum(out[0] * w)

    def f_pow(scale):                   # through the energy coupling
        out = drv._fkpm(ne_j, *rest, scale * S_j)
        return jnp.sum(out[2] * w)

    for nm, f, must_be_live in (("d(sum ne V)/d(ne scale) ", f_ne, True),
                                ("d(sum neps V)/d(P scale)", f_pow, False)):
        g = float(jax.grad(f)(1.0))
        h = 1.0e-5
        fd = float((f(1.0 + h) - f(1.0 - h)) / (2.0 * h))
        rel = abs(g - fd) / max(abs(fd), 1.0e-30)
        print(f"\n  {nm}:  adjoint {g:.8e}   central FD {fd:.8e}   "
              f"rel {rel:.2e}")
        assert np.isfinite(g), f"{nm}: adjoint returned non-finite"
        if must_be_live:
            assert abs(fd) > 0.0, (
                f"{nm}: the functional does not respond to the input at "
                f"all -- the gradient check is vacuous, fix the probe "
                f"before trusting a pass")
        elif fd == 0.0 and g == 0.0:
            print("    (identically zero: Te is on its clamp, so this "
                  "path carries no derivative -- not a transpose result)")
            continue
        assert rel < 2.0e-2, (
            f"{nm}: reverse-mode gradient disagrees with the finite "
            f"difference -- the implicit solve's transpose is wrong (a "
            f"symmetric declaration on this non-symmetric upwind "
            f"operator fails exactly this way)")


# ------------------------------------------------------------------ test 5 --
def test_steady_state_agreement():
    """The two paths must converge to the SAME fixed point: the implicit
    scheme changes only how the pseudo-transient is walked, not the
    steady-state residual it is walking to."""
    if FAST:
        print("\n  [skipped in WAFERKINETIC_FAST mode]")
        return
    mesh, mask = gec_setup()
    case = hybrid.HybridCase(P_set_W=P_SET, p_torr=P_TORR)
    setup = hybrid.build_setup(case, mesh, mask, TURNS)

    runs = {}
    for tag, num in (("explicit",
                      hybrid.HybridNumerics(P_ramp_iters=N_RAMP)),
                     ("implicit",
                      hybrid.HybridNumerics(P_ramp_iters=N_RAMP,
                                            implicit_electrons=True))):
        print(f"\n--- converging {tag} path ---")
        t0 = time.time()
        st, hist, ok = hybrid.run_hybrid(setup, num, verbose=True)
        assert ok, (f"{tag} path did not converge in {num.max_outer} outer "
                    f"iterations (last rel {hist[-1]['rel']:.2e})")
        Q = hybrid.heating_field(setup, st)
        runs[tag] = dict(state=st, hist=hist, wall=time.time() - t0,
                         led=hybrid.power_ledger(setup, st, Q),
                         Te=hybrid.electron_energy.temperature(
                             st.ne, st.n_eps, setup.eparams))

    e, i = runs["explicit"], runs["implicit"]
    act = setup.top.active
    print("\n  converged comparison:")
    print(f"    {'quantity':>16s} {'explicit':>12s} {'implicit':>12s}  rel")
    checks = (("ne peak", e["state"].ne.max(), i["state"].ne.max(), 0.10),
              ("Te peak", e["Te"][act].max(), i["Te"][act].max(), 0.10),
              ("Ar* peak", e["state"].n_ars.max(), i["state"].n_ars.max(),
               0.10),
              ("P_wall (W)", e["led"]["P_wall"], i["led"]["P_wall"], 0.10),
              ("P_inel (W)", e["led"]["P_inel"], i["led"]["P_inel"], 0.10))
    for nm, a, b, tol in checks:
        r = abs(b - a) / max(abs(a), 1.0e-30)
        print(f"    {nm:>16s} {a:12.4e} {b:12.4e}  {r:.2%}")
        assert r < tol, (f"{nm}: the implicit path converged to a "
                         f"different steady state ({r:.1%})")
    for tag, run in runs.items():
        print(f"    {tag:>16s}: {len(run['hist'])} outer iterations, "
              f"{run['wall']:.1f} s, ledger imbalance "
              f"{run['led']['imbalance']:.2%}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(hybrid.FLUID_TIER_NOTICE)
    if FAST:
        print("*** WAFERKINETIC_FAST: reduced warm-up, steady-state "
              "comparison skipped -- NOT validation-grade ***")
    tests = (test_implicit_matches_explicit_at_explicit_clock,
             test_reference_solve_ledger_and_positivity,
             test_stiffness_relief,
             test_reverse_mode_gradient,
             test_steady_state_agreement)
    failed = []
    for t in tests:
        print(f"\n{'=' * 74}\n{t.__name__}\n{'=' * 74}")
        t0 = time.time()
        try:
            t()
            print(f"  PASS ({time.time() - t0:.1f} s)")
        except AssertionError as exc:
            failed.append(t.__name__)
            print(f"  FAIL ({time.time() - t0:.1f} s): {exc}")
    print(f"\n{'=' * 74}")
    print("all implicit-path checks passed" if not failed
          else "FAILED: " + ", ".join(failed))