"""
test_hybrid.py
==============
Validation of the outer hybrid iteration (doc Secs. 12.3-12.5): the
converged, self-consistent, all-fluid 2D argon ICP steady state on the
GEC reference cell.

1. Power ledger closure at convergence (500 W, 10 mTorr):
   P_set (+ the small Option A electrostatic Joule term) = electron wall
   loss + elastic exchange + net inelastic (excitation - superelastic +
   ionization cost + stepwise), imbalance < 2%. The excitation share's
   exit route (Ar* wall de-excitation / radiation / quenching) is
   cross-checked against the Ar* wall-loss integral.
2. GEC targets (doc Sec. 13) at 500 W, 10 mTorr: peak ne in the
   1e11-1e12 cm^-3 band (fluid-tier x2 allowance, Sec. 4.4); plasma
   potential 10-30 V tracking the reactor-averaged Te; toroidal Q_ind
   (~2 W/cm^3 peak) within a skin depth of the dielectric window;
   Ar* > n_i with the stepwise ionization source markedly more uniform
   than the ground-state channel.
3. Power scan 200-1000 W: near-linear peak-ne versus power with a
   sustainment threshold ~100 W by linear extrapolation; pressure point
   at 5 mTorr: plasma potential rises versus 10 mTorr.
4. The doc Sec. 4.4 fluid-tier deviations are printed PROMINENTLY at
   the head of every test (hybrid.FLUID_TIER_NOTICE) so the numbers are
   read against the right fidelity ceiling.

Runtime: each converged case is minutes of CPU (hundreds of jitted
sub-sliced outer iterations); the full suite runs several cases. Set
WAFERKINETIC_FAST=1 for a smoke run (looser tol, capped iterations) --
smoke mode exercises the machinery but is NOT validation-grade.

Run:  python test_hybrid.py
"""

import os

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

try:  # package layout
    from waferkinetic.solvers import hybrid
except ImportError:  # flat layout
    import hybrid

from gec_case import gec_setup, TURNS

FAST = os.environ.get("WAFERKINETIC_FAST", "0") not in ("0", "", "false")

# window bottom face over the coil region (gec_case Z_SLAB_BOT)
Z_WINDOW = 0.040

#: converged states shared across tests, keyed by (P_set_W, p_torr)
_CACHE: dict = {}


def _numerics() -> hybrid.HybridNumerics:
    if FAST:
        return hybrid.HybridNumerics(tol=1.0e-4, max_outer=120,
                                     min_unaccel=5)
    return hybrid.HybridNumerics()


def _converge(P_set: float, p_torr: float,
              warm_from: tuple | None = None):
    """Build + converge one operating point (memoized). Warm starts from
    the cached state of `warm_from` when available, scaling ne/ni/n_eps
    with the power ratio as the near-linear scaling suggests."""
    key = (P_set, p_torr)
    if key in _CACHE:
        return _CACHE[key]
    mesh, mask = gec_setup()
    case = hybrid.HybridCase(P_set_W=P_set, p_torr=p_torr)
    setup = hybrid.build_setup(case, mesh, mask, TURNS)
    num = _numerics()

    state = None
    if warm_from is not None and warm_from in _CACHE:
        from dataclasses import replace
        src = _CACHE[warm_from]
        s = P_set / warm_from[0]
        st = src["state"]
        state = replace(hybrid.seed_state(setup),
                        ne=st.ne * s, ni=st.ni * s, n_eps=st.n_eps * s,
                        n_ars=st.n_ars * s, Phi=st.Phi.copy(),
                        ss_r=st.ss_r.copy(), ss_z=st.ss_z.copy(),
                        A=st.A.copy() * np.sqrt(s), I_coil=st.I_coil,
                        ni_prev=st.ni * s, ars_prev=st.n_ars * s)

    print(f"\n--- converging GEC hybrid @ {P_set:.0f} W, "
          f"{p_torr * 1e3:.0f} mTorr ---")
    state, history, converged = hybrid.run_hybrid(setup, num, state=state,
                                                  verbose=True)
    assert converged, (f"outer iteration did not converge at {P_set} W / "
                       f"{p_torr * 1e3:.0f} mTorr within {num.max_outer} "
                       f"iterations (last rel {history[-1]['rel']:.2e})")
    Q = hybrid.heating_field(setup, state)
    bundle = dict(case=case, setup=setup, state=state, history=history,
                  Q=Q, ledger=hybrid.power_ledger(setup, state, Q))
    _CACHE[key] = bundle
    return bundle


# ------------------------------------------------------------------ test 1 --
def test_power_ledger():
    print(hybrid.FLUID_TIER_NOTICE)
    b = _converge(500.0, 0.010)
    led = b["ledger"]
    ch = led["channels"]
    print(f"[1] power ledger @ 500 W, 10 mTorr (converged):")
    print(f"    P_set  = {led['P_in']:8.2f} W   (+ ES Joule "
          f"{led['P_es']:.2f} W, {led['P_es'] / led['P_in']:.1%} of P_set)")
    print(f"    wall,e = {led['P_wall']:8.2f} W")
    print(f"    elastic= {led['P_el']:8.2f} W")
    print(f"    inel   = {led['P_inel']:8.2f} W  "
          f"(exc {ch.get('excitation', 0.0):.1f}"
          f" + sup {ch.get('superelastic', 0.0):.1f}"
          f" + iz {ch.get('ionization', 0.0):.1f}"
          f" + step {ch.get('stepwise', 0.0):.1f})")
    print(f"    closure (with ES):    imbalance = {led['imbalance']:.2%}")
    print(f"    closure (P_set only): imbalance = "
          f"{led['imbalance_no_es']:.2%}")
    print(f"    Ar* wall/radiation exit channel: {led['P_ars_wall']:.1f} W "
          f"(cross-check vs net excitation "
          f"{ch.get('excitation', 0.0) + ch.get('superelastic', 0.0):.1f} W)")
    # the discrete identity that must close at the fixed point
    assert led["imbalance"] < 0.02, led["imbalance"]
    # the milestone headline: P_set alone closes to < 2% because the
    # Option A ES Joule term is small at ICP conditions
    assert led["imbalance_no_es"] < 0.02, led["imbalance_no_es"]
    assert led["P_es"] < 0.1 * led["P_in"], led["P_es"]


# ------------------------------------------------------------------ test 2 --
def test_gec_targets():
    print(hybrid.FLUID_TIER_NOTICE)
    b = _converge(500.0, 0.010)
    setup, state, Q = b["setup"], b["state"], b["Q"]
    case = b["case"]
    act = setup.top.active
    V = setup.top.volume
    Te = hybrid.electron_energy.temperature(state.ne, state.n_eps,
                                            setup.eparams)

    # (a) peak ne: Sec. 13 band 1e11-1e12 cm^-3 with the Sec. 4.4 x2
    #     fluid-tier headroom on the upper edge
    ne_pk = float(state.ne.max())
    print(f"[2a] peak ne = {ne_pk / 1e6:.3e} cm^-3  "
          f"(target 1e11-1e12 cm^-3; fluid tier may sit up to ~2x high)")
    assert 1.0e17 <= ne_pk <= 2.5e18, ne_pk

    # (b) plasma potential 10-30 V tracking reactor-averaged Te
    Phi_pk = float(state.Phi[act].max())
    w = state.ne * V * act
    Te_avg = float(np.sum(Te * w) / max(np.sum(w), 1.0e-30))
    ratio = Phi_pk / max(Te_avg, 1.0e-30)
    print(f"[2b] plasma potential = {Phi_pk:.1f} V, <Te> = {Te_avg:.2f} eV, "
          f"Phi/<Te> = {ratio:.1f}  (target 10-30 V, sheath-factor ~4-6)")
    assert 8.0 <= Phi_pk <= 35.0, Phi_pk
    assert 2.5 <= ratio <= 8.0, ratio

    # (c) toroidal Q_ind: ~2 W/cm^3 peak within a skin depth of the window
    Q_pk = float(np.where(act, Q, 0.0).max())
    pen, (i, j) = hybrid.field_penetration_depth(setup, Q)
    z_pk = float(setup.mesh.z_c[j])
    delta = hybrid.collisional_skin_depth(ne_pk, case.nu_m, case.omega)
    gap = Z_WINDOW - z_pk
    print(f"[2c] Q_ind peak = {Q_pk / 1e6:.2f} W/cm^3 at z = "
          f"{z_pk * 1e3:.1f} mm ({gap * 1e3:.1f} mm below the window); "
          f"skin depth = {delta * 1e3:.1f} mm, 1/e penetration = "
          f"{pen * 1e3:.1f} mm")
    assert 0.5e6 <= Q_pk <= 10.0e6, Q_pk           # ~2 W/cm^3 target
    assert 0.0 <= gap <= 1.5 * delta, (gap, delta)

    # (d) Ar* exceeds the ion density; stepwise source more uniform than
    #     the ground-state channel
    ars_avg = float(np.sum(state.n_ars * V * act) / np.sum(V * act))
    ni_avg = float(np.sum(state.ni * V * act) / np.sum(V * act))
    print(f"[2d] <Ar*> = {ars_avg:.3e} m^-3 vs <Ar+> = {ni_avg:.3e} m^-3; "
          f"peaks {state.n_ars.max():.3e} vs {state.ni.max():.3e}")
    assert ars_avg > ni_avg
    assert float(state.n_ars.max()) > float(state.ni.max())

    n_Ar = hybrid._heavy_ground_state(setup, state.n_ars)
    S_gs = setup.rxn["ionization"].k(Te) * state.ne * n_Ar
    S_sw = setup.rxn["stepwise"].k(Te) * state.ne * state.n_ars

    def peak_to_mean(S):
        m = float(np.sum(S * V * act) / np.sum(V * act))
        return float(np.where(act, S, 0.0).max()) / max(m, 1.0e-30)

    u_gs, u_sw = peak_to_mean(S_gs), peak_to_mean(S_sw)
    print(f"     source uniformity (peak/mean): ground-state {u_gs:.1f}, "
          f"stepwise {u_sw:.1f}  (stepwise must be flatter)")
    assert u_sw < u_gs, (u_sw, u_gs)


# ------------------------------------------------------------------ test 3 --
def test_power_scan_and_pressure():
    print(hybrid.FLUID_TIER_NOTICE)
    b500 = _converge(500.0, 0.010)
    b200 = _converge(200.0, 0.010, warm_from=(500.0, 0.010))
    b1000 = _converge(1000.0, 0.010, warm_from=(500.0, 0.010))

    P = np.array([200.0, 500.0, 1000.0])
    ne = np.array([float(b["state"].ne.max())
                   for b in (b200, b500, b1000)])
    a, c = np.polyfit(P, ne, 1)                     # ne ~ a P + c
    P0 = -c / a
    resid = np.abs(ne - (a * P + c)) / ne.max()
    print(f"[3a] density-vs-power: ne_pk(200/500/1000 W) = "
          f"{ne[0]:.3e} / {ne[1]:.3e} / {ne[2]:.3e} m^-3")
    print(f"     linear fit ne = a(P - P0): threshold P0 = {P0:.0f} W "
          f"(target ~100 W), max fit residual {resid.max():.1%}")
    assert a > 0.0
    assert 0.0 < P0 < 300.0, P0                     # ~100 W sustainment
    assert resid.max() < 0.15, resid                # near-linear scaling

    # pressure point: plasma potential rises at 5 mTorr (Sec. 13: steep
    # rise below ~4-5 mTorr; 5 vs 10 mTorr must already trend upward)
    b5 = _converge(500.0, 0.005, warm_from=(500.0, 0.010))
    act10 = b500["setup"].top.active
    act5 = b5["setup"].top.active
    Phi10 = float(b500["state"].Phi[act10].max())
    Phi5 = float(b5["state"].Phi[act5].max())
    print(f"[3b] plasma potential: {Phi10:.1f} V @ 10 mTorr -> "
          f"{Phi5:.1f} V @ 5 mTorr (must rise)")
    assert Phi5 > Phi10, (Phi5, Phi10)


if __name__ == "__main__":
    if FAST:
        print("*** WAFERKINETIC_FAST smoke mode: NOT validation-grade ***")
    test_power_ledger()
    test_gec_targets()
    test_power_scan_and_pressure()
    print("\nall hybrid milestone checks passed")
    print("(reminder: read the numbers against the Sec. 4.4 fluid-tier "
          "notice printed above)")