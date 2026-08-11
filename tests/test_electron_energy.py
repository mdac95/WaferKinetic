"""
test_electron_energy.py
=======================
Validation of the fluid electron closure (EETM Option A, doc Sec. 4.1).

1. 0D energy balance: uniform heating vs. elastic exchange -> analytic
   steady Te, exact; continuity wiring (dne = S_e dt) exact; closed
   (re = 1) transport conserves total energy to machine precision.
2. JAX path: jitted stepper matches the NumPy reference bit-for-bit
   tolerance; reverse-mode gradient of total energy w.r.t. the heating
   scale through all substeps agrees with finite differences.
3. GEC integration: inductive solve (1200 W absorbed, prescribed argon
   n_e) -> Q_ind feeds the energy equation -> steady Te(r, z); asserts
   plausible peak Te and global power balance
   P_in = P_wall + P_elastic + P_inelastic, writes gec_icp_te.png.

Run:  python test_electron_energy.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from waferkinetic.mesh.reactor_mesh import (Material, RectRegion,
                                            ReactorGeometry, Mesh2D)
from waferkinetic.solvers.inductive import (build_operator,
                                            coil_current_density,
                                            absorbed_power, power_deposition)
from waferkinetic.solvers.transport import build_transport, to_jax, QE, ME
from waferkinetic.solvers.electron_energy import (
    ElectronParams, electron_rhs, advance, energy_stable_dt, energy_local_dt,
    temperature, wall_energy_power, make_jax_energy_stepper)

from gec_case import gec_setup, gec_plasma_state, OMEGA, TURNS

M_AR = 39.948 * 1.66053906660e-27
MR_AR = ME / M_AR


# ------------------------------------------------------------------ test 1 --
def test_0d_balance_and_conservation():
    mesh = Mesh2D(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4))
    mask = np.full((3, 3), int(Material.PLASMA), dtype=np.int32)
    op = build_transport(mesh, mask)
    shape = (3, 3)
    Er = np.zeros((4, 3))
    Ez = np.zeros((3, 4))

    # (a) heating vs elastic exchange, reflecting walls (re = 1)
    p = ElectronParams(nu_m=5.0e9, mass_ratio=MR_AR, Tg_eV=0.03, re=1.0)
    ne = np.full(shape, 1.0e16)
    dTe = 1.7
    Q = 3.0 * p.mass_ratio * p.nu_m * ne * dTe * QE       # W/m^3
    n_eps = 1.5 * ne * 0.5                                # Te(0) = 0.5 eV
    dt = 1.0e-6
    _, n_eps, Te = advance(op, ne, n_eps, Er, Ez, Q / QE, p, dt, 400)
    err = abs(Te.max() - (p.Tg_eV + dTe)) / (p.Tg_eV + dTe)
    assert err < 1e-8, err
    print(f"[1a] 0D power balance:           Te = {Te.max():.6f} eV vs "
          f"{p.Tg_eV + dTe:.6f} analytic (rel err {err:.1e})")

    # (b) continuity wiring: dne = S_e dt exactly
    S_e = np.full(shape, 2.0e18)
    ne2, _, _ = advance(op, ne.copy(), n_eps.copy(), Er, Ez, Q / QE, p,
                        dt, 10, S_e=S_e, evolve_ne=True)
    err = np.max(np.abs(ne2 - (ne + 10 * dt * S_e))) / ne.max()
    assert err < 1e-12, err
    print(f"[1b] continuity source wiring:   |dne - S_e dt| rel = {err:.1e}")

    # (c) closed-domain energy conservation under nonlinear transport
    mesh = Mesh2D(np.linspace(0.0, 0.05, 13), np.linspace(0.0, 0.05, 13))
    op = build_transport(mesh, np.full((12, 12), int(Material.PLASMA),
                                       np.int32))
    p2 = ElectronParams(nu_m=6.0e7, mass_ratio=0.0, Tg_eV=0.03, re=1.0)
    ne = np.full((12, 12), 5.0e15)
    n_eps = 1.5 * ne * (1.0 + 3.0 * np.exp(
        -((mesh.RC - 0.02) ** 2 + (mesh.ZC - 0.03) ** 2) / 0.01 ** 2))
    Er = np.zeros((13, 12)); Ez = np.zeros((12, 13))
    dt = 0.5 * energy_stable_dt(op, ne, n_eps, Er, Ez, p2, Te_ref=10.0)
    E0 = np.sum(n_eps * op.volume)
    _, n_eps, _ = advance(op, ne, n_eps, Er, Ez, np.zeros((12, 12)), p2,
                          dt, 200)
    rel = abs(np.sum(n_eps * op.volume) - E0) / E0
    assert rel < 1e-12, rel
    print(f"[1c] closed-domain energy:       |dE|/E = {rel:.1e} "
          f"over 200 steps (re = 1)")


# ------------------------------------------------------------------ test 2 --
def test_jax_matches_numpy_and_grad():
    mesh = Mesh2D(np.linspace(0.0, 0.06, 21), np.linspace(0.0, 0.05, 19))
    geo = ReactorGeometry(0.06, 0.0, 0.05)
    geo.add_region(RectRegion(0.02, 0.04, 0.0, 0.012,
                              Material.GROUNDED_METAL, "block"))
    mask = mesh.material_mask(geo)
    op = build_transport(mesh, mask)
    p = ElectronParams(nu_m=6.0e7, mass_ratio=MR_AR, Tg_eV=0.026,
                       ne_floor=1e12, Te_max=20.0)

    g = np.exp(-((mesh.RC - 0.02) ** 2 + (mesh.ZC - 0.03) ** 2) / 0.015 ** 2)
    ne = np.where(op.active, 1e17 * np.maximum(g, 1e-3), 0.0)
    n_eps = 1.5 * ne * 2.0
    Q = np.where(op.active, 1e5 * g, 0.0)              # W/m^3
    Er = 20.0 * np.sin(80 * np.ones((mesh.Nr + 1, 1)) * mesh.z_c[None, :])
    Ez = 20.0 * np.cos(70 * mesh.r_c[:, None] * np.ones((1, mesh.Nz + 1)))

    dt = 0.5 * energy_stable_dt(op, ne, n_eps, Er, Ez, p, Te_ref=p.Te_max)
    n_steps = 60
    _, ref, _ = advance(op, ne.copy(), n_eps.copy(), Er, Ez, Q / QE, p,
                        dt, n_steps)

    coeffs = to_jax(op)
    step = make_jax_energy_stepper(coeffs, p, n_sub=n_steps)
    S_ext = jnp.asarray(Q / QE)
    out = np.asarray(step(jnp.asarray(n_eps), jnp.asarray(ne),
                          jnp.asarray(Er), jnp.asarray(Ez), S_ext, dt))
    err = np.max(np.abs(out - ref)) / ref.max()
    assert err < 1e-10, err
    print(f"[2a] jax stepper vs numpy:       max|d n_eps| rel = {err:.1e} "
          f"({n_steps} substeps)")

    vol = coeffs.volume
    neJ, nepsJ, ErJ, EzJ = map(jnp.asarray, (ne, n_eps, Er, Ez))

    def total(s):
        return jnp.sum(step(nepsJ, neJ, ErJ, EzJ, s * S_ext, dt) * vol)

    gAD = jax.grad(total)(1.0)
    h = 1e-4
    gFD = (total(1.0 + h) - total(1.0 - h)) / (2 * h)
    rel = abs(gAD - gFD) / abs(gFD)
    assert rel < 1e-5, rel
    print(f"[2b] grad through {n_steps} substeps:    d(E_tot)/d(Q scale) = "
          f"{gAD:.4e} eV (FD rel err {rel:.1e})")


# ------------------------------------------------------------------ test 3 --
def test_gec_te_map():
    mesh, mask = gec_setup()
    st = gec_plasma_state(mesh, mask, floor_frac=1e-3)
    ne, sigma, eps_r, ng = st["ne"], st["sigma"], st["eps_r"], st["ng"]

    # inductive solve, scaled to 1200 W absorbed (as in test_inductive)
    iop = build_operator(mesh, mask)
    J = coil_current_density(mesh, TURNS, 1.0)
    A = iop.solve_direct(sigma, OMEGA, J, eps_r)
    P1 = absorbed_power(A, sigma, OMEGA, iop.volume)
    target = 1200.0
    A *= np.sqrt(target / P1)
    Q = power_deposition(A, sigma, OMEGA)              # W/m^3
    P_in = float(np.sum(Q * iop.volume))

    # The energy solve is restricted to the populated prescribed profile:
    # the equation set (Eq. 13) is only consistent where ne is itself a
    # transported quantity; the 1e-3-floored pockets would otherwise
    # receive energy down the prescribed density cliff via grad(D_eps
    # n_eps) with (nearly) no electrons to carry it. Goes away once the
    # transport tier evolves ne self-consistently.
    mask_e = mask.copy()
    mask_e[(mask == int(Material.PLASMA))
           & (ne < 0.02 * st["ne0"])] = int(Material.AIR)
    top = build_transport(mesh, mask_e)
    p = ElectronParams(nu_m=st["nu_e"], mass_ratio=MR_AR, Tg_eV=st["Tg_eV"],
                       re=0.2, ne_floor=1e-4 * st["ne0"],
                       Te_min=0.02, Te_max=15.0)

    def inelastic_np(ne_, Te):
        """Illustrative Lieberman-style argon global-model fits with the
        doc Table 2 thresholds; production runs use Boltzmann tables."""
        kex = 2.48e-14 * Te ** 0.33 * np.exp(-12.78 / Te)
        kiz = 2.34e-14 * Te ** 0.59 * np.exp(-17.44 / Te)
        return ne_ * ng * (11.5 * kex + 15.8 * kiz)

    def inelastic_jx(ne_, Te):
        kex = 2.48e-14 * Te ** 0.33 * jnp.exp(-12.78 / Te)
        kiz = 2.34e-14 * Te ** 0.59 * jnp.exp(-17.44 / Te)
        return ne_ * ng * (11.5 * kex + 15.8 * kiz)

    Er = np.zeros((mesh.Nr + 1, mesh.Nz))              # no ES field yet
    Ez = np.zeros((mesh.Nr, mesh.Nz + 1))
    n_eps = np.where(top.active, 1.5 * ne * 2.0, 0.0)  # Te(0) = 2 eV
    S_ext = np.where(top.active, Q / QE, 0.0)
    P_in = float(np.sum(Q * top.volume * top.active))  # power in solve region

    # steady state via per-cell local pseudo-time (doc Sec. 12.4 spirit);
    # dt_max guards the stiff inelastic sink on coarse cells
    dt = energy_local_dt(top, ne, n_eps, Er, Ez, p, cfl=0.45,
                         Te_ref=p.Te_max, dt_max=3e-9)
    n_sub = 2000
    step = make_jax_energy_stepper(to_jax(top), p, inelastic_fn=inelastic_jx,
                                   n_sub=n_sub)
    neJ, ErJ, EzJ, SJ, dtJ = map(jnp.asarray, (ne, Er, Ez, S_ext, dt))
    P = jnp.asarray(n_eps)
    Te_old = temperature(ne, n_eps, p)
    for k in range(80):
        P = step(P, neJ, ErJ, EzJ, SJ, dtJ)
        Te = temperature(ne, np.asarray(P), p)
        dTe = np.max(np.abs(Te - Te_old)[top.active]) / Te[top.active].max()
        Te_old = Te
        if dTe < 1e-6:
            break
    n_eps = np.asarray(P)
    Te = temperature(ne, n_eps, p)
    print(f"[3] GEC steady state in {(k + 1) * n_sub} local-dt iterations "
          f"(dTe/Te per block = {dTe:.1e})")

    # global power balance at steady state
    P_wall = wall_energy_power(top, n_eps, Te, p)
    P_el = float(np.sum(3.0 * p.mass_ratio * np.asarray(p.nu_m) * ne
                        * (Te - p.Tg_eV) * top.volume * top.active) * QE)
    P_inel = float(np.sum(inelastic_np(ne, Te) * top.volume * top.active)
                   * QE)
    bal = abs(P_in - (P_wall + P_el + P_inel)) / P_in
    Te_pk = Te[top.active].max()
    print(f"    P_in = {P_in:.1f} W ({P_in / 12.0:.1f}% of coil) = wall "
          f"{P_wall:.1f} + elastic {P_el:.2f} + inelastic {P_inel:.1f} W  "
          f"(imbalance {bal:.2%})")
    print(f"    peak Te = {Te_pk:.2f} eV (skin layer), on-axis bulk Te = "
          f"{Te[0, np.searchsorted(mesh.z_faces, 0.020)]:.2f} eV")
    assert 1.0 < Te_pk < 8.0, Te_pk
    assert bal < 0.01, bal

    # figure
    Te_plot = np.where(top.active, Te, np.nan)
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    pc = ax.pcolormesh(mesh.r_faces, mesh.z_faces, Te_plot.T,
                       cmap="plasma", shading="flat")
    fig.colorbar(pc, ax=ax, shrink=0.85, label=r"$T_e$ (eV)")
    lv = ax.contour(mesh.r_c, mesh.z_c, Q.T, levels=5, colors="w",
                    linewidths=0.5)
    for m, cc in ((Material.GROUNDED_METAL, "0.4"), (Material.WAFER, "g"),
                  (Material.RF_ANTENNA, "r"), (Material.DIELECTRIC,
                                               "orange")):
        ax.contour(mesh.r_c[None, :].repeat(mesh.Nz, 0),
                   mesh.z_c[:, None].repeat(mesh.Nr, 1),
                   (mask.T == int(m)).astype(float), levels=[0.5],
                   colors=cc, linewidths=0.7)
    ax.set_xlabel("r (m)"); ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.set_title("Electron temperature, GEC ICP @ 1200 W\n"
                 "(EETM Option A on the inductive heating; white: $Q_{ind}$)")
    fig.tight_layout()
    fig.savefig("gec_icp_te.png", dpi=150, bbox_inches="tight")
    print("    wrote gec_icp_te.png")


if __name__ == "__main__":
    test_0d_balance_and_conservation()
    test_jax_matches_numpy_and_grad()
    test_gec_te_map()
    print("\nall electron-energy checks passed")