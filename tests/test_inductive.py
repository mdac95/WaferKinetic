"""
test_inductive.py
=================
Validation of the A_phi ICP field solver, following the WaferKinetic
"conservation asserts before the next module" convention.

1. Structural: the discrete operator is complex symmetric.
2. Physics:    5-turn coil in vacuum vs. analytic Biot-Savart on axis.
3. Energetics: Re(source complex power) == plasma Joule power (discrete
               energy identity, must hold to solver precision).
4. JAX path:   jitted differentiable solve agrees with the sparse direct
               solve; a gradient d(P_abs)/d(ne0) is finite and sane.
5. GEC demo:   argon-like prescribed n_e on the GEC geometry -> skin
               effect, scaled to 1200 W absorbed, figure written.

Run:  python test_inductive.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from waferkinetic.mesh.reactor_mesh import (Material, RectRegion, ReactorGeometry,
                          tanh_grid, geometric_grid, composite_grid, Mesh2D)
from waferkinetic.solvers.inductive import (MU0, build_operator, coil_current_density,
                       cold_plasma_sigma, electric_field, power_deposition,
                       absorbed_power, source_complex_power, bz_on_axis,
                       to_jax, make_jax_solver)

mm = 1e-3
OMEGA = 2 * np.pi * 13.56e6

# Coil turn rectangles shared by all tests (GEC demo values)
N_TURNS, COIL_W, COIL_H = 5, 3.5 * mm, 4.0 * mm
COIL_PITCH, COIL_R0 = 12.0 * mm, 4.0 * mm
Z_DIEL_TOP = 52.5 * mm
TURNS = [(COIL_R0 + k * COIL_PITCH, COIL_R0 + k * COIL_PITCH + COIL_W,
          Z_DIEL_TOP, Z_DIEL_TOP + COIL_H) for k in range(N_TURNS)]


# ------------------------------------------------------------------ test 1 --
def test_symmetry():
    rng = np.random.default_rng(0)
    zf = composite_grid([np.linspace(0.0, 0.03, 9),
                         geometric_grid(0.03, 0.1, 1e-3, 1.3, "start")])
    rf = tanh_grid(0.0, 0.08, 24, beta=1.5, cluster="end")
    mesh = Mesh2D(rf, zf - 0.02)
    geo = ReactorGeometry(0.08, -0.02, 0.08)
    geo.add_region(RectRegion(0.0, 0.04, -0.02, -0.01,
                              Material.GROUNDED_METAL, "block"))
    op = build_operator(mesh, mesh.material_mask(geo))
    k = op.reaction(cold_plasma_sigma(1e17, 6e7, OMEGA)
                    * np.ones(op.active.shape), OMEGA)
    x = rng.normal(size=op.active.shape) + 1j * rng.normal(size=op.active.shape)
    y = rng.normal(size=op.active.shape) + 1j * rng.normal(size=op.active.shape)
    x, y = np.where(op.active, x, 0), np.where(op.active, y, 0)
    lhs = np.sum(y * op.matvec(k, x))          # y^T L x  (transpose, no conj)
    rhs = np.sum(x * op.matvec(k, y))
    rel = abs(lhs - rhs) / abs(lhs)
    assert rel < 1e-13, rel
    print(f"[1] complex-symmetry:            |y^T Lx - x^T Ly|/|.| = {rel:.2e}")


# ------------------------------------------------------------------ test 2 --
def test_vacuum_biot_savart():
    """Big open box, coil only, everything vacuum -> analytic on-axis B_z."""
    rf = composite_grid([
        np.linspace(0.0, 0.06, 41),
        geometric_grid(0.06, 0.40, 1.6 * mm, 1.15, "start"),
    ])
    zf = composite_grid([
        geometric_grid(-0.30, 0.045, 1.5 * mm, 1.15, "end"),
        np.linspace(0.045, 0.0605, 17),
        geometric_grid(0.0605, 0.40, 1.5 * mm, 1.15, "start"),
    ])
    mesh = Mesh2D(rf, zf)
    mask = np.full((mesh.Nr, mesh.Nz), int(Material.PLASMA), dtype=np.int32)
    op = build_operator(mesh, mask)

    I = 1.0
    J = coil_current_density(mesh, TURNS, I)
    sigma = np.zeros((mesh.Nr, mesh.Nz))
    A = op.solve_direct(sigma, OMEGA, J)

    bz_num = np.real(bz_on_axis(A, mesh))
    z = mesh.z_c
    bz_ana = np.zeros_like(z)
    for r0, r1, z0, z1 in TURNS:
        a, zt = 0.5 * (r0 + r1), 0.5 * (z0 + z1)
        bz_ana += MU0 * I * a**2 / (2.0 * (a**2 + (z - zt)**2) ** 1.5)

    sel = (z > -0.04) & (z < 0.045)            # through & below the coil plane
    err = np.max(np.abs(bz_num[sel] - bz_ana[sel]) / np.max(bz_ana[sel]))
    assert err < 0.03, err
    print(f"[2] vacuum on-axis Bz vs analytic: max rel err = {err:.3%} "
          f"(peak Bz = {bz_num.max()*1e6:.2f} uT/A)")


# ---------------------------------------------------- GEC geometry helper ---
def gec_setup():
    """GEC geometry + mesh, verbatim from demo_gec_icp.py."""
    R_MAX, Z_MIN, Z_MAX = 145 * mm, -40 * mm, 80 * mm
    R_SLAB, Z_SLAB_BOT = 57.5 * mm, 40.0 * mm
    R_STEP, Z_STEP_BOT = 83.5 * mm, 34.0 * mm
    R_AIRBOX = 65.5 * mm
    Z_PLATE_BOT, Z_PLATE_TOP = -3.0 * mm, 0.0 * mm
    R_PLATE, R_WAFER = 83.5 * mm, 50.0 * mm
    R_PED_UP, Z_PED_UP_BOT = 50.0 * mm, -15.0 * mm
    R_PED_LO0, R_PED_LO1 = 10.0 * mm, 50.0 * mm
    Z_PED_LO_BOT = -25.0 * mm
    Z_AIR_BOT_TOP = -25.0 * mm

    geo = ReactorGeometry(r_max=R_MAX, z_min=Z_MIN, z_max=Z_MAX)
    geo.add_region(RectRegion(0.0, R_AIRBOX, Z_DIEL_TOP, Z_MAX,
                              Material.AIR, "air box"))
    geo.add_region(RectRegion(0.0, R_MAX, Z_MIN, Z_AIR_BOT_TOP,
                              Material.AIR, "air below"))
    geo.add_region(RectRegion(0.0, R_SLAB, Z_SLAB_BOT, Z_DIEL_TOP,
                              Material.DIELECTRIC, "window slab"))
    geo.add_region(RectRegion(R_SLAB, R_STEP, Z_STEP_BOT, Z_DIEL_TOP,
                              Material.DIELECTRIC, "window step"))
    for k, t in enumerate(TURNS):
        geo.add_region(RectRegion(*t, Material.RF_ANTENNA, f"coil {k+1}"))
    geo.add_region(RectRegion(0.0, R_PLATE, Z_PLATE_BOT, Z_PLATE_TOP,
                              Material.GROUNDED_METAL, "ground plate"))
    geo.add_region(RectRegion(0.0, R_WAFER, Z_PLATE_BOT, Z_PLATE_TOP,
                              Material.WAFER, "wafer"))
    geo.add_region(RectRegion(0.0, R_PED_UP, Z_PED_UP_BOT, Z_PLATE_BOT,
                              Material.GROUNDED_METAL, "pedestal up"))
    geo.add_region(RectRegion(R_PED_LO0, R_PED_LO1, Z_PED_LO_BOT, Z_PED_UP_BOT,
                              Material.GROUNDED_METAL, "pedestal lo"))

    dz_fine, ratio = 0.3 * mm, 1.20
    z_mid = 0.5 * (Z_PLATE_TOP + Z_SLAB_BOT)
    zf = composite_grid([
        np.linspace(Z_MIN, Z_AIR_BOT_TOP, 5),
        np.linspace(Z_AIR_BOT_TOP, Z_PED_UP_BOT, 5),
        np.linspace(Z_PED_UP_BOT, Z_PLATE_BOT, 6),
        np.linspace(Z_PLATE_BOT, Z_PLATE_TOP, 4),
        geometric_grid(Z_PLATE_TOP, z_mid, dz_fine, ratio, "start"),
        geometric_grid(z_mid, Z_STEP_BOT, dz_fine * 1.5, ratio, "end"),
        np.linspace(Z_STEP_BOT, Z_SLAB_BOT, 5),
        np.linspace(Z_SLAB_BOT, Z_DIEL_TOP, 7),
        np.linspace(Z_DIEL_TOP, Z_DIEL_TOP + COIL_H, 3),
        geometric_grid(Z_DIEL_TOP + COIL_H, Z_MAX, 1.5 * mm, 1.25, "start"),
    ])
    rfaces = composite_grid([
        tanh_grid(0.0, R_WAFER, n_cells=26, beta=1.2, cluster="end"),
        np.linspace(R_WAFER, R_SLAB, 5),
        np.linspace(R_SLAB, R_AIRBOX, 5),
        np.linspace(R_AIRBOX, R_PLATE, 8),
        tanh_grid(R_PLATE, R_MAX, n_cells=20, beta=1.6, cluster="both"),
    ])
    mesh = Mesh2D(rfaces, zf)
    mask = mesh.material_mask(geo)
    return mesh, mask


def gec_plasma_state(mesh, mask):
    """Prescribed argon-like n_e (until the transport module exists):
    peaked at (r=0, z~20 mm), n_e0 ~ reference peak; nu_e from p = 20 mTorr."""
    ne0 = 1.5e17
    prof_r = np.clip(1.0 - (mesh.RC / 0.105) ** 2, 0.0, None)
    prof_z = np.clip(1.0 - ((mesh.ZC - 0.020) / 0.024) ** 2, 0.0, None)
    ne = np.where(mask == int(Material.PLASMA), ne0 * prof_r * prof_z, 0.0)

    p0, T0 = 0.02 * 133.322, 300.0            # 20 mTorr
    ng = p0 / (1.380649e-23 * T0)
    nu_e = ng * 1.0e-13                        # k_mom ~ 1e-13 m^3/s @ few eV
    sigma = cold_plasma_sigma(ne, nu_e, OMEGA)
    eps_r = np.where(mask == int(Material.DIELECTRIC), 4.2, 1.0)
    return ne, nu_e, sigma, eps_r


# ------------------------------------------------------------------ test 3 --
def test_power_balance_and_jax():
    mesh, mask = gec_setup()
    op = build_operator(mesh, mask)
    ne, nu_e, sigma, eps_r = gec_plasma_state(mesh, mask)
    J = coil_current_density(mesh, TURNS, 1.0)

    A = op.solve_direct(sigma, OMEGA, J, eps_r)
    P = absorbed_power(A, sigma, OMEGA, op.volume)
    S = source_complex_power(A, J, OMEGA, op.volume)
    rel = abs(S.real - P) / P
    assert rel < 1e-10, rel
    print(f"[3] energy identity:             |Re(S)-P|/P = {rel:.2e} "
          f"(P = {P:.4f} W/A^2, Z_refl = {2*S.real:.3f} + "
          f"j{2*S.imag:.3f} Ohm)")

    # JAX cross-check + gradient
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    coeffs = to_jax(op)
    solve = make_jax_solver(tol=1e-11)
    A_jax = np.asarray(solve(coeffs, jnp.asarray(sigma), OMEGA,
                             jnp.asarray(J), jnp.asarray(eps_r)))
    scale = np.max(np.abs(A))
    err = np.max(np.abs(A_jax - A)) / scale
    assert err < 1e-6, err
    print(f"[4] jax bicgstab vs direct:      max|dA|/max|A| = {err:.2e}")

    # d(P_abs)/d(ne0): differentiate through sigma -> solve -> power
    ne_shape = jnp.asarray(ne / 1.5e17)        # unit profile

    def p_abs(ne0):
        sig = cold_plasma_sigma(ne0 * ne_shape, nu_e, OMEGA)
        a = solve(coeffs, sig, OMEGA, jnp.asarray(J), jnp.asarray(eps_r))
        q = 0.5 * jnp.real(sig) * (OMEGA * jnp.abs(a)) ** 2
        return jnp.sum(q * coeffs.volume)

    g = jax.grad(p_abs)(1.5e17)
    # finite-difference check
    h = 1e12
    g_fd = (p_abs(1.5e17 + h) - p_abs(1.5e17 - h)) / (2 * h)
    rel_g = abs(g - g_fd) / abs(g_fd)
    assert rel_g < 1e-4, rel_g
    print(f"[5] grad through solve:          dP/dne0 = {g:.3e} W m^3 "
          f"(FD rel err {rel_g:.1e})")
    return mesh, mask, op, sigma, eps_r, J, A


# ------------------------------------------------------------------ demo ----
def gec_demo(mesh, mask, op, sigma, eps_r, J, A):
    P1 = absorbed_power(A, sigma, OMEGA, op.volume)     # at I = 1 A
    target = 1200.0
    I = np.sqrt(target / P1)
    A = A * I                                            # linear in I
    Q = power_deposition(A, sigma, OMEGA)
    E = np.abs(electric_field(A, OMEGA))
    S = source_complex_power(A, J * I, OMEGA, op.volume)
    R_refl, L_refl = 2 * S.real / I**2, 2 * S.imag / (I**2 * OMEGA)

    sig_peak = np.abs(np.real(sigma)).max()
    delta = np.sqrt(2.0 / (MU0 * OMEGA * np.abs(sigma).max()))
    print(f"\nGEC ICP @ 13.56 MHz, prescribed argon n_e (peak 1.5e17 m^-3):")
    print(f"  coil current for {target:.0f} W absorbed: I = {I:.1f} A "
          f"(peak-to-peak {2*np.sqrt(2)*I:.0f} A)")
    print(f"  plasma-reflected impedance: R = {R_refl:.3f} Ohm, "
          f"L = {L_refl*1e6:.3f} uH")
    print(f"  Re(sigma) peak = {sig_peak:.1f} S/m, skin depth ~ "
          f"{delta*100:.2f} cm")
    print(f"  peak |E_phi| = {E.max():.1f} V/m, "
          f"peak Q = {Q.max():.3g} W/m^3")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    extent = None
    for ax, (F, ttl, cm) in zip(axes, [
            (E, r"$|E_\phi|$ (V/m)", "viridis"),
            (Q, r"power deposition $Q$ (W/m$^3$)", "inferno"),
            (np.abs(A), r"$|A_\phi|$ (Wb/m)", "magma")]):
        pc = ax.pcolormesh(mesh.r_faces, mesh.z_faces, F.T,
                           cmap=cm, shading="flat")
        fig.colorbar(pc, ax=ax, shrink=0.9)
        ax.set_title(ttl)
        ax.set_xlabel("r (m)")
        ax.set_aspect("equal")
        for m, c in ((Material.GROUNDED_METAL, "0.4"),
                     (Material.WAFER, "g"), (Material.RF_ANTENNA, "r"),
                     (Material.DIELECTRIC, "orange")):
            ax.contour(mesh.RC.T * 0 + mesh.r_c[None, :].repeat(mesh.Nz, 0),
                       mesh.z_c[:, None].repeat(mesh.Nr, 1),
                       (mask.T == int(m)).astype(float),
                       levels=[0.5], colors=c, linewidths=0.7)
    axes[0].set_ylabel("z (m)")
    fig.suptitle("WaferKinetic inductive field solve, GEC ICP, 1200 W absorbed")
    fig.tight_layout()
    fig.savefig("gec_icp_fields.png", dpi=150, bbox_inches="tight")
    print("  wrote gec_icp_fields.png")


if __name__ == "__main__":
    test_symmetry()
    test_vacuum_biot_savart()
    out = test_power_balance_and_jax()
    gec_demo(*out)
    print("\nall checks passed")