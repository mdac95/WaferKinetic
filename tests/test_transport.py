"""
test_transport.py
=================
Validation of the FKPM flux closures (doc Secs. 5.1, 5.4), following the
WaferKinetic "conservation asserts before the next module" convention.

1. Conservation: closed domain with internal obstacles, random smooth
   coefficients and fields -> inventory change equals the injected source
   exactly (interior FV faces telescope to machine precision).
2. Verification: fundamental (Schottky) diffusion-mode decay in a closed
   cylinder vs. the analytic rate D [ (chi01/R)^2 + (pi/L)^2 ]
   (doc Sec. 13) on a stretched mesh.
3. Drift-diffusion equilibrium: prescribed potential well ->
   discrete steady state matches the Boltzmann relation n ~ exp(Phi/Te).
4. Closure 1B: (a) B = 0 reduces exactly to closure 1A; (b) tensor
   diffusion and gated drift against analytic values on linear fields;
   (c) mode-decay anisotropy with strong B_z matches
   D_perp = D / (1 + (wc/nuc)^2).
5. Ion wall boundary condition (doc Eq. 37): thermal + outward-gated
   drift flux values, exact.

Run:  python test_transport.py
"""

import numpy as np
from scipy.special import j0, jn_zeros

from waferkinetic.mesh.reactor_mesh import (Material, RectRegion,
                                            ReactorGeometry, tanh_grid,
                                            Mesh2D)
from waferkinetic.solvers.transport import (
    build_transport, divergence, flux_drift_diffusion,
    flux_drift_diffusion_magnetized, magnetized_tensors,
    wall_flux_dirichlet0, wall_flux_thermal, wall_flux_ion, stable_dt, QE, ME)

CHI01 = float(jn_zeros(0, 1)[0])          # 2.404825...


# ------------------------------------------------------------------ test 1 --
def test_conservation():
    rng = np.random.default_rng(3)
    mesh = Mesh2D(tanh_grid(0.0, 0.07, 22, beta=1.1, cluster="end"),
                  tanh_grid(-0.04, 0.04, 26, beta=1.3, cluster="both"))
    geo = ReactorGeometry(0.07, -0.04, 0.04)
    geo.add_region(RectRegion(0.02, 0.04, 0.0, 0.015,
                              Material.GROUNDED_METAL, "block"))
    geo.add_region(RectRegion(0.05, 0.07, -0.03, -0.01,
                              Material.DIELECTRIC, "block2"))
    op = build_transport(mesh, mesh.material_mask(geo))

    D = 0.5 + 0.3 * np.sin(40 * mesh.RC) * np.cos(50 * mesh.ZC)
    mu = 0.2 * (1.0 + 0.5 * np.cos(30 * mesh.RC * mesh.ZC))
    Er = 50.0 * rng.standard_normal((mesh.Nr + 1, mesh.Nz))
    Ez = 50.0 * rng.standard_normal((mesh.Nr, mesh.Nz + 1))
    S = 1e3 * (1.0 + np.sin(25 * mesh.ZC))
    S = np.where(op.active, S, 0.0)
    N = np.where(op.active, 1.0 + rng.random(D.shape), 0.0)

    dt = stable_dt(op, D, mu, -1.0, Er, Ez, cfl=0.4)
    I0 = np.sum(N * op.volume)
    n_steps = 300
    for _ in range(n_steps):
        Fr, Fz = flux_drift_diffusion(op, N, D, mu, -1.0, Er, Ez)
        N = N + dt * (-divergence(op, Fr, Fz) + S)
    dI = np.sum(N * op.volume) - I0
    dI_exact = n_steps * dt * np.sum(S * op.volume)
    rel = abs(dI - dI_exact) / abs(dI_exact)
    assert rel < 1e-11, rel
    print(f"[1] closed-domain conservation:  |dI - sum(S V) dt| rel = "
          f"{rel:.2e} over {n_steps} steps")


# ------------------------------------------------------------------ test 2 --
def test_schottky_mode_decay():
    R, L, D = 0.05, 0.08, 2.0
    mesh = Mesh2D(tanh_grid(0.0, R, 30, beta=1.0, cluster="end"),
                  tanh_grid(0.0, L, 36, beta=1.0, cluster="both"))
    mask = np.full((mesh.Nr, mesh.Nz), int(Material.PLASMA), dtype=np.int32)
    op = build_transport(mesh, mask)

    N = j0(CHI01 * mesh.RC / R) * np.sin(np.pi * mesh.ZC / L)
    lam_ana = D * ((CHI01 / R) ** 2 + (np.pi / L) ** 2)

    dt = stable_dt(op, D, cfl=0.4)

    def run(N, T):
        for _ in range(int(round(T / dt))):
            Fr, Fz = flux_drift_diffusion(op, N, D, form="d_gradn")
            Wr, Wz = wall_flux_dirichlet0(op, N, D)
            N = N - dt * divergence(op, Fr + Wr, Fz + Wz)
        return N

    T = 0.5 / lam_ana
    N1 = run(N, T)          # burn-in: discard non-mode content
    N2 = run(N1, T)
    lam_num = np.log(np.sum(N1 * op.volume) / np.sum(N2 * op.volume)) \
        / (int(round(T / dt)) * dt)
    err = abs(lam_num - lam_ana) / lam_ana
    assert err < 0.02, err
    print(f"[2] Schottky mode decay:         lambda = {lam_num:.1f} 1/s vs "
          f"{lam_ana:.1f} analytic (rel err {err:.2%})")


# ------------------------------------------------------------------ test 3 --
def test_boltzmann_equilibrium():
    mesh = Mesh2D(np.linspace(0.0, 0.06, 33), np.linspace(0.0, 0.06, 33))
    mask = np.full((mesh.Nr, mesh.Nz), int(Material.PLASMA), dtype=np.int32)
    op = build_transport(mesh, mask)

    Te, mu = 2.0, 0.05
    Dc = mu * Te
    Phi = 0.6 * np.exp(-((mesh.RC - 0.025) ** 2 + (mesh.ZC - 0.03) ** 2)
                       / 0.012 ** 2)
    Er = np.zeros((mesh.Nr + 1, mesh.Nz))
    Er[1:-1, :] = -(Phi[1:, :] - Phi[:-1, :]) / op.dc_r[1:-1, :]
    Ez = np.zeros((mesh.Nr, mesh.Nz + 1))
    Ez[:, 1:-1] = -(Phi[:, 1:] - Phi[:, :-1]) / op.dc_z[:, 1:-1]

    N = np.full(Phi.shape, 1e15)
    dt = stable_dt(op, Dc, mu, -1.0, Er, Ez, cfl=0.4)
    for _ in range(int(0.02 / dt)):        # ~5 slowest diffusion times
        Fr, Fz = flux_drift_diffusion(op, N, Dc, mu, -1.0, Er, Ez)
        N = N - dt * divergence(op, Fr, Fz)

    dev = np.log(N) - Phi / Te
    dev -= dev.mean()
    err = np.max(np.abs(dev))
    assert err < 0.03, err
    print(f"[3] Boltzmann equilibrium:       max |ln n - Phi/Te| dev = "
          f"{err:.3f} (upwind first-order bound)")


# ------------------------------------------------------------------ test 4 --
def test_magnetized():
    mesh = Mesh2D(np.linspace(0.0, 0.10, 31), np.linspace(0.0, 0.10, 31))
    mask = np.full((mesh.Nr, mesh.Nz), int(Material.PLASMA), dtype=np.int32)
    op = build_transport(mesh, mask)
    rng = np.random.default_rng(1)
    N = 1e14 * (1.0 + rng.random(mask.shape))
    Er = 30.0 * rng.standard_normal((mesh.Nr + 1, mesh.Nz))
    Ez = 30.0 * rng.standard_normal((mesh.Nr, mesh.Nz + 1))
    D0, mu0, nu_c = 1.3, 0.4, 2.0e5

    # (a) B = 0 reduces exactly to closure 1A (uniform coefficients)
    F1 = flux_drift_diffusion(op, N, D0, mu0, 1.0, Er, Ez, form="d_gradn")
    F2 = flux_drift_diffusion_magnetized(op, N, D0, mu0, 1.0, Er, Ez,
                                         0.0, 0.0, nu_c)
    err = max(np.max(np.abs(F1[0] - F2[0])), np.max(np.abs(F1[1] - F2[1]))) \
        / np.max(np.abs(F1[0]))
    assert err < 1e-13, err
    print(f"[4a] B -> 0 reduction:           max flux diff = {err:.2e}")

    # (b) tensor diffusion + gated drift on linear fields, analytic
    a, bR, bZ = 2.0e14, 3.0e15, 5.0e15
    N = a + bR * mesh.RC + bZ * mesh.ZC
    th = np.deg2rad(35.0)
    wc_over_nu = 2.0
    Bmag = wc_over_nu * nu_c * ME / QE
    Br, Bz = Bmag * np.sin(th), Bmag * np.cos(th)
    ratio = 1.0 / (1.0 + wc_over_nu ** 2)
    Drr, Drz, Dzz = magnetized_tensors(np.full(N.shape, D0), Br, Bz,
                                       nu_c, QE, ME)
    E0r, E0z = 40.0, 25.0
    ErU = np.full((mesh.Nr + 1, mesh.Nz), E0r)
    EzU = np.full((mesh.Nr, mesh.Nz + 1), E0z)

    Fd = flux_drift_diffusion_magnetized(op, N, D0, 0.0, 1.0, ErU * 0,
                                         EzU * 0, Br, Bz, nu_c)
    Fr_ana = -(Drr[0, 0] * bR + Drz[0, 0] * bZ)
    Fz_ana = -(Dzz[0, 0] * bZ + Drz[0, 0] * bR)
    sl = np.s_[3:-3, 3:-3]
    errd = max(np.max(np.abs(Fd[0][sl] - Fr_ana)) / abs(Fr_ana),
               np.max(np.abs(Fd[1][sl] - Fz_ana)) / abs(Fz_ana))
    assert errd < 1e-12, errd

    Mrr, Mrz, Mzz = magnetized_tensors(np.full(N.shape, mu0), Br, Bz,
                                       nu_c, QE, ME)
    Fv = flux_drift_diffusion_magnetized(op, N, 0.0, mu0, 1.0, ErU, EzU,
                                         Br, Bz, nu_c)
    vr = Mrr[0, 0] * E0r + Mrz[0, 0] * E0z          # > 0: upwind = west cell
    vz = Mzz[0, 0] * E0z + Mrz[0, 0] * E0r
    assert vr > 0 and vz > 0
    NW = np.zeros((mesh.Nr + 1, mesh.Nz)); NW[1:, :] = N
    NS = np.zeros((mesh.Nr, mesh.Nz + 1)); NS[:, 1:] = N
    errv = max(np.max(np.abs(Fv[0][sl] - vr * NW[sl])) / np.max(vr * NW[sl]),
               np.max(np.abs(Fv[1][sl] - vz * NS[sl])) / np.max(vz * NS[sl]))
    assert errv < 1e-12, errv
    print(f"[4b] tensor flux vs analytic:    diffusion {errd:.1e}, "
          f"drift {errv:.1e} (wc/nuc = {wc_over_nu}, ratio = {ratio:.3f})")

    # (c) mode decay with strong uniform Bz: radial diffusion reduced
    R, L, D = 0.04, 0.30, 1.0
    mesh = Mesh2D(tanh_grid(0.0, R, 24, beta=0.8, cluster="end"),
                  np.linspace(0.0, L, 31))
    op = build_transport(mesh, np.full((mesh.Nr, mesh.Nz),
                                       int(Material.PLASMA), np.int32))
    wc_over_nu = 3.0
    Bz0 = wc_over_nu * nu_c * ME / QE
    ratio = 1.0 / (1.0 + wc_over_nu ** 2)
    lam_ana = D * (ratio * (CHI01 / R) ** 2 + (np.pi / L) ** 2)
    N = j0(CHI01 * mesh.RC / R) * np.sin(np.pi * mesh.ZC / L)
    Dp = ratio * D                                    # radial (perp) diagonal
    dt = stable_dt(op, D, cfl=0.4)
    steps = int(round(0.5 / lam_ana / dt))

    def run(N, steps):
        for _ in range(steps):
            Fr, Fz = flux_drift_diffusion_magnetized(
                op, N, D, 0.0, 1.0, np.zeros((mesh.Nr + 1, mesh.Nz)),
                np.zeros((mesh.Nr, mesh.Nz + 1)), 0.0, Bz0, nu_c)
            Wr, Wz = wall_flux_dirichlet0(op, N, Dp, D_z=D)
            N = N - dt * divergence(op, Fr + Wr, Fz + Wz)
        return N

    N1 = run(N, steps)
    N2 = run(N1, steps)
    lam_num = np.log(np.sum(N1 * op.volume) / np.sum(N2 * op.volume)) \
        / (steps * dt)
    err = abs(lam_num - lam_ana) / lam_ana
    assert err < 0.03, err
    print(f"[4c] B_z mode decay:             lambda = {lam_num:.1f} vs "
          f"{lam_ana:.1f} analytic, D_perp/D = {ratio:.2f} "
          f"(rel err {err:.2%})")


# ------------------------------------------------------------------ test 5 --
def test_ion_wall_bc():
    mesh = Mesh2D(np.linspace(0.0, 0.04, 5), np.linspace(0.0, 0.04, 5))
    op = build_transport(mesh, np.full((4, 4), int(Material.PLASMA),
                                       np.int32))
    N0, vth, gamma, mu = 2.0e15, 600.0, 0.4, 0.12
    N = np.full((4, 4), N0)
    Er = np.full((5, 4), 80.0)
    Ez = np.full((4, 5), 80.0)
    Fr, Fz = wall_flux_ion(op, N, mu, +1.0, Er, Ez, vth=vth, gamma=gamma)

    f_th = 0.25 * gamma * vth * N0                    # 60 * N0
    f_dr = mu * 80.0 * N0                             # 9.6 * N0
    assert np.allclose(Fr[-1, :], f_th + f_dr)        # east: drift outward
    assert np.allclose(Fr[1:-1, :], 0.0) and np.allclose(Fr[0, :], 0.0)
    assert np.allclose(Fz[:, -1], f_th + f_dr)        # north: drift outward
    assert np.allclose(Fz[:, 0], -f_th)               # south: drift gated off
    assert np.allclose(Fz[:, 1:-1], 0.0)
    print(f"[5] ion wall BC (Eq. 37):        thermal {f_th:.3e}, gated "
          f"drift {f_dr:.3e} 1/(m^2 s) -- exact")


if __name__ == "__main__":
    test_conservation()
    test_schottky_mode_decay()
    test_boltzmann_equilibrium()
    test_magnetized()
    test_ion_wall_bc()
    print("\nall transport checks passed")