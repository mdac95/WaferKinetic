"""
gec_case.py
===========
Shared GEC reference-cell ICP case for the test suite: geometry + mesh
(verbatim from demo_gec_icp.py / test_inductive.py) and the prescribed
argon-like plasma state.

`gec_plasma_state` optionally floors the prescribed n_e profile inside
the plasma region (default 1e-3 of the peak) so the electron-energy
module has a well-defined Te everywhere; the inductive solve is affected
only at the 0.1% conductivity level.
"""

import numpy as np

from waferkinetic.mesh.reactor_mesh import (Material, RectRegion,
                                            ReactorGeometry, tanh_grid,
                                            geometric_grid, composite_grid,
                                            Mesh2D)
from waferkinetic.solvers.inductive import cold_plasma_sigma

mm = 1e-3
OMEGA = 2 * np.pi * 13.56e6

N_TURNS, COIL_W, COIL_H = 5, 3.5 * mm, 4.0 * mm
COIL_PITCH, COIL_R0 = 12.0 * mm, 4.0 * mm
Z_DIEL_TOP = 52.5 * mm
TURNS = [(COIL_R0 + k * COIL_PITCH, COIL_R0 + k * COIL_PITCH + COIL_W,
          Z_DIEL_TOP, Z_DIEL_TOP + COIL_H) for k in range(N_TURNS)]


def gec_setup(coarse=False):
    """GEC geometry + mesh, verbatim from demo_gec_icp.py.

    coarse=True returns the same geometry on a ~2x coarser mesh (min
    cell ~1 mm) for the coupled FKPM tier: the explicit electron
    stability floor dt ~ 1/(2 mu_e Te (1/dz^2 + 1/dr^2)) scales with
    the finest cell, and the coupled charged-species stepper must run
    at a UNIFORM pseudo-dt (per-cell local dt destabilizes the coupled
    field-charge update; see poisson.make_jax_fkpm_stepper notes)."""
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

    if coarse:
        z_mid = 0.5 * (Z_PLATE_TOP + Z_SLAB_BOT)
        zf = composite_grid([
            np.linspace(Z_MIN, Z_AIR_BOT_TOP, 4),
            np.linspace(Z_AIR_BOT_TOP, Z_PED_UP_BOT, 4),
            np.linspace(Z_PED_UP_BOT, Z_PLATE_BOT, 5),
            np.linspace(Z_PLATE_BOT, Z_PLATE_TOP, 3),
            geometric_grid(Z_PLATE_TOP, z_mid, 1.0 * mm, 1.25, "start"),
            geometric_grid(z_mid, Z_STEP_BOT, 1.5 * mm, 1.25, "end"),
            np.linspace(Z_STEP_BOT, Z_SLAB_BOT, 4),
            np.linspace(Z_SLAB_BOT, Z_DIEL_TOP, 6),
            np.linspace(Z_DIEL_TOP, Z_DIEL_TOP + COIL_H, 3),
            geometric_grid(Z_DIEL_TOP + COIL_H, Z_MAX, 2.0 * mm, 1.3,
                           "start"),
        ])
        rfaces = composite_grid([
            tanh_grid(0.0, R_WAFER, n_cells=18, beta=1.0, cluster="end"),
            np.linspace(R_WAFER, R_SLAB, 4),
            np.linspace(R_SLAB, R_AIRBOX, 4),
            np.linspace(R_AIRBOX, R_PLATE, 6),
            tanh_grid(R_PLATE, R_MAX, n_cells=12, beta=1.4,
                      cluster="both"),
        ])
        mesh = Mesh2D(rfaces, zf)
        return mesh, mesh.material_mask(geo)

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


def gec_plasma_state(mesh, mask, floor_frac=1e-3):
    """Prescribed argon-like n_e (until the transport tier closes the loop):
    peaked at (r=0, z ~ 20 mm), floored at floor_frac * ne0 inside the
    plasma region; nu_e from p = 20 mTorr, k_mom ~ 1e-13 m^3/s."""
    ne0 = 1.5e17
    prof_r = np.clip(1.0 - (mesh.RC / 0.105) ** 2, 0.0, None)
    prof_z = np.clip(1.0 - ((mesh.ZC - 0.020) / 0.024) ** 2, 0.0, None)
    plasma = mask == int(Material.PLASMA)
    ne = np.where(plasma, ne0 * np.maximum(prof_r * prof_z, floor_frac), 0.0)

    p0, T0 = 0.02 * 133.322, 300.0            # 20 mTorr
    ng = p0 / (1.380649e-23 * T0)
    nu_e = ng * 1.0e-13
    sigma = cold_plasma_sigma(ne, nu_e, OMEGA)
    eps_r = np.where(mask == int(Material.DIELECTRIC), 4.2, 1.0)
    return dict(ne=ne, ne0=ne0, nu_e=nu_e, sigma=sigma, eps_r=eps_r,
                ng=ng, Tg_eV=T0 / 11600.0)