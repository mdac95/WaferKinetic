"""
demo_gec_icp.py
===============
GEC reference-cell ICP geometry, reconstructed from the COMSOL-style
reference figure. Coordinates in meters, read off the figure axes
(adjust the CONSTANTS block below to match your exact CAD numbers).

Layout (r horizontal, z vertical):
  - Domain: r in [0, 0.145], z in [-0.04, 0.08]
  - Air box top-left (r < 65.5 mm, z > 52.5 mm) containing the
    5-turn, 13.56 MHz coil sitting on the dielectric window
  - L-shaped dielectric window: main slab (r < 57.5 mm, z 40-52.5 mm)
    plus a thicker step block (r 57.5-83.5 mm, z 34-52.5 mm)
  - Thin grounded plate at z ~ 0 (r < 83.5 mm), with the wafer
    occupying its top surface for r < 50 mm
  - Two-tier grounded pedestal below the wafer
  - Air below the pedestal plane (z < -25 mm)
  - Plasma (default fill) everywhere else, including the large region
    on the right that wraps around the dielectric step

Run:  python demo_gec_icp.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from waferkinetic.mesh.reactor_mesh import (
    Material, RectRegion, ReactorGeometry,
    tanh_grid, geometric_grid, composite_grid,
    Mesh2D, plot_reactor,
)

mm = 1e-3  # unit helper

# ------------------------------------------------------- CONSTANTS (from fig)
R_MAX, Z_MIN, Z_MAX = 145*mm, -40*mm, 80*mm

# Dielectric window (L-shaped: slab + step)
R_SLAB,  Z_SLAB_BOT           = 57.5*mm, 40.0*mm      # main slab
R_STEP,  Z_STEP_BOT           = 83.5*mm, 34.0*mm      # thicker step block
Z_DIEL_TOP                    = 52.5*mm               # common top face

# Air box (contains the coil) above the window
R_AIRBOX                      = 65.5*mm

# 5-turn coil on top of the window
N_TURNS, COIL_W, COIL_H       = 5, 3.5*mm, 4.0*mm
COIL_PITCH, COIL_R0           = 12.0*mm, 4.0*mm       # first turn inner radius

# Bottom stack: ground plate / wafer / pedestal
Z_PLATE_BOT, Z_PLATE_TOP      = -3.0*mm, 0.0*mm       # thin plate at z ~ 0
R_PLATE                       = 83.5*mm
R_WAFER                       = 50.0*mm               # wafer = plate top, r<50
R_PED_UP,  Z_PED_UP_BOT       = 50.0*mm, -15.0*mm     # upper pedestal tier
R_PED_LO0, R_PED_LO1          = 10.0*mm, 50.0*mm      # lower tier (annular)
Z_PED_LO_BOT                  = -25.0*mm
Z_AIR_BOT_TOP                 = -25.0*mm              # air below this plane

# ---------------------------------------------------------------- geometry --
geo = ReactorGeometry(r_max=R_MAX, z_min=Z_MIN, z_max=Z_MAX)

# Non-plasma "background" regions first (painter's order: later wins)
geo.add_region(RectRegion(0.0, R_AIRBOX, Z_DIEL_TOP, Z_MAX,
                          Material.AIR, "air box (coil region)"))
geo.add_region(RectRegion(0.0, R_MAX, Z_MIN, Z_AIR_BOT_TOP,
                          Material.AIR, "air below pedestal"))

# Dielectric window (slab + step)
geo.add_region(RectRegion(0.0, R_SLAB, Z_SLAB_BOT, Z_DIEL_TOP,
                          Material.DIELECTRIC, "window slab"))
geo.add_region(RectRegion(R_SLAB, R_STEP, Z_STEP_BOT, Z_DIEL_TOP,
                          Material.DIELECTRIC, "window step"))

# 5-turn coil sitting on the window, inside the air box
for k in range(N_TURNS):
    r0 = COIL_R0 + k * COIL_PITCH
    geo.add_region(RectRegion(r0, r0 + COIL_W,
                              Z_DIEL_TOP, Z_DIEL_TOP + COIL_H,
                              Material.RF_ANTENNA, f"coil turn {k+1}"))

# Grounded plate at z ~ 0, then wafer overwrites its inner top section
geo.add_region(RectRegion(0.0, R_PLATE, Z_PLATE_BOT, Z_PLATE_TOP,
                          Material.GROUNDED_METAL, "ground plate"))
geo.add_region(RectRegion(0.0, R_WAFER, Z_PLATE_BOT, Z_PLATE_TOP,
                          Material.WAFER, "wafer"))

# Two-tier pedestal
geo.add_region(RectRegion(0.0, R_PED_UP, Z_PED_UP_BOT, Z_PLATE_BOT,
                          Material.GROUNDED_METAL, "pedestal upper"))
geo.add_region(RectRegion(R_PED_LO0, R_PED_LO1, Z_PED_LO_BOT, Z_PED_UP_BOT,
                          Material.GROUNDED_METAL, "pedestal lower"))

# -------------------------------------------------------------------- mesh --
# z-grid: segment joints land exactly on every horizontal material interface;
# fine (~0.3 mm) at the wafer surface (z=0) and at the window face (z=40 mm).
dz_fine, ratio = 0.3*mm, 1.20
z_mid = 0.5 * (Z_PLATE_TOP + Z_SLAB_BOT)          # coarse midpoint of the gap

z_faces = composite_grid([
    np.linspace(Z_MIN, Z_AIR_BOT_TOP, 5),                       # bottom air
    np.linspace(Z_AIR_BOT_TOP, Z_PED_UP_BOT, 5),                # lower tier
    np.linspace(Z_PED_UP_BOT, Z_PLATE_BOT, 6),                  # upper tier
    np.linspace(Z_PLATE_BOT, Z_PLATE_TOP, 4),                   # plate/wafer
    geometric_grid(Z_PLATE_TOP, z_mid, dz_fine, ratio, "start"),  # sheath @ wafer
    geometric_grid(z_mid, Z_STEP_BOT, dz_fine*1.5, ratio, "end"),
    np.linspace(Z_STEP_BOT, Z_SLAB_BOT, 5),                     # step band
    np.linspace(Z_SLAB_BOT, Z_DIEL_TOP, 7),                     # window
    np.linspace(Z_DIEL_TOP, Z_DIEL_TOP + COIL_H, 3),            # coil band
    geometric_grid(Z_DIEL_TOP + COIL_H, Z_MAX, 1.5*mm, 1.25, "start"),
])

# r-grid: joints on every vertical interface (wafer edge, slab edge, air-box
# edge, step/plate edge); resolution fine enough under the coils that each
# 3.5 mm turn spans ~2 cells; fine near the outer boundary.
r_faces = composite_grid([
    tanh_grid(0.0, R_WAFER, n_cells=26, beta=1.2, cluster="end"),   # wafer edge
    np.linspace(R_WAFER, R_SLAB, 5),
    np.linspace(R_SLAB, R_AIRBOX, 5),
    np.linspace(R_AIRBOX, R_PLATE, 8),
    tanh_grid(R_PLATE, R_MAX, n_cells=20, beta=1.6, cluster="both"),
])

mesh = Mesh2D(r_faces, z_faces)
mask = mesh.material_mask(geo)

# --------------------------------------------------------------- sanity ----
print(mesh)
analytic = np.pi * R_MAX**2 * (Z_MAX - Z_MIN)
assert np.isclose(mesh.total_volume(), analytic, rtol=1e-12)
print(f"sum(V) = {mesh.total_volume():.6e} m^3  (analytic {analytic:.6e})")
for m in Material:
    n = int((mask == m).sum())
    if n:
        print(f"  {m.name:>14s}: {n:5d} cells")
j_wafer = np.searchsorted(mesh.z_faces, Z_PLATE_TOP)
print(f"dz at wafer surface: {mesh.dz[j_wafer]/mm:.3f} mm")

# ----------------------------------------------------------------- plot ----
ax = plot_reactor(mesh, mask,
                  title="GEC reference-cell ICP (5-turn coil, 13.56 MHz)")
ax.figure.savefig("gec_icp_reactor.png", dpi=160, bbox_inches="tight")
print("wrote gec_icp_reactor.png")