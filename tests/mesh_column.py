"""
mesh_column.py
==============
Axial node distribution on the symmetry axis (r = 0), wafer -> window,
for the sheath-resolving ("fine") and analytic-sheath ("asm") grids.

The number that matters after doc P6: the size of the FIRST PLASMA CELL
above the wafer. Measured on the converged ASM state, that single row
sets the whole ion-Courant clock -- every other cell is >= 29x more
permissive -- because it was graded fine to hold a sheath the ASM now
carries outside the mesh.

  python mesh_column.py                    # table + figure, both grids
  python mesh_column.py --outdir outputs_asm
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from waferkinetic.mesh.reactor_mesh import Material
except ImportError:
    from reactor_mesh import Material

from gec_case import gec_setup

WAFER_Z, WINDOW_Z = 0.0, 40.0e-3


def column(mesh, mask, i=0):
    """(z_faces, dz, material) of the r=0 column."""
    return mesh.z_faces, mesh.dz, mask[i, :]


def gap_stats(mesh, mask, i=0):
    pl = mask[i, :] == int(Material.PLASMA)
    j = np.where(pl & (mesh.z_c > WAFER_Z) & (mesh.z_c < WINDOW_Z))[0]
    return dict(n=j.size, first=mesh.dz[j[0]], last=mesh.dz[j[-1]],
                dmin=mesh.dz[j].min(), dmax=mesh.dz[j].max(),
                ratio=float(np.max(np.maximum(
                    mesh.dz[j][1:] / mesh.dz[j][:-1],
                    mesh.dz[j][:-1] / mesh.dz[j][1:]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--rows", type=int, default=14,
                    help="rows of the printed table above the wafer")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    grids = {}
    for tag in ("fine", "asm"):
        mesh, mask = gec_setup(wall=tag)
        grids[tag] = (mesh, mask)
        g = gap_stats(mesh, mask)
        print(f"[{tag:4s}] {mesh.Nr} x {mesh.Nz} = {mesh.Nr * mesh.Nz} "
              f"cells;  discharge gap on axis: {g['n']} cells, "
              f"first(wafer) {g['first'] * 1e3:.3f} mm, "
              f"last(window) {g['last'] * 1e3:.3f} mm, "
              f"span {g['dmin'] * 1e3:.3f}-{g['dmax'] * 1e3:.3f} mm, "
              f"max neighbour ratio {g['ratio']:.2f}")

    print(f"\nnode positions on the axis (r = 0), wafer upward:")
    print(f"{'k':>3} | {'fine z (mm)':>12} {'dz (mm)':>9} | "
          f"{'asm z (mm)':>12} {'dz (mm)':>9}")
    zf_f = grids["fine"][0].z_faces
    zf_a = grids["asm"][0].z_faces
    kf = int(np.argmin(np.abs(zf_f - WAFER_Z)))
    ka = int(np.argmin(np.abs(zf_a - WAFER_Z)))
    for k in range(args.rows):
        a = (f"{zf_f[kf + k] * 1e3:12.3f} "
             f"{(zf_f[kf + k + 1] - zf_f[kf + k]) * 1e3:9.3f}"
             if kf + k + 1 < zf_f.size else " " * 22)
        b = (f"{zf_a[ka + k] * 1e3:12.3f} "
             f"{(zf_a[ka + k + 1] - zf_a[ka + k]) * 1e3:9.3f}"
             if ka + k + 1 < zf_a.size else " " * 22)
        print(f"{k:3d} | {a} | {b}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for tag, c in (("fine", "C0"), ("asm", "C1")):
        mesh, mask = grids[tag]
        zf, dz, _ = column(mesh, mask)
        lab = ("fine (sheath-resolving)" if tag == "fine"
               else "asm (analytic sheath)")
        ax[0].plot(np.arange(zf.size), zf * 1e3, ".-", color=c, ms=3,
                   label=f"{lab}: {mesh.Nz} z-cells")
        ax[1].step(mesh.z_c * 1e3, dz * 1e3, where="mid", color=c,
                   label=lab)
        j = (mask[0, :] == int(Material.PLASMA)) \
            & (mesh.z_c > WAFER_Z) & (mesh.z_c < WINDOW_Z)
        ax[2].plot(mesh.z_c[j] * 1e3, dz[j] * 1e3, "o-", color=c, ms=4,
                   label=lab)
    for a in ax:
        for z, t in ((WAFER_Z, "wafer"), (WINDOW_Z, "window")):
            if a is ax[0]:
                a.axhline(z * 1e3, color="r", ls=":", lw=0.9)
            else:
                a.axvline(z * 1e3, color="r", ls=":", lw=0.9)
        a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[0].set_xlabel("node index k"); ax[0].set_ylabel("z (mm)")
    ax[0].set_title("axial node positions at r = 0\n(red: wafer / window)")
    ax[1].set_xlabel("z (mm)"); ax[1].set_ylabel("dz (mm)")
    ax[1].set_yscale("log"); ax[1].set_title("axial cell size, full column")
    ax[2].set_xlabel("z (mm)"); ax[2].set_ylabel("dz (mm)")
    ax[2].set_title("discharge gap only (wafer -> window)\n"
                    "the wafer-side cell is the clock setter")
    fig.tight_layout()
    path = os.path.join(args.outdir, "mesh_column.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
