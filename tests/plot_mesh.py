""""
plot_mesh.py
============
Draw the mesh actually in use and report the cell size at every
plasma-facing surface -- the number that decides whether a sheath is
resolved or subgrid.

  python plot_mesh.py                       # fine mesh -> outputs/
  python plot_mesh.py --coarse
  python plot_mesh.py --outdir outputs --ne 6.6e17 --te 4.5

Writes <outdir>/mesh.png:
  left    material map with every cell edge drawn (as in the geometry
          figure), so the grid in use is never in doubt
  top-r   dr(r) with the radial material interfaces marked
  bot-z   dz(z) with the axial material interfaces marked, plus the
          Debye length at the quoted (ne, Te) for scale -- where the
          dz curve sits ABOVE the lambda_D line the sheath there is
          subgrid and the wall model is carrying it.
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    from waferkinetic.mesh.reactor_mesh import Material
except ImportError:
    from reactor_mesh import Material

from gec_case import gec_setup

NAMES = ["PLASMA", "DIELECTRIC", "GROUNDED_METAL", "RF_ANTENNA",
         "WAFER", "AIR"]
COLORS = ["#cfe3f7", "#e8a838", "#9aa0a6", "#c0432a", "#2e7d32", "#fdfcf0"]


def surfaces(mesh, mask):
    """Plasma-facing faces: (axis, index, coordinate) for every face with
    plasma on one side and a material on the other."""
    pl = mask == int(Material.PLASMA)
    out = {"r": [], "z": []}
    for i in range(mesh.Nr - 1):
        col = pl[i, :] ^ pl[i + 1, :]
        if col.any():
            out["r"].append((i + 1, mesh.r_faces[i + 1], int(col.sum())))
    for j in range(mesh.Nz - 1):
        row = pl[:, j] ^ pl[:, j + 1]
        if row.any():
            out["z"].append((j + 1, mesh.z_faces[j + 1], int(row.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--wall", default="fine", choices=("fine", "asm"),
                    help="axial wall grading: 'fine' clusters cells at "
                         "the wafer/window for a mesh-resolved sheath; "
                         "'asm' drops that clustering (doc P6 carries "
                         "the sheath analytically)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--ne", type=float, default=6.6e17,
                    help="density for the lambda_D reference line")
    ap.add_argument("--te", type=float, default=4.5,
                    help="Te (eV) for the lambda_D reference line")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    mesh, mask = (gec_setup(coarse=True) if args.coarse
                  else gec_setup(wall=args.wall))
    dr, dz = np.diff(mesh.r_faces), np.diff(mesh.z_faces)
    lam_D = 7430.0 * np.sqrt(args.te / args.ne)

    print(f"mesh: {mesh.Nr} x {mesh.Nz} = {mesh.Nr * mesh.Nz} cells"
          f"{'  (coarse)' if args.coarse else ''}")
    print(f"  dr: {dr.min() * 1e3:.3f} - {dr.max() * 1e3:.3f} mm")
    print(f"  dz: {dz.min() * 1e3:.3f} - {dz.max() * 1e3:.3f} mm")
    print(f"  lambda_D at ne={args.ne:.2e}, Te={args.te} eV: "
          f"{lam_D * 1e3:.4f} mm\n")
    surf = surfaces(mesh, mask)
    print("plasma-facing surfaces (cell size on the PLASMA side):")
    for j, zc, n in surf["z"]:
        d = max(dz[j - 1], dz[min(j, len(dz) - 1)])
        print(f"  z = {zc * 1e3:7.2f} mm ({n:3d} faces): dz = "
              f"{d * 1e3:6.3f} mm   lambda_D/dz = {lam_D / d:6.3f}"
              f"{'   <-- SUBGRID' if lam_D / d < 1 else ''}")
    for i, rc, n in surf["r"]:
        d = max(dr[i - 1], dr[min(i, len(dr) - 1)])
        print(f"  r = {rc * 1e3:7.2f} mm ({n:3d} faces): dr = "
              f"{d * 1e3:6.3f} mm   lambda_D/dr = {lam_D / d:6.3f}"
              f"{'   <-- SUBGRID' if lam_D / d < 1 else ''}")

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[:, 0])
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
    ax.pcolormesh(mesh.r_faces, mesh.z_faces, mask.T, cmap=cmap, norm=norm,
                  edgecolors="0.55", linewidth=0.15)
    ax.set_xlabel("r (m)"); ax.set_ylabel("z (m)"); ax.set_aspect("equal")
    ax.set_title(f"mesh in use: {mesh.Nr} x {mesh.Nz} = "
                 f"{mesh.Nr * mesh.Nz} cells")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.4") for c in COLORS]
    ax.legend(handles, NAMES, loc="upper right", fontsize=8, framealpha=0.9)

    a = fig.add_subplot(gs[0, 1])
    a.step(mesh.r_c, dr * 1e3, where="mid")
    for i, rc, _ in surf["r"]:
        a.axvline(rc, color="r", ls=":", lw=0.8)
    a.axhline(lam_D * 1e3, color="k", ls="--", lw=1,
              label=f"$\\lambda_D$ = {lam_D * 1e3:.3f} mm")
    a.set_yscale("log"); a.set_ylabel("dr (mm)"); a.set_xlabel("r (m)")
    a.set_title("radial cell size (red = plasma-facing surface)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 1])
    a.step(mesh.z_c, dz * 1e3, where="mid")
    for j, zc, _ in surf["z"]:
        a.axvline(zc, color="r", ls=":", lw=0.8)
    a.axhline(lam_D * 1e3, color="k", ls="--", lw=1,
              label=f"$\\lambda_D$ = {lam_D * 1e3:.3f} mm")
    a.set_yscale("log"); a.set_ylabel("dz (mm)"); a.set_xlabel("z (m)")
    a.set_title("axial cell size (red = plasma-facing surface)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout()
    tag = "_coarse" if args.coarse else \
        ("_asmwall" if args.wall == "asm" else "")
    path = os.path.join(args.outdir, f"mesh{tag}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()