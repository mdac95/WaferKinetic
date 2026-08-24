"""
postprocess_fields.py
=====================
Line cuts and sheath diagnostics from a `demo_implicit_diag.py` field
dump (*_fields.npz). Standalone: needs only numpy + matplotlib.

  python postprocess_fields.py outputs/..._fields.npz
  python postprocess_fields.py dump.npz --r 0.0 0.03 0.06 --z 0.02
  python postprocess_fields.py dump.npz --list          # show contents

Writes, next to the input file:
  <stem>_cut_z.png     Te, Phi, ne, ni along z at the requested radii
  <stem>_cut_r.png     the same along r at the requested heights
  <stem>_sheath.png    wall-approach detail: last cells before each
                       surface on the axial cut, with the potential drop
                       measured against the 4.7*Te ambipolar estimate,
                       plus lambda_D per cell vs the local cell size
                       (cells-per-lambda_D is THE sheath-resolution
                       number: < 1 means the sheath is subgrid and the
                       Boltzmann wall factor is doing the work)
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS0, QE = 8.8541878128e-12, 1.602176634e-19


def nearest(arr, v):
    i = int(np.argmin(np.abs(arr - v)))
    return i, float(arr[i])


def cell_line(F, ir=None, iz=None):
    """A cell field along one index line; NaN outside the plasma."""
    return F[ir, :] if iz is None else F[:, iz]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--r", type=float, nargs="+", default=[0.0, 0.04],
                    help="radii (m) for axial cuts (default 0 and 0.04)")
    ap.add_argument("--z", type=float, nargs="+", default=[0.02],
                    help="heights (m) for radial cuts (default mid-gap)")
    ap.add_argument("--list", action="store_true",
                    help="print the arrays in the dump and exit")
    args = ap.parse_args()

    d = np.load(args.npz)
    if args.list:
        for k in d.files:
            a = d[k]
            print(f"  {k:10s} {str(a.shape):>14s} {a.dtype}")
        return
    stem = os.path.splitext(args.npz)[0]

    r_c, z_c = d["r_c"], d["z_c"]
    act = d["active"].astype(bool)
    mask_nan = np.where(act, 1.0, np.nan)
    ne, ni, Te, Phi = (d[k] * mask_nan for k in ("ne", "ni", "Te", "Phi"))
    Phi_all = d["Phi"]                       # defined in dielectric too
    lam_D = 7430.0 * np.sqrt(np.where(act, d["Te"], np.nan)
                             / np.maximum(d["ne"], 1.0))
    dz = np.gradient(z_c)

    # ------------------------------------------------------------ axial --
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for rv in args.r:
        ir, rr = nearest(r_c, rv)
        lbl = f"r = {rr * 1e3:.0f} mm"
        ax[0, 0].plot(z_c, Te[ir, :], label=lbl)
        ax[0, 1].plot(z_c, Phi_all[ir, :], label=lbl)
        ax[1, 0].semilogy(z_c, ne[ir, :], label=lbl)
        ax[1, 1].plot(z_c, (ni - ne)[ir, :]
                      / np.nanmax(np.abs(ne[ir, :])), label=lbl)
    for a, t, yl in ((ax[0, 0], "electron temperature", "Te (eV)"),
                     (ax[0, 1], "potential (incl. dielectric)", "Phi (V)"),
                     (ax[1, 0], "electron density", "ne (m^-3)"),
                     (ax[1, 1], "charge separation",
                      "(ni - ne)/max ne")):
        a.set_title(t); a.set_ylabel(yl); a.grid(alpha=0.3)
        a.legend(fontsize=8)
    for a in ax[1, :]:
        a.set_xlabel("z (m)")
    fig.suptitle("axial cuts")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{stem}_cut_z.png", dpi=140)
    plt.close(fig)
    print(f"wrote {stem}_cut_z.png")

    # ----------------------------------------------------------- radial --
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for zv in args.z:
        iz, zz = nearest(z_c, zv)
        lbl = f"z = {zz * 1e3:.0f} mm"
        ax[0, 0].plot(r_c, Te[:, iz], label=lbl)
        ax[0, 1].plot(r_c, Phi_all[:, iz], label=lbl)
        ax[1, 0].semilogy(r_c, ne[:, iz], label=lbl)
        ax[1, 1].plot(r_c, (ni - ne)[:, iz]
                      / np.nanmax(np.abs(ne[:, iz])), label=lbl)
    for a, t, yl in ((ax[0, 0], "electron temperature", "Te (eV)"),
                     (ax[0, 1], "potential", "Phi (V)"),
                     (ax[1, 0], "electron density", "ne (m^-3)"),
                     (ax[1, 1], "charge separation",
                      "(ni - ne)/max ne")):
        a.set_title(t); a.set_ylabel(yl); a.grid(alpha=0.3)
        a.legend(fontsize=8)
    for a in ax[1, :]:
        a.set_xlabel("r (m)")
    fig.suptitle("radial cuts")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{stem}_cut_r.png", dpi=140)
    plt.close(fig)
    print(f"wrote {stem}_cut_r.png")

    # ----------------------------------------------------------- sheath --
    # Take the FIRST axial cut, find the plasma span, and zoom on the
    # approach to each z-surface.
    ir, rr = nearest(r_c, args.r[0])
    col = act[ir, :]
    if not col.any():
        print("first radius has no plasma cells; skip sheath figure")
        return
    # The active column can contain SEVERAL disconnected plasma regions
    # (e.g. the decaying seed pocket below the wafer plate). Taking the
    # first/last active index mixes surfaces of different regions and
    # produces nonsense like dPhi/Te ~ 1000. Use the contiguous segment
    # that contains the density maximum -- the main discharge.
    j_seg = []
    j = 0
    while j < len(col):
        if col[j]:
            j0 = j
            while j < len(col) and col[j]:
                j += 1
            j_seg.append((j0, j - 1))
        else:
            j += 1
    j_ne = int(np.nanargmax(np.where(col, d["ne"][ir, :], -1.0)))
    jlo, jhi = next((a, b) for a, b in j_seg if a <= j_ne <= b)
    if len(j_seg) > 1:
        others = [f"z=[{z_c[a]:.3f},{z_c[b]:.3f}]"
                  for a, b in j_seg if not (a <= j_ne <= b)]
        print(f"  note: {len(j_seg) - 1} disconnected plasma region(s) "
              f"on this cut ignored ({', '.join(others)})")
    n_zoom = min(8, (jhi - jlo + 1) // 2)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for k, (sl, name) in enumerate(
            ((slice(jlo, jlo + n_zoom), f"bottom surface (z={z_c[jlo]:.3f})"),
             (slice(jhi - n_zoom + 1, jhi + 1),
              f"top surface (z={z_c[jhi]:.3f})"))):
        zz = z_c[sl]
        a = ax[0, k]
        a2 = a.twinx()
        a.plot(zz, Phi_all[ir, sl], "o-", color="tab:blue", label="Phi")
        a2.plot(zz, ne[ir, sl], "s--", color="tab:green", label="ne")
        a2.plot(zz, ni[ir, sl], "^--", color="tab:red", label="ni")
        a2.set_yscale("log")
        a.set_title(f"wall approach, {name}, r={rr * 1e3:.0f} mm")
        a.set_xlabel("z (m)"); a.set_ylabel("Phi (V)", color="tab:blue")
        a2.set_ylabel("n (m^-3)")
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=8)
        a.grid(alpha=0.3)

    # measured wall drop vs ambipolar expectation, both surfaces
    j_pk = jlo + int(np.nanargmax(Phi[ir, jlo:jhi + 1]))
    # per-face flux ratio at the two wall faces: the direct test of the
    # floating condition (see chat) -- Gamma_e/Gamma_i must be ~1 at a
    # floating surface, and the measured dPhi/Te is whatever the wall
    # model NEEDS to get there
    Fz_e, Fz_i = d["Fz_e"], d["Fz_i"]
    # ion delivery estimate must use the field the ions drift in: the
    # Richards effective field when the dump carries one
    Ez = d["Ei_z"] if ("Ei_z" in d.files and d["Ei_z"].size) else d["Ez"]
    QE_, ME_ = 1.602176634e-19, 9.1093837015e-31
    MAR = 39.948 * 1.6605390666e-27
    print(f"  source file: {args.npz}")
    for jw, fac, name in ((jlo, jlo, "bottom"), (jhi, jhi + 1, "top")):
        ge, gi = abs(Fz_e[ir, fac]), abs(Fz_i[ir, fac])
        ni_w, Te_w = d["ni"][ir, jw], max(d["Te"][ir, jw], 1e-3)
        vthi = np.sqrt(8 * QE_ * 0.02585 / (np.pi * MAR))
        uB = np.sqrt(QE_ * Te_w / MAR)
        Ew = abs(Ez[ir, fac])
        mu_i = 14.0        # from FKPMParams at 10 mTorr; adjust if changed
        print(f"  {name:>6s} face: |Ge|={ge:.3e} |Gi|={gi:.3e} "
              f"Ge/Gi={ge / max(gi, 1e-300):.3f}")
        print(f"          ion delivery per ion (m/s): thermal "
              f"{0.25 * vthi:.2e} | Bohm {uB:.2e} | drift mu_i*E "
              f"{mu_i * Ew:.2e}   (actual Gi/ni_w = "
              f"{gi / max(ni_w, 1e-300):.2e})")
        print(f"          -> the LARGEST of these is what carries ions; "
              f"a Bohm floor on the thermal term only matters if "
              f"thermal/Bohm dominates drift")
    rows = []
    for jw, name in ((jlo, "bottom"), (jhi, "top")):
        dPhi = Phi_all[ir, j_pk] - Phi_all[ir, jw]
        Te_w = Te[ir, jw]
        # doc P6 / ASM dumps (chi > 0): the analytic barrier
        # dPhi_b = chi * Te_wall sits OUTSIDE the mesh; the total
        # plasma-to-surface drop is the in-mesh part plus the barrier.
        chi = float(d["chi"]) if "chi" in d.files else 0.0
        dPhi = dPhi + chi * Te_w
        rows.append((name, dPhi, Te_w, dPhi / max(Te_w, 1e-30)))
    a = ax[1, 0]
    x = np.arange(len(rows))
    a.bar(x - 0.15, [r[1] for r in rows], 0.3, label="measured drop (V)")
    a.bar(x + 0.15, [4.7 * r[2] for r in rows], 0.3,
          label="4.7 Te_wall (ambipolar)")
    a.set_xticks(x, [r[0] for r in rows])
    a.set_title("Phi_peak - Phi_wall vs ambipolar estimate\n"
                "(excess = what the wall model asks beyond 4.7 Te)")
    a.set_ylabel("V"); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    for xi, r in zip(x, rows):
        a.text(xi, max(r[1], 4.7 * r[2]) * 1.02,
               f"dPhi/Te = {r[3]:.1f}", ha="center", fontsize=8)

    a = ax[1, 1]
    a.semilogy(z_c[jlo:jhi + 1], lam_D[ir, jlo:jhi + 1], label="lambda_D")
    a.semilogy(z_c[jlo:jhi + 1], dz[jlo:jhi + 1], "--", label="cell dz")
    ratio_lo = lam_D[ir, jlo] / dz[jlo]
    ratio_hi = lam_D[ir, jhi] / dz[jhi]
    a.set_title("sheath resolution along the cut\n"
                f"lambda_D / dz at walls: {ratio_lo:.2f} (bottom), "
                f"{ratio_hi:.2f} (top); < 1 = subgrid sheath")
    a.set_xlabel("z (m)"); a.set_ylabel("m"); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.suptitle(os.path.basename(args.npz), fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{stem}_sheath.png", dpi=140)
    plt.close(fig)
    print(f"wrote {stem}_sheath.png")
    for name, dPhi, Te_w, ratio in rows:
        print(f"  {name:>6s}: dPhi = {dPhi:6.2f} V, Te_wall = "
              f"{Te_w:5.2f} eV, dPhi/Te = {ratio:5.2f} (ambipolar 4.7)")


if __name__ == "__main__":
    main()