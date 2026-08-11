"""
demo_hybrid_gec.py
==================
Milestone demonstration: the converged, self-consistent, all-fluid 2D
argon ICP steady state on the GEC reference cell via the outer hybrid
iteration (hybrid.py, doc Secs. 12.3-12.5).

Produces:
  gec_hybrid_maps.png       2x3 maps: ne, Te, Phi, n_Ar*, Q_ind
                            (+ Ar+/ne quasineutrality panel), with
                            reactor geometry overlays
  gec_hybrid_transient.png  the doc Sec. 5.5 ignition overshoot: peak ne
                            vs iteration, field penetration depth
                            contracting to the collisional skin depth,
                            coil current under Sec. 3.3 power control,
                            power-ledger channels + closure
  gec_hybrid_power_scan.png near-linear density-vs-power with the ~100 W
                            sustainment threshold (unless --no-scan)

Usage:
  python demo_hybrid_gec.py                     # 500 W, 10 mTorr + scan
  python demo_hybrid_gec.py --power 800 --pressure 5 --no-scan
  python demo_hybrid_gec.py --fast              # smoke settings

Runtime: a converged case is minutes of CPU; the power scan runs three
more (warm-started). --fast loosens the tolerance for a quick look.
"""

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

try:  # package layout
    from waferkinetic.solvers import hybrid
    from waferkinetic.mesh.reactor_mesh import Material
except ImportError:  # flat layout
    import hybrid
    from reactor_mesh import Material

from gec_case import gec_setup, TURNS

Z_WINDOW = 0.040          # gec_case window bottom face (m)

GEOM_COLORS = ((Material.GROUNDED_METAL, "0.4"), (Material.WAFER, "g"),
               (Material.RF_ANTENNA, "r"), (Material.DIELECTRIC, "orange"))


def geometry_overlay(ax, mesh, mask):
    for m, cc in GEOM_COLORS:
        ax.contour(mesh.r_c[None, :].repeat(mesh.Nz, 0),
                   mesh.z_c[:, None].repeat(mesh.Nr, 1),
                   (mask.T == int(m)).astype(float), levels=[0.5],
                   colors=cc, linewidths=0.7)


def converge(P_set, p_torr, num, state=None, tag=""):
    mesh, mask = gec_setup()
    case = hybrid.HybridCase(P_set_W=P_set, p_torr=p_torr)
    setup = hybrid.build_setup(case, mesh, mask, TURNS)
    print(f"\n=== hybrid GEC {tag}: {P_set:.0f} W, "
          f"{p_torr * 1e3:.0f} mTorr ===")
    state, history, ok = hybrid.run_hybrid(setup, num, state=state,
                                           verbose=True)
    if not ok:
        print("WARNING: not converged to tolerance; plotting last state")
    return setup, state, history


def warm(setup_src, state_src, P_new, P_old):
    """Warm-start scaling for an adjacent power-scan point."""
    from dataclasses import replace
    s = P_new / P_old
    return replace(hybrid.seed_state(setup_src),
                   ne=state_src.ne * s, ni=state_src.ni * s,
                   n_eps=state_src.n_eps * s, n_ars=state_src.n_ars * s,
                   Phi=state_src.Phi.copy(), ss_r=state_src.ss_r.copy(),
                   ss_z=state_src.ss_z.copy(),
                   A=state_src.A.copy() * np.sqrt(s),
                   I_coil=state_src.I_coil,
                   ni_prev=state_src.ni * s, ars_prev=state_src.n_ars * s)


# --------------------------------------------------------------- figures --
def plot_maps(setup, state, path):
    mesh, mask = setup.mesh, setup.mask
    act = setup.top.active
    Te = hybrid.electron_energy.temperature(state.ne, state.n_eps,
                                            setup.eparams)
    Q = hybrid.heating_field(setup, state)

    panels = (
        (np.where(act, state.ne, np.nan) / 1e6, "viridis",
         r"$n_e$ (cm$^{-3}$)", None),
        (np.where(act, Te, np.nan), "plasma", r"$T_e$ (eV)", None),
        (np.where(setup.pop.solved, state.Phi, np.nan), "cividis",
         r"$\Phi$ (V)", None),
        (np.where(act, state.n_ars, np.nan) / 1e6, "magma",
         r"$n_{Ar^*}$ (cm$^{-3}$)", None),
        (np.where(act, Q, np.nan) / 1e6, "inferno",
         r"$Q_{ind}$ (W/cm$^3$)", None),
        (np.where(act & (state.ne > 1e-3 * state.ne.max()),
                  state.ni / np.maximum(state.ne, 1.0), np.nan),
         "coolwarm", r"$n_{Ar^+}/n_e$", (0.9, 1.1)),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.5))
    for ax, (F, cmap, label, clim) in zip(axes.ravel(), panels):
        pc = ax.pcolormesh(mesh.r_faces, mesh.z_faces, F.T, cmap=cmap,
                           shading="flat")
        if clim is not None:
            pc.set_clim(*clim)
        fig.colorbar(pc, ax=ax, shrink=0.85, label=label)
        geometry_overlay(ax, mesh, mask)
        ax.set_xlabel("r (m)"); ax.set_ylabel("z (m)")
        ax.set_aspect("equal")
    case = setup.case
    fig.suptitle(f"Converged hybrid GEC ICP steady state: "
                 f"{case.P_set_W:.0f} W deposited, "
                 f"{case.p_torr * 1e3:.0f} mTorr argon\n"
                 f"(all-fluid tier; see hybrid.FLUID_TIER_NOTICE for the "
                 f"documented Sec. 4.4 fidelity ceiling)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_transient(setup, state, history, path):
    it = np.array([h["it"] for h in history])
    ne_pk = np.array([h["ne_pk"] for h in history])
    pen = np.array([h["pen_depth"] for h in history])
    I = np.array([h["I_coil"] for h in history])
    imb = np.array([h["imbalance"] for h in history])
    P_wall = np.array([h["P_wall"] for h in history])
    P_el = np.array([h["P_el"] for h in history])
    P_inel = np.array([h["P_inel"] for h in history])

    delta = hybrid.collisional_skin_depth(float(state.ne.max()),
                                          setup.case.nu_m,
                                          setup.case.omega)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    ax = axes[0, 0]
    ax.semilogy(it, ne_pk / 1e6, "C0")
    ax.set_ylabel(r"peak $n_e$ (cm$^{-3}$)")
    ax.set_title("ignition overshoot (doc Sec. 5.5)")

    ax = axes[0, 1]
    ax.plot(it, pen * 1e3, "C1", label="1/e penetration of $Q_{ind}$")
    ax.axhline(delta * 1e3, color="k", ls="--",
               label=f"collisional skin depth {delta * 1e3:.1f} mm")
    ax.set_ylabel("depth (mm)")
    ax.set_title("deep penetration -> skin contraction")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(it, I, "C2")
    ax.set_ylabel("coil current (A)"); ax.set_xlabel("outer iteration")
    ax.set_title("Sec. 3.3 power control: $I \\to I\\sqrt{P_{set}/P_{dep}}$")

    ax = axes[1, 1]
    ax.stackplot(it, P_wall, P_el, P_inel,
                 labels=("wall,e", "elastic", "inelastic (net)"),
                 alpha=0.8)
    ax.axhline(setup.case.P_set_W, color="k", ls=":", label="$P_{set}$")
    ax2 = ax.twinx()
    ax2.semilogy(it, np.maximum(imb, 1e-6), "k", lw=0.8)
    ax2.set_ylabel("ledger imbalance")
    ax.set_ylabel("power (W)"); ax.set_xlabel("outer iteration")
    ax.set_title("global power ledger")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Hybrid outer-iteration transient, GEC ICP "
                 f"{setup.case.P_set_W:.0f} W / "
                 f"{setup.case.p_torr * 1e3:.0f} mTorr", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_power_scan(P, ne_pk, path):
    a, c = np.polyfit(P, ne_pk, 1)
    P0 = -c / a
    Pfit = np.linspace(min(P0, P.min()) * 0.9, P.max() * 1.05, 50)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(P, ne_pk / 1e6, "o", ms=8, label="converged hybrid")
    ax.plot(Pfit, (a * Pfit + c) / 1e6, "k--", lw=1,
            label=f"linear fit, threshold {P0:.0f} W")
    ax.axvline(P0, color="0.6", ls=":")
    ax.set_xlabel("deposited power (W)")
    ax.set_ylabel(r"peak $n_e$ (cm$^{-3}$)")
    ax.set_title("Near-linear density-power scaling, 10 mTorr argon\n"
                 "(stepwise ionization from Ar*, doc Sec. 13)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--power", type=float, default=500.0,
                    help="set-point deposited power (W)")
    ap.add_argument("--pressure", type=float, default=10.0,
                    help="pressure (mTorr)")
    ap.add_argument("--scan", dest="scan", action="store_true",
                    default=True)
    ap.add_argument("--no-scan", dest="scan", action="store_false",
                    help="skip the 200-1000 W power scan")
    ap.add_argument("--max-outer", type=int, default=None)
    ap.add_argument("--fast", action="store_true",
                    help="loose tolerance smoke run (not converged physics)")
    args = ap.parse_args()

    print(hybrid.FLUID_TIER_NOTICE)

    kw = {}
    if args.fast:
        kw = dict(tol=1.0e-4, max_outer=120, min_unaccel=5)
    if args.max_outer is not None:
        kw["max_outer"] = args.max_outer
    num = hybrid.HybridNumerics(**kw)

    p_torr = args.pressure * 1.0e-3
    setup, state, history = converge(args.power, p_torr, num, tag="base")

    led = hybrid.power_ledger(setup, state, hybrid.heating_field(setup,
                                                                 state))
    print(f"\nfinal ledger: P_set {led['P_in']:.1f} (+ES {led['P_es']:.2f})"
          f" = wall {led['P_wall']:.1f} + elastic {led['P_el']:.1f}"
          f" + inelastic {led['P_inel']:.1f} W;"
          f" imbalance {led['imbalance']:.2%}")

    plot_maps(setup, state, "gec_hybrid_maps.png")
    plot_transient(setup, state, history, "gec_hybrid_transient.png")

    if args.scan:
        pts = [p for p in (200.0, 500.0, 1000.0) if p != args.power]
        P_list, ne_list = [args.power], [float(state.ne.max())]
        for P in pts:
            _, st, _ = converge(P, p_torr, num,
                                state=warm(setup, state, P, args.power),
                                tag="scan")
            P_list.append(P); ne_list.append(float(st.ne.max()))
        order = np.argsort(P_list)
        plot_power_scan(np.array(P_list)[order],
                        np.array(ne_list)[order],
                        "gec_hybrid_power_scan.png")


if __name__ == "__main__":
    main()