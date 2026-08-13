"""
demo_implicit_diag.py
=====================
Run a FIXED number of outer iterations on the implicit (P3) path and
plot everything -- no convergence contract, no second path, no waiting.
This is the debugging counterpart to `demo_hybrid_gec.py`: it always
stops when told to and always produces figures, converged or not.

  python demo_implicit_diag.py                      # 60 iters, ramp 30
  python demo_implicit_diag.py --iters 25 --ramp 12
  python demo_implicit_diag.py --explicit           # same, explicit path
  python demo_implicit_diag.py --power 200 --pressure 5

Writes:
  implicit_diag_maps.png       ne, eps_bar (UNCLIPPED), Te, Phi, Ar*, Q
  implicit_diag_transient.png  power ledger, peaks, potential, clamp
                               fraction, pseudo-clock spread

ON PSEUDO-TIME -- read this before interpreting any "time" axis.
There is no global time in this solver. Every cell carries its own dt
(`transport.local_dt`), rebuilt from the local field and density each
outer iteration, and the FKPM re-clips it per substep. So:
  * dt is NOT the same between iterations -- it tracks the state and
    typically shrinks as the sheath sharpens;
  * dt is NOT the same between cells -- the spread across the mesh is
    routinely two decades, which is the entire point of local
    pseudo-time stepping;
  * summing dt over iterations does NOT give a physical elapsed time.
    The trajectory is a relaxation path to a fixed point, not a
    transient. Only the fixed point means anything.
The plots therefore show the dt SPREAD (min/median/max over plasma
cells) per iteration rather than a clock, plus the median cell's
nominal advance through one FKPM slice (n_sub_fkpm * dt) in
microseconds -- useful for judging stiffness relief, not for reading
off physical times.
"""

import argparse
import os
import sys
import time

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

GEOM = ((Material.GROUNDED_METAL, "0.4"), (Material.WAFER, "g"),
        (Material.RF_ANTENNA, "r"), (Material.DIELECTRIC, "orange"))


class Tee:
    """Mirror stdout to a log file: everything printed to the terminal
    is also written to disk, line-buffered so a Ctrl-C mid-run still
    leaves a complete log."""

    def __init__(self, path):
        self.fh = open(path, "w", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


def overlay(ax, mesh, mask):
    for m, cc in GEOM:
        ax.contour(mesh.r_c[None, :].repeat(mesh.Nz, 0),
                   mesh.z_c[:, None].repeat(mesh.Nr, 1),
                   (mask.T == int(m)).astype(float), levels=[0.5],
                   colors=cc, linewidths=0.7)


def check_module_versions():
    """Fail fast, with the file named, if the module set is mixed. The
    last change touched poisson.py, hybrid.py and this demo together;
    any stale one produces a cryptic TypeError deep in a jit trace."""
    import inspect
    msgs = []
    if "ion_field" not in inspect.signature(
            hybrid.poisson.make_jax_fkpm_stepper).parameters:
        msgs.append("poisson.py is STALE (no ion_field on the stepper)")
    if not hasattr(hybrid.HybridNumerics, "ion_field") and \
            "ion_field" not in {f.name for f in
                                __import__("dataclasses")
                                .fields(hybrid.HybridNumerics)}:
        msgs.append("hybrid.py is STALE (no ion_field on HybridNumerics)")
    if "Ei_r" not in {f.name for f in
                      __import__("dataclasses").fields(hybrid.HybridState)}:
        msgs.append("hybrid.py is STALE (no Ei_r on HybridState)")
    if msgs:
        raise SystemExit("MIXED MODULE VERSIONS:\n  "
                         + "\n  ".join(msgs)
                         + "\n  -> re-download ALL of: poisson.py, "
                           "hybrid.py, demo_implicit_diag.py, "
                           "test_implicit_electrons.py, "
                           "postprocess_fields.py")


def backend_banner():
    """Record what this run actually executed on. Set JAX_PLATFORMS=cpu
    (or CUDA_VISIBLE_DEVICES="") to force CPU with no code change --
    worth doing as a control, because at ~1e5 cells a field is under a
    megabyte and fits in CPU cache, while a GPU sees only tens of cells
    per FP64 core and becomes kernel-launch-latency bound. Note also
    that consumer (GeForce) cards run FP64 at 1/64 of FP32 peak, and
    this whole stack runs in float64."""
    dev = jax.devices()
    print(f"backend: {jax.default_backend()}  devices: "
          f"{[d.device_kind for d in dev]}  x64: "
          f"{jax.config.jax_enable_x64}")
    return jax.default_backend()


def run(args):
    mesh, mask = gec_setup()
    case = hybrid.HybridCase(P_set_W=args.power, p_torr=args.pressure)
    setup = hybrid.build_setup(case, mesh, mask, TURNS)
    num = hybrid.HybridNumerics(P_ramp_iters=args.ramp,
                                uniform_dt=args.uniform_dt,
                                es_joule=args.es_joule,
                                eetm_slice=args.eetm_slice,
                                P_ramp_frac0=args.ramp_frac0,
                                chem_thr=args.chem_thr,
                                dt_imp_max=args.dt_cap,
                                n_gummel=args.n_gummel,
                                wall_flux=args.wall_flux,
                                use_accel=args.use_accel,
                                ion_wall=args.ion_wall,
                                ion_field=args.ion_field,
                                n_sub_fkpm=args.n_sub,
                                implicit_electrons=not args.explicit)
    drv = hybrid.HybridDrivers(setup, num)
    state = hybrid.seed_state(setup)
    drv.refresh_chem(state.n_ars, force=True)

    path = "explicit" if args.explicit else "implicit"
    clock = "uniform (global min)" if args.uniform_dt else "per-cell"
    print(f"=== {path} path, {clock} clock, ES Joule '{args.es_joule}', "
          f"{args.iters} outer iterations, "
          f"{args.power:.0f} W over a {args.ramp}-iteration ramp, "
          f"{args.pressure * 1e3:.0f} mTorr ===")
    print(f"{'it':>4s} {'P_W':>7s} {'ne_pk':>10s} {'eps_bar':>9s} "
          f"{'Te':>6s} {'clamp':>6s} {'Phi':>7s} {'P_es':>9s} "
          f"{'P_wall':>9s} {'imb':>9s} {'dt_eff':>9s}")

    hist = []
    globals()["_PARTIAL_HIST"] = hist     # visible if this run raises
    t0 = time.time()
    for it in range(1, args.iters + 1):
        state, diag = hybrid.outer_iteration(setup, num, drv, state, it)
        led = diag["ledger"]
        rec = dict(it=it, **{k: diag[k] for k in
                             ("P_target", "ne_pk", "ars_pk", "Te_pk",
                              "eps_bar_pk", "clamped_frac", "Phi_pk",
                              "qn_dev", "ss_tot",
                              "I_coil", "dt_e_med", "dt_i_med",
                              "dt_eps_med", "dt_e_min", "dt_e_max",
                              "dt_eff_med", "slice_advance")},
                   **{k: led[k] for k in ("P_in", "P_es", "P_wall", "P_el",
                                          "P_inel", "imbalance")})
        hist.append(rec)
        print(f"{it:4d} {rec['P_target']:7.1f} {rec['ne_pk']:10.3e} "
              f"{rec['eps_bar_pk']:9.2f} {rec['Te_pk']:6.2f} "
              f"{rec['clamped_frac']:6.1%} {rec['Phi_pk']:7.2f} "
              f"{rec['P_es']:9.1f} {rec['P_wall']:9.1f} "
              f"{rec['imbalance']:9.1%} {rec['dt_eff_med']:9.2e}")
    print(f"\n{args.iters} iterations in {time.time() - t0:.1f} s "
          f"({(time.time() - t0) / args.iters:.2f} s/iteration)")
    return setup, state, hist


def run_converged(args):
    """Full convergence through `run_hybrid` (acceleration on)."""
    mesh, mask = gec_setup()
    case = hybrid.HybridCase(P_set_W=args.power, p_torr=args.pressure)
    setup = hybrid.build_setup(case, mesh, mask, TURNS)
    num = hybrid.HybridNumerics(P_ramp_iters=args.ramp,
                                uniform_dt=args.uniform_dt,
                                es_joule=args.es_joule,
                                eetm_slice=args.eetm_slice,
                                max_outer=args.max_outer,
                                P_ramp_gate_rel=args.ramp_gate,
                                P_ramp_frac0=args.ramp_frac0,
                                chem_thr=args.chem_thr,
                                dt_imp_max=args.dt_cap,
                                n_gummel=args.n_gummel,
                                n_sub_fkpm=args.n_sub,
                                wall_flux=args.wall_flux,
                                use_accel=args.use_accel,
                                ion_wall=args.ion_wall,
                                ion_field=args.ion_field,
                                implicit_electrons=not args.explicit)
    path = "explicit" if args.explicit else "implicit"
    print(f"=== {path} path -> run_hybrid, wall_flux='{args.wall_flux}', "
          f"n_gummel={args.n_gummel}, "
          f"n_sub={args.n_sub}, dt_cap={args.dt_cap}, "
          f"ES Joule '{args.es_joule}', "
          f"ramp {args.ramp}, max_outer {args.max_outer}, "
          f"{args.power:.0f} W, {args.pressure * 1e3:.0f} mTorr ===")
    print(f"{'it':>4s} {'Phi/Te':>7s} {'qn_dev':>10s} {'ss_tot(C)':>11s} "
          f"{'d(ss)/dit':>11s} {'floored':>8s} {'slice(s)':>10s}"
          f"   <- charge accumulation watch "
          f"(argon ambipolar Phi/Te = 4.7)")
    prev_ss = [None]

    def watch(it, st, rec):
        d = ("       ---" if prev_ss[0] is None
             else f"{rec['ss_tot'] - prev_ss[0]:11.3e}")
        prev_ss[0] = rec["ss_tot"]
        print(f"{it:4d} {rec['Phi_pk'] / max(rec['Te_pk'], 1e-30):7.2f} "
              f"{rec['qn_dev']:10.3e} {rec['ss_tot']:11.3e} {d} "
              f"{rec['floored_frac']:8.1%} "
              f"{rec['slice_advance']:10.2e}")

    hist_live = []
    globals()["_PARTIAL_HIST"] = hist_live   # survives an exception

    def watch2(it, st, rec):
        hist_live.append(rec)
        watch(it, st, rec)

    t0 = time.time()
    state, hist, ok = hybrid.run_hybrid(setup, num, verbose=True,
                                        callback=watch2)
    print(f"\n{'CONVERGED' if ok else 'NOT converged'} after {len(hist)} "
          f"outer iterations in {time.time() - t0:.1f} s")
    nb = sum(1 for h in hist if h["chem_refreshed"])
    print(f"  chemistry re-bakes (JIT retraces): {nb} of {len(hist)} "
          f"iterations -- each one recompiles the whole FKPM stepper")
    if ok:
        led = hist[-1]
        print(f"  final ledger imbalance {led['imbalance']:.3%}; "
              f"ne peak {led['ne_pk']:.3e} m^-3 "
              f"({led['ne_pk'] / 1e6:.3e} cm^-3); "
              f"Te peak {led['Te_pk']:.2f} eV; Phi peak "
              f"{led['Phi_pk']:.1f} V")
        print(hybrid.FLUID_TIER_NOTICE)
    return setup, state, hist


def save_fields(setup, state, path, ion_wall="thermal",
                wall_flux="thermal"):
    """(ion Er/Ez override and E_eff dump handled below)"""
    """Dump the grid and every field to a compressed .npz. Called on the
    state `run_hybrid` hands back -- which on a divergence is the LAST
    FINITE iteration, so the dump doubles as the pre-blowup snapshot.
    Load with np.load(path); cell arrays are (Nr, Nz), r-face arrays
    (Nr+1, Nz), z-face arrays (Nr, Nz+1)."""
    mesh, top, pop = setup.mesh, setup.top, setup.pop
    act = top.active
    Te = hybrid.electron_energy.temperature(state.ne, state.n_eps,
                                            setup.eparams)
    Er, Ez = pop.efield(state.Phi)
    Q = hybrid.heating_field(setup, state)
    # unlimited discrete face fluxes at the final field -- the same
    # Eq. 23/36/37 stencils the stepper uses
    (Fr_e, Fz_e), (Fr_i, Fz_i) = hybrid.poisson._species_fluxes(
        top, pop, state.ne, state.ni, Te, Er, Ez, setup.fkpm_params,
        ion_wall=ion_wall, wall_flux=wall_flux, Phi=state.Phi,
        Er_i=state.Ei_r, Ez_i=state.Ei_z)
    np.savez_compressed(
        path,
        # ---- grid ------------------------------------------------------
        r_c=mesh.r_c, z_c=mesh.z_c,
        r_faces=mesh.r_faces, z_faces=mesh.z_faces,
        mask=setup.mask, active=act, volume=top.volume,
        area_r=top.area_r, area_z=top.area_z,
        # ---- state (cell) ----------------------------------------------
        ne=state.ne, ni=state.ni, n_eps=state.n_eps, n_ars=state.n_ars,
        Phi=state.Phi, Te=Te,
        eps_bar=np.where(act, state.n_eps
                         / np.maximum(state.ne, setup.case.ne_floor), 0.0),
        Q_ind=Q, I_coil=np.float64(state.I_coil),
        # ---- fields / fluxes (faces) -----------------------------------
        Er=Er, Ez=Ez, ss_r=state.ss_r, ss_z=state.ss_z,
        Ei_r=(state.Ei_r if state.Ei_r is not None else np.zeros(0)),
        Ei_z=(state.Ei_z if state.Ei_z is not None else np.zeros(0)),
        Fr_e=Fr_e, Fz_e=Fz_e, Fr_i=Fr_i, Fz_i=Fz_i)
    kb = __import__("os").path.getsize(path) / 1024
    print(f"wrote {path} ({kb:.0f} kB): grid + ne, ni, n_eps, n_ars, "
          f"Te, Phi, Er, Ez, sigma_s, electron/ion face fluxes, Q_ind")


def plot_maps(setup, state, path):
    mesh, mask, act = setup.mesh, setup.mask, setup.top.active
    Te = hybrid.electron_energy.temperature(state.ne, state.n_eps,
                                            setup.eparams)
    eps_bar = np.where(act, state.n_eps
                       / np.maximum(state.ne, setup.case.ne_floor), np.nan)
    Q = hybrid.heating_field(setup, state)
    panels = (
        (np.where(act, state.ne, np.nan) / 1e6, "viridis",
         r"$n_e$ (cm$^{-3}$)"),
        (eps_bar, "hot", r"$\bar\varepsilon$ (eV), UNCLIPPED"),
        (np.where(act, Te, np.nan), "plasma", r"$T_e$ (eV), clamped"),
        (np.where(setup.pop.solved, state.Phi, np.nan), "cividis",
         r"$\Phi$ (V)"),
        (np.where(act, state.n_ars, np.nan) / 1e6, "magma",
         r"$n_{Ar^*}$ (cm$^{-3}$)"),
        (np.where(act, Q, np.nan) / 1e6, "inferno",
         r"$Q_{ind}$ (W/cm$^3$)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.5))
    for ax, (F, cmap, label) in zip(axes.ravel(), panels):
        pc = ax.pcolormesh(mesh.r_faces, mesh.z_faces, F.T, cmap=cmap,
                           shading="flat")
        fig.colorbar(pc, ax=ax, shrink=0.85, label=label)
        overlay(ax, mesh, mask)
        ax.set_xlabel("r (m)"); ax.set_ylabel("z (m)"); ax.set_aspect("equal")
    fig.suptitle("Implicit-path diagnostic state (NOT a converged "
                 "solution unless the ledger says so)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_transient(hist, path):
    g = lambda k: np.array([h[k] for h in hist])
    it = g("it")
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    a = ax[0, 0]
    a.plot(it, g("P_target"), "k-", label="$P_{set}$ (ramp)")
    a.plot(it, g("P_in"), "k--", label="$P_{in}$")
    a.plot(it, g("P_es"), label="$P_{es}$ (ES Joule)")
    a.plot(it, g("P_wall"), label="$P_{wall}$")
    a.plot(it, g("P_inel"), label="$P_{inel}$")
    a.set_yscale("log"); a.set_ylabel("W"); a.legend(fontsize=8)
    a.set_title("power ledger channels")

    a = ax[0, 1]
    a.plot(it, 100 * g("imbalance"), "r-")
    a.axhline(5.0, color="0.6", ls=":")
    a.set_yscale("log"); a.set_ylabel("|residual| / $P_{in}$  (%)")
    a.set_title("ledger imbalance (must fall)")

    a = ax[0, 2]
    a.plot(it, g("ne_pk"), label="$n_e$ peak")
    a.plot(it, g("ars_pk"), label="$n_{Ar^*}$ peak")
    a.set_yscale("log"); a.set_ylabel("m$^{-3}$"); a.legend(fontsize=8)
    a.set_title("peak densities")

    a = ax[1, 0]
    a.plot(it, g("eps_bar_pk"), "r-", label=r"$\bar\varepsilon$ peak")
    a.plot(it, g("Te_pk"), "b--", label="$T_e$ peak (clamped)")
    a.set_yscale("log"); a.set_ylabel("eV"); a.legend(fontsize=8)
    a.set_title("energy: unclipped vs clamped")

    a = ax[1, 1]
    a.plot(it, g("Phi_pk") / np.maximum(g("Te_pk"), 1e-30), "c-",
           label=r"$\Phi/T_e$")
    a.axhline(4.7, color="0.5", ls="--", label="argon ambipolar 4.7")
    a.set_ylabel(r"$\Phi_{pk}/T_{e,pk}$"); a.legend(fontsize=8)
    a.set_title("is the potential ambipolar?\n"
                "(drift away from 4.7 = charge accumulating)")
    a2 = a.twinx()
    a2.plot(it, g("ss_tot"), "r:", label="dielectric charge (C)")
    a2.set_ylabel("$\\sigma_s$ total (C)", color="r")

    a = ax[1, 2]
    a.fill_between(it, g("dt_e_min"), g("dt_e_max"), alpha=0.25,
                   label="$dt_e$ spread over cells")
    a.plot(it, g("dt_e_med"), label="$dt_e$ ceiling (median)")
    a.plot(it, g("dt_i_med"), label="$dt_i$ ceiling (median)")
    a.plot(it, g("dt_eff_med"), "k-", lw=2,
           label="EFFECTIVE common clock")
    a.set_yscale("log"); a.set_ylabel("s"); a.legend(fontsize=8)
    a.set_title("pseudo-clock: per-cell, per-iteration\n"
                "(one COMMON clock; ceilings are not what runs)")

    for a in ax.ravel():
        a.set_xlabel("outer iteration"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--converge", action="store_true",
                    help="hand off to hybrid.run_hybrid: Eq. 56 "
                         "acceleration, convergence contract and final "
                         "module refresh (the bare --iters loop has NO "
                         "acceleration, so it converges very slowly)")
    ap.add_argument("--max-outer", dest="max_outer", type=int, default=400)
    ap.add_argument("--ion-field", dest="ion_field", default="static",
                    choices=("static", "effective"),
                    help="ion drift field: 'static' is doc Eq. 23; "
                         "'effective' is the Richards inertia proxy "
                         "dE_eff/dt = nu_i(E - E_eff) (doc Sec. 5.1 opt. 2; "
                         "requires the implicit path)")
    ap.add_argument("--ion-wall", dest="ion_wall", default="thermal",
                    choices=("thermal", "bohm"),
                    help="ion wall flux: 'thermal' is doc Eq. 37; 'bohm' "
                         "floors the wall speed at u_B = sqrt(qTe/M) "
                         "(presheath criterion) and DEPARTS FROM THE DOC")
    ap.add_argument("--no-accel", dest="use_accel", action="store_false",
                    help="disable Eq. 56 acceleration. During ignition Ar* "
                         "is the FASTEST variable, not a slow one, so "
                         "extrapolating it amplifies the stepwise-ionization "
                         "runaway at any clamp size")
    ap.add_argument("--wall-flux", dest="wall_flux", default="thermal",
                    choices=("thermal", "sheath"),
                    help="electron wall flux: 'thermal' is doc Eq. 36; "
                         "'sheath' applies exp(-dPhi/Te) for the "
                         "unresolved sheath and DEPARTS FROM THE DOC")
    ap.add_argument("--n-gummel", dest="n_gummel", type=int, default=2,
                    help="Gummel block sweeps per FKPM substep (default 2). "
                         "The Phi<->ne coupling stiffens with dt; if the "
                         "step limit is Gummel convergence rather than "
                         "physics, raising this lifts the usable dt")
    ap.add_argument("--n-sub", dest="n_sub", type=int, default=100,
                    help="FKPM substeps per outer iteration. NOTE: total "
                         "work to reach a given pseudo-time is invariant "
                         "to this -- it regroups substeps, it does not "
                         "make them bigger")
    ap.add_argument("--dt-cap", dest="dt_cap", type=float, default=None,
                    help="absolute ceiling (s) on the implicit pseudo-clock. "
                         "Set it to the explicit clock (~4e-12) to compare "
                         "the two paths at EQUAL pseudo-time per iteration "
                         "-- the only comparison that isolates the implicit "
                         "sigma_s ledger from the larger step it takes")
    ap.add_argument("--ramp-frac0", dest="ramp_frac0", type=float,
                    default=0.05,
                    help="deposited fraction of P_set on ramp step 1 "
                         "(default 0.05 = 25 W at 500 W). Note the "
                         "sustainment floor: below it the discharge decays "
                         "rather than settling, so a very low start just "
                         "spends iterations watching the seed die")
    ap.add_argument("--chem-thr", dest="chem_thr", type=float, default=0.05,
                    help="Ar* drift triggering a chemistry re-bake. Each "
                         "re-bake is a full JIT RETRACE of the FKPM "
                         "stepper; during ignition Ar* grows ~40%%/iteration "
                         "so the 0.05 default fires every iteration and "
                         "dominates the runtime. Try 0.5 for transients.")
    ap.add_argument("--ramp-gate", dest="ramp_gate", type=float, default=None,
                    help="advance the power ramp only when the relative "
                         "change falls below this (residual-gated "
                         "continuation, e.g. 0.02); default: one ramp step "
                         "per iteration")
    ap.add_argument("--ramp", type=int, default=30)
    ap.add_argument("--power", type=float, default=500.0)
    ap.add_argument("--pressure", type=float, default=0.010)
    ap.add_argument("--explicit", action="store_true")
    ap.add_argument("--es-joule", dest="es_joule", default="drift",
                    choices=("drift", "flux"),
                    help="electrostatic Joule form: 'drift' is doc Eq. 12 "
                         "(mu_e ne E^2); 'flux' is -Gamma_e.E on the actual "
                         "electron flux and DEPARTS FROM THE DOC")
    ap.add_argument("--no-eetm-slice", dest="eetm_slice",
                    action="store_false",
                    help="drop the standalone EETM relaxation and let the "
                         "FKPM own the energy equation (removes a "
                         "redundant, unclipped, separately-sourced slice)")
    ap.add_argument("--log", default=None,
                    help="also write all terminal output to this file "
                         "(default: implicit_diag_<path>.log)")
    ap.add_argument("--outdir", default="outputs",
                    help="directory for all outputs (log, figures, field "
                         "dump); created if it does not exist "
                         "(default: outputs/)")
    ap.add_argument("--uniform-dt", dest="uniform_dt", action="store_true",
                    help="collapse the per-cell pseudo-clock to one global "
                         "minimum (conservative, slow: control experiment "
                         "for whether local time stepping is implicated)")
    args = ap.parse_args()

    tag = "explicit" if args.explicit else "implicit"
    os.makedirs(args.outdir, exist_ok=True)
    log = args.log or (f"implicit_diag_{tag}_{args.es_joule}"
                       f"{'_conv' if args.converge else ''}"
                       f"{'_udt' if args.uniform_dt else ''}"
                       f"{'_sheath' if args.wall_flux == 'sheath' else ''}"
                       f"{'' if args.eetm_slice else '_noeetm'}.log")
    log = os.path.join(args.outdir, log)
    tee = Tee(log)
    sys.stdout = tee
    try:
        check_module_versions()
        backend_banner()
        try:
            setup, state, hist = (run_converged(args) if args.converge
                                  else run(args))
        except Exception as exc:            # noqa: BLE001 - diagnostics
            import traceback
            traceback.print_exc()
            print(f"\nrun failed: {exc}\n"
                  "plotting whatever was recorded before the failure.")
            setup = state = None
            hist = globals().get("_PARTIAL_HIST", [])
            if not hist:
                raise
        if setup is not None and state is not None:
            plot_maps_ok = True
        else:
            plot_maps_ok = False
        stem = os.path.join(
            args.outdir,
            f"implicit_diag_{tag}_{args.es_joule}"
            f"{'_conv' if args.converge else ''}"
            f"{'_udt' if args.uniform_dt else ''}"
            f"{'_sheath' if args.wall_flux == 'sheath' else ''}"
            f"{'_bohm' if args.ion_wall == 'bohm' else ''}"
            f"{'_eff' if args.ion_field == 'effective' else ''}"
            f"{'' if args.eetm_slice else '_noeetm'}")
        if plot_maps_ok:
            plot_maps(setup, state, f"{stem}_maps.png")
            save_fields(setup, state, f"{stem}_fields.npz",
                        ion_wall=args.ion_wall, wall_flux=args.wall_flux)
        plot_transient(hist, f"{stem}_transient.png")

        last = hist[-1]
        print(f"\nfinal: eps_bar peak {last['eps_bar_pk']:.1f} eV "
              f"({last['clamped_frac']:.0%} of cells on the clamp), "
              f"imbalance {last['imbalance']:.1%}")
        print(f"median cell advanced {1e6 * last['slice_advance']:.4f} us "
              f"per FKPM slice; dt_e spread this iteration "
              f"{last['dt_e_min']:.2e} to {last['dt_e_max']:.2e} s "
              f"({last['dt_e_max'] / max(last['dt_e_min'], 1e-300):.0f}x "
              f"across the mesh)")
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"log written to {log}")


if __name__ == "__main__":
    main()