"""
smoke_synthetic.py
==================
End-to-end smoke on a SYNTHETIC ~10x12 reactor, per the process rule:
build_setup -> seed_state -> HybridDrivers -> run_hybrid -> save_fields,
warnings-as-errors, in every requested flag state. py_compile is not a
test; this is the minimum that is.

  python smoke_synthetic.py                 # implicit + explicit, defaults
  python smoke_synthetic.py --modes implicit,explicit --chem baked,runtime

Exits nonzero on any warning, divergence, or non-finite dump field.
Prints the final peak state per mode so two flag states can be diffed.
"""

import argparse
import sys
import warnings

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

try:
    from waferkinetic.solvers import hybrid
    from waferkinetic.mesh.reactor_mesh import (Material, Mesh2D,
                                                ReactorGeometry, RectRegion)
except ImportError:
    import hybrid
    from reactor_mesh import Material, Mesh2D, ReactorGeometry, RectRegion

mm = 1e-3


def synthetic_setup():
    """Tiny ICP: plasma box, dielectric window under a coil, wafer +
    grounded walls. 10 x 12 cells -- one CPU trace in seconds."""
    r_faces = np.linspace(0.0, 50 * mm, 11)          # 10 r-cells
    z_faces = np.linspace(-5 * mm, 45 * mm, 13)      # 12 z-cells
    mesh = Mesh2D(r_faces, z_faces)

    geo = ReactorGeometry(r_max=50 * mm, z_min=-5 * mm, z_max=45 * mm)
    geo.add_region(RectRegion(0.0, 50 * mm, 35 * mm, 45 * mm,
                              Material.AIR, "air above window"))
    geo.add_region(RectRegion(0.0, 45 * mm, 30 * mm, 35 * mm,
                              Material.DIELECTRIC, "window"))
    geo.add_region(RectRegion(45 * mm, 50 * mm, -5 * mm, 45 * mm,
                              Material.GROUNDED_METAL, "side wall"))
    geo.add_region(RectRegion(0.0, 45 * mm, -5 * mm, 0.0,
                              Material.GROUNDED_METAL, "bottom plate"))
    geo.add_region(RectRegion(0.0, 25 * mm, -5 * mm, 0.0,
                              Material.WAFER, "wafer"))
    turns = [(10 * mm, 20 * mm, 37 * mm, 42 * mm)]
    for t in turns:
        geo.add_region(RectRegion(*t, Material.RF_ANTENNA, "coil"))
    mask = mesh.material_mask(geo)
    return mesh, mask, turns


def run_mode(implicit: bool, chem: str, iters: int, n_sub: int,
             sheath: str = "resolved"):
    mesh, mask, turns = synthetic_setup()
    case = hybrid.HybridCase(P_set_W=30.0, p_torr=0.010)
    setup = hybrid.build_setup(case, mesh, mask, turns)
    kw = {}
    if chem == "runtime":                 # only exists once step 2 lands
        kw["chem_runtime"] = True
    num = hybrid.HybridNumerics(P_ramp_iters=3, max_outer=iters,
                                n_sub_fkpm=n_sub, n_sub_eetm=n_sub,
                                n_sub_ars=n_sub,
                                uniform_dt=True, es_joule="flux",
                                eetm_slice=False, wall_flux="sheath",
                                dt_imp_max=4e-10, use_accel=False,
                                chem_thr=0.05,   # force mid-run re-bakes
                                sheath_model=sheath,
                                implicit_electrons=implicit, **kw)
    drv = hybrid.HybridDrivers(setup, num)
    state = hybrid.seed_state(setup)
    state, hist, _ = hybrid.run_hybrid(setup, num, state=state,
                                       drivers=drv, verbose=False)
    if len(hist) < iters:
        raise RuntimeError(f"run ended early after {len(hist)}/{iters} "
                           "iterations (divergence guard tripped)")
    # save_fields equivalent (demo helper needs matplotlib; keep it lean)
    Te = hybrid.electron_energy.temperature(state.ne, state.n_eps,
                                            setup.eparams)
    Er, Ez = setup.pop.efield(state.Phi)
    (Fr_e, Fz_e), (Fr_i, Fz_i) = hybrid.poisson._species_fluxes(
        setup.top, setup.pop, state.ne, state.ni, Te, Er, Ez,
        setup.fkpm_params, wall_flux=num.wall_flux, Phi=state.Phi,
        sheath_model=num.sheath_model)
    dump = dict(ne=state.ne, ni=state.ni, n_eps=state.n_eps,
                n_ars=state.n_ars, Phi=state.Phi, Te=Te, Er=Er, Ez=Ez,
                ss_r=state.ss_r, ss_z=state.ss_z,
                Fr_e=Fr_e, Fz_e=Fz_e, Fr_i=Fr_i, Fz_i=Fz_i)
    for k, v in dump.items():
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"non-finite {k} in the field dump")
    rec = hist[-1]
    print(f"  [{'imp' if implicit else 'exp'}/{chem}/{sheath}] "
          f"ne_pk={rec['ne_pk']:.6e}  Te_pk={rec['Te_pk']:.4f}  "
          f"Phi_pk={rec['Phi_pk']:.4f}  ss_tot={rec['ss_tot']:.6e}  "
          f"imb={rec['imbalance']:.2%}  bakes={drv.n_chem_bakes}")
    return dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="implicit,explicit")
    ap.add_argument("--chem", default="baked",
                    help="comma list: baked,runtime (runtime needs step 2)")
    ap.add_argument("--sheath", default="resolved",
                    help="comma list: resolved,asm (asm = implicit only)")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--n-sub", type=int, default=30)
    args = ap.parse_args()

    warnings.simplefilter("error")        # any runtime warning is a failure
    results = {}
    for m in args.modes.split(","):
        for c in args.chem.split(","):
            for sh in args.sheath.split(","):
                if sh == "asm" and m != "implicit":
                    continue          # asm is implicit-path only
                results[(m, c, sh)] = run_mode(m == "implicit", c,
                                               args.iters, args.n_sub,
                                               sheath=sh)
    # pairwise chem-mode comparison at fixed path: must agree bitwise-ish
    for m in args.modes.split(","):
        chems = args.chem.split(",")
        if len(chems) == 2:
            a, b = (results[(m, c, "resolved")] for c in chems)
            worst = max((float(np.max(np.abs(a[k] - b[k]))
                               / max(float(np.max(np.abs(a[k]))), 1e-300)), k)
                        for k in a)
            print(f"  {m}: max rel diff {chems[0]} vs {chems[1]} = "
                  f"{worst[0]:.3e} ({worst[1]})")
            if worst[0] > 1e-9:
                raise SystemExit(f"FAIL: {m} chem modes disagree")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
