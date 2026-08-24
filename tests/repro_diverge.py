"""Reproduce the it-248 ASM divergence from the saved it-247 state and
bisect the knob that controls it."""
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from waferkinetic.solvers import hybrid
from gec_case import gec_setup, TURNS

d = np.load(sys.argv[1] if len(sys.argv) > 1 else
            "outputs_asm/asm_plateau_fields.npz")
mesh, mask = gec_setup()
case = hybrid.HybridCase(P_set_W=500.0, p_torr=0.010)
setup = hybrid.build_setup(case, mesh, mask, TURNS)
nr, nz = mesh.Nr, mesh.Nz

def probe(tag, **over):
    kw = dict(P_ramp_iters=0, uniform_dt=True, es_joule="flux",
              eetm_slice=False, wall_flux="sheath", sheath_model="asm",
              dt_imp_max=4e-10, implicit_tol=1e-6, n_gummel=1,
              chem_thr=0.5, use_accel=False, implicit_electrons=True,
              max_outer=4)
    kw.update(over)
    num = hybrid.HybridNumerics(**kw)
    drv = hybrid.HybridDrivers(setup, num)
    state = hybrid.HybridState(
        ne=d["ne"], ni=d["ni"], n_eps=d["n_eps"], n_ars=d["n_ars"],
        ss_r=d["ss_r"], ss_z=d["ss_z"], Phi=d["Phi"],
        A=np.zeros((nr, nz), complex), I_coil=float(d["I_coil"]),
        ne_emm=np.zeros((nr, nz)), ni_prev=d["ni"], ars_prev=d["n_ars"])
    drv.refresh_chem(state.n_ars, force=True)
    state, hist, _ = hybrid.run_hybrid(setup, num, state=state,
                                       drivers=drv, verbose=False)
    n = len(hist)
    last = hist[-1] if hist else {}
    ok = n >= 4 and np.all(np.isfinite(state.ne))
    print(f"{tag:28s}: {'SURVIVES' if ok else f'DIVERGES after {n}'}"
          + (f"  (ne_pk {last.get('ne_pk', 0):.3e}, "
             f"imb {last.get('imbalance', 0):.2%})" if hist else ""))

probe("repro (run settings)")
probe("im_tol 1e-9", implicit_tol=1e-9)
probe("n_gummel 2", n_gummel=2)
probe("dt_cap 2e-10", dt_imp_max=2e-10)
probe("dt_cap 1e-10", dt_imp_max=1e-10)
probe("im_tol 1e-9 + gummel 2", implicit_tol=1e-9, n_gummel=2)
