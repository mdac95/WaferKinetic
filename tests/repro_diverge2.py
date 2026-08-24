"""Localize the divergence: single slices with small n_sub, find the
first non-finite field and its cells."""
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from waferkinetic.solvers import hybrid, electron_energy
from gec_case import gec_setup, TURNS

d = np.load(sys.argv[1] if len(sys.argv) > 1 else
            "outputs_asm/asm_plateau_fields.npz")
mesh, mask = gec_setup()
case = hybrid.HybridCase(P_set_W=500.0, p_torr=0.010)
setup = hybrid.build_setup(case, mesh, mask, TURNS)
nr, nz = mesh.Nr, mesh.Nz

def slice_probe(n_sub):
    num = hybrid.HybridNumerics(P_ramp_iters=0, uniform_dt=True,
                                es_joule="flux", eetm_slice=False,
                                wall_flux="sheath", sheath_model="asm",
                                dt_imp_max=4e-10, implicit_tol=1e-6,
                                n_gummel=1, chem_thr=0.5, use_accel=False,
                                implicit_electrons=True, n_sub_fkpm=n_sub)
    drv = hybrid.HybridDrivers(setup, num)
    state = hybrid.HybridState(
        ne=d["ne"], ni=d["ni"], n_eps=d["n_eps"], n_ars=d["n_ars"],
        ss_r=d["ss_r"], ss_z=d["ss_z"], Phi=d["Phi"],
        A=np.zeros((nr, nz), complex), I_coil=float(d["I_coil"]),
        ne_emm=np.zeros((nr, nz)), ni_prev=d["ni"], ars_prev=d["n_ars"])
    drv.refresh_chem(state.n_ars, force=True)
    state, diag = hybrid.outer_iteration(setup, num, drv, state, None)
    bad = {}
    for k in ("ne", "ni", "n_eps", "Phi"):
        v = getattr(state, k)
        nf = ~np.isfinite(v)
        big = np.isfinite(v) & (np.abs(v) > (1e4 if k == "Phi" else 3e21))
        if nf.any() or big.any():
            cells = np.argwhere(nf | big)
            bad[k] = (int(nf.sum()), int(big.sum()), cells[:4])
    print(f"n_sub={n_sub:4d}: " + ("CLEAN  ne_pk %.4e Phi_pk %.2f" %
          (state.ne.max(), np.nanmax(np.where(setup.pop.solved, state.Phi, np.nan)))
          if not bad else f"BAD {[(k, v[0], v[1]) for k, v in bad.items()]}"))
    for k, (nnf, nbig, cells) in bad.items():
        for c in cells:
            i, j = int(c[0]), int(c[1])
            print(f"    {k}[{i},{j}] r={mesh.r_c[i]*1e3:.1f} z={mesh.z_c[j]*1e3:.1f} "
                  f"val={getattr(state,k)[i,j]:.3e}  (ne={state.ne[i,j]:.2e} "
                  f"ni={state.ni[i,j]:.2e} neps={state.n_eps[i,j]:.2e})")
    return bad

for ns in (1, 2, 5, 10, 25, 50):
    if slice_probe(ns):
        break
