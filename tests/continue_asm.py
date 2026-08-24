"""Continue the ASM run warm from the run-3 dump to reach the plateau:
the ledger imbalance question needs a state that has stopped growing."""
import os
import sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from waferkinetic.solvers import hybrid
from gec_case import gec_setup, TURNS
from demo_implicit_diag import save_fields, plot_maps, Tee

out = "outputs_asm"
d = np.load(sys.argv[2] if len(sys.argv) > 2 else
            f"{out}/implicit_diag_implicit_flux_conv_udt_sheath_asm_noeetm_fields.npz")
mesh, mask = gec_setup(wall=os.environ.get("WK_WALL", "fine"))
case = hybrid.HybridCase(P_set_W=500.0, p_torr=0.010)
setup = hybrid.build_setup(case, mesh, mask, TURNS)
num = hybrid.HybridNumerics(P_ramp_iters=0, uniform_dt=True, es_joule="flux",
                            eetm_slice=False, wall_flux="sheath",
                            sheath_model="asm", dt_imp_max=4e-10,
                            implicit_tol=1e-6, n_gummel=1, chem_thr=0.5,
                            use_accel=False, implicit_electrons=True,
                            max_outer=int(sys.argv[1]))
nr, nz = mesh.Nr, mesh.Nz
state = hybrid.HybridState(
    ne=d["ne"], ni=d["ni"], n_eps=d["n_eps"], n_ars=d["n_ars"],
    ss_r=d["ss_r"], ss_z=d["ss_z"], Phi=d["Phi"],
    A=np.zeros((nr, nz), complex), I_coil=float(d["I_coil"]),
    ne_emm=np.zeros((nr, nz)),        # force EMM re-solve at entry
    ni_prev=d["ni"], ars_prev=d["n_ars"])
tee = Tee(f"{out}/{os.environ.get('WK_STEM', 'asm')}_continue.log"); sys.stdout = tee
t0 = time.time()
state, hist, ok = hybrid.run_hybrid(setup, num, state=state, verbose=True)
print(f"\ncontinuation: {'CONVERGED' if ok else 'not converged'} "
      f"after {len(hist)} iters in {time.time()-t0:.1f} s")
h = hist[-1]
print(f"end: ne={h['ne_pk']:.3e} Te={h['Te_pk']:.2f} Phi={h['Phi_pk']:.2f} "
      f"imb={h['imbalance']:.2%} rel={h['rel']:.2e} "
      f"floored={h['floored_frac']:.1%} slice={h['slice_advance']:.2e}")
save_fields(setup, state, f"{out}/{os.environ.get('WK_STEM', 'asm_plateau')}_fields.npz",
            wall_flux="sheath", sheath_model="asm")
plot_maps(setup, state, f"{out}/{os.environ.get('WK_STEM', 'asm_plateau')}_maps.png")
sys.stdout = tee.stdout; tee.close()
