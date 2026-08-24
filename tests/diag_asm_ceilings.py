"""Step 5 pre-analysis: which mechanism sets the ASM in-loop clock at
the dumped state? Mirrors _local_dts + the step_imp clip chain with the
battery-corrected field."""
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from waferkinetic.solvers import hybrid, electron_energy
import waferkinetic.solvers.transport as transport
from gec_case import gec_setup, TURNS

d = np.load(sys.argv[1])
mesh, mask = gec_setup()
case = hybrid.HybridCase(P_set_W=500.0, p_torr=0.010)
setup = hybrid.build_setup(case, mesh, mask, TURNS)
top, pop, p = setup.top, setup.pop, setup.fkpm_params
act = top.active
ne, ni, Te, Phi, n_eps = d["ne"], d["ni"], d["Te"], d["Phi"], d["n_eps"]
Er, Ez = pop.efield(Phi)
num = hybrid.HybridNumerics(uniform_dt=True, es_joule="flux",
                            eetm_slice=False, wall_flux="sheath",
                            dt_imp_max=None, chem_thr=0.5, use_accel=False,
                            implicit_electrons=True, sheath_model="asm")
state = hybrid.HybridState(ne=ne, ni=ni, n_eps=n_eps, n_ars=d["n_ars"],
                           ss_r=d["ss_r"], ss_z=d["ss_z"], Phi=Phi,
                           A=np.zeros_like(ne, complex), I_coil=1.0,
                           ne_emm=ne, ni_prev=ni, ars_prev=d["n_ars"])
# ceilings exactly as the driver computes them (battery-corrected, no cap)
dt_e, dt_i, dt_eps = hybrid._local_dts(setup, num, state, Te, Er, Ez)
print("ceilings entering the stepper (uniform_dt -> global min):")
for arr, nm in ((dt_e, "dt_e (chem)"), (dt_i, "dt_i (ion Courant, corrected E)"),
                (dt_eps, "dt_eps (sink/chem)")):
    a = np.where(act, arr, np.inf)
    ii, jj = np.unravel_index(np.argmin(a), a.shape)
    print(f"  {nm:34s} min {a.min():.3e} at ({ii:2d},{jj:2d}) "
          f"r={mesh.r_c[ii]*1e3:6.1f} z={mesh.z_c[jj]*1e3:6.1f}")
# ASM rc clip with corrected es scale
MI = 8.0*hybrid.QE*p.T_i_eV/(np.pi*p.vth_i**2)
cp = 0.5*(1-p.re)/(1+p.re)
chi = float(np.log(cp*np.sqrt(8.0*MI/(np.pi*hybrid.ME))))
ub = np.sqrt(hybrid.QE*np.maximum(Te, p.Te_min)/MI)
gw = ub*np.abs(ne-ni)
# corrected field at faces (reuse _local_dts helper logic via pop masks)
live_r = (pop.g_r > 0).astype(float); live_z = (pop.g_z > 0).astype(float)
bW = pop.wpl_r*(1-pop.epl_r)*live_r; bE = pop.epl_r*(1-pop.wpl_r)*live_r
bS = pop.spl_z*(1-pop.npl_z)*live_z; bN = pop.npl_z*(1-pop.spl_z)*live_z
TeW, TeE = transport._face_vals_r(Te); TeS, TeN = transport._face_vals_z(Te)
Erc = Er + (bE-bW)*chi*(bW*TeW+bE*TeE)/pop.dcp_r
Ezc = Ez + (bN-bS)*chi*(bS*TeS+bN*TeN)/pop.dcp_z
gwW, gwE = transport._face_vals_r(gw)
esr = np.maximum(np.abs(Erc), np.maximum(TeW, TeE)/pop.dcp_r)
rq_r = np.where(top.wall_e, hybrid.QE*gwW, 0.0)/(8.854e-12*esr) \
     + np.where(top.wall_w, hybrid.QE*gwE, 0.0)/(8.854e-12*esr)
gwS, gwN = transport._face_vals_z(gw)
esz = np.maximum(np.abs(Ezc), np.maximum(TeS, TeN)/pop.dcp_z)
rq_z = np.where(top.wall_n, hybrid.QE*gwS, 0.0)/(8.854e-12*esz) \
     + np.where(top.wall_s, hybrid.QE*gwN, 0.0)/(8.854e-12*esz)
rmax = max(rq_r.max(), rq_z.max())
kr = np.unravel_index(np.argmax(rq_r), rq_r.shape)
kz = np.unravel_index(np.argmax(rq_z), rq_z.shape)
print(f"  ASM rc-clip: max rate {rmax:.3e} -> dt {0.4/max(rmax,1e-300):.3e} "
      f"(argmax r-face {kr}, z-face {kz})")
# smoothing-sweep effect estimate: global min after 1.25^14 relaxation is
# bounded below by min itself (uniform_dt already collapses), so the
# effective per-substep dt = min(cap, dt_i_min, 0.4/rmax, dt_eps_min)
eff = min(np.where(act, dt_i, np.inf).min(),
          np.where(act, dt_eps, np.inf).min(), 0.4/max(rmax, 1e-300))
print(f"\npredicted in-loop clock (no cap): {eff:.3e} s/substep")
print(f"observed slice advance/100 from log: (compare manually)")
