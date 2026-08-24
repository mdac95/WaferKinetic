"""Locate the ASM pathology: where is the charge blob, what fields does
it make, and which clip term is actually binding at the dumped state."""
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from waferkinetic.solvers import hybrid, electron_energy
import waferkinetic.solvers.transport as transport
from waferkinetic.mesh.reactor_mesh import Material
from gec_case import gec_setup, TURNS

d = np.load(sys.argv[1])
mesh, mask = gec_setup()
case = hybrid.HybridCase(P_set_W=500.0, p_torr=0.010)
setup = hybrid.build_setup(case, mesh, mask, TURNS)
top, pop = setup.top, setup.pop
act = top.active
ne, ni, Te, Phi = d["ne"], d["ni"], d["Te"], d["Phi"]
n_eps = d["n_eps"]; Er, Ez = d["Er"], d["Ez"]

rho = np.where(act, ni - ne, 0.0)
i, j = np.unravel_index(np.argmax(np.abs(rho)), rho.shape)
print(f"max |ni-ne| = {abs(rho[i,j]):.3e} at cell ({i},{j}) "
      f"r={mesh.r_c[i]*1e3:.1f}mm z={mesh.z_c[j]*1e3:.1f}mm")
print(f"  there: ne={ne[i,j]:.3e} ni={ni[i,j]:.3e} Te={Te[i,j]:.2f} "
      f"Phi={Phi[i,j]:.2f} n_eps={n_eps[i,j]:.3e}")
print(f"  neighbors mask (5x5, this cell centered), material codes:")
for jj in range(min(j+2, mesh.Nz-1), max(j-3, -1), -1):
    row = " ".join(f"{Material(mask[ii,jj]).name[:4]:>4s}"
                   for ii in range(max(i-2,0), min(i+3, mesh.Nr)))
    print(f"    z={mesh.z_c[jj]*1e3:6.1f}mm  {row}")
# top-10 charge cells
flat = np.argsort(np.abs(rho).ravel())[::-1][:10]
print("top-10 |ni-ne| cells:")
for k in flat:
    ii, jj = np.unravel_index(k, rho.shape)
    print(f"  ({ii:2d},{jj:2d}) r={mesh.r_c[ii]*1e3:6.1f} z={mesh.z_c[jj]*1e3:6.1f} "
          f"|rho|={abs(rho[ii,jj]):.2e} ne={ne[ii,jj]:.2e} ni={ni[ii,jj]:.2e} "
          f"Te={Te[ii,jj]:.2f} Phi={Phi[ii,jj]:6.2f}")

print(f"\nfield maxima: |Er| {np.abs(Er).max():.3e}  |Ez| {np.abs(Ez).max():.3e} V/m")
for E, name in ((Er, "Er"), (Ez, "Ez")):
    k = np.unravel_index(np.argmax(np.abs(E)), E.shape)
    print(f"  max |{name}| at face {k}")

# --- clip chain at this state (numpy mirror of step_imp) ---
p = setup.fkpm_params
num = hybrid.HybridNumerics(uniform_dt=True, es_joule="flux",
                            eetm_slice=False, wall_flux="sheath",
                            dt_imp_max=4e-10, chem_thr=0.5, use_accel=False,
                            implicit_electrons=True, sheath_model="asm")
state = hybrid.HybridState(ne=ne, ni=ni, n_eps=n_eps, n_ars=d["n_ars"],
                           ss_r=d["ss_r"], ss_z=d["ss_z"], Phi=Phi,
                           A=np.zeros_like(ne, complex), I_coil=1.0,
                           ne_emm=ne, ni_prev=ni, ars_prev=d["n_ars"])
dt_e, dt_i, dt_eps = hybrid._local_dts(setup, num, state, Te, Er, Ez)
print(f"\n_local_dts ceilings (uniform): dt_e {dt_e[act].min():.2e} "
      f"dt_i {dt_i[act].min():.2e} dt_eps {dt_eps[act].min():.2e}")
# where is the dt_eps min (dt_sink or dt_chem)?
n_Ar = hybrid._heavy_ground_state(setup, d["n_ars"])
n = (ne, n_Ar, d["n_ars"], ni)
L = np.maximum(setup.chem.electron_energy_loss(n, Te), 0.0) \
    + np.maximum(3.0*case.mass_ratio*case.nu_m*ne*(Te-case.Tg_eV), 0.0)
dt_sink = 0.2*np.maximum(n_eps,0.0)/np.maximum(L,1e-30)
k_iz = setup.rxn["ionization"].k(Te); k_st = setup.rxn["stepwise"].k(Te)
dt_chem = 0.2/np.maximum(k_iz*n_Ar + k_st*d["n_ars"], 1e-30)
for arr, nm in ((dt_sink, "dt_sink"), (dt_chem, "dt_chem")):
    a = np.where(act, arr, np.inf)
    ii, jj = np.unravel_index(np.argmin(a), a.shape)
    print(f"  {nm} min {a.min():.2e} at ({ii},{jj}) r={mesh.r_c[ii]*1e3:.1f} "
          f"z={mesh.z_c[jj]*1e3:.1f}  ne={ne[ii,jj]:.2e} n_eps={n_eps[ii,jj]:.2e}")
# in-loop: ion Courant stab rate at the dumped field
mu = p.mu_i * np.ones_like(ne)
D = p.D_i * np.ones_like(ne)
dt_ion = transport.local_dt(top, D, mu, +1.0, Er, Ez, cfl=0.4)
a = np.where(act, dt_ion, np.inf)
ii, jj = np.unravel_index(np.argmin(a), a.shape)
print(f"  in-loop ion Courant min {a.min():.2e} at ({ii},{jj}) "
      f"r={mesh.r_c[ii]*1e3:.1f} z={mesh.z_c[jj]*1e3:.1f}")
# ASM rc clip: u_B |ne-ni| vs es scale
MI = 8.0*hybrid.QE*p.T_i_eV/(np.pi*p.vth_i**2)
ub = np.sqrt(hybrid.QE*np.maximum(Te, p.Te_min)/MI)
gw = ub*np.abs(ne-ni)
gwW, gwE = transport._face_vals_r(gw)
TeW, TeE = transport._face_vals_r(Te)
esr = np.maximum(np.abs(Er), np.maximum(TeW, TeE)/pop.dcp_r)
rq_r = np.where(top.wall_e, hybrid.QE*gwW/(8.854e-12*esr), 0.0) \
     + np.where(top.wall_w, hybrid.QE*gwE/(8.854e-12*esr), 0.0)
gwS, gwN = transport._face_vals_z(gw)
TeS, TeN = transport._face_vals_z(Te)
esz = np.maximum(np.abs(Ez), np.maximum(TeS, TeN)/pop.dcp_z)
rq_z = np.where(top.wall_n, hybrid.QE*gwS/(8.854e-12*esz), 0.0) \
     + np.where(top.wall_s, hybrid.QE*gwN/(8.854e-12*esz), 0.0)
rmax = max(rq_r.max(), rq_z.max())
print(f"  ASM rc-clip max rate {rmax:.2e} -> dt {0.4/max(rmax,1e-30):.2e}")
print(f"\nwall-cell quasineutrality: ", end="")
wall_cells = act & (np.roll(~act,1,0)|np.roll(~act,-1,0)|np.roll(~act,1,1)|np.roll(~act,-1,1))
qq = np.abs(ni-ne)/np.maximum(ne, 1e12)
print(f"max |ni-ne|/ne over wall cells {qq[wall_cells].max():.2e}, "
      f"bulk {qq[act & ~wall_cells].max():.2e}")
