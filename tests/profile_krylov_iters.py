"""Effective Krylov iteration counts inside the FKPM stepper, by
maxiter truncation: time and answer saturate once maxiter >= the
iterations the solver actually uses (while_loop exits early)."""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from waferkinetic.solvers import hybrid, poisson, electron_energy
import waferkinetic.solvers.transport as transport
from waferkinetic.chemistry import chemistry
from gec_case import gec_setup, TURNS

d = np.load("outputs/implicit_diag_implicit_flux_conv_udt_sheath_noeetm_fields.npz")
mesh, mask = gec_setup()
case = hybrid.HybridCase(P_set_W=500.0, p_torr=0.010)
setup = hybrid.build_setup(case, mesh, mask, TURNS)
num = hybrid.HybridNumerics(uniform_dt=True, es_joule="flux",
                            eetm_slice=False, wall_flux="sheath",
                            dt_imp_max=4e-10, chem_thr=0.5,
                            use_accel=False, implicit_electrons=True)
state = hybrid.HybridState(
    ne=d["ne"], ni=d["ni"], n_eps=d["n_eps"], n_ars=d["n_ars"],
    ss_r=d["ss_r"], ss_z=d["ss_z"], Phi=d["Phi"],
    A=np.zeros_like(d["ne"], dtype=np.complex128), I_coil=float(d["I_coil"]),
    ne_emm=d["ne"], ni_prev=d["ni"], ars_prev=d["n_ars"])
S_ext = setup.active_f * d["Q_ind"] / hybrid.QE
Er, Ez = setup.pop.efield(state.Phi)
Te = electron_energy.temperature(state.ne, state.n_eps, setup.eparams)
dt_e, dt_i, dt_eps = hybrid._local_dts(setup, num, state, Te, Er, Ez)
n_Ar = setup.active_f * np.maximum(case.ng - state.n_ars,
                                   setup.nparams.n_floor)
src_e = chemistry.make_jax_source_fn(setup.chem, species=chemistry.IE)
n_Ar_j, n_ars_j = jnp.asarray(n_Ar), jnp.asarray(state.n_ars)
def source_fn(ne, Te):
    return src_e((ne, n_Ar_j, n_ars_j, jnp.zeros_like(ne)), Te)
inelastic_fn = chemistry.make_jax_inelastic_fn(setup.chem, n_Ar, state.n_ars)
tc = transport.to_jax(setup.top)
pc = poisson.to_jax(setup.pop)
BASE = dict(source_fn=source_fn, inelastic_fn=inelastic_fn,
            evolve_energy=True, n_sub=100, cg_tol=num.cg_tol,
            courant_clip=num.courant_clip, es_joule="flux",
            wall_flux="sheath", ion_wall="thermal", ion_field="static",
            implicit_electrons=True, n_gummel=2,
            im_tol=num.implicit_tol, im_maxiter=num.implicit_maxiter)
args_j = tuple(jnp.asarray(a) for a in
               (state.ne, state.ni, state.n_eps, state.ss_r, state.ss_z,
                state.Phi, np.zeros_like(state.ss_r),
                np.zeros_like(state.ss_z), dt_e, dt_i, dt_eps, S_ext))

def run(kw):
    step = poisson.make_jax_fkpm_stepper(tc, pc, setup.fkpm_params,
                                         **dict(BASE, **kw))
    out = step(*args_j); jax.block_until_ready(out)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = step(*args_j); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return min(ts), np.asarray(out[0])

t_ref, ne_ref = run({})
print(f"reference (maxiter 2000): {t_ref*1e3:7.1f} ms")
print("\n-- implicit e/eps BiCGStab: im_maxiter sweep --")
for m in (1, 2, 5, 10, 15, 20, 30, 50, 100):
    t, ne = run(dict(im_maxiter=m))
    dv = np.max(np.abs(ne - ne_ref)) / max(np.max(np.abs(ne_ref)), 1e-300)
    print(f"im_maxiter {m:4d}: {t*1e3:7.1f} ms   ne rel dev vs ref {dv:.2e}")
print("\n-- Poisson CG: cg_maxiter sweep --")
for m in (1, 2, 5, 10, 15, 20, 30, 50, 100):
    t, ne = run(dict(cg_maxiter=m))
    dv = np.max(np.abs(ne - ne_ref)) / max(np.max(np.abs(ne_ref)), 1e-300)
    print(f"cg_maxiter {m:4d}: {t*1e3:7.1f} ms   ne rel dev vs ref {dv:.2e}")

print("\n-- tolerance sweep (timing + deviation) --")
for kw, label in ((dict(im_tol=1e-6), "im_tol 1e-6"),
                  (dict(im_tol=1e-4), "im_tol 1e-4"),
                  (dict(cg_tol=1e-5), "cg_tol 1e-5"),
                  (dict(im_tol=1e-6, cg_tol=1e-5), "both loose"),
                  (dict(n_gummel=1), "n_gummel 1"),
                  (dict(n_gummel=1, im_tol=1e-6, cg_tol=1e-5),
                   "gummel1+loose")):
    t, ne = run(kw)
    dv = np.max(np.abs(ne - ne_ref)) / max(np.max(np.abs(ne_ref)), 1e-300)
    print(f"{label:14s}: {t*1e3:7.1f} ms  ({100*t/t_ref:5.1f}% of ref)  "
          f"ne rel dev {dv:.2e}")

print("\n-- revised combination (cg_tol untouched) --")
for kw, label in ((dict(im_tol=1e-6, n_gummel=1), "im1e-6+gummel1"),
                  (dict(im_tol=1e-6, n_gummel=1, cg_tol=1e-7),
                   "  +cg_tol 1e-7"),):
    t, ne = run(kw)
    dv = np.max(np.abs(ne - ne_ref)) / max(np.max(np.abs(ne_ref)), 1e-300)
    print(f"{label:14s}: {t*1e3:7.1f} ms  ({100*t/t_ref:5.1f}% of ref)  "
          f"ne rel dev {dv:.2e}")
