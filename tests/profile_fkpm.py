"""
profile_fkpm.py -- step 3: attribute the FKPM stepper's cost.

Builds variant jitted steppers at the CONVERGED baseline state and times
them. Ablations isolate: Poisson CG, the implicit electron/energy
BiCGStab solves, the in-loop clip machinery (incl. the 14 smoothing
sweeps), and Gummel sweep count. maxiter=1 variants are TIMING-ONLY
(they truncate the solves); the state is discarded after each call.
"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from waferkinetic.solvers import hybrid, poisson, electron_energy
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

# chemistry closures exactly as refresh_chem bakes them
n_Ar = setup.active_f * np.maximum(case.ng - state.n_ars,
                                   setup.nparams.n_floor)
src_e = chemistry.make_jax_source_fn(setup.chem, species=chemistry.IE)
n_Ar_j, n_ars_j = jnp.asarray(n_Ar), jnp.asarray(state.n_ars)
def source_fn(ne, Te):
    return src_e((ne, n_Ar_j, n_ars_j, jnp.zeros_like(ne)), Te)
inelastic_fn = chemistry.make_jax_inelastic_fn(setup.chem, n_Ar, state.n_ars)

tc = drv_tc = None
import waferkinetic.solvers.transport as transport
tc = transport.to_jax(setup.top)
pc = poisson.to_jax(setup.pop)

BASE = dict(source_fn=source_fn, inelastic_fn=inelastic_fn,
            evolve_energy=True, n_sub=100, cg_tol=num.cg_tol,
            courant_clip=num.courant_clip, es_joule="flux",
            wall_flux="sheath", ion_wall="thermal", ion_field="static",
            implicit_electrons=True, n_gummel=2,
            im_tol=num.implicit_tol, im_maxiter=num.implicit_maxiter)

VARIANTS = [
    ("baseline (production)",            {}),
    ("poisson CG maxiter=1",             dict(cg_maxiter=1)),
    ("e+eps BiCGStab maxiter=1",         dict(im_maxiter=1)),
    ("all Krylov maxiter=1",             dict(cg_maxiter=1, im_maxiter=1)),
    ("no courant_clip machinery",        dict(courant_clip=None)),
    ("n_gummel=1",                       dict(n_gummel=1)),
    ("n_sub=200 (linearity check)",      dict(n_sub=200)),
]

args_j = tuple(jnp.asarray(a) for a in
               (state.ne, state.ni, state.n_eps, state.ss_r, state.ss_z,
                state.Phi, np.zeros_like(state.ss_r),
                np.zeros_like(state.ss_z), dt_e, dt_i, dt_eps, S_ext))

print(f"cells {mesh.Nr}x{mesh.Nz}={mesh.Nr*mesh.Nz}, n_sub=100, "
      f"n_gummel=2 -> 600 Krylov solves per baseline call")
rows = []
for name, over in VARIANTS:
    kw = dict(BASE, **over)
    step = poisson.make_jax_fkpm_stepper(tc, pc, setup.fkpm_params, **kw)
    t0 = time.perf_counter()
    out = step(*args_j)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = step(*args_j)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    t = min(ts)
    rows.append((name, t))
    print(f"{name:32s}  {t*1e3:8.1f} ms/call   (1st call {t_compile:5.1f} s)"
          f"   finite={bool(np.all(np.isfinite(np.asarray(out[0]))))}")

base = rows[0][1]
print("\n--- attribution (baseline - ablation) ---")
for name, t in rows[1:]:
    print(f"{name:32s}  saves {1e3*(base-t):8.1f} ms  "
          f"({100*(base-t)/base:5.1f}% of baseline)")
