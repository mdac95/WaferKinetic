"""
hybrid.py
=========
Outer hybrid iteration (model document, Secs. 12.3-12.5): the module
orchestrator that couples the validated WaferKinetic modules into a
converged, self-consistent 2D axisymmetric argon ICP steady state,

    EMM (inductive.py)  <->  EETM-A (electron_energy.py)
                        <->  FKPM (transport.py + poisson.py
                                   + chemistry.py + neutrals.py).

This is the all-fluid milestone: no Monte Carlo tiers, the doc Sec. 4.4
fidelity ceiling applies (see `FLUID_TIER_NOTICE`).

Module sequencing (doc Sec. 12.3)
---------------------------------
One outer iteration is one pass through the hierarchy (the CM and the
kinetic tiers are absent at this milestone):

  1. EMM   : frequency-domain A_phi solve with the *current* cold-plasma
             sigma(ne, nu_m). Power-controlled excitation (doc Sec. 3.3):
             operating conditions specify deposited power P_set, not coil
             current, and since E ~ I at fixed sigma and P ~ I^2, the
             coil current is rescaled every outer iteration,
             I -> I sqrt(P_set / P_dep). The linear solve itself is
             re-entered only when its input has drifted -- ne changed by
             more than `emm_thr` (a few percent) since the last solve
             (doc Sec. 12.3 module re-entry thresholds); between solves
             Q_ind is re-evaluated from the stored A at the current
             sigma and renormalized to P_set, which is exactly the
             power-control rescale.
  2. EETM  : fluid electron-energy relaxation (Option A) at frozen ne on
             the fresh Q_ind -- a short pure-energy sub-slice so the
             FKPM never sees a heating field inconsistent with n_eps.
  3. FKPM  : the coupled (ne, Ar+, n_eps, sigma_s, Phi) semi-implicit P2
             slice of `poisson.make_jax_fkpm_stepper(evolve_energy=True)`
             -- the EETM-FKPM pair advances *together* at the substep
             cadence, which is the doc Sec. 12.5 sub-slicing of the most
             tightly coupled pair taken to its finest limit.
  4. NGM   : the Ar* reaction-diffusion slice of `neutrals.py` at frozen
             (ne, Te, Ar+); ground-state Ar follows by the ideal-gas
             constraint and is never advanced.

Chemistry closures (Eq. 41 sources, Eq. 12 inelastic sum) freeze the
heavy densities for the electron sub-slices and are re-baked -- with the
attendant jit retrace -- only when Ar* has drifted beyond `chem_thr`
since the last bake: the doc Sec. 12.3 re-entry threshold applied to the
chemistry inputs. Refresh frequency decays to zero approaching steady
state, exactly as the doc describes; before convergence is *declared*,
both the EMM and the chemistry are force-refreshed so the answer never
rests on stale module inputs.

Acceleration to steady state (doc Sec. 12.4, scheme (b))
--------------------------------------------------------
Between outer iterations the slow densities (Ar*, Ar+) are extrapolated
from recorded history with clamped fractional change (doc Eq. 56),

    N_A = N_n (1 + delta),   delta = clip(xi (N_n - N_{n-1}) / N_n,
                                          delta_min, delta_max),

with delta_max > |delta_min| playing the role of the doc's gamma_a
(species grow from small seeds; positive acceleration may run faster
than negative, and overshoot to negative densities is impossible by
construction). Following acceleration the charge density rho(r) is
preserved by adjusting the electron density, ne -> ne + (Ar+_A - Ar+),
preventing spurious electric-field transients, and the electron energy
density is rescaled with the density change so Te (velocities and
temperatures) is preserved. Acceleration switches off permanently once
the natural per-iteration change falls below `accel_off_rel`, and a
minimum number of final iterations run unaccelerated before convergence
may be declared. With any low-order acceleration the pseudo-time loses
physical meaning: SS only, never transient studies -- as with the local
pseudo-time stepping of `transport.local_dt` that every sub-slice
already uses for within-module convergence.

Initial conditions (doc Sec. 5.5)
---------------------------------
Seed ne ~ 1e15-1e16 m^-3 (1e9-1e10 cm^-3) as the fundamental diffusion
mode (J0 x half-sine over the main discharge gap) at eps_bar ~ 4-5 eV;
Ar+ from *exact* initial electroneutrality; Ar* = 0; Phi = 0; sigma_s
= 0. Under-seeding fails to sustain. The early transient exhibits the
characteristic overshoot -- low initial conductivity permits deep field
penetration and excess ionization until the skin depth self-consistently
contracts -- which `run_hybrid` records per iteration (peak densities,
EM penetration depth, coil current, power ledger) so the demo can plot
it. In pseudo-time the transient is qualitative only.

Convergence and the power ledger
--------------------------------
Converged when the relative change of ne, n_Ar*, Te, and Phi between
outer iterations is below `tol` (default 1e-5) for `n_consecutive`
iterations, with acceleration off and fresh module inputs. The global
power ledger is evaluated (and, verbose, printed) every iteration: the
volume integral of the discrete steady electron-energy equation gives

    P_set + P_ES = P_wall,e + P_elastic + P_inelastic^net,

where P_ES is the (interior-face) electrostatic Joule term the Option A
closure carries, and P_inelastic^net = excitation - superelastic +
ionization + stepwise is the Eq. 12 sum -- the excitation share leaves
the discharge through Ar* wall de-excitation / radiation / quenching
(reported as a cross-check), the ionization share is the per-pair cost.
That identity is what closes to solver tolerance at the fixed point; the
headline P_set-only closure of the milestone criterion follows because
P_ES is a small fraction of P_set at ICP conditions (both are reported).

Differentiability and purity
----------------------------
Every inner stepper is the existing jitted, reverse-mode-differentiable
machinery (implicit gradients through the EMM/Poisson linear solves via
`lax.custom_linear_solve`, static-trip-count `fori_loop` sub-slices).
The orchestrator itself is deliberately NOT differentiated at this
milestone: it is plain NumPy glue, written pure-functionally --
`HybridState` is a frozen dataclass whose arrays are never mutated in
place; every step returns a new state -- so that a later
implicit-function-theorem / fixed-point adjoint wrapper (differentiate
the *converged* residual, not the iteration path) can be laid over it
without restructuring. The only stateful object is `HybridDrivers`, an
explicitly threaded cache of compiled steppers whose rebuild policy is
the Sec. 12.3 re-entry threshold; it affects performance and input
staleness (bounded by the thresholds), never the converged fixed point.

Units: SI + eV. Densities 1/m^3, powers W, Phi V, Q W/m^3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

try:  # package layout
    from waferkinetic.mesh.reactor_mesh import Material, Mesh2D
    from waferkinetic.solvers import (electron_energy, inductive, neutrals,
                                      poisson, transport)
    from waferkinetic.chemistry import chemistry
except ImportError:  # flat layout (tests / notebooks)
    from reactor_mesh import Material, Mesh2D
    import WaferKinetic.src.waferkinetic.chemistry.chemistry as chemistry
    import electron_energy
    import inductive
    import neutrals
    import poisson
    import transport

# ------------------------------------------------------------------ constants
QE = transport.QE
ME = transport.ME
KB = transport.KB
EPS0 = poisson.EPS0
MU0 = inductive.MU0
M_AR = neutrals.M_AR
C0 = 2.99792458e8
TORR = 133.322

#: Doc Sec. 4.4 fidelity ceiling -- printed by the validation suite so
#: the fluid-tier numbers are never misread as errors.
FLUID_TIER_NOTICE = """\
================== FLUID-TIER FIDELITY NOTICE (doc Sec. 4.4) ==================
This is the all-fluid closure (EETM Option A + drift-diffusion FKPM).
Documented, EXPECTED deviations from the kinetic (EMCS) tier for the
argon ICP benchmark -- these are not errors:
  * peak electron density HIGH by up to a factor of ~2 versus kinetic /
    probe values (the doc Sec. 13 target band is quoted pre-correction);
  * MONOTONIC Te profile: the on-axis Te minimum of nonlocal kinetics is
    absent by construction;
  * NO stochastic / anomalous (collisionless) skin-layer heating -- all
    heating is collisional Re(sigma)|E|^2 plus electrostatic drift work.
Density *profiles* agree well (the SS low-pressure charge density
approaches the fundamental diffusion mode).
==============================================================================="""


# ----------------------------------------------------------------------------
# Case, numerics, setup, state
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridCase:
    """Physical operating point (doc Sec. 13 reference: 500 W, 10 mTorr).

    k_mom is the illustrative constant momentum-transfer rate coefficient
    (nu_m = ng k_mom) matching `chemistry.FIT_ELASTIC`; it is replaced --
    together with every electron-impact rate -- by BOLSIG+ tables in
    production (doc Sec. 7 / chemistry.py docstring).
    """
    P_set_W: float = 500.0
    p_torr: float = 0.010
    Tg_K: float = 300.0
    omega: float = 2.0 * np.pi * 13.56e6
    k_mom: float = 1.0e-13
    re: float = 0.2
    ion_mass: float = M_AR
    mu_i_1torr: float = 0.14          # Ar+ in Ar, doc Eq. 27 reference
    eps_r_window: float = 4.2
    Te_min: float = 0.02
    Te_max: float = 15.0
    ne_floor: float = 1.0e12
    # ---- doc Sec. 5.5 seed ------------------------------------------------
    ne_seed: float = 5.0e15           # 5e9 cm^-3, inside the 1e9-1e10 band
    eps_seed_eV: float = 4.5          # eps_bar ~ 4-5 eV  (Te0 = 3 eV)
    seed_floor: float = 1.0e-2        # relative floor of the seed mode

    @property
    def p_Pa(self) -> float:
        return self.p_torr * TORR

    @property
    def ng(self) -> float:
        return self.p_Pa / (KB * self.Tg_K)

    @property
    def Tg_eV(self) -> float:
        return self.Tg_K / 11600.0

    @property
    def nu_m(self) -> float:
        return self.ng * self.k_mom

    @property
    def mass_ratio(self) -> float:
        return ME / self.ion_mass


@dataclass(frozen=True)
class HybridNumerics:
    """Orchestrator numerics: sub-slice sizes, re-entry thresholds
    (doc Sec. 12.3), Eq. 56 acceleration knobs (doc Sec. 12.4b), and the
    convergence contract."""
    # sub-slice trip counts (jitted fori_loop substeps per outer iteration)
    n_sub_fkpm: int = 100
    n_sub_eetm: int = 200
    n_sub_ars: int = 200
    # local pseudo-time (transport.local_dt machinery)
    cfl: float = 0.4
    courant_clip: float = 0.4         # FKPM in-loop re-clip (poisson.py)
    dt_eps_max: float = 3.0e-9        # stiff-inelastic guard (proven in
                                      # test_electron_energy test 3)
    chem_safety: float = 0.2          # dt <= safety / (chemical rate)
    # module re-entry thresholds (doc Sec. 12.3: "a few percent")
    emm_thr: float = 0.03             # ne drift triggering an EMM re-solve
    chem_thr: float = 0.05            # Ar* drift triggering a chem re-bake
    final_refresh_thr: float = 1.0e-3  # max staleness allowed at convergence
    # Eq. 56 acceleration (scheme b)
    xi: float = 100.0
    delta_min: float = -0.25
    delta_max: float = 0.50           # > |delta_min|: the gamma_a asymmetry
    accel_start: int = 3              # let the seed transient set a history
    accel_off_rel: float = 3.0e-4     # natural rel change disabling accel
    min_unaccel: int = 10             # final unaccelerated iterations (12.4)
    # convergence
    tol: float = 1.0e-5
    n_consecutive: int = 3
    max_outer: int = 400
    # linear solvers
    emm_tol: float = 1.0e-9
    cg_tol: float = 1.0e-8
    use_direct_emm: bool = False      # scipy SuperLU instead of jitted BiCGStab


@dataclass
class HybridSetup:
    """Static (per-case, per-mesh) precomputation: operators, parameter
    packs, coil excitation. Build once via `build_setup`; treat as
    immutable."""
    case: HybridCase
    mesh: Mesh2D
    mask: np.ndarray
    iop: inductive.InductiveOperator
    top: transport.TransportOperator
    pop: poisson.PoissonOperator
    chem: chemistry.ChemistrySet
    eps_r: np.ndarray                 # (Nr, Nz) relative permittivity
    J_unit: np.ndarray                # coil J_phi at I = 1 A
    active_f: np.ndarray              # float 0/1 plasma mask
    fkpm_params: poisson.FKPMParams
    eparams: electron_energy.ElectronParams
    nparams: neutrals.NeutralParams
    D_ars: float                      # flux-limited Ar* diffusion coeff
    Lambda: float                     # reactor diffusion length (Eq. 27)
    rxn: dict = field(default_factory=dict)   # reactions by name


def build_setup(case: HybridCase, mesh: Mesh2D, mask: np.ndarray,
                turns: Sequence, chem: chemistry.ChemistrySet | None = None
                ) -> HybridSetup:
    """Assemble every static operator and parameter pack for the case."""
    chem = chemistry.argon_table2() if chem is None else chem

    iop = inductive.build_operator(mesh, mask)
    top = transport.build_transport(mesh, mask)
    eps_r = np.where(mask == int(Material.DIELECTRIC),
                     case.eps_r_window, 1.0)
    pop = poisson.build_poisson(mesh, mask, eps_r)
    J_unit = inductive.coil_current_density(mesh, turns, 1.0)

    Lambda = neutrals.diffusion_length(top)
    # ion transport: constant-mobility Ar+ in Ar, Eq. 27 flux-limited
    mu_i0 = poisson.ion_mobility_1torr_scaled(case.p_torr, case.mu_i_1torr)
    vth_i = poisson.thermal_speed_heavy(case.Tg_eV, case.ion_mass)
    D_i, mu_i = poisson.flux_limited(mu_i0 * case.Tg_eV, case.Tg_eV,
                                     vth_i, Lambda)
    fkpm_params = poisson.FKPMParams(
        nu_m=case.nu_m, mu_i=float(mu_i), D_i=float(D_i),
        T_i_eV=case.Tg_eV, vth_i=vth_i, re=case.re,
        ne_floor=case.ne_floor, Te_min=case.Te_min, Te_max=case.Te_max,
        mass_ratio=case.mass_ratio, Tg_eV=case.Tg_eV)
    eparams = electron_energy.ElectronParams(
        nu_m=case.nu_m, mass_ratio=case.mass_ratio, Tg_eV=case.Tg_eV,
        re=case.re, ne_floor=case.ne_floor,
        Te_min=case.Te_min, Te_max=case.Te_max)
    nparams = neutrals.NeutralParams(p_Pa=case.p_Pa, Tg_K=case.Tg_K)
    D_ars = neutrals.metastable_D(top, nparams)

    return HybridSetup(
        case=case, mesh=mesh, mask=mask, iop=iop, top=top, pop=pop,
        chem=chem, eps_r=eps_r, J_unit=J_unit,
        active_f=top.active.astype(float),
        fkpm_params=fkpm_params, eparams=eparams, nparams=nparams,
        D_ars=D_ars, Lambda=Lambda,
        rxn={rx.name: rx for rx in chem.reactions})


@dataclass(frozen=True)
class HybridState:
    """The complete outer-iteration state. Frozen; arrays are treated as
    immutable (never modified in place) so the orchestrator stays
    pure-functional -- `replace(state, ...)` is the only update path.

    ni_prev / ars_prev are the recorded history the Eq. 56 extrapolation
    reads (post-acceleration values of the previous iteration, so the
    measured difference is this iteration's natural physics step).
    ne_emm is the electron density baked into the last EMM solve (the
    Sec. 12.3 re-entry reference).
    """
    ne: np.ndarray
    ni: np.ndarray
    n_eps: np.ndarray                 # eV / m^3
    n_ars: np.ndarray
    ss_r: np.ndarray                  # (Nr+1, Nz) dielectric surface charge
    ss_z: np.ndarray                  # (Nr, Nz+1)
    Phi: np.ndarray
    A: np.ndarray                     # complex A_phi phasor
    I_coil: float
    ne_emm: np.ndarray
    ni_prev: np.ndarray
    ars_prev: np.ndarray


def seed_state(setup: HybridSetup) -> HybridState:
    """Doc Sec. 5.5 initial conditions: fundamental-diffusion-mode ne
    seed at eps_bar = eps_seed_eV, Ar+ from exact electroneutrality,
    Ar* = 0, uncharged walls, zero fields."""
    from scipy.special import j0

    case, mesh, top = setup.case, setup.mesh, setup.top
    act = top.active

    # main discharge gap from the on-axis active column (wafer -> window)
    col = act[0, :]
    if not col.any():
        raise ValueError("no active cells on the symmetry axis")
    z_lo = float(mesh.z_c[col].min())
    z_hi = float(mesh.z_c[col].max())
    L = max(z_hi - z_lo, 1e-6)
    # radial extent of the active region at the mid-gap row
    jmid = int(np.argmin(np.abs(mesh.z_c - 0.5 * (z_lo + z_hi))))
    row = act[:, jmid]
    R = float(mesh.r_c[row].max()) if row.any() else float(mesh.r_c.max())

    prof = (np.clip(j0(2.404825557695773 * mesh.RC / R), 0.0, None)
            * np.clip(np.sin(np.pi * (mesh.ZC - z_lo) / L), 0.0, None))
    ne = np.where(act, case.ne_seed * np.maximum(prof, case.seed_floor), 0.0)

    nr, nz = mesh.Nr, mesh.Nz
    return HybridState(
        ne=ne,
        ni=ne.copy(),                              # exact neutrality
        n_eps=case.eps_seed_eV * ne,               # eps_bar ~ 4-5 eV
        n_ars=np.zeros((nr, nz)),                  # excited states negligible
        ss_r=np.zeros((nr + 1, nz)),
        ss_z=np.zeros((nr, nz + 1)),
        Phi=np.zeros((nr, nz)),
        A=np.zeros((nr, nz), dtype=np.complex128),
        I_coil=1.0,
        ne_emm=np.zeros((nr, nz)),                 # forces the first EMM solve
        ni_prev=ne.copy(),
        ars_prev=np.zeros((nr, nz)),
    )


# ----------------------------------------------------------------------------
# Driver cache: compiled inner steppers + Sec. 12.3 re-entry policy
# ----------------------------------------------------------------------------

class HybridDrivers:
    """Explicitly threaded cache of the jitted inner steppers.

    Not hidden state: the object is passed through `run_hybrid` and
    holds only compiled functions plus the reference fields against
    which the doc Sec. 12.3 re-entry thresholds are measured. Rebuilding
    the FKPM/EETM steppers re-bakes the frozen heavy densities into the
    chemistry closures (a jit retrace, so it is done exactly when the
    threshold demands and not per iteration); the converged fixed point
    is refreshed-input by construction (`run_hybrid` forces a final
    refresh before declaring convergence). Every stepper cached here is
    individually reverse-mode differentiable.
    """

    def __init__(self, setup: HybridSetup, num: HybridNumerics):
        import jax.numpy as jnp

        self.setup, self.num = setup, num
        self._jnp = jnp
        self._icoef = inductive.to_jax(setup.iop)
        self._tc = transport.to_jax(setup.top)
        self._pc = poisson.to_jax(setup.pop)
        self._emm = (None if num.use_direct_emm
                     else inductive.make_jax_solver(tol=num.emm_tol))
        self._ars_step = neutrals.make_jax_metastable_stepper(
            self._tc, setup.chem, setup.nparams, setup.D_ars,
            n_sub=num.n_sub_ars)
        self._eps_r_j = jnp.asarray(setup.eps_r)
        self._J_unit_j = jnp.asarray(setup.J_unit)
        self._fkpm = None
        self._eetm = None
        self._ars_ref: np.ndarray | None = None
        self.n_chem_bakes = 0
        self.n_emm_solves = 0

    # ---------------------------------------------------------- drift metrics
    def chem_drift(self, n_ars: np.ndarray) -> float:
        if self._ars_ref is None:
            return np.inf
        scale = max(float(self._ars_ref.max()), float(n_ars.max()), 1.0e14)
        return float(np.abs(n_ars - self._ars_ref).max()) / scale

    def emm_drift(self, state: HybridState) -> float:
        scale = max(float(state.ne.max()), 1.0)
        return float(np.abs(state.ne - state.ne_emm).max()) / scale

    # -------------------------------------------------------------- chemistry
    def refresh_chem(self, n_ars: np.ndarray, force: bool = False) -> bool:
        """Re-bake the frozen heavy densities into the FKPM/EETM
        chemistry closures when Ar* has drifted past `chem_thr` (or on
        demand). Ground-state Ar follows the ideal-gas constraint
        (neutrals.py; the ~1e-5 Ar+ share is below the constraint's own
        accuracy and is omitted, as documented there)."""
        if (not force and self._fkpm is not None
                and self.chem_drift(n_ars) <= self.num.chem_thr):
            return False
        jnp = self._jnp
        setup, case = self.setup, self.setup.case
        n_Ar = setup.active_f * np.maximum(
            case.ng - n_ars, setup.nparams.n_floor)

        src_e = chemistry.make_jax_source_fn(setup.chem,
                                             species=chemistry.IE)
        n_Ar_j = jnp.asarray(n_Ar)
        n_ars_j = jnp.asarray(n_ars)

        def source_fn(ne, Te):
            """S_e (== S_Ar+ for Table 2: ionization + stepwise +
            Penning each create one e and one Ar+) at frozen heavies."""
            return src_e((ne, n_Ar_j, n_ars_j, jnp.zeros_like(ne)), Te)

        inelastic_fn = chemistry.make_jax_inelastic_fn(setup.chem,
                                                       n_Ar, n_ars)
        self._fkpm = poisson.make_jax_fkpm_stepper(
            self._tc, self._pc, setup.fkpm_params,
            source_fn=source_fn, inelastic_fn=inelastic_fn,
            evolve_energy=True, n_sub=self.num.n_sub_fkpm,
            cg_tol=self.num.cg_tol, courant_clip=self.num.courant_clip)
        self._eetm = electron_energy.make_jax_energy_stepper(
            self._tc, setup.eparams, inelastic_fn=inelastic_fn,
            n_sub=self.num.n_sub_eetm)
        self._ars_ref = n_ars.copy()
        self.n_chem_bakes += 1
        return True

    # -------------------------------------------------------------------- EMM
    def solve_emm(self, sigma: np.ndarray, I_coil: float) -> np.ndarray:
        """A_phi at the given sigma and coil current (jitted BiCGStab
        with implicit gradients; SuperLU fallback on breakdown)."""
        self.n_emm_solves += 1
        if self._emm is not None:
            jnp = self._jnp
            A = np.asarray(self._emm(self._icoef, jnp.asarray(sigma),
                                     self.setup.case.omega,
                                     I_coil * self._J_unit_j,
                                     self._eps_r_j))
            if np.all(np.isfinite(A)):
                return A
            # BiCGStab breakdown: fall through to the direct reference path
        return self.setup.iop.solve_direct(sigma, self.setup.case.omega,
                                           I_coil * self.setup.J_unit,
                                           self.setup.eps_r)

    # ----------------------------------------------------------- slice calls
    def eetm_step(self, n_eps, ne, Er, Ez, S_ext_eV, dt) -> np.ndarray:
        jnp = self._jnp
        return np.asarray(self._eetm(
            jnp.asarray(n_eps), jnp.asarray(ne), jnp.asarray(Er),
            jnp.asarray(Ez), jnp.asarray(S_ext_eV), jnp.asarray(dt)))

    def fkpm_step(self, state: HybridState, dt_e, dt_i, dt_eps, S_ext_eV):
        jnp = self._jnp
        out = self._fkpm(
            jnp.asarray(state.ne), jnp.asarray(state.ni),
            jnp.asarray(state.n_eps), jnp.asarray(state.ss_r),
            jnp.asarray(state.ss_z), jnp.asarray(state.Phi),
            jnp.asarray(dt_e), jnp.asarray(dt_i), jnp.asarray(dt_eps),
            jnp.asarray(S_ext_eV))
        return tuple(np.asarray(a) for a in out)

    def ars_step(self, n_ars, ne, Te, ni, dt) -> np.ndarray:
        jnp = self._jnp
        return np.asarray(self._ars_step(
            jnp.asarray(n_ars), jnp.asarray(ne), jnp.asarray(Te),
            jnp.asarray(ni), jnp.asarray(dt)))


# ----------------------------------------------------------------------------
# Local pseudo-time assembly (transport.local_dt + chemistry stiffness caps)
# ----------------------------------------------------------------------------

def _heavy_ground_state(setup: HybridSetup, n_ars: np.ndarray) -> np.ndarray:
    return setup.active_f * np.maximum(setup.case.ng - n_ars,
                                       setup.nparams.n_floor)


def _local_dts(setup: HybridSetup, num: HybridNumerics, state: HybridState,
               Te: np.ndarray, Er: np.ndarray, Ez: np.ndarray):
    """Per-cell pseudo-time arrays for the FKPM slice (doc Sec. 12.4
    local time stepping, reusing `transport.local_dt`), with chemical
    stiffness caps so no substep outruns its stiffest volumetric source.
    The FKPM stepper re-clips against the *evolving* field internally
    (`courant_clip`), so these are ceilings, not exact Courant bounds."""
    case, top, p = setup.case, setup.top, setup.fkpm_params
    mu_e = QE / (ME * case.nu_m) * np.ones_like(state.ne)
    tiny = 1.0e-30

    n_Ar = _heavy_ground_state(setup, state.n_ars)
    k_iz = setup.rxn["ionization"].k(Te)
    k_step = setup.rxn["stepwise"].k(Te)
    nu_iz = k_iz * n_Ar + k_step * state.n_ars           # per-electron growth
    dt_chem = num.chem_safety / np.maximum(nu_iz, tiny)

    dt_e = transport.local_dt(top, mu_e * Te, mu_e, -1.0, Er, Ez,
                              cfl=num.cfl)
    dt_e = np.where(top.active, np.minimum(dt_e, dt_chem), 0.0)

    dt_i = transport.local_dt(top, p.D_i, p.mu_i, +1.0, Er, Ez, cfl=num.cfl)
    dt_i = np.where(top.active, np.minimum(dt_i, dt_chem), 0.0)

    dt_eps = electron_energy.energy_local_dt(top, state.ne, state.n_eps,
                                             Er, Ez, setup.eparams,
                                             cfl=num.cfl,
                                             dt_max=num.dt_eps_max)
    # energy-sink stiffness: never drain more than chem_safety of n_eps
    n = (state.ne, n_Ar, state.n_ars, state.ni)
    L = np.maximum(setup.chem.electron_energy_loss(n, Te), 0.0) \
        + np.maximum(3.0 * case.mass_ratio * case.nu_m * state.ne
                     * (Te - case.Tg_eV), 0.0)
    dt_sink = num.chem_safety * np.maximum(state.n_eps, 0.0) \
        / np.maximum(L, tiny)
    dt_eps = np.where(top.active, np.minimum(dt_eps, dt_sink), 0.0)
    return dt_e, dt_i, dt_eps


def _ars_dt(setup: HybridSetup, num: HybridNumerics, state: HybridState,
            Te: np.ndarray) -> np.ndarray:
    """Ar* local pseudo-time: diffusion bound + per-cell chemical sink
    cap (stepwise + superelastic + quadratic Penning + quenching)."""
    tiny = 1.0e-30
    n_Ar = _heavy_ground_state(setup, state.n_ars)
    sink = (setup.rxn["stepwise"].k(Te) * state.ne
            + setup.rxn["superelastic"].k(Te) * state.ne
            + 2.0 * chemistry.K_PENNING * state.n_ars
            + chemistry.K_QUENCH * n_Ar)
    dt = neutrals.metastable_local_dt(setup.top, setup.nparams,
                                      D_prime=setup.D_ars, cfl=num.cfl)
    cap = num.chem_safety / np.maximum(sink, tiny)
    return np.where(setup.top.active, np.minimum(dt, cap), 0.0)


# ----------------------------------------------------------------------------
# Power ledger (discrete steady electron-energy balance)
# ----------------------------------------------------------------------------

def power_ledger(setup: HybridSetup, state: HybridState,
                 Q: np.ndarray) -> dict:
    """Global power ledger, mirroring the FKPM stepper's discretization
    term by term (same Te clamps, same interior-face-restricted ES Joule
    field, same wall coefficients) so the residual measures convergence,
    not discretization mismatch:

        P_in + P_es = P_wall_e + P_elastic + P_inel_net.

    `imbalance` is |residual| / P_in. The Ar* wall de-excitation power
    (the destination of the net excitation share at SS) is reported as
    a cross-check, not a ledger term."""
    case, top, chem = setup.case, setup.top, setup.chem
    act = top.active
    V = top.volume
    Te = electron_energy.temperature(state.ne, state.n_eps, setup.eparams)

    P_in = float(np.sum(Q * V * act))
    P_wall = electron_energy.wall_energy_power(
        top, np.where(act, state.n_eps, 0.0), Te, setup.eparams)
    P_el = QE * float(np.sum(3.0 * case.mass_ratio * case.nu_m * state.ne
                             * (Te - case.Tg_eV) * V * act))

    n_Ar = _heavy_ground_state(setup, state.n_ars)
    n = (state.ne, n_Ar, state.n_ars, state.ni)
    rates = chem.rates(n, Te)
    channels = {}
    P_inel = 0.0
    for rx, r in zip(chem.reactions, rates):
        if rx.d_eps_eV is None:
            continue
        Pc = QE * float(np.sum(rx.d_eps_eV * r * V * act))
        channels[rx.name] = Pc
        P_inel += Pc

    # interior-face ES Joule term, exactly as inside the FKPM stepper
    Er, Ez = setup.pop.efield(state.Phi)
    ir = top.int_r.astype(float)
    iz = top.int_z.astype(float)
    Erc = 0.5 * (ir[:-1, :] * Er[:-1, :] + ir[1:, :] * Er[1:, :])
    Ezc = 0.5 * (iz[:, :-1] * Ez[:, :-1] + iz[:, 1:] * Ez[:, 1:])
    mu_e = QE / (ME * case.nu_m)
    P_es = QE * float(np.sum(state.ne * mu_e * (Erc ** 2 + Ezc ** 2)
                             * V * act))

    resid = P_in + P_es - (P_wall + P_el + P_inel)
    P_ars_wall = chemistry.EPS_EXC * QE * neutrals.wall_loss_rate(
        top, state.n_ars, setup.nparams)
    return dict(P_in=P_in, P_es=P_es, P_wall=P_wall, P_el=P_el,
                P_inel=P_inel, channels=channels, residual=resid,
                imbalance=abs(resid) / max(P_in, 1.0e-30),
                imbalance_no_es=abs(resid - P_es) / max(P_in, 1.0e-30),
                P_ars_wall=P_ars_wall)


# ----------------------------------------------------------------------------
# EM diagnostics: penetration depth and the collisional skin depth
# ----------------------------------------------------------------------------

def field_penetration_depth(setup: HybridSetup, Q: np.ndarray
                            ) -> tuple[float, tuple[int, int]]:
    """Axial 1/e penetration of Q_ind below its peak, along the peak's
    r-column: the doc Sec. 5.5 overshoot diagnostic (deep at ignition,
    contracting to ~ the skin depth as sigma builds up)."""
    act = setup.top.active
    Qm = np.where(act, Q, 0.0)
    if Qm.max() <= 0.0:
        return np.nan, (0, 0)
    i, j = np.unravel_index(int(np.argmax(Qm)), Qm.shape)
    zc = setup.mesh.z_c
    col = Qm[i, :]
    below = np.where((zc <= zc[j]) & (col < Qm[i, j] / np.e))[0]
    if below.size:
        z_e = zc[below.max()]
    else:                              # field fills the whole column
        z_e = zc[act[i, :]].min() if act[i, :].any() else zc[0]
    return float(zc[j] - z_e), (int(i), int(j))


def collisional_skin_depth(ne_peak: float, nu_m: float,
                           omega: float) -> float:
    """delta = 1 / Im(k), k = (omega/c) sqrt(1 - wpe^2/(w(w - j nu)))."""
    wpe2 = ne_peak * QE ** 2 / (EPS0 * ME)
    k = omega / C0 * np.sqrt(1.0 - wpe2 / (omega * (omega - 1j * nu_m))
                             + 0.0j)
    return float(1.0 / max(abs(k.imag), 1.0e-30))


# ----------------------------------------------------------------------------
# Eq. 56 acceleration (doc Sec. 12.4, scheme b)
# ----------------------------------------------------------------------------

def accelerate(setup: HybridSetup, num: HybridNumerics,
               state: HybridState) -> HybridState:
    """Bounded linear extrapolation of the slow densities (Ar*, Ar+)
    with clamped fractional change; charge density preserved by
    adjusting ne; n_eps rescaled with ne to preserve Te."""
    act = setup.top.active

    def extrap(N, N_prev):
        rel = (N - N_prev) / np.maximum(N, 1.0)
        delta = np.clip(num.xi * rel, num.delta_min, num.delta_max)
        return np.where(act, N * (1.0 + delta), N)

    ni_a = extrap(state.ni, state.ni_prev)
    ars_a = extrap(state.n_ars, state.ars_prev)
    # rho(r) preservation: rho = e (ni - ne) unchanged under ne += d(ni)
    ne_a = np.maximum(state.ne + (ni_a - state.ni), 0.0)
    # preserve Te = (2/3) n_eps / ne under the ne adjustment
    scale = np.where(state.ne > 0.0, ne_a / np.maximum(state.ne, 1.0e-30),
                     1.0)
    n_eps_a = state.n_eps * scale
    return replace(state, ne=ne_a, ni=ni_a, n_ars=ars_a, n_eps=n_eps_a)


# ----------------------------------------------------------------------------
# One outer iteration (doc Sec. 12.3 sequencing)
# ----------------------------------------------------------------------------

def outer_iteration(setup: HybridSetup, num: HybridNumerics,
                    drv: HybridDrivers, state: HybridState
                    ) -> tuple[HybridState, dict]:
    """EMM (power-controlled) -> chem refresh check -> EETM relaxation
    -> FKPM(+EETM sub-sliced) slice -> Ar* slice. Returns the new state
    (pre-acceleration) and a diagnostics dict including the ledger."""
    case = setup.case

    # ---- 1. EMM with re-entry threshold + Sec. 3.3 power control -------- #
    sigma = inductive.cold_plasma_sigma(state.ne, case.nu_m, case.omega)
    emm_refreshed = False
    if drv.emm_drift(state) > num.emm_thr:
        A = drv.solve_emm(sigma, state.I_coil)
        state = replace(state, A=A, ne_emm=state.ne.copy())
        emm_refreshed = True
    Q = inductive.power_deposition(state.A, sigma, case.omega)
    P_dep = float(np.sum(Q * setup.top.volume * setup.top.active))
    if not P_dep > 0.0:
        raise RuntimeError("EMM deposited no power in the plasma; the seed "
                           "may be below the sustainment floor (doc Sec. "
                           "5.5: under-seeding fails to sustain).")
    s2 = case.P_set_W / P_dep
    state = replace(state, A=state.A * np.sqrt(s2),
                    I_coil=state.I_coil * float(np.sqrt(s2)))
    Q = Q * s2                                     # integrates to P_set exactly
    S_ext = setup.active_f * Q / QE                # eV m^-3 s^-1

    # ---- chemistry re-entry check (rebakes FKPM + EETM closures) -------- #
    chem_refreshed = drv.refresh_chem(state.n_ars)

    # ---- 2. EETM relaxation at frozen ne (Sec. 12.3 step 3) ------------- #
    Er, Ez = setup.pop.efield(state.Phi)
    Te = electron_energy.temperature(state.ne, state.n_eps, setup.eparams)
    dt_eps0 = electron_energy.energy_local_dt(setup.top, state.ne,
                                              state.n_eps, Er, Ez,
                                              setup.eparams, cfl=num.cfl,
                                              dt_max=num.dt_eps_max)
    n_eps = drv.eetm_step(state.n_eps, state.ne, Er, Ez, S_ext, dt_eps0)
    state = replace(state, n_eps=n_eps)

    # ---- 3. FKPM slice, EETM sub-sliced inside (Sec. 12.3 step 4) ------- #
    Te = electron_energy.temperature(state.ne, state.n_eps, setup.eparams)
    dt_e, dt_i, dt_eps = _local_dts(setup, num, state, Te, Er, Ez)
    ne, ni, n_eps, ss_r, ss_z, Phi = drv.fkpm_step(state, dt_e, dt_i,
                                                   dt_eps, S_ext)
    state = replace(state, ne=ne, ni=ni, n_eps=n_eps,
                    ss_r=ss_r, ss_z=ss_z, Phi=Phi)

    # ---- 4. Ar* slice at frozen (ne, Te, Ar+) --------------------------- #
    Te = electron_energy.temperature(state.ne, state.n_eps, setup.eparams)
    n_ars = drv.ars_step(state.n_ars, state.ne, Te, state.ni,
                         _ars_dt(setup, num, state, Te))
    state = replace(state, n_ars=n_ars)

    # ---- diagnostics ----------------------------------------------------- #
    ledger = power_ledger(setup, state, Q)
    pen, _ = field_penetration_depth(setup, Q)
    act = setup.top.active
    diag = dict(
        ledger=ledger, Q=Q, Te=Te,
        ne_pk=float(state.ne.max()), ni_pk=float(state.ni.max()),
        ars_pk=float(state.n_ars.max()),
        Te_pk=float(Te[act].max()) if act.any() else np.nan,
        Phi_pk=float(state.Phi[act].max()) if act.any() else np.nan,
        I_coil=state.I_coil, pen_depth=pen,
        emm_refreshed=emm_refreshed, chem_refreshed=chem_refreshed)
    return state, diag


# ----------------------------------------------------------------------------
# The outer loop
# ----------------------------------------------------------------------------

def _rel_change(new: np.ndarray, old: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    denom = max(float(np.abs(new[mask]).max()), 1.0e-30)
    return float(np.abs((new - old)[mask]).max()) / denom


def run_hybrid(setup: HybridSetup, num: HybridNumerics | None = None,
               state: HybridState | None = None,
               drivers: HybridDrivers | None = None,
               verbose: bool = True,
               callback: Callable[[int, HybridState, dict], None] | None
               = None):
    """Drive the outer hybrid iteration to the converged steady state.

    Returns (state, history, converged). `history` is a list of
    per-iteration dicts (peaks, coil current, penetration depth, the
    full power ledger, relative changes, refresh/acceleration flags) --
    the record from which the doc Sec. 5.5 overshoot transient is
    plotted. Warm starting: pass the `state` of a previous run (e.g. the
    adjacent point of a power scan); `drivers` may be reused whenever
    `setup`/`num` are unchanged.

    Pure-functional contract: `state` is never mutated; each iteration
    produces a replacement. `drivers` caches compiled steppers only
    (Sec. 12.3 re-entry policy); before convergence is declared both the
    EMM and the chemistry are force-refreshed so the fixed point never
    rests on inputs staler than `final_refresh_thr`.
    """
    num = HybridNumerics() if num is None else num
    if state is None:
        state = seed_state(setup)
    if drivers is None:
        drivers = HybridDrivers(setup, num)

    history: list[dict] = []
    act = setup.top.active
    prev = None
    accel_enabled = True
    unaccel_since = None
    consecutive = 0
    converged = False

    for it in range(1, num.max_outer + 1):
        t0 = time.time()
        state, diag = outer_iteration(setup, num, drivers, state)

        # -- convergence metric on the pre-acceleration physics state ------ #
        Te = diag["Te"]
        te_mask = act & (state.ne > 1.0e-3 * max(state.ne.max(), 1.0))
        if prev is not None:
            rels = dict(
                ne=_rel_change(state.ne, prev["ne"], act),
                ars=_rel_change(state.n_ars, prev["ars"], act),
                Te=_rel_change(Te, prev["Te"], te_mask),
                Phi=_rel_change(state.Phi, prev["Phi"],
                                setup.pop.solved),
            )
            rel = max(rels.values())
        else:
            rels, rel = {}, np.inf
        prev = dict(ne=state.ne, ars=state.n_ars, Te=Te, Phi=state.Phi)

        # -- acceleration control (doc Sec. 12.4b) -------------------------- #
        if accel_enabled and rel < num.accel_off_rel:
            accel_enabled = False
            unaccel_since = it
            # entering the final unaccelerated phase on fresh inputs
            drivers.refresh_chem(state.n_ars, force=True)
        applied_accel = False
        if accel_enabled and it >= num.accel_start:
            state = accelerate(setup, num, state)
            applied_accel = True
        # Eq. 56 recorded history = end-of-iteration (post-accel) values
        state = replace(state, ni_prev=state.ni, ars_prev=state.n_ars)

        led = diag["ledger"]
        rec = dict(it=it, rel=rel, rels=rels, accel=applied_accel,
                   wall_time=time.time() - t0,
                   **{k: diag[k] for k in
                      ("ne_pk", "ni_pk", "ars_pk", "Te_pk", "Phi_pk",
                       "I_coil", "pen_depth", "emm_refreshed",
                       "chem_refreshed")},
                   **{k: led[k] for k in
                      ("P_in", "P_es", "P_wall", "P_el", "P_inel",
                       "imbalance", "imbalance_no_es", "P_ars_wall")})
        history.append(rec)
        if verbose:
            flags = "".join(f for f, on in
                            (("E", diag["emm_refreshed"]),
                             ("C", diag["chem_refreshed"]),
                             ("A", applied_accel)) if on)
            print(f"[{it:4d}] ne={rec['ne_pk']:.3e}  Ar*={rec['ars_pk']:.3e}"
                  f"  Te={rec['Te_pk']:5.2f}  Phi={rec['Phi_pk']:6.2f} V"
                  f"  | P {led['P_in']:.1f}(+ES {led['P_es']:.1f})"
                  f" = wall {led['P_wall']:.1f} + el {led['P_el']:.1f}"
                  f" + inel {led['P_inel']:.1f} W"
                  f"  imb {led['imbalance']:.2%}"
                  f"  rel {rel:.2e}  [{flags}]")
        if callback is not None:
            callback(it, state, rec)

        # -- convergence contract ------------------------------------------ #
        if not accel_enabled and rel < num.tol:
            consecutive += 1
        else:
            consecutive = 0
        if (consecutive >= num.n_consecutive and unaccel_since is not None
                and it - unaccel_since >= num.min_unaccel):
            stale = max(drivers.chem_drift(state.n_ars),
                        drivers.emm_drift(state))
            if stale > num.final_refresh_thr:
                # never converge on stale module inputs: refresh, keep going
                drivers.refresh_chem(state.n_ars, force=True)
                sigma = inductive.cold_plasma_sigma(state.ne, setup.case.nu_m,
                                                    setup.case.omega)
                state = replace(state, A=drivers.solve_emm(sigma,
                                                           state.I_coil),
                                ne_emm=state.ne.copy())
                consecutive = 0
                continue
            converged = True
            break

    if verbose:
        tag = "CONVERGED" if converged else "NOT converged"
        print(f"run_hybrid: {tag} after {len(history)} outer iterations "
              f"({drivers.n_emm_solves} EMM solves, "
              f"{drivers.n_chem_bakes} chemistry bakes)")
    return state, history, converged


# ----------------------------------------------------------------------------
# Convenience: recompute the converged heating field for diagnostics
# ----------------------------------------------------------------------------

def heating_field(setup: HybridSetup, state: HybridState) -> np.ndarray:
    """Q_ind (W/m^3) of the stored A at the current sigma, renormalized
    to P_set -- identical to what the next outer iteration would feed
    the energy equation."""
    case = setup.case
    sigma = inductive.cold_plasma_sigma(state.ne, case.nu_m, case.omega)
    Q = inductive.power_deposition(state.A, sigma, case.omega)
    P = float(np.sum(Q * setup.top.volume * setup.top.active))
    return Q * (case.P_set_W / max(P, 1.0e-30))