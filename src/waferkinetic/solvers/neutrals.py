"""
neutrals.py
===========
Neutral gas for the argon GEC benchmark: the *maximal reduction* of the
moment hierarchy (model document, Sec. 6, "Regime map and benchmark
reduction"). Closed chamber (v_k = 0, Pe = 0), moderate power, no
Franck-Condon channel, hence uniform Tgas -- the per-species moment
equations collapse to flux-limited reaction-diffusion,

    d(n_k)/dt - div( D'_k grad n_k ) = R_k,                     (doc Eq. 40)

and for the 4-species argon set only Ar* is transported. Ground-state
Ar is *diagnostic*, recovered from the ideal-gas constraint

    sum_k n_k kB Tgas = p     =>   n_Ar = p/(kB Tg) - n_Ar* - n_Ar+,

so it is never advanced and cannot drift from the constraint. This
reduction is a property of the benchmark, not of the model (doc Sec. 6);
the module is deliberately thin so the full moment tier replaces it
without touching the chemistry or transport layers.

Pieces:

* Chapman-Enskog binary diffusion coefficient for Ar* in Ar (first
  Chapman-Enskog approximation with the Neufeld et al. fit for the
  reduced collision integral Omega*(T*)), evaluated at p = 20 mTorr,
  Tg = 300 K by default. Ar Lennard-Jones parameters (sigma = 3.542 A,
  eps/kB = 93.3 K) are used for the Ar*-Ar pair as well -- an
  illustrative choice (the 4s atom is larger); labeled for replacement
  alongside the Boltzmann rate tables.

* The doc Eq. 27 flux limiter, D' = min(v_th Lambda, D) with Lambda the
  reactor diffusion length (fundamental-mode 1/Lambda^2 = (2.405/R)^2 +
  (pi/L)^2 from the active-region extents), guarding drift-diffusion
  transport at Kn ~ 0.1. The Eq. 27 mobility rescaling mu' = q D'/(kB T)
  is moot for neutrals (q = 0).

* Ar* wall loss with sticking gamma = 1 through the Chantry form,

    n . Gamma = gamma / (1 - gamma/2) * (1/4) v_th n,           (doc Eq. 47)

  i.e. the SKM degenerate case of doc Sec. 8.4: every wall contact
  de-excites (Ar* -> Ar with probability 1), and the returned
  ground-state atom is absorbed by the ideal-gas constraint rather than
  tracked as a return flux. Implemented via
  `transport.wall_flux_thermal(coeff = gamma/(1 - gamma/2)/4)`; at
  gamma = 1 the coefficient is 1/2 (twice the free-molecular 1/4,
  the Chantry/Motz-Wise finite-Knudsen correction).

* Volumetric R_Ar* from `chemistry.ChemistrySet.sources` (Eq. 41):
  excitation feeds Ar*; superelastic, stepwise ionization, Penning
  (quadratic), and quenching deplete it.

Discretization: the conservative FV machinery of `transport.py`
(diffusion-only closure 1A, d_gradn form -- D' is spatially uniform
under the benchmark's uniform Tgas, so the two forms coincide).
Static NumPy setup -> `transport.to_jax()` boundary; the jitted stepper
advances Ar* by `n_sub` explicit substeps at frozen (ne, Te, n_Ar+) --
the doc Sec. 12.3 sub-slicing cadence (neutral timescale ~ms versus the
electron sub-slice) -- and is reverse-mode differentiable
(static-trip-count fori_loop) and vmap-able over (ne, Te) batches.

Units: SI; p in Pa, Tg in K, D in m^2/s, densities 1/m^3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # package layout
    from waferkinetic.solvers.transport import (
        KB, TransportOperator, TransportCoeffsJAX, divergence,
        flux_drift_diffusion, wall_flux_thermal, stable_dt, local_dt)
    from waferkinetic.chemistry.chemistry import (
        ChemistrySet, make_jax_source_fn, IARS)
except ImportError:  # flat layout (tests / notebooks)
    from transport import (KB, TransportOperator, TransportCoeffsJAX,
                           divergence, flux_drift_diffusion,
                           wall_flux_thermal, stable_dt, local_dt)
    from WaferKinetic.src.waferkinetic.chemistry.chemistry import ChemistrySet, make_jax_source_fn, IARS

# ------------------------------------------------------------------ constants
M_AR = 39.948 * 1.66053906660e-27   #: argon atomic mass [kg]
SIGMA_LJ_AR = 3.542e-10             #: Ar Lennard-Jones sigma [m]
EPSK_LJ_AR = 93.3                   #: Ar Lennard-Jones eps/kB [K]


@dataclass(frozen=True)
class NeutralParams:
    """Benchmark neutral-gas parameters (uniform Tgas, closed chamber).

    p_Pa / Tg_K   : pressure and gas temperature (defaults: 20 mTorr,
                    300 K -- the GEC argon reference point).
    mass          : transported-species mass (Ar* = Ar).
    sigma_LJ/epsk : Lennard-Jones pair parameters for the Ar*-Ar
                    Chapman-Enskog D (illustrative: ground-state Ar
                    values, see module docstring).
    gamma_wall    : Ar* wall de-excitation probability (doc Sec. 8.4:
                    exactly 1 for the benchmark SKM degeneracy).
    n_floor       : floor for the diagnostic ground-state density,
                    guarding the constraint subtraction.
    """
    p_Pa: float = 0.02 * 133.322
    Tg_K: float = 300.0
    mass: float = M_AR
    sigma_LJ: float = SIGMA_LJ_AR
    epsk_LJ: float = EPSK_LJ_AR
    gamma_wall: float = 1.0
    n_floor: float = 1.0e10

    @property
    def ng_total(self) -> float:
        """Total heavy-particle density p/(kB Tg) [1/m^3]."""
        return self.p_Pa / (KB * self.Tg_K)

    @property
    def vth(self) -> float:
        """Mean thermal speed sqrt(8 kB Tg / (pi m)) [m/s]."""
        return float(np.sqrt(8.0 * KB * self.Tg_K / (np.pi * self.mass)))

    @property
    def chantry_coeff(self) -> float:
        """Wall-flux coefficient gamma/(1 - gamma/2)/4 of doc Eq. 47,
        in the `wall_flux_thermal` convention F = coeff * vth * n."""
        g = self.gamma_wall
        return g / (1.0 - 0.5 * g) / 4.0


# ----------------------------------------------------------------------------
# Chapman-Enskog diffusion + doc Eq. 27 flux limiter
# ----------------------------------------------------------------------------

def omega_diffusion(T_star: float) -> float:
    """Reduced collision integral Omega*(1,1)(T*) for the Lennard-Jones
    potential, Neufeld-Janzen-Aziz fit (standard first-CE closure)."""
    return (1.06036 / T_star ** 0.15610
            + 0.19300 * np.exp(-0.47635 * T_star)
            + 1.03587 * np.exp(-1.52996 * T_star)
            + 1.76474 * np.exp(-3.89411 * T_star))


def chapman_enskog_D(p: NeutralParams, mass_b: float = M_AR,
                     ) -> float:
    """First Chapman-Enskog binary diffusion coefficient [m^2/s] for the
    transported species (mass p.mass) in a bath of mass `mass_b`:

        D = (3/16) sqrt(2 pi kB^3 T^3 / mu_ab) / (p pi sigma^2 Omega*),

    with reduced mass mu_ab and the LJ pair parameters of `p`. At the
    default 20 mTorr / 300 K this evaluates to ~0.7 m^2/s (D ~ 1/p)."""
    mu_ab = p.mass * mass_b / (p.mass + mass_b)
    T_star = p.Tg_K / p.epsk_LJ
    num = 3.0 / 16.0 * np.sqrt(2.0 * np.pi * KB ** 3 * p.Tg_K ** 3 / mu_ab)
    den = p.p_Pa * np.pi * p.sigma_LJ ** 2 * omega_diffusion(T_star)
    return float(num / den)


def diffusion_length(op: TransportOperator) -> float:
    """Reactor fundamental-mode diffusion length Lambda of doc Eq. 27
    from the active-region bounding cylinder:
    1/Lambda^2 = (2.405/R)^2 + (pi/L)^2."""
    if not op.active.any():
        raise ValueError("no active cells")
    nr, nz = op.active.shape
    RC = np.broadcast_to(op.r_c[:, None], (nr, nz))
    ZC = np.broadcast_to(op.z_c[None, :], (nr, nz))
    # wall-face positions from cell centers + center-to-face distances
    r_e = np.where(op.wall_e[1:, :], RC + op.dc_r[1:, :], -np.inf)
    z_n = np.where(op.wall_n[:, 1:], ZC + op.dc_z[:, 1:], -np.inf)
    z_s = np.where(op.wall_s[:, :-1], ZC - op.dc_z[:, :-1], np.inf)
    R = max(float(r_e.max()), 1e-12)
    L = max(float(z_n.max() - z_s.min()), 1e-12)
    inv2 = (2.405 / R) ** 2 + (np.pi / L) ** 2
    return float(1.0 / np.sqrt(inv2))


def flux_limited_D(D: float, vth: float, Lambda: float) -> float:
    """Doc Eq. 27: D' = min(v_th Lambda, D). The companion mobility
    rescaling preserves the Einstein relation for charged species; for
    neutrals it is vacuous."""
    return float(min(vth * Lambda, D))


def metastable_D(op: TransportOperator, p: NeutralParams) -> float:
    """Flux-limited Chapman-Enskog D' for Ar* in Ar on this reactor:
    the single transport coefficient of the Eq. 40 reduction (spatially
    uniform under uniform Tgas and p)."""
    return flux_limited_D(chapman_enskog_D(p), p.vth, diffusion_length(op))


# ----------------------------------------------------------------------------
# Diagnostic ground state (ideal-gas constraint, doc Sec. 6)
# ----------------------------------------------------------------------------

def ground_state_density(p: NeutralParams, n_ars, n_ion=None):
    """n_Ar = p/(kB Tg) - n_Ar* - n_Ar+ (floored): the ideal-gas
    constraint sum_k n_k kB Tg = p over heavy species. Diagnostic, never
    transported -- wall-returned and quenched Ar* reappear here
    automatically. Pass n_ion = None while the ion tier is prescribed
    zero (its ~1e-5 relative contribution at benchmark conditions is
    below the constraint's own accuracy)."""
    n = p.ng_total - n_ars
    if n_ion is not None:
        n = n - n_ion
    return np.maximum(n, p.n_floor)


# ----------------------------------------------------------------------------
# Right-hand side and NumPy reference integrator (doc Eq. 40)
# ----------------------------------------------------------------------------

def metastable_rhs(op: TransportOperator, n_ars: np.ndarray,
                   ne: np.ndarray, Te: np.ndarray, chem: ChemistrySet,
                   p: NeutralParams, D_prime: float,
                   n_ion: np.ndarray | None = None):
    """Tendency d(n_Ar*)/dt of doc Eq. 40 with the Eq. 47 wall loss.

    Returns (dn, S_stack): S_stack is the full Eq. 41 source stack
    (e, Ar, Ar*, Ar+) so callers can wire S_e / S_Ar+ into their own
    continuity equations without re-assembling the chemistry.
    """
    n_Ar = np.where(op.active, ground_state_density(p, n_ars, n_ion), 0.0)
    zero = np.zeros_like(ne)
    n = (ne, n_Ar, n_ars, zero if n_ion is None else n_ion)
    S = chem.sources(n, Te)

    Fr, Fz = flux_drift_diffusion(op, n_ars, D_prime, form="d_gradn")
    Wr, Wz = wall_flux_thermal(op, n_ars, p.vth, p.chantry_coeff)
    dn = np.where(op.active,
                  -divergence(op, Fr + Wr, Fz + Wz) + S[IARS], 0.0)
    return dn, S


def advance(op: TransportOperator, n_ars: np.ndarray, ne: np.ndarray,
            Te: np.ndarray, chem: ChemistrySet, p: NeutralParams,
            dt, n_steps: int, D_prime: float | None = None,
            n_ion: np.ndarray | None = None):
    """Explicit (forward-Euler) NumPy reference integrator at frozen
    (ne, Te, n_ion). `dt` may be a scalar (time-accurate) or the
    (Nr, Nz) array from `metastable_local_dt` (steady-state pseudo-time,
    doc Sec. 12.4 -- transient not physical)."""
    if D_prime is None:
        D_prime = metastable_D(op, p)
    for _ in range(n_steps):
        dn, _ = metastable_rhs(op, n_ars, ne, Te, chem, p, D_prime, n_ion)
        n_ars = np.maximum(n_ars + dt * dn, 0.0)
    return n_ars


def metastable_stable_dt(op: TransportOperator, p: NeutralParams,
                         D_prime: float | None = None,
                         cfl: float = 0.4) -> float:
    """Global explicit step bound for the Ar* diffusion operator."""
    if D_prime is None:
        D_prime = metastable_D(op, p)
    return stable_dt(op, D_prime, cfl=cfl)


def metastable_local_dt(op: TransportOperator, p: NeutralParams,
                        D_prime: float | None = None, cfl: float = 0.4,
                        dt_max: float | None = None) -> np.ndarray:
    """Per-cell local pseudo-time step (steady-state acceleration; cap
    `dt_max` against the stiff quadratic Penning sink where n_Ar* is
    large on coarse cells)."""
    if D_prime is None:
        D_prime = metastable_D(op, p)
    return local_dt(op, D_prime, cfl=cfl, dt_max=dt_max)


def wall_loss_rate(op: TransportOperator, n_ars: np.ndarray,
                   p: NeutralParams) -> float:
    """Total Ar* wall de-excitation rate [1/s] (doc Eq. 47 integrated
    over the walls): the particle balance partner of the volumetric
    sources for steady-state audits."""
    Wr, Wz = wall_flux_thermal(op, n_ars, p.vth, p.chantry_coeff)
    return float(np.sum(op.area_r * np.abs(Wr))
                 + np.sum(op.area_z * np.abs(Wz)))


# ----------------------------------------------------------------------------
# JAX stepper (jitted, differentiable, vmap-able)
# ----------------------------------------------------------------------------

def make_jax_metastable_stepper(coeffs: TransportCoeffsJAX,
                                chem: ChemistrySet, p: NeutralParams,
                                D_prime: float, n_sub: int = 200):
    """Return a jitted stepper

        step(n_ars, ne, Te, n_ion, dt) -> n_ars'

    advancing doc Eq. 40 for Ar* by `n_sub` explicit substeps of `dt`
    (scalar or (Nr, Nz) local pseudo-time array) at frozen (ne, Te,
    n_ion) -- the doc Sec. 12.3 sub-slicing cadence. `D_prime` is the
    precomputed flux-limited coefficient from `metastable_D` (static:
    uniform Tgas). Formulas mirror `metastable_rhs` exactly (same
    d_gradn diffusion, Chantry wall, ideal-gas ground state, and Eq. 41
    source), so the NumPy path is the bitwise reference. Reverse-mode
    gradients propagate through all substeps (static-trip-count
    fori_loop -> scan); `vmap` over (ne, Te) batches for data
    generation.
    """
    import jax
    import jax.numpy as jnp

    src_ars = make_jax_source_fn(chem, species=IARS)
    cw = p.chantry_coeff * p.vth
    ng_tot = p.ng_total

    padW = lambda X: jnp.concatenate(
        [jnp.zeros((1, X.shape[1]), X.dtype), X], axis=0)
    padE = lambda X: jnp.concatenate(
        [X, jnp.zeros((1, X.shape[1]), X.dtype)], axis=0)
    padS = lambda X: jnp.concatenate(
        [jnp.zeros((X.shape[0], 1), X.dtype), X], axis=1)
    padN = lambda X: jnp.concatenate(
        [X, jnp.zeros((X.shape[0], 1), X.dtype)], axis=1)

    def step(n_ars, ne, Te, n_ion, dt):
        c = coeffs
        zero = jnp.zeros_like(ne)
        n_ion_ = zero if n_ion is None else n_ion

        def body(_, N):
            n_Ar = c.active * jnp.maximum(ng_tot - N - n_ion_, p.n_floor)
            S = src_ars((ne, n_Ar, N, n_ion_), Te)
            NW, NE = padW(N), padE(N)
            f = cw * N
            Fr = c.int_r * (-D_prime * (NE - NW) / c.dc_r) \
                + c.wall_e * padW(f) - c.wall_w * padE(f)
            NS, NN = padS(N), padN(N)
            Fz = c.int_z * (-D_prime * (NN - NS) / c.dc_z) \
                + c.wall_n * padS(f) - c.wall_s * padN(f)
            div = c.active * (
                (c.area_r[1:, :] * Fr[1:, :] - c.area_r[:-1, :] * Fr[:-1, :]
                 + c.area_z[:, 1:] * Fz[:, 1:]
                 - c.area_z[:, :-1] * Fz[:, :-1]) / c.volume)
            return jnp.maximum(N + dt * (-div + c.active * S), 0.0)

        return jax.lax.fori_loop(0, n_sub, body, n_ars)

    return jax.jit(step)