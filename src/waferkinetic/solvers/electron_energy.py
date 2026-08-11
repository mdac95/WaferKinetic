"""
electron_energy.py
==================
EETM Option A (model document, Sec. 4.1): the fluid electron closure --
electron continuity with drift-diffusion flux and the electron
energy-density equation.

    d(ne)/dt   + div Gamma_e   = S_e,
        Gamma_e   = -mu_e ne E_S - grad(D_e ne)                (doc Eq. 11)

    d(neps)/dt + div Gamma_eps = S_eps,
        Gamma_eps = -mu_eps neps E_S - grad(D_eps neps),
        mu_eps = (5/3) mu_e,  D_eps = (5/3) D_e                (doc Eq. 13)

with n_eps = ne * eps_bar in eV/m^3, Te = (2/3) eps_bar, D_e = mu_e Te
(Einstein), mu_e = q / (me nu_m), and the energy source (doc Eq. 12)

    S_eps = (Q_ind + Q_es) / q  -  sum_j r_j d_eps_j
            - 3 (me/M) nu_m ne (Te - Tg)          [eV m^-3 s^-1]

where Q_ind is the cycle-averaged inductive heating produced by
`inductive.power_deposition` (identical expression to the bracketed
inductive term of Eq. 12), Q_es = q mu_e ne |E_S|^2 the electrostatic
Joule heating, the sum the inelastic losses (thresholds d_eps_j, supplied
by the chemistry layer or a callable), and the last term elastic exchange
with the gas. Stochastic sheath heating is not represented in this
closure (doc Sec. 4.1); the documented fluid-tier fidelity limits of doc
Sec. 4.4 apply.

Wall boundary conditions (doc Eq. 36, gamma_i = 0 for the argon
benchmark):

    n . Gamma_e   = (1 - re)/(1 + re) * (1/2) v_th ne
    n . Gamma_eps = (1 - re)/(1 + re) * (5/6) v_th n_eps,
    v_th = sqrt(8 q Te / (pi me)),  re = 0.2.

Discretization: the conservative FV machinery of `transport.py`
(closure 1A, grad(D N) form, donor-cell drift). Time integration is
explicit; `energy_stable_dt` provides the step bound.

JAX path: `make_jax_energy_stepper` returns a jitted stepper that
advances n_eps by `n_sub` explicit substeps per call (fixed ne, as when
the transport tier is not yet coupled). It is reverse-mode
differentiable (static-trip-count `fori_loop` lowers to scan) and
`vmap`-able over (ne, S_ext) batches for training-data generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # package layout
    from waferkinetic.solvers.transport import (
        QE, ME, TransportOperator, TransportCoeffsJAX,
        divergence, flux_drift_diffusion, wall_flux_thermal, stable_dt,
        local_dt)
except ImportError:  # flat layout
    from transport import (QE, ME, TransportOperator, TransportCoeffsJAX,
                           divergence, flux_drift_diffusion,
                           wall_flux_thermal, stable_dt, local_dt)

#: Gamma_eps coefficients relative to Gamma_e (doc Eq. 13).
ENERGY_FLUX_FACTOR = 5.0 / 3.0
#: Wall coefficients from doc Eq. 36 (times (1-re)/(1+re)).
WALL_PARTICLE = 0.5
WALL_ENERGY = 5.0 / 6.0


# ----------------------------------------------------------------------------
# Parameters and elementary closures
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ElectronParams:
    """Fluid-electron closure parameters.

    nu_m       : electron momentum-transfer collision frequency (1/s),
                 scalar or (Nr, Nz) array (from gas density and k_mom).
    mass_ratio : me / M_gas (elastic energy exchange).
    Tg_eV      : gas temperature in eV (T_gas[K] / 11600, doc Eq. 12).
    re         : electron wall reflection coefficient (doc Eq. 36).
    ne_floor   : density floor for Te = (2/3) n_eps / ne evaluation.
    Te_min/max : clamps for coefficient evaluation (v_th, D_e) in cells
                 where ne -> floor; the transported quantity n_eps itself
                 is never clamped.
    """
    nu_m: float | np.ndarray
    mass_ratio: float
    Tg_eV: float
    re: float = 0.2
    ne_floor: float = 1.0e12
    Te_min: float = 1.0e-2
    Te_max: float = 30.0


def mobility(nu_m) -> np.ndarray:
    """Collisional electron mobility mu_e = q / (me nu_m)."""
    return QE / (ME * np.asarray(nu_m, dtype=float))


def thermal_speed(Te_eV) -> np.ndarray:
    """v_th = sqrt(8 q Te / (pi me)) (doc Eq. 36)."""
    return np.sqrt(8.0 * QE * np.asarray(Te_eV, dtype=float) / (np.pi * ME))


def temperature(ne: np.ndarray, n_eps: np.ndarray,
                p: ElectronParams) -> np.ndarray:
    """Te = (2/3) n_eps / ne with the floor/clamps of `ElectronParams`."""
    return np.clip((2.0 / 3.0) * n_eps / np.maximum(ne, p.ne_floor),
                   p.Te_min, p.Te_max)


# ----------------------------------------------------------------------------
# Right-hand side (doc Eqs. 11-13 + Eq. 36 walls)
# ----------------------------------------------------------------------------

def electron_rhs(op: TransportOperator, ne: np.ndarray, n_eps: np.ndarray,
                 Er: np.ndarray, Ez: np.ndarray, S_ext_eV: np.ndarray,
                 p: ElectronParams, S_e: np.ndarray | None = None,
                 inelastic=None, evolve_ne: bool = False):
    """Tendencies (dne/dt, dneps/dt, Te).

    S_ext_eV : external heating in eV m^-3 s^-1 -- pass Q_ind / q with
               Q_ind from `inductive.power_deposition` (W/m^3). The
               electrostatic Joule term q mu_e ne |E_S|^2 / q is added
               internally from (Er, Ez).
    inelastic: None, an (Nr, Nz) array [eV m^-3 s^-1], or a callable
               inelastic(ne, Te) -> array (the sum r_j d_eps_j of Eq. 12).
    """
    Te = temperature(ne, n_eps, p)
    mu = mobility(p.nu_m) * np.ones_like(ne)
    vth = thermal_speed(Te)
    ce = WALL_ENERGY * (1.0 - p.re) / (1.0 + p.re)
    cp = WALL_PARTICLE * (1.0 - p.re) / (1.0 + p.re)

    # energy density (Eq. 13 fluxes + Eq. 36 wall)
    Deps = ENERGY_FLUX_FACTOR * mu * Te
    Fr, Fz = flux_drift_diffusion(op, n_eps, Deps, ENERGY_FLUX_FACTOR * mu,
                                  -1.0, Er, Ez)
    Wr, Wz = wall_flux_thermal(op, n_eps, vth, ce)
    Erc = 0.5 * (Er[:-1, :] + Er[1:, :])
    Ezc = 0.5 * (Ez[:, :-1] + Ez[:, 1:])
    S = (S_ext_eV + ne * mu * (Erc ** 2 + Ezc ** 2)
         - 3.0 * p.mass_ratio * np.asarray(p.nu_m) * ne * (Te - p.Tg_eV))
    if inelastic is not None:
        S = S - (inelastic(ne, Te) if callable(inelastic) else inelastic)
    dneps = np.where(op.active, -divergence(op, Fr + Wr, Fz + Wz) + S, 0.0)

    # continuity (Eq. 11)
    if evolve_ne:
        Fr, Fz = flux_drift_diffusion(op, ne, mu * Te, mu, -1.0, Er, Ez)
        Wr, Wz = wall_flux_thermal(op, ne, vth, cp)
        dne = -divergence(op, Fr + Wr, Fz + Wz)
        if S_e is not None:
            dne = dne + S_e
        dne = np.where(op.active, dne, 0.0)
    else:
        dne = np.zeros_like(ne)
    return dne, dneps, Te


def advance(op: TransportOperator, ne: np.ndarray, n_eps: np.ndarray,
            Er: np.ndarray, Ez: np.ndarray, S_ext_eV: np.ndarray,
            p: ElectronParams, dt: float, n_steps: int,
            S_e: np.ndarray | None = None, inelastic=None,
            evolve_ne: bool = False):
    """Explicit (forward-Euler) NumPy reference integrator. `dt` may be a
    scalar (time-accurate) or the (Nr, Nz) array from `energy_local_dt`
    (steady-state pseudo-time acceleration)."""
    for _ in range(n_steps):
        dne, dneps, _ = electron_rhs(op, ne, n_eps, Er, Ez, S_ext_eV, p,
                                     S_e=S_e, inelastic=inelastic,
                                     evolve_ne=evolve_ne)
        if evolve_ne:
            ne = np.maximum(ne + dt * dne, 0.0)
        n_eps = np.maximum(n_eps + dt * dneps, 0.0)
    return ne, n_eps, temperature(ne, n_eps, p)


def energy_stable_dt(op: TransportOperator, ne: np.ndarray,
                     n_eps: np.ndarray, Er: np.ndarray, Ez: np.ndarray,
                     p: ElectronParams, cfl: float = 0.4,
                     Te_ref: float | np.ndarray | None = None) -> float:
    """Explicit step bound for the energy equation; `Te_ref` (e.g. Te_max)
    gives a bound that stays valid as Te evolves."""
    Te = temperature(ne, n_eps, p) if Te_ref is None else Te_ref
    mu = mobility(p.nu_m) * np.ones_like(ne)
    return stable_dt(op, ENERGY_FLUX_FACTOR * mu * Te,
                     ENERGY_FLUX_FACTOR * mu, -1.0, Er, Ez, cfl=cfl)


def energy_local_dt(op: TransportOperator, ne: np.ndarray, n_eps: np.ndarray,
                    Er: np.ndarray, Ez: np.ndarray, p: ElectronParams,
                    cfl: float = 0.4,
                    Te_ref: float | np.ndarray | None = None,
                    dt_max: float | None = None) -> np.ndarray:
    """Per-cell local pseudo-time step for steady-state acceleration
    (see `transport.local_dt`); pass the resulting array as `dt` to
    `advance` or the JAX stepper. Steady state only, not time-accurate."""
    Te = temperature(ne, n_eps, p) if Te_ref is None else Te_ref
    mu = mobility(p.nu_m) * np.ones_like(ne)
    return local_dt(op, ENERGY_FLUX_FACTOR * mu * Te,
                    ENERGY_FLUX_FACTOR * mu, -1.0, Er, Ez, cfl=cfl,
                    dt_max=dt_max)


def wall_energy_power(op: TransportOperator, n_eps: np.ndarray,
                      Te: np.ndarray, p: ElectronParams) -> float:
    """Total electron energy loss to walls (W), doc Eq. 36 second relation.
    Wall fluxes are oriented outward, so the total is sum(A |F|) q."""
    ce = WALL_ENERGY * (1.0 - p.re) / (1.0 + p.re)
    Wr, Wz = wall_flux_thermal(op, n_eps, thermal_speed(Te), ce)
    return float((np.sum(op.area_r * np.abs(Wr))
                  + np.sum(op.area_z * np.abs(Wz))) * QE)


# ----------------------------------------------------------------------------
# JAX stepper (jitted, differentiable, vmap-able)
# ----------------------------------------------------------------------------

def make_jax_energy_stepper(coeffs: TransportCoeffsJAX, p: ElectronParams,
                            inelastic_fn=None, n_sub: int = 200):
    """Return a jitted stepper

        step(n_eps, ne, Er, Ez, S_ext_eV, dt) -> n_eps'

    advancing the energy-density equation by `n_sub` explicit substeps of
    size `dt` -- a scalar (time-accurate) or an (Nr, Nz) local pseudo-time
    array from `energy_local_dt` (steady-state acceleration) -- at fixed
    ne (the doc Sec. 12.3 sub-slicing pattern: the
    EETM integrates on its own timescale between exchanges). Formulas
    mirror `electron_rhs` exactly (same clamps, upwinding, and walls), so
    the NumPy path is the bitwise reference. Reverse-mode gradients
    propagate through all substeps (static-trip-count fori_loop -> scan);
    `vmap` over (ne, S_ext_eV) batches for data generation.

    inelastic_fn: optional jnp-compatible callable (ne, Te) -> eV m^-3 s^-1.
    """
    import jax
    import jax.numpy as jnp

    ce = WALL_ENERGY * (1.0 - p.re) / (1.0 + p.re)
    c53 = ENERGY_FLUX_FACTOR
    nu_m = jnp.asarray(p.nu_m)

    padW = lambda X: jnp.concatenate(
        [jnp.zeros((1, X.shape[1]), X.dtype), X], axis=0)
    padE = lambda X: jnp.concatenate(
        [X, jnp.zeros((1, X.shape[1]), X.dtype)], axis=0)
    padS = lambda X: jnp.concatenate(
        [jnp.zeros((X.shape[0], 1), X.dtype), X], axis=1)
    padN = lambda X: jnp.concatenate(
        [X, jnp.zeros((X.shape[0], 1), X.dtype)], axis=1)

    def step(n_eps, ne, Er, Ez, S_ext_eV, dt):
        c = coeffs
        mu = QE / (ME * nu_m) * jnp.ones_like(ne)
        ne_eff = jnp.maximum(ne, p.ne_floor)
        Erc = 0.5 * (Er[:-1, :] + Er[1:, :])
        Ezc = 0.5 * (Ez[:, :-1] + Ez[:, 1:])
        S0 = S_ext_eV + ne * mu * (Erc ** 2 + Ezc ** 2)
        el = 3.0 * p.mass_ratio * nu_m * ne
        vr = -c53 * 0.5 * (padW(mu) + padE(mu)) * Er   # electron sign = -1
        vz = -c53 * 0.5 * (padS(mu) + padN(mu)) * Ez
        upr, upz = vr > 0.0, vz > 0.0

        def body(_, P):
            Te = jnp.clip((2.0 / 3.0) * P / ne_eff, p.Te_min, p.Te_max)
            G = c53 * mu * Te * P
            f = ce * jnp.sqrt(8.0 * QE * Te / (jnp.pi * ME)) * P
            PW, PE = padW(P), padE(P)
            Fr = c.int_r * (-(padE(G) - padW(G)) / c.dc_r
                            + vr * jnp.where(upr, PW, PE)) \
                + c.wall_e * padW(f) - c.wall_w * padE(f)
            PS, PN = padS(P), padN(P)
            Fz = c.int_z * (-(padN(G) - padS(G)) / c.dc_z
                            + vz * jnp.where(upz, PS, PN)) \
                + c.wall_n * padS(f) - c.wall_s * padN(f)
            div = c.active * (
                (c.area_r[1:, :] * Fr[1:, :] - c.area_r[:-1, :] * Fr[:-1, :]
                 + c.area_z[:, 1:] * Fz[:, 1:]
                 - c.area_z[:, :-1] * Fz[:, :-1]) / c.volume)
            S = S0 - el * (Te - p.Tg_eV)
            if inelastic_fn is not None:
                S = S - inelastic_fn(ne, Te)
            return jnp.maximum(P + dt * (-div + S), 0.0)

        return jax.lax.fori_loop(0, n_sub, body, n_eps)

    return jax.jit(step)