"""
transport.py
============
Species transport flux closures and the conservative divergence for the 2D
axisymmetric (r, z) rectilinear mesh of `reactor_mesh.Mesh2D` -- the flux
layer of the fluid kinetics-Poisson module (FKPM).

Implements (model document, Secs. 5.1, 5.4, 12.1):

* Closure 1A -- drift-diffusion,

      Gamma = -grad(D N) + sign * mu * N * E_S            (doc Eq. 11/23)

  The default diffusive form is grad(D N), matching Eqs. (11)/(13); the
  D grad(N) form (Eq. 23 read literally, harmonic-mean face D) is available
  via `form="d_gradn"`. The drift term is donor-cell upwinded (doc Eq. 53).

* Closure 1B -- drift-diffusion with a static magnetic field B(r, z)
  (doc Eq. 24): tensor mobility/diffusivity with the perpendicular
  reduction D_perp = D_par / (1 + (omega_c / nu_c)^2). For B lying in the
  (r, z) plane the Hall component (b x .) is purely azimuthal and drops
  out of axisymmetric (r, z) transport; the remaining in-plane tensor

      X = X_perp I + (X_par - X_perp) b b^T

  produces the cross-derivative (9-point) coupling noted in doc Sec. 12.2.
  Not active for the unmagnetized GEC benchmark.

* Conservative finite-volume divergence with the exact axisymmetric
  metrics of `Mesh2D` (doc Eq. 52): interior face terms telescope
  exactly, so closed-domain particle conservation holds to machine
  precision (asserted in test_transport.py).

* Charged-particle wall boundary fluxes (doc Eqs. 36-37): thermal outflow
  with partial reflection (electrons/neutrals), thermal + gated outward
  drift (ions), and a Dirichlet-0 wall for verification problems.

Staggering (doc Sec. 12.1): densities and temperatures live at cell
centers, shape (Nr, Nz); fluxes and electrostatic-field components live
at faces, Er (Nr+1, Nz), Ez (Nr, Nz+1). `area_r[0, :] = 0` encodes the
symmetry axis: no radial transport through r = 0 regardless of the flux
value there, so the axis condition is natural (as in `inductive.py`).

Layout follows the package conventions: all setup is static NumPy;
`to_jax()` performs the single NumPy -> JAX conversion (boolean face
masks pre-converted to float multipliers) for jitted steppers.

Units: SI; densities 1/m^3, fluxes 1/(m^2 s), E in V/m, mu in m^2/(V s),
D in m^2/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # package layout: src/waferkinetic/solvers/transport.py
    from waferkinetic.mesh.reactor_mesh import Material, Mesh2D
except ImportError:  # flat layout (tests / notebooks)
    from reactor_mesh import Material, Mesh2D

# ------------------------------------------------------------------ constants
QE = 1.602176634e-19
ME = 9.1093837015e-31
KB = 1.380649e-23

#: Materials in which species are transported; everything else is a wall.
DEFAULT_TRANSPORTED = (Material.PLASMA,)


# ----------------------------------------------------------------------------
# Face-value helpers
# ----------------------------------------------------------------------------

def _face_vals_r(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(west, east) cell values at the Nr+1 radial faces, zero outside."""
    nr, nz = X.shape
    W = np.zeros((nr + 1, nz))
    E = np.zeros((nr + 1, nz))
    W[1:, :] = X
    E[:-1, :] = X
    return W, E


def _face_vals_z(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(south, north) cell values at the Nz+1 axial faces, zero outside."""
    nr, nz = X.shape
    S = np.zeros((nr, nz + 1))
    N = np.zeros((nr, nz + 1))
    S[:, 1:] = X
    N[:, :-1] = X
    return S, N


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = a + b
    return np.where(s > 0.0, 2.0 * a * b / np.where(s > 0.0, s, 1.0), 0.0)


# ----------------------------------------------------------------------------
# Operator (static, NumPy)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TransportOperator:
    """Geometric/topological data for FV transport on a fixed mesh + mask.

    Face classification (r-faces have shape (Nr+1, Nz), z-faces (Nr, Nz+1)):

    - `int_*`  : face between two transported (active) cells,
    - `wall_e` : face is the EAST wall of an active cell (outward normal +r),
    - `wall_w` : face is the WEST wall of an active cell (outward normal -r),
      excluding the symmetry axis (face 0, zero area anyway),
    - `wall_n` / `wall_s` : likewise in z (+z / -z outward).

    `dc_*` holds the center-to-center distance at interior faces and the
    center-to-face (half-cell) distance at wall faces, 1.0 elsewhere.
    """
    area_r: np.ndarray   # (Nr+1, Nz)
    area_z: np.ndarray   # (Nr, Nz+1)
    volume: np.ndarray   # (Nr, Nz)
    active: np.ndarray   # (Nr, Nz) bool
    int_r: np.ndarray
    int_z: np.ndarray
    wall_e: np.ndarray
    wall_w: np.ndarray
    wall_n: np.ndarray
    wall_s: np.ndarray
    dc_r: np.ndarray
    dc_z: np.ndarray
    r_c: np.ndarray      # (Nr,)
    z_c: np.ndarray      # (Nz,)


def build_transport(mesh: Mesh2D, mask: np.ndarray,
                    transported: Sequence[Material] = DEFAULT_TRANSPORTED
                    ) -> TransportOperator:
    """Build the transport operator for a mesh + material mask.

    Cells whose material is in `transported` are active; every face between
    an active cell and anything else (other materials or the domain edge)
    is a wall face carrying the boundary conditions of doc Eqs. 36-37.
    """
    nr, nz = mesh.Nr, mesh.Nz
    active = np.isin(mask, np.array([int(m) for m in transported]))

    # ---- radial faces ---------------------------------------------------- #
    aW = np.zeros((nr + 1, nz), dtype=bool)
    aE = np.zeros((nr + 1, nz), dtype=bool)
    aW[1:, :] = active
    aE[:-1, :] = active
    int_r = aW & aE
    wall_e = aW & ~aE
    wall_w = aE & ~aW
    wall_w[0, :] = False          # r = 0 is the symmetry axis, not a wall

    # ---- axial faces ----------------------------------------------------- #
    aS = np.zeros((nr, nz + 1), dtype=bool)
    aN = np.zeros((nr, nz + 1), dtype=bool)
    aS[:, 1:] = active
    aN[:, :-1] = active
    int_z = aS & aN
    wall_n = aS & ~aN
    wall_s = aN & ~aS

    # ---- distances ------------------------------------------------------- #
    dc_r = np.ones((nr + 1, nz))
    dc_r[1:nr, :] = (mesh.r_c[1:] - mesh.r_c[:-1])[:, None]
    d_e = np.broadcast_to((mesh.r_faces[1:] - mesh.r_c)[:, None], (nr, nz))
    d_w = np.broadcast_to((mesh.r_c - mesh.r_faces[:-1])[:, None], (nr, nz))
    dc_r[1:, :] = np.where(wall_e[1:, :], d_e, dc_r[1:, :])
    dc_r[:nr, :] = np.where(wall_w[:nr, :], d_w, dc_r[:nr, :])

    dc_z = np.ones((nr, nz + 1))
    dc_z[:, 1:nz] = (mesh.z_c[1:] - mesh.z_c[:-1])[None, :]
    d_n = np.broadcast_to((mesh.z_faces[1:] - mesh.z_c)[None, :], (nr, nz))
    d_s = np.broadcast_to((mesh.z_c - mesh.z_faces[:-1])[None, :], (nr, nz))
    dc_z[:, 1:] = np.where(wall_n[:, 1:], d_n, dc_z[:, 1:])
    dc_z[:, :nz] = np.where(wall_s[:, :nz], d_s, dc_z[:, :nz])

    return TransportOperator(
        area_r=mesh.area_r.copy(), area_z=mesh.area_z.copy(),
        volume=mesh.volume.copy(), active=active,
        int_r=int_r, int_z=int_z,
        wall_e=wall_e, wall_w=wall_w, wall_n=wall_n, wall_s=wall_s,
        dc_r=dc_r, dc_z=dc_z,
        r_c=mesh.r_c.copy(), z_c=mesh.z_c.copy(),
    )


# ----------------------------------------------------------------------------
# Conservative divergence (doc Eq. 52)
# ----------------------------------------------------------------------------

def divergence(op: TransportOperator, Fr: np.ndarray,
               Fz: np.ndarray) -> np.ndarray:
    """div(Gamma) at cell centers from face fluxes; zero on inactive cells.

    sum(volume * divergence) telescopes exactly to the boundary (wall)
    flux, so any flux field that vanishes on wall faces conserves the
    total inventory to machine precision.
    """
    d = (op.area_r[1:, :] * Fr[1:, :] - op.area_r[:-1, :] * Fr[:-1, :]
         + op.area_z[:, 1:] * Fz[:, 1:] - op.area_z[:, :-1] * Fz[:, :-1])
    return np.where(op.active, d / op.volume, 0.0)


# ----------------------------------------------------------------------------
# Closure 1A: drift-diffusion (doc Eqs. 11/23, upwinding Eq. 53)
# ----------------------------------------------------------------------------

def flux_drift_diffusion(op: TransportOperator, N: np.ndarray,
                         D: np.ndarray | float,
                         mu: np.ndarray | float | None = None,
                         sign: float = 1.0,
                         Er: np.ndarray | None = None,
                         Ez: np.ndarray | None = None,
                         form: str = "grad_dn"
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Interior-face drift-diffusion fluxes (wall faces are zero here;
    add one of the `wall_flux_*` fields for boundary conditions).

    form = "grad_dn":  F = -( (D N)_E - (D N)_W ) / dc   (doc Eqs. 11/13)
    form = "d_gradn":  F = -harm(D_W, D_E) (N_E - N_W) / dc   (doc Eq. 23)

    Drift: v = sign * mu_face * E_face, donor-cell upwinded density.
    `sign` is the charge sign (electrons: -1). Returns (Fr, Fz), positive
    in +r / +z.
    """
    D = np.broadcast_to(np.asarray(D, dtype=float), N.shape)
    NW, NE = _face_vals_r(N)
    NS, NN = _face_vals_z(N)

    if form == "grad_dn":
        GW, GE = _face_vals_r(D * N)
        GS, GN = _face_vals_z(D * N)
        Fr = -(GE - GW) / op.dc_r
        Fz = -(GN - GS) / op.dc_z
    elif form == "d_gradn":
        DW, DE = _face_vals_r(D)
        DS, DN = _face_vals_z(D)
        Fr = -_harmonic_mean(DW, DE) * (NE - NW) / op.dc_r
        Fz = -_harmonic_mean(DS, DN) * (NN - NS) / op.dc_z
    else:
        raise ValueError(f"unknown diffusion form {form!r}")

    if mu is not None and Er is not None:
        mu = np.broadcast_to(np.asarray(mu, dtype=float), N.shape)
        mW, mE = _face_vals_r(mu)
        v = sign * 0.5 * (mW + mE) * Er
        Fr = Fr + np.where(v > 0.0, NW, NE) * v
        mS, mN = _face_vals_z(mu)
        v = sign * 0.5 * (mS + mN) * Ez
        Fz = Fz + np.where(v > 0.0, NS, NN) * v

    return np.where(op.int_r, Fr, 0.0), np.where(op.int_z, Fz, 0.0)


# ----------------------------------------------------------------------------
# Closure 1B: static-B tensor drift-diffusion (doc Eq. 24)
# ----------------------------------------------------------------------------

def magnetized_tensors(X_par: np.ndarray, Br: np.ndarray, Bz: np.ndarray,
                       nu_c: np.ndarray, charge: float, mass: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-plane transport tensor for a scalar parallel coefficient X_par:

        X = X_perp I + (X_par - X_perp) b b^T,
        X_perp = X_par / (1 + (omega_c / nu_c)^2),  omega_c = |q| |B| / m.

    Returns (X_rr, X_rz, X_zz). The Hall component is azimuthal for B in
    the (r, z) plane and never enters axisymmetric (r, z) fluxes.
    B -> 0 recovers the isotropic scalar exactly.
    """
    B2 = Br * Br + Bz * Bz
    wc = np.abs(charge) * np.sqrt(B2) / mass
    ratio = 1.0 / (1.0 + (wc / nu_c) ** 2)
    Bs = np.sqrt(np.where(B2 > 0.0, B2, 1.0))
    br = np.where(B2 > 0.0, Br / Bs, 0.0)
    bz = np.where(B2 > 0.0, Bz / Bs, 0.0)
    X_perp = X_par * ratio
    dX = X_par - X_perp
    return X_perp + dX * br * br, dX * br * bz, X_perp + dX * bz * bz


def flux_drift_diffusion_magnetized(op: TransportOperator, N: np.ndarray,
                                    D_par, mu_par, sign: float,
                                    Er: np.ndarray, Ez: np.ndarray,
                                    Br, Bz, nu_c,
                                    charge: float = QE, mass: float = ME
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """Closure 1B: Gamma = -D_tensor . grad N + sign * (mu_tensor . E) N,
    with the normal drift component donor-cell upwinded.

    Cross-derivative terms use masked central cell-center gradients
    (zeroed within one cell of a wall, degrading locally to the diagonal
    stencil there). B = 0 reduces exactly to `flux_drift_diffusion`
    with form="d_gradn" and arithmetic == harmonic means (uniform D).
    """
    shape = N.shape
    b = lambda X: np.broadcast_to(np.asarray(X, dtype=float), shape)
    D_par, mu_par = b(D_par), b(mu_par)
    Br, Bz, nu_c = b(Br), b(Bz), b(nu_c)

    Drr, Drz, Dzz = magnetized_tensors(D_par, Br, Bz, nu_c, charge, mass)
    Mrr, Mrz, Mzz = magnetized_tensors(mu_par, Br, Bz, nu_c, charge, mass)

    # masked central tangential gradients at cell centers
    gr = np.zeros(shape)
    gr[1:-1, :] = (N[2:, :] - N[:-2, :]) / (op.r_c[2:] - op.r_c[:-2])[:, None]
    ok = np.zeros_like(op.active)
    ok[1:-1, :] = op.active[1:-1, :] & op.active[2:, :] & op.active[:-2, :]
    gr = np.where(ok, gr, 0.0)

    gz = np.zeros(shape)
    gz[:, 1:-1] = (N[:, 2:] - N[:, :-2]) / (op.z_c[2:] - op.z_c[:-2])[None, :]
    ok = np.zeros_like(op.active)
    ok[:, 1:-1] = op.active[:, 1:-1] & op.active[:, 2:] & op.active[:, :-2]
    gz = np.where(ok, gz, 0.0)

    # tangential E at cell centers
    Erc = 0.5 * (Er[:-1, :] + Er[1:, :])
    Ezc = 0.5 * (Ez[:, :-1] + Ez[:, 1:])

    avr = lambda X: 0.5 * (np.add(*_face_vals_r(X)))
    avz = lambda X: 0.5 * (np.add(*_face_vals_z(X)))

    # ---- radial faces ---------------------------------------------------- #
    NW, NE = _face_vals_r(N)
    Fr = -(avr(Drr) * (NE - NW) / op.dc_r + avr(Drz) * avr(gz))
    v = sign * (avr(Mrr) * Er + avr(Mrz) * avr(Ezc))
    Fr = np.where(op.int_r, Fr + np.where(v > 0.0, NW, NE) * v, 0.0)

    # ---- axial faces ----------------------------------------------------- #
    NS, NN = _face_vals_z(N)
    Fz = -(avz(Dzz) * (NN - NS) / op.dc_z + avz(Drz) * avz(gr))
    v = sign * (avz(Mzz) * Ez + avz(Mrz) * avz(Erc))
    Fz = np.where(op.int_z, Fz + np.where(v > 0.0, NS, NN) * v, 0.0)
    return Fr, Fz


# ----------------------------------------------------------------------------
# Wall boundary fluxes (doc Eqs. 36-37; Dirichlet-0 for verification)
# ----------------------------------------------------------------------------

def wall_flux_dirichlet0(op: TransportOperator, N: np.ndarray,
                         D: np.ndarray | float,
                         D_z: np.ndarray | float | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Purely diffusive wall flux for N = 0 at the wall face (half-cell
    Dirichlet), used for verification against analytic diffusion modes.
    `D_z` allows an anisotropic (e.g. magnetized-tensor diagonal) axial
    coefficient; defaults to `D`.
    """
    D = np.broadcast_to(np.asarray(D, dtype=float), N.shape)
    Dz = D if D_z is None else np.broadcast_to(np.asarray(D_z, float), N.shape)
    NW, NE = _face_vals_r(N)
    DW, DE = _face_vals_r(D)
    Fr = np.where(op.wall_e, DW * NW / op.dc_r, 0.0) \
        + np.where(op.wall_w, -DE * NE / op.dc_r, 0.0)
    NS, NN = _face_vals_z(N)
    DS, DN = _face_vals_z(Dz)
    Fz = np.where(op.wall_n, DS * NS / op.dc_z, 0.0) \
        + np.where(op.wall_s, -DN * NN / op.dc_z, 0.0)
    return Fr, Fz


def wall_flux_thermal(op: TransportOperator, N: np.ndarray,
                      vth: np.ndarray | float, coeff: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Outward thermal wall flux F_out = coeff * vth * N of the adjacent
    active cell (doc Eq. 36 with coeff = (1 - re)/(1 + re) * 1/2 for the
    electron particle flux and * 5/6 for the energy-density flux; doc
    Eq. 47 Chantry form for neutrals via coeff = gamma/(1 - gamma/2)/4).
    """
    f = coeff * np.broadcast_to(np.asarray(vth, dtype=float), N.shape) * N
    fW, fE = _face_vals_r(f)
    Fr = np.where(op.wall_e, fW, 0.0) - np.where(op.wall_w, fE, 0.0)
    fS, fN = _face_vals_z(f)
    Fz = np.where(op.wall_n, fS, 0.0) - np.where(op.wall_s, fN, 0.0)
    return Fr, Fz


def wall_flux_ion(op: TransportOperator, N: np.ndarray,
                  mu: np.ndarray | float, sign: float,
                  Er: np.ndarray, Ez: np.ndarray,
                  vth: np.ndarray | float = 0.0, gamma: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Ion wall flux (doc Eq. 37): one-directional thermal outflow plus
    drift outflow gated on the outward-directed field,

        n . Gamma = (1/4) gamma vth N + sign mu N (n . E) THETA[sign (n.E) > 0].
    """
    mu = np.broadcast_to(np.asarray(mu, dtype=float), N.shape)
    th = 0.25 * gamma * np.broadcast_to(np.asarray(vth, float), N.shape) * N
    NW, NE = _face_vals_r(N)
    thW, thE = _face_vals_r(th)
    mW, mE = _face_vals_r(mu)
    Fr = np.where(op.wall_e, thW + np.maximum(sign * mW * Er, 0.0) * NW, 0.0) \
        - np.where(op.wall_w, thE + np.maximum(-sign * mE * Er, 0.0) * NE, 0.0)
    NS, NN = _face_vals_z(N)
    thS, thN = _face_vals_z(th)
    mS, mN = _face_vals_z(mu)
    Fz = np.where(op.wall_n, thS + np.maximum(sign * mS * Ez, 0.0) * NS, 0.0) \
        - np.where(op.wall_s, thN + np.maximum(-sign * mN * Ez, 0.0) * NN, 0.0)
    return Fr, Fz


# ----------------------------------------------------------------------------
# Explicit stability estimate
# ----------------------------------------------------------------------------

def stability_rate(op: TransportOperator, D: np.ndarray | float,
                   mu: np.ndarray | float | None = None, sign: float = 1.0,
                   Er: np.ndarray | None = None, Ez: np.ndarray | None = None
                   ) -> np.ndarray:
    """Per-cell explicit stability rate lambda_i = sum_f (A_f / V_i)
    (D_f / d_f + |v_f|), wall faces bounded Dirichlet-like (cell coefficient
    over the half-cell distance). The global step bound is cfl / max(rate);
    cfl / rate gives a per-cell local step for steady-state acceleration
    (doc Sec. 12.4 -- pseudo-time only, the transient loses physical
    meaning).
    """
    D = np.broadcast_to(np.asarray(D, dtype=float), op.active.shape)
    DW, DE = _face_vals_r(D)
    rel_r = op.int_r | op.wall_e | op.wall_w
    coef_r = np.where(rel_r, np.maximum(DW, DE) / op.dc_r, 0.0)
    DS, DN = _face_vals_z(D)
    rel_z = op.int_z | op.wall_n | op.wall_s
    coef_z = np.where(rel_z, np.maximum(DS, DN) / op.dc_z, 0.0)

    if mu is not None and Er is not None:
        mu = np.broadcast_to(np.asarray(mu, dtype=float), op.active.shape)
        mW, mE = _face_vals_r(mu)
        coef_r = coef_r + np.where(rel_r, np.abs(0.5 * (mW + mE) * Er), 0.0)
        mS, mN = _face_vals_z(mu)
        coef_z = coef_z + np.where(rel_z, np.abs(0.5 * (mS + mN) * Ez), 0.0)

    rate = (op.area_r[1:, :] * coef_r[1:, :]
            + op.area_r[:-1, :] * coef_r[:-1, :]
            + op.area_z[:, 1:] * coef_z[:, 1:]
            + op.area_z[:, :-1] * coef_z[:, :-1]) / op.volume
    return np.where(op.active, rate, 0.0)


def stable_dt(op: TransportOperator, D: np.ndarray | float,
              mu: np.ndarray | float | None = None, sign: float = 1.0,
              Er: np.ndarray | None = None, Ez: np.ndarray | None = None,
              cfl: float = 0.4) -> float:
    """Global explicit time-step bound dt = cfl / max(stability_rate)."""
    rate = stability_rate(op, D, mu, sign, Er, Ez)
    return float(cfl / rate[op.active].max())


def local_dt(op: TransportOperator, D: np.ndarray | float,
             mu: np.ndarray | float | None = None, sign: float = 1.0,
             Er: np.ndarray | None = None, Ez: np.ndarray | None = None,
             cfl: float = 0.4, dt_max: float | None = None) -> np.ndarray:
    """Per-cell local pseudo-time step cfl / rate for steady-state
    acceleration: the explicit steppers accept a (Nr, Nz) dt array
    elementwise, converging to the same fixed point in far fewer
    iterations on stretched meshes. NOT time-accurate. `dt_max` caps the
    step (e.g. against stiff local sinks such as inelastic collisions).
    """
    rate = stability_rate(op, D, mu, sign, Er, Ez)
    dt = np.where(op.active, cfl / np.where(rate > 0.0, rate, 1.0), 0.0)
    if dt_max is not None:
        dt = np.minimum(dt, dt_max)
    return dt


# ----------------------------------------------------------------------------
# JAX path: pytree of float coefficient arrays
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TransportCoeffsJAX:
    """Pytree of jnp arrays for jitted steppers. Boolean cell/face masks
    are pre-converted to float multipliers (no boolean indexing inside
    jitted code). Build once via `to_jax(op)`; treat as immutable."""
    area_r: object
    area_z: object
    volume: object
    active: object    # float 0/1
    int_r: object     # float 0/1
    int_z: object
    wall_e: object
    wall_w: object
    wall_n: object
    wall_s: object
    dc_r: object
    dc_z: object


def to_jax(op: TransportOperator) -> TransportCoeffsJAX:
    """Single NumPy -> JAX conversion boundary (float64 required)."""
    import jax
    import jax.numpy as jnp

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "Enable float64 first: jax.config.update('jax_enable_x64', True)")
    f = lambda a: jnp.asarray(a.astype(np.float64) if a.dtype == np.bool_
                              else a)
    return TransportCoeffsJAX(
        f(op.area_r), f(op.area_z), f(op.volume), f(op.active),
        f(op.int_r), f(op.int_z),
        f(op.wall_e), f(op.wall_w), f(op.wall_n), f(op.wall_s),
        f(op.dc_r), f(op.dc_z))


def _register_pytree() -> None:
    import jax

    jax.tree_util.register_pytree_node(
        TransportCoeffsJAX,
        lambda c: ((c.area_r, c.area_z, c.volume, c.active,
                    c.int_r, c.int_z, c.wall_e, c.wall_w, c.wall_n, c.wall_s,
                    c.dc_r, c.dc_z), None),
        lambda _, leaves: TransportCoeffsJAX(*leaves),
    )


try:  # register at import time when jax is available
    import jax as _jax  # noqa: F401
    _register_pytree()
except ImportError:
    pass