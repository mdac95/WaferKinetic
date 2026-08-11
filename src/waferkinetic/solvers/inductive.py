"""
inductive.py
============
Frequency-domain azimuthal magnetic vector potential (A_phi) solver for
inductively coupled plasmas on the 2D axisymmetric (r, z) rectilinear mesh
produced by `reactor_mesh.Mesh2D`.

Physics
-------
For a non-magnetized ICP the high-frequency field is A = A_phi(r, z) phi_hat,
solved in the frequency domain (COMSOL mf / HPEM EMM equivalent):

    (j*w*sigma - w^2*eps0*epsr) * A
        - mu0^{-1} * [ (1/r) d/dr(r dA/dr) - A/r^2 + d2A/dz2 ]  =  J_phi^e

with the cold-plasma conductivity

    sigma = ne * q^2 / (me * (nu_e + j*w)).

Discretization
--------------
Finite volume over the exact toroidal cell volumes of `Mesh2D`:

- The curl-curl term integrates *exactly* to face terms
  2*pi*r_face*dz * (dA/dr) and 2*pi*r_c*dr * (dA/dz), so the scheme is
  conservative and the axis boundary condition is natural
  (r_face = 0 => zero flux; regularity A_phi(0) = 0 is implied).
- The +A/r^2 curvature term uses the midpoint rule, which is *exact* in the
  first cell for the physical near-axis behavior A ~ r.
- Metals (GROUNDED_METAL, WAFER by default) and the exterior boundary are
  magnetic insulation, A = 0 (Dirichlet at the interface/boundary face,
  half-cell distance), matching the COMSOL reference which excludes the
  wafer/pedestal from the mf domain.
- The coil is driven with an impressed uniform current density per turn
  ("coil group" / filament approximation: every turn carries the same total
  current I, copper eddy currents not resolved).

The resulting 5-point operator is **complex symmetric** (A^T = A): real,
positive face/curvature couplings plus a complex diagonal. This is exploited
in two ways: (a) a cheap structural unit test, and (b) reverse-mode
differentiability of the JAX solve via `lax.custom_linear_solve(...,
symmetric=True)` -- the transpose solve *is* the forward solve, so
d(solution)/d(sigma, J) gradients come for free.

Layout follows the reactor_mesh conventions: all setup is static NumPy; a
single `to_jax()` conversion produces a pytree of jnp arrays that the jitted
solver consumes. Shapes are (Nr, Nz) throughout, indexing='ij'.

Units: SI. A_phi in Wb/m (T*m), E_phi = -j*w*A in V/m, J in A/m^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # package layout: src/waferkinetic/fields/inductive.py
    from waferkinetic.mesh.reactor_mesh import Material, Mesh2D, RectRegion
except ImportError:  # flat layout (tests / notebooks)
    from reactor_mesh import Material, Mesh2D, RectRegion

# ------------------------------------------------------------------ constants
MU0 = 4.0e-7 * np.pi
EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
ME = 9.1093837015e-31

#: Materials treated as perfect conductors for the RF field (A = 0 inside).
DEFAULT_INACTIVE = (Material.GROUNDED_METAL, Material.WAFER)


# ----------------------------------------------------------------------------
# Cold-plasma conductivity
# ----------------------------------------------------------------------------

def cold_plasma_sigma(ne: np.ndarray, nu_e: np.ndarray | float,
                      omega: float) -> np.ndarray:
    """Complex cold-plasma conductivity sigma = ne q^2 / (me (nu_e + j w)).

    Works for NumPy and JAX arrays (pure arithmetic). `ne` in 1/m^3,
    `nu_e` (momentum-transfer collision frequency) in 1/s.
    """
    return ne * QE**2 / (ME * (nu_e + 1j * omega))


# ----------------------------------------------------------------------------
# Operator construction (static, NumPy)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class InductiveOperator:
    """Geometric part of the discrete A_phi operator on a fixed mesh + mask.

    The full system for a given plasma state is

        [diag_geo + k] * A - cW*A_W - cE*A_E - cS*A_S - cN*A_N = J * volume

    on active cells (identity rows A = 0 on inactive cells), where

        k = (j*w*sigma - w^2*eps0*epsr) * volume       (per-solve, complex)

    All coefficient arrays have shape (Nr, Nz); couplings toward the domain
    edge or an inactive neighbor are zero (their Dirichlet contribution is
    folded into `diag_geo`), so `np.roll`-style shifts are wrap-safe.
    The off-diagonal couplings are symmetric: cE[i, j] == cW[i+1, j] etc.
    """
    diag_geo: np.ndarray   # (Nr, Nz) float64, includes 1/mu0
    cW: np.ndarray         # coupling to (i-1, j)
    cE: np.ndarray         # coupling to (i+1, j)
    cS: np.ndarray         # coupling to (i, j-1)
    cN: np.ndarray         # coupling to (i, j+1)
    active: np.ndarray     # (Nr, Nz) bool
    volume: np.ndarray     # (Nr, Nz) exact toroidal cell volumes

    # ------------------------------------------------------------- solve-time
    def reaction(self, sigma: np.ndarray, omega: float,
                 eps_r: np.ndarray | float = 1.0) -> np.ndarray:
        """Complex diagonal contribution k = (j w sigma - w^2 eps0 epsr) V."""
        return (1j * omega * sigma - omega**2 * EPS0 * eps_r) * self.volume

    def matvec(self, k: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Apply the operator (NumPy reference implementation)."""
        ax = (self.diag_geo + k) * x
        ax = ax - self.cW * np.roll(x, 1, axis=0) \
                - self.cE * np.roll(x, -1, axis=0) \
                - self.cS * np.roll(x, 1, axis=1) \
                - self.cN * np.roll(x, -1, axis=1)
        return np.where(self.active, ax, x)

    # ------------------------------------------------------------ direct path
    def assemble_sparse(self, k: np.ndarray):
        """Assemble the scipy CSR matrix (reference / direct-solve path)."""
        import scipy.sparse as sp

        nr, nz = self.active.shape
        n = nr * nz
        idx = np.arange(n).reshape(nr, nz)
        diag = np.where(self.active, self.diag_geo + k, 1.0).ravel()

        rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag]
        for c, di, dj in ((self.cW, -1, 0), (self.cE, 1, 0),
                          (self.cS, 0, -1), (self.cN, 0, 1)):
            src = np.argwhere(c != 0.0)
            i, j = src[:, 0], src[:, 1]
            rows.append(idx[i, j])
            cols.append(idx[i + di, j + dj])
            vals.append(-c[i, j])
        m = sp.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n), dtype=np.complex128)
        return m.tocsr()

    def solve_direct(self, sigma: np.ndarray, omega: float, J: np.ndarray,
                     eps_r: np.ndarray | float = 1.0) -> np.ndarray:
        """Sparse direct solve (SuperLU). Returns A_phi, shape (Nr, Nz)."""
        from scipy.sparse.linalg import spsolve

        k = self.reaction(np.asarray(sigma, dtype=np.complex128), omega, eps_r)
        rhs = np.where(self.active, J * self.volume, 0.0).astype(np.complex128)
        m = self.assemble_sparse(k)
        return spsolve(m, rhs.ravel()).reshape(self.active.shape)


def build_operator(mesh: Mesh2D, mask: np.ndarray,
                   inactive: Sequence[Material] = DEFAULT_INACTIVE
                   ) -> InductiveOperator:
    """Build the geometric operator for a mesh + material mask.

    Assumes mu_r = 1 everywhere (true for ICP reactors: copper, quartz,
    plasma). Cells whose material is in `inactive` are removed from the
    solve and impose A = 0 on their interfaces.
    """
    nr, nz = mesh.Nr, mesh.Nz
    rc, zc = mesh.r_c, mesh.z_c
    dr, dz = mesh.dr, mesh.dz
    rf, zf = mesh.r_faces, mesh.z_faces

    active = ~np.isin(mask, np.array([int(m) for m in inactive]))

    two_pi = 2.0 * np.pi

    # ---- radial faces --------------------------------------------------- #
    # Interior conductance at face i (between cells i-1 and i), (Nr+1, Nz).
    g_r = np.zeros((nr + 1, nz))
    g_r[1:nr, :] = two_pi * rf[1:nr, None] * dz[None, :] \
        / (rc[1:] - rc[:-1])[:, None]
    # Dirichlet (half-cell) conductances seen from cell i at its own faces.
    g_r_dirW = two_pi * rf[:nr, None] * dz[None, :] \
        / (rc - rf[:nr])[:, None]            # rf[0]=0 -> row of zeros (axis)
    g_r_dirE = two_pi * rf[1:, None] * dz[None, :] \
        / (rf[1:] - rc)[:, None]

    nbW = np.zeros_like(active)
    nbW[1:, :] = active[:-1, :]              # west neighbor active?
    nbE = np.zeros_like(active)
    nbE[:-1, :] = active[1:, :]              # east neighbor active?

    cW = np.where(nbW, g_r[:nr, :], 0.0)
    cE = np.where(nbE, g_r[1:, :], 0.0)
    diag_r = np.where(nbW, g_r[:nr, :], g_r_dirW) \
        + np.where(nbE, g_r[1:, :], g_r_dirE)

    # ---- axial faces ---------------------------------------------------- #
    a_z = two_pi * rc * dr                   # exact annulus area, (Nr,)
    g_z = np.zeros((nr, nz + 1))
    g_z[:, 1:nz] = a_z[:, None] / (zc[1:] - zc[:-1])[None, :]
    g_z_dirS = a_z[:, None] / (zc - zf[:nz])[None, :]
    g_z_dirN = a_z[:, None] / (zf[1:] - zc)[None, :]

    nbS = np.zeros_like(active)
    nbS[:, 1:] = active[:, :-1]
    nbN = np.zeros_like(active)
    nbN[:, :-1] = active[:, 1:]

    cS = np.where(nbS, g_z[:, :nz], 0.0)
    cN = np.where(nbN, g_z[:, 1:], 0.0)
    diag_z = np.where(nbS, g_z[:, :nz], g_z_dirS) \
        + np.where(nbN, g_z[:, 1:], g_z_dirN)

    # ---- curvature (+A/r^2) term ---------------------------------------- #
    curv = two_pi * dr[:, None] * dz[None, :] / rc[:, None]

    inv_mu0 = 1.0 / MU0
    zero = np.zeros((nr, nz))
    return InductiveOperator(
        diag_geo=np.where(active, inv_mu0 * (diag_r + diag_z + curv), 0.0),
        cW=np.where(active, inv_mu0 * cW, zero),
        cE=np.where(active, inv_mu0 * cE, zero),
        cS=np.where(active, inv_mu0 * cS, zero),
        cN=np.where(active, inv_mu0 * cN, zero),
        active=active,
        volume=mesh.volume.copy(),
    )


# ----------------------------------------------------------------------------
# Coil excitation
# ----------------------------------------------------------------------------

def coil_current_density(mesh: Mesh2D,
                         turns: Sequence[RectRegion | tuple],
                         current: float = 1.0) -> np.ndarray:
    """Impressed J_phi (A/m^2) for a series coil: every turn carries `current`.

    Each turn's density is normalized by the *discrete* planar cross-section
    of the cells whose centers fall inside it, so the total current per turn
    is exactly `current` regardless of how the mesh cuts the rectangle.

    `turns` accepts RectRegion objects or (r0, r1, z0, z1) tuples.
    """
    J = np.zeros((mesh.Nr, mesh.Nz))
    planar = mesh.DR * mesh.DZ
    for t in turns:
        if isinstance(t, RectRegion):
            inside = t.contains(mesh.RC, mesh.ZC)
        else:
            r0, r1, z0, z1 = t
            inside = ((mesh.RC >= r0) & (mesh.RC < r1)
                      & (mesh.ZC >= z0) & (mesh.ZC < z1))
        s = planar[inside].sum()
        if s <= 0.0:
            raise ValueError("coil turn contains no cell centers; refine mesh")
        J[inside] += current / s
    return J


# ----------------------------------------------------------------------------
# Derived fields & diagnostics (NumPy or JAX arrays)
# ----------------------------------------------------------------------------

def electric_field(A: np.ndarray, omega: float) -> np.ndarray:
    """Azimuthal RF electric field phasor E_phi = -j w A (V/m)."""
    return -1j * omega * A


def power_deposition(A: np.ndarray, sigma: np.ndarray,
                     omega: float) -> np.ndarray:
    """Time-averaged Joule heating density Q = 1/2 Re(sigma) |E_phi|^2 (W/m^3).

    This is the electron heat source for the mean-energy equation.
    """
    e2 = (omega * np.abs(A)) ** 2
    return 0.5 * np.real(sigma) * e2


def absorbed_power(A: np.ndarray, sigma: np.ndarray, omega: float,
                   volume: np.ndarray) -> float:
    """Total time-averaged power absorbed by the plasma (W)."""
    return float(np.sum(power_deposition(A, sigma, omega) * volume))


def source_complex_power(A: np.ndarray, J: np.ndarray, omega: float,
                         volume: np.ndarray) -> complex:
    """Complex power delivered by the impressed coil current,
    S = -1/2 int E_phi J* dV = 1/2 j w sum(A J* V).

    Re(S) equals the total plasma-absorbed power (discrete energy identity,
    exact up to solver tolerance). Z_reflected = 2 S / |I|^2 gives the
    plasma-loaded coil impedance seen by the drive (filament coil, so this
    excludes the copper's own ohmic resistance).
    """
    return complex(0.5j * omega * np.sum(A * np.conj(J) * volume))


def b_field(A: np.ndarray, mesh: Mesh2D) -> tuple[np.ndarray, np.ndarray]:
    """(B_r, B_z) at cell centers from B = curl(A_phi phi_hat):
    B_r = -dA/dz, B_z = (1/r) d(rA)/dr. Nonuniform central differences;
    diagnostic-grade (the solve itself never uses these).
    """
    Br = -np.gradient(A, mesh.z_c, axis=1)
    Bz = np.gradient(mesh.r_c[:, None] * A, mesh.r_c, axis=0) \
        / mesh.r_c[:, None]
    return Br, Bz


def bz_on_axis(A: np.ndarray, mesh: Mesh2D) -> np.ndarray:
    """B_z(r=0, z) from the near-axis expansion A ~ (Bz/2) r  (2nd order)."""
    return 2.0 * A[0, :] / mesh.r_c[0]


# ----------------------------------------------------------------------------
# JAX path: pytree coefficients + differentiable jitted solve
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class InductiveCoeffsJAX:
    """Pytree of jnp arrays consumed by the jitted solver. Build once via
    `to_jax(op)`; treat as immutable."""
    diag_geo: object
    cW: object
    cE: object
    cS: object
    cN: object
    active: object   # bool
    volume: object


def to_jax(op: InductiveOperator) -> InductiveCoeffsJAX:
    """Single NumPy -> JAX conversion boundary (float64 required)."""
    import jax
    import jax.numpy as jnp

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "Enable float64 first: jax.config.update('jax_enable_x64', True)")
    return InductiveCoeffsJAX(*(jnp.asarray(a) for a in (
        op.diag_geo, op.cW, op.cE, op.cS, op.cN, op.active, op.volume)))


def _register_pytree() -> None:
    import jax

    jax.tree_util.register_pytree_node(
        InductiveCoeffsJAX,
        lambda c: ((c.diag_geo, c.cW, c.cE, c.cS, c.cN, c.active, c.volume),
                   None),
        lambda _, leaves: InductiveCoeffsJAX(*leaves),
    )


try:  # register at import time when jax is available
    import jax as _jax  # noqa: F401
    _register_pytree()
except ImportError:
    pass


def make_jax_solver(tol: float = 1e-9, maxiter: int = 2000):
    """Return a jitted, reverse-mode differentiable solver

        solve(coeffs, sigma, omega, J, eps_r=1.0) -> A_phi (complex, (Nr,Nz))

    Internals: Jacobi-preconditioned BiCGStab wrapped in
    `lax.custom_linear_solve(symmetric=True)`. The operator is complex
    *symmetric* (not Hermitian), so its transpose solve equals the forward
    solve -- gradients w.r.t. sigma (i.e. n_e) and J propagate through the
    linear solve implicitly, with no unrolling. `vmap` over sigma/J batches
    for data generation.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import bicgstab

    def solve(coeffs: InductiveCoeffsJAX, sigma, omega, J, eps_r=1.0):
        k = (1j * omega * sigma - omega**2 * EPS0 * eps_r) * coeffs.volume
        diag = jnp.where(coeffs.active, coeffs.diag_geo + k, 1.0)
        rhs = jnp.where(coeffs.active, J * coeffs.volume, 0.0) + 0.0j

        def mv(x):
            ax = diag * x \
                - coeffs.cW * jnp.roll(x, 1, axis=0) \
                - coeffs.cE * jnp.roll(x, -1, axis=0) \
                - coeffs.cS * jnp.roll(x, 1, axis=1) \
                - coeffs.cN * jnp.roll(x, -1, axis=1)
            return jnp.where(coeffs.active, ax, x)

        def inner_solve(matvec, b):
            x, _ = bicgstab(matvec, b, M=lambda y: y / diag,
                            tol=tol, atol=0.0, maxiter=maxiter)
            return x

        return jax.lax.custom_linear_solve(mv, rhs, solve=inner_solve,
                                           symmetric=True)

    return jax.jit(solve, static_argnames=())