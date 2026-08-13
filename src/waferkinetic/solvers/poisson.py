"""
poisson.py
==========
Electrostatic Poisson solver and the coupled charged-species continuity
update of the fluid kinetics-Poisson module (FKPM) -- closing the loop
around the flux layer of `transport.py` on the 2D axisymmetric (r, z)
rectilinear mesh of `reactor_mesh.Mesh2D`.

Physics (model document, Secs. 5.2-5.4, 12.1-12.2)
--------------------------------------------------
* P1 -- explicit Poisson (doc Eq. 28):

      -div( eps grad Phi(t) ) = sum_i q_i N_i(t) + rho_s(t),

  with rho_s the charge on dielectric surfaces, evolved from the net
  normal charged-species flux (doc Eq. 38, sigma_s = sum_j int q_j phi_j dt).
  Time steps are bounded by the dielectric relaxation time
  dt_d = eps0 / sigma_DC; kept as the verification reference.

* P2 -- semi-implicit Poisson (doc Eqs. 29-30, the default). The charge
  densities on the RHS are predicted with fluxes evaluated at the future
  potential; since q_i * sign_i = +e for every charged species, the drift
  terms *reform the Laplacian* (Eq. 30):

      sum_i q_i Gamma_i^drift = e (sum_i mu_i N_i) E = -sigma_DC grad Phi,

  so the P2 operator is the P1 stencil with the face conductance
  augmented by dt * sigma_DC -- literally the sigma -> eps substitution
  of the `inductive.py` stencil machinery, eps_eff = eps + dt * sigma_DC,
  and the augmentation factor *is* the dielectric-relaxation ratio
  dt / dt_d. Implemented in delta form,

      (G_eps + G_aug) Phi(t+dt) = q_i[N_i + dt(-div Gamma_i(E(t)) + S_i)] V
                                  + sigma_s-terms + G_aug Phi(t) + Dirichlet,

  so that at a stationary point (densities converged, Phi(t+dt) = Phi(t))
  the augmentation cancels exactly and Phi satisfies the *unaugmented*
  Poisson equation with the converged charge -- no fixed-point bias from
  the linearized (non-upwinded) drift prediction. Time steps exceed dt_d
  by factors of 1e2-1e3 (doc Sec. 5.3 headline claim, asserted in
  test_poisson.py test 4), bounded by the Courant limit.

* P3 -- fully implicit electron transport (Kushner 2009, J. Phys. D 42
  194013, Sec. 3.1 option 3 *disambiguated*; selectable, off by
  default). Option 3 as written (Eqs. 9a/9b) is ambiguous: reading
  phi_e as OLD electron density x NEW potential reproduces option 2
  exactly and removes nothing -- the P2 augmentation already makes the
  *drift* term implicit in Phi; what stays explicit in P2 is the
  electron *diffusion* term div grad(D_e n_e), which is precisely what
  sets the ~1 ps limit on 0.3 mm cells at 10 mTorr (D_e = mu_e Te ~
  1e4-1e5 m^2/s). P3 therefore makes n_e itself an unknown: backward
  Euler on the FULL drift-diffusion divergence,

      (1/dt + L_e[Phi]) n_e(t+dt) = n_e(t)/dt + S(t),

  with L_e the SAME discrete operator as the explicit path (grad(D N)
  diffusion, donor-cell drift, Eq. 36 thermal wall), so both schemes
  share one fixed point; the electron energy density gets the identical
  treatment (D_eps = 5/3 mu_e Te carries the same limit -- doing only
  continuity gains nothing). Solved as a Gummel block iteration per
  substep (transport solves <-> the unchanged P2 Poisson), with the P2
  electron augmentation supplying the linearized drift response of the
  already-solved n_e to the updated potential. The transport systems
  are diagonally dominant M-matrices but NOT symmetric (upwinding +
  grad(DN)): Jacobi-preconditioned BiCGStab in `custom_linear_solve`
  with an explicit `transpose_solve` (never symmetric=True). Ions stay
  explicit, so the remaining stability limit is ion sheath drift
  (mu_i E ~ 1e5-1e6 m/s -> ~0.1-1 ns on 0.3 mm cells) plus the
  retained sheath-RC and chemical-stiffness caps: a ~2-3 decade gain
  over the electron-diffusion limit, not five.

* Discretization: cell-centered finite volume on the exact axisymmetric
  metrics of `Mesh2D` (doc Sec. 12.1). Phi is solved in `solved`
  materials (default PLASMA + DIELECTRIC; the dielectric window carries
  its own permittivity, default eps_r = 4.2 from the caller's array).
  Grounded metal, the wafer, and the coil are Dirichlet Phi = 0 (a
  `phi_dirichlet` array supports driven electrodes, doc Sec. 5.4). The
  symmetry axis is natural (r_face = 0 => zero-area face), as in
  `inductive.py`. Interior face conductances use the distance-weighted
  harmonic ("series") permittivity, exact for a permittivity jump on a
  grid-aligned interface, so the normal-D continuity (field jump) at the
  window face is captured without special-casing.

* Surface charge (doc Eqs. 28/38): sigma_s (C/m^2) lives on the
  plasma-dielectric faces and is evolved from the net normal charged
  flux; its charge enters the Poisson RHS split half/half onto the two
  adjacent cells (sheet effectively *on* the face). The Eq. 38 backing-
  metal form is recovered automatically because the dielectric cells and
  the grounded metal behind them are part of the solve. Material
  conduction sigma_M is not implemented (quartz: sigma_M = 0).

* Rarefaction correction (doc Eq. 27): D' = min(v_th Lambda, D), with
  the mobility rescaled through the Einstein relation, mu' = q D'/(kB T)
  = D' / T[eV], so ambipolar fields remain consistent. Lambda is the
  reactor fundamental-mode diffusion length.

* Charged-species continuity (doc Eqs. 22-23, 36-37): electrons
  (sign -1) and one positive ion (sign +1) advanced with
  `transport.flux_drift_diffusion` (donor-cell drift) plus the Eq. 36
  electron wall flux (re = 0.2) and the Eq. 37 ion wall flux (thermal +
  outward-gated drift). The face electrostatic field feeds both the
  species drift and the ES Joule term of the electron energy equation
  (`electron_energy`, doc Eq. 12), replacing the Er = Ez = 0
  placeholders.

Layout follows the package conventions: all setup is static NumPy; a
single `to_jax()` conversion (boolean masks -> float multipliers)
produces the pytree a jitted stepper consumes. The P2 system is real
symmetric positive definite; the JAX path solves it with Jacobi-
preconditioned CG wrapped in `lax.custom_linear_solve(symmetric=True)`,
so reverse-mode gradients propagate implicitly through the Poisson
solve, as in `inductive.py`.

Units: SI. Phi in V, E in V/m, sigma_s in C/m^2, densities 1/m^3,
temperatures in eV where suffixed _eV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # package layout: src/waferkinetic/solvers/poisson.py
    from waferkinetic.mesh.reactor_mesh import Material, Mesh2D
    from waferkinetic.solvers.transport import (
        QE, ME, KB, TransportOperator, TransportCoeffsJAX,
        divergence, flux_drift_diffusion, wall_flux_thermal, wall_flux_ion,
        _face_vals_r, _face_vals_z)
except ImportError:  # flat layout (tests / notebooks)
    from reactor_mesh import Material, Mesh2D
    from transport import (QE, ME, KB, TransportOperator, TransportCoeffsJAX,
                           divergence, flux_drift_diffusion,
                           wall_flux_thermal, wall_flux_ion,
                           _face_vals_r, _face_vals_z)

EPS0 = 8.8541878128e-12
CHI01 = 2.404825557695773  # first zero of J0

#: Materials in which Phi is solved (plasma + dielectric window).
DEFAULT_SOLVED = (Material.PLASMA, Material.DIELECTRIC)
#: Materials imposing Dirichlet Phi (grounded unless `phi_dirichlet` set).
DEFAULT_DIRICHLET = (Material.GROUNDED_METAL, Material.WAFER,
                     Material.RF_ANTENNA)


# ----------------------------------------------------------------------------
# Rarefaction correction (doc Eq. 27) and small closures
# ----------------------------------------------------------------------------

def diffusion_length(R: float, L: float) -> float:
    """Fundamental-mode diffusion length of a closed cylinder,
    1/Lambda^2 = (chi01/R)^2 + (pi/L)^2 (doc Secs. 5.2, 13)."""
    return 1.0 / np.sqrt((CHI01 / R) ** 2 + (np.pi / L) ** 2)


def thermal_speed_heavy(T_eV: float, mass: float) -> float:
    """v_th = sqrt(8 kB T / (pi m)) for a heavy species, T in eV."""
    return float(np.sqrt(8.0 * QE * T_eV / (np.pi * mass)))


def flux_limited(D, T_eV, vth, Lambda: float):
    """Doc Eq. 27: D' = min(v_th Lambda, D); mu' = q D'/(kB T) = D'/T[eV]
    (Einstein relation preserved so ambipolar fields stay consistent)."""
    Dp = np.minimum(np.asarray(D, dtype=float), np.asarray(vth) * Lambda)
    return Dp, Dp / T_eV


def ion_mobility_1torr_scaled(p_torr: float, mu_1torr: float = 0.14) -> float:
    """Constant-mobility ion in its parent gas: mu(p) = mu_1torr / p_torr.
    Default mu_1torr = 0.14 m^2/(V s) at 1 Torr is the low-field Ar+ in
    Ar value (reduced mobility K0 ~ 1.5 cm^2 V^-1 s^-1 at STP; the same
    number expressed per Torr at 300 K). At 20 mTorr this gives
    ~7 m^2/(V s) -- NOT the 760x larger value a 1-atm reference would
    imply; that error makes ion losses outrun any ionization source and
    the discharge cannot ignite."""
    return mu_1torr / p_torr


# backwards-compatible alias (the old name misstated the reference
# pressure; the returned value with default arguments was 760x too big)
def ion_mobility_1atm_scaled(p_torr: float, mu_1torr: float = 0.14) -> float:
    return ion_mobility_1torr_scaled(p_torr, mu_1torr)


# ----------------------------------------------------------------------------
# Operator construction (static, NumPy)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PoissonOperator:
    """Geometric/dielectric data for -div(eps grad Phi) = rho on a fixed
    mesh + mask, in the `inductive.py` coefficient layout.

    Face arrays (r-faces (Nr+1, Nz), z-faces (Nr, Nz+1)):

    g_r, g_z       : eps-conductance eps_face * A / dc on *live* faces
                     (between two solved cells, or solved cell <-> Dirichlet
                     cell / domain edge at the half-cell distance); zero on
                     dead faces and on the axis (zero area).
    ga_r, ga_z     : geometric conductance A / dc of the *augmentable*
                     live faces -- those adjacent to at least one plasma
                     cell, where DC drift current flows. The P2 face
                     augmentation is dt * e * (sum mu_i N_i)_face * ga.
    dcp_r, dcp_z   : the face distances used in g (center-to-center
                     interior, half-cell at Dirichlet faces); 1 on dead
                     faces. E_face = -dPhi/dcp.
    wsol/wdir_r    : west neighbor of the face is a solved / Dirichlet
                     cell (float 0/1); esol/edir_r likewise east; and
                     ssol/sdir_z, nsol/ndir_z for z-faces. Domain-edge
                     Dirichlet ("dirichlet0") faces have both dir masks 0
                     (the boundary value is 0 by definition).
    wpl_r, epl_r,
    spl_z, npl_z   : plasma indicator of the west/east (south/north)
                     neighbor, for plasma-weighted face averages.
    scE_r, scW_r   : surface-charge (plasma|dielectric) faces with the
                     plasma cell WEST (outward normal +r) / EAST (-r);
                     scN_z, scS_z likewise for +z / -z normals.

    Cell arrays (Nr, Nz): `solved` (bool), `plasma` (bool), `volume`,
    and the mesh face areas for charging/deposit bookkeeping.
    """
    g_r: np.ndarray
    g_z: np.ndarray
    ga_r: np.ndarray
    ga_z: np.ndarray
    dcp_r: np.ndarray
    dcp_z: np.ndarray
    wsol_r: np.ndarray
    esol_r: np.ndarray
    wdir_r: np.ndarray
    edir_r: np.ndarray
    ssol_z: np.ndarray
    nsol_z: np.ndarray
    sdir_z: np.ndarray
    ndir_z: np.ndarray
    wpl_r: np.ndarray
    epl_r: np.ndarray
    spl_z: np.ndarray
    npl_z: np.ndarray
    scE_r: np.ndarray
    scW_r: np.ndarray
    scN_z: np.ndarray
    scS_z: np.ndarray
    solved: np.ndarray
    plasma: np.ndarray
    volume: np.ndarray
    area_r: np.ndarray
    area_z: np.ndarray

    # ------------------------------------------------------- assembly helpers
    def couplings(self, gaug_r: np.ndarray | None = None,
                  gaug_z: np.ndarray | None = None):
        """(diag, cW, cE, cS, cN) of the (optionally augmented) operator,
        `inductive.py` layout: diag collects every live face conductance;
        couplings only where the neighbor is a solved cell. Symmetric by
        construction (face-based conductances)."""
        gr = self.g_r if gaug_r is None else self.g_r + gaug_r
        gz = self.g_z if gaug_z is None else self.g_z + gaug_z
        nr, nz = self.solved.shape
        cW = self.wsol_r[:nr, :] * gr[:nr, :]
        cE = self.esol_r[1:, :] * gr[1:, :]
        cS = self.ssol_z[:, :nz] * gz[:, :nz]
        cN = self.nsol_z[:, 1:] * gz[:, 1:]
        diag = gr[:nr, :] + gr[1:, :] + gz[:, :nz] + gz[:, 1:]
        s = self.solved
        z = np.zeros_like(diag)
        return (np.where(s, diag, 1.0), np.where(s, cW, z),
                np.where(s, cE, z), np.where(s, cS, z), np.where(s, cN, z))

    def matvec(self, x: np.ndarray, gaug_r=None, gaug_z=None) -> np.ndarray:
        """Apply the operator (NumPy reference; identity on unsolved)."""
        diag, cW, cE, cS, cN = self.couplings(gaug_r, gaug_z)
        ax = diag * x \
            - cW * np.roll(x, 1, axis=0) - cE * np.roll(x, -1, axis=0) \
            - cS * np.roll(x, 1, axis=1) - cN * np.roll(x, -1, axis=1)
        return np.where(self.solved, ax, x)

    def assemble_sparse(self, gaug_r=None, gaug_z=None):
        """Scipy CSR matrix of the (augmented) operator (direct path)."""
        import scipy.sparse as sp

        diag, cW, cE, cS, cN = self.couplings(gaug_r, gaug_z)
        nr, nz = self.solved.shape
        n = nr * nz
        idx = np.arange(n).reshape(nr, nz)
        rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
        for c, di, dj in ((cW, -1, 0), (cE, 1, 0), (cS, 0, -1), (cN, 0, 1)):
            src = np.argwhere(c != 0.0)
            i, j = src[:, 0], src[:, 1]
            rows.append(idx[i, j])
            cols.append(idx[i + di, j + dj])
            vals.append(-c[i, j])
        m = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
        return m.tocsr()

    # ----------------------------------------------------------- RHS pieces
    def rhs_dirichlet(self, phi_dir: np.ndarray,
                      gaug_r=None, gaug_z=None) -> np.ndarray:
        """RHS contribution g_dir * phi of Dirichlet-cell neighbors (driven
        electrodes, doc Sec. 5.4). Domain-edge Dirichlet faces contribute 0.
        The Dirichlet conductance participates in the P2 augmentation
        (drift to a biased wall responds to the future potential)."""
        gr = self.g_r if gaug_r is None else self.g_r + gaug_r
        gz = self.g_z if gaug_z is None else self.g_z + gaug_z
        nr, nz = self.solved.shape
        pW, pE = _face_vals_r(phi_dir)
        pS, pN = _face_vals_z(phi_dir)
        b = (self.wdir_r[:nr, :] * gr[:nr, :] * pW[:nr, :]
             + self.edir_r[1:, :] * gr[1:, :] * pE[1:, :]
             + self.sdir_z[:, :nz] * gz[:, :nz] * pS[:, :nz]
             + self.ndir_z[:, 1:] * gz[:, 1:] * pN[:, 1:])
        return np.where(self.solved, b, 0.0)

    def rhs_surface_charge(self, ss_r: np.ndarray,
                           ss_z: np.ndarray) -> np.ndarray:
        """Charge sigma_s * A of the dielectric faces, split half/half
        onto the two adjacent cells (sheet on the face)."""
        qr = 0.5 * (self.scE_r + self.scW_r) * ss_r * self.area_r
        qz = 0.5 * (self.scN_z + self.scS_z) * ss_z * self.area_z
        return qr[:-1, :] + qr[1:, :] + qz[:, :-1] + qz[:, 1:]

    # -------------------------------------------------------------- solves
    def solve_direct(self, rho: np.ndarray,
                     ss_r: np.ndarray | None = None,
                     ss_z: np.ndarray | None = None,
                     phi_dir: np.ndarray | None = None,
                     gaug_r=None, gaug_z=None,
                     rhs_extra: np.ndarray | None = None) -> np.ndarray:
        """Sparse direct solve of the (augmented) system with volumetric
        charge rho (C/m^3, zero outside plasma cells is the caller's
        responsibility), face surface charge, Dirichlet values, and an
        optional extra RHS (the P2 delta term G_aug Phi_old)."""
        from scipy.sparse.linalg import spsolve

        b = np.where(self.solved, rho * self.volume, 0.0)
        if ss_r is not None:
            b = b + self.rhs_surface_charge(ss_r, ss_z)
        if phi_dir is not None:
            b = b + self.rhs_dirichlet(phi_dir, gaug_r, gaug_z)
        if rhs_extra is not None:
            b = b + np.where(self.solved, rhs_extra, 0.0)
        m = self.assemble_sparse(gaug_r, gaug_z)
        return spsolve(m, b.ravel()).reshape(self.solved.shape)

    # ------------------------------------------------------------- E-field
    def efield(self, Phi: np.ndarray, phi_dir: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
        """(Er, Ez) on faces, E = -dPhi/dcp across each live face.
        Dirichlet-cell neighbors contribute their phi_dir (default 0);
        domain-edge Dirichlet faces contribute 0. Dead faces carry E = 0."""
        pW, pE = _face_vals_r(Phi)
        pS, pN = _face_vals_z(Phi)
        if phi_dir is None:
            dW = dE = dS = dN = 0.0
        else:
            dWv, dEv = _face_vals_r(phi_dir)
            dSv, dNv = _face_vals_z(phi_dir)
            dW, dE = self.wdir_r * dWv, self.edir_r * dEv
            dS, dN = self.sdir_z * dSv, self.ndir_z * dNv
        live_r = (self.g_r > 0.0).astype(float)
        live_z = (self.g_z > 0.0).astype(float)
        Er = -(self.esol_r * pE + dE - self.wsol_r * pW - dW) / self.dcp_r
        Ez = -(self.nsol_z * pN + dN - self.ssol_z * pS - dS) / self.dcp_z
        return live_r * Er, live_z * Ez

    # ------------------------------------------------------------ charging
    def charging(self, flux_pairs: Sequence[tuple[float, np.ndarray,
                                                  np.ndarray]]
                 ) -> tuple[np.ndarray, np.ndarray]:
        """d(sigma_s)/dt on the dielectric faces from the net *outward*
        normal charge flux (doc Eqs. 28/38). `flux_pairs` is a sequence of
        (charge [C], Fr, Fz) with (Fr, Fz) the species face number-flux
        arrays (wall fluxes included), signed +r/+z as in `transport`."""
        dr = np.zeros_like(self.scE_r)
        dz = np.zeros_like(self.scN_z)
        for q, Fr, Fz in flux_pairs:
            dr = dr + q * (self.scE_r * Fr - self.scW_r * Fr)
            dz = dz + q * (self.scN_z * Fz - self.scS_z * Fz)
        return dr, dz

    # ---------------------------------------------------------- diagnostics
    def dielectric_relaxation_dt(self, M: np.ndarray) -> float:
        """min over plasma cells of eps0 / sigma_DC, sigma_DC = e M,
        M = sum_i mu_i N_i (the P1 step bound, doc Secs. 1/5.3)."""
        s = QE * np.asarray(M)[self.plasma]
        return float(EPS0 / s.max()) if s.size else np.inf


def _series_g(area, dW, dE, eW, eE):
    """Exact two-half-cell series conductance A / (dW/eW + dE/eE)."""
    return area / (dW / eW + dE / eE)


def build_poisson(mesh: Mesh2D, mask: np.ndarray,
                  eps_r: np.ndarray | float = 1.0,
                  solved: Sequence[Material] = DEFAULT_SOLVED,
                  dirichlet: Sequence[Material] = DEFAULT_DIRICHLET,
                  edge: str = "dirichlet0") -> PoissonOperator:
    """Build the electrostatic operator for a mesh + material mask.

    eps_r : relative permittivity per cell ((Nr, Nz) or scalar) -- e.g.
            the gec_case array with 4.2 in the window.
    edge  : 'dirichlet0' (grounded enclosure, default) or 'neumann'
            (zero normal D) at the domain boundary.
    """
    nr, nz = mesh.Nr, mesh.Nz
    eps = EPS0 * np.broadcast_to(np.asarray(eps_r, dtype=float),
                                 (nr, nz)).copy()
    sol = np.isin(mask, np.array([int(m) for m in solved]))
    dir_ = np.isin(mask, np.array([int(m) for m in dirichlet]))
    pla = mask == int(Material.PLASMA)
    die = mask == int(Material.DIELECTRIC)
    if edge not in ("dirichlet0", "neumann"):
        raise ValueError(f"unknown edge condition {edge!r}")
    edge_dir = edge == "dirichlet0"

    f = lambda b: b.astype(float)
    rc, zc = mesh.r_c, mesh.z_c
    rf, zf = mesh.r_faces, mesh.z_faces

    # ---- radial faces ---------------------------------------------------- #
    sW, sE = (np.zeros((nr + 1, nz), bool) for _ in range(2))
    dW, dE = (np.zeros((nr + 1, nz), bool) for _ in range(2))
    pW, pE = (np.zeros((nr + 1, nz), bool) for _ in range(2))
    sW[1:, :], sE[:-1, :] = sol, sol
    dW[1:, :], dE[:-1, :] = dir_, dir_
    pW[1:, :], pE[:-1, :] = pla, pla

    epsW, epsE = _face_vals_r(eps)
    hW = np.zeros((nr + 1, nz))            # center-to-face half distances
    hE = np.zeros((nr + 1, nz))
    hW[1:, :] = (rf[1:] - rc)[:, None]     # west cell center -> face
    hE[:-1, :] = (rc - rf[:-1])[:, None]   # face -> east cell center

    int_r = sW & sE
    dirE_r = sW & dE                       # solved west | Dirichlet cell east
    dirW_r = sE & dW
    # domain-edge Dirichlet: only the outer radial boundary (axis is dead)
    edgeE_r = np.zeros((nr + 1, nz), bool)
    if edge_dir:
        edgeE_r[nr, :] = sol[nr - 1, :]

    dcp_r = np.ones((nr + 1, nz))
    dcp_r = np.where(int_r, hW + hE, dcp_r)
    dcp_r = np.where(dirE_r | edgeE_r, hW, dcp_r)
    dcp_r = np.where(dirW_r, hE, dcp_r)

    area_r = mesh.area_r
    g_r = np.zeros((nr + 1, nz))
    with np.errstate(divide="ignore", invalid="ignore"):
        g_int = _series_g(area_r, np.where(hW > 0, hW, 1.0),
                          np.where(hE > 0, hE, 1.0), epsW + (epsW == 0),
                          epsE + (epsE == 0))
    g_r = np.where(int_r, g_int, g_r)
    g_r = np.where(dirE_r | edgeE_r,
                   area_r * epsW / np.where(hW > 0, hW, 1.0), g_r)
    g_r = np.where(dirW_r, area_r * epsE / np.where(hE > 0, hE, 1.0), g_r)

    ga_r = np.where((g_r > 0.0) & (pW | pE), area_r / dcp_r, 0.0)

    # surface-charge faces: plasma on one side, dielectric cell on the other
    dieW, dieE = (np.zeros((nr + 1, nz), bool) for _ in range(2))
    dieW[1:, :], dieE[:-1, :] = die, die
    scE_r = f(pW & dieE)                   # plasma west, dielectric east (+r)
    scW_r = f(pE & dieW)                   # plasma east, dielectric west (-r)

    # ---- axial faces ------------------------------------------------------ #
    sS, sN = (np.zeros((nr, nz + 1), bool) for _ in range(2))
    dS, dN = (np.zeros((nr, nz + 1), bool) for _ in range(2))
    pS, pN = (np.zeros((nr, nz + 1), bool) for _ in range(2))
    sS[:, 1:], sN[:, :-1] = sol, sol
    dS[:, 1:], dN[:, :-1] = dir_, dir_
    pS[:, 1:], pN[:, :-1] = pla, pla

    epsS, epsN = _face_vals_z(eps)
    hS = np.zeros((nr, nz + 1))
    hN = np.zeros((nr, nz + 1))
    hS[:, 1:] = (zf[1:] - zc)[None, :]
    hN[:, :-1] = (zc - zf[:-1])[None, :]

    int_z = sS & sN
    dirN_z = sS & dN
    dirS_z = sN & dS
    edgeN_z = np.zeros((nr, nz + 1), bool)
    edgeS_z = np.zeros((nr, nz + 1), bool)
    if edge_dir:
        edgeN_z[:, nz] = sol[:, nz - 1]
        edgeS_z[:, 0] = sol[:, 0]

    dcp_z = np.ones((nr, nz + 1))
    dcp_z = np.where(int_z, hS + hN, dcp_z)
    dcp_z = np.where(dirN_z | edgeN_z, hS, dcp_z)
    dcp_z = np.where(dirS_z | edgeS_z, hN, dcp_z)

    area_z = mesh.area_z
    g_z = np.zeros((nr, nz + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        g_int = _series_g(area_z, np.where(hS > 0, hS, 1.0),
                          np.where(hN > 0, hN, 1.0), epsS + (epsS == 0),
                          epsN + (epsN == 0))
    g_z = np.where(int_z, g_int, g_z)
    g_z = np.where(dirN_z | edgeN_z,
                   area_z * epsS / np.where(hS > 0, hS, 1.0), g_z)
    g_z = np.where(dirS_z | edgeS_z,
                   area_z * epsN / np.where(hN > 0, hN, 1.0), g_z)

    ga_z = np.where((g_z > 0.0) & (pS | pN), area_z / dcp_z, 0.0)

    dieS, dieN = (np.zeros((nr, nz + 1), bool) for _ in range(2))
    dieS[:, 1:], dieN[:, :-1] = die, die
    scN_z = f(pS & dieN)                   # plasma south, dielectric north
    scS_z = f(pN & dieS)

    return PoissonOperator(
        g_r=g_r, g_z=g_z, ga_r=ga_r, ga_z=ga_z, dcp_r=dcp_r, dcp_z=dcp_z,
        wsol_r=f(sW), esol_r=f(sE), wdir_r=f(dW & sE), edir_r=f(dE & sW),
        ssol_z=f(sS), nsol_z=f(sN), sdir_z=f(dS & sN), ndir_z=f(dN & sS),
        wpl_r=f(pW), epl_r=f(pE), spl_z=f(pS), npl_z=f(pN),
        scE_r=scE_r, scW_r=scW_r, scN_z=scN_z, scS_z=scS_z,
        solved=sol, plasma=pla, volume=mesh.volume.copy(),
        area_r=mesh.area_r.copy(), area_z=mesh.area_z.copy(),
    )


# ----------------------------------------------------------------------------
# Face augmentation for P2 (doc Eqs. 29-30): eps_eff = eps + dt * sigma_DC
# ----------------------------------------------------------------------------

def p2_augmentation(pop: PoissonOperator, M_e: np.ndarray, M_i: np.ndarray,
                    dt_e, dt_i) -> tuple[np.ndarray, np.ndarray]:
    """Augmentation conductances G_aug = e (dt_e M_e + dt_i M_i)_face * A/dc
    with M = mu N per species and plasma-weighted face averages. `dt_*`
    may be scalars or (Nr, Nz) local pseudo-time arrays; per-face values
    take the *max* of the adjacent cells (face-symmetric, and at least as
    implicit as the fastest-marching neighbor requires)."""
    shape = pop.solved.shape

    def face_avg_r(X):
        XW, XE = _face_vals_r(X * pop.plasma)
        w = pop.wpl_r + pop.epl_r
        return (XW + XE) / np.where(w > 0.0, w, 1.0)

    def face_avg_z(X):
        XS, XN = _face_vals_z(X * pop.plasma)
        w = pop.spl_z + pop.npl_z
        return (XS + XN) / np.where(w > 0.0, w, 1.0)

    def face_max_r(dt):
        dt = np.broadcast_to(np.asarray(dt, dtype=float), shape)
        W, E = _face_vals_r(dt)
        return np.maximum(W, E)

    def face_max_z(dt):
        dt = np.broadcast_to(np.asarray(dt, dtype=float), shape)
        S, N = _face_vals_z(dt)
        return np.maximum(S, N)

    sr = QE * (face_max_r(dt_e) * face_avg_r(M_e)
               + face_max_r(dt_i) * face_avg_r(M_i))
    sz = QE * (face_max_z(dt_e) * face_avg_z(M_e)
               + face_max_z(dt_i) * face_avg_z(M_i))
    return sr * pop.ga_r, sz * pop.ga_z


# ----------------------------------------------------------------------------
# FKPM parameters and the NumPy reference step
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class FKPMParams:
    """Charged-species closure parameters for the e + single-ion FKPM.

    nu_m       : electron momentum-transfer collision frequency (1/s)
                 -> mu_e = q/(me nu_m), D_e = mu_e Te (Einstein).
    mu_i, D_i  : ion mobility/diffusivity (doc Eq. 27-corrected values;
                 use `flux_limited` + `ion_mobility_1atm_scaled`).
    T_i_eV     : ion temperature (= gas temperature, doc Sec. 5.1).
    vth_i      : ion thermal speed for the Eq. 37 wall flux.
    re         : electron wall reflection coefficient (doc Eq. 36).
    ne_floor, Te_min, Te_max : Te = (2/3) n_eps/ne evaluation guards
                 (as in `electron_energy.ElectronParams`).
    mass_ratio, Tg_eV : elastic-exchange parameters of the energy eq.
    """
    nu_m: float | np.ndarray
    mu_i: float
    D_i: float
    T_i_eV: float
    vth_i: float
    re: float = 0.2
    ne_floor: float = 1.0e12
    Te_min: float = 1.0e-2
    Te_max: float = 30.0
    mass_ratio: float = 0.0
    Tg_eV: float = 0.026

    @property
    def mu_e(self):
        return QE / (ME * np.asarray(self.nu_m, dtype=float))


def _limit_outflow(top: TransportOperator, N, Fr, Fz, dt, safety=0.95):
    """Donor-side positivity-preserving flux limiter over ALL faces
    (interior + wall): scale every face flux by the limiter factor of
    its donor (upwind) cell so no cell can lose more than `safety` of
    its content in one step through its total outflow.  Conservative
    (one number per face) and inactive on any cell whose outflow is
    already resolved by dt -- in particular at a converged steady state.
    The limited fluxes must be used for BOTH the density update and the
    sigma_s charging so the discrete charge ledger stays exact.  Beyond
    positivity this is the nonlinear stabilizer of the P2 coupled
    update: the density advance uses explicit donor fluxes at the NEW
    field while any Courant clip necessarily saw the OLD one, and a
    growing field-charge oscillation can otherwise feed itself through
    Courant-violating drift during transients."""
    dtc = np.broadcast_to(np.asarray(dt, float), N.shape)
    outV = (top.area_r[1:, :] * np.maximum(Fr[1:, :], 0.0)
            + top.area_r[:-1, :] * np.maximum(-Fr[:-1, :], 0.0)
            + top.area_z[:, 1:] * np.maximum(Fz[:, 1:], 0.0)
            + top.area_z[:, :-1] * np.maximum(-Fz[:, :-1], 0.0))
    out = outV / top.volume
    hit = dtc * out > safety * N
    den = np.where(hit, dtc * out, 1.0)
    th = np.where(hit, safety * N / den, 1.0)
    thW, thE = _face_vals_r(th)
    thS, thN = _face_vals_z(th)
    fr = np.where(Fr > 0.0, thW, thE) * Fr
    fz = np.where(Fz > 0.0, thS, thN) * Fz
    return fr, fz


# retained name for any external callers
_limit_wall = _limit_outflow


def _species_fluxes(top: TransportOperator, pop: PoissonOperator,
                    ne, ni, Te, Er, Ez, p: FKPMParams,
                    dt_e=None, dt_i=None, ion_wall: str = "thermal",
                    wall_flux: str = "thermal", Phi=None,
                    Er_i=None, Ez_i=None):
    """Full (interior + wall) face fluxes for electrons and the ion at the
    given field: doc Eq. 23 drift-diffusion + Eq. 36/37 walls.  When a
    species dt is given, its wall fluxes are positivity-limited with
    `_limit_outflow` (required whenever the fluxes feed a time
    update)."""
    if ion_wall == "bohm":
        M_i = 8.0 * QE * p.T_i_eV / (np.pi * p.vth_i ** 2)
        vth_eff = np.maximum(p.vth_i,
                             4.0 * np.sqrt(QE * np.maximum(Te, p.Te_min)
                                           / M_i))
    else:
        vth_eff = p.vth_i * np.ones_like(ne)
    mu_e = p.mu_e * np.ones_like(ne)
    cp = 0.5 * (1.0 - p.re) / (1.0 + p.re)
    vth_e = np.sqrt(8.0 * QE * Te / (np.pi * ME))
    Fr_e, Fz_e = flux_drift_diffusion(top, ne, mu_e * Te, mu_e, -1.0, Er, Ez)
    Wr, Wz = wall_flux_thermal(top, ne, vth_e, cp)
    if wall_flux == "sheath" and Phi is not None:
        # Boltzmann throttle of the unresolved sheath, mirroring the
        # jitted `sheath_r`/`sheath_z`: WITHOUT this the dumped electron
        # wall flux is the raw thermal one while the run used the
        # throttled one, and diagnostics built on the dump report
        # spurious charge imbalance at floating surfaces (observed:
        # Ge/Gi = 6.8 at a plateaued dielectric; corrected ~1.05).
        PW, PE = _face_vals_r(Phi)
        TW, TE = _face_vals_r(np.maximum(Te, p.Te_min))
        TW = np.where(TW > 0, TW, 1.0)
        TE = np.where(TE > 0, TE, 1.0)
        br = 1.0 + np.where(top.wall_e, np.exp(-np.maximum(PW - PE, 0.0)
                                               / TW) - 1.0, 0.0)             + np.where(top.wall_w, np.exp(-np.maximum(PE - PW, 0.0)
                                          / TE) - 1.0, 0.0)
        PS, PN = _face_vals_z(Phi)
        TS, TN = _face_vals_z(np.maximum(Te, p.Te_min))
        TS = np.where(TS > 0, TS, 1.0)
        TN = np.where(TN > 0, TN, 1.0)
        bz = 1.0 + np.where(top.wall_n, np.exp(-np.maximum(PS - PN, 0.0)
                                               / TS) - 1.0, 0.0)             + np.where(top.wall_s, np.exp(-np.maximum(PN - PS, 0.0)
                                          / TN) - 1.0, 0.0)
        Wr, Wz = br * Wr, bz * Wz
    Fr_e, Fz_e = Fr_e + Wr, Fz_e + Wz
    if dt_e is not None:
        Fr_e, Fz_e = _limit_outflow(top, ne, Fr_e, Fz_e, dt_e)
    # ions may drift in the Richards effective field (ion_field=
    # "effective"): pass it so dumped fluxes match what the run used
    Eri = Er if Er_i is None else Er_i
    Ezi = Ez if Ez_i is None else Ez_i
    Fr_i, Fz_i = flux_drift_diffusion(top, ni, p.D_i, p.mu_i, +1.0,
                                      Eri, Ezi)
    Wr, Wz = wall_flux_ion(top, ni, p.mu_i, +1.0, Eri, Ezi, vth_eff, 1.0)
    Fr_i, Fz_i = Fr_i + Wr, Fz_i + Wz
    if dt_i is not None:
        Fr_i, Fz_i = _limit_outflow(top, ni, Fr_i, Fz_i, dt_i)
    return (Fr_e, Fz_e), (Fr_i, Fz_i)


def _implicit_electron_solve(top: TransportOperator, ne, Te, Er, Ez, S,
                             p: FKPMParams, dt, tol: float = 1.0e-10,
                             maxiter: int = 2000):
    """NumPy reference backward-Euler electron-continuity solve of the
    P3 scheme (module docstring): (1/dt + L_e[E]) n = ne/dt + S with
    L_e the SAME discrete operator as the explicit path
    (`flux_drift_diffusion` grad(DN) form + donor-cell drift +
    Eq. 36 thermal wall), coefficients frozen at the given (Te, E) so
    the system is linear. Matrix-free scipy BiCGStab (the operator is
    a diagonally dominant M-matrix but not symmetric).

    Returns (n_new, (Fr, Fz)) with the operator fluxes evaluated AT THE
    SOLUTION -- the only fluxes that keep the sigma_s charge ledger
    exact against the solved density (the density change equals
    dt*(-div F(n_new) + S) to solver tolerance, so charging the walls
    with any other flux leaks charge). No positivity limiter: backward
    Euler with donor-cell upwinding is unconditionally positivity-
    preserving for a nonnegative RHS, and limiting the implicit fluxes
    would break the ledger against the solved densities.
    """
    from scipy.sparse.linalg import LinearOperator, bicgstab

    shape = ne.shape
    mu_e = p.mu_e * np.ones(shape)
    D = mu_e * Te
    vth = np.sqrt(8.0 * QE * Te / (np.pi * ME))
    cw = 0.5 * (1.0 - p.re) / (1.0 + p.re)
    dtc = np.broadcast_to(np.asarray(dt, float), shape)
    inv_dt = 1.0 / np.where(dtc > 0.0, dtc, 1.0)

    def op_flux(x):
        Fr, Fz = flux_drift_diffusion(top, x, D, mu_e, -1.0, Er, Ez)
        Wr, Wz = wall_flux_thermal(top, x, vth, cw)
        return Fr + Wr, Fz + Wz

    def mv(xf):
        x = xf.reshape(shape)
        Fr, Fz = op_flux(x)
        return np.where(top.active, inv_dt * x + divergence(top, Fr, Fz),
                        x).ravel()

    # exact diagonal for the Jacobi preconditioner: the cell's own
    # coefficient in its outgoing flux through each face (interior D/dc
    # + outgoing upwind |v|; thermal coefficient on wall faces)
    mW, mE = _face_vals_r(mu_e)
    vr = -0.5 * (mW + mE) * Er
    mS, mN = _face_vals_z(mu_e)
    vz = -0.5 * (mS + mN) * Ez
    cE = np.where(top.int_r[1:, :], D / top.dc_r[1:, :]
                  + np.maximum(vr[1:, :], 0.0), 0.0) \
        + np.where(top.wall_e[1:, :], cw * vth, 0.0)
    cWc = np.where(top.int_r[:-1, :], D / top.dc_r[:-1, :]
                   + np.maximum(-vr[:-1, :], 0.0), 0.0) \
        + np.where(top.wall_w[:-1, :], cw * vth, 0.0)
    cN = np.where(top.int_z[:, 1:], D / top.dc_z[:, 1:]
                  + np.maximum(vz[:, 1:], 0.0), 0.0) \
        + np.where(top.wall_n[:, 1:], cw * vth, 0.0)
    cS = np.where(top.int_z[:, :-1], D / top.dc_z[:, :-1]
                  + np.maximum(-vz[:, :-1], 0.0), 0.0) \
        + np.where(top.wall_s[:, :-1], cw * vth, 0.0)
    diag = np.where(top.active, inv_dt
                    + (top.area_r[1:, :] * cE + top.area_r[:-1, :] * cWc
                       + top.area_z[:, 1:] * cN
                       + top.area_z[:, :-1] * cS) / top.volume,
                    1.0).ravel()

    b = np.where(top.active, inv_dt * ne + S, 0.0).ravel()
    A = LinearOperator((ne.size, ne.size), matvec=mv, dtype=float)
    M = LinearOperator((ne.size, ne.size), matvec=lambda y: y / diag,
                       dtype=float)
    try:
        x, _ = bicgstab(A, b, x0=ne.ravel(), M=M, rtol=tol, atol=0.0,
                        maxiter=maxiter)
    except TypeError:                          # scipy < 1.12 keyword
        x, _ = bicgstab(A, b, x0=ne.ravel(), M=M, tol=tol, atol=0.0,
                        maxiter=maxiter)
    n_new = x.reshape(shape)
    return n_new, op_flux(n_new)


def fkpm_step(top: TransportOperator, pop: PoissonOperator,
              ne, ni, ss_r, ss_z, Phi, Te, S, p: FKPMParams,
              dt_e, dt_i, scheme: str = "p2",
              phi_dir: np.ndarray | None = None,
              n_gummel: int = 2, implicit_tol: float = 1.0e-10):
    """One P1/P2 step of the coupled (ne, ni, sigma_s, Phi) system --
    NumPy reference for the jitted stepper (direct sparse Poisson solve).

    P2 (doc Eq. 29, delta form): the RHS carries the *full* explicit
    charge prediction (upwinded fluxes at E(t)) plus G_aug Phi(t); the
    augmented operator supplies the implicit drift response. The density
    update then uses fluxes at the new field E(t+dt). P1: same update
    with the unaugmented potential of the current charge (doc Eq. 28);
    dt must resolve eps0/sigma_DC.

    P2I (module docstring P3, Kushner option 3 disambiguated): the
    electron density is an unknown of a backward-Euler transport solve
    Gummel-iterated `n_gummel` times against the unchanged P2 Poisson
    solve; each sweep is [Poisson at the latest solved n_e] ->
    [implicit electron solve at the new field], ending on the transport
    solve so the returned density is the exact backward-Euler solution
    at the returned field and sigma_s is charged from the same operator
    fluxes (exact ledger). Ions stay explicit (limiter retained); the
    electron positivity limiter is off (M-matrix). The electron dt is
    no longer diffusion-bounded; ion Courant still binds dt_i.

    Te is held fixed here (doc Sec. 12.3 sub-slicing; the coupled jitted
    stepper also advances n_eps). Returns (ne', ni', ss_r', ss_z', Phi',
    Er, Ez).
    """
    rho = np.where(pop.plasma, QE * (ni - ne), 0.0)
    if scheme == "p2i":
        mu_e = p.mu_e * np.ones_like(ne)
        vth_e = np.sqrt(8.0 * QE * Te / (np.pi * ME))
        cw = 0.5 * (1.0 - p.re) / (1.0 + p.re)
        dtw_e = np.broadcast_to(np.asarray(dt_e, float), ne.shape)
        dtw_i = np.broadcast_to(np.asarray(dt_i, float), ne.shape)
        dtf_er = np.maximum(*_face_vals_r(dtw_e))
        dtf_ez = np.maximum(*_face_vals_z(dtw_e))
        dtf_ir = np.maximum(*_face_vals_r(dtw_i))
        dtf_iz = np.maximum(*_face_vals_z(dtw_i))

        ne_g, Phi_g = ne, Phi
        Fr_e = Fz_e = Er = Ez = None
        for _ in range(max(int(n_gummel), 1)):
            # (a) P2 Poisson at the latest solved electron density; the
            # electron drift response to the future potential is carried
            # by the (unchanged) augmentation, cancelled at a stationary
            # point by the delta form.
            Er0, Ez0 = pop.efield(Phi_g, phi_dir)
            _, (Fr_i, Fz_i) = _species_fluxes(top, pop, ne_g, ni, Te,
                                              Er0, Ez0, p,
                                              dt_e=None, dt_i=dt_i)
            ni_p = ni + dt_i * (-divergence(top, Fr_i, Fz_i) + S)
            Fr_e0, Fz_e0 = flux_drift_diffusion(top, ne_g, mu_e * Te,
                                                mu_e, -1.0, Er0, Ez0)
            Wr, Wz = wall_flux_thermal(top, ne_g, vth_e, cw)
            dsr_e, dsz_e = pop.charging([(-QE, Fr_e0 + Wr, Fz_e0 + Wz)])
            dsr_i, dsz_i = pop.charging([(QE, Fr_i, Fz_i)])
            ssr_p = ss_r + dtf_er * dsr_e + dtf_ir * dsr_i
            ssz_p = ss_z + dtf_ez * dsz_e + dtf_iz * dsz_i
            gaug_r, gaug_z = p2_augmentation(pop, mu_e * ne_g,
                                             p.mu_i * ni, dt_e, dt_i)
            aug_only = pop.matvec(Phi_g, gaug_r, gaug_z) \
                - pop.matvec(Phi_g)
            rho_p = np.where(pop.plasma, QE * (ni_p - ne_g), 0.0)
            Phi_g = pop.solve_direct(rho_p, ssr_p, ssz_p, phi_dir,
                                     gaug_r, gaug_z, rhs_extra=aug_only)
            # (b) backward-Euler electron continuity at the NEW field.
            Er, Ez = pop.efield(Phi_g, phi_dir)
            ne_g, (Fr_e, Fz_e) = _implicit_electron_solve(
                top, ne, Te, Er, Ez, S, p, dt_e, tol=implicit_tol)

        # final explicit ion update at the final field (limiter on);
        # sigma_s charged from each species' ACTUAL update fluxes.
        _, (Fr_i, Fz_i) = _species_fluxes(top, pop, ne_g, ni, Te, Er, Ez,
                                          p, dt_e=None, dt_i=dt_i)
        ne_new = np.maximum(ne_g, 0.0)
        ni_new = np.maximum(ni + dt_i * (-divergence(top, Fr_i, Fz_i)
                                         + S), 0.0)
        dsr_e, dsz_e = pop.charging([(-QE, Fr_e, Fz_e)])
        dsr_i, dsz_i = pop.charging([(QE, Fr_i, Fz_i)])
        ss_r_new = ss_r + dtf_er * dsr_e + dtf_ir * dsr_i
        ss_z_new = ss_z + dtf_ez * dsz_e + dtf_iz * dsz_i
        return ne_new, ni_new, ss_r_new, ss_z_new, Phi_g, Er, Ez
    if scheme == "p1":
        Phi_new = pop.solve_direct(rho, ss_r, ss_z, phi_dir)
    elif scheme == "p2":
        Er0, Ez0 = pop.efield(Phi, phi_dir)
        (Fr_e, Fz_e), (Fr_i, Fz_i) = _species_fluxes(
            top, pop, ne, ni, Te, Er0, Ez0, p, dt_e, dt_i)
        ne_p = ne + dt_e * (-divergence(top, Fr_e, Fz_e) + S)
        ni_p = ni + dt_i * (-divergence(top, Fr_i, Fz_i) + S)
        rho_p = np.where(pop.plasma, QE * (ni_p - ne_p), 0.0)
        dsr_e, dsz_e = pop.charging([(-QE, Fr_e, Fz_e)])
        dsr_i, dsz_i = pop.charging([(QE, Fr_i, Fz_i)])
        dtw_e = np.broadcast_to(np.asarray(dt_e, float), ne.shape)
        dtw_i = np.broadcast_to(np.asarray(dt_i, float), ne.shape)
        ssr_p = ss_r + np.maximum(*_face_vals_r(dtw_e)) * dsr_e \
            + np.maximum(*_face_vals_r(dtw_i)) * dsr_i
        ssz_p = ss_z + np.maximum(*_face_vals_z(dtw_e)) * dsz_e \
            + np.maximum(*_face_vals_z(dtw_i)) * dsz_i
        gaug_r, gaug_z = p2_augmentation(
            pop, p.mu_e * np.ones_like(ne) * ne, p.mu_i * ni, dt_e, dt_i)
        # delta term G_aug Phi(t): apply the augmentation-only operator
        aug_only = pop.matvec(Phi, gaug_r, gaug_z) - pop.matvec(Phi)
        Phi_new = pop.solve_direct(rho_p, ssr_p, ssz_p, phi_dir,
                                   gaug_r, gaug_z, rhs_extra=aug_only)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    Er, Ez = pop.efield(Phi_new, phi_dir)
    (Fr_e, Fz_e), (Fr_i, Fz_i) = _species_fluxes(
        top, pop, ne, ni, Te, Er, Ez, p, dt_e, dt_i)
    ne_new = np.maximum(ne + dt_e * (-divergence(top, Fr_e, Fz_e) + S), 0.0)
    ni_new = np.maximum(ni + dt_i * (-divergence(top, Fr_i, Fz_i) + S), 0.0)
    dsr_e, dsz_e = pop.charging([(-QE, Fr_e, Fz_e)])
    dsr_i, dsz_i = pop.charging([(QE, Fr_i, Fz_i)])
    dtw_e = np.broadcast_to(np.asarray(dt_e, float), ne.shape)
    dtw_i = np.broadcast_to(np.asarray(dt_i, float), ne.shape)
    ss_r_new = ss_r + np.maximum(*_face_vals_r(dtw_e)) * dsr_e \
        + np.maximum(*_face_vals_r(dtw_i)) * dsr_i
    ss_z_new = ss_z + np.maximum(*_face_vals_z(dtw_e)) * dsz_e \
        + np.maximum(*_face_vals_z(dtw_i)) * dsz_i
    return ne_new, ni_new, ss_r_new, ss_z_new, Phi_new, Er, Ez


# ----------------------------------------------------------------------------
# JAX path: pytree + jitted coupled stepper (differentiable Poisson solve)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PoissonCoeffsJAX:
    """Pytree of jnp arrays for the jitted FKPM stepper (float masks)."""
    g_r: object
    g_z: object
    ga_r: object
    ga_z: object
    dcp_r: object
    dcp_z: object
    wsol_r: object
    esol_r: object
    ssol_z: object
    nsol_z: object
    wpl_r: object
    epl_r: object
    spl_z: object
    npl_z: object
    scE_r: object
    scW_r: object
    scN_z: object
    scS_z: object
    solved: object
    plasma: object
    volume: object
    area_r: object
    area_z: object
    live_r: object
    live_z: object


_POISSON_LEAVES = ("g_r", "g_z", "ga_r", "ga_z", "dcp_r", "dcp_z",
                   "wsol_r", "esol_r", "ssol_z", "nsol_z",
                   "wpl_r", "epl_r", "spl_z", "npl_z",
                   "scE_r", "scW_r", "scN_z", "scS_z",
                   "solved", "plasma", "volume", "area_r", "area_z",
                   "live_r", "live_z")


def to_jax(pop: PoissonOperator) -> PoissonCoeffsJAX:
    """Single NumPy -> JAX conversion boundary (float64 required).
    Note: driven electrodes (phi_dir != 0) are not carried on the JAX
    path yet; grounded Dirichlet surfaces contribute 0 to both the RHS
    and the face field, so the dir masks are not needed there."""
    import jax
    import jax.numpy as jnp

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "Enable float64 first: jax.config.update('jax_enable_x64', True)")
    f = lambda a: jnp.asarray(np.asarray(a, dtype=np.float64))
    return PoissonCoeffsJAX(
        f(pop.g_r), f(pop.g_z), f(pop.ga_r), f(pop.ga_z),
        f(pop.dcp_r), f(pop.dcp_z),
        f(pop.wsol_r), f(pop.esol_r), f(pop.ssol_z), f(pop.nsol_z),
        f(pop.wpl_r), f(pop.epl_r), f(pop.spl_z), f(pop.npl_z),
        f(pop.scE_r), f(pop.scW_r), f(pop.scN_z), f(pop.scS_z),
        f(pop.solved), f(pop.plasma), f(pop.volume),
        f(pop.area_r), f(pop.area_z),
        f(pop.g_r > 0.0), f(pop.g_z > 0.0))


def _register_pytree() -> None:
    import jax

    jax.tree_util.register_pytree_node(
        PoissonCoeffsJAX,
        lambda c: (tuple(getattr(c, n) for n in _POISSON_LEAVES), None),
        lambda _, leaves: PoissonCoeffsJAX(*leaves),
    )


try:  # register at import time when jax is available
    import jax as _jax  # noqa: F401
    _register_pytree()
except ImportError:
    pass


def make_jax_poisson_solver(tol: float = 1e-9, maxiter: int = 4000):
    """Return a jitted, reverse-mode differentiable solve

        solve(pc, rhs, gaug_r, gaug_z, x0) -> Phi

    of the (augmented) SPD system via Jacobi-preconditioned CG wrapped in
    `lax.custom_linear_solve(symmetric=True)` -- the same implicit-
    gradient pattern as `inductive.make_jax_solver`, with warm starting
    from x0 (the previous potential)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    def solve(pc: PoissonCoeffsJAX, rhs, gaug_r, gaug_z, x0):
        gr = pc.g_r + gaug_r
        gz = pc.g_z + gaug_z
        nr = pc.solved.shape[0]
        nz = pc.solved.shape[1]
        cW = pc.wsol_r[:nr, :] * gr[:nr, :]
        cE = pc.esol_r[1:, :] * gr[1:, :]
        cS = pc.ssol_z[:, :nz] * gz[:, :nz]
        cN = pc.nsol_z[:, 1:] * gz[:, 1:]
        diag = jnp.where(pc.solved > 0.5,
                         gr[:nr, :] + gr[1:, :] + gz[:, :nz] + gz[:, 1:],
                         1.0)
        b = jnp.where(pc.solved > 0.5, rhs, 0.0)

        def mv(x):
            ax = diag * x \
                - cW * jnp.roll(x, 1, axis=0) - cE * jnp.roll(x, -1, axis=0) \
                - cS * jnp.roll(x, 1, axis=1) - cN * jnp.roll(x, -1, axis=1)
            return jnp.where(pc.solved > 0.5, ax, x)

        def inner(matvec, bb):
            # Warm-start breakdown guard, same disease as `imsolve`: at
            # a converged outer state x0 already solves the system, the
            # initial residual sits at rounding noise, and the CG
            # recurrence forms alpha = <r,z>/<p,Ap> = 0/0 -- returning
            # NaN or finite garbage. A garbage POTENTIAL is worse than a
            # garbage density: the electron solve then faithfully solves
            # its linear system at the insane field, so a residual test
            # on the transport solve alone passes (8e77 really was the
            # solution of its system). Accept CG's answer only if it
            # fits Poisson at least as well as the warm start.
            x, _ = cg(matvec, bb, x0=x0, M=lambda y: y / diag,
                      tol=tol, atol=tol * 1.0e-3, maxiter=maxiter)
            r_new = jnp.linalg.norm(matvec(x) - bb)
            r_ref = jnp.linalg.norm(matvec(x0) - bb)
            ok = jnp.isfinite(r_new) & (r_new <= r_ref)
            return jnp.where(ok, x, x0)

        return jax.lax.custom_linear_solve(mv, b, solve=inner,
                                           symmetric=True)

    return jax.jit(solve)


def make_jax_fkpm_stepper(tc: TransportCoeffsJAX, pc: PoissonCoeffsJAX,
                          p: FKPMParams, source_fn=None, inelastic_fn=None,
                          evolve_energy: bool = False, n_sub: int = 100,
                          cg_tol: float = 1e-8, cg_maxiter: int = 2000,
                          courant_clip: float | None = None,
                          es_joule: str = "drift",
                          wall_flux: str = "thermal",
                          ion_wall: str = "thermal",
                          ion_field: str = "static",
                          implicit_electrons: bool = False,
                          n_gummel: int = 2, im_tol: float = 1e-9,
                          im_maxiter: int = 2000):
    """Return a jitted coupled FKPM stepper

        step(ne, ni, n_eps, ss_r, ss_z, Phi, dt_e, dt_i, dt_eps, S_ext_eV)
            -> (ne', ni', n_eps', ss_r', ss_z', Phi')

    advancing electron + ion continuity, the dielectric surface charge,
    and the semi-implicit (P2) potential by `n_sub` substeps; with
    `evolve_energy=True` the electron energy density n_eps advances
    alongside (doc Eq. 12 with the ES Joule term from the solved field,
    mirroring `electron_energy.make_jax_energy_stepper`), otherwise Te is
    (2/3) n_eps / ne with n_eps frozen. `dt_*` are scalars (time-
    accurate) or (Nr, Nz) local pseudo-time arrays (steady-state
    acceleration, doc Sec. 12.4 -- the P2 delta form makes the
    accelerated fixed point satisfy the exact Poisson equation).

    source_fn(ne, Te) -> ionization source (1/m^3 s), applied to *both*
    species (charge-conserving, so it cancels in the predicted rho);
    inelastic_fn(ne, Te) -> eV/m^3/s energy sink.

    courant_clip : when set (e.g. 0.4), the provided dt arrays are
    additionally clipped each substep to courant_clip / rate with the
    per-cell explicit stability rate evaluated at the CURRENT field and
    Te (the doc Sec. 5.3 Courant bound). Required for local pseudo-time
    runs in which the field develops inside a block: sheath-strength
    fields with the Eq. 27 flux-limited ion mobility give drift speeds
    ~1e7 m/s, so any dt frozen at block start violates Courant as soon
    as the sheaths charge up. Reverse-mode
    differentiable end to end: the Poisson solve is wrapped in
    `custom_linear_solve`, the substep loop is a static-trip-count
    `fori_loop`.

    implicit_electrons (P3, module docstring; default False leaves the
    explicit path bit-for-bit unchanged): electron continuity AND the
    energy density become unknowns of backward-Euler transport solves
    -- the full drift-diffusion divergence acts on the future density,
    removing the explicit electron-diffusion limit (~1 ps on 0.3 mm
    cells at 10 mTorr; D_eps = 5/3 mu_e Te carries the identical limit,
    so both are done). Per substep, `n_gummel` Gummel sweeps of
    [P2 Poisson at the latest solved n_e] -> [implicit electron solve at
    the new field] -> [implicit energy solve], with the P2 electron
    augmentation as the linearized drift response of the solved density
    to the potential update; the delta form still cancels it at a
    stationary point. Coefficients (D_e, face drift velocity, wall
    v_th) are frozen within a sweep at the sweep-entry Te and field, so
    each solve is linear; particle/inelastic/elastic sources are frozen
    at substep entry, S(n^t, Te^t). The solves are Jacobi-preconditioned
    BiCGStab (im_tol/im_maxiter) in `custom_linear_solve` with an
    EXPLICIT transpose_solve -- the upwinded operator is non-symmetric;
    symmetric=True would silently corrupt reverse-mode gradients while
    leaving forward results correct. All species share ONE per-cell
    pseudo-clock: min of the supplied dt ceilings (chemistry caps from
    the caller), the in-loop ion Courant clip, and the sheath-RC
    charging limit (coupled field-charge dynamics -- NOT redundant with
    implicit electron transport); electron/energy transport rates no
    longer enter. The electron positivity limiter is off on this path
    (backward Euler + upwinding is an M-matrix, unconditionally
    positivity-preserving; limiting the implicit fluxes would break the
    charge ledger against the solved densities), the ion limiter stays.
    sigma_s is charged from the operator fluxes reconstructed at the
    SOLVED densities and the final-sweep field -- exactly the fluxes
    that produced the density change -- so the discrete charge ledger
    is exact. Remaining stability limit: ion sheath drift (~0.1-1 ns on
    0.3 mm cells), i.e. a ~2-3 decade gain over the electron-diffusion
    bound.
    """
    import jax
    import jax.numpy as jnp

    psolve = make_jax_poisson_solver(tol=cg_tol, maxiter=cg_maxiter)

    cp = 0.5 * (1.0 - p.re) / (1.0 + p.re)
    ce = (5.0 / 6.0) * (1.0 - p.re) / (1.0 + p.re)
    c53 = 5.0 / 3.0
    nu_m = jnp.asarray(p.nu_m)
    mu_e_s = QE / (ME * nu_m)

    padW = lambda X: jnp.concatenate(
        [jnp.zeros((1, X.shape[1]), X.dtype), X], axis=0)
    padE = lambda X: jnp.concatenate(
        [X, jnp.zeros((1, X.shape[1]), X.dtype)], axis=0)
    padS = lambda X: jnp.concatenate(
        [jnp.zeros((X.shape[0], 1), X.dtype), X], axis=1)
    padN = lambda X: jnp.concatenate(
        [X, jnp.zeros((X.shape[0], 1), X.dtype)], axis=1)

    def div(Fr, Fz):
        return tc.active * (
            (tc.area_r[1:, :] * Fr[1:, :] - tc.area_r[:-1, :] * Fr[:-1, :]
             + tc.area_z[:, 1:] * Fz[:, 1:]
             - tc.area_z[:, :-1] * Fz[:, :-1]) / tc.volume)

    def efield(Phi):
        Er = -(pc.esol_r * padE(Phi) - pc.wsol_r * padW(Phi)) / pc.dcp_r
        Ez = -(pc.nsol_z * padN(Phi) - pc.ssol_z * padS(Phi)) / pc.dcp_z
        return pc.live_r * Er, pc.live_z * Ez

    def limit_outflow(N, Fr, Fz, dt):
        """Donor-side positivity limiter over ALL faces: scale every
        face flux by its donor cell's factor so no cell loses more than
        95% of its content per substep through total outflow.  Feeds
        BOTH the density update and the sigma_s charging (exact charge
        ledger); also the nonlinear stabilizer of the coupled update
        during field transients (see `_limit_outflow`)."""
        dtc = jnp.broadcast_to(jnp.asarray(dt), N.shape)
        outV = (tc.area_r[1:, :] * jnp.maximum(Fr[1:, :], 0.0)
                + tc.area_r[:-1, :] * jnp.maximum(-Fr[:-1, :], 0.0)
                + tc.area_z[:, 1:] * jnp.maximum(Fz[:, 1:], 0.0)
                + tc.area_z[:, :-1] * jnp.maximum(-Fz[:, :-1], 0.0))
        out = outV / tc.volume
        hit = dtc * out > 0.95 * N
        den = jnp.where(hit, dtc * out, 1.0)
        th = jnp.where(hit, 0.95 * N / den, 1.0)
        fr = jnp.where(Fr > 0.0, padW(th), padE(th)) * Fr
        fz = jnp.where(Fz > 0.0, padS(th), padN(th)) * Fz
        return fr, fz

    def e_fluxes(N, D, mu, sign, Er, Ez, vth, cw, dt, br=1.0, bz=1.0):
        """grad(DN) diffusion + donor-cell drift + thermal wall (Eq. 36),
        wall part positivity-limited. br/bz are the `sheath_r`/`sheath_z`
        wall multipliers (1.0 on the thermal path)."""
        G = D * N
        f = cw * vth * N
        vr = sign * 0.5 * (padW(mu) + padE(mu)) * Er
        Fr = tc.int_r * (-(padE(G) - padW(G)) / tc.dc_r
                         + vr * jnp.where(vr > 0.0, padW(N), padE(N)))
        vz = sign * 0.5 * (padS(mu) + padN(mu)) * Ez
        Fz = tc.int_z * (-(padN(G) - padS(G)) / tc.dc_z
                         + vz * jnp.where(vz > 0.0, padS(N), padN(N)))
        Wr = br * (tc.wall_e * padW(f) - tc.wall_w * padE(f))
        Wz = bz * (tc.wall_n * padS(f) - tc.wall_s * padN(f))
        return limit_outflow(N, Fr + Wr, Fz + Wz, dt)

    def i_fluxes(N, Er, Ez, dt, Te=None):
        """Ion drift-diffusion + Eq. 37 wall (thermal + gated drift),
        wall part positivity-limited.

        ion_wall == "bohm": the wall coefficient is floored at the Bohm
        speed u_B = sqrt(q Te / M). Drift-diffusion ions at T_i = T_g
        reach the wall at 0.25 vth_i ~ 100 m/s where the sheath-edge
        criterion demands u_B ~ 3e3 m/s: the missing presheath
        acceleration makes the ion loss ~30x too small, and the
        floating condition dPhi/Te = ln(c_p vth_e / Gamma_i-per-ion)
        then pushes the potential to ~ln(vth_e/vth_i) ~ 8 Te instead of
        ~5 Te (measured: 6.5 at the wafer with the drift assist).
        DEPARTS FROM doc Eq. 37; amend it before quoting results."""
        G = p.D_i * N
        if ion_wall == "bohm" and Te is not None:
            ub = _uB(jnp.maximum(Te, p.Te_min))
            th = jnp.maximum(0.25 * p.vth_i, ub) * N
        else:
            th = 0.25 * p.vth_i * N
        vr = p.mu_i * Er
        Fr = tc.int_r * (-(padE(G) - padW(G)) / tc.dc_r
                         + vr * jnp.where(vr > 0.0, padW(N), padE(N)))
        vz = p.mu_i * Ez
        Fz = tc.int_z * (-(padN(G) - padS(G)) / tc.dc_z
                         + vz * jnp.where(vz > 0.0, padS(N), padN(N)))
        Wr = tc.wall_e * (padW(th)
                          + jnp.maximum(p.mu_i * Er, 0.0) * padW(N)) \
            - tc.wall_w * (padE(th)
                           + jnp.maximum(-p.mu_i * Er, 0.0) * padE(N))
        Wz = tc.wall_n * (padS(th)
                          + jnp.maximum(p.mu_i * Ez, 0.0) * padS(N)) \
            - tc.wall_s * (padN(th)
                           + jnp.maximum(-p.mu_i * Ez, 0.0) * padN(N))
        return limit_outflow(N, Fr + Wr, Fz + Wz, dt)

    def charging(Fr_e, Fz_e, Fr_i, Fz_i):
        dsr = QE * ((pc.scE_r - pc.scW_r) * (Fr_i - Fr_e))
        dsz = QE * ((pc.scN_z - pc.scS_z) * (Fz_i - Fz_e))
        return dsr, dsz

    def rhs_surface(ss_r, ss_z):
        qr = 0.5 * (pc.scE_r + pc.scW_r) * ss_r * pc.area_r
        qz = 0.5 * (pc.scN_z + pc.scS_z) * ss_z * pc.area_z
        return qr[:-1, :] + qr[1:, :] + qz[:, :-1] + qz[:, 1:]

    def aug(M_e, M_i, dte_fr, dte_fz, dti_fr, dti_fz):
        wr = pc.wpl_r + pc.epl_r
        wz = pc.spl_z + pc.npl_z
        MeR = (padW(M_e * pc.plasma) + padE(M_e * pc.plasma)) \
            / jnp.where(wr > 0.0, wr, 1.0)
        MiR = (padW(M_i * pc.plasma) + padE(M_i * pc.plasma)) \
            / jnp.where(wr > 0.0, wr, 1.0)
        MeZ = (padS(M_e * pc.plasma) + padN(M_e * pc.plasma)) \
            / jnp.where(wz > 0.0, wz, 1.0)
        MiZ = (padS(M_i * pc.plasma) + padN(M_i * pc.plasma)) \
            / jnp.where(wz > 0.0, wz, 1.0)
        gaug_r = QE * (dte_fr * MeR + dti_fr * MiR) * pc.ga_r
        gaug_z = QE * (dte_fz * MeZ + dti_fz * MiZ) * pc.ga_z
        return gaug_r, gaug_z

    def aug_matvec(x, gaug_r, gaug_z):
        nr, nz = x.shape
        cW = pc.wsol_r[:nr, :] * gaug_r[:nr, :]
        cE = pc.esol_r[1:, :] * gaug_r[1:, :]
        cS = pc.ssol_z[:, :nz] * gaug_z[:, :nz]
        cN = pc.nsol_z[:, 1:] * gaug_z[:, 1:]
        diag = gaug_r[:nr, :] + gaug_r[1:, :] + gaug_z[:, :nz] \
            + gaug_z[:, 1:]
        ax = diag * x \
            - cW * jnp.roll(x, 1, axis=0) - cE * jnp.roll(x, -1, axis=0) \
            - cS * jnp.roll(x, 1, axis=1) - cN * jnp.roll(x, -1, axis=1)
        return jnp.where(pc.solved > 0.5, ax, 0.0)

    def stab_rate(D, mu, Er, Ez):
        """Per-cell explicit stability rate at the current field
        (mirrors `transport.stability_rate`): sum_f (A_f/V)(D_f/d_f +
        |v_f|) over interior + wall faces."""
        rel_r = tc.int_r + tc.wall_e + tc.wall_w
        DW, DE = padW(D), padE(D)
        mW, mE = padW(mu), padE(mu)
        coef_r = rel_r * (jnp.maximum(DW, DE) / tc.dc_r
                          + jnp.abs(0.5 * (mW + mE) * Er))
        rel_z = tc.int_z + tc.wall_n + tc.wall_s
        DS, DN = padS(D), padN(D)
        mS, mN = padS(mu), padN(mu)
        coef_z = rel_z * (jnp.maximum(DS, DN) / tc.dc_z
                          + jnp.abs(0.5 * (mS + mN) * Ez))
        return tc.active * (
            (tc.area_r[1:, :] * coef_r[1:, :]
             + tc.area_r[:-1, :] * coef_r[:-1, :]
             + tc.area_z[:, 1:] * coef_z[:, 1:]
             + tc.area_z[:, :-1] * coef_z[:, :-1]) / tc.volume)

    if es_joule not in ("drift", "flux"):
        raise ValueError("es_joule must be 'drift' or 'flux'")
    if wall_flux not in ("thermal", "sheath"):
        raise ValueError("wall_flux must be 'thermal' or 'sheath'")
    if ion_wall not in ("thermal", "bohm"):
        raise ValueError("ion_wall must be 'thermal' or 'bohm'")
    if ion_field not in ("static", "effective"):
        raise ValueError("ion_field must be 'static' or 'effective'")
    if ion_field == "effective" and not implicit_electrons:
        raise NotImplementedError(
            "ion_field='effective' is implemented on the implicit (P3) "
            "path only; the explicit stepper is kept verbatim")
    _eff = ion_field == "effective"
    #: Bohm speed u_B = sqrt(q Te / M) from the ion parameters:
    #: mass via the thermal speed identity vth_i = sqrt(8 q Ti / pi M)
    _MI = 8.0 * QE * p.T_i_eV / (jnp.pi * p.vth_i ** 2)

    def _uB(Te):
        return jnp.sqrt(QE * Te / _MI)

    #: Richards et al. (APL 50, 492 (1987); Economou tutorial Eq. 6;
    #: doc Sec. 5.1 option 2 closing remark) ion momentum-transfer
    #: frequency, self-consistent with the code's own mobility via
    #: nu_i = q/(M mu_i). The ions drift in an EFFECTIVE field obeying
    #:     dE_eff/dt = nu_i (E_S - E_eff),
    #: a low-pass filter representing inertia: ions cannot follow field
    #: structure faster than their momentum relaxation (~ {:.1e} s at
    #: these parameters), which is precisely what the drift-diffusion
    #: closure wrongly lets them do inside the smeared sheath.
    _NU_I = QE / (_MI * p.mu_i)

    def sheath_r(Phi, Te):
        """Multiplier on the r-direction electron wall flux.

        "thermal" (doc Eq. 36): 1 -- the one-sided thermal flux
        (1-re)/(1+re) (1/2) v_th n_e, with NO dependence on the
        potential. That is only correct on a sheath-resolving mesh,
        where n_e in the wall cell is already depleted. Here
        lambda_D ~ 0.02-0.2 mm against ~2.5 mm cells, so the depletion
        cannot be represented and the electron loss cannot be throttled
        by the sheath: sigma_s charges without a restoring feedback and
        the potential overshoots.

        "sheath": exp(-(Phi_plasma - Phi_wall)/Te), the Boltzmann factor
        of the unresolved sheath drop across the face. This gives the
        wall flux a Phi dependence, which (a) lets the surface-charge
        balance close, and (b) throttles Gamma_e toward the ion flux,
        cutting the differential response dGamma_e/dPhi = -Gamma_e/Te
        that sets the explicit stability limit. DEPARTS FROM doc
        Eq. 36; amend it before quoting any result produced this way.
        """
        if wall_flux == "thermal":
            return 1.0
        tW = jnp.maximum(padW(Te), p.Te_min)
        tE = jnp.maximum(padE(Te), p.Te_min)
        fE = jnp.exp(-jnp.maximum(padW(Phi) - padE(Phi), 0.0) / tW)
        fW = jnp.exp(-jnp.maximum(padE(Phi) - padW(Phi), 0.0) / tE)
        return 1.0 + tc.wall_e * (fE - 1.0) + tc.wall_w * (fW - 1.0)

    def sheath_z(Phi, Te):
        if wall_flux == "thermal":
            return 1.0
        tS = jnp.maximum(padS(Te), p.Te_min)
        tN = jnp.maximum(padN(Te), p.Te_min)
        fN = jnp.exp(-jnp.maximum(padS(Phi) - padN(Phi), 0.0) / tS)
        fS = jnp.exp(-jnp.maximum(padN(Phi) - padS(Phi), 0.0) / tN)
        return 1.0 + tc.wall_n * (fN - 1.0) + tc.wall_s * (fS - 1.0)

    def es_heating(ne_c, mu_c, Erc, Ezc, Fr_e, Fz_e):
        """Electrostatic heating of the electron fluid, eV m^-3 s^-1.

        "drift" (default, doc Eq. 12 / Kushner 2009 Eq. 5):
            S = mu_e ne |E_S|^2
        the Joule term of a PURE-DRIFT electron flux.

        "flux": S = -Gamma_e . E_S, the work done by the field on the
        actual discrete electron flux -- the same face fluxes that
        advance the density, restricted to interior faces and averaged
        to cell centres like Erc/Ezc. Since Gamma_e = -mu_e ne E
        - grad(De ne), this equals the drift form PLUS the diffusive
        part, and the two very nearly cancel in an ambipolar plasma
        (net Gamma_e is the small ambipolar flux, not the drift flux).
        The drift-only form keeps a positive-definite source that
        cannot switch off; at GEC conditions it overstates the
        electrostatic heating by orders of magnitude, so this option
        exists to quantify that. NOTE: selecting "flux" departs from
        the model document -- Eq. 12 must be amended to match before
        any result produced with it is quoted.
        """
        if es_joule == "drift":
            return ne_c * mu_c * (Erc ** 2 + Ezc ** 2)
        Frc = 0.5 * (tc.int_r[:-1, :] * Fr_e[:-1, :]
                     + tc.int_r[1:, :] * Fr_e[1:, :])
        Fzc = 0.5 * (tc.int_z[:, :-1] * Fz_e[:, :-1]
                     + tc.int_z[:, 1:] * Fz_e[:, 1:])
        return -(Frc * Erc + Fzc * Ezc)

    def clip_dt(dt0, rate):
        # `rate` is exactly zero wherever the clipped mechanism is absent
        # (rate_q vanishes in every cell with no wall face, i.e. the bulk).
        # Guarding only the SELECT is not enough: the unselected branch is
        # still evaluated, and reverse mode then forms 0 * d(1/rate) =
        # 0 * inf = NaN, poisoning the whole adjoint even though the
        # forward value is correct. The denominator itself must be made
        # safe -- the same double-`where` idiom `limit_outflow` already
        # uses. Forward values are bit-identical.
        safe = jnp.where(rate > 0.0, rate, 1.0)
        return jnp.where(rate > 0.0,
                         jnp.minimum(dt0, courant_clip / safe), dt0)

    def step(ne, ni, n_eps, ss_r, ss_z, Phi, dt_e, dt_i, dt_eps, S_ext_eV):
        shape = ne.shape
        dte0 = jnp.broadcast_to(jnp.asarray(dt_e, jnp.float64), shape)
        dti0 = jnp.broadcast_to(jnp.asarray(dt_i, jnp.float64), shape)
        dtp0 = jnp.broadcast_to(jnp.asarray(dt_eps, jnp.float64), shape)

        def body(_, st):
            ne, ni, n_eps, ss_r, ss_z, Phi, dt_acc = st
            ne_eff = jnp.maximum(ne, p.ne_floor)
            Te = jnp.clip((2.0 / 3.0) * n_eps / ne_eff, p.Te_min, p.Te_max)
            mu_e = mu_e_s * jnp.ones_like(ne)
            vth_e = jnp.sqrt(8.0 * QE * Te / (jnp.pi * ME))
            S = source_fn(ne, Te) if source_fn is not None \
                else jnp.zeros_like(ne)

            # ---- P2 potential (delta form) ------------------------------- #
            Er0, Ez0 = efield(Phi)
            if courant_clip is not None:
                # single common charged-species dt: with dt_e != dt_i the
                # shared ionization source, the face charging, and the
                # transport divergence all pump NET charge in pseudo-time
                # (observed: sigma_s runaway to +2e-4 C/m^2 on the window
                # and a 45 kV megastructure). Equal per-cell dt scales
                # the net-charge dynamics uniformly instead -- same fixed
                # point, neutral pseudo-transient.
                dte = jnp.minimum(
                    clip_dt(dte0, stab_rate(mu_e * Te, mu_e, Er0, Ez0)),
                    clip_dt(dti0, stab_rate(
                        p.D_i * jnp.ones_like(ne),
                        p.mu_i * jnp.ones_like(ne), Er0, Ez0)))
                # sheath-RC charging limit: one substep may change a wall
                # face field by at most ~courant_clip x its current scale
                # (dE = e Gamma_wall dt / eps0), else sigma_s overshoots
                # its equilibrium every substep and the potential rings
                # at kV scale during sheath formation from uncharged
                # walls. Self-relaxing: as the sheath charges, E grows
                # and the wall cell depletes, so the limit fades.
                # On the sheath path the electron wall flux is throttled
                # by exp(-dPhi/Te) and, crucially, RESPONDS to Phi: the
                # quantity that limits an explicit sigma_s update is the
                # differential conductance dGamma_e/dPhi = -Gamma_e/Te,
                # not the net current (which nearly vanishes at ambipolar
                # balance and makes the clip ~6x too permissive). Using
                # the throttled Gamma_e here both fixes that estimate and
                # relaxes it, since Gamma_e -> Gamma_i at balance.
                fe_c = cp * vth_e * ne
                if ion_wall == "bohm":
                    fi_c = jnp.maximum(0.25 * p.vth_i,
                                       _uB(jnp.maximum(Te, p.Te_min))) * ni
                else:
                    fi_c = 0.25 * p.vth_i * ni
                mi_ni = p.mu_i * ni
                # NET charging current: gross (fe + fi) would keep
                # throttling a fully formed sheath forever; the field-
                # change limit only cares about the imbalance. During
                # sheath formation ions are static so net ~ fe anyway.
                brq, bzq = sheath_r(Phi, Te), sheath_z(Phi, Te)
                gw_r = brq * (tc.wall_e * jnp.abs(padW(fe_c) - padW(fi_c)
                                                  - padW(mi_ni)
                                                  * jnp.abs(Er0))
                              + tc.wall_w * jnp.abs(padE(fe_c) - padE(fi_c)
                                                    - padE(mi_ni)
                                                    * jnp.abs(Er0)))
                gw_z = bzq * (tc.wall_n * jnp.abs(padS(fe_c) - padS(fi_c)
                                                  - padS(mi_ni)
                                                  * jnp.abs(Ez0))
                              + tc.wall_s * jnp.abs(padN(fe_c) - padN(fi_c)
                                                    - padN(mi_ni)
                                                    * jnp.abs(Ez0)))
                Te_fr = jnp.maximum(padW(Te), padE(Te))
                Te_fz = jnp.maximum(padS(Te), padN(Te))
                es_r = jnp.maximum(jnp.abs(Er0), Te_fr / tc.dc_r)
                es_z = jnp.maximum(jnp.abs(Ez0), Te_fz / tc.dc_z)
                rq_r = QE * gw_r / (EPS0 * es_r)
                rq_z = QE * gw_z / (EPS0 * es_z)
                rate_q = jnp.maximum(
                    jnp.maximum(rq_r[1:, :], rq_r[:-1, :]),
                    jnp.maximum(rq_z[:, 1:], rq_z[:, :-1]))
                dte = clip_dt(dte, rate_q)
                # ... and the energy equation shares the SAME clock: with
                # dtp != dte the ratio Te = (2/3) n_eps / ne acquires
                # spurious pseudo-time dynamics (n_eps and ne marching at
                # different rates), which feeds D_e = mu_e Te and the
                # pressure-gradient fluxes and destabilizes the coupled
                # field-charge update even though each equation is
                # individually within its stability limit.
                dte = jnp.minimum(dte, clip_dt(dtp0, stab_rate(
                    c53 * mu_e * Te, c53 * mu_e, Er0, Ez0)))
                # bound the pseudo-clock ratio between NEIGHBORS: sharp
                # per-cell dt gradients (sheath-clipped cells 30-100x
                # below the bulk) scale the coupled field-charge Jacobian
                # non-uniformly and are themselves linearly destabilizing
                # even with all equations on one clock per cell -- the
                # classic local-time-stepping caveat. A few min-
                # propagation sweeps limit the growth to alpha per cell.
                big = jnp.where(tc.active > 0.0, dte, jnp.inf)
                for _ in range(14):
                    nb = jnp.minimum(
                        jnp.minimum(
                            jnp.concatenate([big[:1, :], big[:-1, :]], 0),
                            jnp.concatenate([big[1:, :], big[-1:, :]], 0)),
                        jnp.minimum(
                            jnp.concatenate([big[:, :1], big[:, :-1]], 1),
                            jnp.concatenate([big[:, 1:], big[:, -1:]], 1)))
                    big = jnp.minimum(big, 1.25 * nb)
                dte = jnp.where(tc.active > 0.0, big, 0.0)
                dti = dte
                dtp = dte
            else:
                dte, dti, dtp = dte0, dti0, dtp0
            dte_fr = jnp.maximum(padW(dte), padE(dte))
            dte_fz = jnp.maximum(padS(dte), padN(dte))
            dti_fr = jnp.maximum(padW(dti), padE(dti))
            dti_fz = jnp.maximum(padS(dti), padN(dti))
            Fr_e, Fz_e = e_fluxes(ne, mu_e * Te, mu_e, -1.0, Er0, Ez0,
                                  vth_e, cp, dte)
            Fr_i, Fz_i = i_fluxes(ni, Er0, Ez0, dti, Te)
            ne_p = ne + dte * (-div(Fr_e, Fz_e) + S)
            ni_p = ni + dti * (-div(Fr_i, Fz_i) + S)
            dsr_e = QE * (pc.scW_r - pc.scE_r) * Fr_e   # electron q = -e
            dsz_e = QE * (pc.scS_z - pc.scN_z) * Fz_e
            dsr_i = QE * (pc.scE_r - pc.scW_r) * Fr_i
            dsz_i = QE * (pc.scN_z - pc.scS_z) * Fz_i
            ssr_p = ss_r + dte_fr * dsr_e + dti_fr * dsr_i
            ssz_p = ss_z + dte_fz * dsz_e + dti_fz * dsz_i
            gaug_r, gaug_z = aug(mu_e * ne, p.mu_i * ni,
                                 dte_fr, dte_fz, dti_fr, dti_fz)
            rhs = pc.plasma * QE * (ni_p - ne_p) * pc.volume \
                + rhs_surface(ssr_p, ssz_p) + aug_matvec(Phi, gaug_r, gaug_z)
            Phi = psolve(pc, rhs, gaug_r, gaug_z, Phi)

            # ---- updates at the new field -------------------------------- #
            Er, Ez = efield(Phi)
            br_w, bz_w = sheath_r(Phi, Te), sheath_z(Phi, Te)
            Fr_e, Fz_e = e_fluxes(ne, mu_e * Te, mu_e, -1.0, Er, Ez,
                                  vth_e, cp, dte, br_w, bz_w)
            Fr_i, Fz_i = i_fluxes(ni, Er, Ez, dti, Te)
            ne = jnp.maximum(ne + dte * (-div(Fr_e, Fz_e) + S), 0.0)
            ni = jnp.maximum(ni + dti * (-div(Fr_i, Fz_i) + S), 0.0)
            ss_r = ss_r + dte_fr * QE * (pc.scW_r - pc.scE_r) * Fr_e \
                + dti_fr * QE * (pc.scE_r - pc.scW_r) * Fr_i
            ss_z = ss_z + dte_fz * QE * (pc.scS_z - pc.scN_z) * Fz_e \
                + dti_fz * QE * (pc.scN_z - pc.scS_z) * Fz_i

            if evolve_energy:
                # ES Joule field from INTERIOR faces only: wall-face E is
                # the numerically smeared sheath (unresolved lambda_D) --
                # physically the electron density there is depleted and
                # the sheath drop does ion acceleration work, not electron
                # heating. Including it rails Te via mu_e E^2 on coarse
                # cells (doc Sec. 4.1 Option A caveat).
                Erc = 0.5 * (tc.int_r[:-1, :] * Er[:-1, :]
                             + tc.int_r[1:, :] * Er[1:, :])
                Ezc = 0.5 * (tc.int_z[:, :-1] * Ez[:, :-1]
                             + tc.int_z[:, 1:] * Ez[:, 1:])
                G = c53 * mu_e * Te * n_eps
                fw = ce * vth_e * n_eps
                vr = -c53 * 0.5 * (padW(mu_e) + padE(mu_e)) * Er
                vz = -c53 * 0.5 * (padS(mu_e) + padN(mu_e)) * Ez
                PW, PE = padW(n_eps), padE(n_eps)
                Fr = tc.int_r * (-(padE(G) - padW(G)) / tc.dc_r
                                 + vr * jnp.where(vr > 0.0, PW, PE)) \
                    + br_w * (tc.wall_e * padW(fw) - tc.wall_w * padE(fw))
                PS, PN = padS(n_eps), padN(n_eps)
                Fz = tc.int_z * (-(padN(G) - padS(G)) / tc.dc_z
                                 + vz * jnp.where(vz > 0.0, PS, PN)) \
                    + bz_w * (tc.wall_n * padS(fw) - tc.wall_s * padN(fw))
                Se = S_ext_eV + es_heating(ne, mu_e, Erc, Ezc,
                                           Fr_e, Fz_e) \
                    - 3.0 * p.mass_ratio * nu_m * ne * (Te - p.Tg_eV)
                if inelastic_fn is not None:
                    Se = Se - inelastic_fn(ne, Te)
                n_eps = jnp.maximum(n_eps + dtp * (-div(Fr, Fz) + Se), 0.0)

            # accumulate the clock ACTUALLY taken: dte after the ion
            # Courant re-clip, the sheath-RC limit and the smoothing
            # sweeps. The dt arrays passed in are only ceilings, so this
            # is the sole honest measure of how far a slice advances.
            dt_acc = dt_acc + jnp.sum(dte * tc.active) \
                / jnp.maximum(jnp.sum(tc.active), 1.0)
            return ne, ni, n_eps, ss_r, ss_z, Phi, dt_acc

        return jax.lax.fori_loop(0, n_sub, body,
                                 (ne, ni, n_eps, ss_r, ss_z, Phi, 0.0))

    if not implicit_electrons:
        return jax.jit(step)

    # ------------------------------------------------------------------ P3 --
    # Fully implicit electron continuity + energy (module docstring P3;
    # Kushner 2009 option 3 disambiguated). The explicit `step` above is
    # untouched and remains the default.
    from jax.scipy.sparse.linalg import bicgstab

    def imsolve(mv, b, x0, diag):
        """Backward-Euler transport solve: Jacobi-preconditioned
        BiCGStab wrapped in `custom_linear_solve`. The operator
        (1/dt + L) with donor-cell upwinding and grad(DN) diffusion is
        a diagonally dominant M-matrix but NOT symmetric, so the
        transpose solve is supplied explicitly (JAX builds vecmat = A^T
        by linear transposition of `mv`); passing symmetric=True here
        would silently corrupt reverse-mode gradients while leaving
        forward results correct. diag(A^T) == diag(A), so one Jacobi
        preconditioner serves both directions."""
        # BiCGStab BREAKDOWN GUARD. The warm start is the previous
        # solution, so as the outer iteration converges the initial
        # residual underflows towards ~1e-16 ||b||; BiCGStab then forms
        # rho = <r0, r> ~ 0 and alpha = rho / <r0, v> = 0/0, which can
        # surface as NaN *or as finite garbage* (observed: 8e77 on an
        # otherwise stationary iteration -- an isfinite() test waved it
        # straight through). The only acceptance test that is actually
        # correct is the RESIDUAL: keep the solver's answer only if it
        # fits the equations at least as well as the warm start did.
        # Breakdown happens precisely when the warm start already solves
        # the system, so falling back to it loses nothing. Cost: one
        # extra matvec per solve. `atol` gets a small absolute floor so
        # the iteration can exit cleanly instead of grinding the
        # recurrence into rounding noise.
        atol_abs = im_tol * 1.0e-3

        def _accept(matvec, bb, x, x_ref):
            r_new = jnp.linalg.norm(matvec(x) - bb)
            r_ref = jnp.linalg.norm(matvec(x_ref) - bb)
            ok = jnp.isfinite(r_new) & (r_new <= r_ref)
            return jnp.where(ok, x, x_ref)

        def fwd(matvec, bb):
            x, _ = bicgstab(matvec, bb, x0=x0, M=lambda y: y / diag,
                            tol=im_tol, atol=atol_abs, maxiter=im_maxiter)
            return _accept(matvec, bb, x, x0)

        def tsp(vecmat, bb):
            x0t = bb / diag                      # Jacobi guess
            x, _ = bicgstab(vecmat, bb, x0=x0t, M=lambda y: y / diag,
                            tol=im_tol, atol=atol_abs, maxiter=im_maxiter)
            return _accept(vecmat, bb, x, x0t)

        return jax.lax.custom_linear_solve(mv, b, solve=fwd,
                                           transpose_solve=tsp)

    def op_fluxes(x, D, vfr, vfz, wf, br=1.0, bz=1.0):
        """Frozen-coefficient transport fluxes of the implicit operator:
        the SAME stencil as `e_fluxes` (grad(DN) diffusion, donor-cell
        drift at the frozen face velocity vfr/vfz, Eq. 36 thermal wall
        with cell coefficient wf = coeff*v_th) MINUS the positivity
        limiter, which is structurally unnecessary here (backward Euler
        + upwinding is an M-matrix) and would break the charge ledger
        against the solved densities. Linear in x by construction: the
        upwind switch tests the frozen vfr/vfz, never x -- required by
        `custom_linear_solve`."""
        G = D * x
        f = wf * x
        Fr = tc.int_r * (-(padE(G) - padW(G)) / tc.dc_r
                         + vfr * jnp.where(vfr > 0.0, padW(x), padE(x))) \
            + br * (tc.wall_e * padW(f) - tc.wall_w * padE(f))
        Fz = tc.int_z * (-(padN(G) - padS(G)) / tc.dc_z
                         + vfz * jnp.where(vfz > 0.0, padS(x), padN(x))) \
            + bz * (tc.wall_n * padS(f) - tc.wall_s * padN(f))
        return Fr, Fz

    def _at(b, which):
        """Slice a face array to the cell-shaped east(1)/west(0) face."""
        if isinstance(b, float):
            return b
        return b[1:, :] if which else b[:-1, :]

    def _atz(b, which):
        if isinstance(b, float):
            return b
        return b[:, 1:] if which else b[:, :-1]

    def op_diag(D, vfr, vfz, wf, inv_dt, br=1.0, bz=1.0):
        """Exact diagonal of (1/dt + L): the cell's own coefficient in
        its outgoing flux through each face (interior D/dc + the
        outgoing upwind half of |v|; thermal coefficient on wall
        faces), for the Jacobi preconditioner of A and A^T alike."""
        cE = tc.int_r[1:, :] * (D / tc.dc_r[1:, :]
                                + jnp.maximum(vfr[1:, :], 0.0)) \
            + tc.wall_e[1:, :] * wf * _at(br, 1)
        cW = tc.int_r[:-1, :] * (D / tc.dc_r[:-1, :]
                                 + jnp.maximum(-vfr[:-1, :], 0.0)) \
            + tc.wall_w[:-1, :] * wf * _at(br, 0)
        cN = tc.int_z[:, 1:] * (D / tc.dc_z[:, 1:]
                                + jnp.maximum(vfz[:, 1:], 0.0)) \
            + tc.wall_n[:, 1:] * wf * _atz(bz, 1)
        cS = tc.int_z[:, :-1] * (D / tc.dc_z[:, :-1]
                                 + jnp.maximum(-vfz[:, :-1], 0.0)) \
            + tc.wall_s[:, :-1] * wf * _atz(bz, 0)
        rate = (tc.area_r[1:, :] * cE + tc.area_r[:-1, :] * cW
                + tc.area_z[:, 1:] * cN + tc.area_z[:, :-1] * cS) \
            / tc.volume
        return jnp.where(tc.active > 0.5, inv_dt + rate, 1.0)

    def step_imp(ne, ni, n_eps, ss_r, ss_z, Phi, Ei_r, Ei_z,
                 dt_e, dt_i, dt_eps, S_ext_eV):
        shape = ne.shape
        dte0 = jnp.broadcast_to(jnp.asarray(dt_e, jnp.float64), shape)
        dti0 = jnp.broadcast_to(jnp.asarray(dt_i, jnp.float64), shape)
        dtp0 = jnp.broadcast_to(jnp.asarray(dt_eps, jnp.float64), shape)

        def body(_, st):
            ne, ni, n_eps, ss_r, ss_z, Phi, Ei_r, Ei_z, dt_acc = st
            ne_eff = jnp.maximum(ne, p.ne_floor)
            Te = jnp.clip((2.0 / 3.0) * n_eps / ne_eff, p.Te_min, p.Te_max)
            mu_e = mu_e_s * jnp.ones_like(ne)
            vth_e = jnp.sqrt(8.0 * QE * Te / (jnp.pi * ME))
            S = source_fn(ne, Te) if source_fn is not None \
                else jnp.zeros_like(ne)

            # ---- ONE common pseudo-clock -------------------------------- #
            # Electron/energy transport rates no longer enter (implicit);
            # kept: the caller's ceilings (chemistry / sink caps), the
            # ion Courant clip at the current field, and the sheath-RC
            # charging limit -- the latter concerns the coupled field-
            # charge dynamics, not electron transport, and stays. All
            # species and the energy equation share the same clock per
            # cell (dt_e != dt_i pumps NET charge through the shared
            # ionization source; dt_eps != dt_e gives Te spurious
            # pseudo-dynamics -- see the explicit body's comments).
            Er0, Ez0 = efield(Phi)
            # field the IONS respond to: the Richards effective field on
            # the "effective" path (lagged one substep -- it relaxes at
            # the END of the substep toward the final field), the true
            # field otherwise. Electrons always see the true field.
            Eir_u = Ei_r if _eff else Er0
            Eiz_u = Ei_z if _eff else Ez0
            if courant_clip is not None:
                dte = jnp.minimum(
                    dte0,
                    clip_dt(dti0, stab_rate(
                        p.D_i * jnp.ones_like(ne),
                        p.mu_i * jnp.ones_like(ne), Eir_u, Eiz_u)))
                # sheath-RC charging limit (verbatim from the explicit
                # body): one substep may change a wall face field by at
                # most ~courant_clip x its current scale.
                fe_c = cp * vth_e * ne
                if ion_wall == "bohm":
                    fi_c = jnp.maximum(0.25 * p.vth_i,
                                       _uB(jnp.maximum(Te, p.Te_min))) * ni
                else:
                    fi_c = 0.25 * p.vth_i * ni
                mi_ni = p.mu_i * ni
                brq, bzq = sheath_r(Phi, Te), sheath_z(Phi, Te)
                gw_r = brq * (tc.wall_e * jnp.abs(padW(fe_c) - padW(fi_c)
                                                  - padW(mi_ni)
                                                  * jnp.abs(Eir_u))
                              + tc.wall_w * jnp.abs(padE(fe_c) - padE(fi_c)
                                                    - padE(mi_ni)
                                                    * jnp.abs(Eir_u)))
                gw_z = bzq * (tc.wall_n * jnp.abs(padS(fe_c) - padS(fi_c)
                                                  - padS(mi_ni)
                                                  * jnp.abs(Eiz_u))
                              + tc.wall_s * jnp.abs(padN(fe_c) - padN(fi_c)
                                                    - padN(mi_ni)
                                                    * jnp.abs(Eiz_u)))
                Te_fr = jnp.maximum(padW(Te), padE(Te))
                Te_fz = jnp.maximum(padS(Te), padN(Te))
                es_r = jnp.maximum(jnp.abs(Er0), Te_fr / tc.dc_r)
                es_z = jnp.maximum(jnp.abs(Ez0), Te_fz / tc.dc_z)
                rq_r = QE * gw_r / (EPS0 * es_r)
                rq_z = QE * gw_z / (EPS0 * es_z)
                rate_q = jnp.maximum(
                    jnp.maximum(rq_r[1:, :], rq_r[:-1, :]),
                    jnp.maximum(rq_z[:, 1:], rq_z[:, :-1]))
                dte = clip_dt(dte, rate_q)
                dte = jnp.minimum(dte, dtp0)
                # neighbour-ratio smoothing sweeps (verbatim): sharp
                # per-cell dt gradients destabilize the coupled field-
                # charge Jacobian even with implicit transport.
                big = jnp.where(tc.active > 0.0, dte, jnp.inf)
                for _s in range(14):
                    nb = jnp.minimum(
                        jnp.minimum(
                            jnp.concatenate([big[:1, :], big[:-1, :]], 0),
                            jnp.concatenate([big[1:, :], big[-1:, :]], 0)),
                        jnp.minimum(
                            jnp.concatenate([big[:, :1], big[:, :-1]], 1),
                            jnp.concatenate([big[:, 1:], big[:, -1:]], 1)))
                    big = jnp.minimum(big, 1.25 * nb)
                dte = jnp.where(tc.active > 0.0, big, 0.0)
            else:
                dte = jnp.where(tc.active > 0.0,
                                jnp.minimum(jnp.minimum(dte0, dti0), dtp0),
                                0.0)
            dt_fr = jnp.maximum(padW(dte), padE(dte))
            dt_fz = jnp.maximum(padS(dte), padN(dte))
            inv_dt = 1.0 / jnp.where(dte > 0.0, dte, 1.0)

            # ---- substep-entry (frozen) sources: S(n^t, Te^t) ----------- #
            if evolve_energy:
                S_stat = S_ext_eV \
                    - 3.0 * p.mass_ratio * nu_m * ne * (Te - p.Tg_eV)
                if inelastic_fn is not None:
                    S_stat = S_stat - inelastic_fn(ne, Te)

            # ---- Gummel block sweeps (static trip count) ---------------- #
            ne_g, neps_g, Phi_g = ne, n_eps, Phi
            for _k in range(max(int(n_gummel), 1)):
                # coefficient refresh between sweeps (sweep 0 == entry)
                Te_g = jnp.clip((2.0 / 3.0) * neps_g
                                / jnp.maximum(ne_g, p.ne_floor),
                                p.Te_min, p.Te_max)
                vth_g = jnp.sqrt(8.0 * QE * Te_g / (jnp.pi * ME))
                D_g = mu_e * Te_g
                wf_e = cp * vth_g
                # frozen for this sweep -> the operator stays LINEAR in x
                br_g = sheath_r(Phi_g, Te_g)
                bz_g = sheath_z(Phi_g, Te_g)

                # (a) P2 Poisson at the latest solved electron density;
                # its drift response to the potential update is the
                # (unchanged) electron augmentation, cancelled at a
                # stationary point by the delta form.
                Er_g, Ez_g = efield(Phi_g)
                Fr_i, Fz_i = i_fluxes(ni,
                                      Eir_u if _eff else Er_g,
                                      Eiz_u if _eff else Ez_g,
                                      dte, Te_g)
                ni_p = ni + dte * (-div(Fr_i, Fz_i) + S)
                vfr = -0.5 * (padW(mu_e) + padE(mu_e)) * Er_g
                vfz = -0.5 * (padS(mu_e) + padN(mu_e)) * Ez_g
                Fr_e, Fz_e = op_fluxes(ne_g, D_g, vfr, vfz, wf_e,
                                       br_g, bz_g)
                ssr_p = ss_r + dt_fr * (QE * (pc.scW_r - pc.scE_r) * Fr_e
                                        + QE * (pc.scE_r - pc.scW_r) * Fr_i)
                ssz_p = ss_z + dt_fz * (QE * (pc.scS_z - pc.scN_z) * Fz_e
                                        + QE * (pc.scN_z - pc.scS_z) * Fz_i)
                gaug_r, gaug_z = aug(mu_e * ne_g, p.mu_i * ni,
                                     dt_fr, dt_fz, dt_fr, dt_fz)
                rhs = pc.plasma * QE * (ni_p - ne_g) * pc.volume \
                    + rhs_surface(ssr_p, ssz_p) \
                    + aug_matvec(Phi_g, gaug_r, gaug_z)
                Phi_g = psolve(pc, rhs, gaug_r, gaug_z, Phi_g)

                # (b) backward-Euler electron continuity at the NEW
                # field; coefficients frozen -> linear system.
                Er_g, Ez_g = efield(Phi_g)
                vfr = -0.5 * (padW(mu_e) + padE(mu_e)) * Er_g
                vfz = -0.5 * (padS(mu_e) + padN(mu_e)) * Ez_g

                def mv_e(x, D_g=D_g, vfr=vfr, vfz=vfz, wf_e=wf_e,
                         inv_dt=inv_dt):
                    Fr, Fz = op_fluxes(x, D_g, vfr, vfz, wf_e,
                                       br_g, bz_g)
                    return jnp.where(tc.active > 0.5,
                                     inv_dt * x + div(Fr, Fz), x)

                b_e = tc.active * (inv_dt * ne + S)
                ne_g = imsolve(mv_e, b_e, ne_g,
                               op_diag(D_g, vfr, vfz, wf_e, inv_dt,
                                       br_g, bz_g))

                if evolve_energy:
                    # (c) backward-Euler energy at the SAME field:
                    # D_eps = (5/3) mu_e Te, mu_eps = (5/3) mu_e, the
                    # Eq. 36 energy wall coefficient (5/6)(1-re)/(1+re);
                    # ES Joule from INTERIOR faces only (explicit-body
                    # caveat), entry ne, sweep field.
                    Erc = 0.5 * (tc.int_r[:-1, :] * Er_g[:-1, :]
                                 + tc.int_r[1:, :] * Er_g[1:, :])
                    Ezc = 0.5 * (tc.int_z[:, :-1] * Ez_g[:, :-1]
                                 + tc.int_z[:, 1:] * Ez_g[:, 1:])
                    Fre_s, Fze_s = op_fluxes(ne_g, D_g, vfr, vfz, wf_e,
                                             br_g, bz_g)
                    S_eps = S_stat + es_heating(ne, mu_e, Erc, Ezc,
                                                Fre_s, Fze_s)
                    wf_p = ce * vth_g

                    def mv_p(x, D_g=D_g, vfr=vfr, vfz=vfz, wf_p=wf_p,
                             inv_dt=inv_dt):
                        Fr, Fz = op_fluxes(x, c53 * D_g, c53 * vfr,
                                           c53 * vfz, wf_p, br_g, bz_g)
                        return jnp.where(tc.active > 0.5,
                                         inv_dt * x + div(Fr, Fz), x)

                    b_p = tc.active * (inv_dt * n_eps + S_eps)
                    neps_g = imsolve(mv_p, b_p, neps_g,
                                     op_diag(c53 * D_g, c53 * vfr,
                                             c53 * vfz, wf_p, inv_dt,
                                             br_g, bz_g))

            # ---- final updates: exact discrete charge ledger ------------ #
            # sigma_s MUST be charged from the same fluxes that produced
            # the solved densities: the electron fluxes are the operator
            # fluxes reconstructed at (ne_g; final-sweep field and
            # coefficients) -- by the linear system these satisfy
            # (ne_g - ne)/dt = -div F(ne_g) + S to solver tolerance, so
            # any other flux (e.g. a limited one, or one at refreshed
            # coefficients) leaks charge against the wall ledger. The
            # ion update stays explicit at the final field with its
            # limiter; its charging uses those same limited fluxes.
            Fr_e, Fz_e = op_fluxes(ne_g, D_g, vfr, vfz, wf_e, br_g, bz_g)
            Fr_i, Fz_i = i_fluxes(ni,
                                  Eir_u if _eff else Er_g,
                                  Eiz_u if _eff else Ez_g,
                                  dte, Te_g)
            # the max(., 0) guards mirror the explicit path; for the
            # Table 2 argon set S >= 0 and the M-matrix solve keeps
            # ne_g, neps_g nonnegative, so they are no-ops here.
            ne = jnp.maximum(ne_g, 0.0)
            ni = jnp.maximum(ni + dte * (-div(Fr_i, Fz_i) + S), 0.0)
            if evolve_energy:
                n_eps = jnp.maximum(neps_g, 0.0)
            ss_r = ss_r + dt_fr * (QE * (pc.scW_r - pc.scE_r) * Fr_e
                                   + QE * (pc.scE_r - pc.scW_r) * Fr_i)
            ss_z = ss_z + dt_fz * (QE * (pc.scS_z - pc.scN_z) * Fz_e
                                   + QE * (pc.scN_z - pc.scS_z) * Fz_i)
            if _eff:
                # Richards relaxation toward the END-of-substep field,
                # exact exponential update of dE/dt = nu_i (E - E_eff)
                # over the face-local dt (unconditionally stable, exact
                # for the linear ODE regardless of nu_i * dt).
                fr = jnp.exp(-_NU_I * dt_fr)
                fz = jnp.exp(-_NU_I * dt_fz)
                Ei_r = Er_g + (Ei_r - Er_g) * fr
                Ei_z = Ez_g + (Ei_z - Ez_g) * fz
            dt_acc = dt_acc + jnp.sum(dte * tc.active) \
                / jnp.maximum(jnp.sum(tc.active), 1.0)
            return ne, ni, n_eps, ss_r, ss_z, Phi_g, Ei_r, Ei_z, dt_acc

        return jax.lax.fori_loop(0, n_sub, body,
                                 (ne, ni, n_eps, ss_r, ss_z, Phi,
                                  Ei_r, Ei_z, 0.0))

    return jax.jit(step_imp)