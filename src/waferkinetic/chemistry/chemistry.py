"""
chemistry.py
============
Plasma chemistry for the 4-species argon benchmark (model document,
Sec. 7): the Table 2 reaction set as *data*, the generic volumetric
source assembly of Eq. 41,

    S_i = sum_j (nu''_ij - nu'_ij) r_j,
    r_j = k_j prod_{l in reactants} n_l^{nu'_lj},                (doc Eq. 41)

and the consistent electron energy-loss export

    L_e = sum_j r_j * d_eps_j     [eV m^-3 s^-1]                 (doc Eq. 12)

that plugs directly into `electron_energy.electron_rhs(..., inelastic=...)`
/ `make_jax_energy_stepper(..., inelastic_fn=...)`. The superelastic
channel (reaction 3, d_eps = -11.5 eV) enters the sum with its negative
sign, so collisional de-excitation *heats* the electrons; stepwise
ionization out of Ar* at 4.24 eV is carried as essential physics (doc
Sec. 7: it sustains the discharge near threshold, flattens the ionization
source, and produces the near-linear density-power scaling).

Species ordering (fixed): (e, Ar, Ar*, Ar+) -- indices IE, IAR, IARS,
IARP. Densities are handled as a stack n of shape (4, ...) so the same
assembly serves 0D states, (Nr, Nz) fields, and vmapped batches.

Rate coefficients
-----------------
Electron-impact k_j(Te) are supplied by the EETM as *callables* (tables
or fits); heavy-particle coefficients are constants (or Arrhenius in
Tgas -- constant for the isothermal benchmark). The bundled fits are the
illustrative Lieberman-style global-model expressions already used in
test_electron_energy.py, clearly labeled for later replacement:

    k_exc  = 2.48e-14 Te^0.33 exp(-12.78/Te)   (rxn 2, threshold 11.5 eV)
    k_iz   = 2.34e-14 Te^0.59 exp(-17.44/Te)   (rxn 4, threshold 15.8 eV)
    k_step = 6.8e-15  Te^0.67 exp(-4.20/Te)    (rxn 5, threshold 4.24 eV)
    k_sup  = detailed balance from k_exc (see `superelastic_from_excitation`)
    k_pen  = 6.2e-16 m^3/s                     (rxn 6, Penning, constant)
    k_qch  = 2.1e-21 m^3/s                     (rxn 7, two-body quench)

ILLUSTRATIVE ONLY: production runs replace every electron-impact entry
with Boltzmann-solver (BOLSIG+) lookup tables under the local
mean-energy approximation. The interface is already the final one --
each reaction carries an opaque `rate` callable k(Te) [m^3/s], so
Option B tables slot in via `argon_table2(rates={...})` without touching
this module.

Superelastic closure (documented choice): the Klein-Rosseland /
detailed-balance relation for a Maxwellian EEDF,

    k_sup(Te) = (g0/g*) * k_exc(Te) * exp(+d_eps / Te),

with g0 = 1 (Ar 1S0) and g* = 12 (statistical weight of the lumped 4s
manifold: 5 + 3 + 1 + 3). Applied to the *fitted* k_exc -- whose
effective Arrhenius energy is 12.78 eV rather than the 11.5 eV
threshold -- this folds into another Arrhenius form,

    k_sup = (2.48e-14/12) Te^0.33 exp(-(12.78 - 11.5)/Te)
          = 2.07e-15 Te^0.33 exp(-1.28/Te),

which stays bounded as Te -> 0 precisely because the fit energy exceeds
the threshold. Detailed balance strictly relates the underlying cross
sections; using it on a rate *fit* is part of the illustrative closure
and is superseded wholesale by BOLSIG tables (which supply k_sup from
the de-excitation cross section directly).

Elastic momentum transfer (reaction 1) is carried as table data for
completeness but contributes neither to Eq. 41 sources (nu'' = nu') nor
to the inelastic sum (d_eps = None): elastic energy exchange is already
the explicit 3 (me/M) nu_m ne (Te - Tg) term of the energy equation
(doc Eq. 12 / electron_energy.py), and counting it here would double it.
Penning (6) and quenching (7) likewise carry d_eps = None -- they are
heavy-particle channels outside the electron-impact loss sum of Eq. 12.

Layout follows the package conventions: plain-NumPy dataclasses for
setup; `jax_rate_callables` / `make_jax_source_fn` /
`make_jax_inelastic_fn` form the JAX boundary for jitted steppers
(static Python loops over the 7 reactions unroll under jit). Bundled
`RateFit` callables dispatch np vs jnp on the argument type; custom
callables used on the JAX path must themselves be jnp-traceable
(anything built from jnp ops or `jnp.interp` tables qualifies).

Units: SI + eV; k two-body m^3/s, r_j m^-3 s^-1, energy loss eV m^-3 s^-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

# ------------------------------------------------------------- species order
SPECIES = ("e", "Ar", "Ar*", "Ar+")
IE, IAR, IARS, IARP = 0, 1, 2, 3
NSPECIES = 4

# --------------------------------------------------- Table 2 thresholds (eV)
EPS_EXC = 11.5     #: e + Ar   -> e + Ar*
EPS_IZ = 15.8      #: e + Ar   -> 2e + Ar+
EPS_STEP = 4.24    #: e + Ar*  -> 2e + Ar+
G_RATIO_4S = 1.0 / 12.0  #: g(Ar 1S0) / g(lumped 4s manifold) = 1 / (5+3+1+3)

#: Heavy-particle coefficients (doc Sec. 7: constant for the benchmark).
K_PENNING = 6.2e-16   #: Ar* + Ar* -> e + Ar + Ar+   [m^3/s]
K_QUENCH = 2.1e-21    #: Ar* + Ar  -> Ar + Ar        [m^3/s]  (illustrative)


def _xp(x):
    """np for NumPy inputs, jnp for JAX arrays/tracers (single dispatch
    point so `RateFit` works identically in `electron_rhs` and inside
    jitted steppers)."""
    try:
        import jax
        if isinstance(x, jax.Array):
            import jax.numpy as jnp
            return jnp
    except ImportError:
        pass
    return np


# ----------------------------------------------------------------------------
# Rate-coefficient callables
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RateFit:
    """Arrhenius/Lieberman-form rate fit  k(Te) = A Te^b exp(-Ea/Te)
    [m^3/s, Te in eV]. ILLUSTRATIVE stand-in for Boltzmann tables; any
    callable k(Te) with the same signature is interchangeable."""
    A: float
    b: float
    Ea_eV: float
    label: str = ""

    def __call__(self, Te):
        return self.A * Te ** self.b * _xp(Te).exp(-self.Ea_eV / Te)


def superelastic_from_excitation(exc: RateFit, d_eps: float = EPS_EXC,
                                 g_ratio: float = G_RATIO_4S) -> RateFit:
    """Detailed-balance superelastic fit from an Arrhenius excitation fit:
    k_sup = g_ratio * A Te^b exp(-(Ea - d_eps)/Te). Requires Ea >= d_eps
    for boundedness at low Te (holds for the bundled fit, 12.78 > 11.5);
    see the module docstring for scope and caveats."""
    if exc.Ea_eV < d_eps:
        raise ValueError("detailed-balance fit diverges as Te -> 0: "
                         f"Ea = {exc.Ea_eV} < d_eps = {d_eps}")
    return RateFit(g_ratio * exc.A, exc.b, exc.Ea_eV - d_eps,
                   label=f"detailed balance from [{exc.label}]")


#: Illustrative fits (module docstring; test_electron_energy.py heritage).
FIT_ELASTIC = RateFit(1.0e-13, 0.0, 0.0, "illustrative constant k_mom")
FIT_EXCITATION = RateFit(2.48e-14, 0.33, 12.78, "illustrative Ar exc 11.5 eV")
FIT_IONIZATION = RateFit(2.34e-14, 0.59, 17.44, "illustrative Ar iz 15.8 eV")
FIT_STEPWISE = RateFit(6.8e-15, 0.67, 4.20, "illustrative Ar* iz 4.24 eV")
FIT_SUPERELASTIC = superelastic_from_excitation(FIT_EXCITATION)


# ----------------------------------------------------------------------------
# Reactions and the Table 2 set
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Reaction:
    """One reaction of doc Table 2 as data.

    nu_p / nu_pp : reactant / product stoichiometric coefficients nu',
                   nu'' over (e, Ar, Ar*, Ar+) -- Eq. 41 uses nu' as the
                   rate exponents and (nu'' - nu') as source weights.
    d_eps_eV     : electron energy loss per event for the Eq. 12 sum
                   (positive = loss, negative = superelastic gain);
                   None excludes the reaction from the sum (elastic --
                   handled by the explicit exchange term -- and
                   heavy-particle channels).
    rate         : callable k(Te) [m^3/s] for electron-impact channels
                   (EETM-supplied fit or Boltzmann table), or a float
                   constant for heavy-particle channels.
    """
    name: str
    nu_p: tuple[int, int, int, int]
    nu_pp: tuple[int, int, int, int]
    rate: Callable[[np.ndarray], np.ndarray] | float
    d_eps_eV: float | None = None

    @property
    def dnu(self) -> tuple[int, ...]:
        return tuple(pp - p for p, pp in zip(self.nu_p, self.nu_pp))

    def k(self, Te):
        """Evaluate k(Te); constants broadcast against Te."""
        if callable(self.rate):
            return self.rate(Te)
        return self.rate * _xp(Te).ones_like(Te)


def argon_table2(rates: Mapping[str, Callable | float] | None = None
                 ) -> "ChemistrySet":
    """The doc Table 2 argon benchmark set, reactions 1-7 in order.

    `rates` overrides any subset by name -- the Boltzmann-table
    replacement path: pass {"excitation": table_fn, ...} built from
    BOLSIG output and nothing else changes.
    """
    r = {
        "elastic": FIT_ELASTIC,
        "excitation": FIT_EXCITATION,
        "superelastic": FIT_SUPERELASTIC,
        "ionization": FIT_IONIZATION,
        "stepwise": FIT_STEPWISE,
        "penning": K_PENNING,
        "quench": K_QUENCH,
    }
    if rates:
        unknown = set(rates) - set(r)
        if unknown:
            raise KeyError(f"unknown reaction name(s): {sorted(unknown)}")
        r.update(rates)

    #                 name            nu'(e,Ar,Ar*,Ar+)  nu''            rate              d_eps
    table = [
        Reaction("elastic",       (1, 1, 0, 0), (1, 1, 0, 0), r["elastic"],      None),
        Reaction("excitation",    (1, 1, 0, 0), (1, 0, 1, 0), r["excitation"],   +EPS_EXC),
        Reaction("superelastic",  (1, 0, 1, 0), (1, 1, 0, 0), r["superelastic"], -EPS_EXC),
        Reaction("ionization",    (1, 1, 0, 0), (2, 0, 0, 1), r["ionization"],   +EPS_IZ),
        Reaction("stepwise",      (1, 0, 1, 0), (2, 0, 0, 1), r["stepwise"],     +EPS_STEP),
        Reaction("penning",       (0, 0, 2, 0), (1, 1, 0, 1), r["penning"],      None),
        Reaction("quench",        (0, 1, 1, 0), (0, 2, 0, 0), r["quench"],       None),
    ]
    return ChemistrySet(tuple(table))


# ----------------------------------------------------------------------------
# Generic source assembly (doc Eq. 41) and energy-loss export (doc Eq. 12)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChemistrySet:
    """An ordered reaction set with the generic Eq. 41 assembly. Nothing
    here is argon-specific beyond the 4-species ordering; extending the
    species tuple (electronegative chemistry) reuses the same machinery.
    """
    reactions: tuple[Reaction, ...]

    def rates(self, n, Te):
        """Reaction rates r_j = k_j prod n_l^nu' (Eq. 41 second relation).

        n  : density stack, shape (NSPECIES, ...) [1/m^3]
        Te : electron temperature [eV], broadcastable to n[0]
        returns list of J arrays [m^-3 s^-1] (a list, not a stacked
        array, so it also serves as the traced-jnp path).
        """
        out = []
        for rx in self.reactions:
            r = rx.k(Te)
            for l, p in enumerate(rx.nu_p):
                if p == 1:
                    r = r * n[l]
                elif p > 1:
                    r = r * n[l] ** p
            out.append(r)
        return out

    def sources(self, n, Te):
        """Volumetric sources S_i = sum_j (nu'' - nu') r_j (Eq. 41),
        returned as a stack shaped like n [m^-3 s^-1]."""
        r = self.rates(n, Te)
        S = [0.0 * n[i] for i in range(len(n))]
        for rx, rj in zip(self.reactions, r):
            for i, d in enumerate(rx.dnu):
                if d != 0:
                    S[i] = S[i] + d * rj
        xp = _xp(n[0])
        return xp.stack(S)

    def electron_energy_loss(self, n, Te):
        """Inelastic electron energy loss sum_j r_j d_eps_j
        [eV m^-3 s^-1] (doc Eq. 12) -- the quantity subtracted in
        electron_energy's source. Superelastic enters with d_eps < 0
        (net electron heating); reactions with d_eps = None (elastic,
        heavy-particle) are excluded by construction."""
        loss = 0.0 * n[IE]
        for rx, rj in zip(self.reactions, self.rates(n, Te)):
            if rx.d_eps_eV is not None:
                loss = loss + rx.d_eps_eV * rj
        return loss

    def make_inelastic(self, n_Ar, n_Ars):
        """Close over frozen heavy densities and return
        inelastic(ne, Te) -> eV m^-3 s^-1 in exactly the
        `electron_energy.electron_rhs(..., inelastic=...)` signature.

        The heavy state is frozen for the EETM sub-slice (doc Sec. 12.3
        cadence: neutrals evolve ~1e3x slower); rebuild the closure each
        outer coupling iteration with the updated Ar/Ar* fields. Ion
        density does not enter the Eq. 12 sum for Table 2 (no
        electron-impact channel consumes Ar+).
        """
        def inelastic(ne, Te):
            zero = 0.0 * ne
            n = (ne, n_Ar, n_Ars, zero)
            return self.electron_energy_loss(n, Te)
        return inelastic


# ----------------------------------------------------------------------------
# JAX boundary
# ----------------------------------------------------------------------------

def jax_rate_callables(chem: ChemistrySet) -> list:
    """Per-reaction jnp-traceable k(Te) callables: `RateFit`s are
    re-expressed in jnp ops, constants become broadcasting closures, and
    other callables (Boltzmann tables) pass through -- they must already
    be jnp-traceable (documented interface contract)."""
    import jax.numpy as jnp

    def from_fit(f: RateFit):
        return lambda Te: f.A * Te ** f.b * jnp.exp(-f.Ea_eV / Te)

    def from_const(c: float):
        return lambda Te: c * jnp.ones_like(Te)

    out = []
    for rx in chem.reactions:
        if isinstance(rx.rate, RateFit):
            out.append(from_fit(rx.rate))
        elif callable(rx.rate):
            out.append(rx.rate)
        else:
            out.append(from_const(float(rx.rate)))
    return out


def make_jax_source_fn(chem: ChemistrySet, species: int | None = None):
    """Return a jnp source function for jitted steppers:

        f(n, Te) -> S              (stack (NSPECIES, ...) if species is
                                    None, else the single S_i array)

    The reaction loop is a static Python loop over Table 2 that unrolls
    under jit; stoichiometry is baked in as compile-time constants.
    """
    import jax.numpy as jnp

    ks = jax_rate_callables(chem)

    def f(n, Te):
        S = [jnp.zeros_like(n[0]) for _ in range(NSPECIES)]
        for rx, k in zip(chem.reactions, ks):
            if species is not None and rx.dnu[species] == 0:
                continue
            r = k(Te)
            for l, p in enumerate(rx.nu_p):
                if p == 1:
                    r = r * n[l]
                elif p > 1:
                    r = r * n[l] ** p
            for i, d in enumerate(rx.dnu):
                if d != 0:
                    S[i] = S[i] + d * r
        if species is not None:
            return S[species]
        return jnp.stack(S)

    return f


def make_jax_inelastic_fn(chem: ChemistrySet, n_Ar, n_Ars):
    """jnp counterpart of `make_inelastic`, for
    `make_jax_energy_stepper(..., inelastic_fn=...)`: heavy densities are
    closed over as jnp constants for the sub-slice; the returned
    inelastic_fn(ne, Te) is jit/vmap-safe. Rebuild each outer coupling
    iteration (cheap: closure construction only, no retrace if shapes
    are static and the arrays are donated as captured constants --
    retracing is acceptable at the outer cadence regardless)."""
    import jax.numpy as jnp

    ks = jax_rate_callables(chem)
    n_Ar = jnp.asarray(n_Ar)
    n_Ars = jnp.asarray(n_Ars)

    def inelastic_fn(ne, Te):
        n = (ne, n_Ar, n_Ars, jnp.zeros_like(ne))
        loss = jnp.zeros_like(ne)
        for rx, k in zip(chem.reactions, ks):
            if rx.d_eps_eV is None:
                continue
            r = k(Te)
            for l, p in enumerate(rx.nu_p):
                if p == 1:
                    r = r * n[l]
                elif p > 1:
                    r = r * n[l] ** p
            loss = loss + rx.d_eps_eV * r
        return loss

    return inelastic_fn