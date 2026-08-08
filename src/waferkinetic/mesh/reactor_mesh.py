"""
reactor_mesh.py
================
Geometry definition and structured rectilinear variable-mesh generation for a
2D axisymmetric (r, z) plasma fluid solver.

Conventions
-----------
- Coordinates: r in [0, r_max] (axis of symmetry at r = 0), z in [z_min, z_max].
- All 2D arrays use shape (Nr, Nz), i.e. index [i, j] -> (r_i, z_j)
  (meshgrid indexing='ij'). Transpose only at plot time.
- "Faces" are the Nr+1 / Nz+1 node coordinates of the rectilinear grid;
  "centers" are the Nr / Nz cell-center coordinates.
- Everything is plain NumPy and fully vectorized; the arrays are trivially
  convertible to jax.numpy (jnp.asarray) since only static construction
  happens here. The mask/metric arrays are what the JAX solver consumes.

Units: SI (meters) throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Literal, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Part 1: Geometry definition
# ----------------------------------------------------------------------------

class Material(IntEnum):
    """Material tags. IntEnum so the mask is a plain integer array."""
    PLASMA = 0
    DIELECTRIC = 1
    GROUNDED_METAL = 2
    RF_ANTENNA = 3
    WAFER = 4
    AIR = 5


@dataclass(frozen=True)
class RectRegion:
    """Axis-aligned rectangular sub-region in the (r, z) plane.

    In 3D this is an annular/cylindrical solid of revolution:
    r in [r_start, r_end], z in [z_start, z_end].
    """
    r_start: float
    r_end: float
    z_start: float
    z_end: float
    material: Material
    name: str = ""

    def __post_init__(self) -> None:
        if self.r_end <= self.r_start:
            raise ValueError(f"RectRegion '{self.name}': r_end <= r_start")
        if self.z_end <= self.z_start:
            raise ValueError(f"RectRegion '{self.name}': z_end <= z_start")
        if self.r_start < 0.0:
            raise ValueError(f"RectRegion '{self.name}': r_start < 0")

    def contains(self, rc: np.ndarray, zc: np.ndarray) -> np.ndarray:
        """Vectorized point-in-region test on broadcastable coordinate arrays."""
        return (
            (rc >= self.r_start) & (rc < self.r_end)
            & (zc >= self.z_start) & (zc < self.z_end)
        )


@dataclass
class ReactorGeometry:
    """Global bounding box, default-filled with PLASMA, plus a stack of
    material regions applied in insertion order (painter's algorithm:
    later regions overwrite earlier ones).
    """
    r_max: float
    z_min: float
    z_max: float
    regions: list[RectRegion] = field(default_factory=list)

    def add_region(self, region: RectRegion) -> "ReactorGeometry":
        # Clip-check against the bounding box (allow exact touching).
        eps = 1e-12
        if (region.r_end > self.r_max + eps
                or region.z_start < self.z_min - eps
                or region.z_end > self.z_max + eps):
            raise ValueError(
                f"Region '{region.name}' exceeds the global bounding box."
            )
        self.regions.append(region)
        return self  # allow chaining

    def material_mask(self, rc_2d: np.ndarray, zc_2d: np.ndarray) -> np.ndarray:
        """Assign a Material integer to every cell center.

        Parameters
        ----------
        rc_2d, zc_2d : (Nr, Nz) arrays of cell-center coordinates.

        Returns
        -------
        mask : (Nr, Nz) int array of Material values.

        Notes
        -----
        Fully vectorized over cells; the outer loop is only over the
        (handful of) regions, which is the natural painter's-algorithm order.
        """
        mask = np.full(rc_2d.shape, int(Material.PLASMA), dtype=np.int32)
        for region in self.regions:
            inside = region.contains(rc_2d, zc_2d)
            mask[inside] = int(region.material)
        return mask


# ----------------------------------------------------------------------------
# Part 2: Variable (stretched) 1D grid generation
# ----------------------------------------------------------------------------

ClusterMode = Literal["start", "end", "both", "interior"]


def tanh_grid(x0: float, x1: float, n_cells: int,
              beta: float = 2.0, cluster: ClusterMode = "both") -> np.ndarray:
    """Hyperbolic-tangent stretched node distribution on [x0, x1].

    Parameters
    ----------
    n_cells : number of cells (returns n_cells + 1 nodes).
    beta    : stretching strength. beta -> 0 gives a uniform grid;
              beta ~ 2-3 is typical; larger packs cells harder at the
              clustered end(s).
    cluster : where the fine spacing goes:
              'start'    -> fine at x0
              'end'      -> fine at x1
              'both'     -> fine at both ends, coarse in the middle
              'interior' -> fine in the middle, coarse at the ends

    Returns
    -------
    (n_cells + 1,) monotone array of node coordinates, exact at endpoints.
    """
    s = np.linspace(0.0, 1.0, n_cells + 1)   # uniform computational coord
    if beta <= 1e-12:
        xi = s
    elif cluster == "both":
        # Symmetric two-sided clustering (Vinokur-style).
        xi = 0.5 * (1.0 + np.tanh(beta * (2.0 * s - 1.0)) / np.tanh(beta))
    elif cluster == "start":
        xi = 1.0 + np.tanh(beta * (s - 1.0)) / np.tanh(beta)
    elif cluster == "end":
        xi = np.tanh(beta * s) / np.tanh(beta)
    elif cluster == "interior":
        # Inverse of the two-sided map: coarse ends, fine middle.
        xi = 0.5 * (1.0 + np.arctanh((2.0 * s - 1.0) * np.tanh(beta)) / beta)
    else:
        raise ValueError(f"Unknown cluster mode: {cluster!r}")
    # Guard against tiny FP drift at the endpoints.
    xi[0], xi[-1] = 0.0, 1.0
    return x0 + (x1 - x0) * xi


def geometric_grid(x0: float, x1: float, dx_min: float,
                   ratio: float = 1.15,
                   cluster: Literal["start", "end"] = "start") -> np.ndarray:
    """Geometric-progression grid: first spacing dx_min at the clustered end,
    each subsequent cell grows by `ratio` until [x0, x1] is covered; the last
    cell is adjusted (the whole grid is rescaled) so the endpoint is exact.

    Number of cells is derived from (dx_min, ratio, length), so the caller
    controls resolution physically rather than by cell count.
    """
    L = x1 - x0
    if dx_min >= L:
        return np.array([x0, x1])
    if abs(ratio - 1.0) < 1e-12:
        n = max(1, int(np.ceil(L / dx_min)))
        return np.linspace(x0, x1, n + 1)
    # Solve dx_min * (ratio^n - 1)/(ratio - 1) >= L for integer n.
    n = int(np.ceil(np.log1p(L * (ratio - 1.0) / dx_min) / np.log(ratio)))
    widths = dx_min * ratio ** np.arange(n)
    widths *= L / widths.sum()               # rescale for exact endpoint
    nodes = x0 + np.concatenate(([0.0], np.cumsum(widths)))
    nodes[-1] = x1
    if cluster == "end":
        nodes = x0 + x1 - nodes[::-1]
    return nodes


def composite_grid(segments: Sequence[np.ndarray], tol: float = 1e-12) -> np.ndarray:
    """Concatenate several 1D node arrays that share endpoints
    (segment k ends where segment k+1 starts) into one grid.

    Useful for aligning grid nodes exactly with material interfaces
    (e.g. the wafer surface or the window face), which every downstream
    field solver will thank you for.
    """
    out = [np.asarray(segments[0], dtype=float)]
    for seg in segments[1:]:
        seg = np.asarray(seg, dtype=float)
        if abs(seg[0] - out[-1][-1]) > tol * max(1.0, abs(seg[0])):
            raise ValueError("composite_grid: segments do not share endpoints")
        out.append(seg[1:])
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# Part 2/3: Mesh with axisymmetric metrics + mask
# ----------------------------------------------------------------------------

class Mesh2D:
    """Structured rectilinear axisymmetric mesh built from 1D node arrays.

    Attributes (all NumPy; shapes noted)
    ------------------------------------
    r_faces : (Nr+1,)      radial node coordinates
    z_faces : (Nz+1,)      axial node coordinates
    r_c     : (Nr,)        radial cell centers (arithmetic mean of faces)
    z_c     : (Nz,)        axial cell centers
    dr      : (Nr,)        radial cell widths
    dz      : (Nz,)        axial cell heights
    RC, ZC  : (Nr, Nz)     2D cell-center coordinates
    DR, DZ  : (Nr, Nz)     2D cell widths/heights
    volume  : (Nr, Nz)     toroidal cell volumes, V = 2*pi*r_c*dr*dz
                           (== pi*(r_{i+1}^2 - r_i^2)*dz exactly, since r_c is
                           the arithmetic mean of the faces)
    area_r  : (Nr+1, Nz)   radial face areas  A_r = 2*pi*r_face*dz
                           (area_r[0, :] = 0 on the symmetry axis)
    area_z  : (Nr, Nz+1)   axial face areas   A_z = 2*pi*r_c*dr
                           (== pi*(r_{i+1}^2 - r_i^2), the true annulus area)
    """

    def __init__(self, r_faces: np.ndarray, z_faces: np.ndarray):
        r_faces = np.asarray(r_faces, dtype=float)
        z_faces = np.asarray(z_faces, dtype=float)
        if r_faces.ndim != 1 or z_faces.ndim != 1:
            raise ValueError("r_faces and z_faces must be 1D node arrays")
        if np.any(np.diff(r_faces) <= 0) or np.any(np.diff(z_faces) <= 0):
            raise ValueError("Node arrays must be strictly increasing")
        if abs(r_faces[0]) > 1e-14:
            raise ValueError("Axisymmetric mesh requires r_faces[0] == 0")

        self.r_faces, self.z_faces = r_faces, z_faces
        self.Nr, self.Nz = r_faces.size - 1, z_faces.size - 1

        # 1D centers and widths
        self.r_c = 0.5 * (r_faces[1:] + r_faces[:-1])
        self.z_c = 0.5 * (z_faces[1:] + z_faces[:-1])
        self.dr = np.diff(r_faces)
        self.dz = np.diff(z_faces)

        # 2D center/width arrays, indexing='ij' -> shape (Nr, Nz)
        self.RC, self.ZC = np.meshgrid(self.r_c, self.z_c, indexing="ij")
        self.DR, self.DZ = np.meshgrid(self.dr, self.dz, indexing="ij")

        # --- Axisymmetric metrics (vectorized via broadcasting) -------------
        # Cell volumes: 2*pi*r_c*dr*dz. Because r_c = (r_i + r_{i+1})/2,
        # 2*pi*r_c*dr == pi*(r_{i+1}^2 - r_i^2), so this is the *exact*
        # volume of the annular cell, not a midpoint approximation.
        self.volume = 2.0 * np.pi * self.RC * self.DR * self.DZ

        # Radial face areas: cylinder side-wall at each r-face, per z-cell.
        # Shape (Nr+1, Nz): area_r[i, j] = 2*pi*r_faces[i]*dz[j]
        self.area_r = 2.0 * np.pi * r_faces[:, None] * self.dz[None, :]

        # Axial face areas: annulus at each z-face, per r-cell.
        # Shape (Nr, Nz+1): independent of z, broadcast across z-faces.
        annulus = np.pi * (r_faces[1:] ** 2 - r_faces[:-1] ** 2)  # (Nr,)
        self.area_z = np.broadcast_to(
            annulus[:, None], (self.Nr, self.Nz + 1)
        ).copy()

    # ------------------------------------------------------------------ mask
    def material_mask(self, geometry: ReactorGeometry) -> np.ndarray:
        """(Nr, Nz) integer Material mask evaluated at cell centers."""
        return geometry.material_mask(self.RC, self.ZC)

    # sanity ----------------------------------------------------------------
    def total_volume(self) -> float:
        return float(self.volume.sum())

    def __repr__(self) -> str:
        return (f"Mesh2D(Nr={self.Nr}, Nz={self.Nz}, "
                f"r=[0, {self.r_faces[-1]:.4g}] m, "
                f"z=[{self.z_faces[0]:.4g}, {self.z_faces[-1]:.4g}] m, "
                f"dr=[{self.dr.min():.3g}, {self.dr.max():.3g}], "
                f"dz=[{self.dz.min():.3g}, {self.dz.max():.3g}])")


# ----------------------------------------------------------------------------
# Part 4: Visualization
# ----------------------------------------------------------------------------

def plot_reactor(mesh: Mesh2D, mask: np.ndarray,
                 title: str = "Reactor geometry & mesh",
                 show_grid: bool = True,
                 ax=None):
    """Color-coded material map with the (stretched) grid lines overlaid.

    r is drawn on the horizontal axis, z on the vertical axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    colors = {
        Material.PLASMA:         "#dce9f5",  # pale blue
        Material.DIELECTRIC:     "#d9a648",  # amber
        Material.GROUNDED_METAL: "#8a8f98",  # grey
        Material.RF_ANTENNA:     "#c0562f",  # copper
        Material.WAFER:          "#3f6d44",  # green
        Material.AIR:            "#fbfbf8",  # off-white
    }
    mats = list(Material)
    cmap = ListedColormap([colors[m] for m in mats])
    norm = BoundaryNorm([m - 0.5 for m in mats] + [mats[-1] + 0.5], cmap.N)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 8.5))

    # pcolormesh wants C with shape (len(y)-1, len(x)-1) -> transpose (Nr,Nz)
    ax.pcolormesh(mesh.r_faces, mesh.z_faces, mask.T, cmap=cmap, norm=norm,
                  shading="flat")

    if show_grid:
        lw, gc = 0.35, (0, 0, 0, 0.35)
        ax.vlines(mesh.r_faces, mesh.z_faces[0], mesh.z_faces[-1],
                  colors=[gc], lw=lw)
        ax.hlines(mesh.z_faces, mesh.r_faces[0], mesh.r_faces[-1],
                  colors=[gc], lw=lw)

    present = np.unique(mask)
    ax.legend(handles=[Patch(facecolor=colors[Material(m)],
                             edgecolor="k", label=Material(m).name)
                       for m in present],
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.figure.tight_layout()
    return ax