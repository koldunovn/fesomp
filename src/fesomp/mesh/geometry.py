"""Mesh geometry data structures and computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fesomp.mesh.topology import Topology

# Earth radius in meters (WGS84 mean radius)
EARTH_RADIUS_M = 6371000.0


@dataclass
class Geometry:
    """
    Mesh geometry information (areas, gradients).

    Attributes
    ----------
    elem_area : np.ndarray
        Area of each element in m^2, shape (nelem,).
    node_area : np.ndarray
        Area associated with each node at each level, shape (nlev, n2d).
    gradient_sca : tuple[np.ndarray, np.ndarray] | None
        Gradient operator for scalar fields (x, y components).
    gradient_vec : tuple[np.ndarray, np.ndarray] | None
        Gradient operator for vector fields (x, y components).
    edge_cross_dxdy : np.ndarray | None
        Edge crossing distances, shape (4, nedges).
    """

    elem_area: np.ndarray
    node_area: np.ndarray
    gradient_sca: tuple[np.ndarray, np.ndarray] | None = None
    gradient_vec: tuple[np.ndarray, np.ndarray] | None = None
    edge_cross_dxdy: np.ndarray | None = None


def spherical_triangle_area(
    lon: np.ndarray, lat: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """
    Compute the spherical area of triangles on a sphere.

    Uses the spherical excess formula for computing the area of spherical
    triangles. The formula is:
        Area = R^2 * E
    where E is the spherical excess (sum of angles - π).

    Parameters
    ----------
    lon : np.ndarray
        Longitude of nodes in degrees, shape (n2d,).
    lat : np.ndarray
        Latitude of nodes in degrees, shape (n2d,).
    triangles : np.ndarray
        Triangle connectivity, shape (nelem, 3), 0-indexed.

    Returns
    -------
    np.ndarray
        Area of each triangle in m^2, shape (nelem,).
    """
    # Convert to radians
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    # Get coordinates of triangle vertices
    # Shape: (nelem, 3)
    tri_lon = lon_rad[triangles]
    tri_lat = lat_rad[triangles]

    # Convert to Cartesian coordinates on unit sphere
    # Shape: (nelem, 3) for each of x, y, z
    cos_lat = np.cos(tri_lat)
    x = cos_lat * np.cos(tri_lon)
    y = cos_lat * np.sin(tri_lon)
    z = np.sin(tri_lat)

    # Stack into vectors for each vertex
    # v[i] has shape (nelem, 3) where the 3 is x, y, z
    v0 = np.stack([x[:, 0], y[:, 0], z[:, 0]], axis=1)
    v1 = np.stack([x[:, 1], y[:, 1], z[:, 1]], axis=1)
    v2 = np.stack([x[:, 2], y[:, 2], z[:, 2]], axis=1)

    # Compute spherical angles using the formula:
    # angle at vertex i = arccos of (cross products dotted)
    # This uses L'Huilier's formula through the tangent half-angle approach

    # Compute arc lengths (central angles) between vertices
    # Using dot product: cos(arc) = v1 · v2
    a = np.arccos(np.clip(np.sum(v1 * v2, axis=1), -1, 1))  # opposite to v0
    b = np.arccos(np.clip(np.sum(v0 * v2, axis=1), -1, 1))  # opposite to v1
    c = np.arccos(np.clip(np.sum(v0 * v1, axis=1), -1, 1))  # opposite to v2

    # Semi-perimeter
    s = (a + b + c) / 2

    # L'Huilier's formula for spherical excess
    # tan(E/4) = sqrt(tan(s/2) * tan((s-a)/2) * tan((s-b)/2) * tan((s-c)/2))
    # Handle numerical issues with small triangles
    eps = 1e-15
    tan_s2 = np.tan(s / 2)
    tan_sa2 = np.tan((s - a) / 2)
    tan_sb2 = np.tan((s - b) / 2)
    tan_sc2 = np.tan((s - c) / 2)

    # Ensure all terms are positive (numerical stability)
    product = np.maximum(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2, eps)
    tan_E4 = np.sqrt(product)

    # Spherical excess
    E = 4 * np.arctan(tan_E4)

    # Area = R^2 * E
    area = EARTH_RADIUS_M**2 * E

    return area


def compute_node_area(
    elem_area: np.ndarray,
    triangles: np.ndarray,
    node_levels: np.ndarray,
    nlev: int,
) -> np.ndarray:
    """
    Compute the area associated with each node at each level.

    Each node receives 1/3 of the area of each adjacent element.
    For nodes with fewer levels, deeper levels have zero area.

    Parameters
    ----------
    elem_area : np.ndarray
        Area of each element, shape (nelem,).
    triangles : np.ndarray
        Triangle connectivity, shape (nelem, 3), 0-indexed.
    node_levels : np.ndarray
        Number of active levels at each node, shape (n2d,).
    nlev : int
        Total number of vertical levels.

    Returns
    -------
    np.ndarray
        Area at each node and level, shape (nlev, n2d).
    """
    n2d = len(node_levels)

    # First compute surface area for each node (sum of 1/3 of adjacent elements)
    node_area_surface = np.zeros(n2d, dtype=np.float64)
    np.add.at(node_area_surface, triangles[:, 0], elem_area / 3)
    np.add.at(node_area_surface, triangles[:, 1], elem_area / 3)
    np.add.at(node_area_surface, triangles[:, 2], elem_area / 3)

    # Create 3D node area array
    node_area = np.zeros((nlev, n2d), dtype=np.float64)

    # For each level, nodes are active if their node_levels >= level+1
    for lev in range(nlev):
        active_mask = node_levels > lev
        node_area[lev, active_mask] = node_area_surface[active_mask]

    return node_area


def compute_geometry(
    lon: np.ndarray,
    lat: np.ndarray,
    triangles: np.ndarray,
    node_levels: np.ndarray,
    nlev: int,
    topology: Topology,
) -> Geometry:
    """
    Compute mesh geometry from coordinates and connectivity.

    Parameters
    ----------
    lon : np.ndarray
        Longitude of nodes in degrees, shape (n2d,).
    lat : np.ndarray
        Latitude of nodes in degrees, shape (n2d,).
    triangles : np.ndarray
        Triangle connectivity, shape (nelem, 3), 0-indexed.
    node_levels : np.ndarray
        Number of active levels at each node, shape (n2d,).
    nlev : int
        Total number of vertical levels.
    topology : Topology
        Pre-computed topology.

    Returns
    -------
    Geometry
        Computed geometry object.
    """
    # Compute element areas using spherical formula
    elem_area = spherical_triangle_area(lon, lat, triangles)

    # Compute node areas
    node_area = compute_node_area(elem_area, triangles, node_levels, nlev)

    return Geometry(
        elem_area=elem_area,
        node_area=node_area,
        gradient_sca=None,
        gradient_vec=None,
        edge_cross_dxdy=None,
    )
