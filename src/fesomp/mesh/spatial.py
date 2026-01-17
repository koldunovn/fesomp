"""Spatial indexing for efficient point queries on mesh data."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def lonlat_to_cartesian(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Convert longitude/latitude to 3D Cartesian coordinates on unit sphere.

    Parameters
    ----------
    lon : np.ndarray
        Longitude in degrees.
    lat : np.ndarray
        Latitude in degrees.

    Returns
    -------
    np.ndarray
        Cartesian coordinates, shape (..., 3).
    """
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.stack([x, y, z], axis=-1)


def chord_to_arc_distance(chord: float, radius: float = EARTH_RADIUS_KM) -> float:
    """
    Convert chord distance to arc (great-circle) distance.

    Parameters
    ----------
    chord : float
        Chord distance on unit sphere.
    radius : float
        Sphere radius (default: Earth radius in km).

    Returns
    -------
    float
        Arc distance in same units as radius.
    """
    # chord = 2 * sin(theta/2) where theta is the central angle
    # arc = radius * theta
    half_angle = np.arcsin(np.clip(chord / 2, -1, 1))
    return 2 * radius * half_angle


def arc_to_chord_distance(arc_km: float, radius: float = EARTH_RADIUS_KM) -> float:
    """
    Convert arc (great-circle) distance to chord distance on unit sphere.

    Parameters
    ----------
    arc_km : float
        Arc distance in kilometers.
    radius : float
        Sphere radius (default: Earth radius in km).

    Returns
    -------
    float
        Chord distance on unit sphere.
    """
    # arc = radius * theta
    # chord = 2 * sin(theta/2)
    theta = arc_km / radius
    return 2 * np.sin(theta / 2)


class SpatialIndex:
    """
    Spatial index for efficient nearest-neighbor queries on mesh nodes.

    Uses a 3D KD-tree in Cartesian coordinates for accurate spherical queries.

    Parameters
    ----------
    lon : np.ndarray
        Longitude of nodes in degrees, shape (n2d,).
    lat : np.ndarray
        Latitude of nodes in degrees, shape (n2d,).
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray) -> None:
        self.lon = lon
        self.lat = lat
        self._coords = lonlat_to_cartesian(lon, lat)
        self._tree = cKDTree(self._coords)

    def find_nearest(
        self, lon: float | np.ndarray, lat: float | np.ndarray, k: int = 1
    ) -> np.ndarray:
        """
        Find the k nearest nodes to given point(s).

        Parameters
        ----------
        lon : float or np.ndarray
            Longitude in degrees.
        lat : float or np.ndarray
            Latitude in degrees.
        k : int, optional
            Number of nearest neighbors to return.

        Returns
        -------
        np.ndarray
            Indices of the k nearest nodes.
            If single point, shape is (k,) for k>1 or scalar for k=1.
            If multiple points, shape is (npoints, k).
        """
        query_coords = lonlat_to_cartesian(np.asarray(lon), np.asarray(lat))
        _, indices = self._tree.query(query_coords, k=k)
        return np.asarray(indices, dtype=np.int32)

    def find_in_radius(
        self, lon: float | np.ndarray, lat: float | np.ndarray, radius_km: float
    ) -> np.ndarray | list[np.ndarray]:
        """
        Find all nodes within a given radius of point(s).

        Parameters
        ----------
        lon : float or np.ndarray
            Longitude in degrees.
        lat : float or np.ndarray
            Latitude in degrees.
        radius_km : float
            Search radius in kilometers.

        Returns
        -------
        np.ndarray or list[np.ndarray]
            Indices of nodes within the radius.
            For single point: 1D array of indices.
            For multiple points: list of arrays.
        """
        # Convert radius to chord distance on unit sphere
        chord_radius = arc_to_chord_distance(radius_km)

        query_coords = lonlat_to_cartesian(np.asarray(lon), np.asarray(lat))
        results = self._tree.query_ball_point(query_coords, chord_radius)

        if np.isscalar(lon):
            return np.array(results, dtype=np.int32)
        else:
            return [np.array(r, dtype=np.int32) for r in results]

    def find_containing_element(
        self,
        lon: float,
        lat: float,
        triangles: np.ndarray,
        mesh_lon: np.ndarray,
        mesh_lat: np.ndarray,
    ) -> int:
        """
        Find the element containing a given point.

        Uses nearest-neighbor search followed by local search of adjacent elements.

        Parameters
        ----------
        lon : float
            Longitude in degrees.
        lat : float
            Latitude in degrees.
        triangles : np.ndarray
            Triangle connectivity, shape (nelem, 3).
        mesh_lon : np.ndarray
            Longitude of mesh nodes.
        mesh_lat : np.ndarray
            Latitude of mesh nodes.

        Returns
        -------
        int
            Index of the containing element, or -1 if not found.
        """
        # Find nearest node
        nearest_node = self.find_nearest(lon, lat, k=1)
        if np.isscalar(nearest_node):
            nearest_node = int(nearest_node)
        else:
            nearest_node = int(nearest_node[0])

        # Find elements containing this node
        # This requires node_elements from topology, so we search by checking
        # which triangles contain the nearest node
        candidate_elems = np.nonzero(np.any(triangles == nearest_node, axis=1))[0]

        # Check each candidate element
        for elem_idx in candidate_elems:
            if _point_in_triangle_spherical(
                lon,
                lat,
                mesh_lon[triangles[elem_idx]],
                mesh_lat[triangles[elem_idx]],
            ):
                return int(elem_idx)

        return -1


def _point_in_triangle_spherical(
    lon: float, lat: float, tri_lon: np.ndarray, tri_lat: np.ndarray
) -> bool:
    """
    Check if a point is inside a spherical triangle.

    Uses barycentric coordinates computed from cross products.

    Parameters
    ----------
    lon, lat : float
        Point coordinates in degrees.
    tri_lon, tri_lat : np.ndarray
        Triangle vertex coordinates in degrees, shape (3,).

    Returns
    -------
    bool
        True if point is inside the triangle.
    """
    # Convert to Cartesian
    p = lonlat_to_cartesian(lon, lat)
    v = lonlat_to_cartesian(tri_lon, tri_lat)  # Shape (3, 3)

    # Check if point is on same side of all edges
    # Using sign of scalar triple product
    def sign(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return np.dot(np.cross(b - a, c - a), a)

    s1 = sign(v[0], v[1], p)
    s2 = sign(v[1], v[2], p)
    s3 = sign(v[2], v[0], p)

    # All same sign (or zero) means inside
    has_neg = (s1 < 0) or (s2 < 0) or (s3 < 0)
    has_pos = (s1 > 0) or (s2 > 0) or (s3 > 0)

    return not (has_neg and has_pos)
