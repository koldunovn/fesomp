"""Interpolation from unstructured to regular grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree


def create_regular_grid(
    box: tuple[float, float, float, float] = (-180, 180, -90, 90),
    res: tuple[int, int] = (360, 180),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a regular lon/lat grid.

    Parameters
    ----------
    box : tuple
        Bounding box as (lon_min, lon_max, lat_min, lat_max).
    res : tuple
        Resolution as (nlon, nlat).

    Returns
    -------
    lon1d : np.ndarray
        1D array of longitudes.
    lat1d : np.ndarray
        1D array of latitudes.
    lon2d : np.ndarray
        2D meshgrid of longitudes.
    lat2d : np.ndarray
        2D meshgrid of latitudes.
    """
    lon_min, lon_max, lat_min, lat_max = box
    nlon, nlat = res

    lon1d = np.linspace(lon_min, lon_max, nlon)
    lat1d = np.linspace(lat_min, lat_max, nlat)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)

    return lon1d, lat1d, lon2d, lat2d


def _lonlat_to_cartesian(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convert lon/lat (degrees) to 3D Cartesian on unit sphere."""
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])


def _meters_to_chord(meters: float, earth_radius: float = 6371000.0) -> float:
    """Convert distance in meters to chord distance on unit sphere."""
    theta = meters / earth_radius
    return 2 * np.sin(theta / 2)


@dataclass
class RegridInterpolator:
    """
    Cached interpolator for regridding unstructured data.

    This class pre-computes and caches the KDTree and interpolation
    indices/weights, allowing fast repeated interpolation of different
    variables on the same grid.

    Parameters
    ----------
    lon : np.ndarray
        Longitudes of source points in degrees.
    lat : np.ndarray
        Latitudes of source points in degrees.
    box : tuple
        Target bounding box as (lon_min, lon_max, lat_min, lat_max).
    res : tuple
        Target resolution as (nlon, nlat).
    method : str
        Interpolation method: 'nn', 'idw', or 'linear'.
    influence : float
        Radius of influence in meters.
    k : int
        Number of neighbors for IDW interpolation.

    Example
    -------
    >>> # Create interpolator once
    >>> interp = RegridInterpolator(mesh.lon, mesh.lat, box=(-180, 180, -90, 90))
    >>>
    >>> # Use many times for different variables
    >>> temp_reg, lon_reg, lat_reg = interp(ds['temp'].values)
    >>> salt_reg, _, _ = interp(ds['salt'].values)
    """

    lon: np.ndarray
    lat: np.ndarray
    box: tuple[float, float, float, float] = (-180, 180, -90, 90)
    res: tuple[int, int] = (360, 180)
    method: Literal["nn", "idw", "linear"] = "nn"
    influence: float = 80000
    k: int = 10

    def __post_init__(self):
        """Build KDTree and compute interpolation indices."""
        self.lon = np.asarray(self.lon).ravel()
        self.lat = np.asarray(self.lat).ravel()

        # Create output grid
        self.lon_reg, self.lat_reg, self.lon2d, self.lat2d = create_regular_grid(
            self.box, self.res
        )

        # Convert to Cartesian
        self._src_xyz = _lonlat_to_cartesian(self.lon, self.lat)
        self._dst_xyz = _lonlat_to_cartesian(
            self.lon2d.ravel(), self.lat2d.ravel()
        )

        # Build KDTree
        self._tree = cKDTree(self._src_xyz)

        # Convert influence to chord distance
        self._influence_chord = _meters_to_chord(self.influence)

        # Pre-compute query results for nn and idw
        if self.method in ("nn", "idw"):
            k_query = 1 if self.method == "nn" else self.k
            self._distances, self._indices = self._tree.query(
                self._dst_xyz, k=k_query
            )

            # Ensure 2D shape for consistency
            if k_query == 1:
                self._distances = self._distances[:, np.newaxis]
                self._indices = self._indices[:, np.newaxis]

            # Pre-compute validity mask for nearest neighbor
            if self.method == "nn":
                self._valid_mask = self._distances[:, 0] <= self._influence_chord

    def __call__(
        self, data: np.ndarray, fill_value: float = np.nan
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate data to regular grid.

        Parameters
        ----------
        data : np.ndarray
            Data values at source points, shape (npoints,).
        fill_value : float
            Value for points outside influence radius.

        Returns
        -------
        data_reg : np.ndarray
            Interpolated data, shape (nlat, nlon).
        lon_reg : np.ndarray
            1D array of output longitudes.
        lat_reg : np.ndarray
            1D array of output latitudes.
        """
        data = np.asarray(data).ravel()

        if len(data) != len(self.lon):
            raise ValueError(
                f"Data length ({len(data)}) doesn't match source grid ({len(self.lon)})"
            )

        if self.method == "nn":
            result = self._interpolate_nn(data, fill_value)
        elif self.method == "idw":
            result = self._interpolate_idw(data, fill_value)
        elif self.method == "linear":
            result = self._interpolate_linear(data, fill_value)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return result.reshape(self.lon2d.shape), self.lon_reg, self.lat_reg

    def _interpolate_nn(
        self, data: np.ndarray, fill_value: float
    ) -> np.ndarray:
        """Nearest neighbor interpolation using cached indices."""
        result = np.full(len(self._dst_xyz), fill_value, dtype=np.float64)
        result[self._valid_mask] = data[self._indices[self._valid_mask, 0]]
        return result

    def _interpolate_idw(
        self, data: np.ndarray, fill_value: float, power: float = 2.0
    ) -> np.ndarray:
        """Inverse distance weighting using cached indices."""
        result = np.full(len(self._dst_xyz), fill_value, dtype=np.float64)

        for i in range(len(self._dst_xyz)):
            dist = self._distances[i]
            idx = self._indices[i]

            # Only use points within influence radius
            valid = dist <= self._influence_chord
            if not np.any(valid):
                continue

            dist_valid = dist[valid]
            idx_valid = idx[valid]
            data_valid = data[idx_valid]

            # Handle exact match
            if np.any(dist_valid == 0):
                result[i] = data_valid[dist_valid == 0][0]
            else:
                weights = 1.0 / (dist_valid ** power)
                result[i] = np.sum(weights * data_valid) / np.sum(weights)

        return result

    def _interpolate_linear(
        self, data: np.ndarray, fill_value: float
    ) -> np.ndarray:
        """Linear interpolation (not cached, uses scipy griddata)."""
        from scipy.interpolate import griddata

        points = np.column_stack([self.lon, self.lat])
        xi = np.column_stack([self.lon2d.ravel(), self.lat2d.ravel()])

        return griddata(
            points, data, xi, method="linear", fill_value=fill_value
        )


def regrid(
    data: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    box: tuple[float, float, float, float] = (-180, 180, -90, 90),
    res: tuple[int, int] = (360, 180),
    method: Literal["nn", "idw", "linear"] = "nn",
    influence: float = 80000,
    fill_value: float = np.nan,
    interpolator: RegridInterpolator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate unstructured data to a regular grid.

    Parameters
    ----------
    data : np.ndarray
        Data values at unstructured points, shape (npoints,).
    lon : np.ndarray
        Longitudes of data points in degrees, shape (npoints,).
    lat : np.ndarray
        Latitudes of data points in degrees, shape (npoints,).
    box : tuple
        Bounding box as (lon_min, lon_max, lat_min, lat_max).
        Default is global (-180, 180, -90, 90).
    res : tuple
        Output resolution as (nlon, nlat). Default is (360, 180).
    method : str
        Interpolation method:
        - 'nn': Nearest neighbor (fast, default)
        - 'idw': Inverse distance weighting
        - 'linear': Linear interpolation (scipy griddata)
    influence : float
        Radius of influence in meters. Points outside this radius
        from any source point will be set to fill_value.
        Default is 80000 (80 km).
    fill_value : float
        Value for grid points with no data. Default is NaN.
    interpolator : RegridInterpolator, optional
        Pre-computed interpolator for caching. If provided, lon, lat,
        box, res, method, and influence are ignored.

    Returns
    -------
    data_reg : np.ndarray
        Interpolated data on regular grid, shape (nlat, nlon).
    lon_reg : np.ndarray
        1D array of output longitudes.
    lat_reg : np.ndarray
        1D array of output latitudes.

    Example
    -------
    >>> # Simple one-off interpolation
    >>> data_reg, lon_reg, lat_reg = regrid(temp, mesh.lon, mesh.lat)
    >>>
    >>> # With caching for multiple variables
    >>> interp = RegridInterpolator(mesh.lon, mesh.lat)
    >>> temp_reg, lon_reg, lat_reg = regrid(temp, mesh.lon, mesh.lat, interpolator=interp)
    >>> salt_reg, _, _ = regrid(salt, mesh.lon, mesh.lat, interpolator=interp)
    """
    if interpolator is not None:
        return interpolator(data, fill_value)

    # Create interpolator and use it
    interp = RegridInterpolator(
        lon=lon,
        lat=lat,
        box=box,
        res=res,
        method=method,
        influence=influence,
    )
    return interp(data, fill_value)
