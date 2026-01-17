"""Transect interpolation and plotting for unstructured 3D ocean data.

This module provides functionality for extracting and visualizing vertical
cross-sections (transects) through unstructured ocean model data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from fesomp.mesh import Mesh

# Earth radius in meters
EARTH_RADIUS = 6371000.0


def _lonlat_to_cartesian(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convert lon/lat (degrees) to 3D Cartesian on unit sphere."""
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])


def _cartesian_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3D Cartesian on unit sphere to lon/lat (degrees)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = np.rad2deg(np.arcsin(np.clip(z, -1, 1)))
    lon = np.rad2deg(np.arctan2(y, x))
    return lon, lat


def _meters_to_chord(meters: float, earth_radius: float = EARTH_RADIUS) -> float:
    """Convert distance in meters to chord distance on unit sphere."""
    theta = meters / earth_radius
    return 2 * np.sin(theta / 2)


def _chord_to_meters(chord: float, earth_radius: float = EARTH_RADIUS) -> float:
    """Convert chord distance on unit sphere to meters."""
    theta = 2 * np.arcsin(np.clip(chord / 2, -1, 1))
    return theta * earth_radius


def great_circle_path(
    start: tuple[float, float],
    end: tuple[float, float],
    npoints: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute points along a great circle path between two points.

    Uses spherical linear interpolation (slerp) on a unit sphere for
    accurate great circle computation.

    Parameters
    ----------
    start : tuple
        Starting point as (lon, lat) in degrees.
    end : tuple
        Ending point as (lon, lat) in degrees.
    npoints : int
        Number of points along the path (including endpoints).

    Returns
    -------
    lon : np.ndarray
        Longitudes along the path in degrees, shape (npoints,).
    lat : np.ndarray
        Latitudes along the path in degrees, shape (npoints,).
    distance : np.ndarray
        Distance from start in meters, shape (npoints,).

    Example
    -------
    >>> lon, lat, dist = great_circle_path((0, 0), (90, 0), npoints=10)
    >>> print(f"Total distance: {dist[-1] / 1000:.0f} km")
    """
    start_lon, start_lat = start
    end_lon, end_lat = end

    # Convert to Cartesian on unit sphere
    p0 = _lonlat_to_cartesian(np.array([start_lon]), np.array([start_lat]))[0]
    p1 = _lonlat_to_cartesian(np.array([end_lon]), np.array([end_lat]))[0]

    # Compute angle between points (great circle arc)
    dot = np.clip(np.dot(p0, p1), -1, 1)
    omega = np.arccos(dot)

    # Handle degenerate case (same point or antipodal)
    if omega < 1e-10:
        # Same point
        lons = np.full(npoints, start_lon)
        lats = np.full(npoints, start_lat)
        distances = np.zeros(npoints)
        return lons, lats, distances

    if omega > np.pi - 1e-10:
        # Antipodal points - great circle is not unique
        raise ValueError(
            "Start and end points are antipodal. Great circle path is not unique."
        )

    # Spherical linear interpolation (slerp)
    t = np.linspace(0, 1, npoints)
    sin_omega = np.sin(omega)

    # P(t) = sin((1-t)*omega)/sin(omega) * P0 + sin(t*omega)/sin(omega) * P1
    w0 = np.sin((1 - t) * omega) / sin_omega
    w1 = np.sin(t * omega) / sin_omega

    xyz = np.outer(w0, p0) + np.outer(w1, p1)

    # Convert back to lon/lat
    lons, lats = _cartesian_to_lonlat(xyz)

    # Compute cumulative distance along path
    distances = t * omega * EARTH_RADIUS

    return lons, lats, distances


def great_circle_distance(
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """
    Compute great circle distance between two points.

    Parameters
    ----------
    start : tuple
        Starting point as (lon, lat) in degrees.
    end : tuple
        Ending point as (lon, lat) in degrees.

    Returns
    -------
    distance : float
        Distance in meters.
    """
    p0 = _lonlat_to_cartesian(np.array([start[0]]), np.array([start[1]]))[0]
    p1 = _lonlat_to_cartesian(np.array([end[0]]), np.array([end[1]]))[0]
    omega = np.arccos(np.clip(np.dot(p0, p1), -1, 1))
    return omega * EARTH_RADIUS


@dataclass
class TransectInterpolator:
    """
    Cached interpolator for extracting vertical transects from unstructured 3D data.

    This class pre-computes the KDTree and interpolation indices/weights,
    allowing fast repeated interpolation of different variables along
    the same transect.

    Parameters
    ----------
    lon : np.ndarray
        Longitudes of source points in degrees, shape (n2d,).
    lat : np.ndarray
        Latitudes of source points in degrees, shape (n2d,).
    start : tuple
        Starting point of transect as (lon, lat) in degrees.
    end : tuple
        Ending point of transect as (lon, lat) in degrees.
    npoints : int
        Number of points along the transect.
    method : str
        Interpolation method: 'nn', 'idw', or 'linear'.
    influence : float
        Radius of influence in meters.
    k : int
        Number of neighbors for IDW interpolation.

    Attributes
    ----------
    transect_lon : np.ndarray
        Longitudes of transect points.
    transect_lat : np.ndarray
        Latitudes of transect points.
    transect_distance : np.ndarray
        Distances from start along transect in meters.

    Example
    -------
    >>> # Create interpolator once
    >>> interp = TransectInterpolator(
    ...     mesh.lon, mesh.lat,
    ...     start=(-30, -60), end=(-30, 60),
    ...     npoints=100,
    ... )
    >>>
    >>> # Use for different 3D variables - data shape: (nlev, n2d)
    >>> temp_transect = interp(temp_3d)  # Returns (nlev, npoints)
    >>> salt_transect = interp(salt_3d)
    """

    lon: np.ndarray
    lat: np.ndarray
    start: tuple[float, float]
    end: tuple[float, float]
    npoints: int = 100
    method: Literal["nn", "idw", "linear"] = "nn"
    influence: float = 80000
    k: int = 10

    # Computed attributes (set in __post_init__)
    transect_lon: np.ndarray = field(init=False, repr=False)
    transect_lat: np.ndarray = field(init=False, repr=False)
    transect_distance: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Build KDTree and compute interpolation indices."""
        self.lon = np.asarray(self.lon).ravel()
        self.lat = np.asarray(self.lat).ravel()

        # Compute transect path
        self.transect_lon, self.transect_lat, self.transect_distance = great_circle_path(
            self.start, self.end, self.npoints
        )

        # Convert source points to Cartesian
        self._src_xyz = _lonlat_to_cartesian(self.lon, self.lat)

        # Convert transect points to Cartesian
        self._dst_xyz = _lonlat_to_cartesian(self.transect_lon, self.transect_lat)

        # Build KDTree on source points
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
    ) -> np.ndarray:
        """
        Interpolate data along the transect.

        Parameters
        ----------
        data : np.ndarray
            Data values at source points. Can be:
            - 1D array of shape (n2d,) for surface data
            - 2D array of shape (nlev, n2d) for 3D data
        fill_value : float
            Value for points outside influence radius.

        Returns
        -------
        data_transect : np.ndarray
            Interpolated data along transect:
            - Shape (npoints,) if input was 1D
            - Shape (nlev, npoints) if input was 2D
        """
        data = np.asarray(data)

        # Handle 1D case (surface data)
        if data.ndim == 1:
            return self._interpolate_1d(data, fill_value)

        # Handle 2D case (3D ocean data: nlev x n2d)
        if data.ndim == 2:
            nlev = data.shape[0]
            if data.shape[1] != len(self.lon):
                raise ValueError(
                    f"Data shape {data.shape} doesn't match source grid. "
                    f"Expected (nlev, {len(self.lon)})"
                )

            # Interpolate each level
            result = np.empty((nlev, self.npoints), dtype=np.float64)
            for lev in range(nlev):
                result[lev] = self._interpolate_1d(data[lev], fill_value)

            return result

        raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")

    def _interpolate_1d(
        self, data: np.ndarray, fill_value: float
    ) -> np.ndarray:
        """Interpolate 1D data (single level)."""
        data = data.ravel()

        if len(data) != len(self.lon):
            raise ValueError(
                f"Data length ({len(data)}) doesn't match source grid ({len(self.lon)})"
            )

        if self.method == "nn":
            return self._interpolate_nn(data, fill_value)
        elif self.method == "idw":
            return self._interpolate_idw(data, fill_value)
        elif self.method == "linear":
            return self._interpolate_linear(data, fill_value)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _interpolate_nn(
        self, data: np.ndarray, fill_value: float
    ) -> np.ndarray:
        """Nearest neighbor interpolation using cached indices."""
        result = np.full(self.npoints, fill_value, dtype=np.float64)
        result[self._valid_mask] = data[self._indices[self._valid_mask, 0]]
        return result

    def _interpolate_idw(
        self, data: np.ndarray, fill_value: float, power: float = 2.0
    ) -> np.ndarray:
        """Inverse distance weighting using cached indices."""
        result = np.full(self.npoints, fill_value, dtype=np.float64)

        for i in range(self.npoints):
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
        xi = np.column_stack([self.transect_lon, self.transect_lat])

        return griddata(
            points, data, xi, method="linear", fill_value=fill_value
        )

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get transect coordinates.

        Returns
        -------
        lon : np.ndarray
            Longitudes along transect.
        lat : np.ndarray
            Latitudes along transect.
        distance : np.ndarray
            Distance from start in meters.
        """
        return self.transect_lon, self.transect_lat, self.transect_distance


def interpolate_transect(
    data: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    npoints: int = 100,
    method: Literal["nn", "idw", "linear"] = "nn",
    influence: float = 80000,
    fill_value: float = np.nan,
    interpolator: TransectInterpolator | None = None,
) -> tuple[np.ndarray, np.ndarray, TransectInterpolator]:
    """
    Interpolate unstructured data along a great circle transect.

    Parameters
    ----------
    data : np.ndarray
        Data values at unstructured points:
        - Shape (n2d,) for surface data
        - Shape (nlev, n2d) for 3D data
    lon : np.ndarray
        Longitudes of data points in degrees.
    lat : np.ndarray
        Latitudes of data points in degrees.
    start : tuple
        Starting point of transect as (lon, lat) in degrees.
    end : tuple
        Ending point of transect as (lon, lat) in degrees.
    npoints : int
        Number of points along the transect. Default is 100.
    method : str
        Interpolation method:
        - 'nn': Nearest neighbor (fast, default)
        - 'idw': Inverse distance weighting
        - 'linear': Linear interpolation (scipy griddata)
    influence : float
        Radius of influence in meters. Default is 80000 (80 km).
    fill_value : float
        Value for transect points with no data. Default is NaN.
    interpolator : TransectInterpolator, optional
        Pre-computed interpolator for caching.

    Returns
    -------
    data_transect : np.ndarray
        Interpolated data along transect:
        - Shape (npoints,) if input was 1D
        - Shape (nlev, npoints) if input was 2D
    transect_distance : np.ndarray
        Distance from start along transect in meters.
    interpolator : TransectInterpolator
        The interpolator used (can be reused).

    Example
    -------
    >>> # 3D ocean data transect
    >>> temp_t, dist, interp = interpolate_transect(
    ...     temp_3d,  # shape (nlev, n2d)
    ...     mesh.lon, mesh.lat,
    ...     start=(-30, -60), end=(-30, 60),
    ... )
    >>> # temp_t has shape (nlev, npoints)
    """
    if interpolator is None:
        interpolator = TransectInterpolator(
            lon=lon,
            lat=lat,
            start=start,
            end=end,
            npoints=npoints,
            method=method,
            influence=influence,
        )

    data_transect = interpolator(data, fill_value)

    return data_transect, interpolator.transect_distance, interpolator


def plot_transect(
    data: np.ndarray,
    distance: np.ndarray,
    depth: np.ndarray,
    *,
    # Plot options
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
    figsize: tuple[float, float] = (12, 5),
    # Style options
    cmap: str | None = None,
    levels: tuple | list | None = None,
    ptype: Literal["cf", "pcm"] = "cf",
    # Labels
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    units: str | None = None,
    colorbar: bool = True,
    # Distance display
    distance_units: Literal["m", "km"] = "km",
    # Depth options
    depth_limits: tuple[float, float] | None = None,
    invert_yaxis: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a vertical transect as a 2D cross-section.

    Parameters
    ----------
    data : np.ndarray
        Data values along transect, shape (nlev, npoints).
    distance : np.ndarray
        Distance along transect in meters, shape (npoints,).
    depth : np.ndarray
        Depth levels in meters (positive downward), shape (nlev,).
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    fig : matplotlib.figure.Figure, optional
        Existing figure to use.
    figsize : tuple
        Figure size in inches if creating new figure.
    cmap : str, optional
        Colormap name. Default is 'RdBu_r'.
    levels : tuple or list, optional
        Contour levels. Can be (min, max, nlevels) or explicit list.
        Default is auto from data.
    ptype : str
        Plot type: 'cf' (contourf) or 'pcm' (pcolormesh).
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label. Default is 'Distance (km)' or 'Distance (m)'.
    ylabel : str, optional
        Y-axis label. Default is 'Depth (m)'.
    units : str, optional
        Units string for colorbar label.
    colorbar : bool
        Whether to show colorbar. Default is True.
    distance_units : str
        Units for distance axis: 'm' or 'km'. Default is 'km'.
    depth_limits : tuple, optional
        Depth range to display as (min_depth, max_depth).
    invert_yaxis : bool
        Whether to invert y-axis (so depth increases downward). Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Example
    -------
    >>> fig, ax = plot_transect(
    ...     temp_transect,  # shape (nlev, npoints)
    ...     transect_distance,
    ...     mesh.depth_levels,
    ...     title="Temperature",
    ...     units="degC",
    ... )
    """
    # Validate input
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D (nlev, npoints), got shape {data.shape}")

    nlev, npoints = data.shape

    if len(distance) != npoints:
        raise ValueError(
            f"Distance length ({len(distance)}) doesn't match data ({npoints})"
        )

    if len(depth) != nlev:
        raise ValueError(
            f"Depth length ({len(depth)}) doesn't match data ({nlev})"
        )

    # Convert distance if needed
    if distance_units == "km":
        dist_plot = distance / 1000
        default_xlabel = "Distance (km)"
    else:
        dist_plot = distance
        default_xlabel = "Distance (m)"

    # Create figure/axes if needed
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    # Default colormap
    if cmap is None:
        cmap = "RdBu_r"

    # Parse levels
    levels_arr = _parse_levels(levels, data)

    # Create meshgrid for plotting
    dist_2d, depth_2d = np.meshgrid(dist_plot, depth)

    # Plot
    if ptype == "cf":
        im = ax.contourf(
            dist_2d,
            depth_2d,
            data,
            levels=levels_arr,
            cmap=cmap,
            extend="both",
        )
    else:  # pcm
        im = ax.pcolormesh(
            dist_2d,
            depth_2d,
            data,
            cmap=cmap,
            vmin=levels_arr.min() if levels_arr is not None else None,
            vmax=levels_arr.max() if levels_arr is not None else None,
            shading="auto",
        )

    # Colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        if units is not None:
            cbar.set_label(units)

    # Labels
    if xlabel is None:
        xlabel = default_xlabel
    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = "Depth (m)"
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    # Depth limits
    if depth_limits is not None:
        ax.set_ylim(depth_limits)

    # Invert y-axis so depth increases downward
    if invert_yaxis:
        ax.invert_yaxis()

    return fig, ax


def _parse_levels(
    levels: tuple | list | None, data: np.ndarray, nlevels: int = 40
) -> np.ndarray | None:
    """Parse levels specification."""
    if levels is None:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        return np.linspace(vmin, vmax, nlevels)

    if len(levels) == 3 and isinstance(levels[2], int):
        return np.linspace(levels[0], levels[1], levels[2])

    return np.array(levels)


def transect(
    data: np.ndarray,
    mesh: Mesh,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    # Depth specification
    depth: np.ndarray | None = None,
    # Interpolation options
    npoints: int = 100,
    method: Literal["nn", "idw", "linear"] = "nn",
    influence: float = 80000,
    fill_value: float = np.nan,
    interpolator: TransectInterpolator | None = None,
    # Plot options
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
    figsize: tuple[float, float] = (12, 5),
    # Style options
    cmap: str | None = None,
    levels: tuple | list | None = None,
    ptype: Literal["cf", "pcm"] = "cf",
    # Labels
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    units: str | None = None,
    colorbar: bool = True,
    # Distance display
    distance_units: Literal["m", "km"] = "km",
    # Depth options
    depth_limits: tuple[float, float] | None = None,
    invert_yaxis: bool = True,
) -> tuple[plt.Figure, plt.Axes, TransectInterpolator]:
    """
    Interpolate and plot a vertical transect through 3D ocean data.

    This is a convenience function combining interpolate_transect and
    plot_transect. Automatically detects:
    - Horizontal location: nodes (n2d points) vs elements (nelem points)
    - Vertical coordinate: levels (interfaces) vs layers (centers)

    Parameters
    ----------
    data : np.ndarray
        Data values at unstructured points:
        - Shape (nlev, n2d) for data on nodes
        - Shape (nlev, nelem) for data on elements
        Can be defined on either levels (interfaces) or layers (centers).
    mesh : Mesh
        The mesh object containing lon, lat, and depth information.
    start : tuple
        Starting point of transect as (lon, lat) in degrees.
    end : tuple
        Ending point of transect as (lon, lat) in degrees.
    depth : np.ndarray, optional
        Depth coordinates in meters. If not provided, automatically
        selects mesh.depth_levels or mesh.depth_layers based on data shape.
    npoints : int
        Number of points along the transect. Default is 100.
    method : str
        Interpolation method: 'nn', 'idw', or 'linear'.
    influence : float
        Radius of influence in meters. Default is 80000 (80 km).
    fill_value : float
        Value for transect points with no data. Default is NaN.
    interpolator : TransectInterpolator, optional
        Pre-computed interpolator for caching.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    fig : matplotlib.figure.Figure, optional
        Existing figure to use.
    figsize : tuple
        Figure size in inches if creating new figure.
    cmap : str, optional
        Colormap name. Default is 'RdBu_r'.
    levels : tuple or list, optional
        Contour levels. Can be (min, max, nlevels) or explicit list.
    ptype : str
        Plot type: 'cf' (contourf) or 'pcm' (pcolormesh).
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    units : str, optional
        Units string for colorbar label.
    colorbar : bool
        Whether to show colorbar. Default is True.
    distance_units : str
        Units for distance axis: 'm' or 'km'. Default is 'km'.
    depth_limits : tuple, optional
        Depth range to display as (min_depth, max_depth).
    invert_yaxis : bool
        Whether to invert y-axis. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    interpolator : TransectInterpolator
        The interpolator used (can be reused).

    Example
    -------
    >>> # Plot temperature transect (data on layers)
    >>> fig, ax, interp = fesomp.transect(
    ...     temp_3d,  # shape (nlev-1, n2d) - on layers
    ...     mesh,
    ...     start=(-30, -60), end=(-30, 60),
    ...     title="Temperature along 30W",
    ...     units="degC",
    ...     depth_limits=(0, 1000),
    ... )
    >>>
    >>> # Reuse interpolator for salinity
    >>> fig2, ax2, _ = fesomp.transect(
    ...     salt_3d, mesh,
    ...     start=(-30, -60), end=(-30, 60),
    ...     interpolator=interp,
    ...     title="Salinity along 30W",
    ... )
    """
    data = np.asarray(data)

    # Auto-detect horizontal location: nodes vs elements
    if data.ndim == 1:
        n_horizontal = len(data)
    else:
        n_horizontal = data.shape[-1]  # Last dimension is horizontal

    if n_horizontal == mesh.n2d:
        # Data is on nodes
        src_lon = mesh.lon
        src_lat = mesh.lat
    elif n_horizontal == mesh.nelem:
        # Data is on elements (triangle centers)
        src_lon = mesh.lon_elem
        src_lat = mesh.lat_elem
    else:
        raise ValueError(
            f"Data has {n_horizontal} horizontal points, but mesh has "
            f"{mesh.n2d} nodes and {mesh.nelem} elements. Cannot determine location."
        )

    # Auto-detect depth coordinate if not provided
    if depth is None:
        nlev_data = data.shape[0] if data.ndim == 2 else 1
        nlev_levels = mesh.nlev
        nlev_layers = nlev_levels - 1

        if nlev_data == nlev_levels:
            # Data is on levels (interfaces)
            depth = mesh.depth_levels
        elif nlev_data == nlev_layers:
            # Data is on layers (centers)
            depth = mesh.depth_layers
        elif nlev_data == 1:
            # Surface data - use first depth level
            depth = mesh.depth_levels[:1]
        else:
            raise ValueError(
                f"Data has {nlev_data} vertical levels, but mesh has "
                f"{nlev_levels} levels and {nlev_layers} layers. "
                "Please specify depth explicitly."
            )

    # Interpolate
    data_t, dist_t, interp = interpolate_transect(
        data=data,
        lon=src_lon,
        lat=src_lat,
        start=start,
        end=end,
        npoints=npoints,
        method=method,
        influence=influence,
        fill_value=fill_value,
        interpolator=interpolator,
    )

    # Plot
    fig, ax = plot_transect(
        data=data_t,
        distance=dist_t,
        depth=depth,
        ax=ax,
        fig=fig,
        figsize=figsize,
        cmap=cmap,
        levels=levels,
        ptype=ptype,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        units=units,
        colorbar=colorbar,
        distance_units=distance_units,
        depth_limits=depth_limits,
        invert_yaxis=invert_yaxis,
    )

    return fig, ax, interp
