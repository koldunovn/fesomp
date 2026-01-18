"""Meridional Overturning Circulation (MOC) diagnostics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from .utils import get_region_mask, get_surface_area, list_available_regions


def moc(
    w: np.ndarray | xr.DataArray,
    node_area: np.ndarray,
    lat: np.ndarray,
    depth_levels: np.ndarray,
    lat_range: tuple[float, float] = (-90, 90),
    nlats: int = 181,
    mask: np.ndarray | None = None,
    cumsum_direction: Literal["bottom_up", "top_down"] = "bottom_up",
) -> tuple[np.ndarray | xr.DataArray, np.ndarray]:
    """
    Compute Meridional Overturning Circulation from vertical velocity.

    In FESOM2, vertical velocity (w) is defined on nodes at level interfaces.
    The MOC streamfunction is computed by:
    1. Binning vertical velocity by latitude
    2. Weighting by node area
    3. Computing cumulative sum in depth

    Parameters
    ----------
    w : array-like
        Vertical velocity in m/s, shape (..., nz, n2d).
        Positive = upward. Data is on level interfaces (nz levels).
    node_area : np.ndarray
        Node areas in m², shape (n2d,) or (nz, n2d).
        If 2D, uses level-dependent areas; if 1D, uses surface area for all levels.
    lat : np.ndarray
        Latitude of nodes in degrees, shape (n2d,).
    depth_levels : np.ndarray
        Depth at level interfaces in meters, shape (nz,).
    lat_range : tuple[float, float]
        (min_lat, max_lat) for output. Default: (-90, 90).
    nlats : int
        Number of latitude bins. Default: 181 (1° resolution).
    mask : np.ndarray, optional
        Boolean mask for nodes, shape (n2d,). True = include.
        Use this to select specific basins (Atlantic, Indo-Pacific, etc.).
    cumsum_direction : {"bottom_up", "top_down"}
        Direction for cumulative sum:
        - "bottom_up": integrate from bottom to surface (standard)
        - "top_down": integrate from surface to bottom

    Returns
    -------
    moc : array-like
        MOC streamfunction in Sv (10⁶ m³/s), shape (..., nz, nlats).
    lat_bins : np.ndarray
        Latitude bin centers, shape (nlats,).

    Notes
    -----
    The MOC streamfunction Ψ(y, z) represents the zonally-integrated
    volume transport. At each latitude and depth:
        Ψ = ∫∫ w dA

    where the integral is over all nodes at that latitude, summed
    cumulatively from bottom (or top).

    To compute Atlantic MOC, use:
        mask = fesomp.diag.get_basin_mask(mesh.lon, mesh.lat, "Atlantic_MOC")
        amoc, lats = fesomp.diag.moc(w, node_area, mesh.lat, depth_levels, mask=mask)

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> w = xr.open_dataset("w.nc")["w"]  # (time, nz, n2d)
    >>> # Global MOC
    >>> moc_global, lats = fesomp.diag.moc(
    ...     w,
    ...     mesh.geometry.node_area[0],  # surface node areas
    ...     mesh.lat,
    ...     mesh.depth_levels
    ... )
    >>> # Atlantic MOC
    >>> atl_mask = fesomp.diag.get_basin_mask(
    ...     mesh.lon, mesh.lat, "Atlantic_MOC"
    ... )
    >>> amoc, lats = fesomp.diag.moc(
    ...     w, mesh.geometry.node_area[0], mesh.lat,
    ...     mesh.depth_levels, mask=atl_mask
    ... )
    """
    # Conversion factor to Sverdrups (1 Sv = 10^6 m³/s)
    SV = 1e6

    # Create latitude bins
    lat_min, lat_max = lat_range
    lat_bins = np.linspace(lat_min, lat_max, nlats)
    lat_edges = np.linspace(lat_min, lat_max, nlats + 1)

    # Get node area (handle 1D or 2D input)
    if node_area.ndim == 2:
        # Level-dependent areas: (nz, n2d)
        area = node_area
    else:
        # Surface area only: broadcast to all levels
        area = node_area

    # Apply mask to areas
    if mask is not None:
        if area.ndim == 2:
            area_masked = area * mask[np.newaxis, :]
        else:
            area_masked = area * mask
    else:
        area_masked = area.copy() if area.ndim == 1 else area.copy()

    # Bin nodes by latitude
    lat_idx = np.digitize(lat, lat_edges) - 1
    # Clip to valid range
    lat_idx = np.clip(lat_idx, 0, nlats - 1)

    is_xarray = isinstance(w, xr.DataArray)

    if is_xarray:
        # Identify dimension names
        dims = w.dims
        # Assume last dim is n2d, second-to-last is nz
        node_dim = dims[-1]
        depth_dim = dims[-2]

        # Use xarray's apply_ufunc for dask compatibility
        result = xr.apply_ufunc(
            _moc_numpy,
            w,
            kwargs=dict(
                node_area=area_masked,
                lat_idx=lat_idx,
                nlats=nlats,
                cumsum_direction=cumsum_direction,
            ),
            input_core_dims=[[depth_dim, node_dim]],
            output_core_dims=[[depth_dim, "lat"]],
            exclude_dims={node_dim},
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs=dict(
                output_sizes={"lat": nlats},
            ),
        )
        # Add coordinates
        result = result.assign_coords(
            lat=("lat", lat_bins),
        )
        if depth_dim in w.coords:
            result = result.assign_coords({depth_dim: w.coords[depth_dim]})

        # Convert to Sv
        result = result / SV
        result.attrs["units"] = "Sv"
        result.attrs["long_name"] = "Meridional Overturning Streamfunction"

        return result, lat_bins

    else:
        # Pure numpy
        moc_sv = _moc_numpy(
            w,
            node_area=area_masked,
            lat_idx=lat_idx,
            nlats=nlats,
            cumsum_direction=cumsum_direction,
        )
        return moc_sv / SV, lat_bins


def _moc_numpy(
    w: np.ndarray,
    node_area: np.ndarray,
    lat_idx: np.ndarray,
    nlats: int,
    cumsum_direction: str,
) -> np.ndarray:
    """
    NumPy implementation of MOC calculation.

    Parameters
    ----------
    w : np.ndarray
        Vertical velocity, shape (..., nz, n2d).
    node_area : np.ndarray
        Node areas (possibly masked), shape (n2d,) or (nz, n2d).
    lat_idx : np.ndarray
        Latitude bin index for each node, shape (n2d,).
    nlats : int
        Number of latitude bins.
    cumsum_direction : str
        "bottom_up" or "top_down".

    Returns
    -------
    np.ndarray
        MOC streamfunction in m³/s, shape (..., nz, nlats).
    """
    # Get shape
    original_shape = w.shape[:-2]
    nz = w.shape[-2]
    n2d = w.shape[-1]

    # Reshape for easier processing
    if len(original_shape) > 0:
        batch_size = int(np.prod(original_shape))
        w_flat = w.reshape(batch_size, nz, n2d)
    else:
        w_flat = w[np.newaxis, :, :]  # Add batch dimension
        batch_size = 1

    # Initialize output
    moc = np.zeros((batch_size, nz, nlats), dtype=np.float64)

    # Handle 1D vs 2D node_area
    if node_area.ndim == 1:
        # Same area for all levels
        for lev in range(nz):
            w_weighted = w_flat[:, lev, :] * node_area[np.newaxis, :]
            for b in range(batch_size):
                np.add.at(moc[b, lev], lat_idx, w_weighted[b])
    else:
        # Level-dependent areas: (nz, n2d)
        for lev in range(nz):
            w_weighted = w_flat[:, lev, :] * node_area[lev, :][np.newaxis, :]
            for b in range(batch_size):
                np.add.at(moc[b, lev], lat_idx, w_weighted[b])

    # Cumulative sum in depth
    if cumsum_direction == "bottom_up":
        # Integrate from bottom to surface
        moc = np.cumsum(moc[:, ::-1, :], axis=1)[:, ::-1, :]
    else:
        # Integrate from surface to bottom
        moc = np.cumsum(moc, axis=1)

    # Reshape back
    if len(original_shape) > 0:
        moc = moc.reshape(*original_shape, nz, nlats)
    else:
        moc = moc[0]

    return moc


def get_basin_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    basin: str,
) -> np.ndarray:
    """
    Get boolean mask for a named ocean basin.

    Uses the MOCBasins.geojson file included with the package.

    Parameters
    ----------
    lon : np.ndarray
        Longitude of points in degrees, shape (n,).
    lat : np.ndarray
        Latitude of points in degrees, shape (n,).
    basin : str
        Name of the basin. Available basins:
        - "Atlantic_MOC": Atlantic Ocean (including Med, Baltic, etc.)
        - "IndoPacific_MOC": Indo-Pacific Ocean
        - "Pacific_MOC": Pacific Ocean
        - "Indian_MOC": Indian Ocean
        Use list_moc_basins() to see all available basins.

    Returns
    -------
    np.ndarray
        Boolean mask, shape (n,). True = inside basin.

    Raises
    ------
    ImportError
        If shapely is not installed.
    ValueError
        If basin name is not found.

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> # Get Atlantic mask for nodes
    >>> atl_mask = fesomp.diag.get_basin_mask(
    ...     mesh.lon, mesh.lat, "Atlantic_MOC"
    ... )
    """
    return get_region_mask(lon, lat, basin, geojson_name="MOCBasins")


def list_moc_basins() -> list[str]:
    """
    List available MOC basin names.

    Returns
    -------
    list[str]
        List of available basin names for use with get_basin_mask().
    """
    return list_available_regions("MOCBasins")


def amoc_index(
    moc_data: np.ndarray | xr.DataArray,
    lat_bins: np.ndarray,
    depth_levels: np.ndarray,
    lat: float = 26.5,
    depth_range: tuple[float, float] = (500, 2000),
) -> np.ndarray | xr.DataArray:
    """
    Compute AMOC index as maximum streamfunction at specified latitude.

    The standard AMOC index is the maximum overturning at 26.5°N
    (RAPID array latitude) in the depth range 500-2000m.

    Parameters
    ----------
    moc_data : array-like
        MOC streamfunction in Sv, shape (..., nz, nlats).
    lat_bins : np.ndarray
        Latitude bin centers, shape (nlats,).
    depth_levels : np.ndarray
        Depth at levels, shape (nz,).
    lat : float
        Latitude for index calculation. Default: 26.5°N.
    depth_range : tuple[float, float]
        Depth range for maximum search. Default: (500, 2000).

    Returns
    -------
    array-like
        AMOC index in Sv, shape (...).

    Examples
    --------
    >>> # Compute Atlantic MOC
    >>> atl_mask = fesomp.diag.get_basin_mask(mesh.lon, mesh.lat, "Atlantic_MOC")
    >>> amoc, lats = fesomp.diag.moc(w, node_area, mesh.lat, depth_levels, mask=atl_mask)
    >>> # Get AMOC index at 26.5N
    >>> amoc_26 = fesomp.diag.amoc_index(amoc, lats, depth_levels)
    """
    # Find nearest latitude index
    lat_idx = np.argmin(np.abs(lat_bins - lat))

    # Find depth range indices
    depth_min, depth_max = depth_range
    depth_mask = (depth_levels >= depth_min) & (depth_levels <= depth_max)
    depth_indices = np.where(depth_mask)[0]

    if len(depth_indices) == 0:
        raise ValueError(
            f"No depth levels found in range {depth_range}. "
            f"Available depths: {depth_levels.min():.0f}-{depth_levels.max():.0f}m"
        )

    is_xarray = isinstance(moc_data, xr.DataArray)

    if is_xarray:
        # Select latitude and depth range
        moc_at_lat = moc_data.isel(lat=lat_idx)
        depth_dim = moc_at_lat.dims[-1]
        moc_depth_sel = moc_at_lat.isel({depth_dim: depth_indices})
        # Maximum over depth
        return moc_depth_sel.max(dim=depth_dim)
    else:
        # Select latitude
        moc_at_lat = moc_data[..., :, lat_idx]  # (..., nz)
        # Select depth range
        moc_depth_sel = moc_at_lat[..., depth_indices]
        # Maximum over depth
        return np.max(moc_depth_sel, axis=-1)
