"""Utility functions for diagnostics module."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

# Type alias for hemisphere selection
Hemisphere = Literal["N", "S", "both"]


def hemisphere_mask(lat: np.ndarray, hemisphere: Hemisphere) -> np.ndarray:
    """
    Create boolean mask for hemisphere selection.

    Parameters
    ----------
    lat : np.ndarray
        Latitude of points in degrees, shape (n,).
    hemisphere : {"N", "S", "both"}
        "N" for Northern Hemisphere (lat >= 0),
        "S" for Southern Hemisphere (lat < 0),
        "both" for global (all points).

    Returns
    -------
    np.ndarray
        Boolean mask, shape (n,). True = include.

    Raises
    ------
    ValueError
        If hemisphere is not one of "N", "S", "both".
    """
    if hemisphere == "N":
        return lat >= 0
    elif hemisphere == "S":
        return lat < 0
    elif hemisphere == "both":
        return np.ones(len(lat), dtype=bool)
    else:
        raise ValueError(f"hemisphere must be 'N', 'S', or 'both', got '{hemisphere}'")


def compute_layer_thickness(depth_levels: np.ndarray) -> np.ndarray:
    """
    Compute layer thickness from level interface depths.

    Parameters
    ----------
    depth_levels : np.ndarray
        Depth at level interfaces in meters, shape (nlev,).
        Should be positive, increasing downward.

    Returns
    -------
    np.ndarray
        Layer thickness in meters, shape (nlev-1,).
    """
    return np.diff(depth_levels)


def select_depth_indices(
    depth_levels: np.ndarray,
    depth_range: tuple[float, float] | None = None,
) -> tuple[int, int]:
    """
    Find level indices corresponding to a depth range.

    Parameters
    ----------
    depth_levels : np.ndarray
        Depth at level interfaces in meters, shape (nlev,).
    depth_range : tuple[float, float], optional
        (min_depth, max_depth) in meters.
        If None, returns (0, nlev-1) for full depth.

    Returns
    -------
    tuple[int, int]
        (start_index, end_index) for slicing.
        Use as depth_levels[start:end+1] or layers[start:end].
    """
    if depth_range is None:
        return 0, len(depth_levels) - 1

    min_depth, max_depth = depth_range

    # Find first level >= min_depth
    start_idx = np.searchsorted(depth_levels, min_depth)
    if start_idx > 0:
        start_idx -= 1  # Include the level just above min_depth

    # Find last level <= max_depth
    end_idx = np.searchsorted(depth_levels, max_depth)
    if end_idx >= len(depth_levels):
        end_idx = len(depth_levels) - 1

    return int(start_idx), int(end_idx)


def get_surface_area(node_area: np.ndarray) -> np.ndarray:
    """
    Extract surface area from node_area array.

    Parameters
    ----------
    node_area : np.ndarray
        Area at each node, shape (n2d,) or (nlev, n2d).

    Returns
    -------
    np.ndarray
        Surface area, shape (n2d,).
    """
    if node_area.ndim == 1:
        return node_area
    elif node_area.ndim == 2:
        return node_area[0]
    else:
        raise ValueError(f"node_area must be 1D or 2D, got {node_area.ndim}D")


def weighted_sum(
    data: np.ndarray | xr.DataArray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
    axis: int = -1,
) -> np.ndarray | xr.DataArray:
    """
    Compute weighted sum along specified axis.

    Handles both numpy arrays and xarray DataArrays (including dask).

    Parameters
    ----------
    data : array-like
        Data array, arbitrary shape.
    weights : np.ndarray
        Weights, must be broadcastable to data shape along axis.
    mask : np.ndarray, optional
        Boolean mask for points to include. Same shape as weights.
    axis : int
        Axis along which to sum.

    Returns
    -------
    array-like
        Weighted sum, with axis removed.
    """
    if mask is not None:
        weights = weights * mask

    if isinstance(data, xr.DataArray):
        # Get the dimension name for the specified axis
        dim = data.dims[axis]
        # Create weights as DataArray with matching dimension
        weights_da = xr.DataArray(weights, dims=[dim])
        return (data * weights_da).sum(dim=dim)
    else:
        return np.sum(data * weights, axis=axis)


def weighted_mean(
    data: np.ndarray | xr.DataArray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
    axis: int = -1,
) -> np.ndarray | xr.DataArray:
    """
    Compute weighted mean along specified axis.

    Handles both numpy arrays and xarray DataArrays (including dask).

    Parameters
    ----------
    data : array-like
        Data array, arbitrary shape.
    weights : np.ndarray
        Weights, must be broadcastable to data shape along axis.
    mask : np.ndarray, optional
        Boolean mask for points to include. Same shape as weights.
    axis : int
        Axis along which to compute mean.

    Returns
    -------
    array-like
        Weighted mean, with axis removed.
    """
    if mask is not None:
        weights = weights * mask

    if isinstance(data, xr.DataArray):
        dim = data.dims[axis]
        weights_da = xr.DataArray(weights, dims=[dim])
        total_weight = weights_da.sum(dim=dim)
        return (data * weights_da).sum(dim=dim) / total_weight
    else:
        total_weight = np.sum(weights)
        return np.sum(data * weights, axis=axis) / total_weight


def load_geojson(name: str) -> dict:
    """
    Load a GeoJSON file from the package data directory.

    Parameters
    ----------
    name : str
        Name of the GeoJSON file (without .geojson extension).
        Available: "MOCBasins", "NinoRegions", "oceanBasins"

    Returns
    -------
    dict
        Parsed GeoJSON data.

    Raises
    ------
    FileNotFoundError
        If the GeoJSON file does not exist.
    """
    try:
        # Python 3.9+ way using importlib.resources
        files = resources.files("fesomp.diag") / "data" / f"{name}.geojson"
        with files.open("r") as f:
            return json.load(f)
    except (TypeError, AttributeError):
        # Fallback for older Python or if resources.files doesn't work
        data_dir = Path(__file__).parent / "data"
        filepath = data_dir / f"{name}.geojson"
        if not filepath.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {filepath}")
        with open(filepath, "r") as f:
            return json.load(f)


def list_available_regions(geojson_name: str = "MOCBasins") -> list[str]:
    """
    List available region names in a GeoJSON file.

    Parameters
    ----------
    geojson_name : str
        Name of GeoJSON file: "MOCBasins", "NinoRegions", or "oceanBasins".

    Returns
    -------
    list[str]
        List of region names available in the file.
    """
    geojson = load_geojson(geojson_name)
    names = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name = props.get("name") or props.get("NAME") or props.get("Name")
        if name:
            names.append(name)
    return names


def get_region_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    region: str,
    geojson_name: str = "MOCBasins",
) -> np.ndarray:
    """
    Create boolean mask for points within a named region.

    Requires shapely to be installed.

    Parameters
    ----------
    lon : np.ndarray
        Longitude of points in degrees, shape (n,).
    lat : np.ndarray
        Latitude of points in degrees, shape (n,).
    region : str
        Name of the region (e.g., "Atlantic_Basin", "Pacific_Basin").
        Use list_available_regions() to see available names.
    geojson_name : str
        Name of GeoJSON file: "MOCBasins", "NinoRegions", or "oceanBasins".

    Returns
    -------
    np.ndarray
        Boolean mask, shape (n,). True = inside region.

    Raises
    ------
    ImportError
        If shapely is not installed.
    ValueError
        If region name is not found.
    """
    try:
        from shapely.geometry import Point, shape
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ImportError:
        raise ImportError(
            "shapely is required for region masks. "
            "Install with: pip install shapely>=2.0"
        )

    geojson = load_geojson(geojson_name)

    # Find matching features
    geometries = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name = props.get("name") or props.get("NAME") or props.get("Name")
        if name == region:
            geometries.append(shape(feature["geometry"]))

    if not geometries:
        available = list_available_regions(geojson_name)
        raise ValueError(
            f"Region '{region}' not found in {geojson_name}.geojson. "
            f"Available regions: {available}"
        )

    # Merge all matching geometries
    region_geom = unary_union(geometries)
    prepared_geom = prep(region_geom)

    # Create mask
    mask = np.array([prepared_geom.contains(Point(x, y)) for x, y in zip(lon, lat)])
    return mask
