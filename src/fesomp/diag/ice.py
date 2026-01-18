"""Sea ice diagnostics: area, volume, and extent."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .utils import Hemisphere, get_surface_area, hemisphere_mask, weighted_sum


def ice_area(
    sic: np.ndarray | xr.DataArray,
    node_area: np.ndarray,
    lat: np.ndarray,
    hemisphere: Hemisphere = "N",
    mask: np.ndarray | None = None,
) -> np.ndarray | xr.DataArray:
    """
    Compute sea ice area (ice concentration × cell area).

    Sea ice area is the sum of (ice concentration × cell area) over all cells.
    This represents the actual area covered by ice.

    Parameters
    ----------
    sic : array-like
        Sea ice concentration [0-1], shape (..., n2d).
        Can have any leading dimensions (time, ensemble, etc.).
    node_area : np.ndarray
        Area of each node in m², shape (n2d,) or (nlev, n2d).
        If 2D, uses the surface level (index 0).
    lat : np.ndarray
        Latitude of nodes in degrees, shape (n2d,).
    hemisphere : {"N", "S", "both"}
        "N" for Northern Hemisphere, "S" for Southern Hemisphere,
        "both" for global. Default is "N".
    mask : np.ndarray, optional
        Additional boolean mask, shape (n2d,). True = include.

    Returns
    -------
    array-like
        Total ice area in m², shape (...).
        Same type as input (numpy or xarray).

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> sic = xr.open_dataset("a_ice.nc")["a_ice"]  # shape: (time, n2d)
    >>> # Compute Northern Hemisphere ice area time series
    >>> nh_area = fesomp.diag.ice_area(
    ...     sic,
    ...     mesh.geometry.node_area[0],
    ...     mesh.lat,
    ...     hemisphere="N"
    ... )
    """
    # Get surface area
    area = get_surface_area(node_area)

    # Create hemisphere mask
    hmask = hemisphere_mask(lat, hemisphere)

    # Combine with optional additional mask
    if mask is not None:
        hmask = hmask & mask

    return weighted_sum(sic, area, mask=hmask, axis=-1)


def ice_volume(
    siv: np.ndarray | xr.DataArray,
    node_area: np.ndarray,
    lat: np.ndarray,
    hemisphere: Hemisphere = "N",
    mask: np.ndarray | None = None,
) -> np.ndarray | xr.DataArray:
    """
    Compute sea ice volume (ice thickness × cell area).

    Sea ice volume is the sum of (effective ice thickness × cell area).
    The input can be either:
    - Effective ice thickness (m): already area-weighted ice thickness
    - Ice volume per unit area (m): same as effective thickness

    Parameters
    ----------
    siv : array-like
        Sea ice effective thickness or volume per unit area in meters,
        shape (..., n2d). Can have any leading dimensions.
    node_area : np.ndarray
        Area of each node in m², shape (n2d,) or (nlev, n2d).
        If 2D, uses the surface level (index 0).
    lat : np.ndarray
        Latitude of nodes in degrees, shape (n2d,).
    hemisphere : {"N", "S", "both"}
        "N" for Northern Hemisphere, "S" for Southern Hemisphere,
        "both" for global. Default is "N".
    mask : np.ndarray, optional
        Additional boolean mask, shape (n2d,). True = include.

    Returns
    -------
    array-like
        Total ice volume in m³, shape (...).
        Same type as input (numpy or xarray).

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> siv = xr.open_dataset("m_ice.nc")["m_ice"]  # shape: (time, n2d)
    >>> # Compute Arctic ice volume time series
    >>> arctic_vol = fesomp.diag.ice_volume(
    ...     siv,
    ...     mesh.geometry.node_area[0],
    ...     mesh.lat,
    ...     hemisphere="N"
    ... )
    """
    # Get surface area
    area = get_surface_area(node_area)

    # Create hemisphere mask
    hmask = hemisphere_mask(lat, hemisphere)

    # Combine with optional additional mask
    if mask is not None:
        hmask = hmask & mask

    return weighted_sum(siv, area, mask=hmask, axis=-1)


def ice_extent(
    sic: np.ndarray | xr.DataArray,
    node_area: np.ndarray,
    lat: np.ndarray,
    hemisphere: Hemisphere = "N",
    threshold: float = 0.15,
    mask: np.ndarray | None = None,
) -> np.ndarray | xr.DataArray:
    """
    Compute sea ice extent (area where concentration exceeds threshold).

    Sea ice extent is the total area of cells where ice concentration
    exceeds the threshold (typically 15%). Unlike ice area, this doesn't
    weight by concentration - a cell is either fully counted or not.

    Parameters
    ----------
    sic : array-like
        Sea ice concentration [0-1], shape (..., n2d).
        Can have any leading dimensions (time, ensemble, etc.).
    node_area : np.ndarray
        Area of each node in m², shape (n2d,) or (nlev, n2d).
        If 2D, uses the surface level (index 0).
    lat : np.ndarray
        Latitude of nodes in degrees, shape (n2d,).
    hemisphere : {"N", "S", "both"}
        "N" for Northern Hemisphere, "S" for Southern Hemisphere,
        "both" for global. Default is "N".
    threshold : float
        Concentration threshold (default 0.15 = 15%).
        Cells with concentration > threshold are counted.
    mask : np.ndarray, optional
        Additional boolean mask, shape (n2d,). True = include.

    Returns
    -------
    array-like
        Total ice extent in m², shape (...).
        Same type as input (numpy or xarray).

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> sic = xr.open_dataset("a_ice.nc")["a_ice"]
    >>> # Standard ice extent with 15% threshold
    >>> extent = fesomp.diag.ice_extent(
    ...     sic,
    ...     mesh.geometry.node_area[0],
    ...     mesh.lat,
    ...     hemisphere="N",
    ...     threshold=0.15
    ... )
    """
    # Get surface area
    area = get_surface_area(node_area)

    # Create hemisphere mask
    hmask = hemisphere_mask(lat, hemisphere)

    # Combine with optional additional mask
    if mask is not None:
        hmask = hmask & mask

    # Create ice presence mask (concentration > threshold)
    if isinstance(sic, xr.DataArray):
        ice_mask = sic > threshold
        # Weights include hemisphere mask
        weights = xr.DataArray(area * hmask, dims=[sic.dims[-1]])
        return (ice_mask * weights).sum(dim=sic.dims[-1])
    else:
        ice_mask = sic > threshold
        return np.sum(ice_mask * area * hmask, axis=-1)
