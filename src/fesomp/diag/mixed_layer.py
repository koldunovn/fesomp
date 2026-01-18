"""Mixed layer depth diagnostics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr


def mixed_layer_depth(
    data: np.ndarray | xr.DataArray,
    depth_levels: np.ndarray,
    threshold: float,
    criterion: Literal["density", "temperature", "threshold"] = "threshold",
    reference_depth: float = 10.0,
) -> np.ndarray | xr.DataArray:
    """
    Compute mixed layer depth using threshold criterion.

    The mixed layer depth is defined as the depth where the property
    (density or temperature) differs from the surface (or reference depth)
    value by more than the threshold.

    Parameters
    ----------
    data : array-like
        Vertical profile data, shape (..., nlev, n2d).
        For density criterion: potential density in kg/m³.
        For temperature criterion: temperature in °C.
    depth_levels : np.ndarray
        Depth at level interfaces in meters, shape (nlev,).
    threshold : float
        Threshold value for criterion:
        - density: typical values 0.03 or 0.125 kg/m³
        - temperature: typical values 0.2 or 0.5 °C
    criterion : {"density", "temperature", "threshold"}
        Type of criterion:
        - "density": MLD where density > surface + threshold
        - "temperature": MLD where temperature < surface - threshold
        - "threshold": MLD where |data - surface| > threshold
    reference_depth : float
        Reference depth in meters for surface value. Default: 10.0
        Uses the level closest to this depth.

    Returns
    -------
    array-like
        Mixed layer depth in meters, shape (..., n2d).

    Notes
    -----
    Common threshold values:
    - de Boyer Montégut (2004): 0.03 kg/m³ density, 0.2°C temperature
    - Kara et al. (2000): 0.8°C temperature
    - Holte & Talley (2009): 0.03 kg/m³ density

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> # Using temperature criterion with 0.2°C threshold
    >>> temp = xr.open_dataset("temp.nc")["temp"]  # (time, lev, n2d)
    >>> mld = fesomp.diag.mixed_layer_depth(
    ...     temp,
    ...     mesh.depth_levels,
    ...     threshold=0.2,
    ...     criterion="temperature"
    ... )
    >>> # Using density criterion with 0.03 kg/m³ threshold
    >>> rho = xr.open_dataset("rho.nc")["rho"]  # (time, lev, n2d)
    >>> mld = fesomp.diag.mixed_layer_depth(
    ...     rho,
    ...     mesh.depth_levels,
    ...     threshold=0.03,
    ...     criterion="density"
    ... )
    """
    is_xarray = isinstance(data, xr.DataArray)

    # Find reference level (closest to reference_depth)
    ref_idx = np.argmin(np.abs(depth_levels - reference_depth))

    if is_xarray:
        dims = data.dims
        depth_dim = dims[-2]
        node_dim = dims[-1]

        # Get surface/reference value
        surface_val = data.isel({depth_dim: ref_idx})

        # Compute difference from surface
        if criterion == "density":
            # MLD where density exceeds surface + threshold
            diff = data - surface_val
            exceeded = diff > threshold
        elif criterion == "temperature":
            # MLD where temperature drops below surface - threshold
            diff = surface_val - data
            exceeded = diff > threshold
        else:  # "threshold"
            diff = np.abs(data - surface_val)
            exceeded = diff > threshold

        # Find first level where threshold is exceeded
        # Create depth array matching data dimensions
        depth_da = xr.DataArray(depth_levels, dims=[depth_dim])

        # Use where to mask depths where threshold not exceeded
        # Then take minimum (first exceedance depth)
        mld = depth_da.where(exceeded).min(dim=depth_dim)

        # Where threshold never exceeded, set to bottom depth
        max_depth = float(depth_levels[-1])
        mld = mld.fillna(max_depth)

        return mld

    else:
        # Pure numpy implementation
        # data shape: (..., nlev, n2d)
        nlev = data.shape[-2]

        # Get surface/reference value: (..., n2d)
        surface_val = data[..., ref_idx, :]

        # Compute difference from surface
        if criterion == "density":
            diff = data - surface_val[..., np.newaxis, :]
            exceeded = diff > threshold
        elif criterion == "temperature":
            diff = surface_val[..., np.newaxis, :] - data
            exceeded = diff > threshold
        else:
            diff = np.abs(data - surface_val[..., np.newaxis, :])
            exceeded = diff > threshold

        # Find first level where threshold exceeded
        # argmax returns first True, but returns 0 if all False
        first_exceeded = np.argmax(exceeded, axis=-2)

        # Check if threshold was ever exceeded
        any_exceeded = np.any(exceeded, axis=-2)

        # Get MLD from depth levels
        mld = depth_levels[first_exceeded]

        # Where threshold never exceeded, set to bottom
        mld = np.where(any_exceeded, mld, depth_levels[-1])

        return mld


def mixed_layer_depth_interpolated(
    data: np.ndarray | xr.DataArray,
    depth_levels: np.ndarray,
    threshold: float,
    criterion: Literal["density", "temperature", "threshold"] = "threshold",
    reference_depth: float = 10.0,
) -> np.ndarray | xr.DataArray:
    """
    Compute mixed layer depth with linear interpolation.

    Similar to mixed_layer_depth but interpolates between levels
    to find the exact depth where threshold is crossed.

    Parameters
    ----------
    data : array-like
        Vertical profile data, shape (..., nlev, n2d).
    depth_levels : np.ndarray
        Depth at level interfaces in meters, shape (nlev,).
    threshold : float
        Threshold value for criterion.
    criterion : {"density", "temperature", "threshold"}
        Type of criterion.
    reference_depth : float
        Reference depth in meters for surface value.

    Returns
    -------
    array-like
        Mixed layer depth in meters, shape (..., n2d).
        Interpolated to exact crossing depth.

    Examples
    --------
    >>> mld = fesomp.diag.mixed_layer_depth_interpolated(
    ...     temp,
    ...     mesh.depth_levels,
    ...     threshold=0.2,
    ...     criterion="temperature"
    ... )
    """
    is_xarray = isinstance(data, xr.DataArray)

    # Convert to numpy for interpolation
    if is_xarray:
        data_np = data.values
        dims = data.dims
        node_dim = dims[-1]
    else:
        data_np = data

    # Find reference level
    ref_idx = np.argmin(np.abs(depth_levels - reference_depth))

    # Get surface value
    surface_val = data_np[..., ref_idx, :]

    # Compute signed difference based on criterion
    if criterion == "density":
        diff = data_np - surface_val[..., np.newaxis, :]
        target = threshold  # Looking for diff > threshold
    elif criterion == "temperature":
        diff = surface_val[..., np.newaxis, :] - data_np
        target = threshold  # Looking for diff > threshold
    else:
        diff = np.abs(data_np - surface_val[..., np.newaxis, :])
        target = threshold

    # Shape: (..., nlev, n2d)
    nlev = diff.shape[-2]
    original_shape = diff.shape[:-2]
    n2d = diff.shape[-1]

    # Reshape for easier processing: (batch, nlev, n2d) -> (batch * n2d, nlev)
    if len(original_shape) > 0:
        batch_size = int(np.prod(original_shape))
        diff_flat = diff.reshape(batch_size, nlev, n2d)
        diff_flat = diff_flat.transpose(0, 2, 1).reshape(-1, nlev)
    else:
        diff_flat = diff.T  # (n2d, nlev)

    # Find interpolated MLD for each column
    mld_flat = np.zeros(diff_flat.shape[0])

    for i in range(diff_flat.shape[0]):
        profile = diff_flat[i]
        # Find first level where diff > target
        exceeded_mask = profile > target
        if not np.any(exceeded_mask):
            mld_flat[i] = depth_levels[-1]
        else:
            idx = np.argmax(exceeded_mask)
            if idx == 0:
                mld_flat[i] = depth_levels[0]
            else:
                # Linear interpolation between levels idx-1 and idx
                d0, d1 = depth_levels[idx - 1], depth_levels[idx]
                v0, v1 = profile[idx - 1], profile[idx]
                if v1 != v0:
                    # Interpolate to find where profile crosses target
                    frac = (target - v0) / (v1 - v0)
                    mld_flat[i] = d0 + frac * (d1 - d0)
                else:
                    mld_flat[i] = d0

    # Reshape back
    if len(original_shape) > 0:
        mld = mld_flat.reshape(batch_size, n2d)
        mld = mld.reshape(*original_shape, n2d)
    else:
        mld = mld_flat

    if is_xarray:
        # Recreate xarray with proper dimensions
        result_dims = dims[:-2] + (node_dim,)
        coords = {k: v for k, v in data.coords.items() if k in result_dims}
        return xr.DataArray(mld, dims=result_dims, coords=coords)

    return mld
