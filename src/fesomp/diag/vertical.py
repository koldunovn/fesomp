"""Vertical diagnostics: Hovmöller diagrams, volume means, heat content."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from .utils import compute_layer_thickness, get_surface_area, select_depth_indices


def hovmoller(
    data: np.ndarray | xr.DataArray,
    node_area: np.ndarray,
    depth: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray | xr.DataArray:
    """
    Compute area-weighted mean profile over time (Hovmöller diagram data).

    Computes the horizontal mean at each depth level for each time step,
    weighted by node areas.

    Parameters
    ----------
    data : array-like
        3D data, shape (time, nlev, n2d) for 3D fields.
        For xarray, dimensions should be named appropriately.
    node_area : np.ndarray
        Area at each node and level, shape (nlev, n2d) or (n2d,).
        If 1D, same area is used for all levels.
    depth : np.ndarray, optional
        Depth values for the vertical coordinate.
        If None and data is xarray, tries to infer from coordinates.
    mask : np.ndarray, optional
        Boolean mask, shape (n2d,). True = include, False = exclude.

    Returns
    -------
    array-like
        Area-weighted mean, shape (time, nlev).
        If xarray input with time and depth coords, returns xarray
        with proper coordinates.

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> temp = xr.open_dataset("temp.nc")["temp"]  # (time, lev, n2d)
    >>> # Compute global mean temperature profile over time
    >>> hovm = fesomp.diag.hovmoller(
    ...     temp,
    ...     mesh.geometry.node_area,
    ...     depth=mesh.depth_layers
    ... )
    """
    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        # Identify dimensions
        dims = data.dims
        # Assume last dim is spatial, second-to-last is depth
        node_dim = dims[-1]
        depth_dim = dims[-2] if len(dims) >= 2 else None
        time_dim = dims[0] if len(dims) >= 3 else None

        # Get depth coordinate if available
        if depth is None and depth_dim is not None:
            if depth_dim in data.coords:
                depth = data.coords[depth_dim].values

        if node_area.ndim == 2 and depth_dim is not None:
            nlev_data = data.sizes[depth_dim]
            nlev_area = node_area.shape[0]
            if nlev_area != nlev_data:
                diff = nlev_area - nlev_data
                if diff != 1:
                    raise ValueError(
                        "node_area has {0} vertical levels but data has {1}; "
                        "only node_area having one extra level is supported "
                        "(levels vs layers).".format(nlev_area, nlev_data)
                    )
                warnings.warn(
                    "node_area has one more vertical level than data; "
                    "using the first {0} levels of node_area to match data "
                    "(levels vs layers).".format(nlev_data),
                    UserWarning,
                    stacklevel=2,
                )
                node_area = node_area[:nlev_data, :]

        # Create weights DataArray
        if node_area.ndim == 1:
            weights = xr.DataArray(node_area, dims=[node_dim])
            if mask is not None:
                weights = weights * xr.DataArray(mask, dims=[node_dim])
        else:
            weights = xr.DataArray(node_area, dims=[depth_dim, node_dim])
            if mask is not None:
                mask_da = xr.DataArray(mask, dims=[node_dim])
                weights = weights * mask_da

        # Compute weighted mean along node dimension
        total_weight = weights.sum(dim=node_dim)
        result = (data * weights).sum(dim=node_dim) / total_weight

        return result

    else:
        # Pure numpy path
        # data shape: (time, nlev, n2d) or (nlev, n2d)
        if data.ndim == 2:
            # Single time step: (nlev, n2d)
            nlev, n2d = data.shape
            if node_area.ndim == 1:
                weights = np.broadcast_to(node_area, (nlev, n2d))
            else:
                nlev_area = node_area.shape[0]
                if nlev_area != nlev:
                    diff = nlev_area - nlev
                    if diff != 1:
                        raise ValueError(
                            "node_area has {0} vertical levels but data has {1}; "
                            "only node_area having one extra level is supported "
                            "(levels vs layers).".format(nlev_area, nlev)
                        )
                    warnings.warn(
                        "node_area has one more vertical level than data; "
                        "using the first {0} levels of node_area to match data "
                        "(levels vs layers).".format(nlev),
                        UserWarning,
                        stacklevel=2,
                    )
                    node_area = node_area[:nlev, :]
                weights = node_area

            if mask is not None:
                weights = weights * mask[np.newaxis, :]

            total_weight = weights.sum(axis=-1)
            return np.sum(data * weights, axis=-1) / total_weight

        elif data.ndim == 3:
            # Time series: (time, nlev, n2d)
            ntime, nlev, n2d = data.shape
            if node_area.ndim == 1:
                weights = np.broadcast_to(node_area, (nlev, n2d))
            else:
                nlev_area = node_area.shape[0]
                if nlev_area != nlev:
                    diff = nlev_area - nlev
                    if diff != 1:
                        raise ValueError(
                            "node_area has {0} vertical levels but data has {1}; "
                            "only node_area having one extra level is supported "
                            "(levels vs layers).".format(nlev_area, nlev)
                        )
                    warnings.warn(
                        "node_area has one more vertical level than data; "
                        "using the first {0} levels of node_area to match data "
                        "(levels vs layers).".format(nlev),
                        UserWarning,
                        stacklevel=2,
                    )
                    node_area = node_area[:nlev, :]
                weights = node_area

            if mask is not None:
                weights = weights * mask[np.newaxis, :]

            total_weight = weights.sum(axis=-1)  # (nlev,)
            # Sum over n2d, result is (time, nlev)
            return np.sum(data * weights[np.newaxis, :, :], axis=-1) / total_weight

        else:
            raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")


def volume_mean(
    data: np.ndarray | xr.DataArray,
    node_area: np.ndarray,
    depth_levels: np.ndarray,
    layer_thickness: np.ndarray | None = None,
    depth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray | xr.DataArray:
    """
    Compute volume-weighted mean over a depth range.

    Parameters
    ----------
    data : array-like
        3D data on layers, shape (..., nlev-1, n2d).
        Can have any leading dimensions (time, ensemble, etc.).
    node_area : np.ndarray
        Area at each node, shape (n2d,) or (nlev, n2d).
        If 2D, must have nlev rows (level interfaces).
    depth_levels : np.ndarray
        Depth at level interfaces in meters, shape (nlev,).
    layer_thickness : np.ndarray, optional
        Thickness of each layer in meters, shape (nlev-1,).
        If None, computed from depth_levels.
    depth_range : tuple[float, float], optional
        (min_depth, max_depth) in meters for averaging.
        If None, uses full depth range.
    mask : np.ndarray, optional
        Boolean mask for nodes, shape (n2d,). True = include.

    Returns
    -------
    array-like
        Volume-weighted mean, shape (...).

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> temp = xr.open_dataset("temp.nc")["temp"]  # (time, lev, n2d)
    >>> # Mean temperature in top 1000m
    >>> mean_temp = fesomp.diag.volume_mean(
    ...     temp,
    ...     mesh.geometry.node_area[0],
    ...     mesh.depth_levels,
    ...     depth_range=(0, 1000)
    ... )
    """
    # Compute layer thickness if not provided
    if layer_thickness is None:
        layer_thickness = compute_layer_thickness(depth_levels)

    # Select depth range
    start_idx, end_idx = select_depth_indices(depth_levels, depth_range)
    # For layers, we need indices into nlev-1 array
    layer_start = start_idx
    layer_end = min(end_idx, len(layer_thickness))

    # Get surface area
    area = get_surface_area(node_area)

    # Apply mask
    if mask is not None:
        area = area * mask

    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        dims = data.dims
        node_dim = dims[-1]
        depth_dim = dims[-2]

        # Select depth range
        data_sel = data.isel({depth_dim: slice(layer_start, layer_end)})
        thickness_sel = layer_thickness[layer_start:layer_end]

        # Create weights: area * thickness for each layer
        weights_2d = np.outer(thickness_sel, area)  # (nlayers_sel, n2d)
        weights = xr.DataArray(weights_2d, dims=[depth_dim, node_dim])

        # Compute volume-weighted mean
        total_volume = weights.sum(dim=[depth_dim, node_dim])
        result = (data_sel * weights).sum(dim=[depth_dim, node_dim]) / total_volume

        return result

    else:
        # Pure numpy
        # Select depth range
        data_sel = data[..., layer_start:layer_end, :]
        thickness_sel = layer_thickness[layer_start:layer_end]

        # Weights: area * thickness
        # thickness_sel: (nlayers,), area: (n2d,)
        # weights: (nlayers, n2d)
        weights = thickness_sel[:, np.newaxis] * area[np.newaxis, :]

        # Sum over depth and nodes
        # data_sel shape: (..., nlayers, n2d)
        total_volume = np.sum(weights)
        weighted_sum = np.sum(data_sel * weights, axis=(-2, -1))

        return weighted_sum / total_volume


def heat_content(
    temperature: np.ndarray | xr.DataArray,
    node_area: np.ndarray,
    depth_levels: np.ndarray,
    layer_thickness: np.ndarray | None = None,
    depth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    rho: float = 1025.0,
    cp: float = 3985.0,
    reference_temp: float = 0.0,
) -> np.ndarray | xr.DataArray:
    """
    Compute ocean heat content over a depth range.

    Heat content is computed as:
        OHC = sum(rho * cp * (T - T_ref) * volume)

    Parameters
    ----------
    temperature : array-like
        Temperature in °C, shape (..., nlev-1, n2d).
    node_area : np.ndarray
        Area at each node in m², shape (n2d,) or (nlev, n2d).
    depth_levels : np.ndarray
        Depth at level interfaces in meters, shape (nlev,).
    layer_thickness : np.ndarray, optional
        Thickness of each layer in meters, shape (nlev-1,).
    depth_range : tuple[float, float], optional
        (min_depth, max_depth) in meters.
        If None, uses full depth range.
    mask : np.ndarray, optional
        Boolean mask for nodes, shape (n2d,). True = include.
    rho : float
        Reference density in kg/m³. Default: 1025.0
    cp : float
        Specific heat capacity in J/(kg·K). Default: 3985.0
    reference_temp : float
        Reference temperature in °C. Default: 0.0

    Returns
    -------
    array-like
        Ocean heat content in Joules, shape (...).

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> temp = xr.open_dataset("temp.nc")["temp"]
    >>> # Heat content in top 700m (standard metric)
    >>> ohc_700 = fesomp.diag.heat_content(
    ...     temp,
    ...     mesh.geometry.node_area[0],
    ...     mesh.depth_levels,
    ...     depth_range=(0, 700)
    ... )
    >>> # Heat content in top 2000m
    >>> ohc_2000 = fesomp.diag.heat_content(
    ...     temp,
    ...     mesh.geometry.node_area[0],
    ...     mesh.depth_levels,
    ...     depth_range=(0, 2000)
    ... )
    """
    # Compute layer thickness if not provided
    if layer_thickness is None:
        layer_thickness = compute_layer_thickness(depth_levels)

    # Select depth range
    start_idx, end_idx = select_depth_indices(depth_levels, depth_range)
    layer_start = start_idx
    layer_end = min(end_idx, len(layer_thickness))

    # Get surface area
    area = get_surface_area(node_area)

    # Apply mask
    if mask is not None:
        area = area * mask

    is_xarray = isinstance(temperature, xr.DataArray)

    if is_xarray:
        dims = temperature.dims
        node_dim = dims[-1]
        depth_dim = dims[-2]

        # Select depth range
        temp_sel = temperature.isel({depth_dim: slice(layer_start, layer_end)})
        thickness_sel = layer_thickness[layer_start:layer_end]

        # Temperature anomaly
        temp_anom = temp_sel - reference_temp

        # Create volume weights: area * thickness for each layer
        weights_2d = np.outer(thickness_sel, area)  # (nlayers_sel, n2d)
        weights = xr.DataArray(weights_2d, dims=[depth_dim, node_dim])

        # Heat content = rho * cp * (T - T_ref) * volume
        ohc = rho * cp * (temp_anom * weights).sum(dim=[depth_dim, node_dim])

        return ohc

    else:
        # Pure numpy
        # Select depth range
        temp_sel = temperature[..., layer_start:layer_end, :]
        thickness_sel = layer_thickness[layer_start:layer_end]

        # Temperature anomaly
        temp_anom = temp_sel - reference_temp

        # Volume weights: (nlayers, n2d)
        volume = thickness_sel[:, np.newaxis] * area[np.newaxis, :]

        # Heat content
        ohc = rho * cp * np.sum(temp_anom * volume, axis=(-2, -1))

        return ohc


def total_volume(
    node_area: np.ndarray,
    depth_levels: np.ndarray,
    layer_thickness: np.ndarray | None = None,
    depth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> float:
    """
    Compute total ocean volume over a depth range.

    Parameters
    ----------
    node_area : np.ndarray
        Area at each node in m², shape (n2d,) or (nlev, n2d).
    depth_levels : np.ndarray
        Depth at level interfaces in meters, shape (nlev,).
    layer_thickness : np.ndarray, optional
        Thickness of each layer in meters, shape (nlev-1,).
    depth_range : tuple[float, float], optional
        (min_depth, max_depth) in meters.
    mask : np.ndarray, optional
        Boolean mask for nodes, shape (n2d,). True = include.

    Returns
    -------
    float
        Total volume in m³.
    """
    # Compute layer thickness if not provided
    if layer_thickness is None:
        layer_thickness = compute_layer_thickness(depth_levels)

    # Select depth range
    start_idx, end_idx = select_depth_indices(depth_levels, depth_range)
    layer_start = start_idx
    layer_end = min(end_idx, len(layer_thickness))
    thickness_sel = layer_thickness[layer_start:layer_end]

    # Get surface area
    area = get_surface_area(node_area)

    # Apply mask
    if mask is not None:
        area = area * mask

    # Total volume = sum(area * thickness)
    return float(np.sum(thickness_sel[:, np.newaxis] * area[np.newaxis, :]))
