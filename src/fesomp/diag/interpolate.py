"""Interpolation between elements and nodes.

This module provides functions to interpolate data between element centers
and node locations on the FESOM2 unstructured mesh.

The interpolation accounts for the varying depth structure of the mesh:
- At deeper levels, some nodes become inactive (shallower than the level)
- At deeper levels, some elements become inactive (all vertices shallower)
- The node area at each level only includes contributions from active elements
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def elem_to_nodes(
    data: np.ndarray | xr.DataArray,
    triangles: np.ndarray,
    elem_area: np.ndarray,
    node_area: np.ndarray,
) -> np.ndarray | xr.DataArray:
    """
    Interpolate 2D data from element centers to nodes.

    Uses an area-weighted scheme where each element contributes its value
    weighted by element area to all three vertices. The result at each node
    is the area-weighted average of values from adjacent elements.

    Parameters
    ----------
    data : array-like
        Data at element centers, shape (..., nelem).
        Can have any leading dimensions (time, ensemble, etc.).
    triangles : np.ndarray
        Triangle connectivity, shape (nelem, 3), 0-indexed.
    elem_area : np.ndarray
        Area of each element in m², shape (nelem,).
    node_area : np.ndarray
        Area associated with each node in m², shape (n2d,).
        This is the sum of 1/3 of adjacent element areas (the "lump").
        Use ``mesh.geometry.node_area[0]`` for surface data.

    Returns
    -------
    array-like
        Data at nodes, shape (..., n2d).
        Same type as input (numpy or xarray).

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> # Data on elements (e.g., computed from velocity gradients)
    >>> data_elem = np.random.rand(mesh.nelem)
    >>> # Interpolate to nodes
    >>> data_nodes = fesomp.diag.elem_to_nodes(
    ...     data_elem,
    ...     mesh.triangles,
    ...     mesh.geometry.elem_area,
    ...     mesh.geometry.node_area[0],
    ... )

    Notes
    -----
    This is an area-weighted interpolation scheme. For each node, the result is:

    .. math::

        f_{node} = \\frac{\\sum_{adj\\_elems} f_{elem} \\cdot A_{elem}}
                        {\\sum_{adj\\_elems} A_{elem}}

    This is equivalent to the "lump" interpolation used in FESOM2.
    """
    is_xarray = isinstance(data, xr.DataArray)
    if is_xarray:
        data_np = data.values.astype(np.float64)
        original_dims = data.dims
        original_coords = {k: v for k, v in data.coords.items() if k != data.dims[-1]}
    else:
        data_np = np.asarray(data, dtype=np.float64)

    triangles = np.asarray(triangles, dtype=np.intp)
    elem_area = np.asarray(elem_area, dtype=np.float64)
    node_area = np.asarray(node_area, dtype=np.float64)

    n2d = len(node_area)
    nelem = len(elem_area)

    # Handle arbitrary leading dimensions
    original_shape = data_np.shape
    if data_np.ndim > 1:
        # Reshape to (n_leading, nelem)
        n_leading = int(np.prod(original_shape[:-1]))
        data_flat = data_np.reshape(n_leading, nelem)
    else:
        data_flat = data_np[np.newaxis, :]
        n_leading = 1

    # Weight data by element area
    weighted_data = data_flat * elem_area  # (n_leading, nelem)

    # Accumulate contributions at nodes
    # Each element contributes to its 3 vertices
    result = np.zeros((n_leading, n2d), dtype=np.float64)
    for i in range(3):
        np.add.at(result, (slice(None), triangles[:, i]), weighted_data)

    # Normalize by node area * 3
    # (node_area is sum of elem_area/3 for adjacent elements,
    # so node_area * 3 = sum of elem_area for adjacent elements)
    result = result / (node_area * 3.0)

    # Reshape back to original leading dimensions
    if data_np.ndim > 1:
        output_shape = original_shape[:-1] + (n2d,)
        result = result.reshape(output_shape)
    else:
        result = result[0]

    if is_xarray:
        new_dims = original_dims[:-1] + ("nod2",)
        result = xr.DataArray(result, dims=new_dims, coords=original_coords)

    return result


def elem_to_nodes_3d(
    data: np.ndarray | xr.DataArray,
    triangles: np.ndarray,
    elem_area: np.ndarray,
    node_area: np.ndarray,
    elem_levels: np.ndarray | None = None,
) -> np.ndarray | xr.DataArray:
    """
    Interpolate 3D data from element centers to nodes.

    Handles data with a vertical dimension, properly accounting for the
    varying depth structure of the mesh. At each level, uses the
    level-appropriate node area for normalization.

    Parameters
    ----------
    data : array-like
        Data at element centers, shape (nelem, nlev) or (time, nelem, nlev).
        The element dimension must be second-to-last, levels last.
    triangles : np.ndarray
        Triangle connectivity, shape (nelem, 3), 0-indexed.
    elem_area : np.ndarray
        Area of each element in m², shape (nelem,).
    node_area : np.ndarray
        Area associated with each node at each level, shape (nlev, n2d).
        Use ``mesh.geometry.node_area`` (the full 2D array).
        This is the ground truth from the model output that already accounts
        for inactive triangles at each level. At each level, inactive nodes
        have area 0.
    elem_levels : np.ndarray, optional
        Number of active levels at each element, shape (nelem,).
        Use ``mesh.elem_levels``. If provided, elements inactive at a
        given level (where elem_levels <= level) will not contribute to
        the interpolation. Recommended for consistency with node_area.

    Returns
    -------
    array-like
        Data at nodes, shape (n2d, nlev) or (time, n2d, nlev).
        Same type as input. Inactive nodes at each level will have NaN values.

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> # 3D data on elements, shape (time, nelem, nlev)
    >>> data_3d = np.random.rand(12, mesh.nelem, mesh.nlev)
    >>> # Interpolate to nodes with level-aware areas
    >>> data_nodes = fesomp.diag.elem_to_nodes_3d(
    ...     data_3d,
    ...     mesh.triangles,
    ...     mesh.geometry.elem_area,
    ...     mesh.geometry.node_area,
    ...     mesh.elem_levels,
    ... )
    >>> # Result has shape (12, n2d, nlev)

    Notes
    -----
    At each level k:
    - Only nodes with node_area[k] > 0 are active (receive valid values)
    - If elem_levels is provided, only elements with elem_levels > k contribute
    - Inactive nodes receive NaN values

    The vertical structure is important because:
    - Shallow nodes become inactive at deeper levels (topography)
    - Elements touching inactive nodes also become inactive
    - The node_area from the model already accounts for this, providing
      the correct normalization at each level
    """
    is_xarray = isinstance(data, xr.DataArray)
    if is_xarray:
        data_np = data.values.astype(np.float64)
        original_dims = data.dims
        original_coords = {k: v for k, v in data.coords.items()}
    else:
        data_np = np.asarray(data, dtype=np.float64)

    triangles = np.asarray(triangles, dtype=np.intp)
    elem_area = np.asarray(elem_area, dtype=np.float64)
    node_area = np.asarray(node_area, dtype=np.float64)

    if elem_levels is not None:
        elem_levels = np.asarray(elem_levels, dtype=np.int32)

    nelem = len(elem_area)
    nlev, n2d = node_area.shape

    # Validate data shape: expect (..., nelem, nlev)
    if data_np.shape[-2] != nelem:
        raise ValueError(
            f"Data shape {data_np.shape} does not match expected (..., nelem, nlev). "
            f"Got {data_np.shape[-2]} elements, expected {nelem}."
        )
    if data_np.shape[-1] != nlev:
        raise ValueError(
            f"Data shape {data_np.shape} does not match expected (..., nelem, nlev). "
            f"Got {data_np.shape[-1]} levels, expected {nlev}."
        )

    # Handle leading dimensions
    original_shape = data_np.shape
    leading_shape = original_shape[:-2]
    if len(leading_shape) > 0:
        n_leading = int(np.prod(leading_shape))
        data_np = data_np.reshape(n_leading, nelem, nlev)
    else:
        data_np = data_np[np.newaxis, :, :]
        n_leading = 1

    # Allocate output: (n_leading, n2d, nlev)
    result = np.zeros((n_leading, n2d, nlev), dtype=np.float64)

    # Process each level with its corresponding node area
    for lev in range(nlev):
        # Get data for this level: (n_leading, nelem)
        data_lev = data_np[:, :, lev]

        # Determine which elements are active at this level
        if elem_levels is not None:
            elem_active = elem_levels > lev
            elem_area_lev = elem_area * elem_active
        else:
            elem_area_lev = elem_area

        # Weight data by element area (masking inactive elements if elem_levels provided)
        weighted_data = data_lev * elem_area_lev  # (n_leading, nelem)

        # Accumulate contributions at nodes
        accum = np.zeros((n_leading, n2d), dtype=np.float64)
        for i in range(3):
            np.add.at(accum, (slice(None), triangles[:, i]), weighted_data)

        # Get node area for this level (from model output, already accounts for
        # inactive triangles - this is the ground truth)
        node_area_lev = node_area[lev]  # (n2d,)

        # Normalize, handling inactive nodes (area = 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[:, :, lev] = accum / (node_area_lev * 3.0)

        # Set inactive nodes to NaN (nodes with zero area at this level)
        inactive_nodes = node_area_lev == 0
        result[:, inactive_nodes, lev] = np.nan

    # Reshape back
    output_shape = leading_shape + (n2d, nlev)
    result = result.reshape(output_shape)

    if is_xarray:
        # Replace elem dimension with nod2
        new_dims = original_dims[:-2] + ("nod2",) + (original_dims[-1],)
        # Keep coords that don't depend on elem dimension
        new_coords = {
            k: v for k, v in original_coords.items()
            if k not in [original_dims[-2]]
        }
        result = xr.DataArray(result, dims=new_dims, coords=new_coords)

    return result


def compute_node_lump(
    triangles: np.ndarray,
    elem_area: np.ndarray,
    n2d: int,
) -> np.ndarray:
    """
    Compute the surface lumped mass (node area) from element connectivity.

    This computes the surface node area, which is the sum of 1/3 of
    adjacent element areas for each node. This is equivalent to
    ``mesh.geometry.node_area[0]``.

    Parameters
    ----------
    triangles : np.ndarray
        Triangle connectivity, shape (nelem, 3), 0-indexed.
    elem_area : np.ndarray
        Area of each element in m², shape (nelem,).
    n2d : int
        Number of nodes.

    Returns
    -------
    np.ndarray
        Lumped mass / surface node area, shape (n2d,).

    Examples
    --------
    >>> import fesomp
    >>> mesh = fesomp.load_mesh("path/to/mesh.nc")
    >>> lump = fesomp.diag.compute_node_lump(
    ...     mesh.triangles,
    ...     mesh.geometry.elem_area,
    ...     mesh.n2d,
    ... )
    >>> # This is equivalent to mesh.geometry.node_area[0]
    >>> np.allclose(lump, mesh.geometry.node_area[0])
    True
    """
    triangles = np.asarray(triangles, dtype=np.intp)
    elem_area = np.asarray(elem_area, dtype=np.float64)

    node_lump = np.zeros(n2d, dtype=np.float64)
    contribution = elem_area / 3.0

    np.add.at(node_lump, triangles[:, 0], contribution)
    np.add.at(node_lump, triangles[:, 1], contribution)
    np.add.at(node_lump, triangles[:, 2], contribution)

    return node_lump
