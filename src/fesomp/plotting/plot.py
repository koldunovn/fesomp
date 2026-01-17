"""Plotting functions for unstructured data."""

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from fesomp.plotting.regrid import RegridInterpolator, regrid

# Map projection aliases
PROJECTIONS = {
    "pc": "PlateCarree",
    "platecarree": "PlateCarree",
    "rob": "Robinson",
    "robinson": "Robinson",
    "merc": "Mercator",
    "mercator": "Mercator",
    "np": "NorthPolarStereo",
    "northpolar": "NorthPolarStereo",
    "sp": "SouthPolarStereo",
    "southpolar": "SouthPolarStereo",
    "ortho": "Orthographic",
    "orthographic": "Orthographic",
}

# Projections that need set_global() instead of set_extent()
GLOBAL_PROJECTIONS = {"rob", "robinson", "ortho", "orthographic"}

# Projections where contourf has GEOS geometry issues - use pcolormesh instead
# This includes all non-rectangular projections
PCOLORMESH_ONLY_PROJECTIONS = {
    "rob", "robinson",
    "ortho", "orthographic",
    "np", "northpolar",
    "sp", "southpolar",
}

# Default bounding boxes for specific projections (display extent)
DEFAULT_BOXES = {
    "np": (-180, 180, 60, 90),
    "northpolar": (-180, 180, 60, 90),
    "sp": (-180, 180, -90, -60),
    "southpolar": (-180, 180, -90, -60),
}


def _get_interpolation_box(box: tuple, mapproj: str) -> tuple:
    """Get expanded box for interpolation to fill map corners.

    For polar stereographic projections, the square map corners extend
    beyond the latitude range of the edges. We expand the interpolation
    box by factor of sqrt(2) to ensure corners are filled.
    """
    lon_min, lon_max, lat_min, lat_max = box
    proj_lower = mapproj.lower()

    if proj_lower in ("np", "northpolar"):
        # North polar: expand southward
        lat_range = 90 - lat_min  # degrees from pole to edge
        expanded_range = lat_range * 1.42  # sqrt(2) ≈ 1.414
        new_lat_min = max(-90, 90 - expanded_range)
        return (lon_min, lon_max, new_lat_min, lat_max)

    elif proj_lower in ("sp", "southpolar"):
        # South polar: expand northward
        lat_range = lat_max - (-90)  # degrees from pole to edge
        expanded_range = lat_range * 1.42
        new_lat_max = min(90, -90 + expanded_range)
        return (lon_min, lon_max, lat_min, new_lat_max)

    return box


def _get_projection(name: str, **kwargs):
    """Get cartopy projection by name."""
    import cartopy.crs as ccrs

    name_lower = name.lower()
    proj_name = PROJECTIONS.get(name_lower, name)

    proj_class = getattr(ccrs, proj_name, None)
    if proj_class is None:
        raise ValueError(
            f"Unknown projection: {name}. Available: {list(PROJECTIONS.keys())}"
        )

    return proj_class(**kwargs)


def _parse_levels(
    levels: tuple | list | None, data: np.ndarray, nlevels: int = 40
) -> np.ndarray | None:
    """Parse levels specification.

    Formats:
    - None: auto from data min/max
    - (min, max, n): linspace(min, max, n)
    - [v1, v2, v3, ...]: explicit levels
    """
    if levels is None:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        return np.linspace(vmin, vmax, nlevels)

    if len(levels) == 3 and isinstance(levels[2], int):
        return np.linspace(levels[0], levels[1], levels[2])

    return np.array(levels)


def plot(
    data: np.ndarray | Sequence[np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    # Interpolation options
    box: tuple[float, float, float, float] | None = None,
    res: tuple[int, int] = (360, 180),
    interp: Literal["nn", "idw", "linear"] = "nn",
    influence: float = 80000,
    interpolator: RegridInterpolator | None = None,
    # Plot options
    cmap: str | None = None,
    levels: tuple | list | None = None,
    ptype: Literal["cf", "pcm"] = "cf",
    mapproj: str = "pc",
    # Figure options
    figsize: tuple[float, float] = (10, 6),
    rowscol: tuple[int, int] = (1, 1),
    # Labels
    titles: str | list[str] | None = None,
    units: str | None = None,
    colorbar: bool = True,
    # Coastlines and features
    coastlines: bool = True,
    land: bool = False,
    gridlines: bool = False,
    # Output
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
) -> tuple[plt.Figure, np.ndarray, RegridInterpolator]:
    """
    Plot unstructured data on a map.

    Data is interpolated to a regular grid and plotted with cartopy.

    Parameters
    ----------
    data : np.ndarray or list of np.ndarray
        Data to plot. Can be a single array (npoints,) or a list of arrays
        for multiple subplots.
    lon : np.ndarray
        Longitudes of data points in degrees.
    lat : np.ndarray
        Latitudes of data points in degrees.
    box : tuple, optional
        Bounding box (lon_min, lon_max, lat_min, lat_max).
        Default depends on projection:
        - 'np': (-180, 180, 60, 90)
        - 'sp': (-180, 180, -90, -60)
        - others: (-180, 180, -90, 90)
    res : tuple
        Interpolation resolution (nlon, nlat). Default is (360, 180).
    interp : str
        Interpolation method: 'nn' (nearest neighbor), 'idw', or 'linear'.
    influence : float
        Radius of influence for interpolation in meters. Default is 80000.
    interpolator : RegridInterpolator, optional
        Pre-computed interpolator for caching. Speeds up repeated plots.
    cmap : str, optional
        Colormap name. Default is 'RdBu_r'.
    levels : tuple or list, optional
        Contour levels. Can be (min, max, nlevels) or explicit list.
        Default is auto from data.
    ptype : str
        Plot type: 'cf' (contourf) or 'pcm' (pcolormesh).
    mapproj : str
        Map projection: 'pc' (Plate Carree), 'rob' (Robinson),
        'merc' (Mercator), 'np' (North Polar), 'sp' (South Polar).
    figsize : tuple
        Figure size in inches.
    rowscol : tuple
        Subplot layout (nrows, ncols).
    titles : str or list
        Title(s) for subplot(s).
    units : str, optional
        Units label for colorbar.
    colorbar : bool
        Whether to show colorbar. Default is True.
    coastlines : bool
        Whether to draw coastlines. Default is True.
    land : bool
        Whether to fill land areas. Default is False.
    gridlines : bool
        Whether to draw gridlines. Default is False.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on (for single plot).
    fig : matplotlib.figure.Figure, optional
        Existing figure to use.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray
        Array of axes objects.
    interpolator : RegridInterpolator
        The interpolator used (can be reused for subsequent plots).

    Example
    -------
    >>> # Simple plot
    >>> fig, axes, interp = fesomp.plot(temp, mesh.lon, mesh.lat)
    >>>
    >>> # Multiple subplots with cached interpolator
    >>> fig, axes, interp = fesomp.plot(
    ...     [temp_surface, temp_100m, temp_500m],
    ...     mesh.lon, mesh.lat,
    ...     rowscol=(1, 3),
    ...     titles=['Surface', '100m', '500m'],
    ...     interpolator=interp,  # reuse from previous
    ... )
    """
    import cartopy.crs as ccrs

    # Set default box based on projection if not specified
    if box is None:
        box = DEFAULT_BOXES.get(mapproj.lower(), (-180, 180, -90, 90))

    # Get expanded box for interpolation (to fill corners in polar projections)
    interp_box = _get_interpolation_box(box, mapproj)

    # Handle single vs multiple data arrays
    if isinstance(data, np.ndarray) and data.ndim == 1:
        data_list = [data]
    else:
        data_list = list(data)

    nplots = len(data_list)
    nrows, ncols = rowscol

    if nrows * ncols < nplots:
        raise ValueError(
            f"rowscol={rowscol} only provides {nrows*ncols} subplots, "
            f"but {nplots} data arrays were given"
        )

    # Default colormap
    if cmap is None:
        cmap = "RdBu_r"

    # Create or reuse interpolator
    if interpolator is None:
        interpolator = RegridInterpolator(
            lon=lon,
            lat=lat,
            box=interp_box,  # Use expanded box for polar projections
            res=res,
            method=interp,
            influence=influence,
        )

    # Get projection
    proj = _get_projection(mapproj)
    data_crs = ccrs.PlateCarree()

    # Create figure and axes
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if ax is not None and nplots == 1:
        axes = np.array([ax])
    else:
        axes = []
        for i in range(nplots):
            ax_i = fig.add_subplot(nrows, ncols, i + 1, projection=proj)
            axes.append(ax_i)
        axes = np.array(axes)

    # Handle titles
    if titles is None:
        titles_list = [None] * nplots
    elif isinstance(titles, str):
        titles_list = [titles] if nplots == 1 else [titles] * nplots
    else:
        titles_list = list(titles)

    # Check if this is a global projection
    is_global_proj = mapproj.lower() in GLOBAL_PROJECTIONS

    # For non-rectangular projections, force pcolormesh - contourf has rendering
    # issues with cartopy/shapely geometry projection (GEOSException on empty points)
    effective_ptype = ptype
    if mapproj.lower() in PCOLORMESH_ONLY_PROJECTIONS and ptype == "cf":
        effective_ptype = "pcm"

    # Interpolate and plot each data array
    mappables = []
    for i, (data_i, ax_i, title_i) in enumerate(zip(data_list, axes.flat, titles_list)):
        # Set global extent BEFORE plotting for global projections
        if is_global_proj:
            ax_i.set_global()

        # Interpolate
        data_reg, lon_reg, lat_reg = interpolator(data_i)

        # Parse levels
        levels_arr = _parse_levels(levels, data_reg)

        # Plot
        if effective_ptype == "cf":
            im = ax_i.contourf(
                lon_reg,
                lat_reg,
                data_reg,
                levels=levels_arr,
                cmap=cmap,
                transform=data_crs,
                extend="both",
            )
        else:  # pcm
            im = ax_i.pcolormesh(
                lon_reg,
                lat_reg,
                data_reg,
                cmap=cmap,
                transform=data_crs,
                vmin=levels_arr.min() if levels_arr is not None else None,
                vmax=levels_arr.max() if levels_arr is not None else None,
            )

        mappables.append(im)

        # Map features
        if coastlines:
            ax_i.coastlines(linewidth=0.5)

        if land:
            import cartopy.feature as cfeature
            ax_i.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)

        if gridlines:
            ax_i.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

        # Set extent based on projection type
        if mapproj.lower() in ("pc", "platecarree", "merc", "mercator"):
            ax_i.set_extent(box, crs=data_crs)
        elif mapproj.lower() in ("np", "northpolar", "sp", "southpolar"):
            # For polar projections, set extent to display box (not expanded interp box)
            ax_i.set_extent(box, crs=data_crs)

        # Title
        if title_i:
            ax_i.set_title(title_i)

    # Colorbar
    if colorbar and mappables:
        # Single colorbar for all subplots
        cbar = fig.colorbar(
            mappables[0],
            ax=axes.tolist(),
            orientation="horizontal",
            fraction=0.05,
            pad=0.08,
            shrink=0.8,
        )
        if units:
            cbar.set_label(units)

    # Note: tight_layout doesn't work well with cartopy, skip it
    # Users can call fig.tight_layout() or use constrained_layout if needed

    return fig, axes, interpolator
