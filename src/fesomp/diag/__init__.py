"""
FESOM2 Diagnostics Module.

This module provides oceanographic diagnostic calculations for FESOM2 model output.

Sea Ice Diagnostics
-------------------
- ice_area : Compute sea ice area (concentration × cell area)
- ice_volume : Compute sea ice volume (thickness × cell area)
- ice_extent : Compute sea ice extent (area above threshold)

Vertical Diagnostics
--------------------
- hovmoller : Area-weighted mean profile over time
- volume_mean : Volume-weighted mean over depth range
- heat_content : Ocean heat content over depth range
- total_volume : Total ocean volume over depth range

Mixed Layer
-----------
- mixed_layer_depth : Mixed layer depth using threshold criterion
- mixed_layer_depth_interpolated : MLD with linear interpolation

Meridional Overturning
----------------------
- moc : Meridional overturning circulation from vertical velocity
- amoc_index : AMOC index at specified latitude
- get_basin_mask : Boolean mask for named ocean basins
- list_moc_basins : List available basin names

Utilities
---------
- hemisphere_mask : Create mask for Northern/Southern hemisphere
- get_region_mask : Create mask for named regions
- list_available_regions : List available region names
- compute_layer_thickness : Compute layer thickness from level depths

Examples
--------
>>> import fesomp
>>> mesh = fesomp.load_mesh("path/to/mesh.nc")
>>>
>>> # Sea ice extent
>>> sic = xr.open_dataset("a_ice.nc")["a_ice"]
>>> extent = fesomp.diag.ice_extent(
...     sic,
...     mesh.geometry.node_area[0],
...     mesh.lat,
...     hemisphere="N"
... )
>>>
>>> # Ocean heat content in top 700m
>>> temp = xr.open_dataset("temp.nc")["temp"]
>>> ohc = fesomp.diag.heat_content(
...     temp,
...     mesh.geometry.node_area[0],
...     mesh.depth_levels,
...     depth_range=(0, 700)
... )
>>>
>>> # Atlantic MOC
>>> w = xr.open_dataset("w.nc")["w"]
>>> atl_mask = fesomp.diag.get_basin_mask(
...     mesh.lon_elem, mesh.lat_elem, "Atlantic_MOC"
... )
>>> amoc, lats = fesomp.diag.moc(
...     w,
...     mesh.geometry.elem_area,
...     mesh.lat_elem,
...     mesh.depth_levels,
...     mask=atl_mask
... )
"""

# Sea ice diagnostics
from fesomp.diag.ice import ice_area, ice_extent, ice_volume

# Mixed layer diagnostics
from fesomp.diag.mixed_layer import mixed_layer_depth, mixed_layer_depth_interpolated

# MOC diagnostics
from fesomp.diag.moc import amoc_index, get_basin_mask, list_moc_basins, moc

# Utilities
from fesomp.diag.utils import (
    compute_layer_thickness,
    get_region_mask,
    get_surface_area,
    hemisphere_mask,
    list_available_regions,
    select_depth_indices,
)

# Vertical diagnostics
from fesomp.diag.vertical import heat_content, hovmoller, total_volume, volume_mean

__all__ = [
    # Sea ice
    "ice_area",
    "ice_volume",
    "ice_extent",
    # Vertical
    "hovmoller",
    "volume_mean",
    "heat_content",
    "total_volume",
    # Mixed layer
    "mixed_layer_depth",
    "mixed_layer_depth_interpolated",
    # MOC
    "moc",
    "amoc_index",
    "get_basin_mask",
    "list_moc_basins",
    # Utilities
    "hemisphere_mask",
    "get_region_mask",
    "list_available_regions",
    "compute_layer_thickness",
    "select_depth_indices",
    "get_surface_area",
]
