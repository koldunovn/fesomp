Quick Start
===========

This guide will help you get started with FESOMP quickly.

Loading a Mesh
--------------

FESOMP can load meshes from NetCDF files or ASCII directories:

.. code-block:: python

   import fesomp

   # From NetCDF (recommended)
   mesh = fesomp.load_mesh("path/to/fesom.mesh.diag.nc")

   # From ASCII directory
   mesh = fesomp.load_mesh("path/to/mesh/directory/")

   print(mesh)
   # Mesh(n2d=126859, nelem=237843, nlev=47, lon=[-180.00, 180.00], lat=[-90.00, 90.00])

Plotting 2D Data
----------------

Create maps with Cartopy integration:

.. code-block:: python

   import xarray as xr

   # Load surface temperature
   sst = xr.open_dataset("sst.fesom.1958.nc")['sst'][0, :].values

   # Create a map
   fig, axes, interp = fesomp.plot(
       sst,
       mesh.lon,
       mesh.lat,
       mapproj="robinson",
       title="Sea Surface Temperature",
       units="°C",
       cmap="RdYlBu_r",
   )

Vertical Transects
------------------

Create vertical cross-sections through 3D data:

.. code-block:: python

   # Load 3D temperature data
   temp_3d = xr.open_dataset("temp.fesom.1958.nc")['temp'][0, :, :].values

   # Atlantic meridional transect
   fig, ax, interp = fesomp.transect(
       temp_3d,  # shape: (nlev, n2d)
       mesh,
       start=(-30, -60),  # 30°W, 60°S
       end=(-30, 60),     # 30°W, 60°N
       title="Temperature along 30°W",
       units="°C",
       depth_limits=(0, 2000),  # Top 2000 meters
   )

Interpolation to Regular Grid
-----------------------------

Interpolate unstructured data to a regular lon/lat grid:

.. code-block:: python

   # Quick interpolation
   data_reg, lon_reg, lat_reg = fesomp.regrid(
       sst,
       mesh.lon,
       mesh.lat,
       res=(360, 180),
       method="idw",
   )

   # Reusable interpolator (faster for multiple variables)
   interp = fesomp.RegridInterpolator(
       mesh.lon, mesh.lat,
       res=(360, 180),
       method="idw",
   )

   temp_reg, lon_reg, lat_reg = interp(sst)
   salt_reg, _, _ = interp(salinity)  # Reuses pre-computed weights

Spatial Queries
---------------

Find nodes by location:

.. code-block:: python

   # Find nearest nodes
   nearest_idx = mesh.find_nearest(lon=10.5, lat=54.3, k=5)

   # Find nodes within radius
   indices = mesh.find_in_radius(lon=0, lat=0, radius_km=100)

   # Bounding box query
   indices = mesh.subset_by_bbox(
       lon_min=-10, lon_max=10,
       lat_min=40, lat_max=60
   )

Diagnostics
-----------

Calculate oceanographic diagnostics:

.. code-block:: python

   # Sea ice extent
   sic = xr.open_dataset("a_ice.nc")["a_ice"]
   extent = fesomp.diag.ice_extent(
       sic,
       mesh.geometry.node_area[0],
       mesh.lat,
       hemisphere="N"
   )

   # Ocean heat content in top 700m
   temp = xr.open_dataset("temp.nc")["temp"]
   ohc = fesomp.diag.heat_content(
       temp,
       mesh.geometry.node_area[0],
       mesh.depth_levels,
       depth_range=(0, 700)
   )

Next Steps
----------

- :doc:`user_guide/mesh` - Learn more about mesh operations
- :doc:`user_guide/plotting` - Advanced plotting options
- :doc:`user_guide/transects` - Transect configuration
- :doc:`user_guide/diagnostics` - Diagnostic calculations
- :doc:`api/fesomp` - Full API reference
