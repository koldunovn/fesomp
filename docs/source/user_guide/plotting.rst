Plotting
========

FESOMP provides visualization tools for unstructured data with Cartopy integration.

2D Map Plots
------------

Basic Plot
~~~~~~~~~~

.. code-block:: python

   import fesomp
   import xarray as xr

   mesh = fesomp.load_mesh("mesh.nc")
   sst = xr.open_dataset("sst.nc")['sst'][0, :].values

   fig, ax, interp = fesomp.plot(
       sst,
       mesh.lon,
       mesh.lat,
       title="Sea Surface Temperature",
   )

Map Projections
~~~~~~~~~~~~~~~

Specify different map projections:

.. code-block:: python

   # Robinson projection (good for global maps)
   fesomp.plot(sst, mesh.lon, mesh.lat, mapproj="robinson")

   # Plate Carree (equirectangular)
   fesomp.plot(sst, mesh.lon, mesh.lat, mapproj="platecarree")

   # North Polar Stereographic
   fesomp.plot(sst, mesh.lon, mesh.lat, mapproj="np")

   # South Polar Stereographic
   fesomp.plot(sst, mesh.lon, mesh.lat, mapproj="sp")

   # Mercator
   fesomp.plot(sst, mesh.lon, mesh.lat, mapproj="mercator")

Customizing Appearance
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, ax, interp = fesomp.plot(
       sst,
       mesh.lon,
       mesh.lat,
       cmap="RdYlBu_r",          # Colormap
       vmin=-2, vmax=30,          # Color limits
       units="°C",                # Colorbar label
       title="Sea Surface Temperature",
       levels=20,                 # Number of contour levels
       extend="both",             # Colorbar extend option
   )

Regional Plots
~~~~~~~~~~~~~~

Specify a region with bounding box:

.. code-block:: python

   fesomp.plot(
       sst,
       mesh.lon,
       mesh.lat,
       box=[-80, 0, 0, 65],  # [lon_min, lon_max, lat_min, lat_max]
       title="North Atlantic SST",
   )

Interpolation Methods
~~~~~~~~~~~~~~~~~~~~~

Control how data is interpolated to the regular grid:

.. code-block:: python

   # Inverse distance weighting (default, smooth)
   fesomp.plot(sst, mesh.lon, mesh.lat, method="idw")

   # Nearest neighbor (fast, preserves extremes)
   fesomp.plot(sst, mesh.lon, mesh.lat, method="nn")

   # Linear interpolation (most accurate)
   fesomp.plot(sst, mesh.lon, mesh.lat, method="linear")

Regridding to Regular Grids
---------------------------

Interpolate unstructured data to a regular lon/lat grid:

.. code-block:: python

   # Quick interpolation
   data_reg, lon_reg, lat_reg = fesomp.regrid(
       sst,
       mesh.lon,
       mesh.lat,
       res=(360, 180),  # Resolution (nlon, nlat)
       method="idw",
   )

Reusable Interpolator
~~~~~~~~~~~~~~~~~~~~~

For processing multiple variables efficiently:

.. code-block:: python

   # Create interpolator once
   interp = fesomp.RegridInterpolator(
       mesh.lon, mesh.lat,
       res=(360, 180),
       method="idw",
   )

   # Use for multiple variables
   temp_reg, lon_reg, lat_reg = interp(temperature)
   salt_reg, _, _ = interp(salinity)
   ssh_reg, _, _ = interp(ssh)

Resolution Options
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Specify resolution as (nlon, nlat)
   fesomp.regrid(sst, mesh.lon, mesh.lat, res=(720, 360))

   # Specify bounding box
   fesomp.regrid(
       sst, mesh.lon, mesh.lat,
       res=(100, 100),
       box=[-80, 0, 0, 65],  # North Atlantic
   )

Influence Radius
~~~~~~~~~~~~~~~~

Control the search radius for IDW interpolation:

.. code-block:: python

   # Default: 80 km
   fesomp.regrid(sst, mesh.lon, mesh.lat, influence=80000)

   # Larger radius for sparse data
   fesomp.regrid(sst, mesh.lon, mesh.lat, influence=150000)
