Vertical Transects
==================

FESOMP provides tools for creating vertical cross-sections through 3D ocean data.

Basic Transect
--------------

.. code-block:: python

   import fesomp
   import xarray as xr

   mesh = fesomp.load_mesh("mesh.nc")
   temp = xr.open_dataset("temp.nc")['temp'][0, :, :].values  # (nlev, n2d)

   fig, ax, interp = fesomp.transect(
       temp,
       mesh,
       start=(-30, -60),  # (lon, lat) - 30°W, 60°S
       end=(-30, 60),     # (lon, lat) - 30°W, 60°N
       title="Temperature along 30°W",
       units="°C",
   )

Customizing Transects
---------------------

Depth Limits
~~~~~~~~~~~~

.. code-block:: python

   # Show only top 2000 meters
   fesomp.transect(
       temp, mesh,
       start=(-30, -60), end=(-30, 60),
       depth_limits=(0, 2000),
   )

   # Full depth
   fesomp.transect(
       temp, mesh,
       start=(-30, -60), end=(-30, 60),
       depth_limits=(0, 6000),
   )

Number of Points
~~~~~~~~~~~~~~~~

Control the resolution of the transect:

.. code-block:: python

   fesomp.transect(
       temp, mesh,
       start=(-30, -60), end=(-30, 60),
       npoints=200,  # Number of horizontal points
   )

Interpolation Method
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Nearest neighbor (fast)
   fesomp.transect(temp, mesh, ..., method="nn")

   # Inverse distance weighting (smooth, default)
   fesomp.transect(temp, mesh, ..., method="idw")

   # Linear interpolation (most accurate)
   fesomp.transect(temp, mesh, ..., method="linear")

Appearance
~~~~~~~~~~

.. code-block:: python

   fesomp.transect(
       temp, mesh,
       start=(-30, -60), end=(-30, 60),
       cmap="RdYlBu_r",
       vmin=-2, vmax=25,
       levels=20,
       title="Atlantic Temperature Section",
       units="°C",
   )

Automatic Detection
-------------------

FESOMP automatically detects:

Data Location
~~~~~~~~~~~~~

.. code-block:: python

   # Data on nodes (n2d)
   temp_nodes = np.random.rand(nlev, n2d)
   fesomp.transect(temp_nodes, mesh, ...)  # Uses mesh.lon, mesh.lat

   # Data on elements (nelem)
   u_velocity = np.random.rand(nlev, nelem)
   fesomp.transect(u_velocity, mesh, ...)  # Uses mesh.lon_elem, mesh.lat_elem

Vertical Coordinate
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Data on levels (nlev points)
   w_velocity = np.random.rand(nlev, n2d)
   fesomp.transect(w_velocity, mesh, ...)  # Uses mesh.depth_levels

   # Data on layers (nlev-1 points)
   temperature = np.random.rand(nlev - 1, n2d)
   fesomp.transect(temperature, mesh, ...)  # Uses mesh.depth_layers

Advanced Usage
--------------

Two-Step Process
~~~~~~~~~~~~~~~~

For more control, separate interpolation from plotting:

.. code-block:: python

   # Step 1: Interpolate along transect path
   data_interp, distance, depth, path_coords = fesomp.interpolate_transect(
       temp,
       mesh.lon,
       mesh.lat,
       mesh.depth_layers,
       start=(-30, -60),
       end=(-30, 60),
       npoints=100,
       method="idw",
   )

   # Step 2: Plot the interpolated data
   fig, ax = fesomp.plot_transect(
       data_interp,
       distance,
       depth,
       title="Temperature",
       units="°C",
       depth_limits=(0, 2000),
   )

Reusable Interpolator
~~~~~~~~~~~~~~~~~~~~~

For multiple variables along the same transect:

.. code-block:: python

   # Create interpolator
   interp = fesomp.TransectInterpolator(
       mesh.lon, mesh.lat,
       start=(-30, -60),
       end=(-30, 60),
       npoints=100,
       method="idw",
   )

   # Interpolate multiple variables
   temp_sec = interp(temperature, mesh.depth_layers)
   salt_sec = interp(salinity, mesh.depth_layers)
   oxy_sec = interp(oxygen, mesh.depth_layers)

Great Circle Calculations
-------------------------

FESOMP uses spherical geometry for accurate transects:

.. code-block:: python

   # Generate points along a great circle path
   lons, lats = fesomp.great_circle_path(
       start=(-30, -60),
       end=(-30, 60),
       npoints=100,
   )

   # Calculate great circle distance between two points
   distance_km = fesomp.great_circle_distance(
       lon1=-30, lat1=-60,
       lon2=-30, lat2=60,
   )
