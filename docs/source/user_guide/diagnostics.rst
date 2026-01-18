Diagnostics
===========

FESOMP provides oceanographic diagnostic calculations in the ``fesomp.diag`` module.

Interpolation
-------------

FESOMP provides functions to interpolate data between element centers and nodes
on the unstructured mesh. This is useful when you have data computed at element
centers (e.g., from velocity gradients) and need it at node locations.

Element to Nodes (2D)
~~~~~~~~~~~~~~~~~~~~~

Interpolate 2D data from element centers to nodes using area-weighted averaging:

.. code-block:: python

   import fesomp

   mesh = fesomp.load_mesh("mesh.nc")

   # Data at element centers (e.g., vorticity computed from velocity)
   data_elem = compute_vorticity(u, v)  # shape: (nelem,) or (time, nelem)

   # Interpolate to nodes
   data_nodes = fesomp.diag.elem_to_nodes(
       data_elem,
       mesh.triangles,
       mesh.geometry.elem_area,
       mesh.geometry.node_area[0],
   )

Element to Nodes (3D)
~~~~~~~~~~~~~~~~~~~~~

For 3D data with vertical levels, use the level-aware interpolation that properly
handles varying depth (inactive nodes/elements at deeper levels):

.. code-block:: python

   # 3D data at element centers, shape: (nelem, nlev) or (time, nelem, nlev)
   data_3d = ...

   # Interpolate with level-aware node areas
   data_nodes_3d = fesomp.diag.elem_to_nodes_3d(
       data_3d,
       mesh.triangles,
       mesh.geometry.elem_area,
       mesh.geometry.node_area,    # Full 2D array: (nlev, n2d)
       mesh.elem_levels,           # Element activity levels
   )
   # Result shape: (n2d, nlev) or (time, n2d, nlev)

The ``node_area`` array from the model output is the ground truth that accounts
for inactive triangles at each level. Inactive nodes at each level receive NaN values.

Compute Node Lump
~~~~~~~~~~~~~~~~~

If you need to compute the surface node area (lumped mass) from element connectivity:

.. code-block:: python

   lump = fesomp.diag.compute_node_lump(
       mesh.triangles,
       mesh.geometry.elem_area,
       mesh.n2d,
   )
   # Equivalent to mesh.geometry.node_area[0]

Sea Ice Diagnostics
-------------------

Ice Area
~~~~~~~~

Compute sea ice area (concentration times cell area):

.. code-block:: python

   import fesomp
   import xarray as xr

   mesh = fesomp.load_mesh("mesh.nc")
   sic = xr.open_dataset("a_ice.nc")["a_ice"]

   # Total ice area in a hemisphere
   ice_area = fesomp.diag.ice_area(
       sic,
       mesh.geometry.node_area[0],
       mesh.lat,
       hemisphere="N",  # "N" or "S"
   )

Ice Volume
~~~~~~~~~~

Compute sea ice volume:

.. code-block:: python

   sit = xr.open_dataset("m_ice.nc")["m_ice"]  # Ice thickness

   ice_vol = fesomp.diag.ice_volume(
       sit,
       mesh.geometry.node_area[0],
       mesh.lat,
       hemisphere="N",
   )

Ice Extent
~~~~~~~~~~

Compute sea ice extent (area where concentration exceeds threshold):

.. code-block:: python

   extent = fesomp.diag.ice_extent(
       sic,
       mesh.geometry.node_area[0],
       mesh.lat,
       hemisphere="N",
       threshold=0.15,  # 15% concentration threshold
   )

Vertical Diagnostics
--------------------

Hovmoller Diagram
~~~~~~~~~~~~~~~~~

Compute area-weighted mean vertical profile over time:

.. code-block:: python

   temp = xr.open_dataset("temp.nc")["temp"]

   hovmoller = fesomp.diag.hovmoller(
       temp,
       mesh.geometry.node_area[0],
       mesh.depth_layers,
   )

Volume-Weighted Mean
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mean_temp = fesomp.diag.volume_mean(
       temp,
       mesh.geometry.node_area[0],
       mesh.depth_levels,
       depth_range=(0, 700),  # Top 700 meters
   )

Ocean Heat Content
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ohc = fesomp.diag.heat_content(
       temp,
       mesh.geometry.node_area[0],
       mesh.depth_levels,
       depth_range=(0, 700),
   )

Total Volume
~~~~~~~~~~~~

.. code-block:: python

   volume = fesomp.diag.total_volume(
       mesh.geometry.node_area[0],
       mesh.depth_levels,
       depth_range=(0, 2000),
   )

Mixed Layer
-----------

Mixed Layer Depth
~~~~~~~~~~~~~~~~~

Calculate mixed layer depth using a density or temperature threshold:

.. code-block:: python

   mld = fesomp.diag.mixed_layer_depth(
       density,  # or temperature
       mesh.depth_layers,
       threshold=0.03,  # kg/m³ for density
       reference_depth=10,  # Reference depth in meters
   )

With Interpolation
~~~~~~~~~~~~~~~~~~

For more accurate MLD using linear interpolation:

.. code-block:: python

   mld = fesomp.diag.mixed_layer_depth_interpolated(
       density,
       mesh.depth_layers,
       threshold=0.03,
       reference_depth=10,
   )

Meridional Overturning Circulation
----------------------------------

Global MOC
~~~~~~~~~~

.. code-block:: python

   w = xr.open_dataset("w.nc")["w"]

   moc, latitudes = fesomp.diag.moc(
       w,
       mesh.geometry.elem_area,
       mesh.lat_elem,
       mesh.depth_levels,
   )

Atlantic MOC
~~~~~~~~~~~~

Use a basin mask for regional MOC:

.. code-block:: python

   # Get Atlantic basin mask
   atl_mask = fesomp.diag.get_basin_mask(
       mesh.lon_elem, mesh.lat_elem, "Atlantic_MOC"
   )

   # Compute Atlantic MOC
   amoc, lats = fesomp.diag.moc(
       w,
       mesh.geometry.elem_area,
       mesh.lat_elem,
       mesh.depth_levels,
       mask=atl_mask,
   )

AMOC Index
~~~~~~~~~~

Extract the AMOC index at a specific latitude:

.. code-block:: python

   amoc_26n = fesomp.diag.amoc_index(
       amoc,  # MOC streamfunction
       lats,  # Latitude array
       latitude=26.5,  # RAPID array latitude
   )

Available Basins
~~~~~~~~~~~~~~~~

.. code-block:: python

   # List available basin names
   basins = fesomp.diag.list_moc_basins()
   print(basins)
   # ['Global', 'Atlantic_MOC', 'Indo-Pacific', ...]

Utility Functions
-----------------

Hemisphere Mask
~~~~~~~~~~~~~~~

.. code-block:: python

   # Create mask for Northern Hemisphere
   nh_mask = fesomp.diag.hemisphere_mask(mesh.lat, "N")

   # Southern Hemisphere
   sh_mask = fesomp.diag.hemisphere_mask(mesh.lat, "S")

Region Mask
~~~~~~~~~~~

.. code-block:: python

   # Get mask for a named region
   mask = fesomp.diag.get_region_mask(
       mesh.lon, mesh.lat, "North_Atlantic"
   )

   # List available regions
   regions = fesomp.diag.list_available_regions()

Layer Thickness
~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute layer thickness from level depths
   thickness = fesomp.diag.compute_layer_thickness(mesh.depth_levels)

Depth Selection
~~~~~~~~~~~~~~~

.. code-block:: python

   # Get indices for a depth range
   indices = fesomp.diag.select_depth_indices(
       mesh.depth_levels,
       depth_range=(100, 500),
   )

Surface Area
~~~~~~~~~~~~

.. code-block:: python

   # Get total surface area
   area = fesomp.diag.get_surface_area(
       mesh.geometry.node_area[0],
       mask=nh_mask,  # Optional regional mask
   )
