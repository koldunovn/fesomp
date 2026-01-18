Working with Meshes
===================

FESOMP provides a comprehensive ``Mesh`` class for working with FESOM2 unstructured meshes.

Loading Meshes
--------------

From NetCDF
~~~~~~~~~~~

The recommended format is NetCDF, typically the ``fesom.mesh.diag.nc`` file:

.. code-block:: python

   import fesomp

   mesh = fesomp.load_mesh("path/to/fesom.mesh.diag.nc")

From ASCII
~~~~~~~~~~

You can also load meshes from ASCII files (the traditional FESOM format):

.. code-block:: python

   # Point to the directory containing nod2d.out, elem2d.out, etc.
   mesh = fesomp.load_mesh("path/to/mesh/directory/")

The loader will automatically detect the format based on whether you provide a file or directory.

Mesh Properties
---------------

Basic Properties
~~~~~~~~~~~~~~~~

.. code-block:: python

   mesh.n2d       # Number of 2D nodes
   mesh.nelem     # Number of triangular elements
   mesh.nlev      # Number of vertical levels

   mesh.lon       # Node longitudes (n2d,)
   mesh.lat       # Node latitudes (n2d,)

   mesh.depth_levels  # Depth at level interfaces (nlev,)
   mesh.depth_layers  # Depth at layer centers (nlev-1,)

Element Coordinates (Lazy)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Element center coordinates are computed on first access:

.. code-block:: python

   mesh.lon_elem  # Element center longitudes (nelem,)
   mesh.lat_elem  # Element center latitudes (nelem,)

Triangulation
~~~~~~~~~~~~~

.. code-block:: python

   mesh.elem      # Triangle connectivity (nelem, 3)

Topology (Lazy)
---------------

Mesh topology is computed on first access for efficiency:

.. code-block:: python

   topology = mesh.topology

   topology.edges           # Unique edges (nedges, 2)
   topology.edge_indices    # Edge index for each triangle edge
   topology.face_neighbors  # Adjacent triangles (-1 for boundary)
   topology.boundary_edges  # Indices of boundary edges
   topology.n_edges         # Total number of edges

Geometry (Lazy)
---------------

Geometric properties are computed on demand:

.. code-block:: python

   geometry = mesh.geometry

   geometry.elem_area   # Triangle areas in m² (nelem,)
   geometry.node_area   # Control volumes around nodes (3, n2d)
                        # Three computation methods
   geometry.gradients   # Gradient operators for elements

Spatial Queries
---------------

FESOMP provides efficient spatial indexing for queries.

Find Nearest Nodes
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find the 5 nearest nodes to a point
   indices = mesh.find_nearest(lon=10.5, lat=54.3, k=5)

   # Find nearest element centers
   indices = mesh.find_nearest(lon=10.5, lat=54.3, k=5, on_elements=True)

Radius Search
~~~~~~~~~~~~~

.. code-block:: python

   # Find all nodes within 100 km of a point
   indices = mesh.find_in_radius(lon=0, lat=0, radius_km=100)

Bounding Box
~~~~~~~~~~~~

.. code-block:: python

   # Find nodes in a region
   indices = mesh.subset_by_bbox(
       lon_min=-10, lon_max=10,
       lat_min=40, lat_max=60
   )

Automatic Data Location Detection
---------------------------------

FESOMP automatically detects whether data is on nodes or elements:

.. code-block:: python

   # Data on nodes (n2d points)
   temp_nodes = np.random.rand(mesh.n2d)
   # Functions will use mesh.lon, mesh.lat

   # Data on elements (nelem points)
   velocity = np.random.rand(mesh.nelem)
   # Functions will use mesh.lon_elem, mesh.lat_elem

   # 3D data on levels (nlev, n2d)
   w_velocity = np.random.rand(mesh.nlev, mesh.n2d)
   # Uses mesh.depth_levels

   # 3D data on layers (nlev-1, n2d)
   temperature = np.random.rand(mesh.nlev - 1, mesh.n2d)
   # Uses mesh.depth_layers

This detection happens automatically in functions like ``transect()`` and ``plot()``.

Coordinate Transformations
--------------------------

FESOM2 meshes often use rotated coordinate systems to avoid singularities at the poles.
FESOMP provides functions to convert between rotated and geographical coordinates.

Scalar Coordinate Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert coordinates between rotated and geographical systems:

.. code-block:: python

   from fesomp.mesh import scalar_r2g, scalar_g2r
   import numpy as np

   # Euler angles defining the rotation (typical FESOM values)
   alpha, beta, gamma = 50, 15, -90

   # Convert from rotated to geographical coordinates
   rlon = np.array([0.0, 10.0, 20.0])
   rlat = np.array([45.0, -30.0, 60.0])
   lon, lat = scalar_r2g(alpha, beta, gamma, rlon, rlat)

   # Convert from geographical to rotated coordinates
   rlon_back, rlat_back = scalar_g2r(alpha, beta, gamma, lon, lat)

Vector Rotation
~~~~~~~~~~~~~~~

Rotate velocity or other vector fields between coordinate systems:

.. code-block:: python

   from fesomp.mesh import vec_rotate_r2g, vec_rotate_g2r

   # Euler angles
   alpha, beta, gamma = 50, 15, -90

   # Coordinates where vectors are defined
   lon = np.array([10.0, 20.0, -30.0])
   lat = np.array([45.0, -30.0, 60.0])

   # Vector components in rotated coordinates
   u_rot = np.array([1.0, 2.0, -1.0])  # eastward component
   v_rot = np.array([0.5, -0.5, 1.5])  # northward component

   # Rotate vectors to geographical coordinates
   # flag=1 means lon/lat are in geographical coordinates
   # flag=0 means lon/lat are in rotated coordinates
   u_geo, v_geo = vec_rotate_r2g(alpha, beta, gamma, lon, lat, u_rot, v_rot, flag=1)

   # Rotate vectors from geographical to rotated coordinates
   u_rot_back, v_rot_back = vec_rotate_g2r(alpha, beta, gamma, lon, lat, u_geo, v_geo, flag=1)

The ``flag`` parameter specifies the coordinate system of the input ``lon``/``lat``:

- ``flag=1``: lon/lat are in geographical coordinates
- ``flag=0``: lon/lat are in rotated coordinates

The functions will automatically compute the coordinates in the other system as needed.
