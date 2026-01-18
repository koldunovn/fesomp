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
