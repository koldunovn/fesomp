FESOMP Documentation
====================

**FESOMP** is a modern Python library for working with FESOM2 (Finite Element Sea ice-Ocean Model) unstructured mesh data.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/mesh
   user_guide/plotting
   user_guide/transects
   user_guide/diagnostics

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/fesomp
   api/mesh
   api/plotting
   api/diag

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog


Features
--------

Mesh Management
~~~~~~~~~~~~~~~
- **Flexible I/O**: Load meshes from NetCDF or ASCII formats
- **Lazy Loading**: Topology, geometry, and spatial indices computed on-demand
- **Dual Coordinates**: Automatic handling of node and element (triangle center) data
- **Spatial Queries**: Fast nearest-neighbor search, radius queries, and bounding box selection

Data Interpolation
~~~~~~~~~~~~~~~~~~
- **Three Methods**: Nearest neighbor (fast), inverse distance weighting (smooth), and linear interpolation
- **Regular Grid**: Interpolate unstructured data to regular lon/lat grids
- **Caching**: Reusable interpolators for efficient multi-variable processing
- **Spherical Geometry**: Accurate great circle calculations on the sphere

Visualization
~~~~~~~~~~~~~
- **Map Plots**: Cartopy integration with multiple projections (PlateCarree, Robinson, Polar, etc.)
- **Transect Plots**: Vertical cross-sections with automatic vertical coordinate detection
- **Customizable**: Full control over colormaps, contours, labels, and styling
- **Auto-Detection**: Automatically determines if data is on nodes/elements and levels/layers

Diagnostics
~~~~~~~~~~~
- **Sea Ice**: Ice area, volume, and extent calculations
- **Vertical**: Hovmoller diagrams, heat content, volume-weighted means
- **Mixed Layer**: Mixed layer depth with various criteria
- **Meridional Overturning**: MOC and AMOC index calculations


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
