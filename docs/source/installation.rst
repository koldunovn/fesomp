Installation
============

From PyPI
---------

The simplest way to install FESOMP is via pip:

.. code-block:: bash

   pip install fesomp

From Source
-----------

To install from source:

.. code-block:: bash

   git clone https://github.com/nkolduno/fesomp.git
   cd fesomp
   pip install -e .

For development with testing tools:

.. code-block:: bash

   pip install -e ".[dev]"

Dependencies
------------

Core dependencies (installed automatically):

- ``numpy >= 1.20``
- ``scipy >= 1.7``
- ``xarray >= 0.19``
- ``pandas >= 1.3``
- ``matplotlib >= 3.5``
- ``cartopy >= 0.20``
- ``netCDF4 >= 1.5``

Optional dependencies:

- ``shapely >= 2.0`` - For geographic operations (install with ``pip install fesomp[geo]``)

Development dependencies:

- ``pytest >= 7.0``
- ``pytest-cov``
- ``hypothesis``

Requirements
------------

- Python 3.10 or higher
- Cartopy may require additional system libraries for map projections. See the `Cartopy installation guide <https://scitools.org.uk/cartopy/docs/latest/installing.html>`_ for details.
