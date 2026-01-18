"""
fesomp: A Python library for working with FESOM2 unstructured ocean model data.
"""

from fesomp import diag
from fesomp.mesh import Mesh, load_mesh
from fesomp.plotting import (
    RegridInterpolator,
    TransectInterpolator,
    great_circle_distance,
    great_circle_path,
    interpolate_transect,
    plot,
    plot_transect,
    regrid,
    transect,
)

__version__ = "0.1.0"
__all__ = [
    # Mesh
    "Mesh",
    "load_mesh",
    # Plotting
    "plot",
    "regrid",
    "RegridInterpolator",
    "transect",
    "interpolate_transect",
    "plot_transect",
    "TransectInterpolator",
    "great_circle_path",
    "great_circle_distance",
    # Diagnostics module
    "diag",
    # Version
    "__version__",
]
