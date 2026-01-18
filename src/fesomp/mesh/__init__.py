"""
FESOM2 mesh module.

This module provides classes and functions for loading and working with
FESOM2 unstructured mesh data.
"""

from fesomp.mesh.coordinates import (
    scalar_g2r,
    scalar_r2g,
    vec_rotate_g2r,
    vec_rotate_r2g,
)
from fesomp.mesh.geometry import Geometry
from fesomp.mesh.mesh import Mesh, load_mesh
from fesomp.mesh.spatial import SpatialIndex
from fesomp.mesh.topology import Topology

__all__ = [
    "Mesh",
    "load_mesh",
    "Topology",
    "Geometry",
    "SpatialIndex",
    # Coordinate transformations
    "scalar_r2g",
    "scalar_g2r",
    "vec_rotate_r2g",
    "vec_rotate_g2r",
]
