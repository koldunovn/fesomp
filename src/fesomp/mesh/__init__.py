"""
FESOM2 mesh module.

This module provides classes and functions for loading and working with
FESOM2 unstructured mesh data.
"""

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
]
