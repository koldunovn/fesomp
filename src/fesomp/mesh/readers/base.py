"""Base class for mesh readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fesomp.mesh.mesh import Mesh


class MeshReader(ABC):
    """Abstract base class for mesh readers."""

    @abstractmethod
    def read(self, path: Path) -> Mesh:
        """
        Read a mesh from the given path.

        Parameters
        ----------
        path : Path
            Path to the mesh file or directory.

        Returns
        -------
        Mesh
            The loaded mesh object.
        """
        ...
