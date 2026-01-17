"""ASCII reader for FESOM2 mesh files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fesomp.mesh.mesh import Mesh
from fesomp.mesh.readers.base import MeshReader


class ASCIIReader(MeshReader):
    """
    Reader for FESOM2 ASCII mesh files.

    Reads from a directory containing:
    - nod2d.out: Node coordinates
    - elem2d.out: Triangle connectivity
    - aux3d.out: Vertical structure and bottom depths
    """

    def read(self, path: Path) -> Mesh:
        """
        Read a mesh from ASCII files in a directory.

        Parameters
        ----------
        path : Path
            Path to directory containing mesh files.

        Returns
        -------
        Mesh
            The loaded mesh object.
        """
        path = Path(path)

        # Read node coordinates
        lon, lat = self._read_nod2d(path / "nod2d.out")

        # Read triangle connectivity
        triangles = self._read_elem2d(path / "elem2d.out")

        # Read vertical structure
        nlev, depth_levels, node_bottom_depth = self._read_aux3d(
            path / "aux3d.out", len(lon)
        )

        # Compute derived quantities
        depth_layers = self._compute_depth_layers(depth_levels)
        node_levels = self._compute_node_levels(depth_levels, node_bottom_depth)
        elem_levels = self._compute_elem_levels(triangles, node_levels)
        elem_bottom_depth = self._compute_elem_bottom_depth(triangles, node_bottom_depth)

        return Mesh(
            lon=lon,
            lat=lat,
            triangles=triangles,
            nlev=nlev,
            depth_levels=depth_levels,
            depth_layers=depth_layers,
            node_levels=node_levels,
            elem_levels=elem_levels,
            node_bottom_depth=node_bottom_depth,
            elem_bottom_depth=elem_bottom_depth,
        )

    def _read_nod2d(self, filepath: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Read node coordinates from nod2d.out.

        Format: `index lon lat flag` per line
        First line may contain the number of nodes.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Longitude and latitude arrays.
        """
        with open(filepath) as f:
            first_line = f.readline().strip()

        # Check if first line is just a count
        parts = first_line.split()
        if len(parts) == 1:
            # First line is node count, skip it
            skiprows = 1
        else:
            skiprows = 0

        # Read data
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            skiprows=skiprows,
            names=["index", "lon", "lat", "flag"],
            usecols=["lon", "lat"],
        )

        lon = df["lon"].values.astype(np.float64)
        lat = df["lat"].values.astype(np.float64)

        # Normalize longitude from [0, 360] to [-180, 180]
        lon = np.where(lon > 180, lon - 360, lon)

        return lon, lat

    def _read_elem2d(self, filepath: Path) -> np.ndarray:
        """
        Read triangle connectivity from elem2d.out.

        Format: `i1 i2 i3` per line (1-indexed)
        First line may contain the number of elements.

        Returns
        -------
        np.ndarray
            Triangle connectivity, shape (nelem, 3), 0-indexed.
        """
        with open(filepath) as f:
            first_line = f.readline().strip()

        # Check if first line is just a count
        parts = first_line.split()
        if len(parts) == 1:
            skiprows = 1
        else:
            skiprows = 0

        # Read data
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            skiprows=skiprows,
            names=["i1", "i2", "i3"],
        )

        # Convert to 0-indexed
        triangles = df.values.astype(np.int32) - 1

        return triangles

    def _read_aux3d(
        self, filepath: Path, n2d: int
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Read vertical structure from aux3d.out.

        Format:
        - Line 1: nlev (number of vertical levels)
        - Lines 2 to nlev+1: depth of each level
        - Lines nlev+2 to end: bottom depth at each node

        Returns
        -------
        tuple[int, np.ndarray, np.ndarray]
            Number of levels, depth levels, and node bottom depths.
        """
        with open(filepath) as f:
            lines = f.readlines()

        # First line: number of levels
        nlev = int(lines[0].strip())

        # Next nlev lines: depth levels (ASCII uses negative, convert to positive)
        depth_levels = np.array(
            [float(lines[i + 1].strip()) for i in range(nlev)], dtype=np.float64
        )
        # Convert to positive depths (matching NetCDF convention)
        depth_levels = np.abs(depth_levels)

        # Remaining lines: node bottom depths (ASCII uses negative, convert to positive)
        node_bottom_depth = np.array(
            [float(lines[i + nlev + 1].strip()) for i in range(n2d)], dtype=np.float64
        )
        # Convert to positive depths (matching NetCDF convention)
        node_bottom_depth = np.abs(node_bottom_depth)

        return nlev, depth_levels, node_bottom_depth

    def _compute_depth_layers(self, depth_levels: np.ndarray) -> np.ndarray:
        """Compute layer center depths from level depths."""
        # Layer center is midpoint between adjacent levels
        return (depth_levels[:-1] + depth_levels[1:]) / 2

    def _compute_node_levels(
        self, depth_levels: np.ndarray, node_bottom_depth: np.ndarray
    ) -> np.ndarray:
        """
        Compute number of active levels at each node.

        A level is active if its depth is <= the node's bottom depth.
        With positive depth convention: depth_levels are 0, 5, 10, ...
        and node_bottom_depth is e.g. 672 (meters deep).
        """
        # For each node, count how many depth_levels are <= node_bottom_depth
        # Using broadcasting: depth_levels[:, None] <= node_bottom_depth[None, :]
        # Result shape: (nlev, n2d), sum along axis 0 gives counts
        node_levels = np.sum(
            depth_levels[:, None] <= node_bottom_depth[None, :], axis=0
        ).astype(np.int32)

        return node_levels

    def _compute_elem_levels(
        self, triangles: np.ndarray, node_levels: np.ndarray
    ) -> np.ndarray:
        """
        Compute number of active levels at each element.

        Element levels = minimum of its three node levels.
        """
        tri_node_levels = node_levels[triangles]
        return np.min(tri_node_levels, axis=1).astype(np.int32)

    def _compute_elem_bottom_depth(
        self, triangles: np.ndarray, node_bottom_depth: np.ndarray
    ) -> np.ndarray:
        """
        Compute bottom depth at each element.

        Element bottom = minimum (shallowest) of its three node bottom depths.
        With positive depth convention, the element's water column extends
        only as deep as the shallowest of its vertices.
        """
        tri_bottom_depths = node_bottom_depth[triangles]
        return np.min(tri_bottom_depths, axis=1)
