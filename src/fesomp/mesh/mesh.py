"""Core Mesh class for FESOM2 unstructured mesh data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fesomp.mesh.geometry import Geometry
    from fesomp.mesh.spatial import SpatialIndex
    from fesomp.mesh.topology import Topology


@dataclass
class Mesh:
    """
    Represents a FESOM2 unstructured mesh.

    This class holds the core mesh data including node coordinates, element
    connectivity, and vertical structure. Topology, geometry, and spatial
    indexing are lazily computed on first access.

    Attributes
    ----------
    lon : np.ndarray
        Longitude of nodes in degrees, shape (n2d,).
    lat : np.ndarray
        Latitude of nodes in degrees, shape (n2d,).
    triangles : np.ndarray
        Triangle connectivity, shape (nelem, 3), 0-indexed.
    nlev : int
        Number of vertical levels.
    depth_levels : np.ndarray
        Depth at each level interface, shape (nlev,).
    depth_layers : np.ndarray
        Depth at layer centers, shape (nlev-1,).
    node_levels : np.ndarray
        Number of active levels at each node, shape (n2d,).
    elem_levels : np.ndarray
        Number of active levels at each element, shape (nelem,).
    node_bottom_depth : np.ndarray
        Bottom depth at each node, shape (n2d,).
    elem_bottom_depth : np.ndarray
        Bottom depth at each element, shape (nelem,).
    """

    # Core 2D (required)
    lon: np.ndarray
    lat: np.ndarray
    triangles: np.ndarray

    # Vertical structure
    nlev: int
    depth_levels: np.ndarray
    depth_layers: np.ndarray
    node_levels: np.ndarray
    elem_levels: np.ndarray

    # Bottom depths
    node_bottom_depth: np.ndarray
    elem_bottom_depth: np.ndarray

    # Pre-loaded components (from NetCDF)
    _preloaded_topology: Topology | None = field(default=None, repr=False)
    _preloaded_geometry: Geometry | None = field(default=None, repr=False)

    # Lazy-loaded components (computed on demand)
    _topology: Topology | None = field(default=None, repr=False)
    _geometry: Geometry | None = field(default=None, repr=False)
    _spatial_index: SpatialIndex | None = field(default=None, repr=False)
    _lon_elem: np.ndarray | None = field(default=None, repr=False)
    _lat_elem: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate mesh data and ensure correct dtypes."""
        self.lon = np.asarray(self.lon, dtype=np.float64)
        self.lat = np.asarray(self.lat, dtype=np.float64)
        self.triangles = np.asarray(self.triangles, dtype=np.int32)
        self.depth_levels = np.asarray(self.depth_levels, dtype=np.float64)
        self.depth_layers = np.asarray(self.depth_layers, dtype=np.float64)
        self.node_levels = np.asarray(self.node_levels, dtype=np.int32)
        self.elem_levels = np.asarray(self.elem_levels, dtype=np.int32)
        self.node_bottom_depth = np.asarray(self.node_bottom_depth, dtype=np.float64)
        self.elem_bottom_depth = np.asarray(self.elem_bottom_depth, dtype=np.float64)

        self._validate()

    def _validate(self) -> None:
        """Validate mesh data consistency."""
        if self.lon.ndim != 1 or self.lat.ndim != 1:
            raise ValueError("lon and lat must be 1D arrays")
        if len(self.lon) != len(self.lat):
            raise ValueError("lon and lat must have the same length")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (nelem, 3)")
        if self.triangles.min() < 0:
            raise ValueError("triangles must be 0-indexed with non-negative values")
        if self.triangles.max() >= self.n2d:
            raise ValueError("triangle indices exceed number of nodes")
        if len(self.node_levels) != self.n2d:
            raise ValueError("node_levels must have length n2d")
        if len(self.elem_levels) != self.nelem:
            raise ValueError("elem_levels must have length nelem")
        if len(self.node_bottom_depth) != self.n2d:
            raise ValueError("node_bottom_depth must have length n2d")
        if len(self.elem_bottom_depth) != self.nelem:
            raise ValueError("elem_bottom_depth must have length nelem")

    @property
    def n2d(self) -> int:
        """Number of 2D nodes."""
        return len(self.lon)

    @property
    def nelem(self) -> int:
        """Number of triangular elements."""
        return len(self.triangles)

    @property
    def topology(self) -> Topology:
        """
        Get mesh topology (edges, neighbors, etc.).

        Computed lazily on first access if not pre-loaded from NetCDF.
        """
        if self._preloaded_topology is not None:
            return self._preloaded_topology
        if self._topology is None:
            from fesomp.mesh.topology import compute_topology

            self._topology = compute_topology(self.triangles)
        return self._topology

    @property
    def geometry(self) -> Geometry:
        """
        Get mesh geometry (areas, gradients, etc.).

        Computed lazily on first access if not pre-loaded from NetCDF.
        """
        if self._preloaded_geometry is not None:
            return self._preloaded_geometry
        if self._geometry is None:
            from fesomp.mesh.geometry import compute_geometry

            self._geometry = compute_geometry(
                self.lon,
                self.lat,
                self.triangles,
                self.node_levels,
                self.nlev,
                self.topology,
            )
        return self._geometry

    @property
    def spatial_index(self) -> SpatialIndex:
        """
        Get spatial index for efficient point queries.

        Built lazily on first access.
        """
        if self._spatial_index is None:
            from fesomp.mesh.spatial import SpatialIndex

            self._spatial_index = SpatialIndex(self.lon, self.lat)
        return self._spatial_index

    @property
    def lon_elem(self) -> np.ndarray:
        """
        Longitude of element (triangle) centers in degrees.

        Computed lazily on first access. Handles cyclic triangles that
        cross the dateline correctly.
        """
        if self._lon_elem is None:
            self._compute_elem_coords()
        return self._lon_elem

    @property
    def lat_elem(self) -> np.ndarray:
        """
        Latitude of element (triangle) centers in degrees.

        Computed lazily on first access.
        """
        if self._lat_elem is None:
            self._compute_elem_coords()
        return self._lat_elem

    def _compute_elem_coords(self) -> None:
        """Compute element center coordinates, handling cyclic triangles."""
        # Get vertex coordinates for each triangle
        tri_lon = self.lon[self.triangles]  # (nelem, 3)
        tri_lat = self.lat[self.triangles]  # (nelem, 3)

        # Simple mean for latitude (no cyclic issues)
        self._lat_elem = tri_lat.mean(axis=1)

        # For longitude, need to handle triangles crossing the dateline
        # First compute simple mean
        lon_mean = tri_lon.mean(axis=1)

        # Find cyclic triangles: where any vertex is far from the mean (>100°)
        max_diff = np.abs(tri_lon - lon_mean[:, np.newaxis]).max(axis=1)
        cyclic_mask = max_diff > 100

        if np.any(cyclic_mask):
            # For cyclic triangles, shift negative longitudes by +360 before averaging
            cyclic_lon = tri_lon[cyclic_mask].copy()
            cyclic_lon_shifted = np.where(cyclic_lon < 0, cyclic_lon + 360, cyclic_lon)
            new_means = cyclic_lon_shifted.mean(axis=1)
            # Shift back to [-180, 180] range
            new_means = np.where(new_means > 180, new_means - 360, new_means)
            lon_mean[cyclic_mask] = new_means

        self._lon_elem = lon_mean

    def find_nearest(self, lon: float, lat: float, k: int = 1) -> np.ndarray:
        """
        Find the k nearest nodes to a given point.

        Parameters
        ----------
        lon : float
            Longitude in degrees.
        lat : float
            Latitude in degrees.
        k : int, optional
            Number of nearest neighbors to return.

        Returns
        -------
        np.ndarray
            Indices of the k nearest nodes.
        """
        return self.spatial_index.find_nearest(lon, lat, k=k)

    def find_in_radius(self, lon: float, lat: float, radius_km: float) -> np.ndarray:
        """
        Find all nodes within a given radius of a point.

        Parameters
        ----------
        lon : float
            Longitude in degrees.
        lat : float
            Latitude in degrees.
        radius_km : float
            Search radius in kilometers.

        Returns
        -------
        np.ndarray
            Indices of nodes within the radius.
        """
        return self.spatial_index.find_in_radius(lon, lat, radius_km)

    def subset_by_bbox(
        self, lon_min: float, lon_max: float, lat_min: float, lat_max: float
    ) -> np.ndarray:
        """
        Find all nodes within a bounding box.

        Parameters
        ----------
        lon_min, lon_max : float
            Longitude bounds in degrees.
        lat_min, lat_max : float
            Latitude bounds in degrees.

        Returns
        -------
        np.ndarray
            Indices of nodes within the bounding box.
        """
        mask = (
            (self.lon >= lon_min)
            & (self.lon <= lon_max)
            & (self.lat >= lat_min)
            & (self.lat <= lat_max)
        )
        return np.nonzero(mask)[0]

    def get_triangulation(self, mask_cyclic: bool = True):
        """
        Create a matplotlib Triangulation object for plotting.

        Parameters
        ----------
        mask_cyclic : bool, optional
            If True (default), mask triangles that cross the dateline
            to prevent ugly lines spanning the globe.

        Returns
        -------
        matplotlib.tri.Triangulation
            Triangulation object ready for use with triplot, tripcolor, etc.

        Example
        -------
        >>> tri = mesh.get_triangulation()
        >>> plt.triplot(tri, 'b-', linewidth=0.2)
        """
        from matplotlib.tri import Triangulation

        tri = Triangulation(self.lon, self.lat, self.triangles)

        if mask_cyclic:
            # Mask triangles that cross the dateline (lon range > 180°)
            tri_lon = self.lon[self.triangles]
            lon_range = tri_lon.max(axis=1) - tri_lon.min(axis=1)
            cyclic_mask = lon_range > 180
            tri.set_mask(cyclic_mask)

        return tri

    def __repr__(self) -> str:
        return (
            f"Mesh(n2d={self.n2d}, nelem={self.nelem}, nlev={self.nlev}, "
            f"lon=[{self.lon.min():.2f}, {self.lon.max():.2f}], "
            f"lat=[{self.lat.min():.2f}, {self.lat.max():.2f}])"
        )


def load_mesh(path: str | Path) -> Mesh:
    """
    Load a FESOM2 mesh from a file or directory.

    Automatically detects the format based on the path:
    - If path is a `.nc` file: loads from NetCDF
    - If path is a directory: loads from ASCII files

    Parameters
    ----------
    path : str or Path
        Path to the mesh file (NetCDF) or directory (ASCII).

    Returns
    -------
    Mesh
        The loaded mesh object.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file format cannot be determined.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file() and path.suffix == ".nc":
        from fesomp.mesh.readers.netcdf import NetCDFReader

        reader = NetCDFReader()
        return reader.read(path)
    elif path.is_dir():
        from fesomp.mesh.readers.ascii import ASCIIReader

        reader = ASCIIReader()
        return reader.read(path)
    else:
        raise ValueError(
            f"Cannot determine mesh format for: {path}. "
            "Expected a .nc file or a directory containing ASCII mesh files."
        )
