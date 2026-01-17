"""NetCDF reader for FESOM2 mesh files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from fesomp.mesh.geometry import Geometry
from fesomp.mesh.mesh import Mesh
from fesomp.mesh.readers.base import MeshReader
from fesomp.mesh.topology import Topology


class NetCDFReader(MeshReader):
    """
    Reader for FESOM2 mesh diagnostic NetCDF files.

    Reads from fesom.mesh.diag.nc which contains pre-computed topology
    and geometry information.
    """

    def read(self, path: Path) -> Mesh:
        """
        Read a mesh from a NetCDF file.

        Parameters
        ----------
        path : Path
            Path to the NetCDF mesh file (e.g., fesom.mesh.diag.nc).

        Returns
        -------
        Mesh
            The loaded mesh object with pre-populated topology and geometry.
        """
        with xr.open_dataset(path) as ds:
            # Core coordinates
            lon = ds["lon"].values.astype(np.float64)
            lat = ds["lat"].values.astype(np.float64)

            # Triangle connectivity - transpose and convert to 0-indexed
            # NetCDF has shape (n3, nelem), we want (nelem, 3)
            # Also convert from 1-indexed (Fortran) to 0-indexed (Python)
            triangles = ds["face_nodes"].values.T.astype(np.int32) - 1

            # Vertical structure
            nlev = ds.sizes["nz"]
            depth_levels = ds["nz"].values.astype(np.float64)
            depth_layers = ds["nz1"].values.astype(np.float64)
            node_levels = ds["nlevels_nod2D"].values.astype(np.int32)
            elem_levels = ds["nlevels"].values.astype(np.int32)

            # Bottom depths
            node_bottom_depth = ds["zbar_n_bottom"].values.astype(np.float64)
            elem_bottom_depth = ds["zbar_e_bottom"].values.astype(np.float64)

            # Load topology
            topology = self._load_topology(ds)

            # Load geometry
            geometry = self._load_geometry(ds, nlev, len(lon))

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
            _preloaded_topology=topology,
            _preloaded_geometry=geometry,
        )

    def _load_topology(self, ds: xr.Dataset) -> Topology:
        """Load pre-computed topology from NetCDF."""
        # Edge nodes - transpose and convert to 0-indexed
        # NetCDF has shape (n2, nedges), we want (nedges, 2)
        edges = ds["edge_nodes"].values.T.astype(np.int32) - 1

        # Face edges - transpose
        # NetCDF has shape (n3, nelem), we want (nelem, 3)
        face_edges = ds["face_edges"].values.T.astype(np.int32) - 1

        # Face neighbors (face_links) - transpose and handle fill values
        # NetCDF uses NaN or large negative values for boundary, we use -1
        # NetCDF has shape (n3, nelem), we want (nelem, 3)
        face_neighbors_raw = ds["face_links"].values.T
        # Replace NaN and invalid values before converting to int
        face_neighbors_raw = np.nan_to_num(face_neighbors_raw, nan=-999)
        face_neighbors = face_neighbors_raw.astype(np.int32)
        # Convert from 1-indexed to 0-indexed, keeping invalid values as boundary marker
        valid_mask = face_neighbors > 0
        face_neighbors[valid_mask] -= 1
        face_neighbors[~valid_mask] = -1  # Convert invalid values to -1

        # Edge faces - transpose
        # NetCDF has shape (n2, nedges), we want (nedges, 2)
        edge_faces_raw = ds["edge_face_links"].values.T
        edge_faces_raw = np.nan_to_num(edge_faces_raw, nan=-999)
        edge_faces = edge_faces_raw.astype(np.int32)
        valid_mask = edge_faces > 0
        edge_faces[valid_mask] -= 1
        edge_faces[~valid_mask] = -1

        # Node elements mapping
        # nod_in_elem2D has shape (max_elems_per_node, n2d)
        # nod_in_elem2D_num has shape (n2d,) - number of elements per node
        nod_in_elem2d = ds["nod_in_elem2D"].values.T.astype(np.int32) - 1
        nod_in_elem2d_num = ds["nod_in_elem2D_num"].values.astype(np.int32)

        n2d = len(nod_in_elem2d_num)
        node_elements = []
        for i in range(n2d):
            count = nod_in_elem2d_num[i]
            elems = nod_in_elem2d[i, :count]
            node_elements.append(elems.copy())

        return Topology(
            edges=edges,
            face_edges=face_edges,
            face_neighbors=face_neighbors,
            edge_faces=edge_faces,
            node_elements=node_elements,
        )

    def _load_geometry(self, ds: xr.Dataset, nlev: int, n2d: int) -> Geometry:
        """Load pre-computed geometry from NetCDF."""
        elem_area = ds["elem_area"].values.astype(np.float64)

        # Node area has shape (nlev, n2d) in NetCDF
        node_area = ds["nod_area"].values.astype(np.float64)

        # Gradient operators (if available)
        gradient_sca = None
        gradient_vec = None
        edge_cross_dxdy = None

        if "gradient_sca_x" in ds and "gradient_sca_y" in ds:
            gradient_sca = (
                ds["gradient_sca_x"].values.astype(np.float64),
                ds["gradient_sca_y"].values.astype(np.float64),
            )

        if "gradient_vec_x" in ds and "gradient_vec_y" in ds:
            gradient_vec = (
                ds["gradient_vec_x"].values.astype(np.float64),
                ds["gradient_vec_y"].values.astype(np.float64),
            )

        if "edge_cross_dxdy" in ds:
            edge_cross_dxdy = ds["edge_cross_dxdy"].values.astype(np.float64)

        return Geometry(
            elem_area=elem_area,
            node_area=node_area,
            gradient_sca=gradient_sca,
            gradient_vec=gradient_vec,
            edge_cross_dxdy=edge_cross_dxdy,
        )
