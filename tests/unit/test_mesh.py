"""Unit tests for the Mesh class."""

import numpy as np
import pytest

from fesomp.mesh import Mesh


class TestMeshCreation:
    """Tests for Mesh object creation and validation."""

    def test_create_single_triangle_mesh(self, single_triangle_mesh_data):
        """Test creating a mesh with a single triangle."""
        mesh = Mesh(**single_triangle_mesh_data)

        assert mesh.n2d == 3
        assert mesh.nelem == 1
        assert mesh.nlev == 3

    def test_create_two_triangle_mesh(self, two_triangle_mesh_data):
        """Test creating a mesh with two triangles."""
        mesh = Mesh(**two_triangle_mesh_data)

        assert mesh.n2d == 4
        assert mesh.nelem == 2
        assert mesh.nlev == 2

    def test_mesh_dtype_conversion(self, single_triangle_mesh_data):
        """Test that mesh data is converted to correct dtypes."""
        # Pass in integer lon/lat (wrong dtype)
        data = single_triangle_mesh_data.copy()
        data["lon"] = [0, 1, 0]  # list of ints
        data["lat"] = [0, 0, 1]

        mesh = Mesh(**data)

        assert mesh.lon.dtype == np.float64
        assert mesh.lat.dtype == np.float64
        assert mesh.triangles.dtype == np.int32

    def test_mesh_repr(self, single_triangle_mesh_data):
        """Test mesh string representation."""
        mesh = Mesh(**single_triangle_mesh_data)
        repr_str = repr(mesh)

        assert "n2d=3" in repr_str
        assert "nelem=1" in repr_str
        assert "nlev=3" in repr_str


class TestMeshValidation:
    """Tests for mesh data validation."""

    def test_invalid_lon_lat_shape(self, single_triangle_mesh_data):
        """Test that mismatched lon/lat shapes raise error."""
        data = single_triangle_mesh_data.copy()
        data["lon"] = np.array([0.0, 1.0])  # Wrong length

        with pytest.raises(ValueError, match="lon and lat must have the same length"):
            Mesh(**data)

    def test_invalid_triangle_shape(self, single_triangle_mesh_data):
        """Test that invalid triangle shape raises error."""
        data = single_triangle_mesh_data.copy()
        data["triangles"] = np.array([[0, 1]])  # Only 2 vertices

        with pytest.raises(ValueError, match="triangles must have shape"):
            Mesh(**data)

    def test_negative_triangle_indices(self, single_triangle_mesh_data):
        """Test that negative triangle indices raise error."""
        data = single_triangle_mesh_data.copy()
        data["triangles"] = np.array([[-1, 1, 2]])

        with pytest.raises(ValueError, match="non-negative values"):
            Mesh(**data)

    def test_triangle_indices_out_of_bounds(self, single_triangle_mesh_data):
        """Test that out-of-bounds triangle indices raise error."""
        data = single_triangle_mesh_data.copy()
        data["triangles"] = np.array([[0, 1, 10]])  # 10 > n2d

        with pytest.raises(ValueError, match="exceed number of nodes"):
            Mesh(**data)


class TestMeshProperties:
    """Tests for mesh computed properties."""

    def test_n2d_property(self, two_triangle_mesh_data):
        """Test n2d property returns correct node count."""
        mesh = Mesh(**two_triangle_mesh_data)
        assert mesh.n2d == 4

    def test_nelem_property(self, two_triangle_mesh_data):
        """Test nelem property returns correct element count."""
        mesh = Mesh(**two_triangle_mesh_data)
        assert mesh.nelem == 2

    def test_lon_elem_property(self, two_triangle_mesh_data):
        """Test lon_elem returns element center longitudes."""
        mesh = Mesh(**two_triangle_mesh_data)

        assert len(mesh.lon_elem) == mesh.nelem
        # Element centers should be within the range of node coordinates
        assert mesh.lon_elem.min() >= mesh.lon.min()
        assert mesh.lon_elem.max() <= mesh.lon.max()

    def test_lat_elem_property(self, two_triangle_mesh_data):
        """Test lat_elem returns element center latitudes."""
        mesh = Mesh(**two_triangle_mesh_data)

        assert len(mesh.lat_elem) == mesh.nelem
        # Element centers should be within the range of node coordinates
        assert mesh.lat_elem.min() >= mesh.lat.min()
        assert mesh.lat_elem.max() <= mesh.lat.max()

    def test_elem_coords_cached(self, two_triangle_mesh_data):
        """Test that element coordinates are cached after first access."""
        mesh = Mesh(**two_triangle_mesh_data)

        # Access lon_elem first time
        lon1 = mesh.lon_elem
        lat1 = mesh.lat_elem

        # Access again - should be same objects (cached)
        lon2 = mesh.lon_elem
        lat2 = mesh.lat_elem

        assert lon1 is lon2
        assert lat1 is lat2


class TestMeshQueries:
    """Tests for mesh query methods."""

    def test_subset_by_bbox(self, two_triangle_mesh_data):
        """Test bounding box subset query."""
        mesh = Mesh(**two_triangle_mesh_data)

        # Query for nodes in left half
        indices = mesh.subset_by_bbox(lon_min=-0.5, lon_max=0.5, lat_min=-0.5, lat_max=1.5)

        assert 0 in indices  # Node at (0, 0)
        assert 1 not in indices  # Node at (1, 0)

    def test_subset_by_bbox_all_nodes(self, two_triangle_mesh_data):
        """Test bbox that includes all nodes."""
        mesh = Mesh(**two_triangle_mesh_data)

        indices = mesh.subset_by_bbox(
            lon_min=-1, lon_max=3, lat_min=-1, lat_max=2
        )

        assert len(indices) == 4
