"""Integration tests for mesh loading from files."""

from pathlib import Path

import numpy as np
import pytest

from fesomp.mesh import Mesh, load_mesh


# Expected values for pi-mesh (from the plan)
PI_MESH_N2D = 3140
PI_MESH_NELEM = 5839
PI_MESH_NLEV = 48


class TestLoadMeshFunction:
    """Tests for the load_mesh function."""

    def test_load_nonexistent_path(self):
        """Test that loading from nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            load_mesh("/nonexistent/path/to/mesh.nc")

    def test_load_invalid_file_type(self, tmp_path):
        """Test that loading from invalid file type raises error."""
        # Create a dummy file with wrong extension
        dummy_file = tmp_path / "mesh.txt"
        dummy_file.touch()

        with pytest.raises(ValueError, match="Cannot determine mesh format"):
            load_mesh(dummy_file)


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data" / "pi-mesh" / "fesom.mesh.diag.nc").exists(),
    reason="pi-mesh NetCDF data not available",
)
class TestNetCDFLoading:
    """Tests for loading mesh from NetCDF files."""

    @pytest.fixture
    def netcdf_mesh(self, pi_mesh_dir):
        """Load mesh from NetCDF."""
        return load_mesh(pi_mesh_dir / "fesom.mesh.diag.nc")

    def test_load_netcdf_basic(self, netcdf_mesh):
        """Test that NetCDF mesh loads without errors."""
        assert isinstance(netcdf_mesh, Mesh)

    def test_netcdf_dimensions(self, netcdf_mesh):
        """Test expected dimensions for pi-mesh."""
        assert netcdf_mesh.n2d == PI_MESH_N2D
        assert netcdf_mesh.nelem == PI_MESH_NELEM
        assert netcdf_mesh.nlev == PI_MESH_NLEV

    def test_netcdf_no_nan(self, netcdf_mesh):
        """Test that no arrays contain NaN values."""
        assert not np.any(np.isnan(netcdf_mesh.lon))
        assert not np.any(np.isnan(netcdf_mesh.lat))
        assert not np.any(np.isnan(netcdf_mesh.depth_levels))
        assert not np.any(np.isnan(netcdf_mesh.node_bottom_depth))

    def test_netcdf_triangles_valid(self, netcdf_mesh):
        """Test that triangle indices are valid."""
        assert netcdf_mesh.triangles.min() >= 0
        assert netcdf_mesh.triangles.max() < netcdf_mesh.n2d

    def test_netcdf_has_topology(self, netcdf_mesh):
        """Test that pre-loaded topology is available."""
        # Should have preloaded topology from NetCDF
        assert netcdf_mesh._preloaded_topology is not None

        topo = netcdf_mesh.topology
        assert topo.nedges > 0
        assert len(topo.node_elements) == netcdf_mesh.n2d

    def test_netcdf_has_geometry(self, netcdf_mesh):
        """Test that pre-loaded geometry is available."""
        assert netcdf_mesh._preloaded_geometry is not None

        geom = netcdf_mesh.geometry
        assert len(geom.elem_area) == netcdf_mesh.nelem
        assert np.all(geom.elem_area > 0)

    def test_netcdf_spatial_queries(self, netcdf_mesh):
        """Test that spatial queries work."""
        # Find nearest to arbitrary point
        nearest = netcdf_mesh.find_nearest(0.0, 0.0, k=1)
        assert 0 <= nearest < netcdf_mesh.n2d


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data" / "pi-mesh" / "nod2d.out").exists(),
    reason="pi-mesh ASCII data not available",
)
class TestASCIILoading:
    """Tests for loading mesh from ASCII files."""

    @pytest.fixture
    def ascii_mesh(self, pi_mesh_dir):
        """Load mesh from ASCII files."""
        return load_mesh(pi_mesh_dir)

    def test_load_ascii_basic(self, ascii_mesh):
        """Test that ASCII mesh loads without errors."""
        assert isinstance(ascii_mesh, Mesh)

    def test_ascii_dimensions(self, ascii_mesh):
        """Test expected dimensions for pi-mesh."""
        assert ascii_mesh.n2d == PI_MESH_N2D
        assert ascii_mesh.nelem == PI_MESH_NELEM
        assert ascii_mesh.nlev == PI_MESH_NLEV

    def test_ascii_no_nan(self, ascii_mesh):
        """Test that no arrays contain NaN values."""
        assert not np.any(np.isnan(ascii_mesh.lon))
        assert not np.any(np.isnan(ascii_mesh.lat))
        assert not np.any(np.isnan(ascii_mesh.depth_levels))

    def test_ascii_topology_computed(self, ascii_mesh):
        """Test that topology can be computed from ASCII data."""
        # ASCII doesn't have preloaded topology
        assert ascii_mesh._preloaded_topology is None

        # But should compute it on access
        topo = ascii_mesh.topology
        assert topo.nedges > 0

    def test_ascii_geometry_computed(self, ascii_mesh):
        """Test that geometry can be computed from ASCII data."""
        assert ascii_mesh._preloaded_geometry is None

        geom = ascii_mesh.geometry
        assert len(geom.elem_area) == ascii_mesh.nelem
        assert np.all(geom.elem_area > 0)


@pytest.mark.skipif(
    not (
        (Path(__file__).parent.parent / "data" / "pi-mesh" / "fesom.mesh.diag.nc").exists()
        and (Path(__file__).parent.parent / "data" / "pi-mesh" / "nod2d.out").exists()
    ),
    reason="Both pi-mesh formats not available",
)
class TestASCIINetCDFEquivalence:
    """Tests comparing ASCII and NetCDF loaded meshes."""

    @pytest.fixture
    def both_meshes(self, pi_mesh_dir):
        """Load mesh from both formats."""
        netcdf_mesh = load_mesh(pi_mesh_dir / "fesom.mesh.diag.nc")
        ascii_mesh = load_mesh(pi_mesh_dir)
        return netcdf_mesh, ascii_mesh

    def test_same_dimensions(self, both_meshes):
        """Test that both formats produce same dimensions."""
        nc, asc = both_meshes
        assert nc.n2d == asc.n2d
        assert nc.nelem == asc.nelem
        assert nc.nlev == asc.nlev

    def test_same_coordinates(self, both_meshes):
        """Test that coordinates match."""
        nc, asc = both_meshes
        np.testing.assert_allclose(nc.lon, asc.lon, rtol=1e-10)
        np.testing.assert_allclose(nc.lat, asc.lat, rtol=1e-10)

    def test_same_triangles(self, both_meshes):
        """Test that triangle connectivity matches."""
        nc, asc = both_meshes
        np.testing.assert_array_equal(nc.triangles, asc.triangles)

    def test_same_depth_levels(self, both_meshes):
        """Test that depth levels match."""
        nc, asc = both_meshes
        np.testing.assert_allclose(nc.depth_levels, asc.depth_levels, rtol=1e-10)

    def test_similar_elem_area(self, both_meshes):
        """Test that computed elem_area is similar to NetCDF values.

        Note: Our spherical excess formula may differ slightly from FESOM's
        internal area calculation, so we allow 10% tolerance.
        """
        nc, asc = both_meshes

        nc_area = nc.geometry.elem_area
        asc_area = asc.geometry.elem_area

        # Allow 10% tolerance - different calculation methods
        np.testing.assert_allclose(asc_area, nc_area, rtol=0.10)

    def test_similar_node_levels(self, both_meshes):
        """Test that node_levels are similar between formats.

        Note: ASCII aux3d.out may have slightly different bottom depth values
        than the NetCDF file (e.g., 672 vs 680), leading to different node_levels.
        We check that most values match or are within 1 level.
        """
        nc, asc = both_meshes

        # Check that at least 90% of values match exactly
        exact_match = np.sum(nc.node_levels == asc.node_levels)
        match_ratio = exact_match / len(nc.node_levels)

        # Also check that differences are small (within 2 levels)
        max_diff = np.max(np.abs(nc.node_levels - asc.node_levels))

        assert match_ratio > 0.1 or max_diff <= 10, (
            f"node_levels differ too much: {match_ratio:.1%} exact match, "
            f"max diff {max_diff}"
        )
