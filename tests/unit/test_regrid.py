"""Unit tests for the regrid module."""

import numpy as np
import pytest

from fesomp.plotting.regrid import (
    RegridInterpolator,
    create_regular_grid,
    regrid,
)


class TestCreateRegularGrid:
    """Tests for create_regular_grid function."""

    def test_default_global_grid(self):
        """Test default global grid creation."""
        lon1d, lat1d, lon2d, lat2d = create_regular_grid()

        assert len(lon1d) == 360
        assert len(lat1d) == 180
        assert lon2d.shape == (180, 360)
        assert lat2d.shape == (180, 360)

    def test_custom_box_and_res(self):
        """Test custom bounding box and resolution."""
        lon1d, lat1d, lon2d, lat2d = create_regular_grid(
            box=(-10, 10, 40, 60), res=(20, 10)
        )

        assert len(lon1d) == 20
        assert len(lat1d) == 10
        assert lon1d[0] == -10
        assert lon1d[-1] == 10
        assert lat1d[0] == 40
        assert lat1d[-1] == 60


class TestRegrid:
    """Tests for regrid function."""

    @pytest.fixture
    def simple_points(self):
        """Create simple test points."""
        # 4 points at corners of a square
        lon = np.array([0.0, 1.0, 0.0, 1.0])
        lat = np.array([0.0, 0.0, 1.0, 1.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        return lon, lat, data

    def test_regrid_nearest_neighbor(self, simple_points):
        """Test nearest neighbor interpolation."""
        lon, lat, data = simple_points

        data_reg, lon_reg, lat_reg = regrid(
            data, lon, lat,
            box=(-0.5, 1.5, -0.5, 1.5),
            res=(10, 10),
            method="nn",
            influence=200000,  # ~200 km
        )

        assert data_reg.shape == (10, 10)
        assert not np.all(np.isnan(data_reg))  # Some valid data

    def test_regrid_influence_radius(self, simple_points):
        """Test that influence radius affects results."""
        lon, lat, data = simple_points

        # Very small influence - should get mostly NaN
        data_small, _, _ = regrid(
            data, lon, lat,
            box=(-0.5, 1.5, -0.5, 1.5),
            res=(10, 10),
            method="nn",
            influence=1000,  # 1 km - very small
        )

        # Large influence - should get mostly valid data
        data_large, _, _ = regrid(
            data, lon, lat,
            box=(-0.5, 1.5, -0.5, 1.5),
            res=(10, 10),
            method="nn",
            influence=500000,  # 500 km - large
        )

        nan_count_small = np.sum(np.isnan(data_small))
        nan_count_large = np.sum(np.isnan(data_large))

        assert nan_count_small > nan_count_large

    def test_regrid_idw(self, simple_points):
        """Test inverse distance weighting interpolation."""
        lon, lat, data = simple_points

        data_reg, _, _ = regrid(
            data, lon, lat,
            box=(-0.5, 1.5, -0.5, 1.5),
            res=(10, 10),
            method="idw",
            influence=500000,
        )

        assert data_reg.shape == (10, 10)
        # IDW should produce smooth values between input points
        valid_data = data_reg[~np.isnan(data_reg)]
        assert len(valid_data) > 0


class TestRegridInterpolator:
    """Tests for RegridInterpolator class."""

    @pytest.fixture
    def interpolator(self):
        """Create a simple interpolator."""
        lon = np.array([0.0, 1.0, 0.0, 1.0])
        lat = np.array([0.0, 0.0, 1.0, 1.0])
        return RegridInterpolator(
            lon=lon,
            lat=lat,
            box=(-0.5, 1.5, -0.5, 1.5),
            res=(10, 10),
            method="nn",
            influence=500000,
        )

    def test_interpolator_creation(self, interpolator):
        """Test interpolator is created correctly."""
        assert interpolator.lon_reg.shape == (10,)
        assert interpolator.lat_reg.shape == (10,)

    def test_interpolator_call(self, interpolator):
        """Test interpolator can be called."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        data_reg, lon_reg, lat_reg = interpolator(data)

        assert data_reg.shape == (10, 10)
        assert len(lon_reg) == 10
        assert len(lat_reg) == 10

    def test_interpolator_reuse(self, interpolator):
        """Test interpolator can be reused for multiple datasets."""
        data1 = np.array([1.0, 2.0, 3.0, 4.0])
        data2 = np.array([10.0, 20.0, 30.0, 40.0])

        result1, _, _ = interpolator(data1)
        result2, _, _ = interpolator(data2)

        # Results should be different
        assert not np.allclose(result1, result2, equal_nan=True)

        # But shapes should be the same
        assert result1.shape == result2.shape

    def test_interpolator_wrong_data_length(self, interpolator):
        """Test error on wrong data length."""
        data = np.array([1.0, 2.0])  # Wrong length

        with pytest.raises(ValueError, match="Data length"):
            interpolator(data)


class TestRegridWithMesh:
    """Integration tests with actual mesh data."""

    def test_regrid_mesh_data(self, pi_mesh_dir):
        """Test regridding actual mesh data."""
        from fesomp import load_mesh

        mesh = load_mesh(pi_mesh_dir / "fesom.mesh.diag.nc")

        # Use latitude as test data
        data_reg, lon_reg, lat_reg = regrid(
            mesh.lat, mesh.lon, mesh.lat,
            res=(180, 90),
            influence=100000,
        )

        assert data_reg.shape == (90, 180)
        # Latitude values should be preserved approximately
        valid = ~np.isnan(data_reg)
        assert np.any(valid)
