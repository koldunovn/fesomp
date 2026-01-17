"""Unit tests for the transect module."""

import numpy as np
import pytest

from fesomp.plotting.transect import (
    TransectInterpolator,
    great_circle_distance,
    great_circle_path,
    interpolate_transect,
    plot_transect,
    transect,
)


class TestGreatCirclePath:
    """Tests for great_circle_path function."""

    def test_equator_path(self):
        """Test path along the equator."""
        lon, lat, dist = great_circle_path(
            start=(0, 0),
            end=(90, 0),
            npoints=10,
        )

        assert len(lon) == 10
        assert len(lat) == 10
        assert len(dist) == 10

        # All latitudes should be 0 on equator
        np.testing.assert_allclose(lat, 0, atol=1e-10)

        # Longitudes should go from 0 to 90
        assert lon[0] == pytest.approx(0, abs=1e-10)
        assert lon[-1] == pytest.approx(90, abs=1e-10)

        # Distance should start at 0 and increase
        assert dist[0] == pytest.approx(0, abs=1e-10)
        assert dist[-1] > 0

    def test_meridian_path(self):
        """Test path along a meridian."""
        lon, lat, dist = great_circle_path(
            start=(0, -45),
            end=(0, 45),
            npoints=10,
        )

        # All longitudes should be 0 on meridian
        np.testing.assert_allclose(lon, 0, atol=1e-10)

        # Latitudes should go from -45 to 45
        assert lat[0] == pytest.approx(-45, abs=1e-10)
        assert lat[-1] == pytest.approx(45, abs=1e-10)

    def test_same_point(self):
        """Test path with same start and end point."""
        lon, lat, dist = great_circle_path(
            start=(10, 20),
            end=(10, 20),
            npoints=5,
        )

        # All points should be the same
        np.testing.assert_allclose(lon, 10, atol=1e-10)
        np.testing.assert_allclose(lat, 20, atol=1e-10)
        np.testing.assert_allclose(dist, 0, atol=1e-10)

    def test_antipodal_raises(self):
        """Test that antipodal points raise an error."""
        with pytest.raises(ValueError, match="antipodal"):
            great_circle_path(start=(0, 0), end=(180, 0), npoints=10)

    def test_distance_calculation(self):
        """Test that distance calculation is accurate."""
        # Quarter of equator: should be ~10000 km
        _, _, dist = great_circle_path(
            start=(0, 0),
            end=(90, 0),
            npoints=100,
        )

        # Earth circumference at equator ~40000 km, so quarter ~10000 km
        expected_dist = np.pi / 2 * 6371000  # radians * radius
        assert dist[-1] == pytest.approx(expected_dist, rel=1e-6)


class TestGreatCircleDistance:
    """Tests for great_circle_distance function."""

    def test_equator_quarter(self):
        """Test quarter equator distance."""
        dist = great_circle_distance((0, 0), (90, 0))
        expected = np.pi / 2 * 6371000
        assert dist == pytest.approx(expected, rel=1e-6)

    def test_same_point(self):
        """Test distance to same point is zero."""
        dist = great_circle_distance((10, 20), (10, 20))
        assert dist == pytest.approx(0, abs=1e-10)

    def test_pole_to_equator(self):
        """Test pole to equator distance."""
        dist = great_circle_distance((0, 90), (0, 0))
        expected = np.pi / 2 * 6371000  # Quarter circumference
        assert dist == pytest.approx(expected, rel=1e-6)


class TestTransectInterpolator:
    """Tests for TransectInterpolator class."""

    @pytest.fixture
    def simple_points(self):
        """Create simple test points along equator."""
        # Points along equator from -10 to 10 degrees
        lon = np.linspace(-10, 10, 21)
        lat = np.zeros(21)
        data_1d = lon.copy()  # Data equals longitude (1D surface data)
        return lon, lat, data_1d

    @pytest.fixture
    def simple_3d_data(self, simple_points):
        """Create simple 3D test data."""
        lon, lat, _ = simple_points
        nlev = 5
        n2d = len(lon)
        # Data varies with depth and longitude
        depth = np.array([0, 10, 50, 100, 500])
        data_3d = np.zeros((nlev, n2d))
        for lev in range(nlev):
            data_3d[lev] = lon + depth[lev] / 100  # Varies with lon and depth
        return data_3d, depth

    @pytest.fixture
    def interpolator(self, simple_points):
        """Create a simple interpolator."""
        lon, lat, _ = simple_points
        return TransectInterpolator(
            lon=lon,
            lat=lat,
            start=(-5, 0),
            end=(5, 0),
            npoints=11,
            method="nn",
            influence=200000,
        )

    def test_interpolator_creation(self, interpolator):
        """Test interpolator is created correctly."""
        assert len(interpolator.transect_lon) == 11
        assert len(interpolator.transect_lat) == 11
        assert len(interpolator.transect_distance) == 11

    def test_interpolator_coordinates(self, interpolator):
        """Test transect coordinates are correct."""
        # Should go from -5 to 5 on equator
        assert interpolator.transect_lon[0] == pytest.approx(-5, abs=1e-10)
        assert interpolator.transect_lon[-1] == pytest.approx(5, abs=1e-10)
        np.testing.assert_allclose(interpolator.transect_lat, 0, atol=1e-10)

    def test_interpolator_1d_data(self, simple_points, interpolator):
        """Test interpolator with 1D surface data."""
        _, _, data_1d = simple_points
        result = interpolator(data_1d)

        assert result.shape == (11,)
        assert not np.all(np.isnan(result))

    def test_interpolator_2d_data(self, simple_points, simple_3d_data, interpolator):
        """Test interpolator with 2D (nlev, n2d) data."""
        data_3d, _ = simple_3d_data
        result = interpolator(data_3d)

        assert result.shape == (5, 11)  # (nlev, npoints)
        assert not np.all(np.isnan(result))

    def test_interpolator_reuse(self, simple_3d_data, interpolator):
        """Test interpolator can be reused for multiple datasets."""
        data_3d, _ = simple_3d_data
        data_3d_2 = data_3d * 10

        result1 = interpolator(data_3d)
        result2 = interpolator(data_3d_2)

        # Results should be different
        assert not np.allclose(result1, result2, equal_nan=True)
        assert result1.shape == result2.shape

    def test_interpolator_wrong_data_shape(self, interpolator):
        """Test error on wrong data shape."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])  # Wrong n2d

        with pytest.raises(ValueError, match="doesn't match"):
            interpolator(data)

    def test_get_coordinates(self, interpolator):
        """Test get_coordinates method."""
        lon, lat, dist = interpolator.get_coordinates()

        assert len(lon) == 11
        assert len(lat) == 11
        assert len(dist) == 11
        np.testing.assert_array_equal(lon, interpolator.transect_lon)
        np.testing.assert_array_equal(lat, interpolator.transect_lat)
        np.testing.assert_array_equal(dist, interpolator.transect_distance)

    def test_idw_method(self, simple_points, simple_3d_data):
        """Test IDW interpolation method."""
        lon, lat, _ = simple_points
        data_3d, _ = simple_3d_data

        interp = TransectInterpolator(
            lon=lon,
            lat=lat,
            start=(-5, 0),
            end=(5, 0),
            npoints=11,
            method="idw",
            influence=200000,
            k=5,
        )

        result = interp(data_3d)
        assert result.shape == (5, 11)
        assert not np.all(np.isnan(result))

    def test_linear_method(self):
        """Test linear interpolation method."""
        # Need 2D scattered points for linear interpolation (not collinear)
        np.random.seed(42)
        lon = np.random.uniform(-10, 10, 50)
        lat = np.random.uniform(-5, 5, 50)

        # Create 3D data
        nlev = 3
        data_3d = np.zeros((nlev, 50))
        for lev in range(nlev):
            data_3d[lev] = lon + lat + lev

        interp = TransectInterpolator(
            lon=lon,
            lat=lat,
            start=(-5, 0),
            end=(5, 0),
            npoints=11,
            method="linear",
            influence=200000,
        )

        result = interp(data_3d)
        assert result.shape == (3, 11)


class TestInterpolateTransect:
    """Tests for interpolate_transect function."""

    @pytest.fixture
    def grid_data(self):
        """Create grid data for testing."""
        lon = np.linspace(-10, 10, 21)
        lat = np.zeros(21)
        nlev = 4
        data_3d = np.zeros((nlev, 21))
        for lev in range(nlev):
            data_3d[lev] = lon + lev * 10
        return lon, lat, data_3d

    def test_basic_transect(self, grid_data):
        """Test basic transect interpolation."""
        lon, lat, data_3d = grid_data

        result, dist, interp = interpolate_transect(
            data_3d, lon, lat,
            start=(-5, 0),
            end=(5, 0),
            npoints=11,
        )

        assert result.shape == (4, 11)
        assert len(dist) == 11
        assert isinstance(interp, TransectInterpolator)

    def test_with_interpolator(self, grid_data):
        """Test reusing interpolator."""
        lon, lat, data_3d = grid_data

        # First call creates interpolator
        _, _, interp = interpolate_transect(
            data_3d, lon, lat,
            start=(-5, 0),
            end=(5, 0),
            npoints=11,
        )

        # Second call reuses interpolator
        result2, _, interp2 = interpolate_transect(
            data_3d * 2, lon, lat,
            start=(-5, 0),
            end=(5, 0),
            interpolator=interp,
        )

        assert interp is interp2
        assert result2.shape == (4, 11)


class TestPlotTransect:
    """Tests for plot_transect function."""

    @pytest.fixture
    def transect_data(self):
        """Create transect data for plotting."""
        npoints = 50
        nlev = 10
        distance = np.linspace(0, 1000000, npoints)  # 1000 km
        depth = np.linspace(0, 500, nlev)
        # Create some pattern
        data = np.zeros((nlev, npoints))
        for lev in range(nlev):
            data[lev] = np.sin(distance / 100000) * (1 - depth[lev] / 500)
        return data, distance, depth

    def test_basic_plot(self, transect_data):
        """Test basic transect plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data, distance, depth = transect_data
        fig, ax = plot_transect(data, distance, depth)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_with_options(self, transect_data):
        """Test transect plot with various options."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data, distance, depth = transect_data
        fig, ax = plot_transect(
            data, distance, depth,
            title="Test Transect",
            xlabel="Distance",
            ylabel="Depth",
            units="degC",
            cmap="viridis",
            ptype="pcm",
            depth_limits=(0, 200),
        )

        assert ax.get_title() == "Test Transect"
        plt.close(fig)

    def test_plot_on_existing_axes(self, transect_data):
        """Test plotting on existing axes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data, distance, depth = transect_data

        fig_pre, ax_pre = plt.subplots()
        fig, ax = plot_transect(data, distance, depth, ax=ax_pre)

        assert ax is ax_pre
        plt.close(fig)

    def test_plot_requires_2d_data(self):
        """Test that 1D data raises error."""
        import matplotlib
        matplotlib.use("Agg")

        data_1d = np.array([1, 2, 3, 4, 5])
        distance = np.linspace(0, 1000, 5)
        depth = np.array([0, 10, 20])

        with pytest.raises(ValueError, match="2D"):
            plot_transect(data_1d, distance, depth)


class TestTransectConvenience:
    """Tests for the transect convenience function."""

    @pytest.fixture
    def mock_mesh(self):
        """Create a mock mesh object."""
        class MockMesh:
            def __init__(self):
                np.random.seed(42)
                n = 100
                self.lon = np.random.uniform(-10, 10, n)
                self.lat = np.random.uniform(-5, 5, n)
                self.depth_levels = np.array([0, 10, 50, 100, 500])
                self.depth_layers = np.array([5, 30, 75, 250])
                self.nlev = len(self.depth_levels)
                # Element properties
                self.nelem = 80  # Fewer elements than nodes
                self.lon_elem = np.random.uniform(-10, 10, self.nelem)
                self.lat_elem = np.random.uniform(-5, 5, self.nelem)

            @property
            def n2d(self):
                return len(self.lon)

        return MockMesh()

    @pytest.fixture
    def mock_3d_data(self, mock_mesh):
        """Create mock 3D data on layers."""
        nlev = len(mock_mesh.depth_layers)  # Data on layers, not levels
        n2d = len(mock_mesh.lon)
        data = np.zeros((nlev, n2d))
        for lev in range(nlev):
            data[lev] = mock_mesh.lon + mock_mesh.lat + lev
        return data

    def test_transect_function(self, mock_mesh, mock_3d_data):
        """Test the convenience transect function."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax, interp = transect(
            mock_3d_data, mock_mesh,
            start=(-5, 0),
            end=(5, 0),
            npoints=20,
            title="Test",
        )

        assert fig is not None
        assert ax is not None
        assert isinstance(interp, TransectInterpolator)
        plt.close(fig)

    def test_transect_reuse_interpolator(self, mock_mesh, mock_3d_data):
        """Test reusing interpolator with convenience function."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # First call
        _, _, interp = transect(
            mock_3d_data, mock_mesh,
            start=(-5, 0),
            end=(5, 0),
        )

        # Second call with same interpolator
        fig2, ax2, interp2 = transect(
            mock_3d_data * 2, mock_mesh,
            start=(-5, 0),
            end=(5, 0),
            interpolator=interp,
        )

        assert interp is interp2
        plt.close(fig2)

    def test_transect_element_data(self, mock_mesh):
        """Test transect with data on elements instead of nodes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Create data on elements (nelem points instead of n2d)
        nlev = len(mock_mesh.depth_layers)
        data_elem = np.zeros((nlev, mock_mesh.nelem))
        for lev in range(nlev):
            data_elem[lev] = mock_mesh.lon_elem + mock_mesh.lat_elem + lev

        fig, ax, interp = transect(
            data_elem, mock_mesh,
            start=(-5, 0),
            end=(5, 0),
            npoints=20,
            title="Element-based data",
        )

        assert fig is not None
        assert ax is not None
        assert isinstance(interp, TransectInterpolator)
        plt.close(fig)


class TestTransectWithMesh:
    """Integration tests with actual mesh data."""

    def test_transect_mesh_data(self, pi_mesh_dir):
        """Test transect with actual mesh data."""
        from fesomp import load_mesh

        mesh = load_mesh(pi_mesh_dir / "fesom.mesh.diag.nc")

        # Create synthetic 3D data (nlev, n2d)
        nlev = len(mesh.depth_levels)
        data_3d = np.zeros((nlev, mesh.n2d))
        for lev in range(nlev):
            # Use latitude as base, add depth variation
            data_3d[lev] = mesh.lat + mesh.depth_levels[lev] / 100

        data_t, dist_t, interp = interpolate_transect(
            data_3d, mesh.lon, mesh.lat,
            start=(0, -60),
            end=(0, 60),
            npoints=50,
            influence=200000,
        )

        assert data_t.shape == (nlev, 50)
        assert len(dist_t) == 50

        # Should have some valid data
        valid = ~np.isnan(data_t)
        assert np.any(valid)
