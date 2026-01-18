"""Tests for the diagnostics module."""

import numpy as np
import pytest
import xarray as xr

from fesomp import diag


class TestHemisphereMask:
    """Tests for hemisphere_mask function."""

    def test_northern_hemisphere(self):
        lat = np.array([-45.0, -10.0, 0.0, 30.0, 60.0])
        mask = diag.hemisphere_mask(lat, "N")
        expected = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_southern_hemisphere(self):
        lat = np.array([-45.0, -10.0, 0.0, 30.0, 60.0])
        mask = diag.hemisphere_mask(lat, "S")
        expected = np.array([True, True, False, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_both_hemispheres(self):
        lat = np.array([-45.0, -10.0, 0.0, 30.0, 60.0])
        mask = diag.hemisphere_mask(lat, "both")
        expected = np.array([True, True, True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_invalid_hemisphere(self):
        lat = np.array([0.0, 30.0])
        with pytest.raises(ValueError, match="hemisphere must be"):
            diag.hemisphere_mask(lat, "invalid")


class TestComputeLayerThickness:
    """Tests for compute_layer_thickness function."""

    def test_uniform_layers(self):
        depth_levels = np.array([0.0, 10.0, 20.0, 30.0])
        thickness = diag.compute_layer_thickness(depth_levels)
        expected = np.array([10.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(thickness, expected)

    def test_nonuniform_layers(self):
        depth_levels = np.array([0.0, 10.0, 50.0, 200.0])
        thickness = diag.compute_layer_thickness(depth_levels)
        expected = np.array([10.0, 40.0, 150.0])
        np.testing.assert_array_almost_equal(thickness, expected)


class TestSelectDepthIndices:
    """Tests for select_depth_indices function."""

    def test_full_depth(self):
        depth_levels = np.array([0.0, 10.0, 50.0, 100.0, 500.0])
        start, end = diag.select_depth_indices(depth_levels, None)
        assert start == 0
        assert end == 4

    def test_partial_depth(self):
        depth_levels = np.array([0.0, 10.0, 50.0, 100.0, 500.0])
        start, end = diag.select_depth_indices(depth_levels, (20, 200))
        assert start == 1  # Level just above 20m
        assert end == 4  # End index for slicing (includes 100m, up to 500m)

    def test_surface_depth(self):
        depth_levels = np.array([0.0, 10.0, 50.0, 100.0])
        start, end = diag.select_depth_indices(depth_levels, (0, 30))
        assert start == 0
        assert end == 2  # Level at 50m


class TestGetSurfaceArea:
    """Tests for get_surface_area function."""

    def test_1d_input(self):
        area = np.array([100.0, 200.0, 300.0])
        result = diag.get_surface_area(area)
        np.testing.assert_array_equal(result, area)

    def test_2d_input(self):
        area = np.array([[100.0, 200.0, 300.0], [90.0, 180.0, 270.0]])
        result = diag.get_surface_area(area)
        np.testing.assert_array_equal(result, area[0])

    def test_invalid_input(self):
        area = np.ones((2, 3, 4))
        with pytest.raises(ValueError):
            diag.get_surface_area(area)


class TestIceArea:
    """Tests for ice_area function."""

    def test_basic_ice_area(self):
        # 4 nodes: 2 in NH, 2 in SH
        sic = np.array([0.5, 0.8, 0.3, 0.0])  # ice concentration
        node_area = np.array([1e10, 1e10, 1e10, 1e10])  # m²
        lat = np.array([45.0, 60.0, -30.0, -60.0])

        # Northern Hemisphere
        nh_area = diag.ice_area(sic, node_area, lat, hemisphere="N")
        expected_nh = (0.5 + 0.8) * 1e10
        np.testing.assert_almost_equal(nh_area, expected_nh)

        # Southern Hemisphere
        sh_area = diag.ice_area(sic, node_area, lat, hemisphere="S")
        expected_sh = (0.3 + 0.0) * 1e10
        np.testing.assert_almost_equal(sh_area, expected_sh)

        # Global
        global_area = diag.ice_area(sic, node_area, lat, hemisphere="both")
        expected_global = (0.5 + 0.8 + 0.3 + 0.0) * 1e10
        np.testing.assert_almost_equal(global_area, expected_global)

    def test_ice_area_with_time(self):
        # Time series: 2 time steps, 3 nodes
        sic = np.array([[0.5, 0.3, 0.1], [0.8, 0.6, 0.4]])
        node_area = np.array([1e10, 1e10, 1e10])
        lat = np.array([30.0, 60.0, 80.0])  # All NH

        result = diag.ice_area(sic, node_area, lat, hemisphere="N")
        expected = np.array([0.9 * 1e10, 1.8 * 1e10])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ice_area_xarray(self):
        sic = xr.DataArray(
            np.array([[0.5, 0.3], [0.8, 0.6]]),
            dims=["time", "node"],
            coords={"time": [0, 1]},
        )
        node_area = np.array([1e10, 1e10])
        lat = np.array([30.0, 60.0])

        result = diag.ice_area(sic, node_area, lat, hemisphere="N")
        assert isinstance(result, xr.DataArray)
        assert "time" in result.dims


class TestIceVolume:
    """Tests for ice_volume function."""

    def test_basic_ice_volume(self):
        siv = np.array([1.0, 2.0, 0.5])  # effective thickness in m
        node_area = np.array([1e10, 1e10, 1e10])
        lat = np.array([60.0, 70.0, 80.0])

        volume = diag.ice_volume(siv, node_area, lat, hemisphere="N")
        expected = (1.0 + 2.0 + 0.5) * 1e10  # m³
        np.testing.assert_almost_equal(volume, expected)


class TestIceExtent:
    """Tests for ice_extent function."""

    def test_basic_ice_extent(self):
        sic = np.array([0.20, 0.10, 0.50, 0.05])  # concentration
        node_area = np.array([1e10, 1e10, 1e10, 1e10])
        lat = np.array([60.0, 70.0, 80.0, 85.0])

        # Default threshold 0.15
        extent = diag.ice_extent(sic, node_area, lat, hemisphere="N", threshold=0.15)
        # Only first and third nodes exceed threshold
        expected = 2e10
        np.testing.assert_almost_equal(extent, expected)

    def test_ice_extent_different_threshold(self):
        sic = np.array([0.20, 0.10, 0.50, 0.05])
        node_area = np.array([1e10, 1e10, 1e10, 1e10])
        lat = np.array([60.0, 70.0, 80.0, 85.0])

        # Higher threshold
        extent = diag.ice_extent(sic, node_area, lat, hemisphere="N", threshold=0.30)
        expected = 1e10  # Only third node
        np.testing.assert_almost_equal(extent, expected)


class TestHovmoller:
    """Tests for hovmoller function."""

    def test_basic_hovmoller(self):
        # Simple 3D data: (nlev, n2d)
        data = np.array([[10.0, 20.0], [5.0, 15.0], [2.0, 8.0]])  # 3 levels, 2 nodes
        node_area = np.array([1e10, 2e10])  # Different areas

        result = diag.hovmoller(data, node_area)

        # Area-weighted mean for each level
        total_area = 3e10
        expected = np.array(
            [
                (10.0 * 1e10 + 20.0 * 2e10) / total_area,
                (5.0 * 1e10 + 15.0 * 2e10) / total_area,
                (2.0 * 1e10 + 8.0 * 2e10) / total_area,
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_hovmoller_with_time(self):
        # 3D with time: (time, nlev, n2d)
        data = np.ones((2, 3, 4))  # 2 times, 3 levels, 4 nodes
        data[1] = 2.0  # Second time step has value 2
        node_area = np.ones(4) * 1e10

        result = diag.hovmoller(data, node_area)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result[0], 1.0)
        np.testing.assert_array_almost_equal(result[1], 2.0)

    def test_hovmoller_node_area_level_mismatch(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 levels
        node_area = np.array(
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]  # 4 levels
        )

        with pytest.warns(UserWarning, match="one more vertical level"):
            result = diag.hovmoller(data, node_area)

        expected = diag.hovmoller(data, node_area[:3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hovmoller_node_area_level_error(self):
        data = np.ones((4, 2))  # 4 levels
        node_area = np.ones((3, 2))  # 3 levels

        with pytest.raises(ValueError, match="only node_area having one extra"):
            diag.hovmoller(data, node_area)


class TestVolumeMean:
    """Tests for volume_mean function."""

    def test_basic_volume_mean(self):
        # Uniform temperature
        temp = np.ones((3, 4)) * 15.0  # 3 layers, 4 nodes
        node_area = np.ones(4) * 1e10
        depth_levels = np.array([0.0, 10.0, 50.0, 100.0])

        result = diag.volume_mean(temp, node_area, depth_levels)
        np.testing.assert_almost_equal(result, 15.0)

    def test_volume_mean_depth_range(self):
        # Temperature varying with depth
        temp = np.array(
            [
                [20.0, 20.0],  # Layer 0: 0-10m
                [15.0, 15.0],  # Layer 1: 10-50m
                [10.0, 10.0],  # Layer 2: 50-100m
            ]
        )
        node_area = np.ones(2) * 1e10
        depth_levels = np.array([0.0, 10.0, 50.0, 100.0])

        # Mean in top 50m (layers 0 and 1)
        result = diag.volume_mean(temp, node_area, depth_levels, depth_range=(0, 50))

        # Volume-weighted: (20*10 + 15*40) / 50 = (200 + 600) / 50 = 16
        expected = 16.0
        np.testing.assert_almost_equal(result, expected)


class TestHeatContent:
    """Tests for heat_content function."""

    def test_basic_heat_content(self):
        # Simple case: uniform 10°C
        temp = np.ones((2, 2)) * 10.0  # 2 layers, 2 nodes
        node_area = np.ones(2) * 1e10  # m²
        depth_levels = np.array([0.0, 100.0, 200.0])  # 100m layers

        result = diag.heat_content(
            temp, node_area, depth_levels, reference_temp=0.0, rho=1025.0, cp=3985.0
        )

        # Volume per node per layer = 1e10 * 100 = 1e12 m³
        # Total volume = 2 nodes * 2 layers * 1e12 = 4e12 m³
        # OHC = rho * cp * dT * volume = 1025 * 3985 * 10 * 4e12
        expected = 1025 * 3985 * 10 * 4e12
        np.testing.assert_almost_equal(result, expected)


class TestTotalVolume:
    """Tests for total_volume function."""

    def test_basic_total_volume(self):
        node_area = np.ones(4) * 1e10  # 4 nodes, each 1e10 m²
        depth_levels = np.array([0.0, 100.0, 500.0])  # 2 layers

        result = diag.total_volume(node_area, depth_levels)

        # Total = 4 * 1e10 * (100 + 400) = 4 * 1e10 * 500 = 2e13 m³
        expected = 2e13
        np.testing.assert_almost_equal(result, expected)


class TestMixedLayerDepth:
    """Tests for mixed_layer_depth function."""

    def test_temperature_criterion(self):
        # Temperature profile: 20°C at surface, decreasing with depth
        temp = np.array(
            [
                [20.0, 20.0],  # Level 0
                [19.9, 18.0],  # Level 1
                [19.7, 15.0],  # Level 2
                [15.0, 10.0],  # Level 3
            ]
        )
        depth_levels = np.array([0.0, 10.0, 50.0, 100.0])

        mld = diag.mixed_layer_depth(
            temp, depth_levels, threshold=0.5, criterion="temperature", reference_depth=0
        )

        # Node 0: T drops by 0.5 at level 3 (100m)
        # Node 1: T drops by 0.5 between level 0 and 1 (10m)
        assert mld.shape == (2,)
        np.testing.assert_almost_equal(mld[1], 10.0)

    def test_density_criterion(self):
        # Density profile: increasing with depth
        rho = np.array(
            [
                [1024.0, 1024.0],
                [1024.02, 1024.05],
                [1024.05, 1024.15],
                [1025.0, 1025.0],
            ]
        )
        depth_levels = np.array([0.0, 10.0, 50.0, 100.0])

        mld = diag.mixed_layer_depth(
            rho, depth_levels, threshold=0.03, criterion="density", reference_depth=0
        )

        assert mld.shape == (2,)


class TestMOC:
    """Tests for MOC function."""

    def test_basic_moc(self):
        # Simple test: uniform upward velocity on nodes at level interfaces
        nz = 3  # nz levels (level interfaces)
        n2d = 10  # nodes
        w = np.ones((nz, n2d)) * 1e-3  # 1 mm/s upward

        node_area = np.ones(n2d) * 1e10  # 10000 km² each
        lat = np.linspace(-80, 80, n2d)
        depth_levels = np.array([0.0, 100.0, 500.0])

        moc_result, lat_bins = diag.moc(
            w, node_area, lat, depth_levels, lat_range=(-90, 90), nlats=19
        )

        assert moc_result.shape == (nz, 19)
        assert len(lat_bins) == 19
        # MOC should be positive (upward transport)
        assert moc_result.sum() > 0

    def test_moc_with_mask(self):
        nz = 2  # nz levels
        n2d = 6  # nodes
        w = np.ones((nz, n2d)) * 1e-3

        node_area = np.ones(n2d) * 1e10
        lat = np.array([-60, -30, 0, 30, 60, 80])
        depth_levels = np.array([0.0, 100.0])

        # Mask: only use half the nodes
        mask = np.array([True, True, True, False, False, False])

        moc_result, _ = diag.moc(
            w, node_area, lat, depth_levels, mask=mask, nlats=10
        )

        # Result should exist but be smaller than without mask
        assert moc_result.shape == (nz, 10)


class TestGeoJSONLoading:
    """Tests for GeoJSON loading utilities."""

    def test_list_moc_basins(self):
        basins = diag.list_moc_basins()
        assert isinstance(basins, list)
        assert len(basins) > 0
        assert "Atlantic_MOC" in basins

    def test_list_available_regions(self):
        regions = diag.list_available_regions("oceanBasins")
        assert isinstance(regions, list)
        assert len(regions) > 0

    def test_list_nino_regions(self):
        regions = diag.list_available_regions("NinoRegions")
        assert "Nino 3.4" in regions


class TestXarrayIntegration:
    """Tests for xarray integration."""

    def test_ice_area_preserves_coords(self):
        time_coords = np.array([0, 1, 2])
        sic = xr.DataArray(
            np.random.rand(3, 10),
            dims=["time", "node"],
            coords={"time": time_coords},
        )
        node_area = np.ones(10) * 1e10
        lat = np.linspace(-80, 80, 10)

        result = diag.ice_area(sic, node_area, lat, hemisphere="both")

        assert isinstance(result, xr.DataArray)
        assert "time" in result.dims
        np.testing.assert_array_equal(result.coords["time"].values, time_coords)

    def test_volume_mean_xarray(self):
        temp = xr.DataArray(
            np.ones((5, 4, 10)) * 15.0, dims=["time", "lev", "node"]  # 5 times, 4 layers
        )
        node_area = np.ones(10) * 1e10
        depth_levels = np.array([0.0, 10.0, 50.0, 100.0, 500.0])

        result = diag.volume_mean(temp, node_area, depth_levels)

        assert isinstance(result, xr.DataArray)
        assert result.shape == (5,)  # One value per time step
