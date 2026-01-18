"""Tests for the diagnostics module."""

import numpy as np
import pytest
import xarray as xr

from fesomp import diag


class TestElemToNodes:
    """Tests for elem_to_nodes function."""

    def test_single_triangle(self):
        # Single triangle: value on element should distribute to all 3 nodes
        triangles = np.array([[0, 1, 2]])
        elem_area = np.array([1e10])  # 1e10 m²

        # Node area is elem_area / 3 for each node (each node touches 1 element)
        node_area = np.array([1e10 / 3, 1e10 / 3, 1e10 / 3])

        # Element has value 6.0
        data = np.array([6.0])

        result = diag.elem_to_nodes(data, triangles, elem_area, node_area)

        # Each node should get the area-weighted average = 6.0
        # (only one element, so it's just the element value)
        np.testing.assert_array_almost_equal(result, [6.0, 6.0, 6.0])

    def test_two_triangles(self):
        # Two triangles sharing edge (nodes 1 and 3)
        #    3
        #   /|\
        #  / | \
        # 0--1--2
        triangles = np.array([[0, 1, 3], [1, 2, 3]])
        elem_area = np.array([1e10, 1e10])

        # Node areas: corner nodes touch 1 element, center nodes touch 2
        # Node 0: 1e10/3
        # Node 1: 2*1e10/3
        # Node 2: 1e10/3
        # Node 3: 2*1e10/3
        node_area = np.array([1e10 / 3, 2e10 / 3, 1e10 / 3, 2e10 / 3])

        # Elements have values 10 and 20
        data = np.array([10.0, 20.0])

        result = diag.elem_to_nodes(data, triangles, elem_area, node_area)

        # Node 0 touches only elem 0: value = 10
        # Node 2 touches only elem 1: value = 20
        # Nodes 1 and 3 touch both: area-weighted average = (10*1e10 + 20*1e10) / 2e10 = 15
        np.testing.assert_almost_equal(result[0], 10.0)
        np.testing.assert_almost_equal(result[2], 20.0)
        np.testing.assert_almost_equal(result[1], 15.0)
        np.testing.assert_almost_equal(result[3], 15.0)

    def test_with_time_dimension(self):
        # Single triangle with time dimension
        triangles = np.array([[0, 1, 2]])
        elem_area = np.array([1e10])
        node_area = np.array([1e10 / 3, 1e10 / 3, 1e10 / 3])

        # 3 time steps, 1 element
        data = np.array([[5.0], [10.0], [15.0]])

        result = diag.elem_to_nodes(data, triangles, elem_area, node_area)

        assert result.shape == (3, 3)
        np.testing.assert_array_almost_equal(result[0], [5.0, 5.0, 5.0])
        np.testing.assert_array_almost_equal(result[1], [10.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(result[2], [15.0, 15.0, 15.0])

    def test_xarray_input(self):
        triangles = np.array([[0, 1, 2]])
        elem_area = np.array([1e10])
        node_area = np.array([1e10 / 3, 1e10 / 3, 1e10 / 3])

        data = xr.DataArray(
            np.array([[7.0], [14.0]]),
            dims=["time", "elem"],
            coords={"time": [0, 1]},
        )

        result = diag.elem_to_nodes(data, triangles, elem_area, node_area)

        assert isinstance(result, xr.DataArray)
        assert "time" in result.dims
        assert result.shape == (2, 3)


class TestElemToNodes3d:
    """Tests for elem_to_nodes_3d function."""

    def test_single_triangle_uniform_levels(self):
        # Single triangle, 2 levels, all nodes active at all levels
        triangles = np.array([[0, 1, 2]])
        elem_area = np.array([1e10])
        elem_levels = np.array([2])  # Element active at levels 0 and 1

        # node_area: (nlev, n2d) - same area at all levels
        node_area = np.array([
            [1e10 / 3, 1e10 / 3, 1e10 / 3],  # level 0
            [1e10 / 3, 1e10 / 3, 1e10 / 3],  # level 1
        ])

        # Data: (nelem, nlev)
        data = np.array([[5.0, 10.0]])

        result = diag.elem_to_nodes_3d(data, triangles, elem_area, node_area, elem_levels)

        assert result.shape == (3, 2)  # (n2d, nlev)
        # Level 0: all nodes get 5.0
        np.testing.assert_array_almost_equal(result[:, 0], [5.0, 5.0, 5.0])
        # Level 1: all nodes get 10.0
        np.testing.assert_array_almost_equal(result[:, 1], [10.0, 10.0, 10.0])

    def test_varying_depth(self):
        # Two triangles, one shallower than the other
        #    3
        #   /|\
        #  / | \
        # 0--1--2
        triangles = np.array([[0, 1, 3], [1, 2, 3]])
        elem_area = np.array([1e10, 1e10])
        elem_levels = np.array([2, 1])  # First element 2 levels, second only 1

        nlev = 2
        # Node area at level 0: all elements active
        # Node area at level 1: only elem 0 active
        # Node 0: touches elem 0 only
        # Node 1: touches elem 0 and 1 at level 0, only elem 0 at level 1
        # Node 2: touches elem 1 only
        # Node 3: touches elem 0 and 1 at level 0, only elem 0 at level 1
        node_area = np.array([
            [1e10 / 3, 2e10 / 3, 1e10 / 3, 2e10 / 3],  # level 0
            [1e10 / 3, 1e10 / 3, 0.0, 1e10 / 3],  # level 1 (node 2 inactive)
        ])

        # Data: (nelem, nlev)
        data = np.array([
            [10.0, 20.0],  # elem 0
            [30.0, 0.0],   # elem 1 (level 1 inactive, value doesn't matter)
        ])

        result = diag.elem_to_nodes_3d(data, triangles, elem_area, node_area, elem_levels)

        assert result.shape == (4, 2)  # (n2d, nlev)

        # Level 0: both elements contribute
        # Node 0: only elem 0 -> 10
        np.testing.assert_almost_equal(result[0, 0], 10.0)
        # Node 2: only elem 1 -> 30
        np.testing.assert_almost_equal(result[2, 0], 30.0)
        # Nodes 1, 3: area-weighted avg of 10 and 30 -> 20
        np.testing.assert_almost_equal(result[1, 0], 20.0)
        np.testing.assert_almost_equal(result[3, 0], 20.0)

        # Level 1: only elem 0 contributes
        # Node 0, 1, 3: value 20
        np.testing.assert_almost_equal(result[0, 1], 20.0)
        np.testing.assert_almost_equal(result[1, 1], 20.0)
        np.testing.assert_almost_equal(result[3, 1], 20.0)
        # Node 2: inactive (NaN)
        assert np.isnan(result[2, 1])

    def test_with_time_dimension(self):
        triangles = np.array([[0, 1, 2]])
        elem_area = np.array([1e10])
        elem_levels = np.array([2])
        node_area = np.array([
            [1e10 / 3, 1e10 / 3, 1e10 / 3],
            [1e10 / 3, 1e10 / 3, 1e10 / 3],
        ])

        # Data: (time, nelem, nlev)
        data = np.array([
            [[1.0, 2.0]],   # time 0
            [[10.0, 20.0]], # time 1
        ])

        result = diag.elem_to_nodes_3d(data, triangles, elem_area, node_area, elem_levels)

        assert result.shape == (2, 3, 2)  # (time, n2d, nlev)
        np.testing.assert_array_almost_equal(result[0, :, 0], [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result[1, :, 1], [20.0, 20.0, 20.0])

    def test_xarray_input(self):
        triangles = np.array([[0, 1, 2]])
        elem_area = np.array([1e10])
        elem_levels = np.array([2])
        node_area = np.array([
            [1e10 / 3, 1e10 / 3, 1e10 / 3],
            [1e10 / 3, 1e10 / 3, 1e10 / 3],
        ])

        data = xr.DataArray(
            np.array([[[5.0, 10.0]]]),
            dims=["time", "elem", "lev"],
            coords={"time": [0]},
        )

        result = diag.elem_to_nodes_3d(data, triangles, elem_area, node_area, elem_levels)

        assert isinstance(result, xr.DataArray)
        assert "time" in result.dims
        assert result.shape == (1, 3, 2)


class TestComputeNodeLump:
    """Tests for compute_node_lump function."""

    def test_single_triangle(self):
        triangles = np.array([[0, 1, 2]])
        elem_area = np.array([3e10])
        n2d = 3

        result = diag.compute_node_lump(triangles, elem_area, n2d)

        # Each node gets 1/3 of the element area
        expected = np.array([1e10, 1e10, 1e10])
        np.testing.assert_array_almost_equal(result, expected)

    def test_two_triangles(self):
        # Two triangles sharing edge
        triangles = np.array([[0, 1, 3], [1, 2, 3]])
        elem_area = np.array([3e10, 6e10])
        n2d = 4

        result = diag.compute_node_lump(triangles, elem_area, n2d)

        # Node 0: 3e10/3 = 1e10
        # Node 1: 3e10/3 + 6e10/3 = 3e10
        # Node 2: 6e10/3 = 2e10
        # Node 3: 3e10/3 + 6e10/3 = 3e10
        expected = np.array([1e10, 3e10, 2e10, 3e10])
        np.testing.assert_array_almost_equal(result, expected)


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

    def test_volume_mean_node_area_level_mismatch(self):
        temp = np.ones((3, 2)) * 5.0  # 3 layers
        node_area = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )
        depth_levels = np.array([0.0, 10.0, 20.0, 30.0])

        with pytest.warns(UserWarning, match="node_area has one more vertical level"):
            result = diag.volume_mean(temp, node_area, depth_levels)

        expected = diag.volume_mean(temp, node_area[:3], depth_levels)
        np.testing.assert_allclose(result, expected)

    def test_volume_mean_node_area_level_error(self):
        temp = np.ones((4, 2))  # 4 layers
        node_area = np.ones((3, 2))  # 3 levels
        depth_levels = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

        with pytest.raises(ValueError, match="only node_area having one extra"):
            diag.volume_mean(temp, node_area, depth_levels)


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

    def test_heat_content_node_area_level_mismatch(self):
        temp = np.ones((3, 2)) * 8.0  # 3 layers
        node_area = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )
        depth_levels = np.array([0.0, 10.0, 20.0, 30.0])

        with pytest.warns(UserWarning, match="node_area has one more vertical level"):
            result = diag.heat_content(temp, node_area, depth_levels)

        expected = diag.heat_content(temp, node_area[:3], depth_levels)
        np.testing.assert_allclose(result, expected)

    def test_heat_content_node_area_level_error(self):
        temp = np.ones((4, 2)) * 8.0  # 4 layers
        node_area = np.ones((3, 2))  # 3 levels
        depth_levels = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

        with pytest.raises(ValueError, match="only node_area having one extra"):
            diag.heat_content(temp, node_area, depth_levels)


class TestTotalVolume:
    """Tests for total_volume function."""

    def test_basic_total_volume(self):
        node_area = np.ones(4) * 1e10  # 4 nodes, each 1e10 m²
        depth_levels = np.array([0.0, 100.0, 500.0])  # 2 layers

        result = diag.total_volume(node_area, depth_levels)

        # Total = 4 * 1e10 * (100 + 400) = 4 * 1e10 * 500 = 2e13 m³
        expected = 2e13
        np.testing.assert_almost_equal(result, expected)

    def test_total_volume_node_area_level_mismatch(self):
        node_area = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )
        depth_levels = np.array([0.0, 10.0, 20.0, 30.0])

        with pytest.warns(UserWarning, match="node_area has one more vertical level"):
            result = diag.total_volume(node_area, depth_levels)

        expected = diag.total_volume(node_area[:3], depth_levels)
        np.testing.assert_allclose(result, expected)

    def test_total_volume_node_area_level_error(self):
        node_area = np.ones((2, 2))  # 2 levels
        depth_levels = np.array([0.0, 10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="only node_area having one extra"):
            diag.total_volume(node_area, depth_levels)


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
