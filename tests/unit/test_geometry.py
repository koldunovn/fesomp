"""Unit tests for the Geometry class and computation."""

import numpy as np
import pytest

from fesomp.mesh.geometry import (
    EARTH_RADIUS_M,
    Geometry,
    compute_geometry,
    compute_node_area,
    spherical_triangle_area,
)
from fesomp.mesh.topology import compute_topology


class TestSphericalTriangleArea:
    """Tests for spherical triangle area computation."""

    def test_small_triangle_near_equator(self):
        """Test area of a small triangle near the equator."""
        # Small 1-degree triangle at equator
        lon = np.array([0.0, 1.0, 0.5])
        lat = np.array([0.0, 0.0, 0.866])  # ~1 degree
        triangles = np.array([[0, 1, 2]])

        areas = spherical_triangle_area(lon, lat, triangles)

        assert len(areas) == 1
        assert areas[0] > 0

        # Rough estimate: 1 degree ~ 111 km at equator
        # Triangle area ~ 0.5 * base * height ~ 0.5 * 111 * 96 ~ 5300 km^2
        # In m^2: ~ 5.3e9 m^2
        area_km2 = areas[0] / 1e6
        assert 4000 < area_km2 < 7000  # Reasonable range

    def test_known_spherical_triangle(self):
        """Test against a known spherical triangle area."""
        # Triangle from pole to equator (octant of sphere)
        # Area = 1/8 of sphere = (4*pi*R^2) / 8 = pi*R^2 / 2
        lon = np.array([0.0, 90.0, 0.0])
        lat = np.array([0.0, 0.0, 90.0])
        triangles = np.array([[0, 1, 2]])

        areas = spherical_triangle_area(lon, lat, triangles)

        expected_area = np.pi * EARTH_RADIUS_M**2 / 2
        relative_error = abs(areas[0] - expected_area) / expected_area

        # Should be accurate to within 1%
        assert relative_error < 0.01

    def test_all_areas_positive(self):
        """Test that all computed areas are positive."""
        # Multiple triangles
        lon = np.array([0.0, 1.0, 0.5, 2.0])
        lat = np.array([0.0, 0.0, 1.0, 1.0])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])

        areas = spherical_triangle_area(lon, lat, triangles)

        assert len(areas) == 2
        assert np.all(areas > 0)

    def test_tiny_triangle(self):
        """Test that tiny triangles still have positive area."""
        # Very small triangle (0.001 degrees)
        lon = np.array([0.0, 0.001, 0.0005])
        lat = np.array([0.0, 0.0, 0.001])
        triangles = np.array([[0, 1, 2]])

        areas = spherical_triangle_area(lon, lat, triangles)

        assert areas[0] > 0


class TestNodeArea:
    """Tests for node area computation."""

    def test_single_triangle_node_area(self):
        """Test node area for single triangle."""
        lon = np.array([0.0, 1.0, 0.5])
        lat = np.array([0.0, 0.0, 1.0])
        triangles = np.array([[0, 1, 2]])

        elem_area = spherical_triangle_area(lon, lat, triangles)
        node_levels = np.array([2, 2, 2], dtype=np.int32)
        nlev = 2

        node_area = compute_node_area(elem_area, triangles, node_levels, nlev)

        # Each node gets 1/3 of the element area
        expected_node_area = elem_area[0] / 3

        # Check surface level
        np.testing.assert_allclose(
            node_area[0, :], expected_node_area, rtol=1e-10
        )

        # Sum of node areas at surface = total elem area
        np.testing.assert_allclose(
            node_area[0, :].sum(), elem_area[0], rtol=1e-10
        )

    def test_node_area_depth_masking(self):
        """Test that node area is zero below active depth."""
        lon = np.array([0.0, 1.0, 0.5])
        lat = np.array([0.0, 0.0, 1.0])
        triangles = np.array([[0, 1, 2]])

        elem_area = spherical_triangle_area(lon, lat, triangles)

        # Node 0 has only 1 level, others have 3
        node_levels = np.array([1, 3, 3], dtype=np.int32)
        nlev = 3

        node_area = compute_node_area(elem_area, triangles, node_levels, nlev)

        # Node 0 should have area at level 0 but not levels 1, 2
        assert node_area[0, 0] > 0
        assert node_area[1, 0] == 0
        assert node_area[2, 0] == 0

        # Node 1 should have area at all levels
        assert np.all(node_area[:, 1] > 0)


class TestGeometryComputation:
    """Tests for full geometry computation."""

    def test_compute_geometry(self, two_triangle_mesh_data):
        """Test complete geometry computation."""
        data = two_triangle_mesh_data
        topo = compute_topology(data["triangles"])

        geom = compute_geometry(
            data["lon"],
            data["lat"],
            data["triangles"],
            data["node_levels"],
            data["nlev"],
            topo,
        )

        assert isinstance(geom, Geometry)
        assert len(geom.elem_area) == 2
        assert geom.node_area.shape == (2, 4)
        assert np.all(geom.elem_area > 0)


class TestLazyGeometryLoading:
    """Tests for lazy geometry loading in Mesh."""

    def test_geometry_computed_on_access(self, two_triangle_mesh_data):
        """Test that geometry is computed lazily."""
        from fesomp.mesh import Mesh

        mesh = Mesh(**two_triangle_mesh_data)

        # Internal geometry should be None before access
        assert mesh._geometry is None
        assert mesh._preloaded_geometry is None

        # Access geometry
        geom = mesh.geometry

        # Now it should be computed
        assert mesh._geometry is not None
        assert len(geom.elem_area) == 2

    def test_geometry_cached(self, two_triangle_mesh_data):
        """Test that geometry is cached after first access."""
        from fesomp.mesh import Mesh

        mesh = Mesh(**two_triangle_mesh_data)

        geom1 = mesh.geometry
        geom2 = mesh.geometry

        # Should be the same object
        assert geom1 is geom2
