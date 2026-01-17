"""Unit tests for the SpatialIndex class."""

import numpy as np
import pytest

from fesomp.mesh.spatial import (
    EARTH_RADIUS_KM,
    SpatialIndex,
    arc_to_chord_distance,
    chord_to_arc_distance,
    lonlat_to_cartesian,
)


class TestCoordinateConversion:
    """Tests for coordinate conversion utilities."""

    def test_lonlat_to_cartesian_equator(self):
        """Test conversion at equator."""
        # Point at (0, 0) should be (1, 0, 0)
        coords = lonlat_to_cartesian(0.0, 0.0)
        np.testing.assert_allclose(coords, [1.0, 0.0, 0.0], atol=1e-10)

    def test_lonlat_to_cartesian_north_pole(self):
        """Test conversion at North Pole."""
        coords = lonlat_to_cartesian(0.0, 90.0)
        np.testing.assert_allclose(coords, [0.0, 0.0, 1.0], atol=1e-10)

    def test_lonlat_to_cartesian_array(self):
        """Test conversion with arrays."""
        lon = np.array([0.0, 90.0, 0.0])
        lat = np.array([0.0, 0.0, 90.0])

        coords = lonlat_to_cartesian(lon, lat)

        assert coords.shape == (3, 3)
        np.testing.assert_allclose(coords[0], [1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(coords[1], [0.0, 1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(coords[2], [0.0, 0.0, 1.0], atol=1e-10)

    def test_cartesian_unit_length(self):
        """Test that converted points lie on unit sphere."""
        lon = np.array([45.0, -120.0, 180.0])
        lat = np.array([30.0, -45.0, 60.0])

        coords = lonlat_to_cartesian(lon, lat)

        # All points should have unit length
        lengths = np.linalg.norm(coords, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-10)


class TestDistanceConversion:
    """Tests for chord/arc distance conversion."""

    def test_chord_to_arc_small_distance(self):
        """Test chord to arc for small distances."""
        # For small distances, chord ≈ arc
        chord = 0.01  # Small chord on unit sphere
        arc = chord_to_arc_distance(chord, radius=1.0)

        # Should be approximately equal for small distances
        np.testing.assert_allclose(arc, chord, rtol=0.01)

    def test_arc_to_chord_roundtrip(self):
        """Test arc to chord and back."""
        original_arc = 1000.0  # 1000 km

        chord = arc_to_chord_distance(original_arc)
        recovered_arc = chord_to_arc_distance(chord)

        np.testing.assert_allclose(recovered_arc, original_arc, rtol=1e-10)

    def test_quarter_earth_arc(self):
        """Test conversion for quarter Earth circumference."""
        # Quarter Earth = 90 degrees = pi*R/2
        quarter_arc = np.pi * EARTH_RADIUS_KM / 2

        chord = arc_to_chord_distance(quarter_arc)

        # Chord for 90 degrees = sqrt(2) on unit sphere
        expected_chord = np.sqrt(2)
        np.testing.assert_allclose(chord, expected_chord, rtol=1e-10)


class TestSpatialIndex:
    """Tests for SpatialIndex class."""

    def test_create_index(self):
        """Test creating a spatial index."""
        lon = np.array([0.0, 1.0, 2.0])
        lat = np.array([0.0, 1.0, 2.0])

        index = SpatialIndex(lon, lat)

        assert index.lon is lon
        assert index.lat is lat

    def test_find_nearest_single(self):
        """Test finding single nearest neighbor."""
        lon = np.array([0.0, 10.0, 20.0])
        lat = np.array([0.0, 0.0, 0.0])

        index = SpatialIndex(lon, lat)

        # Query near first point
        nearest = index.find_nearest(0.1, 0.1, k=1)

        assert nearest == 0

    def test_find_nearest_multiple(self):
        """Test finding multiple nearest neighbors."""
        lon = np.array([0.0, 10.0, 20.0, 30.0])
        lat = np.array([0.0, 0.0, 0.0, 0.0])

        index = SpatialIndex(lon, lat)

        # Query near first point, get 2 nearest
        nearest = index.find_nearest(0.1, 0.1, k=2)

        assert len(nearest) == 2
        assert 0 in nearest  # First point should be nearest

    def test_find_in_radius(self):
        """Test finding points within radius."""
        lon = np.array([0.0, 1.0, 10.0, 20.0])
        lat = np.array([0.0, 0.0, 0.0, 0.0])

        index = SpatialIndex(lon, lat)

        # Query with 200 km radius (about 2 degrees at equator)
        in_radius = index.find_in_radius(0.5, 0.0, radius_km=200.0)

        # Should find points 0 and 1
        assert len(in_radius) >= 2
        assert 0 in in_radius
        assert 1 in in_radius

    def test_find_in_radius_none(self):
        """Test finding no points within small radius."""
        lon = np.array([0.0, 10.0, 20.0])
        lat = np.array([0.0, 0.0, 0.0])

        index = SpatialIndex(lon, lat)

        # Query far from any point with tiny radius
        in_radius = index.find_in_radius(5.0, 5.0, radius_km=1.0)

        assert len(in_radius) == 0


class TestSpatialIndexWithMesh:
    """Tests for spatial index integrated with Mesh."""

    def test_mesh_spatial_index_lazy(self, two_triangle_mesh_data):
        """Test that spatial index is created lazily."""
        from fesomp.mesh import Mesh

        mesh = Mesh(**two_triangle_mesh_data)

        # Should be None before access
        assert mesh._spatial_index is None

        # Access through property
        index = mesh.spatial_index

        # Now should exist
        assert mesh._spatial_index is not None
        assert isinstance(index, SpatialIndex)

    def test_mesh_find_nearest(self, two_triangle_mesh_data):
        """Test finding nearest node through mesh."""
        from fesomp.mesh import Mesh

        mesh = Mesh(**two_triangle_mesh_data)

        # Find nearest to (0, 0) which is node 0
        nearest = mesh.find_nearest(0.0, 0.0, k=1)

        assert nearest == 0

    def test_mesh_find_in_radius(self, two_triangle_mesh_data):
        """Test finding nodes in radius through mesh."""
        from fesomp.mesh import Mesh

        mesh = Mesh(**two_triangle_mesh_data)

        # Find nodes within large radius (should find all)
        in_radius = mesh.find_in_radius(1.0, 0.5, radius_km=500.0)

        assert len(in_radius) == 4  # All 4 nodes
