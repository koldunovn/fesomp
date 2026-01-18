"""Unit tests for coordinate transformations between rotated and geographical coordinates."""

import numpy as np
import pytest

from fesomp.mesh.coordinates import (
    scalar_g2r,
    scalar_r2g,
    vec_rotate_g2r,
    vec_rotate_r2g,
)


def assert_lon_close(actual, desired, atol=1e-10):
    """Assert longitudes are close, accounting for -180/180 wrapping."""
    # Normalize difference to [-180, 180]
    diff = actual - desired
    diff = np.mod(diff + 180, 360) - 180
    np.testing.assert_allclose(diff, 0, atol=atol)


class TestScalarR2G:
    """Tests for scalar_r2g (rotated to geographical coordinate conversion)."""

    def test_identity_transformation(self):
        """Test that zero Euler angles give identity transformation."""
        rlon = np.array([10.0, 20.0, -30.0])
        rlat = np.array([45.0, -30.0, 60.0])

        lon, lat = scalar_r2g(0, 0, 0, rlon, rlat)

        np.testing.assert_allclose(lon, rlon, atol=1e-10)
        np.testing.assert_allclose(lat, rlat, atol=1e-10)

    def test_single_point(self):
        """Test conversion of a single point."""
        rlon = np.array([0.0])
        rlat = np.array([0.0])

        lon, lat = scalar_r2g(50, 15, -90, rlon, rlat)

        # Result should be valid coordinates
        assert -180 <= lon[0] <= 180
        assert -90 <= lat[0] <= 90

    def test_array_input(self):
        """Test that array inputs work correctly."""
        rlon = np.array([0.0, 10.0, 20.0, 30.0])
        rlat = np.array([0.0, 10.0, 20.0, 30.0])

        lon, lat = scalar_r2g(50, 15, -90, rlon, rlat)

        assert lon.shape == rlon.shape
        assert lat.shape == rlat.shape
        assert np.all(np.isfinite(lon))
        assert np.all(np.isfinite(lat))

    def test_pole_handling(self):
        """Test that poles are handled correctly."""
        rlon = np.array([0.0])
        rlat = np.array([90.0])  # North pole in rotated coords

        lon, lat = scalar_r2g(50, 15, -90, rlon, rlat)

        # Should return valid coordinates
        assert np.isfinite(lon[0])
        assert np.isfinite(lat[0])


class TestScalarG2R:
    """Tests for scalar_g2r (geographical to rotated coordinate conversion)."""

    def test_identity_transformation(self):
        """Test that zero Euler angles give identity transformation."""
        lon = np.array([10.0, 20.0, -30.0])
        lat = np.array([45.0, -30.0, 60.0])

        rlon, rlat = scalar_g2r(0, 0, 0, lon, lat)

        np.testing.assert_allclose(rlon, lon, atol=1e-10)
        np.testing.assert_allclose(rlat, lat, atol=1e-10)

    def test_single_point(self):
        """Test conversion of a single point."""
        lon = np.array([0.0])
        lat = np.array([0.0])

        rlon, rlat = scalar_g2r(50, 15, -90, lon, lat)

        # Result should be valid coordinates
        assert -180 <= rlon[0] <= 180
        assert -90 <= rlat[0] <= 90


class TestScalarRoundtrip:
    """Tests for roundtrip consistency of scalar coordinate conversions."""

    @pytest.mark.parametrize(
        "alpha,beta,gamma",
        [
            (0, 0, 0),
            (50, 15, -90),
            (180, 0, 0),
            (0, 90, 0),
            (-45, 30, 60),
        ],
    )
    def test_r2g_g2r_roundtrip(self, alpha, beta, gamma):
        """Test that r2g followed by g2r returns original coordinates."""
        rlon_orig = np.array([10.0, 20.0, -30.0, 0.0, 180.0])
        rlat_orig = np.array([45.0, -30.0, 60.0, 0.0, -45.0])

        lon, lat = scalar_r2g(alpha, beta, gamma, rlon_orig, rlat_orig)
        rlon_back, rlat_back = scalar_g2r(alpha, beta, gamma, lon, lat)

        assert_lon_close(rlon_back, rlon_orig)
        np.testing.assert_allclose(rlat_back, rlat_orig, atol=1e-10)

    @pytest.mark.parametrize(
        "alpha,beta,gamma",
        [
            (0, 0, 0),
            (50, 15, -90),
            (180, 0, 0),
            (0, 90, 0),
            (-45, 30, 60),
        ],
    )
    def test_g2r_r2g_roundtrip(self, alpha, beta, gamma):
        """Test that g2r followed by r2g returns original coordinates."""
        lon_orig = np.array([10.0, 20.0, -30.0, 0.0, 180.0])
        lat_orig = np.array([45.0, -30.0, 60.0, 0.0, -45.0])

        rlon, rlat = scalar_g2r(alpha, beta, gamma, lon_orig, lat_orig)
        lon_back, lat_back = scalar_r2g(alpha, beta, gamma, rlon, rlat)

        assert_lon_close(lon_back, lon_orig)
        np.testing.assert_allclose(lat_back, lat_orig, atol=1e-10)


class TestVecRotateR2G:
    """Tests for vec_rotate_r2g (vector rotation from rotated to geographical)."""

    def test_identity_transformation(self):
        """Test that zero Euler angles give identity transformation."""
        lon = np.array([10.0, 20.0])
        lat = np.array([45.0, -30.0])
        urot = np.array([1.0, 2.0])
        vrot = np.array([0.5, -0.5])

        u, v = vec_rotate_r2g(0, 0, 0, lon, lat, urot, vrot, flag=1)

        np.testing.assert_allclose(u, urot, atol=1e-10)
        np.testing.assert_allclose(v, vrot, atol=1e-10)

    def test_flag_0_rotated_coords(self):
        """Test with flag=0 (lon/lat in rotated coordinates)."""
        rlon = np.array([10.0, 20.0])
        rlat = np.array([45.0, -30.0])
        urot = np.array([1.0, 2.0])
        vrot = np.array([0.5, -0.5])

        u, v = vec_rotate_r2g(50, 15, -90, rlon, rlat, urot, vrot, flag=0)

        assert u.shape == urot.shape
        assert v.shape == vrot.shape
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))

    def test_flag_1_geographical_coords(self):
        """Test with flag=1 (lon/lat in geographical coordinates)."""
        lon = np.array([10.0, 20.0])
        lat = np.array([45.0, -30.0])
        urot = np.array([1.0, 2.0])
        vrot = np.array([0.5, -0.5])

        u, v = vec_rotate_r2g(50, 15, -90, lon, lat, urot, vrot, flag=1)

        assert u.shape == urot.shape
        assert v.shape == vrot.shape
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))


class TestVecRotateG2R:
    """Tests for vec_rotate_g2r (vector rotation from geographical to rotated)."""

    def test_identity_transformation(self):
        """Test that zero Euler angles give identity transformation."""
        lon = np.array([10.0, 20.0])
        lat = np.array([45.0, -30.0])
        ugeo = np.array([1.0, 2.0])
        vgeo = np.array([0.5, -0.5])

        u, v = vec_rotate_g2r(0, 0, 0, lon, lat, ugeo, vgeo, flag=1)

        np.testing.assert_allclose(u, ugeo, atol=1e-10)
        np.testing.assert_allclose(v, vgeo, atol=1e-10)

    def test_flag_0_rotated_coords(self):
        """Test with flag=0 (lon/lat in rotated coordinates)."""
        rlon = np.array([10.0, 20.0])
        rlat = np.array([45.0, -30.0])
        ugeo = np.array([1.0, 2.0])
        vgeo = np.array([0.5, -0.5])

        u, v = vec_rotate_g2r(50, 15, -90, rlon, rlat, ugeo, vgeo, flag=0)

        assert u.shape == ugeo.shape
        assert v.shape == vgeo.shape
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))


class TestVecRotateRoundtrip:
    """Tests for roundtrip consistency of vector rotations."""

    @pytest.mark.parametrize(
        "alpha,beta,gamma",
        [
            (0, 0, 0),
            (50, 15, -90),
            (180, 0, 0),
            (-45, 30, 60),
        ],
    )
    def test_r2g_g2r_roundtrip_flag0(self, alpha, beta, gamma):
        """Test vector roundtrip with flag=0 (rotated coordinates)."""
        rlon = np.array([10.0, 20.0, -30.0])
        rlat = np.array([45.0, -30.0, 60.0])
        urot = np.array([1.0, 2.0, -1.0])
        vrot = np.array([0.5, -0.5, 1.5])

        ugeo, vgeo = vec_rotate_r2g(alpha, beta, gamma, rlon, rlat, urot, vrot, flag=0)
        urot_back, vrot_back = vec_rotate_g2r(
            alpha, beta, gamma, rlon, rlat, ugeo, vgeo, flag=0
        )

        np.testing.assert_allclose(urot_back, urot, atol=1e-10)
        np.testing.assert_allclose(vrot_back, vrot, atol=1e-10)

    @pytest.mark.parametrize(
        "alpha,beta,gamma",
        [
            (0, 0, 0),
            (50, 15, -90),
            (180, 0, 0),
            (-45, 30, 60),
        ],
    )
    def test_g2r_r2g_roundtrip_flag1(self, alpha, beta, gamma):
        """Test vector roundtrip with flag=1 (geographical coordinates)."""
        lon = np.array([10.0, 20.0, -30.0])
        lat = np.array([45.0, -30.0, 60.0])
        ugeo = np.array([1.0, 2.0, -1.0])
        vgeo = np.array([0.5, -0.5, 1.5])

        urot, vrot = vec_rotate_g2r(alpha, beta, gamma, lon, lat, ugeo, vgeo, flag=1)
        ugeo_back, vgeo_back = vec_rotate_r2g(
            alpha, beta, gamma, lon, lat, urot, vrot, flag=1
        )

        np.testing.assert_allclose(ugeo_back, ugeo, atol=1e-10)
        np.testing.assert_allclose(vgeo_back, vgeo, atol=1e-10)


class TestVectorMagnitudePreservation:
    """Tests that vector rotation preserves magnitude."""

    @pytest.mark.parametrize(
        "alpha,beta,gamma",
        [
            (50, 15, -90),
            (180, 0, 0),
            (-45, 30, 60),
        ],
    )
    def test_magnitude_preserved_r2g(self, alpha, beta, gamma):
        """Test that vector magnitude is preserved during r2g rotation."""
        rlon = np.array([10.0, 20.0, -30.0])
        rlat = np.array([45.0, -30.0, 60.0])
        urot = np.array([1.0, 2.0, -1.0])
        vrot = np.array([0.5, -0.5, 1.5])

        mag_orig = np.sqrt(urot**2 + vrot**2)

        ugeo, vgeo = vec_rotate_r2g(alpha, beta, gamma, rlon, rlat, urot, vrot, flag=0)
        mag_rotated = np.sqrt(ugeo**2 + vgeo**2)

        np.testing.assert_allclose(mag_rotated, mag_orig, rtol=1e-10)

    @pytest.mark.parametrize(
        "alpha,beta,gamma",
        [
            (50, 15, -90),
            (180, 0, 0),
            (-45, 30, 60),
        ],
    )
    def test_magnitude_preserved_g2r(self, alpha, beta, gamma):
        """Test that vector magnitude is preserved during g2r rotation."""
        lon = np.array([10.0, 20.0, -30.0])
        lat = np.array([45.0, -30.0, 60.0])
        ugeo = np.array([1.0, 2.0, -1.0])
        vgeo = np.array([0.5, -0.5, 1.5])

        mag_orig = np.sqrt(ugeo**2 + vgeo**2)

        urot, vrot = vec_rotate_g2r(alpha, beta, gamma, lon, lat, ugeo, vgeo, flag=1)
        mag_rotated = np.sqrt(urot**2 + vrot**2)

        np.testing.assert_allclose(mag_rotated, mag_orig, rtol=1e-10)


class TestImportFromMesh:
    """Tests that functions are properly exported from fesomp.mesh."""

    def test_import_from_mesh_module(self):
        """Test importing functions from fesomp.mesh."""
        from fesomp.mesh import (
            scalar_g2r,
            scalar_r2g,
            vec_rotate_g2r,
            vec_rotate_r2g,
        )

        # Just verify they are callable
        assert callable(scalar_r2g)
        assert callable(scalar_g2r)
        assert callable(vec_rotate_r2g)
        assert callable(vec_rotate_g2r)
