"""Pytest configuration and fixtures for mesh tests."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def single_triangle_mesh_data():
    """Fixture providing data for a single triangle mesh."""
    # Simple equilateral-ish triangle near equator
    lon = np.array([0.0, 1.0, 0.5], dtype=np.float64)
    lat = np.array([0.0, 0.0, 0.866], dtype=np.float64)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    # Minimal vertical structure
    nlev = 3
    depth_levels = np.array([0.0, -10.0, -50.0], dtype=np.float64)
    depth_layers = np.array([-5.0, -30.0], dtype=np.float64)
    node_levels = np.array([3, 3, 3], dtype=np.int32)
    elem_levels = np.array([3], dtype=np.int32)
    node_bottom_depth = np.array([-50.0, -50.0, -50.0], dtype=np.float64)
    elem_bottom_depth = np.array([-50.0], dtype=np.float64)

    return {
        "lon": lon,
        "lat": lat,
        "triangles": triangles,
        "nlev": nlev,
        "depth_levels": depth_levels,
        "depth_layers": depth_layers,
        "node_levels": node_levels,
        "elem_levels": elem_levels,
        "node_bottom_depth": node_bottom_depth,
        "elem_bottom_depth": elem_bottom_depth,
    }


@pytest.fixture
def two_triangle_mesh_data():
    """Fixture providing data for two adjacent triangles."""
    # Two triangles sharing an edge (nodes 1 and 2)
    #    3
    #   /|\
    #  / | \
    # 0--1--2
    lon = np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float64)
    lat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    triangles = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)

    nlev = 2
    depth_levels = np.array([0.0, -100.0], dtype=np.float64)
    depth_layers = np.array([-50.0], dtype=np.float64)
    node_levels = np.array([2, 2, 2, 2], dtype=np.int32)
    elem_levels = np.array([2, 2], dtype=np.int32)
    node_bottom_depth = np.array([-100.0, -100.0, -100.0, -100.0], dtype=np.float64)
    elem_bottom_depth = np.array([-100.0, -100.0], dtype=np.float64)

    return {
        "lon": lon,
        "lat": lat,
        "triangles": triangles,
        "nlev": nlev,
        "depth_levels": depth_levels,
        "depth_layers": depth_layers,
        "node_levels": node_levels,
        "elem_levels": elem_levels,
        "node_bottom_depth": node_bottom_depth,
        "elem_bottom_depth": elem_bottom_depth,
    }


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def pi_mesh_dir(test_data_dir):
    """Path to pi-mesh test data directory."""
    pi_dir = test_data_dir / "pi-mesh"
    required = pi_dir / "fesom.mesh.diag.nc"
    if not required.exists():
        pytest.skip(
            f"pi-mesh test data not available at {required}",
            allow_module_level=True,
        )
    return pi_dir
