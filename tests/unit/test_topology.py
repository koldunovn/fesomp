"""Unit tests for the Topology class and computation."""

import numpy as np
import pytest

from fesomp.mesh.topology import Topology, compute_topology


class TestTopologyComputation:
    """Tests for topology computation from triangles."""

    def test_single_triangle_edges(self):
        """Test edge computation for single triangle."""
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        topo = compute_topology(triangles)

        # Single triangle has 3 edges
        assert topo.nedges == 3

        # Check edges are sorted (min, max)
        for edge in topo.edges:
            assert edge[0] < edge[1]

    def test_single_triangle_face_edges(self):
        """Test face_edges for single triangle."""
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        topo = compute_topology(triangles)

        # face_edges[0] should have 3 edge indices
        assert topo.face_edges.shape == (1, 3)

        # All edge indices should be valid
        assert all(0 <= e < topo.nedges for e in topo.face_edges[0])

    def test_single_triangle_no_neighbors(self):
        """Test that single triangle has no neighbors."""
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        topo = compute_topology(triangles)

        # All face_neighbors should be -1 (boundary)
        assert np.all(topo.face_neighbors == -1)

    def test_two_triangles_shared_edge(self):
        """Test topology for two triangles sharing an edge."""
        # Triangles sharing edge (1, 3)
        triangles = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
        topo = compute_topology(triangles)

        # Two triangles, 3 edges each, but 1 shared = 5 edges total
        assert topo.nedges == 5

        # Find the shared edge (nodes 1 and 3)
        shared_edge_idx = None
        for i, edge in enumerate(topo.edges):
            if set(edge) == {1, 3}:
                shared_edge_idx = i
                break

        assert shared_edge_idx is not None

        # Both faces should reference this edge
        shared_edge_faces = topo.edge_faces[shared_edge_idx]
        assert set(shared_edge_faces) == {0, 1}

    def test_two_triangles_face_neighbors(self):
        """Test face neighbors for two adjacent triangles."""
        triangles = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
        topo = compute_topology(triangles)

        # Each triangle should have exactly one neighbor (the other)
        # and two boundary edges (-1)
        neighbors_0 = topo.face_neighbors[0]
        neighbors_1 = topo.face_neighbors[1]

        assert 1 in neighbors_0  # Triangle 0 neighbors triangle 1
        assert 0 in neighbors_1  # Triangle 1 neighbors triangle 0

        # Each should have exactly 2 boundary edges
        assert np.sum(neighbors_0 == -1) == 2
        assert np.sum(neighbors_1 == -1) == 2

    def test_node_elements_mapping(self):
        """Test node-to-elements mapping."""
        triangles = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
        topo = compute_topology(triangles)

        # Node 0: only in triangle 0
        assert list(topo.node_elements[0]) == [0]

        # Node 1: in both triangles
        assert set(topo.node_elements[1]) == {0, 1}

        # Node 2: only in triangle 1
        assert list(topo.node_elements[2]) == [1]

        # Node 3: in both triangles
        assert set(topo.node_elements[3]) == {0, 1}


class TestTopologyProperties:
    """Tests for Topology class properties and methods."""

    def test_boundary_edges_single_triangle(self):
        """Test boundary edge detection for single triangle."""
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        topo = compute_topology(triangles)

        boundary_edges = topo.get_boundary_edges()

        # All 3 edges are boundary edges
        assert len(boundary_edges) == 3

    def test_boundary_edges_two_triangles(self):
        """Test boundary edge detection for two triangles."""
        triangles = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
        topo = compute_topology(triangles)

        boundary_edges = topo.get_boundary_edges()

        # 5 edges total, 1 interior = 4 boundary
        assert len(boundary_edges) == 4

    def test_boundary_nodes_single_triangle(self):
        """Test boundary node detection for single triangle."""
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        topo = compute_topology(triangles)

        boundary_nodes = topo.get_boundary_nodes()

        # All 3 nodes are boundary nodes
        assert len(boundary_nodes) == 3
        assert set(boundary_nodes) == {0, 1, 2}

    def test_boundary_nodes_two_triangles(self):
        """Test boundary node detection for two triangles."""
        triangles = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
        topo = compute_topology(triangles)

        boundary_nodes = topo.get_boundary_nodes()

        # All 4 nodes are on the boundary
        assert len(boundary_nodes) == 4
        assert set(boundary_nodes) == {0, 1, 2, 3}


class TestLazyTopologyLoading:
    """Tests for lazy topology loading in Mesh."""

    def test_topology_computed_on_access(self, two_triangle_mesh_data):
        """Test that topology is computed lazily."""
        from fesomp.mesh import Mesh

        mesh = Mesh(**two_triangle_mesh_data)

        # Internal topology should be None before access
        assert mesh._topology is None
        assert mesh._preloaded_topology is None

        # Access topology
        topo = mesh.topology

        # Now it should be computed
        assert mesh._topology is not None
        assert topo.nedges == 5

    def test_topology_cached(self, two_triangle_mesh_data):
        """Test that topology is cached after first access."""
        from fesomp.mesh import Mesh

        mesh = Mesh(**two_triangle_mesh_data)

        topo1 = mesh.topology
        topo2 = mesh.topology

        # Should be the same object
        assert topo1 is topo2
