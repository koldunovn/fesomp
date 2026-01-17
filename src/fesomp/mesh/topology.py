"""Mesh topology data structures and computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class Topology:
    """
    Mesh topology information (edges, connectivity, neighbors).

    Attributes
    ----------
    edges : np.ndarray
        Edge node pairs, shape (nedges, 2), 0-indexed.
        Each row contains [node_i, node_j] where node_i < node_j.
    face_edges : np.ndarray
        Edge indices for each face, shape (nelem, 3).
        face_edges[i, j] is the edge index opposite to vertex j of face i.
    face_neighbors : np.ndarray
        Neighbor face indices, shape (nelem, 3).
        face_neighbors[i, j] is the face adjacent to face i across edge j.
        -1 indicates a boundary edge with no neighbor.
    edge_faces : np.ndarray
        Face indices for each edge, shape (nedges, 2).
        edge_faces[e, 0] and edge_faces[e, 1] are the faces sharing edge e.
        -1 indicates a boundary edge with only one adjacent face.
    node_elements : list[np.ndarray]
        For each node, the list of element indices containing that node.
    """

    edges: np.ndarray
    face_edges: np.ndarray
    face_neighbors: np.ndarray
    edge_faces: np.ndarray
    node_elements: list[np.ndarray]

    @property
    def nedges(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def get_boundary_edges(self) -> np.ndarray:
        """
        Get indices of boundary edges.

        Returns
        -------
        np.ndarray
            Indices of edges that have only one adjacent face.
        """
        return np.nonzero(self.edge_faces[:, 1] == -1)[0]

    def get_boundary_nodes(self) -> np.ndarray:
        """
        Get indices of boundary nodes.

        Returns
        -------
        np.ndarray
            Sorted unique indices of nodes on the boundary.
        """
        boundary_edges = self.get_boundary_edges()
        boundary_nodes = self.edges[boundary_edges].ravel()
        return np.unique(boundary_nodes)


def compute_topology(triangles: np.ndarray) -> Topology:
    """
    Compute mesh topology from triangle connectivity.

    Parameters
    ----------
    triangles : np.ndarray
        Triangle connectivity, shape (nelem, 3), 0-indexed.

    Returns
    -------
    Topology
        Computed topology object.
    """
    nelem = len(triangles)

    # Build node-to-elements mapping
    n2d = triangles.max() + 1
    node_elem_lists: list[list[int]] = [[] for _ in range(n2d)]
    for elem_idx, tri in enumerate(triangles):
        for node in tri:
            node_elem_lists[node].append(elem_idx)
    node_elements = [np.array(lst, dtype=np.int32) for lst in node_elem_lists]

    # Build edge dictionary: (min_node, max_node) -> edge_index
    # and track which faces use each edge
    edge_dict: dict[tuple[int, int], int] = {}
    edge_to_faces: dict[int, list[int]] = {}
    edges_list: list[tuple[int, int]] = []

    # face_edges[i, j] = edge index opposite to vertex j
    face_edges = np.zeros((nelem, 3), dtype=np.int32)

    # Edge opposite to vertex j connects vertices (j+1)%3 and (j+2)%3
    for face_idx, tri in enumerate(triangles):
        for j in range(3):
            # Edge opposite to vertex j
            n1, n2 = tri[(j + 1) % 3], tri[(j + 2) % 3]
            edge_key = (min(n1, n2), max(n1, n2))

            if edge_key not in edge_dict:
                edge_idx = len(edges_list)
                edge_dict[edge_key] = edge_idx
                edges_list.append(edge_key)
                edge_to_faces[edge_idx] = []
            else:
                edge_idx = edge_dict[edge_key]

            face_edges[face_idx, j] = edge_idx
            edge_to_faces[edge_idx].append(face_idx)

    nedges = len(edges_list)
    edges = np.array(edges_list, dtype=np.int32)

    # Build edge_faces array
    edge_faces = np.full((nedges, 2), -1, dtype=np.int32)
    for edge_idx, faces in edge_to_faces.items():
        edge_faces[edge_idx, 0] = faces[0]
        if len(faces) > 1:
            edge_faces[edge_idx, 1] = faces[1]

    # Build face_neighbors from edge_faces
    face_neighbors = np.full((nelem, 3), -1, dtype=np.int32)
    for face_idx in range(nelem):
        for j in range(3):
            edge_idx = face_edges[face_idx, j]
            f1, f2 = edge_faces[edge_idx]
            if f1 == face_idx:
                face_neighbors[face_idx, j] = f2
            else:
                face_neighbors[face_idx, j] = f1

    return Topology(
        edges=edges,
        face_edges=face_edges,
        face_neighbors=face_neighbors,
        edge_faces=edge_faces,
        node_elements=node_elements,
    )
