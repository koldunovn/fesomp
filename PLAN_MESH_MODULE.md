# Mesh Module Detailed Plan

## Overview

The mesh module is the foundation of `fesomp`. It handles loading, representing, and querying FESOM2 unstructured mesh data.

## Core Design Principles

1. **Immutable mesh objects** - Once loaded, mesh doesn't change
2. **Lazy computation** - Derived properties (areas, neighbors) computed on first access (for example if we read from legacy ASCII files, otherwise trust what is provided in netCDF files.) and cached if needed.
3. **Clear separation** - I/O separate from mesh representation
4. **Type hints everywhere** - Full typing for IDE support and documentation
5. **Comprehensive testing** - Unit tests with synthetic data + integration tests with real pi-mesh

---

## Module Structure

```
fesomp/
├── __init__.py
├── mesh/
│   ├── __init__.py
│   ├── mesh.py          # Main Mesh class
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── base.py      # Abstract reader interface
│   │   ├── netcdf.py    # NetCDF reader (fesom.mesh.diag.nc)
│   │   └── ascii.py     # ASCII reader (nod2d.out, elem2d.out, etc.)
│   ├── topology.py      # Topology computations (neighbors, edges)
│   ├── geometry.py      # Geometric computations (areas, volumes)
│   └── spatial.py       # Spatial indexing (KDTree wrapper)
│
tests/
├── conftest.py          # Shared fixtures
├── data/
│   └── pi-mesh/         # Real pi-mesh files for integration tests
├── unit/
│   ├── test_mesh.py
│   ├── test_readers.py
│   ├── test_topology.py
│   ├── test_geometry.py
│   └── test_spatial.py
└── integration/
    └── test_mesh_loading.py
```

---

## Class Design

### `Mesh` - Main Class

```python
@dataclass(frozen=True)  # or attrs with frozen=True
class Mesh:
    """Immutable representation of a FESOM2 mesh."""

    # Core 2D data (required)
    lon: np.ndarray          # (n2d,) node longitudes
    lat: np.ndarray          # (n2d,) node latitudes
    triangles: np.ndarray    # (nelem, 3) element node indices

    # 3D structure (optional, None for 2D-only operations)
    nlev: int | None                    # number of levels
    depth_levels: np.ndarray | None     # (nlev,) depth of each level
    node_levels: np.ndarray | None      # (n2d,) number of levels at each node

    # Cached/lazy properties (computed on demand if needed, otherwise taken from fesom.mesh.diag.nc file)
    _topology: Topology | None = None
    _geometry: Geometry | None = None
    _spatial_index: SpatialIndex | None = None

    # Properties with lazy initialization
    @property
    def n2d(self) -> int:
        """Number of 2D nodes."""
        return len(self.lon)

    @property
    def nelem(self) -> int:
        """Number of triangular elements."""
        return len(self.triangles)

    @property
    def topology(self) -> Topology:
        """Mesh topology (neighbors, edges). Computed on first access if needed (if we load from ASCII files)."""
        ...

    @property
    def geometry(self) -> Geometry:
        """Mesh geometry (areas, volumes). Computed on first if needed (if we load from ASCII files)."""
        ...

    @property
    def spatial_index(self) -> SpatialIndex:
        """Spatial index for fast queries. Built on first access."""
        ...

    # Core methods
    def find_nearest_node(self, lon: float, lat: float) -> int: ...
    def find_containing_element(self, lon: float, lat: float) -> int | None: ...
    def subset_by_bbox(self, lon_min, lon_max, lat_min, lat_max) -> "Mesh": ...
    def subset_by_polygon(self, polygon: Polygon) -> "Mesh": ...
```

### `Topology` - Connectivity Information

```python
@dataclass
class Topology:
    """Mesh connectivity and topology."""

    node_neighbors: list[np.ndarray]   # neighbors of each node
    elem_neighbors: np.ndarray         # (nelem, 3) neighboring elements (-1 for boundary)
    edges: np.ndarray                  # (nedges, 2) unique edges
    boundary_nodes: np.ndarray         # indices of boundary nodes

    @classmethod
    def from_triangles(cls, triangles: np.ndarray) -> "Topology": ...
```

### `Geometry` - Geometric Properties

```python
@dataclass
class Geometry:
    """Geometric properties of the mesh."""

    node_areas: np.ndarray     # (n2d,) area associated with each node (cluster area)
    elem_areas: np.ndarray     # (nelem,) area of each triangle

    # 3D properties (only if mesh has 3D structure)
    node_volumes: np.ndarray | None   # (n3d,) volume of each 3D node
    layer_thicknesses: np.ndarray | None  # layer thicknesses

    @classmethod
    def compute(cls, mesh: Mesh) -> "Geometry": ...
```

### `SpatialIndex` - Fast Spatial Queries

```python
class SpatialIndex:
    """Spatial index for fast nearest-neighbor and region queries."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray): ...

    def find_nearest(self, lon: float, lat: float, k: int = 1) -> np.ndarray: ...
    def find_in_radius(self, lon: float, lat: float, radius_km: float) -> np.ndarray: ...
    def find_in_bbox(self, lon_min, lon_max, lat_min, lat_max) -> np.ndarray: ...
```

---

## Reader Interface

```python
# base.py
from abc import ABC, abstractmethod

class MeshReader(ABC):
    """Abstract base class for mesh readers."""

    @abstractmethod
    def read(self, path: Path | str) -> Mesh:
        """Read mesh from file(s) and return Mesh object."""
        ...

    @staticmethod
    def detect_format(path: Path | str) -> str:
        """Detect mesh format from path."""
        ...

# Factory function
def load_mesh(path: str | Path) -> Mesh:
    """Load mesh from file, auto-detecting format."""
    ...
```

---

## Testing Strategy

### 1. Unit Tests (with synthetic data)

**Synthetic test meshes created in fixtures:**
- Single triangle (3 nodes, 1 element)
- Two triangles sharing an edge (4 nodes, 2 elements)
- Small regular triangular grid (e.g., 5x5 = 25 nodes)
- Known geometry for area verification

**What to test:**
- Mesh creation with valid/invalid data
- Property calculations (n2d, nelem)
- Topology computation (neighbors, edges, boundary detection)
- Geometry computation (areas - compare with analytical values)
- Spatial index queries

### 2. Integration Tests (with pi-mesh)

**What to test:**
- NetCDF reader loads all fields correctly
- ASCII reader loads all fields correctly
- NetCDF and ASCII readers produce identical Mesh objects
- Real mesh has expected properties (n2d, nelem, depth range)
- Spatial queries return sensible results
- No NaN/Inf values in computed properties

### 3. Property-Based Tests (with hypothesis)

- Any valid triangles array produces valid topology
- Computed areas are always positive
- Number of boundary nodes < total nodes
- Spatial index always finds the query point's nearest neighbor

---

## Test Fixtures Example

```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def single_triangle():
    """Minimal mesh: one equilateral triangle."""
    return {
        "lon": np.array([0.0, 1.0, 0.5]),
        "lat": np.array([0.0, 0.0, np.sqrt(3)/2]),
        "triangles": np.array([[0, 1, 2]]),
    }

@pytest.fixture
def two_triangles():
    """Two triangles sharing an edge."""
    return {
        "lon": np.array([0.0, 1.0, 0.5, 0.5]),
        "lat": np.array([0.0, 0.0, np.sqrt(3)/2, -np.sqrt(3)/2]),
        "triangles": np.array([[0, 1, 2], [0, 3, 1]]),
    }

@pytest.fixture
def pi_mesh_path():
    """Path to real pi-mesh files."""
    return Path(__file__).parent / "data" / "pi-mesh"

@pytest.fixture
def pi_mesh_netcdf(pi_mesh_path):
    """Load pi-mesh from NetCDF."""
    from fesomp import load_mesh
    return load_mesh(pi_mesh_path / "fesom.mesh.diag.nc")
```

---

## Dependencies

**Required:**
- `numpy` - array operations
- `scipy` - KDTree for spatial indexing
- `xarray` - NetCDF reading (lazy loading)
- `pandas` - for fast ASCII data reading (important for large meshes)

**Optional:**
- `hypothesis` - property-based testing
- `shapely` - polygon operations (for subset_by_polygon)

---

## Implementation Order

1. **Phase 1: Core Mesh + NetCDF Reader**
   - [ ] Basic Mesh dataclass with core properties
   - [ ] NetCDF reader for fesom.mesh.diag.nc
   - [ ] Unit tests with synthetic data
   - [ ] Integration tests with pi-mesh NetCDF

2. **Phase 2: Topology & Geometry**
   - [ ] Topology computation (neighbors, edges, boundaries)
   - [ ] Geometry computation (areas, volumes)
   - [ ] Tests with known analytical values

3. **Phase 3: ASCII Reader**
   - [ ] ASCII reader for nod2d.out, elem2d.out, etc.
   - [ ] Tests comparing NetCDF vs ASCII loading

4. **Phase 4: Spatial Index & Queries**
   - [ ] SpatialIndex class with KDTree
   - [ ] find_nearest_node, find_containing_element
   - [ ] Region subset methods

---

## Questions for You

1. Should I also include `nod3d.out` parsing for full 3D node coordinates, or is level-based 3D structure sufficient?
 Answer: No we don't need it. We only work with FESOM2 data, that do not use nod3d.out

2. For element areas, should we use:
   - Flat Cartesian approximation (fast, less accurate for large elements)
   - Spherical excess formula (accurate, slower)
   - Both with an option to choose?
Answer: we should use Spherical, and when possible computations should be compared to results of data contained in the netCDF file, that will be the source of truth for computations base on ASCII files. If parameter exists already in netCDF file, don't recompute it.

3. What precision: float32 (memory efficient) or float64 (more precise) for coordinates?
Answer: For some computation precision of the coordinates is important, so keep float64