# FESOMP

A modern Python library for working with FESOM2 (Finite Element Sea ice-Ocean Model) unstructured mesh data.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Mesh Management
- 🔄 **Flexible I/O**: Load meshes from NetCDF or ASCII formats
- ⚡ **Lazy Loading**: Topology, geometry, and spatial indices computed on-demand
- 📍 **Dual Coordinates**: Automatic handling of node and element (triangle center) data
- 🔍 **Spatial Queries**: Fast nearest-neighbor search, radius queries, and bounding box selection

### Data Interpolation
- 🎯 **Three Methods**: Nearest neighbor (fast), inverse distance weighting (smooth), and linear interpolation
- 🌐 **Regular Grid**: Interpolate unstructured data to regular lon/lat grids
- 💾 **Caching**: Reusable interpolators for efficient multi-variable processing
- 🌍 **Spherical Geometry**: Accurate great circle calculations on the sphere

### Visualization
- 🗺️ **Map Plots**: Cartopy integration with multiple projections (PlateCarree, Robinson, Polar, etc.)
- 📊 **Transect Plots**: Vertical cross-sections with automatic vertical coordinate detection
- 🎨 **Customizable**: Full control over colormaps, contours, labels, and styling
- 🔄 **Auto-Detection**: Automatically determines if data is on nodes/elements and levels/layers

## Installation

### From PyPI

```bash
pip install fesomp
```

### From source

```bash
git clone https://github.com/nkolduno/fesomp.git
cd fesomp
pip install -e .
```

### Dependencies

Core dependencies:
- `numpy >= 1.20`
- `scipy >= 1.7`
- `xarray >= 0.19`
- `pandas >= 1.3`
- `matplotlib >= 3.5`
- `cartopy >= 0.20`
- `netCDF4 >= 1.5`

Development dependencies:
```bash
pip install -e ".[dev]"  # Includes pytest, pytest-cov, hypothesis
```

## Quick Start

### Load a mesh

```python
import fesomp

# From NetCDF
mesh = fesomp.load_mesh("path/to/fesom.mesh.diag.nc")

# From ASCII directory
mesh = fesomp.load_mesh("path/to/mesh/directory/")

print(mesh)
# Mesh(n2d=126859, nelem=237843, nlev=47, lon=[-180.00, 180.00], lat=[-90.00, 90.00])
```

### Plot 2D data on a map

```python
import xarray as xr

# Load surface temperature
sst = xr.open_dataset("sst.fesom.1958.nc")['sst'][0, :].values

# Create a map
fig, axes, interp = fesomp.plot(
    sst,
    mesh.lon,
    mesh.lat,
    mapproj="robinson",
    title="Sea Surface Temperature",
    units="°C",
    cmap="RdYlBu_r",
)
```

### Create a vertical transect

```python
# Load 3D temperature data
temp_3d = xr.open_dataset("temp.fesom.1958.nc")['temp'][0, :, :].values

# Atlantic meridional transect
fig, ax, interp = fesomp.transect(
    temp_3d,  # shape: (nlev, n2d)
    mesh,
    start=(-30, -60),  # 30°W, 60°S
    end=(-30, 60),     # 30°W, 60°N
    title="Temperature along 30°W",
    units="°C",
    depth_limits=(0, 2000),  # Top 2000 meters
)
```

### Interpolate to regular grid

```python
# Quick interpolation
data_reg, lon_reg, lat_reg = fesomp.regrid(
    sst,
    mesh.lon,
    mesh.lat,
    res=(360, 180),
    method="idw",
)

# Reusable interpolator (faster for multiple variables)
interp = fesomp.RegridInterpolator(
    mesh.lon, mesh.lat,
    res=(360, 180),
    method="idw",
)

temp_reg, lon_reg, lat_reg = interp(sst)
salt_reg, _, _ = interp(salinity)  # Reuses pre-computed weights
```

## Advanced Features

### Automatic Detection

The library automatically detects:

**Horizontal location:**
```python
# Data on nodes (n2d points)
temp_nodes = temp_3d  # shape: (nlev, n2d)
fesomp.transect(temp_nodes, mesh, ...)  # Uses mesh.lon, mesh.lat

# Data on elements (nelem points)
u_velocity = u_3d  # shape: (nlev, nelem)
fesomp.transect(u_velocity, mesh, ...)  # Uses mesh.lon_elem, mesh.lat_elem
```

**Vertical coordinate:**
```python
# Data on levels (interfaces) - nlev points
w_velocity  # shape: (nlev, n2d)

# Data on layers (centers) - nlev-1 points
temperature  # shape: (nlev-1, n2d)

# Automatically uses mesh.depth_levels or mesh.depth_layers
```

### Interpolation Methods

```python
# Nearest neighbor - fastest, best for categorical data
fesomp.transect(..., method="nn")

# Inverse distance weighting - smooth, good balance (default)
fesomp.transect(..., method="idw", influence=80000)  # 80 km radius

# Linear - most accurate but slower
fesomp.transect(..., method="linear")
```

### Spatial Queries

```python
# Find nearest nodes
nearest_idx = mesh.find_nearest(lon=10.5, lat=54.3, k=5)

# Find nodes within radius
indices = mesh.find_in_radius(lon=0, lat=0, radius_km=100)

# Bounding box query
indices = mesh.subset_by_bbox(
    lon_min=-10, lon_max=10,
    lat_min=40, lat_max=60
)

# Access mesh topology
edges = mesh.topology.edges
neighbors = mesh.topology.face_neighbors

# Access mesh geometry
areas = mesh.geometry.elem_area
node_areas = mesh.geometry.node_area
```

## Examples

See the [examples/](examples/) directory for Jupyter notebooks:

- `mesh_tutorial.ipynb` - Mesh loading and exploration
- `plotting_tutorial.ipynb` - 2D map plotting
- `transect_plotting.ipynb` - Vertical transect visualization

## Documentation

### Project Structure

```
fesomp/
├── src/fesomp/
│   ├── mesh/           # Mesh handling
│   │   ├── mesh.py           # Core Mesh class
│   │   ├── topology.py       # Topology computation
│   │   ├── geometry.py       # Geometric calculations
│   │   ├── spatial.py        # Spatial indexing
│   │   └── readers/          # I/O for NetCDF and ASCII
│   └── plotting/       # Visualization
│       ├── plot.py           # 2D map plotting
│       ├── regrid.py         # Grid interpolation
│       └── transect.py       # Vertical transects
├── tests/              # Test suite (108 tests)
└── examples/           # Jupyter notebooks
```

### API Overview

**Mesh Operations:**
- `load_mesh(path)` - Load mesh from file or directory
- `mesh.find_nearest(lon, lat, k)` - Find k nearest nodes
- `mesh.find_in_radius(lon, lat, radius_km)` - Radius search
- `mesh.subset_by_bbox(...)` - Bounding box query
- `mesh.lon_elem`, `mesh.lat_elem` - Element center coordinates (lazy)
- `mesh.topology` - Edge and neighbor information (lazy)
- `mesh.geometry` - Areas and gradients (lazy)

**Interpolation:**
- `regrid(data, lon, lat, ...)` - Interpolate to regular grid
- `RegridInterpolator(lon, lat, ...)` - Reusable interpolator

**Visualization:**
- `plot(data, lon, lat, ...)` - 2D map with cartopy
- `transect(data, mesh, start, end, ...)` - Vertical cross-section
- `interpolate_transect(...)` - Interpolate along transect path
- `plot_transect(data, distance, depth, ...)` - Plot pre-interpolated transect

## Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=fesomp --cov-report=html

# Specific test file
pytest tests/unit/test_transect.py -v
```

Current test coverage: **108 passing tests**

## Performance Tips

1. **Reuse interpolators** when processing multiple variables:
   ```python
   interp = fesomp.RegridInterpolator(mesh.lon, mesh.lat)
   temp_grid = interp(temperature)[0]
   salt_grid = interp(salinity)[0]  # Much faster!
   ```

2. **Choose appropriate methods**:
   - Use `method="nn"` for fastest interpolation
   - Use `method="idw"` for smooth fields (default)
   - Use `method="linear"` for highest accuracy

3. **Adjust influence radius** for sparse data:
   ```python
   fesomp.transect(..., influence=150000)  # 150 km
   ```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Citation

If you use FESOMP in your research, please cite:

```bibtex
@software{fesomp2026,
  title = {FESOMP: A Python library for FESOM2 unstructured mesh data},
  author = {Koldunov, Nikolay},
  year = {2026},
  url = {https://github.com/yourusername/fesomp}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed with assistance from Claude (Anthropic)
- Built for the FESOM2 ocean modeling community
- Inspired by PyFVCOM and other unstructured grid tools

## Related Projects

- [FESOM2](https://github.com/FESOM/fesom2) - The Finite Element Sea ice-Ocean Model
- [PyFVCOM](https://github.com/pwcazenave/PyFVCOM) - Python tools for FVCOM data
- [xarray](https://github.com/pydata/xarray) - N-D labeled arrays and datasets in Python

---

**Questions or issues?** Please open an issue on GitHub or contact the maintainers.
