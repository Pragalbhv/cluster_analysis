# Voronoi-Weighted Coordination Number Analysis

This module implements various coordination number calculations based on Voronoi tessellation geometries.

## Implemented Methods

### 1. Topological Coordination Number (CN_faces)
Simple count of Voronoi faces, where each face corresponds to one nearest neighbor:
```
CN_faces = F
```
where F is the number of faces of the atom's Voronoi cell.

### 2. Face-Area Weighted Coordination Number (CN_A)
Weighted by the areas of Voronoi faces:
```
CN_A = (Σ A_i)² / Σ A_i²
```
where A_i is the area of the Voronoi face shared with neighbor i.

### 3. Solid-Angle Weighted Coordination Number (CN_Ω)
Weighted by solid angles subtended by Voronoi faces:
```
CN_Ω = (Σ Ω_i)² / Σ Ω_i²
```
where Ω_i = A_i / r_i² is the solid angle subtended by face i.

### 4. Solid-Angle Thresholding (O'Keeffe/VoronoiNN)
Keeps neighbors based on a threshold τ of the maximum solid angle:
```
w_i = Ω_i
CN = Σ Θ(w_i - τ * max_j w_j)
```
where Θ is the Heaviside step function and τ ≈ 0.5 is typical.

### 5. CrystalNN Weighted Scheme
Combines Voronoi solid-angle and face-area weighting:
```
w_i^Vor = Ω_i² / A_i
```
This provides a more sophisticated weighting scheme that can be further refined with distance and chemical considerations.

## Usage

### Basic Usage

```python
from voronoi_ovito_utils import (
    build_voronoi_graph_from_pipeline,
    analyze_voronoi_coordination_all_methods,
    compute_coordination_statistics,
)

# Build Voronoi graph
graph = build_voronoi_graph_from_pipeline(pipeline, frame=0, min_area=0.0)

# Extract positions
阮sitions = np.array([graph.nodes[idx]['position'] for idx in graph.nodes()])

# Compute all coordination numbers
results = analyze_voronoi_coordination_all_methods(
    graph,
    positions=positions,
    tau=0.5  # threshold for solid-angle method
)

# Access results
cn_faces = results['CN_faces']
cn_area = results['CN_A']
cn_omega = results['CN_Ω']
cn_threshold = results['CN_threshold']
crystal_weights = results['CrystalNN_weights']
```

### Individual Method Usage

```python
from voronoi_ovito_utils import (
    compute_topological_coordination_number,
    compute_face_area_weighted_cn,
    compute_solid_angle_weighted_cn,
    compute_solid_angle_threshold_cn,
    compute_crystal_nn_weight,
)

# Topological CN
cn_top = compute_topological_coordination_number(graph)

# Face-area weighted CN
cn_a = compute_face_area_weighted_cn(graph)

# Solid-angle weighted CN
cn_omega = compute_solid_angle_weighted_cn(graph, positions=positions)

# Threshold-based CN
cn_thresh = compute_solid_angle_threshold_cn(graph, positions=positions, tau=0.5)

# CrystalNN weights
crystal_weights = compute_crystal_nn_weight(graph, positions=positions)
```

### Statistical Analysis

```python
from voronoi_ovito_utils import compute_coordination_statistics

# Get species information
species = {idx: graph.nodes[idx]['species'] for idx in graph.nodes()}

# Compute statistics
stats = compute_coordination_statistics(cn_faces, species=species)

print(f"Mean: {stats['mean']:.2f}")
print(f"Std: {stats['std']:.2f}")
print(f"By species: {stats['by_species']}")
```

## Output Format

All coordination number functions return dictionaries mapping atom indices to values:
- `CN_faces`: `{atom_idx: int}` - integer coordination numbers
- `CN_A`: `{atom_idx: float}` - continuous coordination numbers
- `CN_Ω`: `{atom_idx: float}` - continuous coordination numbers
- `CN_threshold`: `{atom_idx: int}` - integer coordination numbers
- `CrystalNN_weights`: `{atom_idx: {neighbor_idx: weight}}` - nested dictionary of weights

## Files

- `voronoi_coordination.py`: Core implementation of all coordination analysis functions
- `voronoi_ovito_utils.py`: Integration and wrapper functions (updated to export coordination functions)
- `vornoi_ovito_cannon.ipynb`: Example usage in Jupyter notebook (cells 9-13 added)

## Mathematical Background

The face-area and solid-angle weighted coordination numbers use an effective coordination number formula that is equivalent to the inverse participation ratio:
```
CN_eff = 1 / Σ p_i²
```
where p_i are normalized weights (either by area or solid angle).

This provides a continuous measure that accounts for the distribution of neighbor strengths, making it useful for analyzing disordered systems where traditional integer coordination numbers may be ambiguous.
