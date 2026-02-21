# Cluster Analysis Package Reference Documentation

## Table of Contents

1. [Package Overview](#package-overview)
2. [Installation and Dependencies](#installation-and-dependencies)
3. [Architecture](#architecture)
4. [Module Reference](#module-reference)
   - [Bond-Based Analysis](#bond-based-analysis)
   - [Voronoi-Based Analysis](#voronoi-based-analysis)
   - [Utilities](#utilities)
5. [Common Workflows](#common-workflows)
6. [Function Index](#function-index)

---

## Package Overview

The `cluster_analysis` package provides comprehensive tools for analyzing atomic clusters in NaCl-PuCl3 systems using two primary approaches:

1. **Bond-Based Analysis**: Uses radial distribution functions (RDFs) to determine bond cutoffs and constructs shared-anion connectivity graphs
2. **Voronoi-Based Analysis**: Uses Voronoi tessellation to define neighbor relationships and analyze coordination environments

The package is designed to work with OVITO data pipelines and ASE Atoms objects, providing flexible interfaces for cluster detection, coordination analysis, and visualization.

### Key Capabilities

- **RDF Computation**: Compute partial radial distribution functions for all species pairs
- **Cutoff Detection**: Automatic detection of first-minimum cutoffs from RDFs
- **Bond Configuration**: Configure OVITO bond modifiers with pairwise cutoffs
- **Graph Construction**: Build connectivity graphs using shared-anion (Cl) or Voronoi-based methods
- **Cluster Analysis**: Identify and analyze clusters of metal atoms
- **Coordination Analysis**: Multiple methods for computing coordination numbers
- **Visualization**: Comprehensive plotting utilities for clusters, coordination, and graphs

---

## Installation and Dependencies

### Required Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing (signal processing, filtering)
- **networkx**: Graph theory and network analysis
- **matplotlib**: Plotting and visualization
- **ovito**: OVITO Python API (for bond-based analysis)
- **freud**: Voronoi tessellation (for freud-based Voronoi analysis)
- **ASE**: Atomic Simulation Environment (optional, for Atoms objects)

### Installation

The package is typically used within a research environment. Ensure all dependencies are installed:

```bash
pip install numpy scipy networkx matplotlib ovito freud ase
```

---

## Architecture

### Module Organization

```
cluster_analysis/
├── bond_based/
│   ├── bondmodifier_utils.py    # RDF computation, cutoff detection, bond configuration
│   └── shell_utils.py            # Shell analysis, coordination numbers, neighbor distributions
├── voronoi/
│   ├── frued/
│   │   └── voronoi_utils.py     # Freud-based Voronoi tessellation
│   ├── ovito/
│   │   └── voronoi_ovito_utils.py  # OVITO-based Voronoi tessellation
│   └── voronoi_coordination.py  # Weighted coordination number methods
└── utils/
    └── plots.py                  # Visualization utilities
```

### Design Principles

1. **Separation of Concerns**: Analysis logic separated from visualization
2. **Multiple Backends**: Support for both freud and OVITO Voronoi implementations
3. **Flexible Interfaces**: Functions accept OVITO pipelines, ASE Atoms, or raw arrays
4. **Graph-Based**: Uses NetworkX graphs for cluster and connectivity analysis
5. **Composable**: Functions can be combined into custom workflows

### Data Flow

```
Simulation Data (OVITO Pipeline / ASE Atoms)
    ↓
RDF Computation / Voronoi Tessellation
    ↓
Cutoff Detection / Graph Construction
    ↓
Cluster Identification
    ↓
Analysis & Visualization
```

---

## Module Reference

### Bond-Based Analysis

#### Module: `bond_based.bondmodifier_utils`

Utilities for RDF-based bond cutoff detection, OVITO bond creation, and graph/cluster analysis built around shared-anion connectivity.

#### `compute_partial_rdfs`

```python
compute_partial_rdfs(
    pipeline: Any,
    nsamples: int = 100,
    cutoff: float = 8.0,
    bins: int = 200
) -> Dict[str, np.ndarray]
```

Compute partial RDFs over the last `nsamples` frames.

**Parameters:**
- `pipeline` (Any): OVITO pipeline object containing simulation data
- `nsamples` (int, optional): Number of frames to sample. Defaults to 100.
- `cutoff` (float, optional): Maximum distance for RDF computation. Defaults to 8.0 Å.
- `bins` (int, optional): Number of bins for RDF histogram. Defaults to 200.

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary with keys like 'Cl-Na', 'Cl-Pu', etc., plus key 'r' containing the radial distance array.

**Example:**
```python
from ovito.io import import_file
from bond_based.bondmodifier_utils import compute_partial_rdfs

pipeline = import_file("trajectory.xyz")
rdf = compute_partial_rdfs(pipeline, nsamples=50, cutoff=10.0)

# Access RDF for specific pairs
r = rdf['r']
g_cl_na = rdf['Cl-Na']
g_cl_pu = rdf['Cl-Pu']
```

**Notes:**
- Automatically clears pipeline modifiers after computation
- Normalizes pair names alphabetically (e.g., 'Cl-Na' rather than 'Na-Cl')
- Averages RDFs over all sampled frames

---

#### `find_first_minimum`

```python
find_first_minimum(
    r: np.ndarray,
    g: np.ndarray,
    min_distance: float = 1.0,
    max_distance: float = 8.0,
    prominence: float = 0.1,
    smooth_window: int = 11,
    poly_order: int = 2,
    smoothing_method: str = 'gaussian',
    gaussian_sigma: float = 3.0
) -> Optional[float]
```

Find first minimum using scipy's find_peaks with prominence, with fallback to custom implementation.

**Parameters:**
- `r` (np.ndarray): Distance array
- `g` (np.ndarray): RDF values
- `min_distance` (float, optional): Minimum distance to consider. Defaults to 1.0 Å.
- `max_distance` (float, optional): Maximum distance to consider. Defaults to 8.0 Å.
- `prominence` (float, optional): Minimum prominence for peaks (scipy method only). Defaults to 0.1.
- `smooth_window` (int, optional): Window size for smoothing (must be odd for Savitzky-Golay). Defaults to 11.
- `poly_order` (int, optional): Polynomial order for Savitzky-Golay filter. Defaults to 2.
- `smoothing_method` (str, optional): Method to use ('moving_avg', 'savgol', 'gaussian'). Defaults to 'gaussian'.
- `gaussian_sigma` (float, optional): Standard deviation for Gaussian smoothing. Defaults to 3.0.

**Returns:**
- `Optional[float]`: Distance of first minimum, or None if not found.

**Algorithm:**
1. Restricts RDF to specified window
2. Applies smoothing based on selected method
3. Inverts g(r) to find minima as peaks
4. Uses scipy's find_peaks with prominence filtering
5. Refines minimum location using quadratic interpolation

**Example:**
```python
from bond_based.bondmodifier_utils import find_first_minimum

r = rdf['r']
g_cl_pu = rdf['Cl-Pu']
cutoff = find_first_minimum(r, g_cl_pu, min_distance=2.0, max_distance=6.0)
print(f"Pu-Cl cutoff: {cutoff:.2f} Å")
```

---

#### `find_first_shell_minimum`

```python
find_first_shell_minimum(
    r: np.ndarray,
    g: np.ndarray,
    min_distance: float,
    max_distance: float,
    smooth_window: int = 5,
    baseline_level: float = 1.0,
    min_peak_height_above_baseline: float = 0.1
) -> Optional[float]
```

Select the first-shell cutoff as the first local minimum after the first peak.

**Parameters:**
- `r` (np.ndarray): Distance array
- `g` (np.ndarray): RDF values
- `min_distance` (float): Minimum distance to consider
- `max_distance` (float): Maximum distance to consider
- `smooth_window` (int, optional): Window size for smoothing. Defaults to 5.
- `baseline_level` (float, optional): Baseline RDF level (~1 in liquids/solid tails). Defaults to 1.0.
- `min_peak_height_above_baseline` (float, optional): Minimum peak height above baseline. Defaults to 0.1.

**Returns:**
- `Optional[float]`: Distance of first minimum after first peak, or None if not found.

**Algorithm:**
1. Smooths g(r) slightly to suppress noise
2. Identifies first local maximum above baseline
3. Searches forward for first local minimum
4. Refines minimum location using quadratic interpolation

**Example:**
```python
from bond_based.bondmodifier_utils import find_first_shell_minimum

cutoff = find_first_shell_minimum(
    r, g_cl_na,
    min_distance=2.0,
    max_distance=5.0,
    baseline_level=1.0
)
```

---

#### `determine_cutoffs_from_rdf`

```python
determine_cutoffs_from_rdf(
    rdf: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    window: Tuple[float, float] = (2.0, 8.0),
    fallback: float = 3.10,
    pair_windows: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None
) -> Dict[Tuple[str, str], float]
```

For each (A,B) pair, determine a symmetric cutoff using first minimum.

**Parameters:**
- `rdf` (Dict[str, np.ndarray]): RDF dictionary with 'r' key and pair keys
- `pairs` (List[Tuple[str, str]]): List of species pairs to analyze
- `window` (Tuple[float, float], optional): Default search window (min, max). Defaults to (2.0, 8.0).
- `fallback` (float, optional): Fallback cutoff if RDF not found. Defaults to 3.10 Å.
- `pair_windows` (Optional[Dict], optional): Pair-specific search windows. Defaults to None.

**Returns:**
- `Dict[Tuple[str, str], float]`: Mapping of (A,B) pairs to cutoff distances.

**Example:**
```python
from bond_based.bondmodifier_utils import determine_cutoffs_from_rdf

pairs = [("Pu", "Cl"), ("Na", "Cl"), ("Pu", "Pu")]
cutoffs = determine_cutoffs_from_rdf(
    rdf,
    pairs=pairs,
    window=(2.0, 6.0),
    fallback=3.5
)

# Access cutoffs
pu_cl_cutoff = cutoffs[("Pu", "Cl")]
```

---

#### `configure_bonds_modifier_from_cutoffs`

```python
configure_bonds_modifier_from_cutoffs(
    pipeline: Any,
    pair_cutoffs: Dict[Tuple[str, str], float],
    disable_pairs: Optional[List[Tuple[str, str]]] = None
) -> None
```

Clear pipeline modifiers and append a CreateBondsModifier with specified pairwise cutoffs.

**Parameters:**
- `pipeline` (Any): OVITO pipeline object
- `pair_cutoffs` (Dict[Tuple[str, str], float]): Dictionary mapping species pairs to cutoff distances
- `disable_pairs` (Optional[List[Tuple[str, str]]], optional): Pairs to set cutoff 0.0. Defaults to None.

**Returns:**
- `None`: Modifies pipeline in-place.

**Notes:**
- Automatically blocks direct metal-metal bonds (Pu-Pu, Na-Na, Pu-Na, Na-Pu)
- `pair_cutoffs` should include symmetric entries (A,B) and (B,A) if desired
- Clears all existing modifiers before adding bond modifier

**Example:**
```python
from bond_based.bondmodifier_utils import configure_bonds_modifier_from_cutoffs

pair_cutoffs = {
    ("Pu", "Cl"): 3.2,
    ("Cl", "Pu"): 3.2,
    ("Na", "Cl"): 3.4,
    ("Cl", "Na"): 3.4,
}

# Disable Pu-Na direct bonds
disable_pairs = [("Pu", "Na"), ("Na", "Pu")]

configure_bonds_modifier_from_cutoffs(pipeline, pair_cutoffs, disable_pairs)
```

---

#### `build_shared_anion_graph`

```python
build_shared_anion_graph(
    data: Any,
    names: np.ndarray,
    anion: str = 'Cl',
    metals: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, nx.Graph]
```

Construct shared-anion connectivity graph among metal atoms.

**Parameters:**
- `data` (Any): OVITO data object with bonds already computed
- `names` (np.ndarray): Array of atom names/types
- `anion` (str, optional): Anion species name. Defaults to 'Cl'.
- `metals` (Optional[List[str]], optional): List of metal species. Defaults to ['Pu', 'Na'].

**Returns:**
- `Tuple[np.ndarray, np.ndarray, nx.Graph]`: 
  - `sizes`: Array of cluster sizes
  - `cluster_ids`: Array mapping atom indices to cluster IDs (-1 for non-metals/unclustered)
  - `G`: NetworkX graph of metal connectivity

**Algorithm:**
1. Identifies metal atoms and anions
2. For each anion, finds all bonded metal neighbors
3. Connects metal atoms pairwise if they share an anion
4. Identifies connected components as clusters

**Example:**
```python
from bond_based.bondmodifier_utils import build_shared_anion_graph

data = pipeline.compute(pipeline.source.num_frames - 1)
names = extract_names_array(data.particles)

sizes, cluster_ids, G = build_shared_anion_graph(
    data,
    names,
    anion='Cl',
    metals=['Pu', 'Na']
)

print(f"Found {len(sizes)} clusters")
print(f"Largest cluster size: {max(sizes) if len(sizes) > 0 else 0}")
```

---

#### `canonical_cluster_workflow`

```python
canonical_cluster_workflow(
    pipeline: Any,
    disable_pair: Optional[Tuple[str, str]] = None,
    metals: Optional[List[str]] = None,
    anion: str = 'Cl',
    rdf_samples: int = 100,
    pair_cutoffs: Optional[Dict[Tuple[str, str], float]] = None
) -> Dict[str, Any]
```

End-to-end workflow: compute RDFs, choose cutoffs, create bonds, compute shared-anion clusters.

**Parameters:**
- `pipeline` (Any): OVITO pipeline object
- `disable_pair` (Optional[Tuple[str, str]], optional): Pair to disable (e.g., ("Pu", "Na")). Defaults to None.
- `metals` (Optional[List[str]], optional): List of metal species. Defaults to ['Pu', 'Na'].
- `anion` (str, optional): Anion species name. Defaults to 'Cl'.
- `rdf_samples` (int, optional): Number of frames to sample for RDF. Defaults to 100.
- `pair_cutoffs` (Optional[Dict], optional): Pre-computed cutoffs. If None, will compute from RDFs. Defaults to None.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `rdf`: Computed RDF dictionary
  - `pair_cutoffs`: Used cutoff dictionary
  - `data`: Final computed data object
  - `names`: Array of atom names
  - `sizes`: Array of cluster sizes
  - `cluster_ids`: Array of cluster IDs
  - `graph`: NetworkX graph

**Example:**
```python
from bond_based.bondmodifier_utils import canonical_cluster_workflow

result = canonical_cluster_workflow(
    pipeline,
    disable_pair=("Pu", "Na"),  # Disable direct Pu-Na bonds
    rdf_samples=50
)

print(f"Found {len(result['sizes'])} clusters")
print(f"Largest cluster: {max(result['sizes']) if len(result['sizes']) > 0 else 0} atoms")
```

---

#### Module: `bond_based.shell_utils`

Shell analysis utilities for bond-based cluster analysis, including coordination number calculations and shell detection.

#### `get_species_densities_from_data`

```python
get_species_densities_from_data(data: Any) -> Dict[str, float]
```

Calculate species densities from simulation data.

**Parameters:**
- `data` (Any): OVITO data object containing particle information

**Returns:**
- `Dict[str, float]`: Dictionary mapping species names to number densities (atoms/Å³)

**Example:**
```python
from bond_based.shell_utils import get_species_densities_from_data

densities = get_species_densities_from_data(data)
print(f"Pu density: {densities['Pu']:.4f} atoms/Å³")
print(f"Na density: {densities['Na']:.4f} atoms/Å³")
```

---

#### `calculate_coordination_number`

```python
calculate_coordination_number(
    r: np.ndarray,
    g_r: np.ndarray,
    rho_j: float,
    r1: Optional[float] = None,
    r2: Optional[float] = None
) -> float
```

Calculate coordination number using the formula: CN_ij = 4πρ_j ∫[r1 to r2] r² g_ij(r) dr

**Parameters:**
- `r` (np.ndarray): Radial distances
- `g_r` (np.ndarray): Radial distribution function values
- `rho_j` (float): Number density of species j
- `r1` (Optional[float], optional): Lower integration limit. If None, defaults to 0.0.
- `r2` (Optional[float], optional): Upper integration limit. If None, determined automatically from first minimum.

**Returns:**
- `float`: Coordination number

**Mathematical Background:**
The coordination number CN_ij represents the average number of atoms of species j surrounding an atom of species i:

CN_ij = 4πρ_j ∫[r1 to r2] r² g_ij(r) dr

where:
- ρ_j is the number density of species j
- g_ij(r) is the radial distribution function
- The integration limits r1 and r2 typically span the first coordination shell

**Example:**
```python
from bond_based.shell_utils import calculate_coordination_number

r = rdf['r']
g_pu_cl = rdf['Pu-Cl']
rho_cl = densities['Cl']

# Automatic integration limits
cn = calculate_coordination_number(r, g_pu_cl, rho_cl)

# Manual integration limits
cn_manual = calculate_coordination_number(r, g_pu_cl, rho_cl, r1=0.0, r2=3.5)
```

---

#### `calculate_coordination_number_vs_r`

```python
calculate_coordination_number_vs_r(
    r: np.ndarray,
    g_r: np.ndarray,
    rho_j: float,
    r_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]
```

Calculate coordination number as a function of radial distance r.

**Parameters:**
- `r` (np.ndarray): Radial distances
- `g_r` (np.ndarray): Radial distribution function values
- `rho_j` (float): Number density of species j
- `r_max` (Optional[float], optional): Maximum integration distance. If None, uses full r range.

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (r_values, cn_values) arrays

**Mathematical Background:**
Computes cumulative coordination number CN(r) by integrating from 0 to r:

CN(r) = 4πρ_j ∫[0 to r] r'² g_ij(r') dr'

This provides a cumulative view of coordination as a function of distance.

**Example:**
```python
from bond_based.shell_utils import calculate_coordination_number_vs_r

r_vals, cn_vals = calculate_coordination_number_vs_r(r, g_pu_cl, rho_cl, r_max=10.0)

# Plot coordination number vs distance
import matplotlib.pyplot as plt
plt.plot(r_vals, cn_vals)
plt.xlabel('r (Å)')
plt.ylabel('CN(r)')
plt.show()
```

---

#### `detect_proper_shells`

```python
detect_proper_shells(
    first_deriv: np.ndarray,
    min_prominence: float = 0.1,
    smoothing_method: str = 'gaussian',
    gaussian_sigma: float = 3.0
) -> np.ndarray
```

Detect shells using proper pattern: derivative goes up, peaks, then goes down to zero.

**Parameters:**
- `first_deriv` (np.ndarray): First derivative array of coordination number
- `min_prominence` (float, optional): Minimum prominence for peaks. Defaults to 0.1.
- `smoothing_method` (str, optional): Smoothing method ('gaussian'). Defaults to 'gaussian'.
- `gaussian_sigma` (float, optional): Standard deviation for Gaussian smoothing. Defaults to 3.0.

**Returns:**
- `np.ndarray`: Array of shell boundary indices

**Algorithm:**
1. Smooths the first derivative using Gaussian filter
2. Finds peaks in first derivative (where coordination growth is fastest)
3. For each peak, finds subsequent minimum (where derivative goes to zero)
4. Returns indices where shells are detected

**Example:**
```python
from bond_based.shell_utils import detect_proper_shells
import numpy as np

# Calculate first derivative of coordination
cn_gradient = np.gradient(cn_values)

# Detect shells
shell_boundaries = detect_proper_shells(cn_gradient, min_prominence=0.1)
shell_radii = r_values[shell_boundaries]

print(f"Detected {len(shell_boundaries)} shells at radii: {shell_radii}")
```

---

#### `analyze_pu_neighbor_shells`

```python
analyze_pu_neighbor_shells(
    pipe: Any,
    names: np.ndarray,
    cutoffs: Dict[Tuple[str, str], float],
    max_radius: float = 10.0
) -> Dict[str, Any]
```

Analyze Pu neighbor shells by varying PuCl bond cutoff and counting all Pu neighbors.

**Parameters:**
- `pipe` (Any): OVITO pipeline object
- `names` (np.ndarray): Array of atom names/types
- `cutoffs` (Dict[Tuple[str, str], float]): Dictionary of current cutoff values
- `max_radius` (float, optional): Maximum radius to analyze. Defaults to 10.0 Å.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `radius_range`: Array of radii analyzed
  - `pu_pu_neighbors`: Average Pu-Pu neighbors vs radius
  - `pu_na_neighbors`: Average Pu-Na neighbors vs radius
  - `pu_cl_neighbors`: Average Pu-Cl neighbors vs radius
  - `pu_total_neighbors`: Average total Pu neighbors vs radius
  - `pu_pu_shells`: Detected Pu-Pu shell boundaries
  - `pu_cl_shells`: Detected Pu-Cl shell boundaries
  - `pu_total_shells`: Detected total shell boundaries
  - `pu_atom_count`: Number of Pu atoms

**Example:**
```python
from bond_based.shell_utils import analyze_pu_neighbor_shells

analysis = analyze_pu_neighbor_shells(pipe, names, cutoffs, max_radius=12.0)

# Access results
radius_range = analysis['radius_range']
pu_cl_neighbors = analysis['pu_cl_neighbors']
pu_cl_shells = analysis['pu_cl_shells']
```

---

#### `analyze_pu_neighbor_distributions`

```python
analyze_pu_neighbor_distributions(
    pipe: Any,
    names: np.ndarray,
    cutoffs: Dict[Tuple[str, str], float],
    max_radius: float = 10.0
) -> Dict[str, Any]
```

Build histograms showing Pu neighbor distributions at key distances.

**Parameters:**
- `pipe` (Any): OVITO pipeline object
- `names` (np.ndarray): Array of atom names/types
- `cutoffs` (Dict[Tuple[str, str], float]): Dictionary of current cutoff values
- `max_radius` (float, optional): Maximum radius to analyze. Defaults to 10.0 Å.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `key_distances`: List of distances analyzed
  - `pu_pu_distributions`: Dict mapping distance to coordination counts
  - `pu_na_distributions`: Dict mapping distance to coordination counts
  - `pu_cl_distributions`: Dict mapping distance to coordination counts
  - `pu_total_distributions`: Dict mapping distance to coordination counts
  - `colors`: Color scheme for plotting
  - `distance_labels`: Labels for each distance

**Example:**
```python
from bond_based.shell_utils import analyze_pu_neighbor_distributions

distributions = analyze_pu_neighbor_distributions(pipe, names, cutoffs)

# Access coordination distributions at specific distance
coord_at_3A = distributions['pu_cl_distributions'][3.0]
print(f"Mean Pu-Cl coordination at 3.0 Å: {np.mean(coord_at_3A):.2f}")
```

---

#### `analyze_bond_coordination`

```python
analyze_bond_coordination(
    data: Any,
    names: np.ndarray,
    pair_cutoffs: Optional[Dict[Tuple[str, str], float]] = None
) -> Dict[str, Dict[str, List[int]]]
```

Analyze coordination using bond-based analysis.

**Parameters:**
- `data` (Any): OVITO data object
- `names` (np.ndarray): Array of atom names/types
- `pair_cutoffs` (Optional[Dict], optional): Dictionary of bond cutoffs for different species pairs. Defaults to None (uses default 4.0 Å).

**Returns:**
- `Dict[str, Dict[str, List[int]]]`: Coordination data organized by central species and neighbor species. Includes:
  - `coord_data[sp_c]['any']`: Total coordination for each central species
  - `coord_data['any'][sp_n]`: Coordination from any central species to each neighbor species
  - `coord_data['any']['any']`: Total coordination across all species pairs

**Example:**
```python
from bond_based.shell_utils import analyze_bond_coordination

pair_cutoffs = {
    ("Pu", "Cl"): 3.2,
    ("Na", "Cl"): 3.4,
}

coord_data = analyze_bond_coordination(data, names, pair_cutoffs)

# Access Pu-Cl coordination
pu_cl_coords = coord_data['Pu']['Cl']
print(f"Average Pu-Cl coordination: {np.mean(pu_cl_coords):.2f}")
```

---

### Voronoi-Based Analysis

#### Module: `voronoi.frued.voronoi_utils`

Utilities for Voronoi-based analysis using freud library.

#### `build_voronoi_graph`

```python
build_voronoi_graph(
    atoms: Any,
    min_area: float = 0.0
) -> nx.Graph
```

Build a Voronoi graph for all atoms using freud.

**Parameters:**
- `atoms` (Any): ASE Atoms-like object with PBC enabled in all directions
- `min_area` (float, optional): Minimum facet area to accept an edge. Defaults to 0.0.

**Returns:**
- `nx.Graph`: NetworkX graph with:
  - Nodes: attributes `position`, `species`, `index`
  - Edges: attributes `area` (facet area), `species_pair`

**Example:**
```python
from voronoi.frued.voronoi_utils import build_voronoi_graph
from ase import Atoms

# Create or load atoms object
atoms = Atoms(...)  # Must have PBC enabled

G = build_voronoi_graph(atoms, min_area=0.01)
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
```

---

#### `build_voronoi_graph_metals_only`

```python
build_voronoi_graph_metals_only(
    atoms: Any,
    min_area: float = 0.0,
    metal_species: Iterable[str] | None = None
) -> nx.Graph
```

Build a Voronoi graph using only selected metal atoms as points.

**Parameters:**
- `atoms` (Any): ASE Atoms-like object
- `min_area` (float, optional): Minimum facet area to accept an edge. Defaults to 0.0.
- `metal_species` (Iterable[str] | None, optional): Species to include (e.g., ["Pu", "Na"]). Defaults to ["Pu", "Na"].

**Returns:**
- `nx.Graph`: Metals-only Voronoi graph

**Example:**
```python
from voronoi.frued.voronoi_utils import build_voronoi_graph_metals_only

G_metals = build_voronoi_graph_metals_only(
    atoms,
    min_area=0.01,
    metal_species=["Pu", "Na"]
)

# This graph only contains Pu and Na atoms, connected by Voronoi edges
```

---

#### `analyze_voronoi_coordination`

```python
analyze_voronoi_coordination(
    atoms_list: List[Any],
    at_list: Optional[Iterable[str]] = None,
    min_area: float = 0.0
) -> Dict[str, Dict[str, List[int]]]
```

Compute neighbor-count distributions by species using Voronoi.

**Parameters:**
- `atoms_list` (List[Any]): List of ASE Atoms objects
- `at_list` (Optional[Iterable[str]], optional): List of species to analyze. If None, uses all species found. Defaults to None.
- `min_area` (float, optional): Minimum facet area threshold. Defaults to 0.0.

**Returns:**
- `Dict[str, Dict[str, List[int]]]`: Mapping central_species -> neighbor_species -> list of coordination counts

**Example:**
```python
from voronoi.frued.voronoi_utils import analyze_voronoi_coordination

# Analyze multiple frames
atoms_list = [atoms_frame_0, atoms_frame_1, atoms_frame_2]

coord_data = analyze_voronoi_coordination(atoms_list, min_area=0.01)

# Access Pu-Cl coordination counts
pu_cl_counts = coord_data['Pu']['Cl']
print(f"Mean Pu-Cl coordination: {np.mean(pu_cl_counts):.2f}")
```

---

#### `analyze_voronoi_clusters`

```python
analyze_voronoi_clusters(
    atoms: Any,
    min_area: float = 0.0,
    metal_species: Iterable[str] | None = None
) -> Tuple[np.ndarray, np.ndarray, nx.Graph]
```

Cluster analysis using metals-only Voronoi graph.

**Parameters:**
- `atoms` (Any): ASE Atoms object
- `min_area` (float, optional): Minimum facet area threshold. Defaults to 0.0.
- `metal_species` (Iterable[str] | None, optional): Metal species to include. Defaults to ["Pu", "Na"].

**Returns:**
- `Tuple[np.ndarray, np.ndarray, nx.Graph]`:
  - `cluster_sizes`: Array of component sizes
  - `cluster_ids`: Array of cluster id per atom (-1 for non-metals/unclustered)
  - `G`: Metals-only Voronoi graph used for clustering

**Example:**
```python
from voronoi.frued.voronoi_utils import analyze_voronoi_clusters

sizes, cluster_ids, G = analyze_voronoi_clusters(
    atoms,
    min_area=0.01,
    metal_species=["Pu", "Na"]
)

print(f"Found {len(sizes)} clusters")
print(f"Largest cluster: {max(sizes) if len(sizes) > 0 else 0} atoms")
```

---

#### `analyze_graph_properties`

```python
analyze_graph_properties(
    G: nx.Graph,
    species_filter: Optional[Iterable[str]] = None
) -> Dict[str, Any]
```

Summarize graph properties useful for analysis.

**Parameters:**
- `G` (nx.Graph): NetworkX graph
- `species_filter` (Optional[Iterable[str]], optional): Filter nodes by species. Defaults to None.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `num_nodes`, `num_edges`: Graph size
  - `density`: Graph density
  - `is_connected`: Whether graph is connected
  - `num_components`: Number of connected components
  - `largest_component_size`: Size of largest component
  - `avg_degree`, `std_degree`: Degree statistics
  - `avg_facet_area`: Average Voronoi facet area
  - `species_counts`: Counts by species
  - And more...

**Example:**
```python
from voronoi.frued.voronoi_utils import analyze_graph_properties

props = analyze_graph_properties(G)
print(f"Graph density: {props['density']:.4f}")
print(f"Average degree: {props['avg_degree']:.2f}")
print(f"Number of components: {props['num_components']}")
```

---

#### Module: `voronoi.ovito.voronoi_ovito_utils`

OVITO-based Voronoi utilities mirroring the public API of `voronoi_utils.py`.

#### `build_voronoi_graph_from_pipeline`

```python
build_voronoi_graph_from_pipeline(
    pipeline: Any,
    frame: int = 0,
    min_area: float = 0.0,
    use_radii: bool = False,
    edge_threshold: float = 0.0
) -> nx.Graph
```

Build Voronoi graph directly from an OVITO pipeline frame.

**Parameters:**
- `pipeline` (Any): OVITO pipeline object
- `frame` (int, optional): Frame index to analyze. Defaults to 0.
- `min_area` (float, optional): Minimum facet area threshold. Defaults to 0.0.
- `use_radii` (bool, optional): Use atomic radii for weighted Voronoi. Defaults to False.
- `edge_threshold` (float, optional): Minimum edge length threshold. Defaults to 0.0.

**Returns:**
- `nx.Graph`: Voronoi graph (same structure as freud version)

**Example:**
```python
from voronoi.ovito.voronoi_ovito_utils import build_voronoi_graph_from_pipeline

G = build_voronoi_graph_from_pipeline(pipeline, frame=100, min_area=0.01)
```

---

#### Module: `voronoi.voronoi_coordination`

Voronoi-weighted coordination number analysis methods.

#### `compute_topological_coordination_number`

```python
compute_topological_coordination_number(
    voronoi_graph: Any,
    atom_indices: Optional[List[int]] = None
) -> Dict[int, int]
```

Compute topological coordination number (CN_faces).

**Mathematical Background:**
CN_faces = F, where F is the number of faces of the atom's Voronoi cell. Each Voronoi face corresponds to one nearest neighbor.

**Parameters:**
- `voronoi_graph` (Any): NetworkX graph with Voronoi tessellation
- `atom_indices` (Optional[List[int]], optional): Specific atoms to compute. If None, computes for all. Defaults to None.

**Returns:**
- `Dict[int, int]`: Dictionary mapping atom index to coordination number

**Example:**
```python
from voronoi.voronoi_coordination import compute_topological_coordination_number

cn_faces = compute_topological_coordination_number(G)
print(f"Average CN_faces: {np.mean(list(cn_faces.values())):.2f}")
```

---

#### `compute_face_area_weighted_cn`

```python
compute_face_area_weighted_cn(
    voronoi_graph: Any,
    atom_indices: Optional[List[int]] = None,
    return_normalized: bool = False
) -> Dict[int, float]
```

Compute face-area weighted coordination number (CN_A).

**Mathematical Background:**
CN_A = (Σ A_i)² / Σ A_i²

Equivalent form using normalized weights p_i = A_i / Σ A_j:
CN_A = 1 / Σ p_i²

This provides a continuous measure that accounts for the distribution of neighbor strengths.

**Parameters:**
- `voronoi_graph` (Any): NetworkX graph with 'area' edge attribute
- `atom_indices` (Optional[List[int]], optional): Specific atoms to compute. Defaults to None.
- `return_normalized` (bool, optional): Return normalized weights along with CN_A. Defaults to False.

**Returns:**
- `Dict[int, float]`: Dictionary mapping atom index to CN_A (and optionally weights)

**Example:**
```python
from voronoi.voronoi_coordination import compute_face_area_weighted_cn

cn_a = compute_face_area_weighted_cn(G)
cn_a_normalized, weights = compute_face_area_weighted_cn(G, return_normalized=True)
```

---

#### `compute_solid_angle_weighted_cn`

```python
compute_solid_angle_weighted_cn(
    voronoi_graph: Any,
    positions: Optional[np.ndarray] = None,
    atom_indices: Optional[List[int]] = None,
    return_normalized: bool = False
) -> Dict[int, float]
```

Compute solid-angle weighted coordination number (CN_Ω).

**Mathematical Background:**
CN_Ω = (Σ Ω_i)² / Σ Ω_i²

where Ω_i = A_i / r_i² is the solid angle subtended by face i.

Equivalent form using normalized weights q_i = Ω_i / Σ Ω_j:
CN_Ω = 1 / Σ q_i²

**Parameters:**
- `voronoi_graph` (Any): NetworkX graph with 'area' edge attribute
- `positions` (Optional[np.ndarray], optional): Nx3 array of atomic positions. If None, extracts from graph. Defaults to None.
- `atom_indices` (Optional[List[int]], optional): Specific atoms to compute. Defaults to None.
- `return_normalized` (bool, optional): Return normalized weights along with CN_Ω. Defaults to False.

**Returns:**
- `Dict[int, float]`: Dictionary mapping atom index to CN_Ω

**Example:**
```python
from voronoi.voronoi_coordination import compute_solid_angle_weighted_cn

positions = np.array([G.nodes[idx]['position'] for idx in G.nodes()])
cn_omega = compute_solid_angle_weighted_cn(G, positions=positions)
```

---

#### `compute_solid_angle_threshold_cn`

```python
compute_solid_angle_threshold_cn(
    voronoi_graph: Any,
    positions: Optional[np.ndarray] = None,
    tau: float = 0.5,
    atom_indices: Optional[List[int]] = None
) -> Dict[int, int]
```

Compute coordination number using solid-angle thresholding (O'Keeffe/VoronoiNN).

**Mathematical Background:**
Defines neighbor weights w_i = Ω_i and keeps neighbors satisfying:
w_i >= τ * max_j w_j

Then CN = Σ Θ(w_i - τ * max_j w_j), where Θ is the Heaviside step function and τ ≈ 0.5 is typical.

**Parameters:**
- `voronoi_graph` (Any): NetworkX graph with 'area' edge attribute
- `positions` (Optional[np.ndarray], optional): Nx3 array of atomic positions. Defaults to None.
- `tau` (float, optional): Threshold fraction (typical value ~0.5). Defaults to 0.5.
- `atom_indices` (Optional[List[int]], optional): Specific atoms to compute. Defaults to None.

**Returns:**
- `Dict[int, int]`: Dictionary mapping atom index to threshold-based coordination number

**Example:**
```python
from voronoi.voronoi_coordination import compute_solid_angle_threshold_cn

cn_thresh = compute_solid_angle_threshold_cn(G, positions=positions, tau=0.5)
```

---

#### `compute_crystal_nn_weight`

```python
compute_crystal_nn_weight(
    voronoi_graph: Any,
    positions: Optional[np.ndarray] = None,
    atom_indices: Optional[List[int]] = None
) -> Dict[int, Dict[int, float]]
```

Compute CrystalNN weights for neighbors.

**Mathematical Background:**
w_i^Vor = Ω_i² / A_i

This combines Voronoi solid-angle and face-area weighting. Additional distance and chemical weighting can be applied later.

**Parameters:**
- `voronoi_graph` (Any): NetworkX graph with 'area' edge attribute
- `positions` (Optional[np.ndarray], optional): Nx3 array of atomic positions. Defaults to None.
- `atom_indices` (Optional[List[int]], optional): Specific atoms to compute. Defaults to None.

**Returns:**
- `Dict[int, Dict[int, float]]`: Dictionary mapping atom index to dict of {neighbor_index: weight}

**Example:**
```python
from voronoi.voronoi_coordination import compute_crystal_nn_weight

crystal_weights = compute_crystal_nn_weight(G, positions=positions)
# Access weights for atom 0
weights_atom_0 = crystal_weights[0]
```

---

#### `analyze_voronoi_coordination_all_methods`

```python
analyze_voronoi_coordination_all_methods(
    voronoi_graph: Any,
    positions: Optional[np.ndarray] = None,
    tau: float = 0.5,
    atom_indices: Optional[List[int]] = None
) -> Dict[str, Dict[int, Any]]
```

Comprehensive coordination analysis using all methods.

**Parameters:**
- `voronoi_graph` (Any): NetworkX graph with Voronoi tessellation
- `positions` (Optional[np.ndarray], optional): Nx3 array of atomic positions
- `tau` (float, optional): Threshold for solid-angle method. Defaults to 0.5.
- `atom_indices` (Optional[List[int]], optional): Specific atoms to analyze. Defaults to None.

**Returns:**
- `Dict[str, Dict[int, Any]]`: Dictionary with all coordination metrics:
  - `CN_faces`: Topological coordination numbers
  - `CN_A`: Face-area weighted coordination numbers
  - `CN_Ω`: Solid-angle weighted coordination numbers
  - `CN_threshold`: Threshold-based coordination numbers
  - `CrystalNN_weights`: CrystalNN neighbor weights

**Example:**
```python
from voronoi.voronoi_coordination import analyze_voronoi_coordination_all_methods

results = analyze_voronoi_coordination_all_methods(G, positions=positions, tau=0.5)

cn_faces = results['CN_faces']
cn_area = results['CN_A']
cn_omega = results['CN_Ω']
cn_threshold = results['CN_threshold']
crystal_weights = results['CrystalNN_weights']
```

---

#### `compute_coordination_statistics`

```python
compute_coordination_statistics(
    cn_values: Dict[int, Any],
    species: Optional[Dict[int, str]] = None
) -> Dict[str, Any]
```

Compute statistical summary of coordination numbers.

**Parameters:**
- `cn_values` (Dict[int, Any]): Dictionary mapping atom index to CN value
- `species` (Optional[Dict[int, str]], optional): Dictionary mapping atom index to species

**Returns:**
- `Dict[str, Any]`: Dictionary with mean, std, min, max, and optionally species breakdown

**Example:**
```python
from voronoi.voronoi_coordination import compute_coordination_statistics

species_dict = {idx: G.nodes[idx]['species'] for idx in G.nodes()}
stats = compute_coordination_statistics(cn_faces, species=species_dict)

print(f"Overall mean: {stats['mean']:.2f}")
print(f"Pu mean: {stats['by_species']['Pu']['mean']:.2f}")
```

---

### Utilities

#### Module: `utils.plots`

Merged plotting utilities for cluster visualization, coordination analysis, and graph visualization.

#### `plot_cluster_size_distribution`

```python
plot_cluster_size_distribution(
    sizes: List[int],
    title: str = "Cluster Size Distribution"
) -> None
```

Plot histogram of cluster sizes.

**Parameters:**
- `sizes` (List[int]): List of cluster sizes
- `title` (str, optional): Plot title. Defaults to "Cluster Size Distribution".

**Example:**
```python
from utils.plots import plot_cluster_size_distribution

plot_cluster_size_distribution(sizes, title="Pu Clusters")
```

---

#### `plot_cluster_composition_analysis`

```python
plot_cluster_composition_analysis(
    data: Any,
    cluster_ids: np.ndarray,
    names: np.ndarray
) -> List[dict]
```

Analyze and plot cluster composition (Pu/Na fractions).

**Parameters:**
- `data` (Any): OVITO data object
- `cluster_ids` (np.ndarray): Array mapping atoms to cluster IDs
- `names` (np.ndarray): Array of atom names/types

**Returns:**
- `List[dict]`: List of cluster composition dictionaries

**Example:**
```python
from utils.plots import plot_cluster_composition_analysis

compositions = plot_cluster_composition_analysis(data, cluster_ids, names)
```

---

#### `plot_3d_cluster_visualization`

```python
plot_3d_cluster_visualization(
    data: Any,
    cluster_ids: np.ndarray,
    names: np.ndarray,
    max_clusters: int = 10
) -> None
```

Visualize clusters in 3D.

**Parameters:**
- `data` (Any): OVITO data object
- `cluster_ids` (np.ndarray): Array mapping atoms to cluster IDs
- `names` (np.ndarray): Array of atom names/types
- `max_clusters` (int, optional): Maximum number of clusters to display. Defaults to 10.

**Example:**
```python
from utils.plots import plot_3d_cluster_visualization

plot_3d_cluster_visualization(data, cluster_ids, names, max_clusters=5)
```

---

#### `plot_coordination_histograms`

```python
plot_coordination_histograms(
    coord_data: Dict[str, Dict[str, List[int]]],
    central_type: str
) -> None
```

Plot coordination histograms for a central species.

**Parameters:**
- `coord_data` (Dict[str, Dict[str, List[int]]]): Coordination data dictionary
- `central_type` (str): Central species to plot

**Example:**
```python
from utils.plots import plot_coordination_histograms

plot_coordination_histograms(coord_data, central_type='Pu')
```

---

#### `analyze_bond_network`

```python
analyze_bond_network(
    data: Any,
    names: np.ndarray
) -> Counter
```

Analyze and plot bond network statistics.

**Parameters:**
- `data` (Any): OVITO data object with bonds
- `names` (np.ndarray): Array of atom names/types

**Returns:**
- `Counter`: Dictionary-like counter of bond types

**Example:**
```python
from utils.plots import analyze_bond_network

bond_counts = analyze_bond_network(data, names)
print(f"Number of Pu-Cl bonds: {bond_counts[('Pu', 'Cl')]}")
```

---

#### `plot_cluster_evolution_analysis`

```python
plot_cluster_evolution_analysis(
    temporal_results: Dict[str, Any],
    title: str = "Cluster Evolution Analysis"
) -> None
```

Plot temporal evolution of cluster properties.

**Parameters:**
- `temporal_results` (Dict[str, Any]): Dictionary containing:
  - `frame_indices`: List of frame indices
  - `cluster_statistics`: List of statistics dictionaries
- `title` (str, optional): Plot title. Defaults to "Cluster Evolution Analysis".

**Example:**
```python
from utils.plots import plot_cluster_evolution_analysis

temporal_results = {
    'frame_indices': [0, 10, 20, 30],
    'cluster_statistics': [...]
}
plot_cluster_evolution_analysis(temporal_results)
```

---

## Common Workflows

### Workflow 0: Unified Pu-Cl Cluster Analysis (Recommended)

The `pu_cluster_analysis_unified.ipynb` notebook provides a unified interface for Pu-Cl cluster analysis using three different graph-building methods. This is the recommended starting point for new users.

**Three Methods Available:**

1. **Voronoi OVITO**: Uses OVITO's `VoronoiAnalysisModifier` for Voronoi tessellation
2. **Voronoi freud**: Uses `freud` library for Voronoi tessellation
3. **Bond-based**: Uses OVITO's `CreateBondsModifier` with RDF-based cutoffs

**Key Features:**
- All methods produce Pu-Cl graphs with both Pu and Cl nodes connected by Pu-Cl edges
- Cluster sizes count only Pu atoms (Pu-Cl-Pu connections count as one Pu oligomer)
- Cluster IDs are assigned to both Pu and Cl atoms in the same component
- Unified analysis functions work identically regardless of graph building method

**Usage Example:**

```python
# Configuration
INPUT_PATH = 'trajectory.xyz'
FRAME_INDEX = -1  # Use last frame
METHOD = 'bond_based'  # or 'ovito_voronoi' or 'freud_voronoi'
MIN_AREA = 0.2  # For Voronoi methods
RDF_SAMPLES = 100  # For bond-based method

# Load data based on method
if METHOD == 'ovito_voronoi':
    from ovito.io import import_file
    pipeline = import_file(INPUT_PATH, multiple_frames=True)
    # ... build graph using build_mixed_voronoi_graph_from_pipeline
elif METHOD == 'freud_voronoi':
    from ase.io import read
    atoms = read(INPUT_PATH, index=FRAME_INDEX)
    # ... build graph using build_pu_cl_graph
elif METHOD == 'bond_based':
    from bond_based.bondmodifier_utils import canonical_cluster_workflow
    result = canonical_cluster_workflow(pipeline, metals=['Pu'], anion='Cl')
    G = result['graph']

# Unified analysis (same for all methods)
from utils.plots import (
    std_plot_cluster_size_distribution,
    std_plot_graph_structure,
    std_plot_3d_graph_components_pu_only
)

# Calculate cluster sizes (Pu atoms only)
cluster_sizes = calculate_pu_cluster_sizes_from_graph(G)

# Visualize
std_plot_cluster_size_distribution(cluster_sizes)
std_plot_3d_graph_components_pu_only(G, data, names, max_components=6)
```

**Note:** The unified notebook references functions from a `pu_cl_analysis` module that provides:
- `calculate_pu_cluster_sizes_from_graph`: Calculate cluster sizes counting only Pu atoms
- `calculate_pu_cluster_ids_from_graph`: Assign cluster IDs to both Pu and Cl atoms
- `analyze_pu_cluster_properties`: Analyze graph properties

See `pu_cluster_analysis_unified.ipynb` for the complete implementation.

---

### Workflow 1: Bond-Based Cluster Analysis

```python
from ovito.io import import_file
from bond_based.bondmodifier_utils import (
    canonical_cluster_workflow,
    extract_names_array
)
from utils.plots import plot_cluster_size_distribution

# Load trajectory
pipeline = import_file("trajectory.xyz")

# Run complete workflow
result = canonical_cluster_workflow(
    pipeline,
    disable_pair=("Pu", "Na"),  # Disable direct Pu-Na bonds
    rdf_samples=50
)

# Visualize results
plot_cluster_size_distribution(result['sizes'])
print(f"Found {len(result['sizes'])} clusters")
```

### Workflow 2: Voronoi-Based Coordination Analysis

```python
from voronoi.frued.voronoi_utils import (
    build_voronoi_graph,
    analyze_voronoi_clusters
)
from voronoi.voronoi_coordination import (
    analyze_voronoi_coordination_all_methods,
    compute_coordination_statistics
)
import numpy as np

# Build Voronoi graph
G = build_voronoi_graph(atoms, min_area=0.01)

# Extract positions
positions = np.array([G.nodes[idx]['position'] for idx in G.nodes()])

# Compute all coordination methods
results = analyze_voronoi_coordination_all_methods(G, positions=positions)

# Analyze statistics
species_dict = {idx: G.nodes[idx]['species'] for idx in G.nodes()}
stats = compute_coordination_statistics(results['CN_faces'], species=species_dict)

print(f"Overall mean CN: {stats['mean']:.2f}")
```

### Workflow 3: Shell Analysis

```python
from bond_based.shell_utils import (
    analyze_pu_neighbor_shells,
    plot_shell_diagram,
    calculate_coordination_number_vs_r,
    get_species_densities_from_data
)
from bond_based.bondmodifier_utils import compute_partial_rdfs

# Compute RDFs
rdf = compute_partial_rdfs(pipeline, nsamples=50)

# Get densities
densities = get_species_densities_from_data(data)

# Calculate coordination number vs distance
r = rdf['r']
g_pu_cl = rdf['Pu-Cl']
rho_cl = densities['Cl']
r_vals, cn_vals = calculate_coordination_number_vs_r(r, g_pu_cl, rho_cl)

# Analyze neighbor shells
analysis = analyze_pu_neighbor_shells(pipe, names, cutoffs, max_radius=12.0)
plot_shell_diagram(analysis)
```

### Workflow 4: Mixed Tessellation Analysis

```python
from voronoi.ovito.voronoi_ovito_utils import (
    build_mixed_voronoi_graph_from_pipeline,
    build_pu_only_cluster_graph_from_mixed,
    analyze_pu_cluster_properties_from_mixed,
    plot_mixed_tessellation_analysis
)

# Build mixed Voronoi tessellation (all atoms)
mixed_graph = build_mixed_voronoi_graph_from_pipeline(
    pipeline, frame=0, min_area=0.01
)

# Extract Pu-only cluster graph
pu_cluster_graph = build_pu_only_cluster_graph_from_mixed(mixed_graph, min_area=0.01)

# Comprehensive analysis
analysis_results = analyze_pu_cluster_properties_from_mixed(
    pipeline, frame=0, min_area=0.01
)

# Visualize
plot_mixed_tessellation_analysis(analysis_results)
```

---

## Notebooks and Examples

### Unified Analysis Notebook

**`pu_cluster_analysis_unified.ipynb`**: A comprehensive notebook that demonstrates unified Pu-Cl cluster analysis using three different methods:

- **Voronoi OVITO Method**: Uses OVITO's built-in Voronoi analysis
- **Voronoi freud Method**: Uses freud library for Voronoi tessellation  
- **Bond-Based Method**: Uses RDF-based bond cutoffs

The notebook provides:
- Unified interface for all three methods
- Consistent output format regardless of method
- Comparison capabilities between methods
- Temporal analysis examples
- Comprehensive visualization

**Key Functions Used:**
- Graph building: `build_mixed_voronoi_graph_from_pipeline`, `build_pu_cl_graph`, `canonical_cluster_workflow`
- Analysis: `calculate_pu_cluster_sizes_from_graph`, `calculate_pu_cluster_ids_from_graph`, `analyze_pu_cluster_properties`
- Visualization: `std_plot_cluster_size_distribution`, `std_plot_3d_graph_components_pu_only`

---

## Function Index

### Bond-Based Analysis

**bondmodifier_utils.py:**
- `compute_partial_rdfs` - Compute partial RDFs
- `find_first_minimum` - Find first minimum in RDF
- `find_first_shell_minimum` - Find first-shell minimum
- `determine_cutoffs_from_rdf` - Determine cutoffs from RDFs
- `configure_bonds_modifier_from_cutoffs` - Configure OVITO bonds
- `build_shared_anion_graph` - Build shared-anion connectivity graph
- `canonical_cluster_workflow` - Complete workflow
- `summarize_bonds` - Summarize bond statistics
- `extract_names_array` - Extract names from OVITO data

**shell_utils.py:**
- `get_species_densities_from_data` - Calculate species densities
- `calculate_coordination_number` - Calculate CN from RDF
- `calculate_coordination_number_vs_r` - Calculate CN vs distance
- `detect_proper_shells` - Detect coordination shells
- `analyze_pu_neighbor_shells` - Analyze Pu neighbor shells
- `analyze_pu_neighbor_distributions` - Analyze neighbor distributions
- `analyze_bond_coordination` - Analyze bond-based coordination
- `plot_shell_diagram` - Plot shell analysis
- `plot_coordination_derivatives_2nd_order` - Plot derivatives
- `plot_coordination_with_derivatives` - Plot coordination with derivatives
- `plot_bond_coordination_histograms` - Plot bond coordination histograms
- `plot_coordination_number_vs_r` - Plot CN vs distance
- `plot_multiple_cn_vs_r` - Plot multiple CN curves
- `save_cn_r_data` - Save CN(r) data

### Voronoi-Based Analysis

**voronoi_utils.py (freud):**
- `build_voronoi_graph` - Build full Voronoi graph
- `build_voronoi_graph_metals_only` - Build metals-only graph
- `build_pu_cl_graph` - Build Pu-Cl graph (Pu and Cl nodes, Pu-Cl edges only)
- `analyze_voronoi_coordination` - Analyze coordination
- `analyze_voronoi_coordination_metals_only` - Analyze metals-only coordination
- `analyze_graph_properties` - Analyze graph properties
- `analyze_temporal_graph_properties` - Analyze temporal properties
- `plot_temporal_graph_properties` - Plot temporal properties
- `analyze_voronoi_clusters` - Analyze clusters
- `summarize_voronoi_edge_network` - Summarize edge network

**voronoi_ovito_utils.py:**
- `build_voronoi_graph` - Build graph from atoms
- `build_voronoi_graph_metals_only` - Build metals-only graph
- `build_voronoi_graph_from_pipeline` - Build from OVITO pipeline
- `build_voronoi_graph_metals_only_from_pipeline` - Build metals-only from pipeline
- `analyze_voronoi_coordination_from_pipeline` - Analyze coordination from pipeline
- `analyze_temporal_graph_properties_from_pipeline` - Analyze temporal from pipeline
- `build_mixed_voronoi_graph_from_pipeline` - Build mixed tessellation
- `build_pu_only_cluster_graph_from_mixed` - Extract Pu-only clusters
- `build_pu_cluster_graph_from_mixed` - Extract Pu-Cl cluster graph from mixed tessellation
- `analyze_na_pu_interactions_from_mixed` - Analyze Na-Pu interactions
- `analyze_pu_cluster_properties_from_mixed` - Analyze Pu cluster properties
- `analyze_pu_coordination_in_mixed` - Analyze Pu coordination in mixed environment
- `build_neighbor_list_from_mixed` - Build neighbor list from mixed graph
- `plot_mixed_tessellation_analysis` - Plot mixed tessellation analysis
- `analyze_mixed_tessellation_from_pipeline` - Temporal analysis of mixed tessellation

**voronoi_coordination.py:**
- `compute_topological_coordination_number` - Topological CN
- `compute_face_area_weighted_cn` - Face-area weighted CN
- `compute_solid_angle_weighted_cn` - Solid-angle weighted CN
- `compute_solid_angle_threshold_cn` - Threshold-based CN
- `compute_crystal_nn_weight` - CrystalNN weights
- `analyze_voronoi_coordination_all_methods` - All methods
- `compute_coordination_statistics` - Statistical analysis
- `analyze_weighted_coordination_by_species` - By-species analysis
- `compute_coordination_histogram_data` - Histogram data

### Utilities

**plots.py:**
- `plot_cluster_size_distribution` - Cluster size histogram
- `std_plot_cluster_size_distribution` - Standardized cluster size histogram
- `plot_cluster_composition_analysis` - Composition analysis
- `plot_3d_cluster_visualization` - 3D cluster visualization
- `plot_3d_cluster_with_graph` - 3D with graph edges
- `std_plot_3d_graph_components` - Standardized 3D graph components
- `std_plot_3d_graph_components_pu_only` - Standardized 3D Pu-only clusters
- `std_plot_3d_graph_components_pu_cl` - Standardized 3D Pu-Cl clusters
- `plot_coordination_histograms` - Coordination histograms
- `std_plot_coordination_histograms` - Standardized coordination histograms
- `plot_graph_structure` - Graph structure plot
- `std_plot_graph_structure` - Standardized graph structure plot
- `analyze_bond_network` - Bond network analysis
- `plot_rdfs` - RDF plots
- `plot_cluster_evolution_analysis` - Temporal evolution
- `plot_pu_clusters_with_na_context` - Pu clusters with Na background
- `plot_na_pu_interaction_network` - Na-Pu interaction network
- `plot_pu_coordination_analysis` - Pu coordination analysis
- `plot_mixed_tessellation_structure` - Mixed tessellation structure
- `plot_3d_mixed_tessellation` - 3D mixed tessellation visualization
- `plot_neighbor_list_analysis` - Neighbor list analysis
- `setup_plot_style` - Setup plotting style
- `get_standard_colors` - Get color scheme
- `extract_positions_from_data` - Extract positions from data objects
- `extract_names_from_data` - Extract names from data objects

---

## Mathematical Background

### Radial Distribution Function (RDF)

The radial distribution function g_ij(r) describes the probability of finding an atom of species j at distance r from an atom of species i:

g_ij(r) = (1 / (4πr²ρ_j)) * (dN_ij(r) / dr)

where ρ_j is the number density of species j.

### Coordination Number

The coordination number CN_ij represents the average number of atoms of species j surrounding an atom of species i:

CN_ij = 4πρ_j ∫[r1 to r2] r² g_ij(r) dr

The integration limits r1 and r2 typically span the first coordination shell.

### Voronoi Coordination Numbers

**Topological CN (CN_faces):**
Each Voronoi face corresponds to one nearest neighbor:
CN_faces = F, where F is the number of faces

**Face-Area Weighted CN (CN_A):**
CN_A = (Σ A_i)² / Σ A_i² = 1 / Σ p_i²

where p_i = A_i / Σ A_j are normalized face areas.

**Solid-Angle Weighted CN (CN_Ω):**
CN_Ω = (Σ Ω_i)² / Σ Ω_i² = 1 / Σ q_i²

where Ω_i = A_i / r_i² is the solid angle and q_i = Ω_i / Σ Ω_j are normalized weights.

**Solid-Angle Threshold CN:**
CN = Σ Θ(w_i - τ * max_j w_j)

where w_i = Ω_i and Θ is the Heaviside step function.

**CrystalNN Weight:**
w_i^Vor = Ω_i² / A_i

This combines solid-angle and face-area weighting.

---

## References

- OVITO Documentation: https://www.ovito.org/docs/
- NetworkX Documentation: https://networkx.org/documentation/
- Freud Documentation: https://freud.readthedocs.io/
- ASE Documentation: https://wiki.fysik.dtu.dk/ase/

---

## License

This documentation is part of the cluster_analysis package. Please refer to the package license for usage terms.

---

*Last Updated: 2024*

