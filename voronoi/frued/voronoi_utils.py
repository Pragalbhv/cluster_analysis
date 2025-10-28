"""
Utilities for Voronoi-based analysis and plotting.

This module centralizes Voronoi tessellation helpers that were previously
embedded in notebooks (e.g., `vornoi_personality.ipynb`). It exposes simple
function calls for:

- Building Voronoi graphs (full system or metals-only)
- Coordination analysis across frames
- Graph property summaries and temporal evolution
- Cluster analysis on metals-only Voronoi graphs
- Voronoi "bond" summaries (edge counts by species pair) with plotting
- Thin wrappers that reuse plotting functions from `plot_utils`

Notes
-----
- Graphs use NetworkX where nodes carry `position`, `species`, and `index`.
- Voronoi neighbors are extracted via freud and stored as graph edges with
  edge attribute `area` (facet area).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    raise ImportError("voronoi_utils requires networkx to be installed") from exc

try:
    import freud
except Exception as exc:  # pragma: no cover
    raise ImportError("voronoi_utils requires freud to be installed") from exc


# Optional plotting utilities (reused for consistency)
try:
    from plot_utils import (
        plot_coordination_histograms as _plot_coordination_histograms,
        plot_graph_structure as _plot_graph_structure,
        plot_cluster_size_distribution as _plot_cluster_size_distribution,
        plot_cluster_composition_analysis as _plot_cluster_composition_analysis,
        plot_3d_cluster_visualization as _plot_3d_cluster_visualization,
        plot_3d_cluster_with_graph as _plot_3d_cluster_with_graph,
        analyze_bond_network as _analyze_bond_network,  # OVITO bonds (not Voronoi)
    )
except Exception:
    # Allow compute-only usage if plotting module is unavailable
    _plot_coordination_histograms = None  # type: ignore
    _plot_graph_structure = None  # type: ignore
    _plot_cluster_size_distribution = None  # type: ignore
    _plot_cluster_composition_analysis = None  # type: ignore
    _plot_3d_cluster_visualization = None  # type: ignore
    _plot_3d_cluster_with_graph = None  # type: ignore
    _analyze_bond_network = None  # type: ignore


# -----------------------------
# Voronoi graph construction
# -----------------------------

def build_voronoi_graph(atoms: Any, min_area: float = 0.0) -> nx.Graph:
    """Build a Voronoi graph for all atoms.

    Parameters
    ----------
    atoms : ASE Atoms-like
        Must have PBC enabled in all directions.
    min_area : float, optional
        Minimum facet area to accept an edge, by default 0.0.

    Returns
    -------
    networkx.Graph
        Nodes include attributes: `position`, `species`, `index`.
        Edges include attribute: `area` (facet area) and `species_pair`.
    """
    assert atoms.pbc.all(), "freud Voronoi expects PBC in all directions."

    box = freud.box.Box.from_matrix(atoms.get_cell().array)
    positions = atoms.get_positions()
    species = atoms.get_chemical_symbols()

    v = freud.locality.Voronoi()
    v.compute((box, positions))

    G = nx.Graph()

    for i, (pos, spec) in enumerate(zip(positions, species)):
        G.add_node(i, position=np.asarray(pos), species=str(spec), index=int(i))

    i_indices = v.nlist[:, 0]
    j_indices = v.nlist[:, 1]
    areas = v.nlist.weights

    mask = areas >= float(min_area)
    for i, j, area in zip(i_indices[mask], j_indices[mask], areas[mask]):
        G.add_edge(int(i), int(j), area=float(area), species_pair=f"{species[i]}-{species[j]}")

    return G


def build_voronoi_graph_metals_only(
    atoms: Any,
    min_area: float = 0.0,
    metal_species: Iterable[str] | None = None,
) -> nx.Graph:
    """Build a Voronoi graph using only selected metal atoms as points.

    Parameters
    ----------
    atoms : ASE Atoms-like
    min_area : float
        Minimum facet area to accept an edge.
    metal_species : iterable of str
        Species to include (e.g., ["Pu", "Na"]). Defaults to ["Pu", "Na"].
    """
    assert atoms.pbc.all(), "freud Voronoi expects PBC in all directions."

    if metal_species is None:
        metal_species = ["Pu", "Na"]

    species = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    metal_mask = np.array([sp in set(metal_species) for sp in species], dtype=bool)
    metal_indices = np.where(metal_mask)[0]
    if metal_indices.size == 0:
        return nx.Graph()

    metal_positions = positions[metal_indices]
    metal_species_list = [species[i] for i in metal_indices]

    box = freud.box.Box.from_matrix(atoms.get_cell().array)
    v = freud.locality.Voronoi()
    v.compute((box, metal_positions))

    G = nx.Graph()
    for idx_in_subset, (pos, spec) in enumerate(zip(metal_positions, metal_species_list)):
        original_index = int(metal_indices[idx_in_subset])
        G.add_node(original_index, position=np.asarray(pos), species=str(spec), index=original_index)

    i_indices = v.nlist[:, 0]
    j_indices = v.nlist[:, 1]
    areas = v.nlist.weights

    mask = areas >= float(min_area)
    for i, j, area in zip(i_indices[mask], j_indices[mask], areas[mask]):
        orig_i = int(metal_indices[int(i)])
        orig_j = int(metal_indices[int(j)])
        G.add_edge(orig_i, orig_j, area=float(area), species_pair=f"{metal_species_list[int(i)]}-{metal_species_list[int(j)]}")

    return G


# -----------------------------
# Coordination analysis
# -----------------------------

def analyze_voronoi_coordination(
    atoms_list: List[Any],
    at_list: Optional[Iterable[str]] = None,
    min_area: float = 0.0,
) -> Dict[str, Dict[str, List[int]]]:
    """Compute neighbor-count distributions by species using Voronoi.

    Returns mapping central_species -> neighbor_species -> list of coordination counts.
    """
    if at_list is None:
        all_species = sorted({s for at in atoms_list for s in at.get_chemical_symbols()})
    else:
        all_species = list(at_list)

    sp2idx = {sp: i for i, sp in enumerate(all_species)}
    coord_data: Dict[str, Dict[str, List[int]]] = {sp_c: {sp_n: [] for sp_n in all_species} for sp_c in all_species}

    for atoms in atoms_list:
        assert atoms.pbc.all(), "freud Voronoi expects PBC in all directions."
        box = freud.box.Box.from_matrix(atoms.get_cell().array)
        pos = atoms.get_positions()
        species = np.array(atoms.get_chemical_symbols())
        species_idx = np.array([sp2idx[s] for s in species], dtype=int)

        v = freud.locality.Voronoi()
        v.compute((box, pos))

        i = v.nlist[:, 0]
        j = v.nlist[:, 1]
        areas = v.nlist.weights

        mask = areas >= float(min_area)
        i = i[mask]
        j = j[mask]

        N = len(atoms)
        S = len(all_species)
        counts = np.zeros((N, S), dtype=int)
        np.add.at(counts, (i, species_idx[j]), 1)

        for sp_c, c_idx in sp2idx.items():
            rows = np.where(species_idx == c_idx)[0]
            if rows.size == 0:
                continue
            sub = counts[rows]
            for sp_n, n_idx in sp2idx.items():
                coord_data[sp_c][sp_n].extend(sub[:, n_idx].tolist())

    return coord_data


def analyze_voronoi_coordination_metals_only(
    atoms_list: List[Any],
    metal_species: Iterable[str] | None = None,
    min_area: float = 0.0,
) -> Dict[str, Dict[str, List[int]]]:
    """Coordination distributions restricted to selected metal species."""
    if metal_species is None:
        metal_species = ["Pu", "Na"]

    metals = list(metal_species)
    coord_data: Dict[str, Dict[str, List[int]]] = {sp_c: {sp_n: [] for sp_n in metals} for sp_c in metals}

    for atoms in atoms_list:
        assert atoms.pbc.all(), "freud Voronoi expects PBC in all directions."
        species = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        metal_mask = np.array([sp in set(metals) for sp in species], dtype=bool)
        metal_indices = np.where(metal_mask)[0]
        if metal_indices.size == 0:
            continue

        metal_positions = positions[metal_indices]
        metal_species_list = [species[i] for i in metal_indices]

        box = freud.box.Box.from_matrix(atoms.get_cell().array)
        v = freud.locality.Voronoi()
        v.compute((box, metal_positions))

        # For each metal atom, count neighbors by species with area threshold
        for i_center, central_spec in enumerate(metal_species_list):
            neighbors_mask = v.nlist.query_point_indices == i_center
            neighbor_indices = v.nlist.point_indices[neighbors_mask]
            neighbor_areas = v.nlist.weights[neighbors_mask]

            valid_neighbors = neighbor_indices[neighbor_areas >= float(min_area)]
            neighbor_species = [metal_species_list[j] for j in valid_neighbors]

            for neighbor_spec in metals:
                count = int(neighbor_species.count(neighbor_spec))
                coord_data[central_spec][neighbor_spec].append(count)

    return coord_data


# -----------------------------
# Graph properties and temporal evolution
# -----------------------------

def analyze_graph_properties(G: nx.Graph, species_filter: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Summarize graph properties useful for personality analysis."""
    properties: Dict[str, Any] = {}

    if species_filter is not None:
        allowed = set(species_filter)
        nodes_to_keep = [n for n in G.nodes if G.nodes[n].get("species") in allowed]
        G = G.subgraph(nodes_to_keep).copy()

    properties["num_nodes"] = G.number_of_nodes()
    properties["num_edges"] = G.number_of_edges()
    properties["density"] = nx.density(G) if G.number_of_nodes() > 1 else 0.0

    species_counts: Dict[str, int] = defaultdict(int)
    for node in G.nodes:
        species_counts[G.nodes[node]["species"]] += 1
    properties["species_counts"] = dict(species_counts)

    if properties["num_nodes"] > 0 and nx.is_connected(G):
        properties["is_connected"] = True
        properties["diameter"] = nx.diameter(G)
        properties["radius"] = nx.radius(G)
    else:
        properties["is_connected"] = False
        properties["num_components"] = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
        if G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            properties["largest_component_size"] = len(largest_cc)
            properties["largest_component_fraction"] = len(largest_cc) / G.number_of_nodes()

    if properties["num_nodes"] > 0:
        degrees = [G.degree(n) for n in G.nodes]
        properties["avg_degree"] = float(np.mean(degrees)) if degrees else 0.0
        properties["std_degree"] = float(np.std(degrees)) if degrees else 0.0
        properties["max_degree"] = int(max(degrees)) if degrees else 0
        properties["min_degree"] = int(min(degrees)) if degrees else 0

        species_degrees: Dict[str, List[int]] = defaultdict(list)
        for node in G.nodes:
            species_degrees[str(G.nodes[node]["species"])].append(G.degree(node))
        properties["species_avg_degrees"] = {sp: float(np.mean(d)) for sp, d in species_degrees.items() if d}

        if G.number_of_edges() > 0:
            edge_areas = [G.edges[e]["area"] for e in G.edges]
            properties["avg_facet_area"] = float(np.mean(edge_areas))
            properties["std_facet_area"] = float(np.std(edge_areas))
            properties["min_facet_area"] = float(np.min(edge_areas))
            properties["max_facet_area"] = float(np.max(edge_areas))
        else:
            properties["avg_facet_area"] = properties["std_facet_area"] = 0.0
            properties["min_facet_area"] = properties["max_facet_area"] = 0.0

    return properties


def analyze_temporal_graph_properties(
    atoms_list: List[Any], min_area: float = 0.0, sample_frames: int = 10
) -> Dict[str, Any]:
    """Compute temporal evolution of selected graph properties."""
    frame_indices = np.linspace(0, len(atoms_list) - 1, sample_frames, dtype=int)

    temporal: Dict[str, Any] = {
        "frame_indices": frame_indices,
        "num_nodes": [],
        "num_edges": [],
        "density": [],
        "avg_degree": [],
        "is_connected": [],
        "largest_component_fraction": [],
        "avg_facet_area": [],
    }

    for frame_idx in frame_indices:
        atoms = atoms_list[frame_idx]
        G = build_voronoi_graph(atoms, min_area=min_area)
        props = analyze_graph_properties(G)

        temporal["num_nodes"].append(props.get("num_nodes", 0))
        temporal["num_edges"].append(props.get("num_edges", 0))
        temporal["density"].append(props.get("density", 0.0))
        temporal["avg_degree"].append(props.get("avg_degree", 0.0))
        temporal["is_connected"].append(props.get("is_connected", False))
        temporal["largest_component_fraction"].append(props.get("largest_component_fraction", 1.0))
        temporal["avg_facet_area"].append(props.get("avg_facet_area", 0.0))

    return temporal


def plot_temporal_graph_properties(temporal: Dict[str, Any]) -> None:
    """Plot a minimal set of temporal curves for quick inspection."""
    idx = np.arange(len(temporal.get("frame_indices", [])))

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(idx, temporal.get("density", []), label="density")
    ax1.set_title("Graph density over time")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(idx, temporal.get("avg_degree", []), label="avg_degree", color="tab:orange")
    ax2.set_title("Average degree over time")
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(idx, temporal.get("num_edges", []), label="#edges", color="tab:green")
    ax3.set_title("Edges over time")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(idx, temporal.get("avg_facet_area", []), label="avg_facet_area", color="tab:red")
    ax4.set_title("Avg facet area over time")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Clusters on metals-only graphs
# -----------------------------

def analyze_voronoi_clusters(
    atoms: Any, min_area: float = 0.0, metal_species: Iterable[str] | None = None
) -> Tuple[np.ndarray, np.ndarray, nx.Graph]:
    """Cluster analysis using metals-only Voronoi graph.

    Returns
    -------
    cluster_sizes : (K,) array of component sizes
    cluster_ids : (N_atoms,) array of cluster id per atom, -1 for non-metals/unclustered
    G : metals-only Voronoi graph used for clustering
    """
    if metal_species is None:
        metal_species = ["Pu", "Na"]

    G = build_voronoi_graph_metals_only(atoms, min_area=min_area, metal_species=metal_species)
    if G.number_of_nodes() == 0:
        return np.array([], dtype=int), -np.ones(len(atoms), dtype=int), G

    components = list(nx.connected_components(G))
    cluster_sizes = np.array([len(comp) for comp in components], dtype=int)
    cluster_ids = -np.ones(len(atoms), dtype=int)

    for cid, comp in enumerate(components):
        for atom_idx in comp:
            cluster_ids[int(atom_idx)] = int(cid)

    return cluster_sizes, cluster_ids, G


# -----------------------------
# Voronoi edge/bond summaries
# -----------------------------

def summarize_voronoi_edge_network(G: nx.Graph, plot: bool = True) -> Counter:
    """Count Voronoi edges by species pair and optionally plot a bar chart.

    Returns a Counter with keys like ("Cl","Na") sorted alphabetically.
    """
    pair_counts: Counter = Counter()
    for u, v, data in G.edges(data=True):
        su = str(G.nodes[u].get("species"))
        sv = str(G.nodes[v].get("species"))
        key = tuple(sorted([su, sv]))
        pair_counts[key] += 1

    if plot and pair_counts:
        labels = [f"{a}-{b}" for (a, b) in pair_counts.keys()]
        values = list(pair_counts.values())

        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, values, alpha=0.8)
        plt.xlabel("Voronoi edge pair")
        plt.ylabel("Count")
        plt.title("Voronoi Edge Network Summary")
        plt.xticks(rotation=45)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(value), ha="center", va="bottom")
        plt.tight_layout()
        plt.show()

    # Console summary
    if pair_counts:
        print("Voronoi Edge Network Summary:")
        for (a, b), count in pair_counts.items():
            print(f"{a}-{b}: {count} edges")

    return pair_counts


# -----------------------------
# Thin plotting wrappers (optional)
# -----------------------------

def plot_coordination_histograms(coord_data: Dict[str, Dict[str, List[int]]], central_type: str) -> None:
    if _plot_coordination_histograms is None:
        raise ImportError("plot_utils is required for plotting coordination histograms")
    _plot_coordination_histograms(coord_data, central_type)


def plot_graph_structure(G: nx.Graph, title: str = "Voronoi Graph Structure") -> None:
    if _plot_graph_structure is None:
        raise ImportError("plot_utils is required for graph plotting")
    _plot_graph_structure(G, title)


def plot_cluster_size_distribution(sizes: List[int], title: str = "Cluster Size Distribution") -> None:
    if _plot_cluster_size_distribution is None:
        raise ImportError("plot_utils is required for plotting cluster size distribution")
    _plot_cluster_size_distribution(sizes, title)


def plot_cluster_composition_analysis(data: Any, cluster_ids: np.ndarray, names: np.ndarray) -> List[dict]:
    if _plot_cluster_composition_analysis is None:
        raise ImportError("plot_utils is required for plotting cluster composition")
    return _plot_cluster_composition_analysis(data, cluster_ids, names)


def plot_3d_cluster_visualization(data: Any, cluster_ids: np.ndarray, names: np.ndarray, max_clusters: int = 10) -> None:
    if _plot_3d_cluster_visualization is None:
        raise ImportError("plot_utils is required for 3D plotting")
    _plot_3d_cluster_visualization(data, cluster_ids, names, max_clusters=max_clusters)


def plot_3d_cluster_with_graph(data: Any, cluster_ids: np.ndarray, names: np.ndarray, G: Optional[Any], max_clusters: int = 10) -> None:
    if _plot_3d_cluster_with_graph is None:
        raise ImportError("plot_utils is required for 3D plotting")
    _plot_3d_cluster_with_graph(data, cluster_ids, names, G, max_clusters=max_clusters)


def analyze_bond_network_via_ovito(data: Any, names: np.ndarray) -> Counter:
    """Wrapper to reuse OVITO bond network analysis from plot_utils.

    This analyzes topological bonds from OVITO (not Voronoi edges).
    """
    if _analyze_bond_network is None:
        raise ImportError("plot_utils.analyze_bond_network is unavailable")
    return _analyze_bond_network(data, names)


# -----------------------------
# Helpers
# -----------------------------

def extract_info(path: str) -> Tuple[Optional[float], Optional[int]]:
    """Extract composition and temperature information from path.

    Matches segments like: NaCl-PuCl3/x{fraction}/T{temp}K/ ...
    """
    import re

    match = re.search(r"NaCl-PuCl3/x(\d*\.?\d+)/T(\d+)K/", path)
    if match:
        x = float(match.group(1))
        temperature = int(match.group(2))
        return x, temperature
    return None, None


__all__ = [
    # graph/build
    "build_voronoi_graph",
    "build_voronoi_graph_metals_only",
    # coordination
    "analyze_voronoi_coordination",
    "analyze_voronoi_coordination_metals_only",
    # properties & temporal
    "analyze_graph_properties",
    "analyze_temporal_graph_properties",
    "plot_temporal_graph_properties",
    # clusters
    "analyze_voronoi_clusters",
    # edge/bond summaries
    "summarize_voronoi_edge_network",
    "analyze_bond_network_via_ovito",
    # wrappers
    "plot_coordination_histograms",
    "plot_graph_structure",
    "plot_cluster_size_distribution",
    "plot_cluster_composition_analysis",
    "plot_3d_cluster_visualization",
    "plot_3d_cluster_with_graph",
    # helpers
    "extract_info",
]


