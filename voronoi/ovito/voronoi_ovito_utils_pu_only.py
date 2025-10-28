"""
OVITO-based Voronoi utilities for Pu-only clustering with mixed Na/Pu tessellation.

This module extends the base voronoi_ovito_utils.py to support:
- Voronoi tessellation using both Na and Pu atoms
- Cluster formation restricted to Pu atoms only
- Enhanced neighbor list tracking for Na-Pu interactions
- Specialized analysis functions for Pu clustering patterns

Key differences from base implementation:
- Mixed tessellation: Uses all atoms (Na, Pu, Cl) for Voronoi construction
- Pu-only clustering: Only Pu atoms participate in cluster formation
- Neighbor tracking: Tracks Na-Pu interactions separately
- Enhanced analysis: Specialized functions for Pu cluster analysis
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    raise ImportError("voronoi_ovito_utils_pu_only requires networkx to be installed") from exc

try:
    # OVITO 3.x Python API
    from ovito.modifiers import VoronoiAnalysisModifier
    from ovito.pipeline import Pipeline, PythonSource
    from ovito.data import Particles, SimulationCell
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "voronoi_ovito_utils_pu_only requires ovito to be installed (pip install ovito)"
    ) from exc

# Import base utilities
import sys
import os
sys.path.append('/pscratch/sd/p/pvashi/irp/irp_mace_l_2/irp/density/cluster_analysis/')
from voronoi_ovito_utils import (
    _build_ovito_pipeline_from_arrays,
    _compute_voronoi,
    _extract_bond_pairs_and_areas,
    _names_from_data_particles,
    analyze_graph_properties,
)

# Optional plotting utilities
try:
    from plot_utils import (
        plot_coordination_histograms as _plot_coordination_histograms,
        plot_graph_structure as _plot_graph_structure,
        plot_cluster_size_distribution as _plot_cluster_size_distribution,
        plot_cluster_composition_analysis as _plot_cluster_composition_analysis,
        plot_3d_cluster_visualization as _plot_3d_cluster_visualization,
        plot_3d_cluster_with_graph as _plot_3d_cluster_with_graph,
        analyze_bond_network as _analyze_bond_network,
    )
except Exception:
    _plot_coordination_histograms = None  # type: ignore
    _plot_graph_structure = None  # type: ignore
    _plot_cluster_size_distribution = None  # type: ignore
    _plot_cluster_composition_analysis = None  # type: ignore
    _plot_3d_cluster_visualization = None  # type: ignore
    _plot_3d_cluster_with_graph = None  # type: ignore
    _analyze_bond_network = None  # type: ignore


# -----------------------------
# Mixed tessellation with Pu-only clustering
# -----------------------------

def build_mixed_voronoi_graph_from_pipeline(
    pipeline: Any,
    frame: int = 0,
    min_area: float = 0.0,
    use_radii: bool = False,
    edge_threshold: float = 0.0,
) -> nx.Graph:
    """Build Voronoi graph using all atoms (Na, Pu, Cl) for tessellation.
    
    This creates a complete Voronoi tessellation but maintains species information
    for later filtering to Pu-only clusters.
    """
    # Ensure Voronoi bonds are generated for this frame
    voro = VoronoiAnalysisModifier(
        compute_indices=True,
        generate_bonds=True,
        use_radii=bool(use_radii),
        edge_threshold=float(edge_threshold),
    )
    pipeline.modifiers.append(voro)
    try:
        data = pipeline.compute(frame)
    finally:
        try:
            pipeline.modifiers.remove(voro)
        except Exception:
            pass
    
    parts = data.particles
    names = _names_from_data_particles(parts)
    pos = np.asarray(parts.positions)

    pairs, areas = _extract_bond_pairs_and_areas(data)
    mask = areas >= float(min_area)
    pairs = pairs[mask]
    areas = areas[mask]

    G = nx.Graph()
    for i, (p, spec) in enumerate(zip(pos, names)):
        G.add_node(int(i), position=np.asarray(p), species=str(spec), index=int(i))
    
    for (i, j), a in zip(pairs, areas):
        G.add_edge(int(i), int(j), area=float(a), species_pair=f"{names[int(i)]}-{names[int(j)]}")
    
    return G


def build_pu_only_cluster_graph_from_mixed(
    mixed_graph: nx.Graph,
    min_area: float = 0.0,
) -> nx.Graph:
    """Extract Pu-only cluster graph from mixed Voronoi tessellation.
    
    This function filters the mixed graph to only include Pu atoms and their
    connections, creating a Pu-only cluster network.
    """
    # Find all Pu nodes
    pu_nodes = [node for node in mixed_graph.nodes() 
                if mixed_graph.nodes[node].get("species") == "Pu"]
    
    if not pu_nodes:
        return nx.Graph()
    
    # Create subgraph with only Pu nodes
    pu_subgraph = mixed_graph.subgraph(pu_nodes).copy()
    
    # Filter edges by area threshold
    edges_to_remove = []
    for u, v, data in pu_subgraph.edges(data=True):
        if data.get("area", 0.0) < min_area:
            edges_to_remove.append((u, v))
    
    for u, v in edges_to_remove:
        pu_subgraph.remove_edge(u, v)
    
    return pu_subgraph


def analyze_na_pu_interactions_from_mixed(
    mixed_graph: nx.Graph,
    min_area: float = 0.0,
) -> Dict[str, Any]:
    """Analyze Na-Pu interactions from mixed Voronoi tessellation.
    
    Returns statistics about Na-Pu neighbor relationships and their
    facet areas.
    """
    na_pu_edges = []
    na_pu_areas = []
    
    for u, v, data in mixed_graph.edges(data=True):
        u_species = mixed_graph.nodes[u].get("species")
        v_species = mixed_graph.nodes[v].get("species")
        area = data.get("area", 0.0)
        
        # Check for Na-Pu interactions (both directions)
        if ((u_species == "Na" and v_species == "Pu") or 
            (u_species == "Pu" and v_species == "Na")):
            na_pu_edges.append((u, v))
            na_pu_areas.append(area)
    
    # Filter by area threshold
    na_pu_areas = np.array(na_pu_areas)
    mask = na_pu_areas >= min_area
    filtered_areas = na_pu_areas[mask]
    
    # Count Pu atoms with Na neighbors
    pu_with_na_neighbors = set()
    for u, v in na_pu_edges:
        u_species = mixed_graph.nodes[u].get("species")
        v_species = mixed_graph.nodes[v].get("species")
        if u_species == "Pu":
            pu_with_na_neighbors.add(u)
        elif v_species == "Pu":
            pu_with_na_neighbors.add(v)
    
    # Count Na atoms with Pu neighbors
    na_with_pu_neighbors = set()
    for u, v in na_pu_edges:
        u_species = mixed_graph.nodes[u].get("species")
        v_species = mixed_graph.nodes[v].get("species")
        if u_species == "Na":
            na_with_pu_neighbors.add(u)
        elif v_species == "Na":
            na_with_pu_neighbors.add(v)
    
    return {
        "num_na_pu_edges": len([a for a in na_pu_areas if a >= min_area]),
        "avg_na_pu_area": float(np.mean(filtered_areas)) if len(filtered_areas) > 0 else 0.0,
        "std_na_pu_area": float(np.std(filtered_areas)) if len(filtered_areas) > 0 else 0.0,
        "min_na_pu_area": float(np.min(filtered_areas)) if len(filtered_areas) > 0 else 0.0,
        "max_na_pu_area": float(np.max(filtered_areas)) if len(filtered_areas) > 0 else 0.0,
        "pu_atoms_with_na_neighbors": len(pu_with_na_neighbors),
        "na_atoms_with_pu_neighbors": len(na_with_pu_neighbors),
        "pu_with_na_fraction": len(pu_with_na_neighbors) / len([n for n in mixed_graph.nodes() 
                                                               if mixed_graph.nodes[n].get("species") == "Pu"]) if len([n for n in mixed_graph.nodes() if mixed_graph.nodes[n].get("species") == "Pu"]) > 0 else 0.0,
    }


def build_neighbor_list_from_mixed(
    mixed_graph: nx.Graph,
    min_area: float = 0.0,
) -> Dict[int, List[Tuple[int, str, float]]]:
    """Build comprehensive neighbor list from mixed Voronoi tessellation.
    
    Returns a dictionary mapping atom index to list of (neighbor_index, species, area).
    """
    neighbor_list = defaultdict(list)
    
    for u, v, data in mixed_graph.edges(data=True):
        area = data.get("area", 0.0)
        if area < min_area:
            continue
            
        u_species = mixed_graph.nodes[u].get("species")
        v_species = mixed_graph.nodes[v].get("species")
        
        neighbor_list[u].append((v, v_species, area))
        neighbor_list[v].append((u, u_species, area))
    
    return dict(neighbor_list)


def analyze_pu_cluster_properties_from_mixed(
    pipeline: Any,
    frame: int = 0,
    min_area: float = 0.0,
    use_radii: bool = False,
    edge_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Comprehensive analysis of Pu clusters using mixed tessellation approach.
    
    This function performs the complete analysis workflow:
    1. Build mixed Voronoi tessellation
    2. Extract Pu-only clusters
    3. Analyze Na-Pu interactions
    4. Return comprehensive statistics
    """
    # Build mixed tessellation
    mixed_graph = build_mixed_voronoi_graph_from_pipeline(
        pipeline, frame=frame, min_area=min_area, 
        use_radii=use_radii, edge_threshold=edge_threshold
    )
    
    # Extract Pu-only clusters
    pu_cluster_graph = build_pu_only_cluster_graph_from_mixed(mixed_graph, min_area)
    
    # Analyze Na-Pu interactions
    na_pu_stats = analyze_na_pu_interactions_from_mixed(mixed_graph, min_area)
    
    # Build neighbor list
    neighbor_list = build_neighbor_list_from_mixed(mixed_graph, min_area)
    
    # Analyze Pu cluster properties
    pu_cluster_props = analyze_graph_properties(pu_cluster_graph)
    
    # Get connected components (clusters)
    pu_components = list(nx.connected_components(pu_cluster_graph))
    cluster_sizes = [len(comp) for comp in pu_components]
    
    # Analyze Pu coordination in mixed environment
    pu_coordination_stats = analyze_pu_coordination_in_mixed(mixed_graph, min_area)
    
    return {
        "mixed_graph_properties": analyze_graph_properties(mixed_graph),
        "pu_cluster_properties": pu_cluster_props,
        "na_pu_interactions": na_pu_stats,
        "cluster_sizes": cluster_sizes,
        "num_clusters": len(pu_components),
        "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "avg_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "pu_coordination_stats": pu_coordination_stats,
        "neighbor_list": neighbor_list,
    }


def analyze_pu_coordination_in_mixed(
    mixed_graph: nx.Graph,
    min_area: float = 0.0,
) -> Dict[str, Any]:
    """Analyze Pu coordination patterns in mixed Na/Pu/Cl environment.
    
    Returns coordination statistics for Pu atoms considering all neighbor types.
    """
    pu_nodes = [node for node in mixed_graph.nodes() 
                if mixed_graph.nodes[node].get("species") == "Pu"]
    
    if not pu_nodes:
        return {}
    
    coordination_data = defaultdict(list)
    
    for pu_node in pu_nodes:
        neighbors = mixed_graph.neighbors(pu_node)
        neighbor_counts = defaultdict(int)
        
        for neighbor in neighbors:
            edge_data = mixed_graph.edges[pu_node, neighbor]
            area = edge_data.get("area", 0.0)
            if area >= min_area:
                neighbor_species = mixed_graph.nodes[neighbor].get("species")
                neighbor_counts[neighbor_species] += 1
        
        # Store coordination numbers for each neighbor type
        for species, count in neighbor_counts.items():
            coordination_data[f"Pu-{species}"].append(count)
    
    # Compute statistics
    stats = {}
    for species_pair, counts in coordination_data.items():
        if counts:
            stats[f"{species_pair}_mean"] = float(np.mean(counts))
            stats[f"{species_pair}_std"] = float(np.std(counts))
            stats[f"{species_pair}_min"] = int(np.min(counts))
            stats[f"{species_pair}_max"] = int(np.max(counts))
    
    return stats


def plot_mixed_tessellation_analysis(
    analysis_results: Dict[str, Any],
    title_prefix: str = "Mixed Tessellation Analysis"
) -> None:
    """Plot comprehensive analysis results from mixed tessellation approach."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{title_prefix} - Pu Clusters with Mixed Na/Pu Tessellation", fontsize=16)
    
    # Plot 1: Cluster size distribution
    cluster_sizes = analysis_results.get("cluster_sizes", [])
    if cluster_sizes:
        axes[0, 0].hist(cluster_sizes, bins=range(1, max(cluster_sizes) + 2), alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel("Cluster Size")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Pu Cluster Size Distribution")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Na-Pu interaction areas
    na_pu_stats = analysis_results.get("na_pu_interactions", {})
    if na_pu_stats.get("num_na_pu_edges", 0) > 0:
        # This would need actual area data - placeholder for now
        axes[0, 1].text(0.5, 0.5, f"Na-Pu Edges: {na_pu_stats['num_na_pu_edges']}\n"
                                   f"Avg Area: {na_pu_stats['avg_na_pu_area']:.3f}\n"
                                   f"Pu with Na: {na_pu_stats['pu_atoms_with_na_neighbors']}", 
                        ha='center', va='center', transform=axes[0, 1].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[0, 1].set_title("Na-Pu Interaction Summary")
    
    # Plot 3: Pu coordination statistics
    pu_coord_stats = analysis_results.get("pu_coordination_stats", {})
    if pu_coord_stats:
        species_pairs = [k.replace("_mean", "") for k in pu_coord_stats.keys() if k.endswith("_mean")]
        means = [pu_coord_stats[f"{sp}_mean"] for sp in species_pairs]
        
        axes[0, 2].bar(species_pairs, means, alpha=0.7)
        axes[0, 2].set_xlabel("Neighbor Species")
        axes[0, 2].set_ylabel("Average Coordination Number")
        axes[0, 2].set_title("Pu Coordination Statistics")
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Graph properties comparison
    mixed_props = analysis_results.get("mixed_graph_properties", {})
    pu_props = analysis_results.get("pu_cluster_properties", {})
    
    properties = ["num_nodes", "num_edges", "avg_degree"]
    mixed_values = [mixed_props.get(prop, 0) for prop in properties]
    pu_values = [pu_props.get(prop, 0) for prop in properties]
    
    x = np.arange(len(properties))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, mixed_values, width, label='Mixed Tessellation', alpha=0.7)
    axes[1, 0].bar(x + width/2, pu_values, width, label='Pu-only Clusters', alpha=0.7)
    axes[1, 0].set_xlabel("Property")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Graph Properties Comparison")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(properties)
    axes[1, 0].legend()
    
    # Plot 5: Cluster statistics
    cluster_stats = [
        analysis_results.get("num_clusters", 0),
        analysis_results.get("largest_cluster_size", 0),
        int(analysis_results.get("avg_cluster_size", 0))
    ]
    cluster_labels = ["Total Clusters", "Largest Cluster", "Avg Cluster Size"]
    
    axes[1, 1].bar(cluster_labels, cluster_stats, alpha=0.7, color=['blue', 'red', 'green'])
    axes[1, 1].set_ylabel("Count/Size")
    axes[1, 1].set_title("Cluster Statistics")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Summary text
    summary_text = f"""
    Analysis Summary:
    
    Mixed Tessellation:
    • Total atoms: {mixed_props.get('num_nodes', 0)}
    • Total edges: {mixed_props.get('num_edges', 0)}
    • Avg degree: {mixed_props.get('avg_degree', 0):.2f}
    
    Pu Clusters:
    • Number of clusters: {analysis_results.get('num_clusters', 0)}
    • Largest cluster: {analysis_results.get('largest_cluster_size', 0)}
    • Average size: {analysis_results.get('avg_cluster_size', 0):.2f}
    
    Na-Pu Interactions:
    • Na-Pu edges: {na_pu_stats.get('num_na_pu_edges', 0)}
    • Pu atoms with Na neighbors: {na_pu_stats.get('pu_atoms_with_na_neighbors', 0)}
    """
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    axes[1, 2].set_title("Summary")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


# -----------------------------
# Pipeline-based analysis functions
# -----------------------------

def analyze_mixed_tessellation_from_pipeline(
    pipeline: Any,
    frames: Optional[List[int]] = None,
    min_area: float = 0.0,
    use_radii: bool = False,
    edge_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Analyze mixed tessellation over multiple frames."""
    nframes = pipeline.source.num_frames if hasattr(pipeline, "source") else 1
    if frames is None:
        frames = list(range(nframes))
    
    results = {
        "frame_indices": frames,
        "mixed_graph_properties": [],
        "pu_cluster_properties": [],
        "na_pu_interactions": [],
        "cluster_statistics": [],
    }
    
    for frame in frames:
        frame_results = analyze_pu_cluster_properties_from_mixed(
            pipeline, frame=frame, min_area=min_area,
            use_radii=use_radii, edge_threshold=edge_threshold
        )
        
        results["mixed_graph_properties"].append(frame_results["mixed_graph_properties"])
        results["pu_cluster_properties"].append(frame_results["pu_cluster_properties"])
        results["na_pu_interactions"].append(frame_results["na_pu_interactions"])
        results["cluster_statistics"].append({
            "num_clusters": frame_results["num_clusters"],
            "largest_cluster_size": frame_results["largest_cluster_size"],
            "avg_cluster_size": frame_results["avg_cluster_size"],
        })
    
    return results


# -----------------------------
# Export functions
# -----------------------------

__all__ = [
    "build_mixed_voronoi_graph_from_pipeline",
    "build_pu_only_cluster_graph_from_mixed",
    "analyze_na_pu_interactions_from_mixed",
    "build_neighbor_list_from_mixed",
    "analyze_pu_cluster_properties_from_mixed",
    "analyze_pu_coordination_in_mixed",
    "plot_mixed_tessellation_analysis",
    "analyze_mixed_tessellation_from_pipeline",
]
