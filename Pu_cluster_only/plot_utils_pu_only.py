"""
Specialized plotting utilities for Pu-only cluster analysis with mixed tessellation.

This module extends the base plot_utils.py to support:
- Mixed tessellation visualization (Na/Pu/Cl)
- Pu-only cluster highlighting
- Na-Pu interaction analysis
- Enhanced 3D visualizations for Pu clustering patterns
"""

from pathlib import Path
from itertools import cycle
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx  # Optional; only needed for graph plotting
except Exception:  # pragma: no cover - optional dependency in some environments
    nx = None  # type: ignore

# Import base plotting functions
import sys
import os
sys.path.append('/pscratch/sd/p/pvashi/irp/irp_mace_l_2/irp/density/cluster_analysis/')
from plot_utils import (
    plot_coordination_histograms,
    plot_graph_structure,
    plot_cluster_size_distribution,
    plot_cluster_composition_analysis,
    plot_3d_cluster_visualization,
    plot_3d_cluster_with_graph,
    plot_3d_graph_components,
    analyze_bond_network,
    plot_rdfs,
)


__all__ = [
    "plot_mixed_tessellation_structure",
    "plot_pu_clusters_with_na_context",
    "plot_na_pu_interaction_network",
    "plot_pu_coordination_analysis",
    "plot_3d_mixed_tessellation",
    "plot_cluster_evolution_analysis",
    "plot_neighbor_list_analysis",
    "plot_3d_graph_components_pu_only",
]


def plot_mixed_tessellation_structure(
    mixed_graph: Any, 
    pu_cluster_graph: Any,
    title: str = "Mixed Tessellation Structure"
) -> None:
    """Plot the mixed tessellation structure with Pu clusters highlighted.
    
    Shows all atoms (Na, Pu, Cl) in the tessellation but highlights
    Pu clusters with different colors.
    """
    if nx is None:
        raise ImportError("networkx is required for plot_mixed_tessellation_structure")

    plt.figure(figsize=(15, 10))

    # Get 2D positions (x, y) from stored 3D positions
    pos = {}
    species_colors = {"Na": "blue", "Cl": "green", "Pu": "red"}
    
    # Get positions from mixed graph
    for node in mixed_graph.nodes():
        x, y, z = mixed_graph.nodes[node]["position"]
        pos[node] = (x, y)  # 2D projection

    # Draw all nodes first (background)
    for species in species_colors.keys():
        nodes_of_species = [n for n in mixed_graph.nodes() if mixed_graph.nodes[n]["species"] == species]
        if nodes_of_species:
            nx.draw_networkx_nodes(
                mixed_graph,
                pos,
                nodelist=nodes_of_species,
                node_color=species_colors[species],
                node_size=30,
                alpha=0.3,
                label=f"{species} (all)",
            )

    # Highlight Pu clusters with different colors
    pu_components = list(nx.connected_components(pu_cluster_graph))
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(pu_components)))
    
    for i, component in enumerate(pu_components):
        if component:
            nx.draw_networkx_nodes(
                mixed_graph,
                pos,
                nodelist=list(component),
                node_color=[cluster_colors[i]],
                node_size=100,
                alpha=0.8,
                label=f"Pu Cluster {i+1}",
            )

    # Draw edges for Pu clusters
    nx.draw_networkx_edges(pu_cluster_graph, pos, alpha=0.6, width=2, edge_color="red")

    # Draw all other edges (lighter)
    other_edges = [(u, v) for u, v in mixed_graph.edges() 
                   if not pu_cluster_graph.has_edge(u, v)]
    nx.draw_networkx_edges(mixed_graph, pos, edgelist=other_edges, alpha=0.2, width=0.5, edge_color="gray")

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_pu_clusters_with_na_context(
    data: Any, 
    cluster_ids: np.ndarray, 
    names: np.ndarray,
    max_clusters: int = 10
) -> None:
    """3D visualization of Pu clusters with Na atoms shown for context.
    
    Shows Pu clusters in different colors with Na atoms as smaller,
    semi-transparent spheres for spatial context.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Extract positions
    try:
        positions = data.particles["Position"]  # OVITO
    except Exception:
        try:
            positions = data.get_positions()  # ASE
        except Exception:
            positions = getattr(data, "positions", None)
            if positions is None:
                raise AttributeError("Could not extract positions from 'data' for 3D plotting.")
    positions = np.asarray(positions)

    # Get Pu clusters
    pu_mask = names == "Pu"
    pu_cluster_ids = cluster_ids[pu_mask]
    unique_pu_clusters, counts = np.unique(pu_cluster_ids[pu_cluster_ids >= 0], return_counts=True)
    
    if len(unique_pu_clusters) == 0:
        print("No Pu clusters to visualize")
        return

    largest_cluster_indices = np.argsort(counts)[-max_clusters:][::-1]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Pu Clusters with Na Context", fontsize=16, fontweight='bold')

    for i, cluster_idx in enumerate(largest_cluster_indices):
        cid = unique_pu_clusters[cluster_idx]
        
        ax = fig.add_subplot(2, 5, i + 1, projection="3d")

        # Plot Na atoms as context (smaller, semi-transparent)
        na_mask = names == "Na"
        if np.any(na_mask):
            ax.scatter(
                positions[na_mask, 0],
                positions[na_mask, 1],
                positions[na_mask, 2],
                c="lightblue",
                s=20,
                alpha=0.3,
                label="Na (context)",
            )

        # Plot Pu atoms in this cluster
        pu_cluster_mask = (names == "Pu") & (cluster_ids == cid)
        if np.any(pu_cluster_mask):
            ax.scatter(
                positions[pu_cluster_mask, 0],
                positions[pu_cluster_mask, 1],
                positions[pu_cluster_mask, 2],
                c="red",
                s=80,
                alpha=0.9,
                label=f"Pu Cluster {cid}",
                edgecolors='darkred',
                linewidth=0.5
            )

        # Plot other Pu atoms (not in this cluster) as smaller points
        other_pu_mask = (names == "Pu") & (cluster_ids != cid) & (cluster_ids >= 0)
        if np.any(other_pu_mask):
            ax.scatter(
                positions[other_pu_mask, 0],
                positions[other_pu_mask, 1],
                positions[other_pu_mask, 2],
                c="orange",
                s=30,
                alpha=0.5,
                label="Other Pu",
            )

        ax.set_title(f"Pu Cluster {cid}\nSize: {counts[cluster_idx]}")
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")

        if i == 0:  # legend once
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_na_pu_interaction_network(
    mixed_graph: Any,
    min_area: float = 0.0,
    title: str = "Na-Pu Interaction Network"
) -> None:
    """Plot the network of Na-Pu interactions from mixed tessellation."""
    if nx is None:
        raise ImportError("networkx is required for plot_na_pu_interaction_network")

    plt.figure(figsize=(12, 8))

    # Get positions
    pos = {}
    for node in mixed_graph.nodes():
        x, y, z = mixed_graph.nodes[node]["position"]
        pos[node] = (x, y)  # 2D projection

    # Separate nodes by species
    na_nodes = [n for n in mixed_graph.nodes() if mixed_graph.nodes[n]["species"] == "Na"]
    pu_nodes = [n for n in mixed_graph.nodes() if mixed_graph.nodes[n]["species"] == "Pu"]
    cl_nodes = [n for n in mixed_graph.nodes() if mixed_graph.nodes[n]["species"] == "Cl"]

    # Draw nodes by species
    if na_nodes:
        nx.draw_networkx_nodes(mixed_graph, pos, nodelist=na_nodes, 
                              node_color="blue", node_size=50, alpha=0.7, label="Na")
    if pu_nodes:
        nx.draw_networkx_nodes(mixed_graph, pos, nodelist=pu_nodes, 
                              node_color="red", node_size=80, alpha=0.8, label="Pu")
    if cl_nodes:
        nx.draw_networkx_nodes(mixed_graph, pos, nodelist=cl_nodes, 
                              node_color="green", node_size=30, alpha=0.5, label="Cl")

    # Draw edges with different styles
    na_pu_edges = []
    other_edges = []
    
    for u, v, data in mixed_graph.edges(data=True):
        area = data.get("area", 0.0)
        if area < min_area:
            continue
            
        u_species = mixed_graph.nodes[u]["species"]
        v_species = mixed_graph.nodes[v]["species"]
        
        if ((u_species == "Na" and v_species == "Pu") or 
            (u_species == "Pu" and v_species == "Na")):
            na_pu_edges.append((u, v))
        else:
            other_edges.append((u, v))

    # Draw Na-Pu edges prominently
    if na_pu_edges:
        nx.draw_networkx_edges(mixed_graph, pos, edgelist=na_pu_edges, 
                              alpha=0.8, width=3, edge_color="purple", label=f"Na-Pu ({len(na_pu_edges)})")

    # Draw other edges lightly
    if other_edges:
        nx.draw_networkx_edges(mixed_graph, pos, edgelist=other_edges, 
                              alpha=0.2, width=0.5, edge_color="gray")

    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_pu_coordination_analysis(
    coordination_stats: Dict[str, Any],
    title: str = "Pu Coordination Analysis"
) -> None:
    """Plot Pu coordination statistics in mixed environment."""
    if not coordination_stats:
        print("No coordination statistics to plot")
        return

    # Extract species pairs and their statistics
    species_pairs = []
    means = []
    stds = []
    
    for key, value in coordination_stats.items():
        if key.endswith("_mean"):
            species_pair = key.replace("_mean", "")
            species_pairs.append(species_pair)
            means.append(value)
            
            std_key = key.replace("_mean", "_std")
            stds.append(coordination_stats.get(std_key, 0))

    if not species_pairs:
        print("No coordination data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Mean coordination numbers
    bars1 = ax1.bar(species_pairs, means, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel("Neighbor Species")
    ax1.set_ylabel("Average Coordination Number")
    ax1.set_title("Mean Coordination Numbers")
    ax1.tick_params(axis='x', rotation=45)
    
    # Add error bars
    ax1.errorbar(range(len(species_pairs)), means, yerr=stds, 
                fmt='none', color='red', capsize=5)

    # Add value labels on bars
    for bar, mean, std in zip(bars1, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom')

    # Plot 2: Coordination distribution (if we had the raw data)
    ax2.text(0.5, 0.5, "Coordination Distribution\n(requires raw coordination data)", 
             ha='center', va='center', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax2.set_title("Coordination Distribution")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def plot_3d_mixed_tessellation(
    data: Any,
    mixed_graph: Any,
    pu_cluster_graph: Any,
    max_clusters: int = 6
) -> None:
    """3D visualization of mixed tessellation with Pu clusters highlighted."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Extract positions
    try:
        positions = data.particles["Position"]  # OVITO
    except Exception:
        try:
            positions = data.get_positions()  # ASE
        except Exception:
            positions = getattr(data, "positions", None)
            if positions is None:
                raise AttributeError("Could not extract positions from 'data' for 3D plotting.")
    positions = np.asarray(positions)

    # Get Pu clusters
    pu_components = list(nx.connected_components(pu_cluster_graph))
    pu_components = pu_components[:max_clusters]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("3D Mixed Tessellation with Pu Clusters", fontsize=16, fontweight='bold')

    n = len(pu_components)
    nrows, ncols = 2, 3
    nplots = max(n, 1)

    for i, component in enumerate(pu_components[:nplots]):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        
        # Plot all atoms as background (small, transparent)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='lightgray', s=10, alpha=0.3, label='All atoms')

        # Highlight Pu cluster
        if component:
            cluster_positions = []
            for node in component:
                if node < len(positions):
                    cluster_positions.append(positions[node])
            
            if cluster_positions:
                cluster_positions = np.vstack(cluster_positions)
                ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], cluster_positions[:, 2],
                          c='red', s=100, alpha=0.9, label=f'Pu Cluster {i+1}')

        ax.set_title(f"Pu Cluster {i+1}\nSize: {len(component)}")
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")

        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_cluster_evolution_analysis(
    temporal_results: Dict[str, Any],
    title: str = "Cluster Evolution Analysis"
) -> None:
    """Plot temporal evolution of cluster properties."""
    if not temporal_results:
        print("No temporal data to plot")
        return

    frame_indices = temporal_results.get("frame_indices", [])
    cluster_stats = temporal_results.get("cluster_statistics", [])
    
    if not cluster_stats:
        print("No cluster statistics found")
        return

    # Extract data
    num_clusters = [stat.get("num_clusters", 0) for stat in cluster_stats]
    largest_sizes = [stat.get("largest_cluster_size", 0) for stat in cluster_stats]
    avg_sizes = [stat.get("avg_cluster_size", 0) for stat in cluster_stats]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Number of clusters over time
    axes[0, 0].plot(frame_indices, num_clusters, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel("Frame Index")
    axes[0, 0].set_ylabel("Number of Clusters")
    axes[0, 0].set_title("Cluster Count Evolution")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Largest cluster size over time
    axes[0, 1].plot(frame_indices, largest_sizes, 'r-s', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel("Frame Index")
    axes[0, 1].set_ylabel("Largest Cluster Size")
    axes[0, 1].set_title("Largest Cluster Evolution")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Average cluster size over time
    axes[1, 0].plot(frame_indices, avg_sizes, 'g-^', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel("Frame Index")
    axes[1, 0].set_ylabel("Average Cluster Size")
    axes[1, 0].set_title("Average Cluster Size Evolution")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    summary_text = f"""
    Evolution Summary:
    
    Initial State:
    • Clusters: {num_clusters[0] if num_clusters else 0}
    • Largest: {largest_sizes[0] if largest_sizes else 0}
    • Average: {avg_sizes[0]:.2f if avg_sizes else 0:.2f}
    
    Final State:
    • Clusters: {num_clusters[-1] if num_clusters else 0}
    • Largest: {largest_sizes[-1] if largest_sizes else 0}
    • Average: {avg_sizes[-1]:.2f if avg_sizes else 0:.2f}
    
    Changes:
    • Δ Clusters: {num_clusters[-1] - num_clusters[0] if len(num_clusters) > 1 else 0}
    • Δ Largest: {largest_sizes[-1] - largest_sizes[0] if len(largest_sizes) > 1 else 0}
    • Δ Average: {avg_sizes[-1] - avg_sizes[0]:.2f if len(avg_sizes) > 1 else 0:.2f}
    """
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    axes[1, 1].set_title("Summary")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_neighbor_list_analysis(
    neighbor_list: Dict[int, List[Tuple[int, str, float]]],
    names: np.ndarray,
    title: str = "Neighbor List Analysis"
) -> None:
    """Plot analysis of neighbor list from mixed tessellation."""
    if not neighbor_list:
        print("No neighbor list data to plot")
        return

    # Analyze neighbor distributions
    pu_neighbors = defaultdict(list)
    na_neighbors = defaultdict(list)
    
    for atom_idx, neighbors in neighbor_list.items():
        if atom_idx >= len(names):
            continue
            
        atom_species = names[atom_idx]
        for neighbor_idx, neighbor_species, area in neighbors:
            if neighbor_idx >= len(names):
                continue
                
            if atom_species == "Pu":
                pu_neighbors[neighbor_species].append(area)
            elif atom_species == "Na":
                na_neighbors[neighbor_species].append(area)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Pu neighbor areas by species
    pu_species = list(pu_neighbors.keys())
    if pu_species:
        pu_data = [pu_neighbors[sp] for sp in pu_species]
        axes[0, 0].boxplot(pu_data, labels=pu_species)
        axes[0, 0].set_xlabel("Neighbor Species")
        axes[0, 0].set_ylabel("Facet Area")
        axes[0, 0].set_title("Pu Neighbor Facet Areas")
        axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Na neighbor areas by species
    na_species = list(na_neighbors.keys())
    if na_species:
        na_data = [na_neighbors[sp] for sp in na_species]
        axes[0, 1].boxplot(na_data, labels=na_species)
        axes[0, 1].set_xlabel("Neighbor Species")
        axes[0, 1].set_ylabel("Facet Area")
        axes[0, 1].set_title("Na Neighbor Facet Areas")
        axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot 3: Coordination number distributions
    pu_coord_counts = defaultdict(int)
    na_coord_counts = defaultdict(int)
    
    for atom_idx, neighbors in neighbor_list.items():
        if atom_idx >= len(names):
            continue
            
        atom_species = names[atom_idx]
        coord_count = len(neighbors)
        
        if atom_species == "Pu":
            pu_coord_counts[coord_count] += 1
        elif atom_species == "Na":
            na_coord_counts[coord_count] += 1

    if pu_coord_counts:
        coord_nums = list(pu_coord_counts.keys())
        coord_counts = list(pu_coord_counts.values())
        axes[1, 0].bar(coord_nums, coord_counts, alpha=0.7, color='red', label='Pu')
        axes[1, 0].set_xlabel("Coordination Number")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Pu Coordination Distribution")
        axes[1, 0].legend()

    if na_coord_counts:
        coord_nums = list(na_coord_counts.keys())
        coord_counts = list(na_coord_counts.values())
        axes[1, 1].bar(coord_nums, coord_counts, alpha=0.7, color='blue', label='Na')
        axes[1, 1].set_xlabel("Coordination Number")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Na Coordination Distribution")
    plt.tight_layout()
    plt.show()


def plot_3d_graph_components_pu_only(
    pu_cluster_graph: Any, 
    data: Any, 
    names: np.ndarray, 
    max_components: int = 10,
    show_na_context: bool = True
) -> None:
    """3D visualization of Pu-only graph components with optional Na context.
    
    This function visualizes the largest connected components of the Pu cluster graph,
    optionally showing Na atoms for spatial context.
    
    Args:
        pu_cluster_graph: NetworkX graph containing only Pu atoms and their connections
        data: OVITO data object containing particle positions
        names: Array of atom names/types
        max_components: Maximum number of components to display
        show_na_context: Whether to show Na atoms as background context
    """
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components_pu_only")

    # Local import to avoid global dependency in headless environments
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import

    if pu_cluster_graph.number_of_nodes() == 0:
        print("Pu cluster graph is empty")
        return

    # Get particle positions
    try:
        positions = data.particles.positions  # OVITO
    except Exception:
        try:
            positions = data.get_positions()  # ASE
        except Exception:
            positions = getattr(data, "positions", None)
            if positions is None:
                raise AttributeError("Could not extract positions from 'data' for 3D plotting.")
    positions = np.asarray(positions)
    
    # Ensure graph nodes have position and species data
    for node in pu_cluster_graph.nodes():
        if node < len(positions) and node < len(names):
            pu_cluster_graph.nodes[node]['position'] = positions[node]
            pu_cluster_graph.nodes[node]['species'] = names[node]

    # Connected components sorted by size (only Pu components)
    components = sorted(nx.connected_components(pu_cluster_graph), key=len, reverse=True)
    components = components[:max_components]

    n = len(components)
    nrows, ncols = 2, 5
    nplots = max(n, 1)

    fig = plt.figure(figsize=(20, 15))
    title = "Pu Metal Clusters (Pu-only connectivity)"
    if show_na_context:
        title += " with Na Context"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Color scheme for Pu atoms (different shades of red/orange)
    pu_colors = ["#FF0000", "#FF4500", "#FF6347", "#FF7F50", "#FFA500", 
                 "#FFB347", "#FFC0CB", "#FFD700", "#FFE4B5", "#FFF8DC"]

    def lw_from_area(a: float, a0: float, a1: float) -> float:
        if a1 <= a0:
            return 1.0
        t = (float(a) - a0) / (a1 - a0)
        return 0.6 + 2.4 * max(0.0, min(1.0, t))

    for i, nodes in enumerate(components):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        subG = pu_cluster_graph.subgraph(nodes)

        # Show Na atoms as background context if requested
        if show_na_context:
            na_mask = names == "Na"
            if np.any(na_mask):
                ax.scatter(
                    positions[na_mask, 0],
                    positions[na_mask, 1],
                    positions[na_mask, 2],
                    c="lightblue",
                    s=15,
                    alpha=0.3,
                    label="Na (context)" if i == 0 else "",
                )

        # Collect positions for Pu atoms in this component
        pu_positions = []
        for n in subG.nodes:
            pos = np.asarray(subG.nodes[n].get("position"))
            if pos is not None and pos.shape[0] == 3:
                pu_positions.append(pos)

        if not pu_positions:
            ax.set_title(f"Component {i} - No valid positions")
            continue

        # Plot Pu atoms in this component
        P = np.vstack(pu_positions)
        color = pu_colors[i % len(pu_colors)]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, s=80, alpha=0.9, 
                  label=f'Pu Cluster {i+1} (n={len(pu_positions)})', 
                  edgecolors='darkred', linewidth=0.5)

        # Edge linewidths by area when available
        areas = [edata.get("area", 1.0) for _, _, edata in subG.edges(data=True)]
        a_min = float(np.min(areas)) if len(areas) > 0 else 1.0
        a_max = float(np.max(areas)) if len(areas) > 0 else 1.0

        # Draw edges between Pu atoms
        for u, v, edata in subG.edges(data=True):
            p1 = np.asarray(subG.nodes[u].get("position"))
            p2 = np.asarray(subG.nodes[v].get("position"))
            if p1 is not None and p2 is not None:
                lw = lw_from_area(edata.get("area", 1.0), a_min, a_max)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       color="darkred", alpha=0.7, linewidth=lw)

        ax.set_title(f"Pu Component {i+1} (|V|={subG.number_of_nodes()}, |E|={subG.number_of_edges()})")
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        
        # Add legend for the first subplot
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()
