"""
Standardized plotting utilities for Pu cluster analysis notebooks.

This module provides consistent, well-styled plotting functions for:
- Coordination histograms
- Graph visualizations  
- Cluster size distributions
- 3D component visualizations (both general and Pu-only)

All functions follow consistent styling guidelines:
- Standard figure sizes and layouts
- Consistent color schemes
- Proper error handling
- Clear labels and legends
- Professional appearance suitable for publications
"""

from pathlib import Path
from itertools import cycle
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import networkx as nx  # Optional; only needed for graph plotting
except Exception:  # pragma: no cover - optional dependency in some environments
    nx = None  # type: ignore

# Import base plotting functions for fallback
import sys
import os
sys.path.append('/pscratch/sd/p/pvashi/irp/irp_mace_l_2/irp/density/cluster_analysis/')
from plot_utils import (
    plot_coordination_histograms as base_plot_coordination_histograms,
    plot_graph_structure as base_plot_graph_structure,
    plot_cluster_size_distribution as base_plot_cluster_size_distribution,
    plot_3d_graph_components as base_plot_3d_graph_components,
)

__all__ = [
    "plot_coordination_histograms",
    "plot_graph_structure", 
    "plot_cluster_size_distribution",
    "plot_3d_graph_components",
    "plot_3d_graph_components_pu_only",
    "setup_plot_style",
    "get_standard_colors",
    "extract_positions_from_data",
    "extract_names_from_data",
]

# Standard styling configuration
PLOT_CONFIG = {
    'figure_size_large': (15, 10),
    'figure_size_medium': (12, 8), 
    'figure_size_small': (10, 6),
    'figure_size_3d': (20, 15),
    'dpi': 100,
    'font_size_title': 16,
    'font_size_labels': 12,
    'font_size_legend': 10,
    'line_width': 2,
    'marker_size': 6,
    'alpha_main': 0.8,
    'alpha_background': 0.3,
    'grid_alpha': 0.3,
}

# Standard color schemes
COLORS = {
    'species': {
        'Na': '#1f77b4',  # Blue
        'Pu': '#d62728',  # Red  
        'Cl': '#2ca02c',  # Green
        'unknown': '#7f7f7f'  # Gray
    },
    'clusters': [
        '#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896'
    ],
    'coordination': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'background': '#f0f0f0',
    'grid': '#cccccc'
}

def setup_plot_style():
    """Set up consistent matplotlib styling."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': PLOT_CONFIG['dpi'],
        'font.size': PLOT_CONFIG['font_size_labels'],
        'axes.titlesize': PLOT_CONFIG['font_size_title'],
        'axes.labelsize': PLOT_CONFIG['font_size_labels'],
        'xtick.labelsize': PLOT_CONFIG['font_size_labels'],
        'ytick.labelsize': PLOT_CONFIG['font_size_labels'],
        'legend.fontsize': PLOT_CONFIG['font_size_legend'],
        'lines.linewidth': PLOT_CONFIG['line_width'],
        'lines.markersize': PLOT_CONFIG['marker_size'],
        'grid.alpha': PLOT_CONFIG['grid_alpha'],
        'axes.grid': True,
        'axes.facecolor': COLORS['background'],
        'figure.facecolor': 'white',
    })

def get_standard_colors():
    """Get the standard color scheme."""
    return COLORS

def extract_positions_from_data(data: Any) -> np.ndarray:
    """Extract positions from various data formats (OVITO, ASE, etc.)."""
    try:
        # Try OVITO format first
        positions = data.particles.positions
    except Exception:
        try:
            # Try ASE format
            positions = data.get_positions()
        except Exception:
            try:
                # Try direct attribute access
                positions = getattr(data, "positions", None)
            except Exception:
                positions = None
    
    if positions is None:
        raise ValueError("Could not extract positions from data object")
    
    return np.asarray(positions)

def extract_names_from_data(data: Any) -> np.ndarray:
    """Extract atom names/types from various data formats."""
    try:
        # Try OVITO format first
        names = data.particles["Particle Type"]
        # Convert numeric types to string names if needed
        if hasattr(names, 'dtype') and np.issubdtype(names.dtype, np.integer):
            # Map numeric types to species names (common OVITO convention)
            type_map = {1: "Na", 2: "Pu", 3: "Cl"}  # Adjust as needed
            names = np.array([type_map.get(t, f"Type{t}") for t in names])
    except Exception:
        try:
            # Try ASE format
            names = np.array(data.get_chemical_symbols())
        except Exception:
            try:
                # Try direct attribute access
                names = getattr(data, "names", None)
                if names is not None:
                    names = np.asarray(names)
            except Exception:
                names = None
    
    if names is None:
        raise ValueError("Could not extract atom names from data object")
    
    return np.asarray(names)

def plot_coordination_histograms(
    coord_data: Dict[str, Dict[str, List[int]]], 
    central_type: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot standardized coordination histograms for a given central atom type.
    
    Args:
        coord_data: Mapping of central species -> neighbor species -> list of counts
        central_type: Central atom type (e.g., "Pu", "Na")
        title: Optional custom title
        save_path: Optional path to save the plot
    """
    if not coord_data or central_type not in coord_data:
        print(f"No coordination data found for {central_type}")
        return
    
    setup_plot_style()
    
    plt.figure(figsize=PLOT_CONFIG['figure_size_medium'])
    
    colors = COLORS['coordination']
    species = sorted(coord_data[central_type].keys())
    
    if not species:
        print(f"No neighbor species found for {central_type}")
        return
    
    # Plot histograms for each neighbor species
    for idx, neighbor_type in enumerate(species):
        data = coord_data[central_type][neighbor_type]
        if not data:
            continue
            
        mean = np.mean(data)
        std = np.std(data)
        color = colors[idx % len(colors)]
        
        # Create histogram
        plt.hist(
            data,
            bins=range(0, max(data) + 2),
            alpha=PLOT_CONFIG['alpha_main'],
            label=f"{neighbor_type} (μ={mean:.2f}±{std:.2f})",
            color=color,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add mean line
        plt.axvline(mean, color=color, linestyle="--", linewidth=2, alpha=0.8)
    
    # Customize plot
    plot_title = title or f"Coordination Histogram: {central_type}"
    plt.title(plot_title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    plt.xlabel("Coordination Number", fontsize=PLOT_CONFIG['font_size_labels'])
    plt.ylabel("Frequency", fontsize=PLOT_CONFIG['font_size_labels'])
    plt.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    plt.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
    
    # Add summary statistics box
    total_atoms = sum(len(data) for data in coord_data[central_type].values())
    stats_text = f"Total {central_type} atoms: {total_atoms}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=PLOT_CONFIG['font_size_legend'],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()

def plot_graph_structure(
    G: Any, 
    title: str = "Graph Structure",
    show_edges: bool = True,
    show_labels: bool = False,
    save_path: Optional[str] = None
) -> None:
    """Plot standardized graph structure with species-colored nodes.
    
    Args:
        G: NetworkX graph with node attributes 'position' and 'species'
        title: Plot title
        show_edges: Whether to show edges
        show_labels: Whether to show node labels
        save_path: Optional path to save the plot
    """
    if nx is None:
        raise ImportError("networkx is required for plot_graph_structure")
    
    if G.number_of_nodes() == 0:
        print("Graph is empty")
        return
    
    setup_plot_style()
    plt.figure(figsize=PLOT_CONFIG['figure_size_large'])
    
    # Extract 2D positions (x, y projection)
    pos = {}
    species_colors = COLORS['species']
    
    for node in G.nodes():
        try:
            x, y, z = G.nodes[node]["position"]
            pos[node] = (x, y)  # 2D projection
        except (KeyError, ValueError, TypeError):
            print(f"Warning: Invalid position data for node {node}")
            continue
    
    if not pos:
        print("No valid positions found in graph")
        return
    
    # Draw nodes by species
    species_counts = {}
    for species in species_colors.keys():
        nodes_of_species = [n for n in G.nodes() 
                           if G.nodes[n].get("species") == species]
        if nodes_of_species:
            species_counts[species] = len(nodes_of_species)
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes_of_species,
                node_color=species_colors[species],
                node_size=100,
                alpha=PLOT_CONFIG['alpha_main'],
                label=f"{species} ({len(nodes_of_species)})",
            )
    
    # Draw edges
    if show_edges and G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G, pos, 
            alpha=0.4, 
            width=1.0, 
            edge_color="gray"
        )
    
    # Draw labels if requested
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Customize plot
    plt.title(title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    plt.legend(fontsize=PLOT_CONFIG['font_size_legend'], loc='upper right')
    plt.axis("off")
    
    # Add summary statistics
    stats_text = f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
             fontsize=PLOT_CONFIG['font_size_legend'],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()

def plot_cluster_size_distribution(
    sizes: List[int], 
    title: str = "Cluster Size Distribution",
    bins: Optional[List[float]] = None,
    log_scale: bool = False,
    save_path: Optional[str] = None
) -> None:
    """Plot standardized cluster size distribution histogram.
    
    Args:
        sizes: List of cluster sizes
        title: Plot title
        bins: Optional custom bins for histogram
        log_scale: Whether to use log scale on y-axis
        save_path: Optional path to save the plot
    """
    if len(sizes) == 0:
        print("No clusters to plot")
        return
    
    setup_plot_style()
    plt.figure(figsize=PLOT_CONFIG['figure_size_medium'])
    
    # Prepare data
    sizes_array = np.array(sizes)
    max_size = np.max(sizes_array)
    
    # Set bins
    if bins is None:
        bins = np.arange(0, max_size + 2) - 0.5
    
    # Create histogram
    plt.hist(sizes, bins=bins, alpha=PLOT_CONFIG['alpha_main'], 
             edgecolor='black', linewidth=0.5, color=COLORS['species']['Pu'])
    
    # Set scale
    if log_scale:
        plt.yscale('log')
    
    # Customize plot
    plt.title(title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    plt.xlabel("Cluster Size", fontsize=PLOT_CONFIG['font_size_labels'])
    plt.ylabel("Frequency", fontsize=PLOT_CONFIG['font_size_labels'])
    plt.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
    
    # Add statistics
    mean_size = np.mean(sizes_array)
    std_size = np.std(sizes_array)
    median_size = np.median(sizes_array)
    
    stats_text = (f"Total clusters: {len(sizes)}\n"
                 f"Largest cluster: {max_size}\n"
                 f"Mean size: {mean_size:.2f}±{std_size:.2f}\n"
                 f"Median size: {median_size:.1f}")
    
    plt.text(0.7, 0.9, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=PLOT_CONFIG['font_size_legend'],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()

def plot_3d_graph_components(
    G: Any, 
    max_components: int = 6,
    title: str = "3D Graph Components",
    show_edges: bool = True,
    save_path: Optional[str] = None
) -> None:
    """Plot standardized 3D visualization of graph components.
    
    Args:
        G: NetworkX graph with node attributes 'position' and 'species'
        max_components: Maximum number of components to display
        title: Plot title
        show_edges: Whether to show edges
        save_path: Optional path to save the plot
    """
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components")
    
    if G.number_of_nodes() == 0:
        print("Graph is empty")
        return
    
    setup_plot_style()
    
    # Get connected components
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    components = components[:max_components]
    
    if not components:
        print("No connected components found")
        return
    
    # Create subplot layout
    n = len(components)
    nrows = min(2, n)
    ncols = min(5, n)
    
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size_3d'])
    fig.suptitle(title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    
    species_colors = COLORS['species']
    cluster_colors = COLORS['clusters']
    
    def lw_from_area(a: float, a0: float, a1: float) -> float:
        """Convert area to line width."""
        if a1 <= a0:
            return 1.0
        t = (float(a) - a0) / (a1 - a0)
        return 0.6 + 2.4 * max(0.0, min(1.0, t))
    
    for i, nodes in enumerate(components):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        subG = G.subgraph(nodes)
        
        # Collect positions by species
        pos_by_species: Dict[str, List[np.ndarray]] = {}
        for n in subG.nodes:
            try:
                pos = np.asarray(subG.nodes[n].get("position"))
                if pos is None or pos.shape[0] != 3:
                    continue
                sp = str(subG.nodes[n].get("species", "unknown"))
                pos_by_species.setdefault(sp, []).append(pos)
            except (KeyError, ValueError, TypeError):
                continue
        
        # Plot atoms by species
        for sp, arrs in pos_by_species.items():
            if not arrs:
                continue
            P = np.vstack(arrs)
            color = species_colors.get(sp, COLORS['species']['unknown'])
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], 
                      c=color, s=80, alpha=PLOT_CONFIG['alpha_main'], 
                      label=sp if i == 0 else "")
        
        # Draw edges if requested
        if show_edges and subG.number_of_edges() > 0:
            areas = [edata.get("area", 1.0) for _, _, edata in subG.edges(data=True)]
            a_min = float(np.min(areas)) if len(areas) > 0 else 1.0
            a_max = float(np.max(areas)) if len(areas) > 0 else 1.0
            
            for u, v, edata in subG.edges(data=True):
                try:
                    p1 = np.asarray(subG.nodes[u].get("position"))
                    p2 = np.asarray(subG.nodes[v].get("position"))
                    if p1 is None or p2 is None:
                        continue
                    lw = lw_from_area(edata.get("area", 1.0), a_min, a_max)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                           color="gray", alpha=0.6, linewidth=lw)
                except (KeyError, ValueError, TypeError):
                    continue
        
        # Customize subplot
        ax.set_title(f"Component {i+1}\n(|V|={subG.number_of_nodes()}, |E|={subG.number_of_edges()})",
                    fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_xlabel("X (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_ylabel("Y (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_zlabel("Z (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        
        if i == 0 and pos_by_species:
            ax.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()

def plot_3d_graph_components_pu_only(
    pu_cluster_graph: Any, 
    data: Any, 
    names: np.ndarray, 
    max_components: int = 6,
    show_na_context: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot standardized 3D visualization of Pu-only graph components.
    
    Args:
        pu_cluster_graph: NetworkX graph containing only Pu atoms
        data: Data object with particle positions
        names: Array of atom names/types
        max_components: Maximum number of components to display
        show_na_context: Whether to show Na atoms as background
        title: Optional custom title
        save_path: Optional path to save the plot
    """
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components_pu_only")
    
    if pu_cluster_graph.number_of_nodes() == 0:
        print("Pu cluster graph is empty")
        return
    
    setup_plot_style()
    
    # Extract positions
    try:
        positions = extract_positions_from_data(data)
    except ValueError as e:
        print(f"Error extracting positions: {e}")
        return
    
    # Ensure graph nodes have position and species data
    for node in pu_cluster_graph.nodes():
        if node < len(positions) and node < len(names):
            pu_cluster_graph.nodes[node]['position'] = positions[node]
            pu_cluster_graph.nodes[node]['species'] = names[node]
    
    # Get connected components
    components = sorted(nx.connected_components(pu_cluster_graph), key=len, reverse=True)
    components = components[:max_components]
    
    if not components:
        print("No Pu clusters found")
        return
    
    # Create subplot layout
    n = len(components)
    nrows = min(2, n)
    ncols = min(5, n)
    
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size_3d'])
    
    plot_title = title or "Pu Metal Clusters (Pu-only connectivity)"
    if show_na_context:
        plot_title += " with Na Context"
    fig.suptitle(plot_title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    
    # Color scheme for Pu clusters
    pu_colors = COLORS['clusters']
    
    def lw_from_area(a: float, a0: float, a1: float) -> float:
        """Convert area to line width."""
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
                    c=COLORS['species']['Na'],
                    s=20,
                    alpha=PLOT_CONFIG['alpha_background'],
                    label="Na (context)" if i == 0 else "",
                )
        
        # Collect positions for Pu atoms in this component
        pu_positions = []
        for n in subG.nodes:
            try:
                pos = np.asarray(subG.nodes[n].get("position"))
                if pos is not None and pos.shape[0] == 3:
                    pu_positions.append(pos)
            except (KeyError, ValueError, TypeError):
                continue
        
        if not pu_positions:
            ax.set_title(f"Component {i+1} - No valid positions")
            continue
        
        # Plot Pu atoms in this component
        P = np.vstack(pu_positions)
        color = pu_colors[i % len(pu_colors)]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], 
                  c=color, s=100, alpha=PLOT_CONFIG['alpha_main'], 
                  label=f'Pu Cluster {i+1} (n={len(pu_positions)})', 
                  edgecolors='darkred', linewidth=0.5)
        
        # Draw edges between Pu atoms
        if subG.number_of_edges() > 0:
            areas = [edata.get("area", 1.0) for _, _, edata in subG.edges(data=True)]
            a_min = float(np.min(areas)) if len(areas) > 0 else 1.0
            a_max = float(np.max(areas)) if len(areas) > 0 else 1.0
            
            for u, v, edata in subG.edges(data=True):
                try:
                    p1 = np.asarray(subG.nodes[u].get("position"))
                    p2 = np.asarray(subG.nodes[v].get("position"))
                    if p1 is not None and p2 is not None:
                        lw = lw_from_area(edata.get("area", 1.0), a_min, a_max)
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                               color="darkred", alpha=0.7, linewidth=lw)
                except (KeyError, ValueError, TypeError):
                    continue
        
        # Customize subplot
        ax.set_title(f"Pu Component {i+1}\n(|V|={subG.number_of_nodes()}, |E|={subG.number_of_edges()})",
                    fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_xlabel("X (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_ylabel("Y (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_zlabel("Z (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        
        # Add legend for the first subplot
        if i == 0:
            ax.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()

# Initialize plot style when module is imported
setup_plot_style()
