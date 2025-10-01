"""
Centralized plotting utilities for cluster_analysis notebooks.

Functions here consolidate plotting logic that previously lived inside notebooks.
Import from this module instead of redefining plotting functions in notebooks.
"""

from pathlib import Path
from itertools import cycle
from collections import Counter
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx  # Optional; only needed for graph plotting
except Exception:  # pragma: no cover - optional dependency in some environments
    nx = None  # type: ignore


__all__ = [
    "plot_coordination_histograms",
    "plot_graph_structure",
    "plot_cluster_size_distribution",
    "plot_cluster_composition_analysis",
    "plot_3d_cluster_visualization",
    "plot_3d_cluster_with_graph",
    "plot_3d_graph_components",
    "plot_3d_graph_components_pu_only",
    "analyze_bond_network",
    "plot_rdfs",
]


def plot_coordination_histograms(coord_data: Dict[str, Dict[str, List[int]]], central_type: str) -> None:
    """Plot histograms of coordination numbers for a given central atom type.

    Parameters:
        coord_data: Mapping of central species -> neighbor species -> list of counts
        central_type: e.g., "Na"
    """
    plt.figure(figsize=(10, 6))
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
    species = sorted(coord_data[central_type].keys())

    for idx, neighbor_type in enumerate(species):
        data = coord_data[central_type][neighbor_type]
        if not data:
            continue
        mean = np.mean(data)
        # std is computed but not shown beyond legend; keep for completeness
        std = np.std(data)

        plt.hist(
            data,
            bins=range(0, max(data) + 2),
            alpha=0.5,
            label=f"{neighbor_type} (μ={mean:.2f}, σ={std:.2f})",
            color=colors[idx % len(colors)],
        )
        plt.axvline(mean, color=colors[idx % len(colors)], linestyle="--", linewidth=2)

    plt.title(f"Voronoi Coordination Histogram for {central_type}")
    plt.xlabel("Coordination Number")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_graph_structure(G: Any, title: str = "Voronoi Graph Structure") -> None:
    """Plot the graph structure with species-colored nodes.

    Expects NetworkX-like graph G with node attributes:
      - 'position': iterable (x, y, z)
      - 'species': str like 'Na', 'Cl', 'Pu'
    """
    if nx is None:
        raise ImportError("networkx is required for plot_graph_structure")

    plt.figure(figsize=(12, 8))

    # Get 2D positions (x, y) from stored 3D positions
    pos = {}
    species_colors = {"Na": "blue", "Cl": "green", "Pu": "red"}

    for node in G.nodes():
        x, y, z = G.nodes[node]["position"]
        pos[node] = (x, y)  # 2D projection

    # Draw nodes colored by species
    for species in species_colors.keys():
        nodes_of_species = [n for n in G.nodes() if G.nodes[n]["species"] == species]
        if nodes_of_species:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes_of_species,
                node_color=species_colors[species],
                node_size=50,
                alpha=0.7,
                label=species,
            )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color="black")

    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_cluster_size_distribution(sizes: List[int], title: str = "Cluster Size Distribution") -> None:
    """Plot histogram of cluster sizes."""
    if len(sizes) == 0:
        print("No clusters to plot")
        return

    plt.figure(figsize=(10, 6))

    bins = np.arange(0, max(sizes) + 2) - 0.5
    plt.hist(sizes, bins=bins, alpha=0.7, edgecolor="black")

    plt.xlabel("Cluster Size")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add statistics
    plt.text(
        0.7,
        0.9,
        f"Total clusters: {len(sizes)}\nLargest cluster: {max(sizes)}\nMean size: {np.mean(sizes):.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )

    plt.tight_layout()
    plt.show()


def plot_cluster_composition_analysis(data: Any, cluster_ids: np.ndarray, names: np.ndarray) -> List[dict]:
    """Analyze and plot cluster composition (Pu vs Na).

    Returns a list of composition dicts per cluster.
    """
    is_pu = names == "Pu"
    is_na = names == "Na"

    # Get unique cluster IDs (excluding -1 for unclustered)
    unique_clusters = np.unique(cluster_ids[cluster_ids >= 0])

    cluster_compositions: List[dict] = []
    for cid in unique_clusters:
        cluster_mask = cluster_ids == cid
        pu_count = int(np.sum(is_pu & cluster_mask))
        na_count = int(np.sum(is_na & cluster_mask))
        total = pu_count + na_count

        cluster_compositions.append(
            {
                "cluster_id": int(cid),
                "size": int(total),
                "pu_count": pu_count,
                "na_count": na_count,
                "pu_fraction": (pu_count / total) if total > 0 else 0.0,
                "na_fraction": (na_count / total) if total > 0 else 0.0,
            }
        )

    # Convert to arrays for plotting
    sizes = [c["size"] for c in cluster_compositions]
    pu_fractions = [c["pu_fraction"] for c in cluster_compositions]
    na_fractions = [c["na_fraction"] for c in cluster_compositions]

    # Create composition plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Pu fraction vs cluster size
    ax1.scatter(sizes, pu_fractions, alpha=0.6, s=50)
    ax1.set_xlabel("Cluster Size")
    ax1.set_ylabel("Pu Fraction")
    ax1.set_title("Pu Fraction vs Cluster Size")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Na fraction vs cluster size
    ax2.scatter(sizes, na_fractions, alpha=0.6, s=50, color="orange")
    ax2.set_xlabel("Cluster Size")
    ax2.set_ylabel("Na Fraction")
    ax2.set_title("Na Fraction vs Cluster Size")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()

    # Print composition statistics
    print("Cluster Composition Analysis:")
    print(f"Total clusters: {len(cluster_compositions)}")
    print(f"Pure Pu clusters: {sum(1 for c in cluster_compositions if c['pu_fraction'] == 1.0)}")
    print(f"Pure Na clusters: {sum(1 for c in cluster_compositions if c['na_fraction'] == 1.0)}")
    print(
        f"Mixed clusters: {sum(1 for c in cluster_compositions if 0 < c['pu_fraction'] < 1.0)}"
    )

    return cluster_compositions


def plot_3d_cluster_visualization(
    data: Any, cluster_ids: np.ndarray, names: np.ndarray, max_clusters: int = 10
) -> None:
    """Create 3D visualization of largest clusters (Pu vs Na colored)."""
    # Local import to avoid global dependency in headless environments
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import

    # Accept both OVITO DataCollection-like objects (with .particles)
    # and ASE Atoms-like objects (with .get_positions()).
    try:
        positions = data.particles["Position"]  # OVITO
    except Exception:
        try:
            positions = data.get_positions()  # ASE
        except Exception:
            # Fallbacks: try common attribute names or mapping styles
            positions = getattr(data, "positions", None)
            if positions is None and hasattr(data, "particles"):
                # Some objects expose positions under a generic key
                part = getattr(data, "particles")
                try:
                    positions = part["position"]
                except Exception:
                    pass
            if positions is None:
                raise AttributeError("Could not extract positions from 'data' for 3D plotting.")
    positions = np.asarray(positions)

    unique_clusters, counts = np.unique(cluster_ids[cluster_ids >= 0], return_counts=True)
    if len(unique_clusters) == 0:
        print("No clusters to visualize")
        return

    largest_cluster_indices = np.argsort(counts)[-max_clusters:][::-1]

    fig = plt.figure(figsize=(20, 15))

    for i, cluster_idx in enumerate(largest_cluster_indices):
        cid = unique_clusters[cluster_idx]
        cluster_mask = cluster_ids == cid

        ax = fig.add_subplot(2, 5, i + 1, projection="3d")

        # Plot Pu atoms in red
        pu_mask = cluster_mask & (names == "Pu")
        if np.any(pu_mask):
            ax.scatter(
                positions[pu_mask, 0],
                positions[pu_mask, 1],
                positions[pu_mask, 2],
                c="red",
                s=50,
                alpha=0.7,
                label="Pu",
            )

        # Plot Na atoms in blue
        na_mask = cluster_mask & (names == "Na")
        if np.any(na_mask):
            ax.scatter(
                positions[na_mask, 0],
                positions[na_mask, 1],
                positions[na_mask, 2],
                c="blue",
                s=50,
                alpha=0.7,
                label="Na",
            )

        ax.set_title(f"Cluster {cid}\nSize: {counts[cluster_idx]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if i == 0:  # legend once
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_3d_cluster_with_graph(data, cluster_ids, names, G, max_clusters):
    """3D cluster visualization drawing nodes and edges from graph G.

    Nodes are colored by species (Pu red, Na blue). Edges are drawn only when
    both endpoints lie in the displayed cluster. If graph nodes are missing
    position/species attributes, the function falls back to `data`/`names`.
    """
    # Local import to avoid global dependency in headless environments
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import

    if G is None:
        raise ValueError("A NetworkX graph 'G' must be provided to draw edges.")

    # Optional fallback source for positions if not stored on graph nodes
    positions_src = None
    try:
        positions_src = data.particles["Position"]  # OVITO
    except Exception:
        try:
            positions_src = data.get_positions()  # ASE
        except Exception:
            positions_src = getattr(data, "positions", None)

    # Determine clusters to display (largest first)
    unique_clusters, counts = np.unique(cluster_ids[cluster_ids >= 0], return_counts=True)
    if len(unique_clusters) == 0:
        print("No clusters to visualize")
        return
    largest_cluster_indices = np.argsort(counts)[-max_clusters:][::-1]

    # Prepare figure layout
    n_to_show = min(len(largest_cluster_indices), max_clusters)
    nrows, ncols = 2, 5
    n_plots = max(n_to_show, 1)

    fig = plt.figure(figsize=(20, 15))

    # Helper mappings
    def node_to_atom_index(n):
        idx = G.nodes[n].get("index", None)
        if idx is not None:
            try:
                return int(idx)
            except Exception:
                return None
        try:
            return int(n)
        except Exception:
            return None

    def get_node_position(n):
        pos = G.nodes[n].get("position", None)
        if pos is not None:
            return np.asarray(pos)
        ai = node_to_atom_index(n)
        if positions_src is not None and ai is not None and 0 <= ai < len(positions_src):
            return np.asarray(positions_src[ai])
        return None

    def get_node_species(n):
        sp = G.nodes[n].get("species", None)
        if sp is not None:
            return str(sp)
        ai = node_to_atom_index(n)
        if ai is not None and 0 <= ai < len(names):
            return str(names[ai])
        return "unknown"

    for i, cluster_idx in enumerate(largest_cluster_indices[:n_plots]):
        cid = unique_clusters[cluster_idx]
        cluster_mask = cluster_ids == cid
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")

        # Select nodes that belong to this cluster (by atom index)
        cluster_atom_indices = set(np.where(cluster_mask)[0].tolist())
        cluster_node_ids = []
        for n in G.nodes:
            ai = node_to_atom_index(n)
            if ai is not None and ai in cluster_atom_indices:
                cluster_node_ids.append(n)

        # Scatter nodes by species
        if cluster_node_ids:
            pu_positions = []
            na_positions = []
            for n in cluster_node_ids:
                pos = get_node_position(n)
                if pos is None:
                    continue
                sp = get_node_species(n)
                if sp == "Pu":
                    pu_positions.append(pos)
                elif sp == "Na":
                    na_positions.append(pos)

            if len(pu_positions) > 0:
                pu_positions = np.vstack(pu_positions)
                ax.scatter(pu_positions[:, 0], pu_positions[:, 1], pu_positions[:, 2], c="red", s=50, alpha=0.8, label="Pu")
            if len(na_positions) > 0:
                na_positions = np.vstack(na_positions)
                ax.scatter(na_positions[:, 0], na_positions[:, 1], na_positions[:, 2], c="blue", s=50, alpha=0.8, label="Na")

            # Draw edges where both endpoints lie in this cluster
            subG = G.subgraph(cluster_node_ids)
            if subG.number_of_edges() == 0:
                print(f"Cluster {cid}: no edges to draw")
            else:
                # linewidth mapped by facet area if available
                areas = [edata.get("area", 1.0) for _, _, edata in subG.edges(data=True)]
                a_min = float(np.min(areas)) if len(areas) > 0 else 1.0
                a_max = float(np.max(areas)) if len(areas) > 0 else 1.0
                def lw_from_area(a, a0=a_min, a1=a_max):
                    if a1 <= a0:
                        return 1.0
                    t = (float(a) - a0) / (a1 - a0)
                    return 0.6 + 2.4 * max(0.0, min(1.0, t))

                for u, v, edata in subG.edges(data=True):
                    p1 = get_node_position(u)
                    p2 = get_node_position(v)
                    if p1 is None or p2 is None:
                        continue
                    lw = lw_from_area(edata.get("area", 1.0))
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="black", alpha=0.6, linewidth=lw)

        ax.set_title(f"Cluster {cid}\nSize: {counts[cluster_idx]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_3d_graph_components(G: Any, max_components: int = 10) -> None:
    """3D visualization of the largest connected components of a graph.

    Uses only the graph `G`. Nodes must have `position` (3-vector) and
    optionally `species` (e.g., "Na", "Pu"). Edges are drawn with linewidth
    mapped to edge "area" when present.
    """
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components")

    # Local import to avoid global dependency in headless environments
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import

    if G.number_of_nodes() == 0:
        print("Graph is empty")
        return

    # Connected components sorted by size
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    components = components[:max_components]

    n = len(components)
    nrows, ncols = 2, 5
    nplots = max(n, 1)

    fig = plt.figure(figsize=(20, 15))

    species_colors = {"Na": "blue", "Pu": "red", "Cl": "green"}

    def lw_from_area(a: float, a0: float, a1: float) -> float:
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
            pos = np.asarray(subG.nodes[n].get("position"))
            if pos is None or pos.shape[0] != 3:
                continue
            sp = str(subG.nodes[n].get("species", "unknown"))
            pos_by_species.setdefault(sp, []).append(pos)

        # Scatter by species
        for sp, arrs in pos_by_species.items():
            P = np.vstack(arrs) if len(arrs) > 0 else None
            if P is None:
                continue
            color = species_colors.get(sp, "#666666")
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, s=50, alpha=0.9, label=sp)

        # Edge linewidths by area when available
        areas = [edata.get("area", 1.0) for _, _, edata in subG.edges(data=True)]
        a_min = float(np.min(areas)) if len(areas) > 0 else 1.0
        a_max = float(np.max(areas)) if len(areas) > 0 else 1.0

        for u, v, edata in subG.edges(data=True):
            p1 = np.asarray(subG.nodes[u].get("position"))
            p2 = np.asarray(subG.nodes[v].get("position"))
            if p1 is None or p2 is None:
                continue
            lw = lw_from_area(edata.get("area", 1.0), a_min, a_max)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="black", alpha=0.6, linewidth=lw)

        ax.set_title(f"Component {i} (|V|={subG.number_of_nodes()}, |E|={subG.number_of_edges()})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if i == 0 and len(pos_by_species) > 0:
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_3d_graph_components_pu_only(G: Any, data: Any, names: np.ndarray, max_components: int = 10) -> None:
    """3D visualization of the largest connected components showing only Pu metal atoms.
    
    This function filters the graph to show only Pu atoms and their connections,
    making it easier to visualize Pu clustering patterns.
    
    Args:
        G: NetworkX graph with metal atom connectivity
        data: OVITO data object containing particle positions
        names: Array of atom names/types
        max_components: Maximum number of components to display
    """
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components_pu_only")

    # Local import to avoid global dependency in headless environments
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import

    if G.number_of_nodes() == 0:
        print("Graph is empty")
        return

    # Get particle positions
    positions = data.particles.positions
    
    # Add position and species data to graph nodes
    for node in G.nodes():
        if node < len(positions) and node < len(names):
            G.nodes[node]['position'] = positions[node]
            G.nodes[node]['species'] = names[node]

    # Filter graph to only include Pu atoms
    pu_nodes = [node for node in G.nodes() if node < len(names) and names[node] == 'Pu']
    
    if not pu_nodes:
        print("No Pu atoms found in the graph")
        return
    
    # Create subgraph with only Pu atoms
    pu_subgraph = G.subgraph(pu_nodes)
    
    if pu_subgraph.number_of_nodes() == 0:
        print("No Pu atoms in the filtered graph")
        return

    # Connected components sorted by size (only Pu components)
    components = sorted(nx.connected_components(pu_subgraph), key=len, reverse=True)
    components = components[:max_components]

    n = len(components)
    nrows, ncols = 2, 5
    nplots = max(n, 1)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Pu Metal Clusters (Pu-only connectivity)", fontsize=16, fontweight='bold')

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
        subG = pu_subgraph.subgraph(nodes)

        # Collect positions for Pu atoms
        pu_positions = []
        for n in subG.nodes:
            pos = np.asarray(subG.nodes[n].get("position"))
            if pos is not None and pos.shape[0] == 3:
                pu_positions.append(pos)

        if not pu_positions:
            ax.set_title(f"Component {i} - No valid positions")
            continue

        # Plot Pu atoms
        P = np.vstack(pu_positions)
        color = pu_colors[i % len(pu_colors)]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, s=80, alpha=0.9, 
                  label=f'Pu (n={len(pu_positions)})', edgecolors='black', linewidth=0.5)

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

        ax.set_title(f"Pu Component {i} (|V|={subG.number_of_nodes()}, |E|={subG.number_of_edges()})")
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        
        # Add legend for the first subplot
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()


def analyze_bond_network(data: Any, names: np.ndarray) -> Counter:
    """Analyze the bond network between different atom types and plot bar chart.

    Returns Counter of bond type -> count.
    """
    bonds = data.particles.bonds
    if not bonds or len(bonds) == 0:
        print("No bonds found!")
        return Counter()

    bond_types: List[tuple] = []
    for bond in bonds.topology:
        a, b = bond
        type_a = names[a] if a < len(names) else "out_of_range"
        type_b = names[b] if b < len(names) else "out_of_range"
        # Normalize bond types to avoid duplicates like "Na-Cl" vs "Cl-Na"
        bond_type = tuple(sorted([type_a, type_b]))
        bond_types.append(bond_type)

    # Count bond types
    bond_counts: Counter = Counter(bond_types)

    # Create bond type plot
    plt.figure(figsize=(12, 6))

    bond_labels = [f"{t[0]}-{t[1]}" for t in bond_counts.keys()]
    bond_values = list(bond_counts.values())

    bars = plt.bar(bond_labels, bond_values, alpha=0.7)
    plt.xlabel("Bond Type")
    plt.ylabel("Number of Bonds")
    plt.title("Bond Network Analysis")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, bond_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    print("Bond Network Summary:")
    for bond_type, count in bond_counts.items():
        print(f"{bond_type[0]}-{bond_type[1]}: {count} bonds")

    return bond_counts


def plot_rdfs(rdf_data: Dict[str, np.ndarray], x: Optional[float] = None, P: Optional[float] = None, cutoffs: Optional[Dict[Any, float]] = None) -> None:
    """Plot RDF curves with consistent coloring and save figure.

    Parameters:
        rdf_data: Dict containing key 'r' for radii and other keys for pair curves
        x: Optional composition fraction (e.g., Pu fraction). Defaults to 0.
        P: Optional pressure or similar scalar for title/filename. Defaults to 0.
        cutoffs: Optional mapping of pair -> cutoff distance. Keys may be strings like
                 'Cl-Na' or tuple pairs like ('Na','Cl'). Vertical lines will be drawn
                 for provided cutoffs with informative labels.
    """
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rdf_pairs = ["Cl-Cl", "Cl-Na", "Cl-Pu", "Na-Na", "Na-Pu", "Pu-Pu"]
    pair_to_color = {pair: color for pair, color in zip(rdf_pairs, cycle(color_cycle))}

    with plt.style.context("default"):
        # Single panel layout using subplot_mosaic for flexibility
        fig, axes = plt.subplot_mosaic(
            """
            a
            """,
            figsize=(4, 3),
            constrained_layout=True,
        )

        iax = "a"
        r = rdf_data["r"]

        for name, y in sorted(rdf_data.items(), key=lambda e: e[0]):
            if name == "r":
                continue
            axes[iax].plot(r, y, label=name, color=pair_to_color.get(name, None))

        # Optionally draw cutoff lines akin to notebook visuals
        if cutoffs:
            # normalize cutoff keys to 'A-B' alphabetical strings
            norm_cutoffs: Dict[str, float] = {}
            for k, v in cutoffs.items():
                if isinstance(k, (tuple, list)) and len(k) == 2:
                    key = "-".join(sorted([str(k[0]), str(k[1])]))
                else:
                    parts = str(k).split("-")
                    key = "-".join(sorted(parts)) if len(parts) == 2 else str(k)
                try:
                    norm_cutoffs[key] = float(v)
                except Exception:
                    continue

            for pair_key, cutoff in sorted(norm_cutoffs.items()):
                color = pair_to_color.get(pair_key, "black")
                axes[iax].axvline(cutoff, color=color, linestyle="--", alpha=0.7, label=f"{pair_key} cutoff: {cutoff:.3f} Å")

        axes[iax].legend()
        if x is not None and P is not None:
            axes[iax].set(title=f"x = {x}, P = {int(P):d}Gpa", xlabel="$r$ ($\\AA$)", ylabel="$g(r)$")
        else:
            axes[iax].set(xlabel="$r$ ($\\AA$)", ylabel="$g(r)$")
        plt.show()


