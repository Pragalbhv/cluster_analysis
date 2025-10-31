"""
Merged plotting utilities consolidating:
- plot_utils.py
- plot_utils_pu_only.py
- standardized_plots.py

All functions are preserved. Standardized variants are exported with std_ prefixes.
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

# -----------------------------
# Standard style config (from standardized_plots)
# -----------------------------
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

COLORS = {
    'species': {
        'Na': '#1f77b4',
        'Pu': '#d62728',
        'Cl': '#2ca02c',
        'unknown': '#7f7f7f'
    },
    'clusters': [
        '#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896'
    ],
    'coordination': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'background': '#f0f0f0',
    'grid': '#cccccc'
}

# -----------------------------
# Standardized helper functions (std_*)
# -----------------------------

def setup_plot_style():
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
    return COLORS

def extract_positions_from_data(data: Any) -> np.ndarray:
    try:
        positions = data.particles.positions
    except Exception:
        try:
            positions = data.get_positions()
        except Exception:
            try:
                positions = getattr(data, "positions", None)
            except Exception:
                positions = None
    if positions is None:
        raise ValueError("Could not extract positions from data object")
    return np.asarray(positions)

def extract_names_from_data(data: Any) -> np.ndarray:
    try:
        names = data.particles["Particle Type"]
        if hasattr(names, 'dtype') and np.issubdtype(names.dtype, np.integer):
            type_map = {1: "Na", 2: "Pu", 3: "Cl"}
            names = np.array([type_map.get(t, f"Type{t}") for t in names])
    except Exception:
        try:
            names = np.array(data.get_chemical_symbols())
        except Exception:
            try:
                names = getattr(data, "names", None)
                if names is not None:
                    names = np.asarray(names)
            except Exception:
                names = None
    if names is None:
        raise ValueError("Could not extract atom names from data object")
    return np.asarray(names)

# -----------------------------
# Original plot_utils functions
# -----------------------------

def plot_coordination_histograms(coord_data: Dict[str, Dict[str, List[int]]], central_type: str) -> None:
    plt.figure(figsize=(10, 6))
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
    species = sorted(coord_data[central_type].keys())
    for idx, neighbor_type in enumerate(species):
        data = coord_data[central_type][neighbor_type]
        if not data:
            continue
        mean = np.mean(data)
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
    if nx is None:
        raise ImportError("networkx is required for plot_graph_structure")
    plt.figure(figsize=(12, 8))
    pos = {}
    species_colors = {"Na": "blue", "Cl": "green", "Pu": "red"}
    for node in G.nodes():
        x, y, z = G.nodes[node]["position"]
        pos[node] = (x, y)
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
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color="black")
    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_cluster_size_distribution(sizes: List[int], title: str = "Cluster Size Distribution") -> None:
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
    is_pu = names == "Pu"
    is_na = names == "Na"
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
    sizes = [c["size"] for c in cluster_compositions]
    pu_fractions = [c["pu_fraction"] for c in cluster_compositions]
    na_fractions = [c["na_fraction"] for c in cluster_compositions]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(sizes, pu_fractions, alpha=0.6, s=50)
    ax1.set_xlabel("Cluster Size")
    ax1.set_ylabel("Pu Fraction")
    ax1.set_title("Pu Fraction vs Cluster Size")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    ax2.scatter(sizes, na_fractions, alpha=0.6, s=50, color="orange")
    ax2.set_xlabel("Cluster Size")
    ax2.set_ylabel("Na Fraction")
    ax2.set_title("Na Fraction vs Cluster Size")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()
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
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import
    try:
        positions = data.particles["Position"]
    except Exception:
        try:
            positions = data.get_positions()
        except Exception:
            positions = getattr(data, "positions", None)
            if positions is None and hasattr(data, "particles"):
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
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.show()


def plot_3d_cluster_with_graph(data, cluster_ids, names, G, max_clusters):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import
    if G is None:
        raise ValueError("A NetworkX graph 'G' must be provided to draw edges.")
    positions_src = None
    try:
        positions_src = data.particles["Position"]
    except Exception:
        try:
            positions_src = data.get_positions()
        except Exception:
            positions_src = getattr(data, "positions", None)
    unique_clusters, counts = np.unique(cluster_ids[cluster_ids >= 0], return_counts=True)
    if len(unique_clusters) == 0:
        print("No clusters to visualize")
        return
    largest_cluster_indices = np.argsort(counts)[-max_clusters:][::-1]
    n_to_show = min(len(largest_cluster_indices), max_clusters)
    nrows, ncols = 2, 5
    n_plots = max(n_to_show, 1)
    fig = plt.figure(figsize=(20, 15))
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
        cluster_atom_indices = set(np.where(cluster_mask)[0].tolist())
        cluster_node_ids = []
        for n in G.nodes:
            ai = node_to_atom_index(n)
            if ai is not None and ai in cluster_atom_indices:
                cluster_node_ids.append(n)
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
            subG = G.subgraph(cluster_node_ids)
            if subG.number_of_edges() == 0:
                print(f"Cluster {cid}: no edges to draw")
            else:
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
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import
    if G.number_of_nodes() == 0:
        print("Graph is empty")
        return
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
        pos_by_species: Dict[str, List[np.ndarray]] = {}
        for n in subG.nodes:
            pos = np.asarray(subG.nodes[n].get("position"))
            if pos is None or pos.shape[0] != 3:
                continue
            sp = str(subG.nodes[n].get("species", "unknown"))
            pos_by_species.setdefault(sp, []).append(pos)
        for sp, arrs in pos_by_species.items():
            P = np.vstack(arrs) if len(arrs) > 0 else None
            if P is None:
                continue
            color = species_colors.get(sp, "#666666")
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, s=50, alpha=0.9, label=sp)
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
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components_pu_only")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import
    if G.number_of_nodes() == 0:
        print("Graph is empty")
        return
    positions = data.particles.positions
    for node in G.nodes():
        if node < len(positions) and node < len(names):
            G.nodes[node]['position'] = positions[node]
            G.nodes[node]['species'] = names[node]
    pu_nodes = [node for node in G.nodes() if node < len(names) and names[node] == 'Pu']
    if not pu_nodes:
        print("No Pu atoms found in the graph")
        return
    pu_subgraph = G.subgraph(pu_nodes)
    if pu_subgraph.number_of_nodes() == 0:
        print("No Pu atoms in the filtered graph")
        return
    components = sorted(nx.connected_components(pu_subgraph), key=len, reverse=True)
    components = components[:max_components]
    n = len(components)
    nrows, ncols = 2, 5
    nplots = max(n, 1)
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Pu Metal Clusters (Pu-only connectivity)", fontsize=16, fontweight='bold')
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
        pu_positions = []
        for n in subG.nodes:
            pos = np.asarray(subG.nodes[n].get("position"))
            if pos is not None and pos.shape[0] == 3:
                pu_positions.append(pos)
        if not pu_positions:
            ax.set_title(f"Component {i} - No valid positions")
            continue
        P = np.vstack(pu_positions)
        color = pu_colors[i % len(pu_colors)]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, s=80, alpha=0.9,
                  label=f'Pu (n={len(pu_positions)})', edgecolors='black', linewidth=0.5)
        areas = [edata.get("area", 1.0) for _, _, edata in subG.edges(data=True)]
        a_min = float(np.min(areas)) if len(areas) > 0 else 1.0
        a_max = float(np.max(areas)) if len(areas) > 0 else 1.0
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
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.show()


def analyze_bond_network(data: Any, names: np.ndarray) -> Counter:
    bonds = data.particles.bonds
    if not bonds or len(bonds) == 0:
        print("No bonds found!")
        return Counter()
    bond_types: List[tuple] = []
    for bond in bonds.topology:
        a, b = bond
        type_a = names[a] if a < len(names) else "out_of_range"
        type_b = names[b] if b < len(names) else "out_of_range"
        bond_type = tuple(sorted([type_a, type_b]))
        bond_types.append(bond_type)
    bond_counts: Counter = Counter(bond_types)
    plt.figure(figsize=(12, 6))
    bond_labels = [f"{t[0]}-{t[1]}" for t in bond_counts.keys()]
    bond_values = list(bond_counts.values())
    bars = plt.bar(bond_labels, bond_values, alpha=0.7)
    plt.xlabel("Bond Type")
    plt.ylabel("Number of Bonds")
    plt.title("Bond Network Analysis")
    plt.xticks(rotation=45)
    for bar, value in zip(bars, bond_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()
    print("Bond Network Summary:")
    for bond_type, count in bond_counts.items():
        print(f"{bond_type[0]}-{bond_type[1]}: {count} bonds")
    return bond_counts


def plot_rdfs(rdf_data: Dict[str, np.ndarray], x: Optional[float] = None, P: Optional[float] = None, cutoffs: Optional[Dict[Any, float]] = None, figsize: Tuple[float, float] = (4, 3)) -> None:
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rdf_pairs = ["Cl-Cl", "Cl-Na", "Cl-Pu", "Na-Na", "Na-Pu", "Pu-Pu"]
    pair_to_color = {pair: color for pair, color in zip(rdf_pairs, cycle(color_cycle))}
    with plt.style.context("default"):
        fig, axes = plt.subplot_mosaic(
            """
            a
            """,
            figsize=figsize,
            constrained_layout=True,
        )
        iax = "a"
        r = rdf_data["r"]
        for name, y in sorted(rdf_data.items(), key=lambda e: e[0]):
            if name == "r":
                continue
            if name == "complete":
                color = "red"
                linewidth = 2.5
                linestyle = "-"
            else:
                color = pair_to_color.get(name, None)
                linewidth = 1.5
                linestyle = "-"
            axes[iax].plot(r, y, label=name, color=color, linewidth=linewidth, linestyle=linestyle)
        if cutoffs:
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

# -----------------------------
# Standardized variants (std_*) from standardized_plots
# -----------------------------

def std_plot_coordination_histograms(
    coord_data: Dict[str, Dict[str, List[int]]], 
    central_type: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
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
    for idx, neighbor_type in enumerate(species):
        data = coord_data[central_type][neighbor_type]
        if not data:
            continue
        mean = np.mean(data)
        std = np.std(data)
        color = colors[idx % len(colors)]
        plt.hist(
            data,
            bins=range(0, max(data) + 2),
            alpha=PLOT_CONFIG['alpha_main'],
            label=f"{neighbor_type} (μ={mean:.2f}±{std:.2f})",
            color=color,
            edgecolor='black',
            linewidth=0.5
        )
        plt.axvline(mean, color=color, linestyle="--", linewidth=2, alpha=0.8)
    plot_title = title or f"Coordination Histogram: {central_type}"
    plt.title(plot_title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    plt.xlabel("Coordination Number", fontsize=PLOT_CONFIG['font_size_labels'])
    plt.ylabel("Frequency", fontsize=PLOT_CONFIG['font_size_labels'])
    plt.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    plt.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
    total_atoms = sum(len(data) for data in coord_data[central_type].values())
    stats_text = f"Total {central_type} atoms: {total_atoms}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=PLOT_CONFIG['font_size_legend'],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()


def std_plot_graph_structure(
    G: Any, 
    title: str = "Graph Structure",
    show_edges: bool = True,
    show_labels: bool = False,
    save_path: Optional[str] = None
) -> None:
    if nx is None:
        raise ImportError("networkx is required for plot_graph_structure")
    if G.number_of_nodes() == 0:
        print("Graph is empty")
        return
    setup_plot_style()
    plt.figure(figsize=PLOT_CONFIG['figure_size_large'])
    pos = {}
    species_colors = COLORS['species']
    for node in G.nodes():
        try:
            x, y, z = G.nodes[node]["position"]
            pos[node] = (x, y)
        except (KeyError, ValueError, TypeError):
            print(f"Warning: Invalid position data for node {node}")
            continue
    if not pos:
        print("No valid positions found in graph")
        return
    for species in species_colors.keys():
        nodes_of_species = [n for n in G.nodes() if G.nodes[n].get("species") == species]
        if nodes_of_species:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes_of_species,
                node_color=species_colors[species],
                node_size=100,
                alpha=PLOT_CONFIG['alpha_main'],
                label=f"{species} ({len(nodes_of_species)})",
            )
    if show_edges and G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.0, edge_color="gray")
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    plt.legend(fontsize=PLOT_CONFIG['font_size_legend'], loc='upper right')
    plt.axis("off")
    stats_text = f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
             fontsize=PLOT_CONFIG['font_size_legend'],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()


def std_plot_cluster_size_distribution(
    sizes: List[int], 
    title: str = "Cluster Size Distribution",
    bins: Optional[List[float]] = None,
    log_scale: bool = False,
    save_path: Optional[str] = None
) -> None:
    if len(sizes) == 0:
        print("No clusters to plot")
        return
    setup_plot_style()
    plt.figure(figsize=PLOT_CONFIG['figure_size_medium'])
    sizes_array = np.array(sizes)
    max_size = np.max(sizes_array)
    if bins is None:
        bins = np.arange(0, max_size + 2) - 0.5
    plt.hist(sizes, bins=bins, alpha=PLOT_CONFIG['alpha_main'], 
             edgecolor='black', linewidth=0.5, color=COLORS['species']['Pu'])
    if log_scale:
        plt.yscale('log')
    plt.title(title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    plt.xlabel("Cluster Size", fontsize=PLOT_CONFIG['font_size_labels'])
    plt.ylabel("Frequency", fontsize=PLOT_CONFIG['font_size_labels'])
    plt.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
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


def std_plot_3d_graph_components(
    G: Any, 
    max_components: int = 6,
    title: str = "3D Graph Components",
    show_edges: bool = True,
    save_path: Optional[str] = None
) -> None:
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components")
    if G.number_of_nodes() == 0:
        print("Graph is empty")
        return
    setup_plot_style()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    components = components[:max_components]
    if not components:
        print("No connected components found")
        return
    n = len(components)
    nrows = min(2, n)
    ncols = min(5, n)
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size_3d'])
    fig.suptitle(title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    species_colors = COLORS['species']
    cluster_colors = COLORS['clusters']
    def lw_from_area(a: float, a0: float, a1: float) -> float:
        if a1 <= a0:
            return 1.0
        t = (float(a) - a0) / (a1 - a0)
        return 0.6 + 2.4 * max(0.0, min(1.0, t))
    for i, nodes in enumerate(components):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        subG = G.subgraph(nodes)
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
        for sp, arrs in pos_by_species.items():
            if not arrs:
                continue
            P = np.vstack(arrs)
            color = species_colors.get(sp, COLORS['species']['unknown'])
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], 
                      c=color, s=80, alpha=PLOT_CONFIG['alpha_main'], 
                      label=sp if i == 0 else "")
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


def std_plot_3d_graph_components_pu_only(
    pu_cluster_graph: Any, 
    data: Any, 
    names: np.ndarray, 
    max_components: int = 6,
    show_na_context: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    if nx is None:
        raise ImportError("networkx is required for plot_3d_graph_components_pu_only")
    if pu_cluster_graph.number_of_nodes() == 0:
        print("Pu cluster graph is empty")
        return
    setup_plot_style()
    try:
        positions = extract_positions_from_data(data)
    except ValueError as e:
        print(f"Error extracting positions: {e}")
        return
    for node in pu_cluster_graph.nodes():
        if node < len(positions) and node < len(names):
            pu_cluster_graph.nodes[node]['position'] = positions[node]
            pu_cluster_graph.nodes[node]['species'] = names[node]
    components = sorted(nx.connected_components(pu_cluster_graph), key=len, reverse=True)
    components = components[:max_components]
    if not components:
        print("No Pu clusters found")
        return
    n = len(components)
    nrows = min(2, n)
    ncols = min(5, n)
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size_3d'])
    plot_title = title or "Pu Metal Clusters (Pu-only connectivity)"
    if show_na_context:
        plot_title += " with Na Context"
    fig.suptitle(plot_title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    pu_colors = COLORS['clusters']
    def lw_from_area(a: float, a0: float, a1: float) -> float:
        if a1 <= a0:
            return 1.0
        t = (float(a) - a0) / (a1 - a0)
        return 0.6 + 2.4 * max(0.0, min(1.0, t))
    for i, nodes in enumerate(components):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        subG = pu_cluster_graph.subgraph(nodes)
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
        P = np.vstack(pu_positions)
        color = pu_colors[i % len(pu_colors)]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], 
                  c=color, s=100, alpha=PLOT_CONFIG['alpha_main'], 
                  label=f'Pu Cluster {i+1} (n={len(pu_positions)})', 
                  edgecolors='darkred', linewidth=0.5)
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
        ax.set_title(f"Pu Component {i+1}\n(|V|={subG.number_of_nodes()}, |E|={subG.number_of_edges()})",
                    fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_xlabel("X (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_ylabel("Y (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_zlabel("Z (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
        if i == 0:
            ax.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.show()

# -----------------------------
# Pu-only specialized functions from plot_utils_pu_only
# -----------------------------

def plot_mixed_tessellation_structure(
    mixed_graph: Any, 
    pu_cluster_graph: Any,
    title: str = "Mixed Tessellation Structure"
) -> None:
    if nx is None:
        raise ImportError("networkx is required for plot_mixed_tessellation_structure")
    plt.figure(figsize=(15, 10))
    pos = {}
    species_colors = {"Na": "blue", "Cl": "green", "Pu": "red"}
    for node in mixed_graph.nodes():
        x, y, z = mixed_graph.nodes[node]["position"]
        pos[node] = (x, y)
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
    nx.draw_networkx_edges(pu_cluster_graph, pos, alpha=0.6, width=2, edge_color="red")
    other_edges = [(u, v) for u, v in mixed_graph.edges() if not pu_cluster_graph.has_edge(u, v)]
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
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    try:
        positions = data.particles["Position"]
    except Exception:
        try:
            positions = data.get_positions()
        except Exception:
            positions = getattr(data, "positions", None)
            if positions is None:
                raise AttributeError("Could not extract positions from 'data' for 3D plotting.")
    positions = np.asarray(positions)
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
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.show()


def plot_na_pu_interaction_network(
    mixed_graph: Any,
    min_area: float = 0.0,
    title: str = "Na-Pu Interaction Network"
) -> None:
    if nx is None:
        raise ImportError("networkx is required for plot_na_pu_interaction_network")
    plt.figure(figsize=(12, 8))
    pos = {}
    for node in mixed_graph.nodes():
        x, y, z = mixed_graph.nodes[node]["position"]
        pos[node] = (x, y)
    na_nodes = [n for n in mixed_graph.nodes() if mixed_graph.nodes[n]["species"] == "Na"]
    pu_nodes = [n for n in mixed_graph.nodes() if mixed_graph.nodes[n]["species"] == "Pu"]
    cl_nodes = [n for n in mixed_graph.nodes() if mixed_graph.nodes[n]["species"] == "Cl"]
    if na_nodes:
        nx.draw_networkx_nodes(mixed_graph, pos, nodelist=na_nodes, 
                              node_color="blue", node_size=50, alpha=0.7, label="Na")
    if pu_nodes:
        nx.draw_networkx_nodes(mixed_graph, pos, nodelist=pu_nodes, 
                              node_color="red", node_size=80, alpha=0.8, label="Pu")
    if cl_nodes:
        nx.draw_networkx_nodes(mixed_graph, pos, nodelist=cl_nodes, 
                              node_color="green", node_size=30, alpha=0.5, label="Cl")
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
    if na_pu_edges:
        nx.draw_networkx_edges(mixed_graph, pos, edgelist=na_pu_edges, 
                              alpha=0.8, width=3, edge_color="purple", label=f"Na-Pu ({len(na_pu_edges)})")
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
    if not coordination_stats:
        print("No coordination statistics to plot")
        return
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
    bars1 = ax1.bar(species_pairs, means, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel("Neighbor Species")
    ax1.set_ylabel("Average Coordination Number")
    ax1.set_title("Mean Coordination Numbers")
    ax1.tick_params(axis='x', rotation=45)
    ax1.errorbar(range(len(species_pairs)), means, yerr=stds, fmt='none', color='red', capsize=5)
    for bar, mean, std in zip(bars1, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom')
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
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    try:
        positions = data.particles["Position"]
    except Exception:
        try:
            positions = data.get_positions()
        except Exception:
            positions = getattr(data, "positions", None)
            if positions is None:
                raise AttributeError("Could not extract positions from 'data' for 3D plotting.")
    positions = np.asarray(positions)
    pu_components = list(nx.connected_components(pu_cluster_graph))
    pu_components = pu_components[:max_clusters]
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("3D Mixed Tessellation with Pu Clusters", fontsize=16, fontweight='bold')
    n = len(pu_components)
    nrows, ncols = 2, 3
    nplots = max(n, 1)
    for i, component in enumerate(pu_components[:nplots]):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='lightgray', s=10, alpha=0.3, label='All atoms')
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
    if not temporal_results:
        print("No temporal data to plot")
        return
    frame_indices = temporal_results.get("frame_indices", [])
    cluster_stats = temporal_results.get("cluster_statistics", [])
    if not cluster_stats:
        print("No cluster statistics found")
        return
    num_clusters = [stat.get("num_clusters", 0) for stat in cluster_stats]
    largest_sizes = [stat.get("largest_cluster_size", 0) for stat in cluster_stats]
    avg_sizes = [stat.get("avg_cluster_size", 0) for stat in cluster_stats]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    axes[0, 0].plot(frame_indices, num_clusters, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel("Frame Index")
    axes[0, 0].set_ylabel("Number of Clusters")
    axes[0, 0].set_title("Cluster Count Evolution")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(frame_indices, largest_sizes, 'r-s', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel("Frame Index")
    axes[0, 1].set_ylabel("Largest Cluster Size")
    axes[0, 1].set_title("Largest Cluster Evolution")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(frame_indices, avg_sizes, 'g-^', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel("Frame Index")
    axes[1, 0].set_ylabel("Average Cluster Size")
    axes[1, 0].set_title("Average Cluster Size Evolution")
    axes[1, 0].grid(True, alpha=0.3)
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
    if not neighbor_list:
        print("No neighbor list data to plot")
        return
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
    pu_species = list(pu_neighbors.keys())
    if pu_species:
        pu_data = [pu_neighbors[sp] for sp in pu_species]
        axes[0, 0].boxplot(pu_data, labels=pu_species)
        axes[0, 0].set_xlabel("Neighbor Species")
        axes[0, 0].set_ylabel("Facet Area")
        axes[0, 0].set_title("Pu Neighbor Facet Areas")
        axes[0, 0].tick_params(axis='x', rotation=45)
    na_species = list(na_neighbors.keys())
    if na_species:
        na_data = [na_neighbors[sp] for sp in na_species]
        axes[0, 1].boxplot(na_data, labels=na_species)
        axes[0, 1].set_xlabel("Neighbor Species")
        axes[0, 1].set_ylabel("Facet Area")
        axes[0, 1].set_title("Na Neighbor Facet Areas")
        axes[0, 1].tick_params(axis='x', rotation=45)
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

# -----------------------------
# Exports
# -----------------------------
__all__ = [
    # base/original names
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
    # standardized helpers and variants
    "setup_plot_style",
    "get_standard_colors",
    "extract_positions_from_data",
    "extract_names_from_data",
    "std_plot_coordination_histograms",
    "std_plot_graph_structure",
    "std_plot_cluster_size_distribution",
    "std_plot_3d_graph_components",
    "std_plot_3d_graph_components_pu_only",
    # pu-only specialized
    "plot_mixed_tessellation_structure",
    "plot_pu_clusters_with_na_context",
    "plot_na_pu_interaction_network",
    "plot_pu_coordination_analysis",
    "plot_3d_mixed_tessellation",
    "plot_cluster_evolution_analysis",
    "plot_neighbor_list_analysis",
]
