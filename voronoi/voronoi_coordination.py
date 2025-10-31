"""
Voronoi-Weighted Coordination Number Analysis

This module implements various coordination number calculations based on
Voronoi tessellation, including  

References:
- Topological CN: Count of Voronoi faces
- Area-weighted CN: Weighted by face areas
- Solid-angle weighted CN: Weighted by solid angles
- O'Keeffe/VoronoiNN: Threshold-based solid angle method
- CrystalNN: Combined solid-angle and face-area scheme
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_topological_coordination_number(
    voronoi_graph: Any,
    atom_indices: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Compute topological coordination number (CN_faces).
    
    Each Voronoi face corresponds to one nearest neighbor:
    CN_faces = F, where F is the number of faces
    
    Args:
        voronoi_graph: NetworkX graph with Voronoi tessellation
        atom_indices: Specific atoms to compute (None = all atoms)
    
    Returns:
        Dictionary mapping atom index to coordination number
    """
    if atom_indices is None:
        atom_indices = list(voronoi_graph.nodes())
    
    cn_dict = {}
    for atom_idx in atom_indices:
        if atom_idx in voronoi_graph:
            # Number of edges = number of faces
            cn_dict[atom_idx] = voronoi_graph.degree(atom_idx)
        else:
            cn_dict[atom_idx] = 0
    
    return cn_dict


def compute_face_area_weighted_cn(
    voronoi_graph: Any,
    atom_indices: Optional[List[int]] = None,
    return_normalized: bool = False,
) -> Dict[int, float]:
    """
    Compute face-area weighted coordination number (CN_A).
    
    Formula:
        CN_A = (Σ A_i)² / Σ A_i²
    
    Equivalent form using normalized weights p_i = A_i / Σ A_j:
        CN_A = 1 / Σ p_i²
    
    Args:
        voronoi_graph: NetworkX graph with 'area' edge attribute
        atom_indices: Specific atoms to compute (None = all atoms)
        return_normalized: Return normalized weights along with CN_A
    
    Returns:
        Dictionary mapping atom index to CN_A (and optionally weights)
    """
    if atom_indices is None:
        atom_indices = list(voronoi_graph.nodes())
    
    cn_dict = {}
    weights_dict = {} if return_normalized else None
    
    for atom_idx in atom_indices:
        if atom_idx not in voronoi_graph:
            cn_dict[atom_idx] = 0.0
            if return_normalized:
                weights_dict[atom_idx] = np.array([])
            continue
        
        # Get all neighbors and their face areas
        neighbors = list(voronoi_graph.neighbors(atom_idx))
        areas = []
        
        for neighbor_idx in neighbors:
            if atom_idx < neighbor_idx:
                edge_data = voronoi_graph.edges[atom_idx, neighbor_idx]
            else:
                edge_data = voronoi_graph.edges[neighbor_idx, atom_idx]
            area = edge_data.get("area", 0.0)
            if area > 0:
                areas.append(area)
        
        areas = np.array(areas)
        
        if len(areas) == 0:
            cn_dict[atom_idx] = 0.0
            if return_normalized:
                weights_dict[atom_idx] = np.array([])
            continue
        
        # Compute using the formula: CN_A = (Σ A_i)² / Σ A_i²
        sum_areas = np.sum(areas)
        sum_areas_squared = np.sum(areas ** 2)
        
        if sum_areas_squared > 0:
            cn_a = (sum_areas ** 2) / sum_areas_squared
        else:
            cn_a = 0.0
        
        cn_dict[atom_idx] = float(cn_a)
        
        if return_normalized:
            # Compute normalized weights p_i = A_i / Σ A_j
            normalized_weights = areas / sum_areas if sum_areas > 0 else areas
            weights_dict[atom_idx] = normalized_weights
    
    if return_normalized:
        return cn_dict, weights_dict
    return cn_dict


def compute_solid_angle(
    center_pos: np.ndarray,
    neighbor_pos: np.ndarray,
    face_area: float,
    distance: float,
) -> float:
    """
    Compute the solid angle subtended by a Voronoi face.
    
    For a face with area A at distance r:
        Ω = A / r²
    
    Args:
        center_pos: Position of central atom
        neighbor_pos: Position of neighbor atom
        face_area: Area of the Voronoi face
        distance: Distance between atoms (optional, will compute if None)
    
    Returns:
        Solid angle in steradians
    """
    if distance is None:
        distance = np.linalg.norm(neighbor_pos - center_pos)
    
    if distance <= 0 or face_area <= 0:
        return 0.0
    
    # Solid angle: Ω = A / r²
    omega = face_area / (distance ** 2)
    
    return float(omega)


def compute_solid_angle_weighted_cn(
    voronoi_graph: Any,
    positions: Optional[np.ndarray] = None,
    atom_indices: Optional[List[int]] = None,
    return_normalized: bool = False,
) -> Dict[int, float]:
    """
    Compute solid-angle weighted coordination number (CN_Ω).
    
    Formula:
        CN_Ω = (Σ Ω_i)² / Σ Ω_i²
    
    Equivalent form using normalized weights q_i = Ω_i / Σ Ω_j:
        CN_Ω = 1 / Σ q_i²
    
    Args:
        voronoi_graph: NetworkX graph with 'area' edge attribute
        positions: Nx3 array of atomic positions (if None, extracts from graph)
        atom_indices: Specific atoms to compute (None = all atoms)
        return_normalized: Return normalized weights along with CN_Ω
    
    Returns:
        Dictionary mapping atom index to CN_Ω (and optionally weights)
    """
    if atom_indices is None:
        atom_indices = list(voronoi_graph.nodes())
    
    # Extract positions if not provided
    if positions is None:
        positions = np.array([
            voronoi_graph.nodes[idx].get("position", np.zeros(3))
            for idx in voronoi_graph.nodes()
        ])
    
    cn_dict = {}
    weights_dict = {} if return_normalized else None
    
    for atom_idx in atom_indices:
        if atom_idx not in voronoi_graph or atom_idx >= len(positions):
            cn_dict[atom_idx] = 0.0
            if return_normalized:
                weights_dict[atom_idx] = np.array([])
            continue
        
        center_pos = positions[atom_idx]
        neighbors = list(voronoi_graph.neighbors(atom_idx))
        solid_angles = []
        
        for neighbor_idx in neighbors:
            if neighbor_idx >= len(positions):
                continue
                
            neighbor_pos = positions[neighbor_idx]
            
            # Get face area
            if atom_idx < neighbor_idx:
                edge_data = voronoi_graph.edges[atom_idx, neighbor_idx]
            else:
                edge_data = voronoi_graph.edges[neighbor_idx, atom_idx]
            face_area = edge_data.get("area", 0.0)
            
            if face_area > 0:
                # Compute solid angle: Ω = A / r²
                distance = np.linalg.norm(neighbor_pos - center_pos)
                if distance > 0:
                    omega = face_area / (distance ** 2)
                    solid_angles.append(omega)
        
        solid_angles = np.array(solid_angles)
        
        if len(solid_angles) == 0:
            cn_dict[atom_idx] = 0.0
            if return_normalized:
                weights_dict[atom_idx] = np.array([])
            continue
        
        # Compute using the formula: CN_Ω = (Σ Ω_i)² / Σ Ω_i²
        sum_omega = np.sum(solid_angles)
        sum_omega_squared = np.sum(solid_angles ** 2)
        
        if sum_omega_squared > 0:
            cn_omega = (sum_omega ** 2) / sum_omega_squared
        else:
            cn_omega = 0.0
        
        cn_dict[atom_idx] = float(cn_omega)
        
        if return_normalized:
            # Compute normalized weights q_i = Ω_i / Σ Ω_j
            normalized_weights = solid_angles / sum_omega if sum_omega > 0 else solid_angles
            weights_dict[atom_idx] = normalized_weights
    
    if return_normalized:
        return cn_dict, weights_dict
    return cn_dict


def compute_solid_angle_threshold_cn(
    voronoi_graph: Any,
    positions: Optional[np.ndarray] = None,
    tau: float = 0.5,
    atom_indices: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Compute coordination number using solid-angle thresholding (O'Keeffe/VoronoiNN).
    
    Defines neighbor weights w_i = Ω_i and keeps neighbors satisfying:
        w_i >= τ * max_j w_j
    
    Then CN = Σ Θ(w_i - τ * max_j w_j), where Θ is the Heaviside step function.
    
    Args:
        voronoi_graph: NetworkX graph with 'area' edge attribute
        positions: Nx3 array of atomic positions (if None, extracts from graph)
        tau: Threshold fraction (typical value ~0.5)
        atom_indices: Specific atoms to compute (None = all atoms)
    
    Returns:
        Dictionary mapping atom index to threshold-based coordination number
    """
    if atom_indices is None:
        atom_indices = list(voronoi_graph.nodes())
    
    # Extract positions if not provided
    if positions is None:
        positions = np.array([
            voronoi_graph.nodes[idx].get("position", np.zeros(3))
            for idx in voronoi_graph.nodes()
        ])
    
    cn_dict = {}
    
    for atom_idx in atom_indices:
        if atom_idx not in voronoi_graph or atom_idx >= len(positions):
            cn_dict[atom_idx] = 0
            continue
        
        center_pos = positions[atom_idx]
        neighbors = list(voronoi_graph.neighbors(atom_idx))
        weights = []
        
        for neighbor_idx in neighbors:
            if neighbor_idx >= len(positions):
                continue
                
            neighbor_pos = positions[neighbor_idx]
            
            # Get face area
            if atom_idx < neighbor_idx:
                edge_data = voronoi_graph.edges[atom_idx, neighbor_idx]
            else:
                edge_data = voronoi_graph.edges[neighbor_idx, atom_idx]
            face_area = edge_data.get("area", 0.0)
            
            if face_area > 0:
                # Compute solid angle: Ω = A / r²
                distance = np.linalg.norm(neighbor_pos - center_pos)
                if distance > 0:
                    omega = face_area / (distance ** 2)
                    weights.append(omega)
        
        if len(weights) == 0:
            cn_dict[atom_idx] = 0
            continue
        
        weights = np.array(weights)
        
        # Compute threshold: τ * max_j w_j
        max_weight = np.max(weights)
        threshold = tau * max_weight
        
        # Count neighbors above threshold
        cn = np.sum(weights >= threshold)
        cn_dict[atom_idx] = int(cn)
    
    return cn_dict


def compute_crystal_nn_weight(
    voronoi_graph: Any,
    positions: Optional[np.ndarray] = None,
    atom_indices: Optional[List[int]] = None,
) -> Dict[int, Dict[int, float]]:
    """
    Compute CrystalNN weights for neighbors.
    
    Formula: w_i^Vor = Ω_i² / A_i
    
    This combines Voronoi solid-angle and face-area weighting.
    Additional distance and chemical weighting can be applied later.
    
    Args:
        voronoi_graph: NetworkX graph with 'area' edge attribute
        positions: Nx3 array of atomic positions (if None, extracts from graph)
        atom_indices: Specific atoms to compute (None = all atoms)
    
    Returns:
        Dictionary mapping atom index to dict of {neighbor_index: weight}
    """
    if atom_indices is None:
        atom_indices = list(voronoi_graph.nodes())
    
    # Extract positions if not provided
    if positions is None:
        positions = np.array([
            voronoi_graph.nodes[idx].get("position", np.zeros(3))
            for idx in voronoi_graph.nodes()
        ])
    
    weights_dict = {}
    
    for atom_idx in atom_indices:
        if atom_idx not in voronoi_graph or atom_idx >= len(positions):
            weights_dict[atom_idx] = {}
            continue
        
        center_pos = positions[atom_idx]
        neighbors = list(voronoi_graph.neighbors(atom_idx))
        neighbor_weights = {}
        
        for neighbor_idx in neighbors:
            if neighbor_idx >= len(positions):
                continue
                
            neighbor_pos = positions[neighbor_idx]
            
            # Get face area
            if atom_idx < neighbor_idx:
                edge_data = voronoi_graph.edges[atom_idx, neighbor_idx]
            else:
                edge_data = voronoi_graph.edges[neighbor_idx, atom_idx]
            face_area = edge_data.get("area", 0.0)
            
            if face_area > 0:
                # Compute solid angle: Ω = A / r²
                distance = np.linalg.norm(neighbor_pos - center_pos)
                if distance > 0:
                    omega = face_area / (distance ** 2)
                    
                    # CrystalNN weight: w_i^Vor = Ω_i² / A_i
                    weight = (omega ** 2) / face_area
                    neighbor_weights[neighbor_idx] = float(weight)
        
        weights_dict[atom_idx] = neighbor_weights
    
    return weights_dict


def analyze_voronoi_coordination_all_methods(
    voronoi_graph: Any,
    positions: Optional[np.ndarray] = None,
    tau: float = 0.5,
    atom_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[int, Any]]:
    """
    Comprehensive coordination analysis using all methods.
    
    Computes all coordination number metrics for comparison:
    - Topological (CN_faces)
    - Face-area weighted (CN_A)
    - Solid-angle weighted (CN_Ω)
    - Solid-angle threshold (tau-based)
    - CrystalNN weights
    
    Args:
        voronoi_graph: NetworkX graph with Voronoi tessellation
        positions: Nx3 array of atomic positions
        tau: Threshold for solid-angle method (default 0.5)
        atom_indices: Specific atoms to analyze (None = all)
    
    Returns:
        Dictionary with all coordination metrics
    """
    results = {}
    
    # 1. Topological coordination number
    results["CN_faces"] = compute_topological_coordination_number(
        voronoi_graph, atom_indices
    )
    
    # 2. Face-area weighted coordination number
    cn_a, weights_a = compute_face_area_weighted_cn(
        voronoi_graph, atom_indices, return_normalized=True
    )
    results["CN_A"] = cn_a
    results["CN_A_weights"] = weights_a
    
    # 3. Solid-angle weighted coordination number
    cn_omega, weights_omega = compute_solid_angle_weighted_cn(
        voronoi_graph, positions, atom_indices, return_normalized=True
    )
    results["CN_Ω"] = cn_omega
    results["CN_Ω_weights"] = weights_omega
    
    # 4. Solid-angle threshold coordination number
    results["CN_threshold"] = compute_solid_angle_threshold_cn(
        voronoi_graph, positions, tau, atom_indices
    )
    
    # 5. CrystalNN weights
    results["CrystalNN_weights"] = compute_crystal_nn_weight(
        voronoi_graph, positions, atom_indices
    )
    
    return results


def compute_coordination_statistics(
    cn_values: Dict[int, Any],
    species: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Compute statistical summary of coordination numbers.
    
    Args:
        cn_values: Dictionary mapping atom index to CN value
        species: Optional dictionary mapping atom index to species
    
    Returns:
        Dictionary with mean, std, min, max, and optionally species breakdown
    """
    values = np.array(list(cn_values.values()))
    
    stats = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": len(values),
    }
    
    if species is not None:
        stats["by_species"] = {}
        unique_species = set(species.values())
        for spec in unique_species:
            spec_indices = [idx for idx in cn_values.keys() if species.get(idx) == spec]
            spec_values = [cn_values[idx] for idx in spec_indices]
            if len(spec_values) > 0:
                stats["by_species"][spec] = {
                    "mean": float(np.mean(spec_values)),
                    "std": float(np.std(spec_values)),
                    "min": float(np.min(spec_values)),
                    "max": float(np.max(spec_values)),
                    "count": len(spec_values),
                }
    
    return stats


def analyze_weighted_coordination_by_species(
    cn_dict: Dict[int, Any],
    voronoi_graph: Any,
    species_dict: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze coordination numbers by central species for easy histogramming.
    
    Organizes coordination number data by the species of the central atom,
    making it straightforward to create histograms and comparisons.
    
    Args:
        cn_dict: Dictionary mapping atom index to coordination number
        voronoi_graph: NetworkX graph with Voronoi tessellation (to get species)
        species_dict: Optional dict mapping atom index to species. If None, extracts from graph
    
    Returns:
        Dictionary organized by species:
        {
            "species_name": {
                "values": [cn1, cn2, ...],  # List of coordination numbers
                "statistics": {...},         # Mean, std, min, max, etc.
                "count": N,                  # Number of atoms of this species
            },
            ...
        }
    
    Example:
        >>> results = analyze_voronoi_coordination_all_methods(graph, positions)
        >>> cn_faces = results['CN_faces']
        >>> by_species = analyze_weighted_coordination_by_species(cn_faces, graph)
        >>> # Now easy to plot histograms:
        >>> import matplotlib.pyplot as plt
        >>> plt.hist(by_species['Pu']['values'], bins=20, label='Pu')
        >>> plt.hist(by_species['Na']['values'], bins=20, label='Na')
    """
    # Extract species information if not provided
    if species_dict is None:
        species_dict = {}
        for atom_idx in voronoi_graph.nodes():
            species_dict[atom_idx] = voronoi_graph.nodes[atom_idx].get("species", "Unknown")
    
    # Initialize results structure
    by_species_data = {}
    
    # Group coordination numbers by species
    for atom_idx, cn_value in cn_dict.items():
        species = species_dict.get(atom_idx, "Unknown")
        
        if species not in by_species_data:
            by_species_data[species] = {"values": [], "count": 0}
        
        by_species_data[species]["values"].append(cn_value)
        by_species_data[species]["count"] += 1
    
    # Compute statistics for each species
    results = {}
    for species, data in by_species_data.items():
        values = np.array(data["values"])
        
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "count": len(values),
        }
        
        # Compute additional percentiles if needed
        if len(values) > 0:
            stats["q25"] = float(np.percentile(values, 25))
            stats["q75"] = float(np.percentile(values, 75))
        
        results[species] = {
            "values": data["values"],  # Keep as list for easy histogramming
            "statistics": stats,
            "count": data["count"],
        }
    
    return results


def compute_coordination_histogram_data(
    cn_dict: Dict[int, Any],
    voronoi_graph: Any,
    bins: int = 20,
    species_dict: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute histogram data for coordination numbers by species.
    
    Pre-computes histogram bins and counts for each species, making plotting
    even more straightforward.
    
    Args:
        cn_dict: Dictionary mapping atom index to coordination number
        voronoi_graph: NetworkX graph with Voronoi tessellation
        bins: Number of bins for histograms (default 20)
        species_dict: Optional dict mapping atom index to species
    
    Returns:
        Dictionary organized by species with histogram data:
        {
            "species_name": {
                "counts": [...],           # Bin counts
                "bin_edges": [...],        # Bin edges
                "bin_centers": [...],      # Bin centers (for plotting)
            },
            ...
        }
    
    Example:
        >>> results = analyze_voronoi_coordination_all_methods(graph, positions)
        >>> cn_a = results['CN_A']
        >>> hist_data = compute_coordination_histogram_data(cn_a, graph, bins=30)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(hist_data['Pu']['bin_centers'], hist_data['Pu']['counts'], label='Pu')
    """
    by_species = analyze_weighted_coordination_by_species(cn_dict, voronoi_graph, species_dict)
    
    hist_results = {}
    for species, data in by_species.items():
        values = np.array(data["values"])
        
        if len(values) == 0:
            hist_results[species] = {
                "counts": np.array([]),
                "bin_edges": np.array([]),
                "bin_centers": np.array([]),
            }
            continue
        
        # Compute histogram
        counts, bin_edges = np.histogram(values, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        hist_results[species] = {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_centers": bin_centers.tolist(),
        }
    
    return hist_results


__all__ = [
    "compute_topological_coordination_number",
    "compute_face_area_weighted_cn",
    "compute_solid_angle",
    "compute_solid_angle_weighted_cn",
    "compute_solid_angle_threshold_cn",
    "compute_crystal_nn_weight",
    "analyze_voronoi_coordination_all_methods",
    "compute_coordination_statistics",
    "analyze_weighted_coordination_by_species",
    "compute_coordination_histogram_data"
]

