"""
Shell Analysis Utilities for Bond-Based Cluster Analysis

This module contains functions for analyzing coordination shells in NaCl-PuCl3 systems
using bond analysis and RDF integration.

Functions are extracted from:
- ovito_bond_shell_analysis.ipynb
- rdf_integrate.ipynb
- rdf_shell_analysis.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from bondmodifier_utils import gaussian_smooth


# =============================================================================
# Species Density Calculation
# =============================================================================

def get_species_densities_from_data(data):
    """
    Calculate species densities from simulation data.
    
    Parameters:
    -----------
    data : OVITO data object
        Simulation data containing particle information
    
    Returns:
    --------
    dict : Dictionary of species densities {species: density}
    """
    import numpy as np
    
    # Get particle types and names
    particle_types = data.particles.particle_types
    type_names = [particle_types.type_by_id(t).name for t in particle_types]
    
    # Get unique species and counts
    unique_species, counts = np.unique(type_names, return_counts=True)
    
    # Calculate box volume from simulation cell
    cell = data.cell
    if cell is not None:
        cell_matrix = cell.matrix
        volume = np.abs(np.linalg.det(cell_matrix[:3, :3]))
    else:
        raise ValueError("Cannot determine box volume from data")
    
    # Calculate densities
    densities = {}
    for species, count in zip(unique_species, counts):
        densities[species] = count / volume
    
    return densities


# =============================================================================
# Coordination Number Calculations
# =============================================================================

def calculate_coordination_number(r, g_r, rho_j, r1=None, r2=None):
    """
    Calculate coordination number using the formula:
    CN_ij = 4πρ_j ∫[r1 to r2] r² g_ij(r) dr
    
    Parameters:
    -----------
    r : array
        Radial distances
    g_r : array  
        Radial distribution function values
    rho_j : float
        Number density of species j
    r1, r2 : float, optional
        Integration limits. If None, will be determined automatically.
    
    Returns:
    --------
    float : Coordination number
    """
    
    # Determine integration limits if not provided
    if r1 is None or r2 is None:
        # Find first minimum after first peak using scipy
        # Smooth the data slightly
        smooth_window = 5
        g_smooth = np.convolve(g_r, np.ones(smooth_window)/smooth_window, mode='same')
        
        
        # Invert g(r) to find minima as peaks
        g_inverted = -g_smooth
        
        # Find peaks (minima in original) with prominence filtering
        peaks, _ = find_peaks(g_inverted, prominence=0.5, distance=1)
        
        if len(peaks) > 0:
            # Find first minimum after the first peak
            for peak in peaks:
                r2 = r[peak]
                break
            else:
                r2 = r[-1]
        else:
            r2 = r[-1]
        
        r1 = 0.0
    
    # Ensure integration limits are within data range
    r1 = max(r1, r[0])
    r2 = min(r2, r[-1])
    
    # Find indices for integration
    idx1 = np.argmin(np.abs(r - r1))
    idx2 = np.argmin(np.abs(r - r2))
    
    # Ensure idx1 < idx2
    if idx1 >= idx2:
        idx1 = 0
        idx2 = min(idx1 + 10, len(r) - 1)
    
    # Perform integration
    r_integration = r[idx1:idx2+1]
    g_integration = g_r[idx1:idx2+1]
    
    # Calculate coordination number: CN_ij = 4πρ_j ∫ r² g_ij(r) dr
    integrand = 4 * np.pi * rho_j * r_integration**2 * g_integration
    coordination_number = np.trapz(integrand, r_integration)
    
    return coordination_number


def calculate_coordination_number_vs_r(r, g_r, rho_j, r_max=None):
    """
    Calculate coordination number as a function of radial distance r.
    
    This function computes the cumulative coordination number CN(r) by integrating
    the radial distribution function from 0 to r:
    CN(r) = 4πρ_j ∫[0 to r] r'² g_ij(r') dr'
    
    Parameters:
    -----------
    r : array
        Radial distances
    g_r : array  
        Radial distribution function values
    rho_j : float
        Number density of species j
    r_max : float, optional
        Maximum integration distance. If None, uses the full r range.
    
    Returns:
    --------
    tuple : (r_values, cn_values)
        r_values: array of radial distances
        cn_values: array of coordination numbers at each r
    """
    import numpy as np
    
    # Determine integration range
    if r_max is None:
        r_max = r[-1]
    
    # Find the index corresponding to r_max
    r_max_idx = np.argmin(np.abs(r - r_max))
    
    # Use r values up to r_max
    r_values = r[:r_max_idx+1]
    g_values = g_r[:r_max_idx+1]
    
    # Calculate coordination number at each r
    cn_values = np.zeros_like(r_values)
    
    for i in range(len(r_values)):
        # Integrate from 0 to r_values[i]
        r_integration = r_values[:i+1]
        g_integration = g_values[:i+1]
        
        # Calculate integrand: 4πρ_j * r² * g(r)
        integrand = 4 * np.pi * rho_j * r_integration**2 * g_integration
        
        # Perform integration using trapezoidal rule
        cn_values[i] = np.trapz(integrand, r_integration)
    
    return r_values, cn_values


# =============================================================================
# Shell Detection
# =============================================================================

def detect_proper_shells(first_deriv, min_prominence=0.1, smoothing_method='gaussian', gaussian_sigma=3.0):
    """
    Detect shells using proper pattern: derivative goes up, peaks, then goes down to zero.
    
    Args:
        first_deriv: First derivative array
        min_prominence: Minimum prominence for peaks
        
    Returns:
        Array of shell boundary indices
    """
    # Find peaks in first derivative (where coordination growth is fastest)
    if smoothing_method == 'gaussian':
        first_deriv = gaussian_smooth(first_deriv, gaussian_sigma, mode='nearest')
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}. Choose from: 'moving_avg', 'savgol', 'gaussian'")
        
    peaks, properties = find_peaks(first_deriv, prominence=min_prominence, distance=10)
    
    if len(peaks) == 0:
        return np.array([])
    
    # For each peak, find the subsequent minimum (where derivative goes to zero)
    shell_boundaries = []
    
    for peak_idx in peaks:
        # Look for minima after this peak
        search_start = peak_idx + 5  # Start searching a bit after the peak
        search_end = min(peak_idx + 50, len(first_deriv))  # Don't search too far
        
        if search_start >= len(first_deriv):
            continue
            
        # Find minima in the region after the peak
        region_deriv = first_deriv[search_start:search_end]
        if len(region_deriv) < 5:
            continue
            
        # Look for points where derivative is close to zero (plateau)
        zero_threshold = np.std(first_deriv) * 0.2
        zero_crossings = np.where(np.abs(region_deriv) < zero_threshold)[0]
        
        if len(zero_crossings) > 0:
            # Take the first significant zero crossing after the peak
            shell_idx = search_start + zero_crossings[0]
            shell_boundaries.append(shell_idx)
    
    return np.array(shell_boundaries)


# =============================================================================
# Neighbor Shell Analysis (from ovito_bond_shell_analysis.ipynb)
# =============================================================================

def analyze_pu_neighbor_shells(pipe, names, cutoffs, max_radius=10.0):
    
    """
    Analyze Pu neighbor shells by varying PuCl bond cutoff and counting all Pu neighbors.
    
    This function:
    1. Varies PuCl bond cutoff from 0.5 to max_radius
    2. Counts Pu neighbors by type: Pu-Pu, Pu-Na, Pu-Cl
    3. Detects shells based on neighbor count changes
    
    Args:
        pipe: OVITO pipeline object
        names: Array of atom names/types
        cutoffs: Dictionary of current cutoff values
        max_radius: Maximum radius to analyze
    
    Returns:
        Dictionary with Pu neighbor analysis results
    """
    print("=" * 60)
    print("PU NEIGHBOR SHELL ANALYSIS")
    print("=" * 60)
    
    # High resolution radius analysis
    radius_range = np.linspace(0.5, max_radius, 201)  # 0.025 Å resolution
    
    # Initialize data storage for Pu neighbors
    pu_pu_neighbors = []    # Pu-Pu neighbors
    pu_na_neighbors = []    # Pu-Na neighbors  
    pu_cl_neighbors = []    # Pu-Cl neighbors
    pu_total_neighbors = [] # Total Pu neighbors
    
    print(f"Analyzing {len(radius_range)} radius points for Pu neighbors...")
    
    for r in radius_range:
        # Configure bonds with varying PuCl cutoff
        pair_cutoffs = {
            ("Pu","Cl"): float(r), ("Cl","Pu"): float(r),
            ("Na","Pu"): float(r), ("Pu","Na"): float(r),
            ("Pu","Pu"): float(r)
        }
        # configure_bonds_modifier_from_cutoffs(pipe, pair_cutoffs)
        data_temp = pipe.compute(pipe.source.num_frames - 1)
        
        # Get atom positions
        positions = data_temp.particles.positions
        
        # Analyze Pu neighbors
        pu_atoms = np.where(names == "Pu")[0]
        pu_pu_count = 0
        pu_na_count = 0
        pu_cl_count = 0
        pu_total_count = 0
        
        for pu_idx in pu_atoms:
            pu_pos = positions[pu_idx]
            
            # Count Pu-Pu neighbors
            pu_positions = positions[names == "Pu"]
            pu_distances = np.linalg.norm(pu_positions - pu_pos, axis=1)
            pu_pu_neighbors_at_r = np.sum((pu_distances <= r) & (pu_distances > 0))  # Exclude self
            pu_pu_count += pu_pu_neighbors_at_r
            
            # Count Pu-Na neighbors
            na_positions = positions[names == "Na"]
            na_distances = np.linalg.norm(na_positions - pu_pos, axis=1)
            pu_na_neighbors_at_r = np.sum(na_distances <= r)
            pu_na_count += pu_na_neighbors_at_r
            
            # Count Pu-Cl neighbors
            cl_positions = positions[names == "Cl"]
            cl_distances = np.linalg.norm(cl_positions - pu_pos, axis=1)
            pu_cl_neighbors_at_r = np.sum(cl_distances <= r)
            pu_cl_count += pu_cl_neighbors_at_r
            
            # Total neighbors
            pu_total_count += pu_pu_neighbors_at_r + pu_na_neighbors_at_r + pu_cl_neighbors_at_r
        
        # Average per Pu atom
        avg_pu_pu = pu_pu_count / len(pu_atoms) if len(pu_atoms) > 0 else 0
        avg_pu_na = pu_na_count / len(pu_atoms) if len(pu_atoms) > 0 else 0
        avg_pu_cl = pu_cl_count / len(pu_atoms) if len(pu_atoms) > 0 else 0
        avg_pu_total = pu_total_count / len(pu_atoms) if len(pu_atoms) > 0 else 0
        
        pu_pu_neighbors.append(avg_pu_pu)
        pu_na_neighbors.append(avg_pu_na)
        pu_cl_neighbors.append(avg_pu_cl)
        pu_total_neighbors.append(avg_pu_total)

    pu_pu_gradients = np.gradient(pu_pu_neighbors)
    pu_pu_second_deriv = np.gradient(pu_pu_gradients)
    pu_pu_shells = detect_proper_shells(pu_pu_gradients)

    pu_cl_gradients = np.gradient(pu_cl_neighbors)
    pu_cl_second_deriv = np.gradient(pu_cl_gradients)
    pu_cl_shells = detect_proper_shells(pu_cl_gradients)

    pu_total_gradients = np.gradient(pu_total_neighbors)
    pu_total_second_deriv = np.gradient(pu_total_gradients)
    pu_total_shells = detect_proper_shells(pu_total_gradients)
    

    return {
        'radius_range': radius_range,
        'pu_pu_neighbors': pu_pu_neighbors,
        'pu_na_neighbors': pu_na_neighbors,
        'pu_cl_neighbors': pu_cl_neighbors,
        'pu_total_neighbors': pu_total_neighbors,
        'pu_pu_shells': pu_pu_shells,
        'pu_cl_shells': pu_cl_shells,
        'pu_total_shells': pu_total_shells,
        'pu_atom_count': len(pu_atoms)
    }


def plot_shell_diagram(pu_shell_analysis):
    
    radius_range = pu_shell_analysis['radius_range']
    pu_pu_neighbors = pu_shell_analysis['pu_pu_neighbors']
    pu_na_neighbors = pu_shell_analysis['pu_na_neighbors']
    pu_cl_neighbors = pu_shell_analysis['pu_cl_neighbors']
    pu_total_neighbors = pu_shell_analysis['pu_total_neighbors']
    pu_pu_shells = pu_shell_analysis['pu_pu_shells']
    pu_cl_shells = pu_shell_analysis['pu_cl_shells']
    pu_total_shells = pu_shell_analysis['pu_total_shells']


    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All Pu neighbor types vs radius
    ax1.plot(radius_range, pu_pu_neighbors, 'b-', linewidth=2, label='Pu-Pu neighbors')
    ax1.plot(radius_range, pu_na_neighbors, 'g-', linewidth=2, label='Pu-Na neighbors')
    ax1.plot(radius_range, pu_cl_neighbors, 'r-', linewidth=2, label='Pu-Cl neighbors')
    ax1.plot(radius_range, pu_total_neighbors, 'purple', linewidth=2, label='Pu-Total neighbors')
    ax1.set_xlabel('PuCl Bond Cutoff Radius (Å)')
    ax1.set_ylabel('Average Neighbor Count per Pu')
    ax1.set_title('Pu Neighbors vs PuCl Bond Cutoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pu-Pu neighbors with shell detection
    ax2.plot(radius_range, pu_pu_neighbors, 'b-', linewidth=2, label='Pu-Pu neighbors')
    if len(pu_pu_shells) > 0:
        pu_pu_shell_radii = radius_range[pu_pu_shells]
        pu_pu_shell_counts = [pu_pu_neighbors[i] for i in pu_pu_shells]
        ax2.scatter(pu_pu_shell_radii, pu_pu_shell_counts, color='red', s=100, zorder=5, label='Pu-Pu shells')
        for i, (x, y) in enumerate(zip(pu_pu_shell_radii, pu_pu_shell_counts)):
            ax2.annotate(f'({x:.2f}, {y:.1f})', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, ha='left', va='bottom', color='red')
    ax2.set_xlabel('PuCl Bond Cutoff Radius (Å)')
    ax2.set_ylabel('Average Pu-Pu Neighbors')
    ax2.set_title('Pu-Pu Shell Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    

    # Plot 3: Pu-Cl neighbors with shell detection
    ax3.plot(radius_range, pu_cl_neighbors, 'r-', linewidth=2, label='Pu-Cl neighbors')
    if len(pu_cl_shells) > 0:
        pu_cl_shell_radii = radius_range[pu_cl_shells]
        pu_cl_shell_counts = [pu_cl_neighbors[i] for i in pu_cl_shells]
        ax3.scatter(pu_cl_shell_radii, pu_cl_shell_counts, color='red', s=100, zorder=5, label='Pu-Cl shells')
        for i, (x, y) in enumerate(zip(pu_cl_shell_radii, pu_cl_shell_counts)):
            ax3.annotate(f'({x:.2f}, {y:.1f})', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, ha='left', va='bottom', color='red')
    ax3.set_xlabel('PuCl Bond Cutoff Radius (Å)')
    ax3.set_ylabel('Average Pu-Cl Neighbors')
    ax3.set_title('Pu-Cl Shell Detection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total Pu neighbors with shell detection

    
    ax4.plot(radius_range, pu_total_neighbors, 'purple', linewidth=2, label='Pu-Total neighbors')
    if len(pu_total_shells) > 0:
        pu_total_shell_radii = radius_range[pu_total_shells]
        pu_total_shell_counts = [pu_total_neighbors[i] for i in pu_total_shells]
        ax4.scatter(pu_total_shell_radii, pu_total_shell_counts, color='red', s=100, zorder=5, label='Pu-Total shells')
        for i, (x, y) in enumerate(zip(pu_total_shell_radii, pu_total_shell_counts)):
            ax4.annotate(f'({x:.2f}, {y:.1f})', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, ha='left', va='bottom', color='red')
    ax4.set_xlabel('PuCl Bond Cutoff Radius (Å)')
    ax4.set_ylabel('Average Total Pu Neighbors')
    ax4.set_title('Pu-Total Shell Detection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Pu Neighbor Shell Analysis (Varying PuCl Bond Cutoff)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis results
    print(f"\nPU NEIGHBOR SHELL ANALYSIS RESULTS:")
    print(f"=" * 50)
    
    print(f"\nPu-Pu Neighbors:")
    print(f"  Max Pu-Pu neighbors: {max(pu_pu_neighbors):.2f}")
    print(f"  Shells detected: {len(pu_pu_shells)}")
    if len(pu_pu_shells) > 0:
        pu_pu_shell_radii = radius_range[pu_pu_shells]
        print(f"  Shell radii: {[f'{r:.3f}' for r in pu_pu_shell_radii]} Å")
    
    print(f"\nPu-Na Neighbors:")
    print(f"  Max Pu-Na neighbors: {max(pu_na_neighbors):.2f}")
    
    print(f"\nPu-Cl Neighbors:")
    print(f"  Max Pu-Cl neighbors: {max(pu_cl_neighbors):.2f}")
    print(f"  Shells detected: {len(pu_cl_shells)}")
    if len(pu_cl_shells) > 0:
        pu_cl_shell_radii = radius_range[pu_cl_shells]
        print(f"  Shell radii: {[f'{r:.3f}' for r in pu_cl_shell_radii]} Å")
    
    print(f"\nPu-Total Neighbors:")
    print(f"  Max Pu-Total neighbors: {max(pu_total_neighbors):.2f}")
    print(f"  Shells detected: {len(pu_total_shells)}")
    if len(pu_total_shells) > 0:
        pu_total_shell_radii = radius_range[pu_total_shells]
        print(f"  Shell radii: {[f'{r:.3f}' for r in pu_total_shell_radii]} Å")


# =============================================================================
# Neighbor Distribution Analysis (from ovito_bond_shell_analysis.ipynb)
# =============================================================================

def analyze_pu_neighbor_distributions(pipe, names, cutoffs, max_radius=10.0):
    """
    Build histograms showing Pu neighbor distributions at key distances.
    
    Shows coordination number distributions for Pu-Pu, Pu-Na, Pu-Cl, and Pu-Total
    neighbors at different PuCl bond cutoff radii.
    
    Args:
        pipe: OVITO pipeline object
        names: Array of atom names/types
        cutoffs: Dictionary of current cutoff values
        max_radius: Maximum radius to analyze
    
    Returns:
        Dictionary with neighbor distribution data
    """
    print("\n" + "=" * 60)
    print("PU NEIGHBOR DISTRIBUTION HISTOGRAMS")
    print("=" * 60)
    
    # Key distances to analyze with distinct colors and clear labels
    key_distances = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'brown']
    distance_labels = ['2.0 Å', '3.0 Å', '4.0 Å', '5.0 Å', '6.0 Å', '8.0 Å', '10.0 Å']
    
    # Store coordination distributions for each distance
    pu_pu_distributions = {}
    pu_na_distributions = {}
    pu_cl_distributions = {}
    pu_total_distributions = {}
    
    print(f"Analyzing Pu neighbor distributions at {len(key_distances)} key distances...")
    
    for r in key_distances:
        print(f"  Processing radius {r:.1f} Å...")
        
        # Set up bonds for this radius
        pair_cutoffs = {
            ("Pu","Cl"): float(r), ("Cl","Pu"): float(r),
            ("Na","Cl"): cutoffs.get(("Na","Cl"), 3.42), ("Cl","Na"): cutoffs.get(("Na","Cl"), 3.42),
        }
        # configure_bonds_modifier_from_cutoffs(pipe, pair_cutoffs)
        data_temp = pipe.compute(pipe.source.num_frames - 1)
        
        positions = data_temp.particles.positions
        
        # Analyze Pu neighbors
        pu_atoms = np.where(names == "Pu")[0]
        pu_pu_coords = []
        pu_na_coords = []
        pu_cl_coords = []
        pu_total_coords = []
        
        for pu_idx in pu_atoms:
            pu_pos = positions[pu_idx]
            
            # Count Pu-Pu neighbors
            pu_positions = positions[names == "Pu"]
            pu_distances = np.linalg.norm(pu_positions - pu_pos, axis=1)
            pu_pu_neighbors_at_r = np.sum((pu_distances <= r) & (pu_distances > 0))  # Exclude self
            pu_pu_coords.append(pu_pu_neighbors_at_r)
            
            # Count Pu-Na neighbors
            na_positions = positions[names == "Na"]
            na_distances = np.linalg.norm(na_positions - pu_pos, axis=1)
            pu_na_neighbors_at_r = np.sum(na_distances <= r)
            pu_na_coords.append(pu_na_neighbors_at_r)
            
            # Count Pu-Cl neighbors
            cl_positions = positions[names == "Cl"]
            cl_distances = np.linalg.norm(cl_positions - pu_pos, axis=1)
            pu_cl_neighbors_at_r = np.sum(cl_distances <= r)
            pu_cl_coords.append(pu_cl_neighbors_at_r)
            
            # Total neighbors
            pu_total_neighbors_at_r = pu_pu_neighbors_at_r + pu_na_neighbors_at_r + pu_cl_neighbors_at_r
            pu_total_coords.append(pu_total_neighbors_at_r)
        
        # Store distributions
        pu_pu_distributions[r] = pu_pu_coords
        pu_na_distributions[r] = pu_na_coords
        pu_cl_distributions[r] = pu_cl_coords
        pu_total_distributions[r] = pu_total_coords
    
    # Create histogram plots with clearer styling
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Pu-Pu coordination distributions
    ax1.set_title('Pu-Pu Coordination Distributions', fontsize=14, fontweight='bold')
    for i, r in enumerate(key_distances):
        coord_dist = pu_pu_distributions[r]
        if len(coord_dist) > 0:
            max_coord = max(coord_dist)
            bins = range(max_coord + 2)
            ax1.hist(coord_dist, bins=bins, alpha=0.7, color=colors[i], 
                    label=distance_labels[i], density=True, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Pu-Pu Coordination Number', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('white')
    
    # Plot 2: Pu-Na coordination distributions
    ax2.set_title('Pu-Na Coordination Distributions', fontsize=14, fontweight='bold')
    for i, r in enumerate(key_distances):
        coord_dist = pu_na_distributions[r]
        if len(coord_dist) > 0:
            max_coord = max(coord_dist)
            bins = range(max_coord + 2)
            ax2.hist(coord_dist, bins=bins, alpha=0.7, color=colors[i], 
                    label=distance_labels[i], density=True, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Pu-Na Coordination Number', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('white')
    
    # Plot 3: Pu-Cl coordination distributions
    ax3.set_title('Pu-Cl Coordination Distributions', fontsize=14, fontweight='bold')
    for i, r in enumerate(key_distances):
        coord_dist = pu_cl_distributions[r]
        if len(coord_dist) > 0:
            max_coord = max(coord_dist)
            bins = range(max_coord + 2)
            ax3.hist(coord_dist, bins=bins, alpha=0.7, color=colors[i], 
                    label=distance_labels[i], density=True, edgecolor='black', linewidth=0.8)
    ax3.set_xlabel('Pu-Cl Coordination Number', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('white')
    
    # Plot 4: Pu-Total coordination distributions
    ax4.set_title('Pu-Total Coordination Distributions', fontsize=14, fontweight='bold')
    for i, r in enumerate(key_distances):
        coord_dist = pu_total_distributions[r]
        if len(coord_dist) > 0:
            max_coord = max(coord_dist)
            bins = range(max_coord + 2)
            ax4.hist(coord_dist, bins=bins, alpha=0.7, color=colors[i], 
                    label=distance_labels[i], density=True, edgecolor='black', linewidth=0.8)
    ax4.set_xlabel('Pu-Total Coordination Number', fontsize=12)
    ax4.set_ylabel('Probability Density', fontsize=12)
    ax4.legend(fontsize=10, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('white')
    
    plt.suptitle('Pu Neighbor Coordination Distributions at Different Radii', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()
    
    # Create a separate legend reference plot
    fig_legend, ax_legend = plt.subplots(figsize=(8, 2))
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis('off')
    
    # Create legend patches
    legend_elements = []
    for i, (r, color) in enumerate(zip(key_distances, colors)):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, 
                                           edgecolor='black', linewidth=0.8))
    
    ax_legend.legend(legend_elements, distance_labels, loc='center', ncol=len(key_distances), 
                   fontsize=12, framealpha=0.9)
    ax_legend.set_title('Distance Legend', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Print distribution statistics
    print(f"\nPU NEIGHBOR DISTRIBUTION STATISTICS:")
    print(f"=" * 50)
    
    for r in key_distances:
        print(f"\nAt {r:.1f} Å:")
        print(f"  Pu-Pu: mean={np.mean(pu_pu_distributions[r]):.2f}, std={np.std(pu_pu_distributions[r]):.2f}, range={min(pu_pu_distributions[r])}-{max(pu_pu_distributions[r])}")
        print(f"  Pu-Na: mean={np.mean(pu_na_distributions[r]):.2f}, std={np.std(pu_na_distributions[r]):.2f}, range={min(pu_na_distributions[r])}-{max(pu_na_distributions[r])}")
        print(f"  Pu-Cl: mean={np.mean(pu_cl_distributions[r]):.2f}, std={np.std(pu_cl_distributions[r]):.2f}, range={min(pu_cl_distributions[r])}-{max(pu_cl_distributions[r])}")
        print(f"  Total: mean={np.mean(pu_total_distributions[r]):.2f}, std={np.std(pu_total_distributions[r]):.2f}, range={min(pu_total_distributions[r])}-{max(pu_total_distributions[r])}")
    
    return {
        'key_distances': key_distances,
        'pu_pu_distributions': pu_pu_distributions,
        'pu_na_distributions': pu_na_distributions,
        'pu_cl_distributions': pu_cl_distributions,
        'pu_total_distributions': pu_total_distributions,
        'colors': colors,
        'distance_labels': distance_labels
    }


# =============================================================================
# Coordination Derivatives Analysis (from ovito_bond_shell_analysis.ipynb)
# =============================================================================

def plot_coordination_derivatives_2nd_order(pu_shell_analysis):
    """
    Plot first and second derivatives of Pu coordination curves to visualize shell detection.
    
    Shows how the derivatives change with radius and where shells are detected.
    Shell boundaries are detected where first derivative peaks then goes to zero (proper shell formation pattern).
    
    Args:
        pu_shell_analysis: Results from analyze_pu_neighbor_shells function
    """
    print("\n" + "=" * 60)
    print("COORDINATION DERIVATIVES ANALYSIS")
    print("=" * 60)
    
    radius_range = pu_shell_analysis['radius_range']
    pu_pu_neighbors = pu_shell_analysis['pu_pu_neighbors']
    pu_na_neighbors = pu_shell_analysis['pu_na_neighbors']
    pu_cl_neighbors = pu_shell_analysis['pu_cl_neighbors']
    pu_total_neighbors = pu_shell_analysis['pu_total_neighbors']
    
    # Calculate first derivatives
    pu_pu_first_deriv = np.gradient(pu_pu_neighbors)
    pu_na_first_deriv = np.gradient(pu_na_neighbors)
    pu_cl_first_deriv = np.gradient(pu_cl_neighbors)
    pu_total_first_deriv = np.gradient(pu_total_neighbors)
    
    # Calculate second derivatives
    pu_pu_second_deriv = np.gradient(pu_pu_first_deriv)
    pu_na_second_deriv = np.gradient(pu_na_first_deriv)
    pu_cl_second_deriv = np.gradient(pu_cl_first_deriv)
    pu_total_second_deriv = np.gradient(pu_total_first_deriv)
    
    # Detect shells using proper pattern
    pu_pu_shells = detect_proper_shells(pu_pu_first_deriv)
    pu_na_shells = detect_proper_shells(pu_na_first_deriv)
    pu_cl_shells = detect_proper_shells(pu_cl_first_deriv)
    pu_total_shells = detect_proper_shells(pu_total_first_deriv)
    
    # Create derivative plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Pu-Pu derivatives
    ax1.plot(radius_range, pu_pu_first_deriv, 'b-', linewidth=2, label='1st derivative', alpha=0.7)
    ax1.plot(radius_range, pu_pu_second_deriv, 'r-', linewidth=2, label='2nd derivative', alpha=0.7)
    
    # Mark detected shells (where first derivative goes to zero after peaking)
    if len(pu_pu_shells) > 0:
        shell_radii = radius_range[pu_pu_shells]
        shell_first_derivs = pu_pu_first_deriv[pu_pu_shells]
        ax1.scatter(shell_radii, shell_first_derivs, color='blue', s=100, zorder=5, 
                  label=f'Shells ({len(pu_pu_shells)})', marker='o')
    
    # Mark peaks in first derivative
    pu_pu_peaks, _ = find_peaks(pu_pu_first_deriv, prominence=0.1, distance=10)
    if len(pu_pu_peaks) > 0:
        peak_radii = radius_range[pu_pu_peaks]
        peak_derivs = pu_pu_first_deriv[pu_pu_peaks]
        ax1.scatter(peak_radii, peak_derivs, color='red', s=80, zorder=5, 
                   label=f'Peaks ({len(pu_pu_peaks)})', marker='^', alpha=0.7)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('PuCl Bond Cutoff Radius (Å)', fontsize=12)
    ax1.set_ylabel('Derivative Value', fontsize=12)
    ax1.set_title('Pu-Pu Coordination Derivatives', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('white')
    
    # Plot 2: Pu-Na derivatives
    ax2.plot(radius_range, pu_na_first_deriv, 'g-', linewidth=2, label='1st derivative', alpha=0.7)
    ax2.plot(radius_range, pu_na_second_deriv, 'orange', linewidth=2, label='2nd derivative', alpha=0.7)
    
    # Mark detected shells
    if len(pu_na_shells) > 0:
        shell_radii = radius_range[pu_na_shells]
        shell_first_derivs = pu_na_first_deriv[pu_na_shells]
        ax2.scatter(shell_radii, shell_first_derivs, color='green', s=100, zorder=5, 
                  label=f'Shells ({len(pu_na_shells)})', marker='o')
    
    # Mark peaks in first derivative
    pu_na_peaks, _ = find_peaks(pu_na_first_deriv, prominence=0.1, distance=10)
    if len(pu_na_peaks) > 0:
        peak_radii = radius_range[pu_na_peaks]
        peak_derivs = pu_na_first_deriv[pu_na_peaks]
        ax2.scatter(peak_radii, peak_derivs, color='orange', s=80, zorder=5, 
                   label=f'Peaks ({len(pu_na_peaks)})', marker='^', alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('PuCl Bond Cutoff Radius (Å)', fontsize=12)
    ax2.set_ylabel('Derivative Value', fontsize=12)
    ax2.set_title('Pu-Na Coordination Derivatives', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('white')
    
    # Plot 3: Pu-Cl derivatives
    ax3.plot(radius_range, pu_cl_first_deriv, 'purple', linewidth=2, label='1st derivative', alpha=0.7)
    ax3.plot(radius_range, pu_cl_second_deriv, 'red', linewidth=2, label='2nd derivative', alpha=0.7)
    
    # Mark detected shells
    if len(pu_cl_shells) > 0:
        shell_radii = radius_range[pu_cl_shells]
        shell_first_derivs = pu_cl_first_deriv[pu_cl_shells]
        ax3.scatter(shell_radii, shell_first_derivs, color='purple', s=100, zorder=5, 
                  label=f'Shells ({len(pu_cl_shells)})', marker='o')
    
    # Mark peaks in first derivative
    pu_cl_peaks, _ = find_peaks(pu_cl_first_deriv, prominence=0.1, distance=10)
    if len(pu_cl_peaks) > 0:
        peak_radii = radius_range[pu_cl_peaks]
        peak_derivs = pu_cl_first_deriv[pu_cl_peaks]
        ax3.scatter(peak_radii, peak_derivs, color='red', s=80, zorder=5, 
                   label=f'Peaks ({len(pu_cl_peaks)})', marker='^', alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('PuCl Bond Cutoff Radius (Å)', fontsize=12)
    ax3.set_ylabel('Derivative Value', fontsize=12)
    ax3.set_title('Pu-Cl Coordination Derivatives', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('white')
    
    # Plot 4: Pu-Total derivatives
    ax4.plot(radius_range, pu_total_first_deriv, 'brown', linewidth=2, label='1st derivative', alpha=0.7)
    ax4.plot(radius_range, pu_total_second_deriv, 'darkred', linewidth=2, label='2nd derivative', alpha=0.7)
    
    # Mark detected shells
    if len(pu_total_shells) > 0:
        shell_radii = radius_range[pu_total_shells]
        shell_first_derivs = pu_total_first_deriv[pu_total_shells]
        ax4.scatter(shell_radii, shell_first_derivs, color='brown', s=100, zorder=5, 
                  label=f'Shells ({len(pu_total_shells)})', marker='o')
    
    # Mark peaks in first derivative
    pu_total_peaks, _ = find_peaks(pu_total_first_deriv, prominence=0.1, distance=10)
    if len(pu_total_peaks) > 0:
        peak_radii = radius_range[pu_total_peaks]
        peak_derivs = pu_total_first_deriv[pu_total_peaks]
        ax4.scatter(peak_radii, peak_derivs, color='darkred', s=80, zorder=5, 
                   label=f'Peaks ({len(pu_total_peaks)})', marker='^', alpha=0.7)
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('PuCl Bond Cutoff Radius (Å)', fontsize=12)
    ax4.set_ylabel('Derivative Value', fontsize=12)
    ax4.set_title('Pu-Total Coordination Derivatives', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('white')
    
    plt.suptitle('Coordination Derivatives and Shell Detection (Shells = Peak → Zero Pattern)', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()
    
    # Print derivative analysis
    print(f"\nPROPER SHELL DETECTION ANALYSIS:")
    print(f"=" * 50)
    
    derivative_data = {
        'Pu-Pu': {'first': pu_pu_first_deriv, 'shells': pu_pu_shells, 'peaks': pu_pu_peaks},
        'Pu-Na': {'first': pu_na_first_deriv, 'shells': pu_na_shells, 'peaks': pu_na_peaks},
        'Pu-Cl': {'first': pu_cl_first_deriv, 'shells': pu_cl_shells, 'peaks': pu_cl_peaks},
        'Pu-Total': {'first': pu_total_first_deriv, 'shells': pu_total_shells, 'peaks': pu_total_peaks}
    }
    
    for coord_type, data in derivative_data.items():
        print(f"\n{coord_type} Coordination:")
        print(f"  Derivative peaks found: {len(data['peaks'])}")
        print(f"  Shell boundaries detected: {len(data['shells'])}")
        
        if len(data['peaks']) > 0:
            peak_radii = radius_range[data['peaks']]
            print(f"  Peak radii: {[f'{r:.3f}' for r in peak_radii]} Å")
        
        if len(data['shells']) > 0:
            shell_radii = radius_range[data['shells']]
            print(f"  Shell radii: {[f'{r:.3f}' for r in shell_radii]} Å")
    
    return {
        'radius_range': radius_range,
        'pu_pu_first_deriv': pu_pu_first_deriv,
        'pu_pu_second_deriv': pu_pu_second_deriv,
        'pu_na_first_deriv': pu_na_first_deriv,
        'pu_na_second_deriv': pu_na_second_deriv,
        'pu_cl_first_deriv': pu_cl_first_deriv,
        'pu_cl_second_deriv': pu_cl_second_deriv,
        'pu_total_first_deriv': pu_total_first_deriv,
        'pu_total_second_deriv': pu_total_second_deriv,
        'shells': {
            'pu_pu': pu_pu_shells,
            'pu_na': pu_na_shells,
            'pu_cl': pu_cl_shells,
            'pu_total': pu_total_shells
        },
        'peaks': {
            'pu_pu': pu_pu_peaks,
            'pu_na': pu_na_peaks,
            'pu_cl': pu_cl_peaks,
            'pu_total': pu_total_peaks
        }
    }


def plot_coordination_with_derivatives(pu_shell_analysis,smooth=False):
    """
    Plot coordination curves with overlayed first derivatives to visualize shell formation.
    
    Shows coordination on left y-axis and first derivative on right y-axis.
    Shell boundaries are detected using proper pattern: derivative peaks then goes to zero.
    
    Args:
        pu_shell_analysis: Results from analyze_pu_neighbor_shells function
    """
    print("\n" + "=" * 60)
    print("COORDINATION WITH FIRST DERIVATIVE OVERLAY")
    print("=" * 60)
    
    radius_range = pu_shell_analysis['radius_range']
    pu_pu_neighbors = pu_shell_analysis['pu_pu_neighbors']
    pu_na_neighbors = pu_shell_analysis['pu_na_neighbors']
    pu_cl_neighbors = pu_shell_analysis['pu_cl_neighbors']
    pu_total_neighbors = pu_shell_analysis['pu_total_neighbors']
    
    # Calculate first derivatives
    pu_pu_first_deriv = np.gradient(pu_pu_neighbors)
    pu_na_first_deriv = np.gradient(pu_na_neighbors)
    pu_cl_first_deriv = np.gradient(pu_cl_neighbors)
    pu_total_first_deriv = np.gradient(pu_total_neighbors)

    pu_pu_second_deriv = np.gradient(pu_pu_first_deriv)
    pu_na_second_deriv = np.gradient(pu_na_first_deriv)
    pu_cl_second_deriv = np.gradient(pu_cl_first_deriv)
    pu_total_second_deriv = np.gradient(pu_total_first_deriv)

    if smooth:
        pu_pu_first_deriv = gaussian_smooth(pu_pu_first_deriv, sigma=3)
        pu_na_first_deriv = gaussian_smooth(pu_na_first_deriv, sigma=3)
        pu_cl_first_deriv = gaussian_smooth(pu_cl_first_deriv, sigma=3)
        pu_total_first_deriv = gaussian_smooth(pu_total_first_deriv, sigma=3)

        pu_pu_second_deriv = gaussian_smooth(pu_pu_second_deriv, sigma=3)
        pu_na_second_deriv = gaussian_smooth(pu_na_second_deriv, sigma=3)
        pu_cl_second_deriv = gaussian_smooth(pu_cl_second_deriv, sigma=3)
        pu_total_second_deriv = gaussian_smooth(pu_total_second_deriv, sigma=3)
    
    # Detect shells using proper pattern
    pu_pu_shells = detect_proper_shells(pu_pu_first_deriv)
    pu_na_shells = detect_proper_shells(pu_na_first_deriv)
    pu_cl_shells = detect_proper_shells(pu_cl_first_deriv)
    pu_total_shells = detect_proper_shells(pu_total_first_deriv)
    
    # Create overlay plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Pu-Pu coordination with first derivative
    ax1_twin = ax1.twinx()
    
    # Coordination curve
    ax1.plot(radius_range, pu_pu_neighbors, 'b-', linewidth=5, label='Pu-Pu coordination', alpha=0.8)
    ax1.set_xlabel('PuCl Bond Cutoff Radius (Å)', fontsize=12)
    ax1.set_ylabel('Pu-Pu Coordination Number', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # First derivative
    ax1_twin.plot(radius_range, pu_pu_first_deriv, 'red', linewidth=2, label='1st derivative', alpha=0.7)
    ax1_twin.plot(radius_range, pu_pu_second_deriv, 'red', linewidth=2, label='2nd derivative', alpha=0.2)
    ax1_twin.set_ylabel('First & Second Derivative', fontsize=12, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Mark shell boundaries
    if len(pu_pu_shells) > 0:
        shell_radii = radius_range[pu_pu_shells]
        shell_coords = [pu_pu_neighbors[i] for i in pu_pu_shells]
        ax1.scatter(shell_radii, shell_coords, color='blue', s=100, zorder=5, 
                   label=f'Shells ({len(pu_pu_shells)})', marker='o', edgecolor='black')
    
    # Mark peaks in first derivative
    pu_pu_peaks, _ = find_peaks(pu_pu_first_deriv, prominence=0.1, distance=10)
    if len(pu_pu_peaks) > 0:
        peak_radii = radius_range[pu_pu_peaks]
        peak_derivs = pu_pu_first_deriv[pu_pu_peaks]
        ax1_twin.scatter(peak_radii, peak_derivs, color='red', s=80, zorder=5, 
                        label=f'Peaks ({len(pu_pu_peaks)})', marker='^', alpha=0.7)
    
    ax1_twin.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Pu-Pu Coordination with First Derivative', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1_twin.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Plot 2: Pu-Na coordination with first derivative
    ax2_twin = ax2.twinx()
    
    # Coordination curve
    ax2.plot(radius_range, pu_na_neighbors, 'g-', linewidth=3, label='Pu-Na coordination', alpha=0.8)
    ax2.set_xlabel('PuCl Bond Cutoff Radius (Å)', fontsize=12)
    ax2.set_ylabel('Pu-Na Coordination Number', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # First derivative
    ax2_twin.plot(radius_range, pu_na_first_deriv, 'orange', linewidth=2, label='1st derivative', alpha=0.7)
    ax2_twin.plot(radius_range, pu_na_second_deriv, 'orange', linewidth=2, label='2nd derivative', alpha=0.2)
    ax2_twin.set_ylabel('First & Second Derivative', fontsize=12, color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    # Mark shell boundaries
    if len(pu_na_shells) > 0:
        shell_radii = radius_range[pu_na_shells]
        shell_coords = [pu_na_neighbors[i] for i in pu_na_shells]
        ax2.scatter(shell_radii, shell_coords, color='green', s=100, zorder=5, 
                   label=f'Shells ({len(pu_na_shells)})', marker='o', edgecolor='black')
    
    # Mark peaks in first derivative
    pu_na_peaks, _ = find_peaks(pu_na_first_deriv, prominence=0.1, distance=10)
    if len(pu_na_peaks) > 0:
        peak_radii = radius_range[pu_na_peaks]
        peak_derivs = pu_na_first_deriv[pu_na_peaks]
        ax2_twin.scatter(peak_radii, peak_derivs, color='orange', s=80, zorder=5, 
                        label=f'Peaks ({len(pu_na_peaks)})', marker='^', alpha=0.7)
    
    ax2_twin.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Pu-Na Coordination with First Derivative', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2_twin.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Plot 3: Pu-Cl coordination with first derivative
    ax3_twin = ax3.twinx()
    
    # Coordination curve
    ax3.plot(radius_range, pu_cl_neighbors, 'purple', linewidth=3, label='Pu-Cl coordination', alpha=0.8)
    ax3.set_xlabel('PuCl Bond Cutoff Radius (Å)', fontsize=12)
    ax3.set_ylabel('Pu-Cl Coordination Number', fontsize=12, color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    
    # First derivative
    ax3_twin.plot(radius_range, pu_cl_first_deriv, 'red', linewidth=2, label='1st derivative', alpha=0.7)
    ax3_twin.plot(radius_range, pu_cl_second_deriv, 'red', linewidth=2, label='2nd derivative', alpha=0.2)
    ax3_twin.set_ylabel('First & Second Derivative', fontsize=12, color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    # Mark shell boundaries
    if len(pu_cl_shells) > 0:
        shell_radii = radius_range[pu_cl_shells]
        shell_coords = [pu_cl_neighbors[i] for i in pu_cl_shells]
        ax3.scatter(shell_radii, shell_coords, color='purple', s=100, zorder=5, 
                   label=f'Shells ({len(pu_cl_shells)})', marker='o', edgecolor='black')
    
    # Mark peaks in first derivative
    pu_cl_peaks, _ = find_peaks(pu_cl_first_deriv, prominence=0.1, distance=10)
    if len(pu_cl_peaks) > 0:
        peak_radii = radius_range[pu_cl_peaks]
        peak_derivs = pu_cl_first_deriv[pu_cl_peaks]
        ax3_twin.scatter(peak_radii, peak_derivs, color='red', s=80, zorder=5, 
                        label=f'Peaks ({len(pu_cl_peaks)})', marker='^', alpha=0.7)
    
    ax3_twin.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Pu-Cl Coordination with First Derivative', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax3_twin.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Plot 4: Pu-Total coordination with first derivative
    ax4_twin = ax4.twinx()
    
    # Coordination curve
    ax4.plot(radius_range, pu_total_neighbors, 'brown', linewidth=3, label='Pu-Total coordination', alpha=0.8)
    ax4.set_xlabel('PuCl Bond Cutoff Radius (Å)', fontsize=12)
    ax4.set_ylabel('Pu-Total Coordination Number', fontsize=12, color='brown')
    ax4.tick_params(axis='y', labelcolor='brown')
    
    # First derivative
    ax4_twin.plot(radius_range, pu_total_first_deriv, 'darkred', linewidth=2, label='1st derivative', alpha=0.7)
    ax4_twin.plot(radius_range, pu_total_second_deriv, 'darkred', linewidth=2, label='2nd derivative', alpha=0.2)
    ax4_twin.set_ylabel('First Derivative', fontsize=12, color='darkred')
    ax4_twin.tick_params(axis='y', labelcolor='darkred')
    
    # Mark shell boundaries
    if len(pu_total_shells) > 0:
        shell_radii = radius_range[pu_total_shells]
        shell_coords = [pu_total_neighbors[i] for i in pu_total_shells]
        ax4.scatter(shell_radii, shell_coords, color='brown', s=100, zorder=5, 
                   label=f'Shells ({len(pu_total_shells)})', marker='o', edgecolor='black')
    
    # Mark peaks in first derivative
    pu_total_peaks, _ = find_peaks(pu_total_first_deriv, prominence=0.1, distance=10)
    if len(pu_total_peaks) > 0:
        peak_radii = radius_range[pu_total_peaks]
        peak_derivs = pu_total_first_deriv[pu_total_peaks]
        ax4_twin.scatter(peak_radii, peak_derivs, color='darkred', s=80, zorder=5, 
                        label=f'Peaks ({len(pu_total_peaks)})', marker='^', alpha=0.7)
    
    ax4_twin.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Pu-Total Coordination with First Derivative', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax4_twin.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.suptitle('Coordination Curves with First Derivative Overlay (Shells = Peak → Zero Pattern)', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()
    
    # Print shell analysis
    print(f"\nPROPER SHELL BOUNDARY ANALYSIS:")
    print(f"=" * 50)
    
    shell_data = {
        'Pu-Pu': {'shells': pu_pu_shells, 'coords': pu_pu_neighbors, 'peaks': pu_pu_peaks},
        'Pu-Na': {'shells': pu_na_shells, 'coords': pu_na_neighbors, 'peaks': pu_na_peaks},
        'Pu-Cl': {'shells': pu_cl_shells, 'coords': pu_cl_neighbors, 'peaks': pu_cl_peaks},
        'Pu-Total': {'shells': pu_total_shells, 'coords': pu_total_neighbors, 'peaks': pu_total_peaks}
    }
    
    for coord_type, data in shell_data.items():
        print(f"\n{coord_type} Coordination:")
        print(f"  Derivative peaks found: {len(data['peaks'])}")
        print(f"  Shell boundaries detected: {len(data['shells'])}")
        
        if len(data['peaks']) > 0:
            peak_radii = radius_range[data['peaks']]
            print(f"  Peak radii: {[f'{r:.3f}' for r in peak_radii]} Å")
        
        if len(data['shells']) > 0:
            shell_radii = radius_range[data['shells']]
            shell_coords = [data['coords'][i] for i in data['shells']]
            print(f"  Shell radii: {[f'{r:.3f}' for r in shell_radii]} Å")
            print(f"  Coordination at shells: {[f'{c:.1f}' for c in shell_coords]}")
    
    return {
        'radius_range': radius_range,
        'shells': {
            'pu_pu': pu_pu_shells,
            'pu_na': pu_na_shells,
            'pu_cl': pu_cl_shells,
            'pu_total': pu_total_shells
        },
        'peaks': {
            'pu_pu': pu_pu_peaks,
            'pu_na': pu_na_peaks,
            'pu_cl': pu_cl_peaks,
            'pu_total': pu_total_peaks
        }
    }


# =============================================================================
# Bond Coordination Analysis (from rdf_integrate.ipynb)
# =============================================================================

def analyze_bond_coordination(data, names, pair_cutoffs=None):
    """
    Analyze coordination using bond-based analysis (similar to analyze_voronoi_coordination).
    
    This function computes neighbor-count distributions by species using bond cutoffs
    instead of Voronoi tessellation, making it suitable for bond-based cluster analysis.
    
    Parameters:
        data: OVITO data object containing particle information
        names: Array of atom names/types
        pair_cutoffs: Dictionary of bond cutoffs for different species pairs
    
    Returns:
        dict: Coordination data organized by central species and neighbor species
              Includes additional 'any' coordination structures:
              - coord_data[sp_c]['any']: total coordination for each central species
              - coord_data['any'][sp_n]: coordination from any central species to each neighbor species
              - coord_data['any']['any']: total coordination across all species pairs
    """
    import numpy as np
    from collections import defaultdict
    
    # Get unique species
    unique_species = sorted(list(set(names)))
    print(f"Analyzing coordination for species: {unique_species}")
    
    # Initialize coordination data structure
    coord_data = {sp_c: {sp_n: [] for sp_n in unique_species} for sp_c in unique_species}
    
    # Add 'any' coordination structures
    coord_data['any'] = {sp_n: [] for sp_n in unique_species}
    coord_data['any']['any'] = []
    
    # Get positions and create species mapping
    positions = data.particles.positions[:]
    species_idx = {sp: i for i, sp in enumerate(unique_species)}
    
    # Create species array for each atom
    atom_species = np.array([species_idx[name] for name in names])
    
    # Calculate coordination for each atom
    for i, (pos_i, sp_i) in enumerate(zip(positions, names)):
        neighbors = []
        
        # Find neighbors within cutoff distance
        for j, (pos_j, sp_j) in enumerate(zip(positions, names)):
            if i == j:
                continue
                
            # Calculate distance
            dist = np.linalg.norm(pos_j - pos_i)
            
            # Check if within cutoff (use default or specific pair cutoff)
            if pair_cutoffs:
                cutoff_key = (sp_i, sp_j)
                if cutoff_key in pair_cutoffs:
                    cutoff = pair_cutoffs[cutoff_key]
                else:
                    # Use default cutoff if specific pair not found
                    cutoff = 4.0  # Default cutoff
            else:
                cutoff = 4.0  # Default cutoff
                
            if dist <= cutoff:
                neighbors.append(sp_j)
        
        # Store coordination counts for specific species pairs
        for neighbor_sp in unique_species:
            count = neighbors.count(neighbor_sp)
            coord_data[sp_i][neighbor_sp].append(count)
        
        # Store total coordination for this central species (any neighbors)
        total_coord = len(neighbors)
        coord_data[sp_i]['any'] = coord_data[sp_i].get('any', [])
        coord_data[sp_i]['any'].append(total_coord)
        
        # Store coordination counts for 'any' central species to each neighbor species
        for neighbor_sp in unique_species:
            count = neighbors.count(neighbor_sp)
            coord_data['any'][neighbor_sp].append(count)
        
        # Store total coordination for 'any'-'any' (all coordination)
        coord_data['any']['any'].append(total_coord)
    
    return coord_data


def plot_bond_coordination_histograms(coord_data, central_species, title_suffix=""):
    """
    Plot coordination histograms for bond-based analysis.
    
    Parameters:
        coord_data: Coordination data from analyze_bond_coordination
        central_species: Species to plot coordination for
        title_suffix: Optional suffix for plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if central_species not in coord_data:
        print(f"No coordination data found for species: {central_species}")
        return
    
    # Get coordination data for the central species
    species_data = coord_data[central_species]
    print(species_data)
    
    # Filter out empty neighbor species
    neighbor_species = [sp for sp, counts in species_data.items() if counts and any(counts)]
    print(neighbor_species)
    
    if not neighbor_species:
        print(f"No coordination data found for {central_species}")
        return
    
    # Create subplots
    n_neighbors = len(neighbor_species)
    fig, axes = plt.subplots(1, n_neighbors, figsize=(5*n_neighbors, 4))
    if n_neighbors == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, neighbor_sp in enumerate(neighbor_species):
        counts = species_data[neighbor_sp]
        
        if not counts:
            continue
            
        # Create histogram
        max_coord = max(counts) if counts else 0
        bins = np.arange(0, max_coord + 2) - 0.5
        
        axes[idx].hist(counts, bins=bins, alpha=0.7, color=colors[idx % len(colors)], 
                      edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean_coord = np.mean(counts)
        std_coord = np.std(counts)
        axes[idx].axvline(mean_coord, color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {mean_coord:.2f} ± {std_coord:.2f}')
        
        axes[idx].set_xlabel(f'{central_species}-{neighbor_sp} Coordination Number')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{central_species}-{neighbor_sp} Coordination{title_suffix}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        # Print statistics
        print(f'{central_species}-{neighbor_sp}: {mean_coord:.2f} ± {std_coord:.2f} (n={len(counts)})')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Plotting Functions (from rdf_integrate.ipynb)
# =============================================================================

def plot_coordination_number_vs_r(r_values, cn_values, pair_name, r_max=None, ax=None, **plot_kwargs):
    """
    Plot coordination number as a function of radial distance.
    
    Version from rdf_integrate.ipynb
    
    Parameters:
    -----------
    r_values : array
        Radial distances
    cn_values : array
        Coordination number values
    pair_name : str
        Name of the species pair (e.g., 'Pu-Cl')
    r_max : float, optional
        Maximum r value to plot
    ax : matplotlib.axes, optional
        Axes to plot on. If None, creates new figure.
    **plot_kwargs : dict
        Additional plotting arguments passed to plt.plot()
    """
    import matplotlib.pyplot as plt
    
    # Determine plotting range
    if r_max is not None:
        mask = r_values <= r_max
        r_plot = r_values[mask]
        cn_plot = cn_values[mask]
    else:
        r_plot = r_values
        cn_plot = cn_values
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Default plotting parameters
    default_kwargs = {
        'linewidth': 2,
        'color': 'blue',
        'label': f'CN({pair_name})'
    }
    default_kwargs.update(plot_kwargs)
    
    ax.plot(r_plot, cn_plot, **default_kwargs)
    
    # Add horizontal line at final coordination number
    final_cn = cn_plot[-1]
    ax.axhline(y=final_cn, color='red', linestyle='--', alpha=0.7, 
               label=f'Final CN = {final_cn:.3f}')
    
    ax.set_xlabel('Radial Distance r (Å)')
    ax.set_ylabel('Coordination Number CN(r)')
    ax.set_title(f'Coordination Number vs Radial Distance: {pair_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax


def plot_coordination_number_vs_r_with_shell(r_values, cn_values, pair_name, shell_cutoff=None, r_max=None, ax=None, **plot_kwargs):
    """
    Plot coordination number as a function of radial distance.
    
    Version from rdf_shell_analysis.ipynb with shell cutoff support
    
    Parameters:
    -----------
    r_values : array
        Radial distances
    cn_values : array
        Coordination number values
    pair_name : str
        Name of the species pair (e.g., 'Pu-Cl')
    shell_cutoff : int, optional
        Index for shell-based coordination number
    r_max : float, optional
        Maximum r value to plot
    ax : matplotlib.axes, optional
        Axes to plot on. If None, creates new figure.
    **plot_kwargs : dict
        Additional plotting arguments passed to plt.plot()
    """
    
    # Determine plotting range
    if r_max is not None:
        mask = r_values <= r_max
        r_plot = r_values[mask]
        cn_plot = cn_values[mask]
    else:
        r_plot = r_values
        cn_plot = cn_values
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Default plotting parameters
    default_kwargs = {
        'linewidth': 2,
        'color': 'blue',
        'label': f'CN({pair_name})'
    }
    default_kwargs.update(plot_kwargs)
    
    ax.plot(r_plot, cn_plot, **default_kwargs)
    
    # Add horizontal line at final coordination number
    if shell_cutoff is not None:

        final_cn = cn_plot[shell_cutoff]
        ax.axhline(y=final_cn, color='red', linestyle='--', alpha=0.7,
        label=f'Shell based CN = {final_cn:.3f}')
    
    ax.set_xlabel('Radial Distance r (Å)')
    ax.set_ylabel('Coordination Number CN(r)')
    ax.set_title(f'Coordination Number vs Radial Distance: {pair_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    
    
    return ax


def plot_multiple_cn_vs_r(cn_data_dict, r_max=None, figsize=(12, 8)):
    """
    Plot multiple coordination number curves on the same figure.
    
    Parameters:
    -----------
    cn_data_dict : dict
        Dictionary with pair names as keys and (r_values, cn_values) tuples as values
    r_max : float, optional
        Maximum r value to plot
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    for i, (pair_name, (r_values, cn_values)) in enumerate(cn_data_dict.items()):
        color = colors[i % len(colors)]
        plot_coordination_number_vs_r(r_values, cn_values, pair_name, r_max=r_max, 
                                    ax=ax, color=color, linewidth=2)
    
    ax.set_title('Coordination Number vs Radial Distance (Multiple Pairs)')
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def save_cn_r_data(cn_r_data, filename, pair_name=None):
    """
    Save CN(r) data to a file for further analysis.
    
    Parameters:
    -----------
    cn_r_data : dict or tuple
        Either a dictionary of CN(r) data or a single (r_values, cn_values) tuple
    filename : str
        Output filename
    pair_name : str, optional
        Name of the species pair (for single data)
    """
    import numpy as np
    import json
    
    if isinstance(cn_r_data, tuple):
        # Single pair data
        r_values, cn_values = cn_r_data
        data_to_save = {
            'pair_name': pair_name or 'unknown',
            'r_values': r_values.tolist(),
            'cn_values': cn_values.tolist(),
            'final_cn': float(cn_values[-1]),
            'r_max': float(r_values[-1])
        }
    else:
        # Multiple pairs data
        data_to_save = {}
        for pair, (r_values, cn_values) in cn_r_data.items():
            data_to_save[pair] = {
                'r_values': r_values.tolist(),
                'cn_values': cn_values.tolist(),
                'final_cn': float(cn_values[-1]),
                'r_max': float(r_values[-1])
            }
    
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"CN(r) data saved to: {filename}")

