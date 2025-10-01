"""
Utilities for RDF-based bond cutoff detection, OVITO bond creation, and
graph/cluster analysis built around shared-anion connectivity.

This module centralizes logic that was previously embedded in notebooks like
`ovito_personality.ipynb`, `ovito_personality_no_NaCl.ipynb`, and
`arxiv/ovito_personality_no_PuCl.ipynb`.

Key capabilities:
- Compute partial RDFs using OVITO's CoordinationAnalysisModifier
- Detect first-minimum cutoffs without SciPy
- Configure CreateBondsModifier with pairwise cutoffs (including disabling pairs)
- Build a metal connectivity graph via shared anions (e.g., Cl)
- Summarize bonds and provide quick plotting hooks (delegates to plot_utils)
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List

import numpy as np

try:
    from ovito.io import import_file  # noqa: F401  # re-export convenience
    from ovito.modifiers import CoordinationAnalysisModifier, CreateBondsModifier
    from ovito.data import BondsEnumerator
except Exception as exc:  # pragma: no cover - runtime environment may not have ovito
    raise ImportError("OVITO is required for bondmodifier_utils") from exc

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    raise ImportError("networkx is required for graph construction") from exc

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    find_peaks = None  # type: ignore

# Optional plotting imports are deferred to plot_utils to keep a single style
try:
    from .plot_utils import plot_cluster_size_distribution, analyze_bond_network
except Exception:
    # Fallback: allow import without plot_utils, but plotting helpers will be unavailable
    plot_cluster_size_distribution = None  # type: ignore
    analyze_bond_network = None  # type: ignore


def compute_partial_rdfs(pipeline: Any, nsamples: int = 100, cutoff: float = 8.0, bins: int = 200) -> Dict[str, np.ndarray]:
    """Compute partial RDFs over the last `nsamples` frames.

    Returns a dict with keys like 'Cl-Na', 'Cl-Pu', etc., plus key 'r'.
    """
    # Ensure the pipeline has an RDF modifier appended
    rdf_modifier = CoordinationAnalysisModifier(cutoff=cutoff, number_of_bins=bins, partial=True)
    pipeline.modifiers.append(rdf_modifier)

    nframes = pipeline.source.num_frames
    nsamples = min(nsamples, nframes)
    start = max(0, nframes - nsamples)

    rdf_accumulator: Dict[str, List[np.ndarray]] = {}
    r_axis: Optional[np.ndarray] = None

    for frame in range(start, nframes):
        data = pipeline.compute(frame)
        rdf_table = data.tables['coordination-rdf']
        component_names = rdf_table.y.component_names

        # cache r-axis from xy() once
        if r_axis is None:
            r_axis = rdf_table.xy()[:, 0]

        y_all = rdf_table.y  # shape: (num_bins, num_components)
        for component, name in enumerate(component_names):
            # normalize naming to alphabetical A-B
            pair = '-'.join(sorted(name.split('-')))
            series = np.asarray(y_all[:, component])
            rdf_accumulator.setdefault(pair, []).append(series)

    # Average accumulated RDFs
    rdf_out: Dict[str, np.ndarray] = {k: np.vstack(v).mean(axis=0) for k, v in rdf_accumulator.items()}
    if r_axis is None:
        raise RuntimeError("Failed to obtain r-axis from RDF table")
    rdf_out["r"] = r_axis
    # remove the RDF modifier to not affect downstream usage unless desired
    pipeline.modifiers.clear()
    return rdf_out


def find_first_minimum_with_scipy_fallback(
    r: np.ndarray, 
    g: np.ndarray, 
    min_distance: float = 1.0, 
    max_distance: float = 6.0,
    prominence: float = 0.5,
    smooth_window: int = 5
) -> Optional[float]:
    """Find first minimum using scipy's find_peaks with prominence, fallback to custom implementation.
    
    Primary method: Use scipy.signal.find_peaks on inverted g(r) with prominence filtering
    Fallback: Use the existing find_first_minimum_no_scipy implementation
    
    Args:
        r: Distance array
        g: RDF values
        min_distance: Minimum distance to consider
        max_distance: Maximum distance to consider  
        prominence: Minimum prominence for peaks (scipy method only)
        smooth_window: Window size for smoothing (scipy method only)
        
    Returns:
        Distance of first minimum, or None if not found
    """
    if not SCIPY_AVAILABLE or find_peaks is None:
        return find_first_minimum_no_scipy(r, g, min_distance, max_distance)
    
    # Restrict to window
    mask = (r >= min_distance) & (r <= max_distance)
    r_win = r[mask]
    g_win = g[mask]
    if r_win.size < 5:
        return find_first_minimum_no_scipy(r, g, min_distance, max_distance)
    
    try:
        # Smooth the data slightly to reduce noise
        g_smooth = _moving_average(g_win, smooth_window)
        
        # Find the first peak (maximum) to establish search region
        peak_idx = int(np.argmax(g_smooth))
        
        # Invert g(r) to find minima as peaks
        g_inverted = -g_smooth
        
        # Find peaks (minima in original) with prominence filtering
        peaks, properties = find_peaks(
            g_inverted, 
            prominence=prominence,
            distance=1  # Minimum distance between peaks
        )
        
        if len(peaks) > 0:
            # Find the first minimum after the first peak
            for peak in peaks:
                if peak > peak_idx:  # Must be after the first peak
                    # Refine using quadratic interpolation
                    return _quadratic_refine_minimum(r_win, g_smooth, peak)
        
        # If no suitable peaks found, fall back to custom method
        return find_first_minimum_no_scipy(r, g, min_distance, max_distance)
        
    except Exception:
        # If scipy fails for any reason, fall back to custom method
        return find_first_minimum_no_scipy(r, g, min_distance, max_distance)


def find_first_minimum_no_scipy(r: np.ndarray, g: np.ndarray, min_distance: float = 1.0, max_distance: float = 6.0) -> Optional[float]:
    """Locate a first minimum in g(r) within [min_distance, max_distance] without SciPy.

    Strategy:
    - Restrict r, g to the window.
    - Smooth slightly via a small moving average to reduce noise.
    - Find the first index after the global maximum where g dips below 1.0; if not,
      detect a local minimum by checking sign change of discrete derivative.
    """
    mask = (r >= min_distance) & (r <= max_distance)
    r_win = r[mask]
    g_win = g[mask]
    if r_win.size < 3:
        return None

    # simple smoothing (window size 3)
    if g_win.size >= 3:
        g_smooth = np.convolve(g_win, np.ones(3) / 3.0, mode='same')
    else:
        g_smooth = g_win

    peak_idx = int(np.argmax(g_smooth))
    # heuristic: first point after peak where g < 1.0
    for i in range(peak_idx + 1, g_smooth.size):
        if g_smooth[i] < 1.0:
            return float(r_win[i])

    # fallback: discrete derivative sign change (max -> min)
    dg = np.diff(g_smooth)
    for i in range(max(peak_idx + 1, 1), dg.size - 1):
        if dg[i - 1] < 0.0 and dg[i] > 0.0:
            return float(r_win[i])

    return None


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """Simple centered moving average with reflection at boundaries."""
    if window_size <= 1:
        return values
    pad = window_size // 2
    if pad == 0:
        return values
    extended = np.pad(values, pad_width=pad, mode='reflect')
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    smoothed = np.convolve(extended, kernel, mode='valid')
    return smoothed


def _quadratic_refine_minimum(r: np.ndarray, g: np.ndarray, i: int) -> float:
    """Refine the minimum location using quadratic interpolation around index i.

    Assumes near-uniform spacing in r, which holds for OVITO's RDF bins.
    Falls back to r[i] if refinement fails or i is at an edge.
    """
    if i <= 0 or i >= len(r) - 1:
        return float(r[i])
    g_im1, g_i, g_ip1 = float(g[i - 1]), float(g[i]), float(g[i + 1])
    denom = (g_im1 - 2.0 * g_i + g_ip1)
    if denom == 0.0:
        return float(r[i])
    # Estimate assuming uniform spacing
    dr = float(r[i] - r[i - 1])
    delta = 0.5 * dr * (g_im1 - g_ip1) / denom
    return float(r[i] + delta)


def find_first_shell_minimum(
    r: np.ndarray,
    g: np.ndarray,
    min_distance: float,
    max_distance: float,
    smooth_window: int = 5,
    baseline_level: float = 1.0,
    min_peak_height_above_baseline: float = 0.1,
) -> Optional[float]:
    """Select the first-shell cutoff as the first local minimum after the first peak.

    Steps (science-informed):
    - g_AB(r) exhibits a first-neighbor peak; the bond shell ends at the
      subsequent local minimum where the first shell separates from the second.
    - Smooth g(r) slightly to suppress noise without shifting extrema materially.
    - Identify the first local maximum above the baseline (~1 in liquids/solid tails).
    - Search forward for the first local minimum and refine by quadratic fit.
    """
    # Restrict window
    mask = (r >= min_distance) & (r <= max_distance)
    r_win = r[mask]
    g_win = g[mask]
    if r_win.size < 5:
        return None

    g_s = _moving_average(g_win, smooth_window)

    # Discrete derivative
    dg = np.diff(g_s)

    # Find candidate local maxima: dg[i-1] > 0 and dg[i] <= 0 at index i
    peak_idx: Optional[int] = None
    for i in range(1, len(g_s) - 1):
        if dg[i - 1] > 0.0 and dg[i] <= 0.0 and g_s[i] >= (baseline_level + min_peak_height_above_baseline):
            peak_idx = i
            break
    if peak_idx is None:
        # Fallback: use global maximum within the window
        peak_idx = int(np.argmax(g_s))

    # Now find the first local minimum after the peak
    min_idx: Optional[int] = None
    for i in range(peak_idx + 1, len(g_s) - 1):
        if dg[i - 1] < 0.0 and dg[i] >= 0.0:
            min_idx = i
            break

    if min_idx is not None:
        # Map back to original r indices
        return _quadratic_refine_minimum(r_win, g_s, min_idx)

    # Secondary fallback: first point after peak where g crosses below baseline
    for i in range(peak_idx + 1, len(g_s)):
        if g_s[i] < baseline_level:
            return float(r_win[i])

    # Final fallback: minimal value after peak within window
    if peak_idx + 1 < len(g_s):
        j = peak_idx + 1 + int(np.argmin(g_s[peak_idx + 1:]))
        return float(r_win[j])

    return None


def determine_cutoffs_from_rdf(
    rdf: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    window: Tuple[float, float] = (2.0, 5.0),
    fallback: float = 3.10,
    pair_windows: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
) -> Dict[Tuple[str, str], float]:
    """For each (A,B) pair, determine a symmetric cutoff using first minimum.

    Returns mapping {(A,B): cutoff}. If RDF not found, uses fallback.
    """
    r = rdf.get("r")
    if r is None:
        raise ValueError("RDF dict missing 'r' axis")

    cutoffs: Dict[Tuple[str, str], float] = {}
    for A, B in pairs:
        key = '-'.join(sorted([A, B]))
        g = rdf.get(key)
        if g is not None:
            # Use pair-specific window if provided
            win = window
            if pair_windows is not None and (A, B) in pair_windows:
                win = pair_windows[(A, B)]
            elif pair_windows is not None and (B, A) in pair_windows:
                win = pair_windows[(B, A)]
            a, b = win
            # Try scipy-based method first, with fallback to custom implementation
            val = find_first_minimum_with_scipy_fallback(r, g, min_distance=a, max_distance=b)
            if val is None:
                # Additional fallback to the more sophisticated shell minimum detector
                val = find_first_shell_minimum(r, g, min_distance=a, max_distance=b)
            if val is None:
                # Final fallback to legacy detector
                val = find_first_minimum_no_scipy(r, g, min_distance=a, max_distance=b)
            cutoffs[(A, B)] = float(val) if val is not None else float(fallback)
        else:
            cutoffs[(A, B)] = float(fallback)
    return cutoffs


def configure_bonds_modifier_from_cutoffs(pipeline: Any, pair_cutoffs: Dict[Tuple[str, str], float], disable_pairs: Optional[List[Tuple[str, str]]] = None) -> None:
    """Clear pipeline modifiers and append a CreateBondsModifier with specified pairwise cutoffs.

    - `pair_cutoffs` should include symmetric entries (A,B) and (B,A) if desired.
    - `disable_pairs` can list pairs to set cutoff 0.0.
    """
    pipeline.modifiers.clear()
    cb = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)

    disable_set = set(disable_pairs or [])
    # Apply configured cutoffs
    for (A, B), rcut in pair_cutoffs.items():
        cutoff_val = 0.0 if (A, B) in disable_set else float(rcut)
        cb.set_pairwise_cutoff(A, B, cutoff_val)

    # Common: block direct metal-metal bonds
    for (A, B) in [("Pu", "Pu"), ("Na", "Na"), ("Pu", "Na"), ("Na", "Pu")]:
        cb.set_pairwise_cutoff(A, B, 0.0)

    pipeline.modifiers.append(cb)


def build_shared_anion_graph(data: Any, names: np.ndarray, anion: str = 'Cl', metals: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, nx.Graph]:
    """Construct shared-anion connectivity graph among metal atoms.

    - Connect two metal atoms if they share an anion neighbor via bonds.
    Returns (sizes, cluster_id, G)
    """
    if metals is None:
        metals = ['Pu', 'Na']

    parts = data.particles
    bonds = parts.bonds
    if not bonds or len(bonds) == 0:
        return np.array([], dtype=int), -np.ones(len(names), dtype=int), nx.Graph()

    enum = BondsEnumerator(bonds)

    is_metal = np.isin(names, metals)
    is_anion = (names == anion)

    G = nx.Graph()
    metal_indices = np.where(is_metal)[0]
    G.add_nodes_from(metal_indices.tolist())

    topo = bonds.topology
    for cl_idx in np.where(is_anion)[0]:
        neighbors: List[int] = []
        for bidx in enum.bonds_of_particle(int(cl_idx)):
            a, b = topo[bidx]
            nb = int(a) if int(b) == int(cl_idx) else int(b)
            if nb < len(is_metal) and is_metal[nb]:
                neighbors.append(nb)
        # connect pairwise
        for i in range(len(neighbors)):
            ni = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                nj = neighbors[j]
                G.add_edge(int(ni), int(nj))

    comps = list(nx.connected_components(G))
    sizes = np.array([len(c) for c in comps], dtype=int)

    cluster_id = -np.ones(len(names), dtype=int)
    for cid, comp in enumerate(comps):
        for idx in comp:
            if idx < len(cluster_id):
                cluster_id[int(idx)] = int(cid)

    return sizes, cluster_id, G


def summarize_bonds(data: Any, names: np.ndarray) -> Dict[str, int]:
    """Return a mapping like {"Cl-Pu": count, ...}. Optionally plot if plot_utils is present."""
    bonds = data.particles.bonds
    if not bonds or len(bonds) == 0:
        return {}

    topo = bonds.topology
    counts: Dict[str, int] = {}
    for a, b in topo:
        ta = names[int(a)] if int(a) < len(names) else 'out_of_range'
        tb = names[int(b)] if int(b) < len(names) else 'out_of_range'
        label = '-'.join(sorted([ta, tb]))
        counts[label] = counts.get(label, 0) + 1

    # Optional visualization using plot_utils
    if analyze_bond_network is not None:
        try:
            analyze_bond_network(data, names)
        except Exception:
            pass

    return counts


def extract_names_array(parts: Any) -> np.ndarray:
    """Helper to convert OVITO particle 'Particle Type' IDs to names array."""
    particle_types = parts['Particle Type']
    types = parts.particle_types
    return np.array([types.type_by_id(t).name for t in particle_types])


def canonical_cluster_workflow(pipeline: Any, disable_pair: Optional[Tuple[str, str]] = None, metals: Optional[List[str]] = None, anion: str = 'Cl', rdf_samples: int = 100) -> Dict[str, Any]:
    """End-to-end workflow:
    1) compute RDFs, 2) choose cutoffs for (Pu,Cl) and (Na,Cl), 3) create bonds
    honoring an optional disabled pair, 4) compute shared-anion clusters and return artifacts.
    """
    # Step 1: RDFs
    rdf = compute_partial_rdfs(pipeline, nsamples=rdf_samples)
    # Step 2: determine cutoffs for key pairs
    pairs = [("Pu", "Cl"), ("Na", "Cl"), ("Cl", "Pu"), ("Cl", "Na")]
    # compute for unique unordered, then mirror
    base = determine_cutoffs_from_rdf(rdf, pairs=[("Pu", "Cl"), ("Na", "Cl")])
    # mirror
    pair_cutoffs = {
        ("Pu", "Cl"): base[("Pu", "Cl")],
        ("Cl", "Pu"): base[("Pu", "Cl")],
        ("Na", "Cl"): base[("Na", "Cl")],
        ("Cl", "Na"): base[("Na", "Cl")],
    }

    disable_pairs = []
    if disable_pair is not None:
        A, B = disable_pair
        disable_pairs.extend([(A, B), (B, A)])

    # Step 3: configure bonds
    configure_bonds_modifier_from_cutoffs(pipeline, pair_cutoffs, disable_pairs=disable_pairs)

    # Step 4: compute one frame and build clusters
    data = pipeline.compute(0)
    names = extract_names_array(data.particles)
    sizes, cluster_ids, G = build_shared_anion_graph(data, names, anion=anion, metals=metals)

    result = {
        "rdf": rdf,
        "pair_cutoffs": pair_cutoffs,
        "data": data,
        "names": names,
        "sizes": sizes,
        "cluster_ids": cluster_ids,
        "graph": G,
    }

    return result


