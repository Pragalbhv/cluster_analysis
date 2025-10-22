#!/usr/bin/env python3
"""
Simple example showing how to add Pu-only plotting to existing workflows.

This can be used in Jupyter notebooks or as a standalone script.
"""

# Example usage in a notebook or script:

# 1. Import the required functions
from bondmodifier_utils import canonical_cluster_workflow, extract_names_array
from plot_utils import plot_3d_graph_components_pu_only

# 2. Assuming you already have a pipeline and data loaded:
# pipeline = import_file("your_data.xyz")
# data = pipeline.compute(0)
# names = extract_names_array(data.particles)

# 3. Run the canonical workflow to get the graph
# result = canonical_cluster_workflow(pipeline, metals=['Pu', 'Na'], anion='Cl')
# graph = result['graph']

# 4. Plot Pu-only clusters
# plot_3d_graph_components_pu_only(graph, data, names, max_components=6)

print("Pu-only plotting example ready!")
print("Copy the commented code above into your notebook or script.")
print()
print("Key features of the Pu-only plotting:")
print("- Shows only Pu atoms and their connections")
print("- Different colors for different cluster components")
print("- 3D visualization with proper axes labels")
print("- Displays cluster size and connectivity information")
print("- Filters out Na and Cl atoms for cleaner visualization")

