#!/usr/bin/env python3

import sys
import os

# Add the paths like in the notebook
sys.path.append('/pscratch/sd/p/pvashi/irp/irp_mace_l_2/irp/density/cluster_analysis/')
sys.path.append('/pscratch/sd/p/pvashi/irp/irp_mace_l_2/irp/density/cluster_analysis/Pu_cluster_only/')

print('Python path:')
for p in sys.path:
    print(p)

print('\nTrying to import...')
try:
    from voronoi_ovito_utils_pu_only import build_mixed_voronoi_graph_from_pipeline
    print('Import successful!')
    print('Function available:', callable(build_mixed_voronoi_graph_from_pipeline))
except Exception as e:
    print('Import failed:', str(e))
    import traceback
    traceback.print_exc()
