#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import re


PLOT_FUNCS = [
    "plot_coordination_histograms",
    "plot_graph_structure",
    "plot_cluster_size_distribution",
    "plot_cluster_composition_analysis",
    "plot_3d_cluster_visualization",
    "plot_3d_cluster_with_graph",
    "analyze_bond_network",
    "plot_rdfs",
]

MODULE_NAME = "plot_utils"
IMPORT_CELL_SOURCE = ("from " + MODULE_NAME + " import " + ", ".join(PLOT_FUNCS) + "\n")


def is_plot_def_cell(cell_source: str) -> bool:
    # Match any def plot_* at start of a line
    return bool(re.search(r"^def\s+plot_\w+\s*\(", cell_source, re.M)) or (
        "analyze_bond_network" in cell_source and "def analyze_bond_network(" in cell_source
    )


def already_imports(cell_source: str) -> bool:
    return cell_source.strip().startswith(f"from {MODULE_NAME} import ")


def refactor_notebook(nb_path: Path) -> bool:
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    new_cells = []
    removed_any = False
    modified = False
    has_import_cell = False

    for cell in cells:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue

        source = "".join(cell.get("source", []))

        # Rewrite any existing imports from old module name to new module name
        if re.search(r"^\s*from\s+plot\s+import\s+", source, re.M):
            source = re.sub(r"^\s*from\s+plot\s+import\s+", f"from {MODULE_NAME} import ", source, flags=re.M)
            cell["source"] = [source]
            has_import_cell = True
            modified = True
            new_cells.append(cell)
            continue

        if already_imports(source):
            has_import_cell = True
            new_cells.append(cell)
            continue

        if is_plot_def_cell(source):
            removed_any = True
            continue  # drop this cell

        new_cells.append(cell)

    if removed_any and not has_import_cell:
        import_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [IMPORT_CELL_SOURCE],
        }
        # Prepend import cell near the top (after any markdown title if present)
        insert_idx = 0
        for i, c in enumerate(new_cells[:5]):
            if c.get("cell_type") == "markdown":
                insert_idx = i + 1
        new_cells.insert(insert_idx, import_cell)
        modified = True

    if modified:
        nb["cells"] = new_cells
        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return True
    return False


def main(paths):
    changed = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for nb in sorted(path.rglob("*.ipynb")):
                if refactor_notebook(nb):
                    changed.append(str(nb))
        else:
            if path.suffix == ".ipynb" and refactor_notebook(path):
                changed.append(str(path))
    if changed:
        print("Refactored:")
        for c in changed:
            print(" -", c)
    else:
        print("No changes needed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: refactor_notebooks_import_plots.py <notebook_or_dir> [more...]")
        sys.exit(1)
    main(sys.argv[1:])


