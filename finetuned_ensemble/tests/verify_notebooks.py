"""Validation des notebooks livrés, sans exécuter de commande Kaggle."""
import ast
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def verify():
    notebooks = sorted(p for p in ROOT.rglob("*.ipynb") if not any(part in {"dist", "local"} for part in p.relative_to(ROOT).parts))
    if len(notebooks) != 17:
        raise ValueError(f"Dix-sept notebooks attendus, trouvé {len(notebooks)}")
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path.name}:cell{i}")
                if cell.outputs or cell.execution_count is not None:
                    raise ValueError(f"Sortie d'exécution non attendue : {path.name}")
        print(f"OK — {path.relative_to(ROOT)}")
    return len(notebooks)


if __name__ == "__main__":
    verify()
