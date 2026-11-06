"""Valide le notebook sans l'exécuter ; seule sa cellule de configuration est testée."""

import ast
from pathlib import Path
from unittest.mock import patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = nbformat.read(ROOT / "baseline_candidate.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    for cell in code_cells:
        ast.parse(cell.source)
        assert cell.execution_count is None and not cell.outputs, "Le livrable doit rester non exécuté."
    for path in (ROOT / "knee_baseline").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    # La configuration seule ne traite aucune image. Les sous-processus sont interdits ici.
    namespace = {"__name__": "notebook_configuration_check"}
    with patch("subprocess.run", side_effect=AssertionError("Sous-processus interdit pendant ce contrôle")):
        exec(compile(code_cells[0].source, "configuration_cell", "exec"), namespace)
        assert namespace["MODE"] == "inspection"
        assert namespace["STEPS_TO_RUN"] == set()
        assert namespace["ALLOW_DOWNLOAD"] is False
        for step in ["prepare", "assets", "benchmark", "extract_train", "train", "evaluate", "predict"]:
            namespace["run_step"](step, step)
    print(f"Notebook valide : {len(code_cells)} cellules compilables ; configuration inspection sûre.")
    print("Aucune autre cellule exécutée ; aucun téléchargement, extraction ou entraînement.")


if __name__ == "__main__":
    main()
