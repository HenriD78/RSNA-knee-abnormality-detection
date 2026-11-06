"""Contrôles statiques et archives ; n'exécute aucun notebook ou modèle."""

import ast
import importlib
import json
import sys
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))


def main():
    for path in (ROOT / "runtime").rglob("*.py"):
        ast.parse(path.read_text())
    for name in ["bootstrap", "config", "constants", "data", "features", "models", "train", "evaluate", "session", "cloud"]:
        importlib.import_module(name if name == "bootstrap" else "knee_baseline." + name)
    for name in ["KAGGLE_GPU.ipynb", "COLAB_GPU.ipynb"]:
        nb = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.execution_count is None and not cell.outputs
    if (ROOT / "dist/UPLOAD_TO_KAGGLE.zip").exists():
        with zipfile.ZipFile(ROOT / "dist/UPLOAD_TO_KAGGLE.zip") as archive:
            assert archive.testzip() is None
            names = set(archive.namelist())
            assert {"bundle_manifest.json", "data_payload.bin", "assets_payload.bin", "seed_payload.bin",
                    "code_payload.bin", "KAGGLE_GPU.ipynb", "COLAB_GPU.ipynb"} <= names
            assert not any(".venv" in n or ".git/" in n or "kaggle.json" in n for n in names)
    print("Syntaxe, notebooks non exécutés et archive ZIP : contrôles réussis.")


if __name__ == "__main__":
    main()
