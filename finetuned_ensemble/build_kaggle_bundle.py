"""Assemble les entrées Kaggle ; ce script ne génère aucun notebook."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rsna3.common import ROOT, SPECS, atomic_json, read_json, require, sha256, spec


def download_wheels(destination, python_version="3.13"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
                    "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_28_x86_64",
                    "--platform", "manylinux_2_27_x86_64", "--platform", "manylinux_2_17_x86_64",
                    "--python-version", python_version, "--implementation", "cp", "--abi", "cp" + python_version.replace(".", ""),
                    "--abi", "abi3", "--dest", str(destination), "-r", str(ROOT / "requirements-kaggle-offline.txt")], check=True)


def build(destination, wheels, project=None, inference=None):
    destination, wheels = Path(destination), Path(wheels)
    project = Path(project or ROOT.parent)
    require(not destination.exists(), "Dossier de destination déjà présent : choisir un nouveau chemin")
    require(list(wheels.glob("*.whl")), "Wheels Linux absentes")
    destination.mkdir(parents=True)
    shutil.copytree(ROOT / "rsna3", destination / "rsna3", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "kaggle_bootstrap.py", destination)
    shutil.copytree(wheels, destination / "wheels")
    if inference:
        require(read_json(Path(inference) / "manifest.json")["format"] == "rsna3-full-models-v1", "Export d'inférence invalide")
        shutil.copytree(inference, destination, dirs_exist_ok=True)
        marker = "offline_manifest.json"
        format_name = "rsna3-inference-offline-v1"
    else:
        for filename in ("entrees_manifest.json", "budget_24h.json", "PROTOCOLE_COMMUN_FR.md", "requirements-kaggle-offline.txt"):
            shutil.copy2(ROOT / filename, destination / filename)
        for model_id, dirname in SPECS.items():
            (destination / dirname).mkdir()
            shutil.copy2(ROOT / dirname / "experience_planifiee.json", destination / dirname)
            cfg = spec(model_id)
            weight = cfg["initialization"]["weight_path"]
            require(sha256(project / weight) == cfg["initialization"]["weight_sha256"], "Poids génériques corrompus")
            target = destination / weight
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / weight, target)
        for entry in read_json(ROOT / "entrees_manifest.json")["files"].values():
            require(sha256(project / entry["path"]) == entry["sha256"], "CSV corrompu")
            target = destination / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / entry["path"], target)
        baseline = "baseline_candidate/online_kaggle/rsna_reports/reports/cohort_predictions"
        for cohort in ("gold", "weak_proxy"):
            path = Path(baseline) / cohort / "candidate05.csv"
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / path, target)
        marker, format_name = "bundle_manifest.json", "rsna3-training-bundle-v1"
    files = {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    atomic_json(destination / marker, {"format": format_name, "files": files,
                                      "bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())})
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download-wheels", action="store_true")
    p.add_argument("--wheels", type=Path, default=ROOT / "wheels-linux-cp313")
    p.add_argument("--python-version", default="3.13")
    p.add_argument("--destination", type=Path)
    p.add_argument("--inference", type=Path)
    args = p.parse_args()
    if args.download_wheels:
        download_wheels(args.wheels, args.python_version)
    if args.destination:
        print(build(args.destination, args.wheels, inference=args.inference))
