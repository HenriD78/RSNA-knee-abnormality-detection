"""Installation hors ligne, avant d'importer timm/pydicom et leurs plugins."""

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def find_bundle(input_root="/kaggle/input", explicit=""):
    if explicit:
        found = [Path(explicit)] if (Path(explicit) / "submission_bundle.json").is_file() else []
    else:
        found = []
        # Ne jamais parcourir les centaines de milliers de DICOM de la compétition.
        for directory, subdirs, files in os.walk(input_root):
            path = Path(directory)
            subdirs[:] = [x for x in subdirs if x not in {
                "competitions", "rsna-knee-abnormality-detection", "train_series", "test_series"}]
            if len(path.relative_to(input_root).parts) >= 5:
                subdirs[:] = []
            if "submission_bundle.json" in files:
                found.append(path)
    if len(found) != 1:
        raise FileNotFoundError("Attacher le NOUVEAU Dataset de soumission ; renseigner BUNDLE_DIR si nécessaire.")
    return found[0]


def find_competition(input_root="/kaggle/input", explicit=""):
    root = Path(input_root)
    candidates = ([Path(explicit)] if explicit else [
        root / "competitions/rsna-knee-abnormality-detection",
        root / "rsna-knee-abnormality-detection"])
    found = [p for p in candidates if all((p / name).is_file() for name in [
        "test.csv", "test_series.csv", "sample_submission.csv"]) and (p / "test_series").is_dir()]
    if len(found) != 1:
        raise FileNotFoundError("Ajouter la compétition RSNA Knee dans Add Input, ou préciser COMPETITION_DIR.")
    return found[0]


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "submission_bundle.json").read_text())
    payload = {k: v for k, v in manifest.items() if k != "bundle_id"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if (manifest.get("format") != "rsna-four-reader-inference-v1"
            or hashlib.sha256(canonical.encode()).hexdigest() != manifest.get("bundle_id")):
        raise ValueError("Manifeste de soumission altéré ou inconnu.")
    for name, record in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Fichier de soumission manquant ou altéré : {name}")
    return manifest


def install(bundle, work_dir="/kaggle/temp/rsna_submission"):
    """Garde torch/torchvision du service ; installe les wheels incluses sans réseau."""
    if sys.version_info < (3, 11) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Ce paquet d'installation vise Kaggle Linux x86_64, Python 3.11 à 3.13.")
    bundle, work_dir = Path(bundle), Path(work_dir)
    manifest = verify_bundle(bundle)
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    codecs = [n for n in manifest["files"] if n.startswith("wheels/python_gdcm-") and f"-{tag}-{tag}-" in n]
    if len(codecs) != 1:
        raise RuntimeError(f"Wheel GDCM absente pour {tag} ; utiliser Python 3.11 à 3.13.")
    for name, wanted in [("timm", "1.0.29"), ("pydicom", "3.0.2"), ("cv2", "5.0.0")]:
        if name in sys.modules and getattr(sys.modules[name], "__version__", None) != wanted:
            raise RuntimeError(f"{name} déjà chargé dans une autre version : redémarrer le noyau.")
    vendor = work_dir / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    wheels = [bundle / n for n in manifest["files"] if n.startswith("wheels/")
              and (not n.startswith("wheels/python_gdcm-") or n in codecs)]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                    "--disable-pip-version-check", "--upgrade", "--target", str(vendor),
                    *map(str, wheels)], check=True)
    sys.path[:0] = [str(vendor), str(bundle / "code")]
    importlib.invalidate_caches()
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "HF_HOME": str(work_dir / "cache/huggingface"),
                       "TORCH_HOME": str(work_dir / "cache/torch"),
                       "XDG_CACHE_HOME": str(work_dir / "cache/xdg")})
    import gdcm
    print("Installation hors ligne prête ; GDCM", gdcm.Version.GetVersion(), flush=True)
    return manifest
