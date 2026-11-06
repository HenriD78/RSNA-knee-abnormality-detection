from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

UID = "StudyInstanceUID"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SPECS = {"coatnet2": "01_coatnet2", "convnext_small": "02_convnext_small",
         "dinov2_small": "03_dinov2_small"}
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_csv(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    table.to_csv(temp, index=False)
    os.replace(temp, path)


def source_hash():
    return digest({p.name: sha256(p) for p in sorted((ROOT / "rsna3").glob("*.py"))})


def versions():
    return {name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "timm", "numpy", "pandas", "pydicom", "safetensors", "scipy")}


def spec(model):
    return read_json(ROOT / SPECS[model] / "experience_planifiee.json")


def input_tables(project, verify=True):
    project = Path(project)
    manifest = read_json(ROOT / "entrees_manifest.json")
    if verify:
        for name, entry in manifest["files"].items():
            require(sha256(project / entry["path"]) == entry["sha256"], f"SHA-256 incompatible : {name}")
    folds = pd.read_csv(project / manifest["files"]["master_folds"]["path"], dtype={UID: str, "group_id": str})
    targets = pd.read_csv(project / manifest["files"]["targets"]["path"], dtype={UID: str, "group_id": str})
    require(len(folds) == manifest["studies"] and folds[UID].is_unique, "Couverture des folds invalide")
    require(folds.groupby("group_id").fold.nunique().max() == 1, "Groupe traversant les folds")
    require(set(folds.fold) == set(range(5)), "Cinq folds requis")
    require(targets[UID].is_unique and set(targets[UID]) == set(folds[UID]), "UID des cibles invalides")
    targets = targets.set_index(UID).loc[folds[UID]].reset_index()
    for key in ("fold", "group_id", "is_gold"):
        require(np.array_equal(folds[key], targets[key]), f"Métadonnées désalignées : {key}")
    y = targets[LABELS].to_numpy(float)
    w = targets[[v + "__conf" for v in LABELS]].to_numpy(float)
    require(np.isfinite(w).all() and ((w >= 0) & (w <= 1)).all(), "Confiances invalides")
    require((w[~np.isfinite(y)] == 0).all(), "Cibles inconnues pondérées")
    require(((y[np.isfinite(y)] >= 0) & (y[np.isfinite(y)] <= 1)).all(), "Cibles invalides")
    gold = targets.is_gold.to_numpy(bool)
    require((w[gold][np.isfinite(y[gold])] == 1).all(), "Confiance gold différente de 1")
    return targets, manifest


def validate_predictions(table, uids):
    require(UID in table and set(LABELS).issubset(table), "Colonnes de prédictions absentes")
    require(table[UID].is_unique and set(table[UID]) == set(uids) and len(table) == len(uids),
            "Couverture des prédictions incomplète ou dupliquée")
    aligned = table.set_index(UID).loc[list(uids), LABELS]
    values = aligned.to_numpy(float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Probabilités invalides")
    return aligned.reset_index()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32)
