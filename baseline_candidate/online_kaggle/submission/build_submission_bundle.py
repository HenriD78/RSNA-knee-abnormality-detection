"""Exporte les vingt têtes finales et un ZIP Kaggle d'inférence uniquement.

Ce conditionneur permanent ne génère pas de notebook, ne décode aucune image
et n'entraîne aucun modèle. Les checkpoints originaux restent en lecture seule.
"""

import json
import sys
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ONLINE = HERE.parent
BASELINE = ONLINE.parent
PROJECT = BASELINE.parent
sys.path.insert(0, str(HERE / "runtime"))
from knee_submit.constants import LABELS, READERS
from knee_submit.io import fingerprint, read_json, sha256, write_json
from knee_submit.models import StudyHead


def build():
    staging, dist = HERE / "bundle", HERE / "dist"
    staging.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)
    source = ONLINE / "rsna_resume"
    state = read_json(source / "session_status.json")
    files, heads, specifications = {}, {}, {}
    for reader, spec in READERS.items():
        if state["features"].get(reader + "/train") != 4407:
            raise ValueError(f"Le lecteur {reader} n'a pas une extraction train complète.")
        directory = BASELINE / "assets" / reader
        asset = read_json(directory / "manifest.json")
        encoder = read_json(source / "features" / reader / "encoder.json")
        if asset != encoder["asset"] or asset["model"] != spec["model"]:
            raise ValueError("Les poids génériques diffèrent de ceux de l'expérience entraînée.")
        if sha256(directory / asset["weight_file"]) != asset["weight_sha256"]:
            raise ValueError("Poids générique altéré.")
        for name in ["manifest.json", asset["weight_file"]]:
            files[f"assets/{reader}/{name}"] = directory / name
        files[f"assets/{reader}/encoder.json"] = source / "features" / reader / "encoder.json"
        specifications[reader] = {**spec, "feature_dim": encoder["feature_dim"]}
        heads[reader] = []
        for fold in range(5):
            checkpoint_path = source / "checkpoints" / reader / f"fold{fold}/last.pt"
            receipt = read_json(source / "predictions/oof" / reader / f"fold{fold}.json")
            source_sha = sha256(checkpoint_path)
            if source_sha != receipt["checkpoint_sha256"]:
                raise ValueError("Le checkpoint ne correspond plus à l'évaluation OOF.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (checkpoint["reader"] != reader or checkpoint["fold"] != fold
                    or checkpoint["epoch"] != 20 or checkpoint["identity"] != receipt["identity"]):
                raise ValueError("Lecteur/fold/époque/identité de checkpoint incompatible.")
            model = StudyHead(**checkpoint["spec"])
            model.load_state_dict(checkpoint["head"], strict=True)
            tensors = {name: tensor.detach().cpu().contiguous().clone()
                       for name, tensor in model.state_dict().items()}
            if not all(torch.isfinite(t).all() for t in tensors.values()):
                raise ValueError("Paramètres non finis dans une tête.")
            relative = f"heads/{reader}/fold{fold}.safetensors"
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(path))
            files[relative] = path
            heads[reader].append({"file": relative, "fold": fold, "epoch": 20,
                                  "spec": checkpoint["spec"], "source_checkpoint_sha256": source_sha})
        print(f"{reader} : poids génériques et cinq têtes exportés.", flush=True)
    for path in sorted((HERE / "runtime").rglob("*.py")):
        if "__pycache__" not in path.parts:
            files["code/" + path.relative_to(HERE / "runtime").as_posix()] = path
    # Les petites dépendances sont locales ; aucun install/torch macOS n'est copié.
    wheels = list((HERE / "wheels").glob("*.whl"))
    wheels += list((ONLINE / "wheels").glob("timm-*.whl"))
    codec_dir = PROJECT / "submission/dicom_codecs"
    codec_meta = read_json(codec_dir / "codecs_manifest.json")
    for name, expected in codec_meta["files"].items():
        path = codec_dir / name
        if sha256(path) != expected:
            raise ValueError("Wheel GDCM altérée.")
        wheels.append(path)
    required = ["timm-1.0.29-", "pydicom-3.0.2-", "safetensors-0.8.0-",
                "opencv_python_headless-5.0.0.93-"]
    if not all(any(w.name.startswith(prefix) for w in wheels) for prefix in required):
        raise FileNotFoundError("Une wheel hors ligne requise n'a pas été téléchargée.")
    files.update({"wheels/" + w.name: w for w in wheels})
    manifest = {"format": "rsna-four-reader-inference-v1", "created": "2026-10-19",
                "labels": LABELS, "readers": specifications, "heads": heads,
                "seed": 42, "samples_per_slot": 8, "primary_blend": "candidate05",
                "training_bundle": read_json(source / "cloud_identity.json"),
                "preprocessing": {"name": "minmax_uint8_area_jpeg95", "size": 224,
                                  "status": "empirically audited; original JPEG converter unavailable"},
                "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                          for name, path in sorted(files.items())}}
    manifest["bundle_id"] = fingerprint(manifest)
    write_json(staging / "submission_bundle.json", manifest)
    target = dist / "UPLOAD_SUBMISSION_TO_KAGGLE.zip"
    temporary = target.with_suffix(".partial")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.write(staging / "submission_bundle.json", "submission_bundle.json")
        for name in ["KAGGLE_SUBMISSION.ipynb", "GUIDE_SOUMISSION_FR.md", "VERIFICATION_FR.md"]:
            archive.write(HERE / name, name)
    temporary.replace(target)
    write_json(dist / "UPLOAD_SHA256.json", {"file": target.name, "sha256": sha256(target),
                                            "bytes": target.stat().st_size,
                                            "bundle_id": manifest["bundle_id"]})
    print(f"ZIP prêt : {target} ({target.stat().st_size / 1e6:.1f} Mo)", flush=True)


if __name__ == "__main__":
    build()
