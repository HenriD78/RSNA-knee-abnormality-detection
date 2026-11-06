"""Installe le paquet hors ligne dans un espace temporaire Kaggle/Colab.

Les seuls résultats à conserver sont rsna_resume.tar et rsna_reports.zip.
Les JPEG et poids ne sont ni téléchargés à nouveau ni publiés par ce module.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_tar(archive_path, destination):
    """Refuse liens et chemins sortants avant de décompresser une archive."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Entrée d'archive interdite : {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def safe_zip(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Chemin ZIP sortant.")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def find_bundle(input_root="/kaggle/input", explicit=""):
    if explicit:
        candidates = [Path(explicit) / "bundle_manifest.json"]
    else:
        candidates = list(Path(input_root).rglob("bundle_manifest.json"))
    if len(candidates) != 1 or not candidates[0].exists():
        raise FileNotFoundError("Attacher le Dataset du paquet ; définir BUNDLE_DIR si plusieurs paquets existent.")
    return candidates[0].parent


def find_resume(input_root="/kaggle/input", explicit=""):
    if explicit:
        result = Path(explicit)
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    candidates = list(Path(input_root).rglob("rsna_resume.tar"))
    if len(candidates) > 1:
        raise ValueError("Plusieurs reprises trouvées : définir RESUME_FILE explicitement.")
    return candidates[0] if candidates else None


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    payload = {k: v for k, v in manifest.items() if k != "bundle_id"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()
    if manifest.get("format") != "rsna-kaggle-upload-v1" or manifest.get("bundle_id") != expected:
        raise ValueError("Manifeste de paquet altéré ou inconnu.")
    for name, sha in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or digest(path) != sha:
            raise ValueError(f"Fichier de paquet altéré : {name}")
    return manifest


def setup(bundle, session_dir, results_dir, resume=None, install=True, require_gpu=True):
    """Prépare les données/poids, sans entraînement ni extraction de features."""
    if require_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Activer le GPU avant l'installation du paquet.")
    bundle, session_dir, results_dir = Path(bundle), Path(session_dir), Path(results_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(session_dir).free < 8 * 1024**3:
        raise RuntimeError("Prévoir au moins 8 Go libres pour données, poids, caches et reprise.")
    manifest = verify_bundle(bundle)
    runtime = session_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    vendor = runtime / "vendor"
    if install:
        # Conserver le torch/torchvision CUDA du service ; installer uniquement 2 wheels pures.
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        "--target", str(vendor), "--upgrade", "--find-links", str(bundle / "wheels"),
                        "timm==1.0.29", "iterative-stratification==0.1.9"], check=True)
    sys.path[:0] = [str(vendor), str(runtime)]
    directories = {"data_payload.bin": session_dir / "inputs", "assets_payload.bin": session_dir / "assets",
                   "seed_payload.bin": session_dir / "seed"}
    for name, directory in directories.items():
        marker = directory / ".payload_sha256"
        expected = manifest["files"][name]
        if marker.exists() and marker.read_text() == expected:
            continue
        print(f"Décompression locale : {name}", flush=True)
        safe_tar(bundle / name, directory)
        marker.write_text(expected)
    output = session_dir / "artifacts/full"
    # Un rerun dans la même session réutilise ses résultats plus récents, sans les écraser.
    if resume is not None and not (output / "cloud_identity.json").exists():
        print(f"Restauration : {resume}", flush=True)
        safe_tar(resume, output)
    identity = output / "cloud_identity.json"
    output.mkdir(parents=True, exist_ok=True)
    if identity.exists() and json.loads(identity.read_text())["bundle_id"] != manifest["bundle_id"]:
        raise ValueError("Reprise issue d'un autre paquet : refus de mélanger les expériences.")
    identity.write_text(json.dumps({"bundle_id": manifest["bundle_id"], "prepared": manifest["prepared"]}))
    os.environ.update({"RSNA_DATA_ROOT": str(session_dir / "inputs"),
                       "RSNA_ASSET_ROOT": str(session_dir / "assets"),
                       "RSNA_SEED_ROOT": str(session_dir / "seed"),
                       "RSNA_OUTPUT_ROOT": str(output), "RSNA_RESULTS_DIR": str(results_dir),
                       "RSNA_CACHE_ROOT": str(session_dir / "cache")})
    results_dir.mkdir(parents=True, exist_ok=True)
    print("Paquet prêt. Features CPU conservées :", manifest["seed_studies"])
    return {"runtime": str(runtime), "output": str(output), "results": str(results_dir),
            "bundle_id": manifest["bundle_id"]}
