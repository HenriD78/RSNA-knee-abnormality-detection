"""Une archive de reprise autonome, sans JPEG/poids à recopier dans les sorties."""

import os
import tarfile
import time
from pathlib import Path

from .config import output_root
from .constants import READERS
from .io import read_json, write_json

_last_snapshot = 0.


def result_dir():
    path = Path(os.environ.get("RSNA_RESULTS_DIR", "/kaggle/working"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot(cfg):
    """Écriture atomique ; conserve l'archive précédente si l'écriture échoue."""
    global _last_snapshot
    root = output_root(cfg)
    if not root.exists():
        return None
    destination = result_dir() / "rsna_resume.tar"
    temporary = destination.with_suffix(".partial")
    status = {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "features": {}, "completed_epochs": {}}
    for reader in READERS:
        for split in ["train", "test"]:
            path = root / "features" / reader / split / "manifest.json"
            if path.exists():
                status["features"][reader + "/" + split] = len(read_json(path)["studies"])
        for path in (root / "checkpoints" / reader).glob("fold*/history.csv"):
            status["completed_epochs"][reader + "/" + path.parent.name] = max(0, len(path.read_text().splitlines()) - 1)
    write_json(root / "session_status.json", status)
    with tarfile.open(temporary, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.startswith(".partial"):
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    temporary.replace(destination)
    write_json(result_dir() / "SESSION_STATUS.json", status)
    # Rapports et soumissions séparés : consultation sans ouvrir le gros cache.
    import zipfile
    with zipfile.ZipFile(result_dir() / "rsna_reports.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for subdirectory in ["reports", "submissions", "predictions", "checkpoints"]:
            for path in sorted((root / subdirectory).rglob("*")):
                if path.is_file() and path.suffix in {".csv", ".json"}:
                    archive.write(path, path.relative_to(root).as_posix())
    _last_snapshot = time.monotonic()
    print(f"Reprise sauvegardée : {destination} ({destination.stat().st_size / 1e6:.0f} Mo)", flush=True)
    return destination


def maybe_snapshot(cfg):
    interval = float(os.environ.get("RSNA_SNAPSHOT_SECONDS", "1200"))
    if time.monotonic() - _last_snapshot >= interval:
        snapshot(cfg)
