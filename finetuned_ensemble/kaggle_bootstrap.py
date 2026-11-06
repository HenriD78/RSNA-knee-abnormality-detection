"""Bootstrap hors ligne, bibliothèque standard uniquement avant installation."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def locate(kind="training", explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        marker = "bundle_manifest.json" if kind == "training" else "manifest.json"
        candidates = []
        for path in Path("/kaggle/input").rglob(marker):
            try:
                meta = json.loads(path.read_text())
                expected = "rsna3-training-bundle-v1" if kind == "training" else "rsna3-full-models-v1"
                if meta.get("format") == expected:
                    candidates.append(path.parent)
            except (OSError, ValueError):
                continue
    if len(candidates) != 1:
        raise ValueError(f"Choisir explicitement un paquet {kind} parmi {candidates}")
    return candidates[0]


def select_wheels(wheels, supported_tags=None):
    """Select one compatible wheel per distribution before invoking pip."""
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
    except ImportError:
        from pip._vendor.packaging.tags import sys_tags
        from pip._vendor.packaging.utils import parse_wheel_filename
    supported = set(sys_tags() if supported_tags is None else supported_tags)
    projects, selected = set(), {}
    for path in wheels:
        path = Path(path)
        project, version, _, tags = parse_wheel_filename(path.name)
        projects.add(project)
        if not supported.intersection(tags):
            continue
        if project in selected:
            previous, previous_version = selected[project]
            if version != previous_version or checksum(path) != checksum(previous):
                raise ValueError(f"Conflicting compatible wheels for {project}: {previous.name}, {path.name}")
        selected[project] = (path, version)
    missing = projects - selected.keys()
    if missing:
        raise ValueError(
            f"No compatible wheels for {', '.join(sorted(missing))}. "
            f"Runtime: Python {sys.version_info.major}.{sys.version_info.minor}, {sys.platform}. "
            "Attach the matching wheels patch and pass wheels_root=PATCH_ROOT."
        )
    return [selected[name][0] for name in sorted(selected)]


def install(root, target="/kaggle/working/rsna3-env", wheels_root=None):
    root, target = Path(root), Path(target)
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.exists():
        manifest_path = root / "offline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        if checksum(root / name) != expected:
            raise ValueError(f"SHA-256 invalide : {name}")
    wheels = sorted((root / "wheels").glob("*.whl"))
    signatures = [checksum(manifest_path)]
    if wheels_root is not None:
        wheels_root = Path(wheels_root)
        patch_path = wheels_root / "wheels_manifest.json"
        patch = json.loads(patch_path.read_text())
        if patch.get("format") != "rsna3-offline-wheels-patch-v1":
            raise ValueError("Invalid offline wheels patch")
        for name, expected in patch["files"].items():
            if checksum(wheels_root / name) != expected:
                raise ValueError(f"Patch SHA-256 mismatch: {name}")
        signatures.append(checksum(patch_path))
        wheels += sorted((wheels_root / "wheels").glob("*.whl"))
    if not wheels:
        raise ValueError("Aucune wheel : préparer les dépendances Linux avant Save & Run All")
    if any(p.name.lower().startswith(("torch-", "torchvision-", "nvidia_", "triton-", "numpy-")) for p in wheels):
        raise ValueError("Le paquet ne doit pas remplacer Torch CUDA/NumPy du service")
    wheels = select_wheels(wheels)
    signatures += [sys.implementation.cache_tag, *[checksum(wheel) for wheel in wheels]]
    stamp = hashlib.sha256("|".join(signatures).encode()).hexdigest()
    if not (target / ".manifest").exists() or (target / ".manifest").read_text() != stamp:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--upgrade",
                        "--target", str(target), *map(str, wheels)], check=True)
        (target / ".manifest").write_text(stamp)
    os.environ["PYTHONPATH"] = str(target) + os.pathsep + str(root)
    sys.path[:0] = [str(target), str(root)]
    # Perform the import check in a fresh process to avoid stale notebook imports.
    subprocess.run([sys.executable, "-c", "import torch,timm,gdcm,pydicom; "
                    "from pydicom.pixels import get_decoder; "
                    "uids=['1.2.840.10008.1.2.4.70','1.2.840.10008.1.2.4.90','1.2.840.10008.1.2.5']; "
                    "assert all(get_decoder(u).is_available for u in uids); "
                    "assert timm.__version__=='1.0.29'; "
                    "print('Offline OK:',torch.__version__,timm.__version__,pydicom.__version__)"], check=True)
    return root
