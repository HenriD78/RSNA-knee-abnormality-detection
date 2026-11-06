"""Wheel selection against Linux CPython ABI tags, without importing binaries."""
import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from packaging.tags import compatible_tags, cpython_tags

from rsna3.common import ROOT


def bootstrap():
    spec = importlib.util.spec_from_file_location("rsna3_abi_bootstrap", ROOT / "kaggle_bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tags(minor):
    platforms = ["manylinux_2_28_x86_64", "manylinux_2_27_x86_64",
                 "manylinux_2_17_x86_64", "manylinux2014_x86_64"]
    return [*cpython_tags((3, minor), abis=[f"cp3{minor}"], platforms=platforms),
            *compatible_tags((3, minor), interpreter=f"cp3{minor}", platforms=platforms)]


@pytest.mark.parametrize("minor", [12, 13])
@pytest.mark.local_assets
def test_mixed_wheel_pool_selects_only_matching_binaries(minor):
    pool = list((ROOT / "wheels-linux-cp312").glob("*.whl"))
    pool += list((ROOT / "wheels-linux-cp313").glob("*.whl"))
    assert len(pool) == 16
    selected = bootstrap().select_wheels(pool, tags(minor))
    assert len(selected) == 8
    codecs = [p for p in selected if any(name in p.name for name in ("_libjpeg-", "_openjpeg-", "_rle-", "python_gdcm-"))]
    assert len(codecs) == 4
    assert all(f"-cp3{minor}-cp3{minor}-" in p.name for p in codecs)
    assert any("cp310-abi3" in p.name for p in selected)  # compatible safetensors


@pytest.mark.local_assets
def test_python313_rejects_old_bundle_before_pip():
    pool = list((ROOT / "wheels-linux-cp312").glob("*.whl"))
    with pytest.raises(ValueError, match="No compatible wheels.*pylibjpeg-libjpeg"):
        bootstrap().select_wheels(pool, tags(13))


def test_duplicate_package_conflict_is_rejected(tmp_path):
    first = tmp_path / "a/example-1.0-py3-none-any.whl"
    second = tmp_path / "b/example-1.0-py3-none-any.whl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"different")
    with pytest.raises(ValueError, match="Conflicting compatible wheels"):
        bootstrap().select_wheels([first, second], tags(13))


def test_patch_install_replaces_incompatible_wheels_and_is_idempotent(tmp_path, monkeypatch):
    module = bootstrap()
    root, patch, target = (tmp_path / name for name in ("original", "patch", "env"))
    for folder in (root, patch):
        (folder / "wheels").mkdir(parents=True)
    old = root / "wheels/example-1.0-cp312-cp312-manylinux_2_17_x86_64.whl"
    new = patch / "wheels/example-1.0-cp313-cp313-manylinux_2_17_x86_64.whl"
    old.write_bytes(b"old ABI")
    new.write_bytes(b"new ABI")
    for folder, filename, extra in ((root, "bundle_manifest.json", {}),
                                    (patch, "wheels_manifest.json", {"format": "rsna3-offline-wheels-patch-v1"})):
        entries = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (folder / "wheels").glob("*.whl")}
        (folder / filename).write_text(json.dumps({**extra, "files": entries}))
    original_select = module.select_wheels
    monkeypatch.setattr(module, "select_wheels", lambda pool: original_select(pool, tags(13)))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    calls = []
    def run(argv, **_):
        calls.append(argv)
        if "pip" in argv:
            assert str(new) in argv and str(old) not in argv
            target.mkdir(exist_ok=True)
    monkeypatch.setattr(module.subprocess, "run", run)
    module.install(root, target, wheels_root=patch)
    assert (target / ".manifest").is_file()
    module.install(root, target, wheels_root=patch)
    assert sum("pip" in call for call in calls) == 1
    new.write_bytes(b"corrupted patch")
    with pytest.raises(ValueError, match="Patch SHA-256 mismatch"):
        module.install(root, target, wheels_root=patch)
