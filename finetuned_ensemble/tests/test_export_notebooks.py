"""Guard export completeness and the offline Python ABI used on Kaggle."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from packaging.tags import compatible_tags, cpython_tags

from rsna3.common import ROOT, atomic_json, read_json, sha256, source_hash, versions


MODELS = ("convnext_small", "coatnet2", "dinov2_small")


def tagged(name):
    nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    return {tag: c.source for c in nb.cells if c.cell_type == "code"
            for tag in c.metadata.get("tags", [])}


def training_metadata(tmp_path, dino_folds):
    inputs = tmp_path / "input"
    runs = inputs / "last-training/rsna3-runs"
    for model in MODELS:
        for fold in range(5) if model != "dinov2_small" else dino_folds:
            atomic_json(runs / model / f"fold{fold}/completed.json", {
                "status": "completed", "contract": {"source_hash": source_hash(), "versions": versions()},
                "fixture": "Metadata only; no real trained weights."})
    work = tmp_path / "work"
    work.mkdir()

    def path(value):
        return inputs if str(value) == "/kaggle/input" else Path(value)

    env = {"Path": path, "WORK": work, "PROJECT": ROOT, "json": json}
    cells = tagged("03_evaluer_exporter_cpu.ipynb")
    exec(cells["configuration"], env)
    return cells, env, runs


def test_cpu_export_rejects_successful_session_with_only_twelve_folds(tmp_path):
    cells, env, _ = training_metadata(tmp_path, [0, 1])
    with pytest.raises(ValueError, match="Training is still incomplete:.*dinov2_small/fold2"):
        exec(cells["export-preflight"], env)
    assert not (env["WORK"] / "rsna3-inference").exists()
    assert not (env["WORK"] / "rsna3-reports").exists()


def test_cpu_export_requires_all_folds_and_unchanged_training_environment(tmp_path):
    cells, env, runs = training_metadata(tmp_path, range(5))
    exec(cells["export-preflight"], env)
    assert env["RUNS"] == runs
    commands = []
    env["cli"] = lambda *args: commands.append(args)
    exec(cells["evaluate-export"], env)
    assert [cmd[0] for cmd in commands] == ["evaluate", "export"]
    assert commands[0][-2:] == ("--repetitions", 2000)
    filename = runs / "dinov2_small/fold4/completed.json"
    meta = read_json(filename)
    meta["contract"]["versions"]["torch"] = "different-CPU-build"
    atomic_json(filename, meta)
    with pytest.raises(ValueError, match="Training and export package versions differ"):
        exec(cells["export-preflight"], env)


@pytest.mark.local_assets
def test_offline_export_uses_patched_bootstrap_and_only_python313_wheels(tmp_path):
    module_spec = importlib.util.spec_from_file_location("export_bootstrap", ROOT / "kaggle_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(bootstrap)
    platforms = ["manylinux_2_28_x86_64", "manylinux_2_27_x86_64",
                 "manylinux_2_17_x86_64", "manylinux2014_x86_64"]
    tags = [*cpython_tags((3, 13), abis=["cp313"], platforms=platforms),
            *compatible_tags((3, 13), interpreter="cp313", platforms=platforms)]
    patch = ROOT / "dist/rsna3-python313-patch"
    export = tmp_path / "rsna3-inference"
    export.mkdir()
    atomic_json(export / "manifest.json", {"fixture": "No actual model weights in this packaging test."})
    (export / "__pycache__").mkdir()
    (export / "__pycache__/irrelevant.pyc").write_bytes(b"fixture")
    report = tmp_path / "reports"
    atomic_json(report / "evaluation.json", {"cohorts": {"gold": {"fixture": True}}})
    env = {"BOOTSTRAP_ROOT": patch, "BUNDLE_ROOT": ROOT / "dist/rsna3-training",
           "WHEELS_PATCH_ROOT": patch, "EXPORT": export, "REPORT": report, "json": json,
           "bootstrap": SimpleNamespace(select_wheels=lambda pool: bootstrap.select_wheels(pool, tags))}
    exec(tagged("03_evaluer_exporter_cpu.ipynb")["offline-package"], env)
    wheels = list((export / "wheels").glob("*.whl"))
    assert len(wheels) == 8
    assert not any("-cp312-" in p.name for p in wheels)
    assert sum("-cp313-cp313-" in p.name for p in wheels) == 4
    assert sha256(export / "kaggle_bootstrap.py") == sha256(patch / "kaggle_bootstrap.py")
    manifest = read_json(export / "offline_manifest.json")
    assert all("__pycache__" not in name for name in manifest["files"])
    assert all(sha256(export / name) == checksum for name, checksum in manifest["files"].items())


def test_gpu_benchmark_refreshes_offline_manifest_after_admission_changes(tmp_path):
    export = tmp_path / "rsna3-inference"
    profile = {"scope": "decode_render_all_15_models_global_ranks", "studies": 50}
    atomic_json(export / "manifest.json", {"files": {str(i): "fixture" for i in range(15)}})
    previous_hash = sha256(export / "manifest.json")
    atomic_json(export / "offline_manifest.json", {"files": {"manifest.json": previous_hash}})
    manifest = read_json(export / "manifest.json")
    manifest["admission_profile"] = profile
    atomic_json(export / "manifest.json", manifest)
    assert sha256(export / "manifest.json") != previous_hash
    env = {"EXPORT": export, "atomic_json": atomic_json, "sha256": sha256, "json": json,
           "require": lambda condition, message: condition or pytest.fail(message),
           "load_bundle": lambda folder: read_json(folder / "manifest.json")}
    exec(tagged("03b_mesurer_inference_gpu.ipynb")["finalize"], env)
    offline = read_json(export / "offline_manifest.json")
    assert offline["files"]["manifest.json"] == sha256(export / "manifest.json")
    assert all(sha256(export / name) == checksum for name, checksum in offline["files"].items())
