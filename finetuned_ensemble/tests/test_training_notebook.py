"""Exercise recipe serialization and checkpoint recovery without CUDA training."""
import json
import shutil
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from rsna3.cache import STORAGE
from rsna3.common import ROOT, UID, atomic_json, digest, read_json, sha256, source_hash, spec, versions
from rsna3.dicom import RENDER


METRICS = {
    "convnext_small": (.11061963937499897, .22806981550961652, .13518417631249946,
                       82.703469584, .2774499589999664),
    "coatnet2": (.11931453775000023, .13391070041346062, .34772906112499413,
                 213.8957813510001, .4221221160000823),
    "dinov2_small": (.11842553281252322, .10145846540384494, .11684530662500947,
                      25.261385984999833, .12848489099997096),
}
MODELS = list(METRICS)


def sources(notebook="02_entrainer_ensemble_pilote.ipynb"):
    nb = nbformat.read(ROOT / "notebooks" / notebook, as_version=4)
    return {tag: c.source for c in nb.cells if c.cell_type == "code"
            for tag in c.metadata.get("tags", [])} | {
                "cache": next(c.source for c in nb.cells if c.cell_type == "code" and c.source.startswith("CACHE = None"))}


def fixtures(tmp_path):
    project = ROOT / "dist/rsna3-training"
    inputs = tmp_path / "input"
    cache = inputs / "cpu" / "rsna3-cache"
    uids = pd.read_csv(project / "data/train.csv", dtype=str)[UID].tolist()
    atomic_json(cache / "manifest.json", {
        "complete": True, "entries": {uid: {} for uid in uids},
        "contract": {"render": RENDER, "storage": STORAGE, "source_hash": source_hash(),
                     "purpose": "training", "centres": 2}})
    profile_root = inputs / "pilots/profiles"
    for model, values in METRICS.items():
        profile = {
            "model_id": model, "config": spec(model), "source_hash": source_hash(), "versions": versions(),
            "cache_centres": 2, "cache_manifest_sha256": sha256(cache / "manifest.json"),
            "resume_verified": True, "external_fold_evaluated": False,
            "fixture": "Synthetic profile metadata, reported throughput; never a real CUDA benchmark"}
        profile.update(zip(("head_seconds_per_study", "tail_seconds_per_study", "eval_seconds_per_study",
                            "reload_seconds", "checkpoint_seconds"), values))
        atomic_json(profile_root / f"profile_{model}.json", profile)
    return project, inputs, profile_root


def environment(tmp_path, project, inputs, work_name):
    work = tmp_path / work_name
    work.mkdir()
    def path(value):
        return inputs if str(value) == "/kaggle/input" else Path(value)
    env = {"Path": path, "WORK": work, "PROJECT": project, "json": json}
    exec(sources()["configuration"], env)
    return env


def execute_preflight(env):
    cells = sources()
    for name in ("cache", "profiles", "restore", "freeze"):
        exec(compile(cells[name], f"training-notebook:{name}", "exec"), env)


def simulated_queue(env):
    def cli(*args):
        assert args[0] == "queue"
        assert args[args.index("--gpus") + 1:args.index("--gpus") + 3] == (0, 1)
        assert args[args.index("--quota-available") + 1] == env["QUOTA_AVAILABLE"]
        output = env["OUTPUT"]
        recipes = {model: digest(read_json(env["FROZEN"] / f"{model}.json")) for model in MODELS}
        checkpoint = output / "convnext_small/fold0/active.pt"
        if checkpoint.exists():
            assert checkpoint.read_bytes() == b"fixture: checkpoint retained across sessions"
            assert read_json(output / "queue_state.json")["recipes"] == recipes
        else:
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture: checkpoint retained across sessions")
        atomic_json(output / "queue_state.json", {
            "recipes": recipes, "quota_estimated": {model: .25 for model in MODELS},
            "jobs": {"convnext_small/fold0": {"status": "paused", "returncode": 75}},
            "sessions": [{"elapsed_seconds": 900, "quota_estimated_hours": .25}],
            "fixture": True})
    env["cli"] = cli
    for name in ("training", "summary"):
        exec(compile(sources()[name], f"training-notebook:{name}", "exec"), env)


@pytest.mark.local_assets
def test_frozen_recipes_are_serializable_keep_pilot_snapshot_and_resume_exactly(tmp_path):
    project, inputs, profile_root = fixtures(tmp_path)
    original_profiles = {model: sha256(profile_root / f"profile_{model}.json") for model in MODELS}
    env = environment(tmp_path, project, inputs, "first-work")
    execute_preflight(env)
    frozen_hashes = {}
    for model, expected in zip(MODELS, (7, 9, 10)):
        filename = env["FROZEN"] / f"{model}.json"
        cfg = read_json(filename)
        frozen_hashes[model] = sha256(filename)
        assert cfg["training"]["epochs_target"] == expected
        assert cfg["profile"]["config"]["training"]["epochs_target"] == spec(model)["training"]["epochs_target"]
        assert cfg["projected_quota_hours"] <= cfg["budget_quota_hours"]
        assert cfg["source_hash"] == source_hash()
        assert sha256(profile_root / f"profile_{model}.json") == original_profiles[model]
    assert env["projection_sum"] == pytest.approx(14.013142254748484)
    simulated_queue(env)
    assert read_json(env["WORK"] / "training_session_summary.json")["checkpoint_resume_needed"]
    previous = inputs / "previous-training"
    shutil.copytree(env["WORK"], previous)
    resumed = environment(tmp_path, project, inputs, "second-work")
    resumed["QUOTA_AVAILABLE"] = 17.
    execute_preflight(resumed)
    assert resumed["RESUME_FROM"] == previous / "rsna3-runs"
    for model in MODELS:
        assert sha256(resumed["FROZEN"] / f"{model}.json") == frozen_hashes[model]
    simulated_queue(resumed)
    summary = read_json(resumed["WORK"] / "training_session_summary.json")
    assert summary["completed_folds"] == 0 and not summary["all_folds_complete"]
    assert summary["budget_stop_models"] == []


@pytest.mark.local_assets
def test_training_preflight_rejects_mismatched_profile(tmp_path):
    project, inputs, profile_root = fixtures(tmp_path)
    filename = profile_root / "profile_convnext_small.json"
    p = read_json(filename)
    p["source_hash"] = "wrong-source"
    atomic_json(filename, p)
    env = environment(tmp_path, project, inputs, "work")
    with pytest.raises(ValueError, match="Pilot/code/environment differ"):
        execute_preflight(env)


@pytest.mark.local_assets
def test_training_admission_protects_eight_hour_reserve(tmp_path):
    project, inputs, _ = fixtures(tmp_path)
    env = environment(tmp_path, project, inputs, "work")
    env["QUOTA_AVAILABLE"] = 20.
    with pytest.raises(ValueError, match="protected reserve"):
        execute_preflight(env)


@pytest.mark.local_assets
def test_common_pilot_notebook_freeze_path_serializes_without_mutating_profiles(tmp_path):
    project, inputs, profile_root = fixtures(tmp_path)
    env = environment(tmp_path, project, inputs, "work")
    shutil.copytree(profile_root, env["WORK"] / "profiles")
    notebook = nbformat.read(ROOT / "01_piloter_et_figer.ipynb", as_version=4)
    cell = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("MODE = 'profile'"))
    cell = cell.replace("MODE = 'profile'", "MODE = 'freeze'", 1)
    cell = cell.replace("QUOTA_BEFORE = None", "QUOTA_BEFORE = 24 + 6 / 60", 1)
    cell = cell.replace("QUOTA_AFTER = None", "QUOTA_AFTER = 24 + 3 / 60", 1)
    cell = cell.replace("QUOTA_ELAPSED_HOURS = None", "QUOTA_ELAPSED_HOURS = 3 / 60", 1)
    exec(compile(cell, "common-pilot-freeze-cell", "exec"), env)
    for model, epochs in zip(MODELS, (7, 9, 10)):
        cfg = read_json(env["FROZEN"] / f"{model}.json")
        assert cfg["training"]["epochs_target"] == epochs
        assert cfg["profile"]["config"]["status"] == "specification_only_not_a_training_cli_config"


def resume_preflight(env):
    cells = sources("02_reprendre_ensemble.ipynb")
    for name in ("configuration", "cache", "profiles", "restore", "freeze"):
        exec(compile(cells[name], f"resume-notebook:{name}", "exec"), env)
    return cells


@pytest.mark.local_assets
def test_resume_notebook_refuses_to_start_without_saved_training(tmp_path):
    project, inputs, _ = fixtures(tmp_path)
    env = environment(tmp_path, project, inputs, "resume-work")
    with pytest.raises(ValueError, match="Resume requires exactly one saved training output"):
        resume_preflight(env)
    assert not (env["WORK"] / "rsna3-runs").exists()
    assert not list(env["WORK"].rglob("convnext_small.json"))


@pytest.mark.local_assets
def test_resume_notebook_keeps_twelve_results_and_runs_only_three_dino_folds(tmp_path, monkeypatch):
    from rsna3 import queue

    project, inputs, _ = fixtures(tmp_path)
    first = environment(tmp_path, project, inputs, "first-work")
    execute_preflight(first)
    completed = {"convnext_small": list(range(5)), "coatnet2": list(range(5)), "dinov2_small": [0, 1]}

    def finish(output, cfg, fold):
        output.mkdir(parents=True, exist_ok=True)
        weights = output / f"final_fold{fold}.safetensors"
        weights.write_bytes(b"fixture: completed weights, not an actual trained model")
        oof = output / "oof.csv"
        oof.write_text("fixture: completed OOF, not actual predictions\n")
        atomic_json(output / "completed.json", {"contract": {"config_hash": digest(cfg)},
                    "checkpoint_sha256": sha256(weights), "oof_sha256": sha256(oof)})

    for model, folds in completed.items():
        for fold in folds:
            finish(first["OUTPUT"] / model / f"fold{fold}", first["RECIPES"][model], fold)
    for fold in (2, 3):
        active = first["OUTPUT"] / "dinov2_small" / f"fold{fold}" / "active.pt"
        active.parent.mkdir(parents=True)
        active.write_bytes(f"fixture: saved DINOv2 fold {fold} optimizer and RNG state".encode())
    initial_quota = {model: .25 for model in MODELS}
    previous_session = {"elapsed_seconds": 900, "quota_estimated_hours": .75, "fixture": True}
    atomic_json(first["OUTPUT"] / "queue_state.json", {
        "recipes": {model: digest(cfg) for model, cfg in first["RECIPES"].items()},
        "quota_estimated": initial_quota, "sessions": [previous_session],
        "jobs": {f"dinov2_small/fold{fold}": {"status": "paused", "returncode": 75} for fold in (2, 3)}})
    saved = inputs / "previous-training"
    shutil.copytree(first["WORK"], saved)
    expected_hashes = {p.relative_to(first["WORK"]): sha256(p) for p in first["WORK"].rglob("*")
                       if p.is_file() and (p.parent.name == "frozen" or p.name == "completed.json")}

    # An incomplete upload must not silently restart a paused fold from generic weights.
    missing = saved / "rsna3-runs/dinov2_small/fold3/active.pt"
    checkpoint = missing.read_bytes()
    missing.unlink()
    broken = environment(tmp_path, project, inputs, "missing-checkpoint-work")
    with pytest.raises(ValueError, match="Missing saved DINOv2 checkpoint"):
        resume_preflight(broken)
    assert not (broken["WORK"] / "rsna3-runs").exists()
    missing.write_bytes(checkpoint)

    resumed = environment(tmp_path, project, inputs, "resume-work")
    cells = resume_preflight(resumed)
    assert resumed["QUOTA_AVAILABLE"] == pytest.approx(16.05)
    assert read_json(resumed["OUTPUT"] / "queue_state.json")["quota_estimated"] == initial_quota
    launched = []

    class FakeProcess:
        def __init__(self, command, env, **_):
            cfg = read_json(command[command.index("--experiment") + 1])
            fold = int(command[command.index("--fold") + 1])
            assert cfg["model_id"] == "dinov2_small" and fold in (2, 3, 4)
            output = resumed["OUTPUT"] / cfg["model_id"] / f"fold{fold}"
            if fold in (2, 3):
                assert (output / "active.pt").read_bytes() == (
                    f"fixture: saved DINOv2 fold {fold} optimizer and RNG state".encode())
            launched.append((cfg["model_id"], fold))
            self.pid = 100 + fold
            finish(output, cfg, fold)

        def poll(self):
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(queue.time, "sleep", lambda *_: None)

    def cli(*args):
        assert args[0] == "queue"
        return queue.run_queue([resumed["FROZEN"] / f"{model}.json" for model in MODELS],
                               project, resumed["CACHE"], resumed["OUTPUT"], [0, 1],
                               quota_available=resumed["QUOTA_AVAILABLE"], session_hours=resumed["SESSION_HOURS"])

    resumed["cli"] = cli
    for name in ("training", "summary"):
        exec(compile(cells[name], f"resume-notebook:{name}", "exec"), resumed)
    assert launched == [("dinov2_small", fold) for fold in (2, 3, 4)]
    summary = read_json(resumed["WORK"] / "training_session_summary.json")
    assert summary["all_folds_complete"] and summary["completed_folds"] == 15
    assert not summary["checkpoint_resume_needed"]
    state = read_json(resumed["OUTPUT"] / "queue_state.json")
    assert state["sessions"][0] == previous_session and len(state["sessions"]) == 2
    assert state["quota_estimated"]["convnext_small"] == initial_quota["convnext_small"]
    assert state["quota_estimated"]["coatnet2"] == initial_quota["coatnet2"]
    assert state["quota_estimated"]["dinov2_small"] >= initial_quota["dinov2_small"]
    assert all(sha256(resumed["WORK"] / path) == expected for path, expected in expected_hashes.items())
