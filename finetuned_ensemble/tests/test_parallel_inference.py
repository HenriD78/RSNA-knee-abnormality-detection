"""Real process spawning and pixel equivalence; no actual CUDA run is claimed."""
import copy
import json
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

import rsna3_parallel_inference as parallel
from rsna3.cache import Cache, prepare
from rsna3.common import LABELS, ROOT, UID, atomic_json, read_json, sha256, source_hash
from test_pipeline import write_dicom


def dicom_fixture(tmp_path):
    ids, rows = [f"study-{i}" for i in range(5)], []
    dicom_root = tmp_path / "dicom"
    for uid in ids:
        for sid, plane, iop, direction, side in (
                ("sag", "Sagittal", [0, 1, 0, 0, 0, -1], [1, 0, 0], "R"),
                ("cor", "Coronal", [1, 0, 0, 0, 0, -1], [0, 1, 0], "L")):
            rows.append({UID: uid, "SeriesInstanceUID": sid, "Anatomical_Plane": plane, "Fat_Suppression": 1})
            for index in range(5):
                path = dicom_root / uid / sid / f"{index}.dcm"
                ds = write_dicom(path, np.asarray(direction) * index, iop)
                ds.Laterality = side
                ds.save_as(path, enforce_file_format=True)
    series = tmp_path / "series.csv"
    pd.DataFrame(rows).to_csv(series, index=False)
    return ids, series, dicom_root


def test_spawned_parallel_cache_matches_serial_pixels_payloads_and_warnings(tmp_path):
    ids, series, dicom_root = dicom_fixture(tmp_path)
    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"
    serial = prepare(ids, series, dicom_root, serial_dir, centres=2, inference=True)
    combined = parallel.prepare_parallel(ids, series, dicom_root, parallel_dir, centres=2,
                                         workers=2, batch_size=2, max_hours=.05)
    assert combined["contract"] == serial["contract"]
    assert set(combined["entries"]) == set(ids)
    assert combined["laterality_conflict_studies"] == serial["laterality_conflict_studies"] == ids
    assert len(combined["shards"]) == 3
    assert not parallel_dir.with_name(parallel_dir.name + ".parts").exists()
    left, right = Cache(serial_dir, verify=True), Cache(parallel_dir, verify=True)
    try:
        for uid in ids:
            a, b = left.read(uid), right.read(uid)
            for key in ("images", "slots", "positions"):
                np.testing.assert_array_equal(a[key], b[key])
                assert a[key].dtype == b[key].dtype
            # The entire study NPZ is retained verbatim, including geometry and warnings.
            assert serial["entries"][uid]["sha256"] == combined["entries"][uid]["sha256"]
    finally:
        left.close()
        right.close()
    resumed = parallel.prepare_parallel(ids, series, dicom_root, parallel_dir, centres=2,
                                        workers=2, batch_size=2, max_hours=.05)
    assert resumed == combined
    with pytest.raises(ValueError, match="different inputs or execution settings"):
        parallel.prepare_parallel(ids, series, dicom_root, parallel_dir, centres=2, workers=3, batch_size=2)


def test_parallel_worker_failure_does_not_publish_complete_cache(tmp_path):
    ids, series, dicom_root = dicom_fixture(tmp_path)
    import shutil
    shutil.rmtree(dicom_root / ids[0])
    target = tmp_path / "parallel"
    with pytest.raises(ValueError, match="inexploitable"):
        parallel.prepare_parallel(ids, series, dicom_root, target, centres=2, workers=2, batch_size=2, max_hours=.05)
    assert not (target / "manifest.json").exists()


def test_parallel_profile_guard_keeps_no_stale_submission_and_validates_execution(tmp_path, monkeypatch):
    from rsna3 import inference
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({UID: ["a", "b"]}).to_csv(test_csv, index=False)
    series = tmp_path / "series.csv"
    series.write_text("fixture")
    execution = parallel.execution_contract(2, 2)
    manifest = {"required_execution_pipeline": parallel.PIPELINE, "centres": 2,
                "admission_profile": {"scope": parallel.SCOPE, "execution": execution,
                                      "startup_seconds": 1., "seconds_per_study": .1}}
    monkeypatch.setattr(inference, "load_bundle", lambda *_: manifest)
    calls = []
    monkeypatch.setattr(parallel, "prepare_parallel", lambda *args, **kwargs: calls.append(args))
    predictions = np.repeat(np.array([[.25], [.75]]), len(LABELS), axis=1)
    monkeypatch.setattr(inference, "predict_bundle", lambda *_: predictions)
    output = tmp_path / "submission.csv"
    output.write_text("old submission")
    with pytest.raises(ValueError, match="differs from benchmark"):
        parallel.submit_parallel(tmp_path, test_csv, series, tmp_path, tmp_path / "work", output,
                                 workers=3, batch_size=2)
    assert not output.exists() and not calls
    receipt = parallel.submit_parallel(tmp_path, test_csv, series, tmp_path, tmp_path / "work", output,
                                       workers=2, batch_size=2)
    frame = pd.read_csv(output)
    assert frame[UID].tolist() == ["a", "b"] and list(frame) == [UID] + LABELS
    np.testing.assert_array_equal(frame[LABELS], predictions)
    assert receipt["execution"] == execution and len(calls) == 1
    manifest["admission_profile"]["seconds_per_study"] = 100000.
    with pytest.raises(ValueError, match="exceeds its measured runtime budget"):
        parallel.submit_parallel(tmp_path, test_csv, series, tmp_path, tmp_path / "work", output,
                                 workers=2, batch_size=2)
    assert not output.exists() and len(calls) == 1


def test_benchmark_refuses_existing_cache_instead_of_reporting_warm_reuse(tmp_path, monkeypatch):
    from rsna3 import inference
    monkeypatch.setattr(inference, "load_bundle", lambda *_: {"centres": 2})
    studies = tmp_path / "studies.csv"
    pd.DataFrame({UID: [str(i) for i in range(50)]}).to_csv(studies, index=False)
    (tmp_path / "work/parallel-cache").mkdir(parents=True)
    with pytest.raises(ValueError, match="requires a fresh cache directory"):
        parallel.benchmark_parallel(tmp_path, studies, tmp_path, tmp_path, tmp_path / "work")


def test_notebook_embeds_exact_helper_and_original_submit_rejects_parallel_profile():
    nb = nbformat.read(ROOT / "notebooks/03c_mesurer_inference_parallele.ipynb", as_version=4)
    cell = next(c.source for c in nb.cells if "parallel-helper" in c.metadata.get("tags", []))
    import ast
    tree = ast.parse(cell)
    write = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text")
    assert ast.literal_eval(write.args[0]) == (ROOT / "rsna3_parallel_inference.py").read_text()
    original = nbformat.read(ROOT / "04_soumettre.ipynb", as_version=4)
    assert "serial submission cannot use its parallel timing profile" in original.cells[1].source
    assert "rglob" not in next(c.source for c in nb.cells if "bootstrap-copy" in c.metadata.get("tags", []))
    assert source_hash() == "1f9c80ee22795f8702039ab37d64f84fd865755cd0c1f517134b2c6a6e1912a5"


def test_submission_discovers_test_only_competition_layout(tmp_path):
    from rsna3.common import require
    inputs = tmp_path / "input"
    competition = inputs / "competitions/rsna-knee-abnormality-detection"
    (competition / "test_series").mkdir(parents=True)
    (competition / "test.csv").write_text("fixture")
    (competition / "test_series.csv").write_text("fixture")

    def path(value):
        return inputs if str(value) == "/kaggle/input" else Path(value)

    nb = nbformat.read(ROOT / "notebooks/04_soumettre_parallele.ipynb", as_version=4)
    env = {"Path": path, "require": require}
    exec(nb.cells[2].source, env)
    assert env["COMPETITION_ROOT"] == competition
    assert env["dicom_directory"]("test") == competition / "test_series"
    assert not (competition / "train.csv").exists()
