"""Intégration miniature : fichiers artificiels, sans extraction ni entraînement."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from knee_baseline import config
from knee_baseline.constants import LABELS, SERIES, UID
from knee_baseline.data import load_prepared, prepare


def test_preparation_publish_reuse_and_image_drift(tmp_path, monkeypatch, label_tables):
    cfg = config.load_config("configs/smoke.json")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    cfg.update(data_dir="data", output_dir="artifacts/smoke")
    root = tmp_path / "data"
    (root / "derived").mkdir(parents=True)
    train, weak = label_tables
    train.to_csv(root / "train.csv", index=False)
    weak.to_csv(root / "derived/report_labels_v2.csv", index=False)
    test = pd.DataFrame({UID: ["test_1", "test_2", "test_3"]})
    test.to_csv(root / "test.csv", index=False)
    sample = test.copy()
    sample[LABELS] = .5
    sample.to_csv(root / "sample_submission.csv", index=False)
    for split, frame in [("train", train), ("test", test)]:
        rows = []
        for uid in frame[UID]:
            series = "series_" + uid
            directory = root / f"{split}_series" / uid / series
            directory.mkdir(parents=True)
            Image.new("L", (224, 224), color=32).save(directory / "slice.jpeg")
            rows.append({UID: uid, SERIES: series, "Anatomical_Plane": "Sagittal",
                         "Fat_Suppression": 1, "Fluid_Sensitive": 1})
        pd.DataFrame(rows).to_csv(root / f"{split}_series.csv", index=False)
    destination = prepare(cfg)
    assert destination.name == "prepared"
    manifest, tables = load_prepared(cfg, check_images=True)
    assert len(tables["train_folds"]) == 40
    assert prepare(cfg) == destination
    modified = root / tables["selected_images"].iloc[0].path
    modified.write_bytes(b"modified")
    with pytest.raises(ValueError, match="modifiée"):
        load_prepared(cfg, check_images=True)


def test_evaluate_filters_cohorts_and_keeps_fixed_weights(tmp_path, monkeypatch, prediction_frames):
    from knee_baseline import evaluate as evaluation
    cfg = config.load_config("configs/smoke.json")
    cfg["bootstrap_samples"] = 20
    ids = prediction_frames["coatnet"][UID].tolist()
    master = pd.DataFrame({UID: ids, "fold": [i % 5 for i in range(12)],
                           "group_id": [f"g{i}" for i in range(12)],
                           "is_gold": [i < 6 for i in range(12)], "active": True})
    gold = pd.DataFrame({UID: ids})
    gold[LABELS] = [[float(i % 2)] * 12 if i < 6 else [np.nan] * 12 for i in range(12)]
    targets = pd.DataFrame({UID: ids})
    targets[LABELS] = [[.8 if i % 2 else .2] * 12 for i in range(12)]
    for c in LABELS:
        targets[c + "__conf"] = 1.
    monkeypatch.setattr(evaluation, "load_prepared", lambda _: ({}, {
        "train_folds": master, "targets": targets, "gold_labels": gold}))
    monkeypatch.setattr(evaluation, "collect_oof", lambda _, reader: prediction_frames[reader])
    monkeypatch.setattr(evaluation, "output_root", lambda _: tmp_path)
    result = evaluation.evaluate(cfg)
    assert result["gold"]["baseline3"]["studies"] == 6
    assert result["weak_proxy"]["candidate05"]["studies"] == 6
    assert result["weight_selection"] == "fixed_before_gold_evaluation"
    assert (tmp_path / "reports/gold_auc_by_fold.csv").exists()
    assert set(result["gold"]) == set(prediction_frames) | {
        "baseline2", "baseline3", "candidate05", "candidate10", "candidate15"}


def test_oof_and_test_exports_from_untrained_synthetic_heads(tmp_path, monkeypatch):
    """Teste la provenance et le format, avec états artificiels, sans optimizer.step."""
    import torch
    from knee_baseline import train as training
    from knee_baseline import evaluate as evaluation
    from knee_baseline.constants import READERS
    from knee_baseline.features import FeatureStore
    from knee_baseline.models import StudyHead, head_spec
    from knee_baseline.io import sha256, write_checkpoint, write_csv, write_json

    cfg = config.load_config("configs/smoke.json")
    cfg["hidden_dim"] = 16
    train_ids = [f"train_{i}" for i in range(10)]
    test_ids = ["example_1", "example_2", "example_3"]
    master = pd.DataFrame({UID: train_ids, "fold": [i % 5 for i in range(10)],
                           "group_id": [f"g{i}" for i in range(10)],
                           "is_gold": [i < 5 for i in range(10)], "active": True})
    prepared = {"signature": "synthetic_preparation"}
    tables = {"train_folds": master, "test_ids": pd.DataFrame({UID: test_ids})}
    stores = {}
    rng = np.random.default_rng(42)
    for split, ids in [("train", train_ids), ("test", test_ids)]:
        store = FeatureStore(tmp_path / "features" / split, split, 4, 2, create=True)
        for uid in ids:
            store.put(uid, rng.normal(size=(6, 2, 4)), np.ones((6, 2), bool))
        store.flush()
        stores[split] = store
    for module in [training, evaluation]:
        monkeypatch.setattr(module, "load_prepared", lambda *a, **kw: (prepared, tables))
        monkeypatch.setattr(module, "output_root", lambda _: tmp_path)
        monkeypatch.setattr(module, "open_store", lambda _, reader, split: stores[split])
    for reader in READERS:
        for fold in range(5):
            torch.manual_seed(42 + fold)
            spec = head_spec(cfg, reader, 4)
            model = StudyHead(**spec)
            identity = training.training_identity(cfg, prepared, stores["train"], reader, fold)
            checkpoint = tmp_path / "checkpoints" / reader / f"fold{fold}/last.pt"
            write_checkpoint(checkpoint, {"identity": identity, "epoch": cfg["epochs"],
                                          "spec": spec, "head": model.state_dict()})
            subset = master[master.fold == fold]
            frame = training.predict_head(model, subset[UID].tolist(), stores["train"], 2)
            frame = frame.merge(subset, on=UID).drop(columns="active")
            path = tmp_path / "predictions/oof" / reader / f"fold{fold}.csv"
            write_csv(path, frame)
            write_json(path.with_suffix(".json"), {"identity": identity, "sha256": sha256(path),
                                                  "checkpoint_sha256": sha256(checkpoint)})
        combined = training.collect_oof(cfg, reader)
        assert set(combined[UID]) == set(train_ids) and len(combined) == 10
    training.predict_test(cfg, list(READERS))
    evaluation.export_submissions(cfg)
    for name in ["baseline2", "baseline3", "candidate05", "candidate10", "candidate15"]:
        frame = pd.read_csv(tmp_path / "submissions" / f"submission_{name}.csv")
        assert frame[UID].tolist() == test_ids
        assert frame.columns.tolist() == [UID] + LABELS
        assert np.isfinite(frame[LABELS]).all().all()
        assert ((frame[LABELS] >= 0) & (frame[LABELS] <= 1)).all().all()
