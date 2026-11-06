"""Contrôles du split et des images sur quelques fichiers synthétiques."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from knee_baseline.constants import LABELS, SERIES, UID
from knee_baseline.data import build_folds, choose_images, normalize_report


def test_report_groups_and_deterministic_split(label_tables):
    train, weak = label_tables
    master, targets, _ = build_folds(train, weak)
    shuffled, _, _ = build_folds(train.sample(frac=1, random_state=1), weak.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(master, shuffled)
    assert master.fold.nunique() == 5
    assert master.groupby("group_id").fold.nunique().max() == 1
    assert master.set_index(UID).loc["study_00", "fold"] == master.set_index(UID).loc["study_20", "fold"]
    for fold in range(5):
        assert not set(master[master.fold == fold].group_id) & set(master[master.fold != fold].group_id)


def test_gold_overrides_and_unknown_not_negative(label_tables):
    train, weak = label_tables
    weak.loc[weak[UID] == "study_21", LABELS[0]] = np.nan
    _, targets, gold = build_folds(train, weak)
    target = targets.set_index(UID)
    assert target.loc["study_00", LABELS[0]] == train.iloc[0][LABELS[0]]
    assert target.loc["study_00", LABELS[0] + "__conf"] == 1
    assert np.isnan(target.loc["study_21", LABELS[0]])
    assert target.loc["study_21", LABELS[0] + "__conf"] == 0
    assert gold[gold[UID] == "study_21"][LABELS].isna().all().all()


def test_empty_reports_do_not_form_one_group(label_tables):
    train, weak = label_tables
    train.loc[[1, 2], "Report"] = [np.nan, "  "]
    master, _, _ = build_folds(train, weak)
    assert master.set_index(UID).loc["study_01", "group_id"] != master.set_index(UID).loc["study_02", "group_id"]
    assert normalize_report("Ａ\n B ") == "a b"


@pytest.mark.parametrize("kind", ["duplicates", "missing", "partial_gold", "confidence", "scores"])
def test_invalid_labels_fail(label_tables, kind):
    train, weak = label_tables
    if kind == "duplicates":
        weak = pd.concat([weak, weak.iloc[[0]]])
    elif kind == "missing":
        weak = weak.iloc[:-1]
    elif kind == "partial_gold":
        train.loc[0, LABELS[0]] = np.nan
    elif kind == "confidence":
        weak.loc[0, LABELS[0] + "__conf"] = 2
    else:
        weak.loc[0, LABELS[0]] = np.inf
    with pytest.raises(ValueError):
        build_folds(train, weak)


def test_series_selection_and_missing_slots(tmp_path):
    uid = "study"
    rows = []
    for series, count in [("b", 3), ("a", 3), ("c", 1)]:
        directory = tmp_path / "train_series" / uid / series
        directory.mkdir(parents=True)
        for i in range(count):
            Image.new("L", (224, 224), color=20 + i).save(directory / f"{i}.jpeg")
        rows.append({UID: uid, SERIES: series, "Anatomical_Plane": "Sagittal", "Fat_Suppression": 1})
    (tmp_path / "train_series" / uid / "b/broken.jpeg").write_bytes(b"not an image")
    cfg = {"samples_per_slot": 2, "seed": 42}
    first, counts, errors, digest = choose_images(pd.DataFrame(rows), tmp_path, "train", {uid}, cfg)
    second, _, _, digest2 = choose_images(pd.DataFrame(rows[::-1]), tmp_path, "train", {uid}, cfg)
    assert first == second and digest == digest2
    assert {r[SERIES] for r in first} == {"a"}  # départage stable par UID
    assert {r["slot"] for r in first} == {0}  # aucun slot inventé
    assert len(first) == 2 and len(errors) == 1
    assert next(r for r in counts if r[SERIES] == "b")["readable_images"] == 3
