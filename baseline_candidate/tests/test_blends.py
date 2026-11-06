"""Arithmétique des mélanges et bootstrap sur douze études fictives."""

import numpy as np
import pandas as pd
import pytest

from knee_baseline.constants import LABELS, TINY_WEIGHTS, UID
from knee_baseline.evaluate import align_predictions, build_blends, metrics, paired_bootstrap


def test_align_and_shuffled_rows(prediction_frames):
    expected = build_blends(prediction_frames)
    shuffled = {k: v.sample(frac=1, random_state=i) for i, (k, v) in enumerate(prediction_frames.items())}
    actual = build_blends(shuffled)
    for name in expected:
        pd.testing.assert_frame_equal(expected[name].sort_values(UID).reset_index(drop=True),
                                      actual[name].sort_values(UID).reset_index(drop=True))


def test_arithmetic_and_ties(prediction_frames):
    for frame in prediction_frames.values():
        frame.loc[:2, LABELS[0]] = .5
    actual = build_blends(prediction_frames)
    raw = {k: v.set_index(UID)[LABELS] for k, v in prediction_frames.items()}
    r = lambda x: x.rank(method="average", pct=True)
    b2 = .7 * r(raw["coatnet"]) + .3 * r(raw["small"])
    w = pd.Series(TINY_WEIGHTS).reindex(LABELS)
    b3 = (1 - w) * r(b2) + w * r(raw["tiny"])
    expected = r(.95 * r(b3) + .05 * r(raw["dino"]))
    pd.testing.assert_frame_equal(actual["candidate05"].set_index(UID), expected)
    assert r(raw["coatnet"])[LABELS[0]].iloc[0] == r(raw["coatnet"])[LABELS[0]].iloc[1]


def test_wrong_uids_fail(prediction_frames):
    frame = prediction_frames["coatnet"].copy()
    frame.loc[0, UID] = frame.loc[1, UID]
    with pytest.raises(ValueError):
        align_predictions(frame, prediction_frames["small"][UID])
    with pytest.raises(ValueError):
        align_predictions(prediction_frames["small"].iloc[:-1], prediction_frames["small"][UID])


def test_undefined_auc_is_not_hidden(prediction_frames):
    truth = prediction_frames["coatnet"].copy()
    truth[LABELS] = 0.
    rows, summary = metrics(truth, prediction_frames["coatnet"], "gold")
    assert summary["macro_auc"] is None
    assert summary["defined_targets"] == 0


def test_group_bootstrap_reproducible(prediction_frames):
    truth = prediction_frames["coatnet"].copy()
    truth[LABELS] = np.array([[i % 2] * 12 for i in range(len(truth))])
    groups = pd.DataFrame({UID: truth[UID], "group_id": [str(i // 2) for i in range(len(truth))]})
    result = paired_bootstrap(truth, groups, prediction_frames["coatnet"], prediction_frames["coatnet"], 20)
    assert result["valid"] == 20 and result["ci95"] == [0., 0.]
    assert result == paired_bootstrap(truth, groups, prediction_frames["coatnet"], prediction_frames["coatnet"], 20)
