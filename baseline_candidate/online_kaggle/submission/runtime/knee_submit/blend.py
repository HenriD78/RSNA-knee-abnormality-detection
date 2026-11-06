"""Mêmes rangs et coefficients que l'expérience OOF, sur tout le test courant."""

import numpy as np
import pandas as pd

from .constants import LABELS, READERS, TINY_WEIGHTS, UID


def align_predictions(frame, ids):
    if frame[UID].duplicated().any() or set(frame[UID]) != set(ids) or len(ids) != len(set(ids)):
        raise ValueError("Les prédictions ne couvrent pas exactement les UID du test courant.")
    aligned = frame.set_index(UID).loc[list(ids), LABELS].astype(float)
    values = aligned.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Les scores doivent être finis et compris dans [0, 1].")
    return aligned


def ranks(frame):
    # Calcul GLOBAL par finding, jamais à l'intérieur d'un lot GPU ou d'un fold.
    return frame[LABELS].rank(method="average", pct=True)


def build_blends(predictions):
    if set(predictions) != set(READERS):
        raise ValueError("Les quatre lecteurs sont nécessaires au mélange prévu.")
    ids = predictions["coatnet"][UID].tolist()
    raw = {name: align_predictions(frame, ids) for name, frame in predictions.items()}
    baseline2 = .7 * ranks(raw["coatnet"]) + .3 * ranks(raw["small"])
    weight = pd.Series(TINY_WEIGHTS).reindex(LABELS)
    baseline3 = (1 - weight) * ranks(baseline2) + weight * ranks(raw["tiny"])
    result = {"baseline2": ranks(baseline2), "baseline3": ranks(baseline3)}
    for alpha in [.05, .10, .15]:
        result[f"candidate{int(round(100 * alpha)):02d}"] = ranks(
            (1 - alpha) * ranks(baseline3) + alpha * ranks(raw["dino"]))
    return {name: frame.reset_index()[[UID] + LABELS] for name, frame in result.items()}


def validate_submission(frame, ids):
    if list(frame.columns) != [UID] + LABELS or frame[UID].tolist() != list(ids):
        raise ValueError("Colonnes ou ordre des études incorrects dans submission.csv.")
    align_predictions(frame, ids)
