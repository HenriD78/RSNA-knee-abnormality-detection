"""Petites données artificielles : aucun accès aux 819 000 JPEG pour les tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knee_baseline.constants import LABELS, UID


@pytest.fixture
def label_tables():
    train, weak = [], []
    for i in range(40):
        uid = f"study_{i:02d}"
        # Une répétition gold/weak permet de tester la fuite par rapport partagé.
        report = "  Même RAPPORT\n " if i in [0, 20] else f"Rapport unique {i}"
        gold = i < 15
        train.append({UID: uid, "Report": report,
                      **{c: float((i + j) % 2) if gold else np.nan for j, c in enumerate(LABELS)}})
        weak.append({UID: uid, **{c: .8 if (i + j) % 2 else .2 for j, c in enumerate(LABELS)},
                     **{c + "__conf": .75 for c in LABELS}})
    return pd.DataFrame(train), pd.DataFrame(weak)


@pytest.fixture
def prediction_frames():
    rng = np.random.default_rng(42)
    frames = {}
    for reader in ["coatnet", "small", "tiny", "dino"]:
        frame = pd.DataFrame(rng.random((12, 12)), columns=LABELS)
        frame.insert(0, UID, [f"study_{i}" for i in range(12)])
        frames[reader] = frame
    return frames
