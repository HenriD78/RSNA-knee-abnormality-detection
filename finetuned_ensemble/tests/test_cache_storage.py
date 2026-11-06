"""Lossless cache compression must preserve every byte delivered to the models."""
import io
import json

import numpy as np
import pandas as pd
import pytest

from rsna3.cache import DELTA_KEY, STORAGE, pack, prepare, unpack
from rsna3.common import UID, atomic_json
from rsna3.dicom import RENDER


@pytest.mark.parametrize("kind", ["noise", "gradient", "zeros", "wraparound", "noncontiguous"])
def test_lossless_storage_exact_roundtrip_and_raw_size_bound(kind):
    rng = np.random.default_rng(42)
    shape = (2, 3, 32, 384)
    images = rng.integers(0, 256, shape, dtype=np.uint8)
    if kind == "gradient":
        images = np.broadcast_to(np.arange(384, dtype=np.uint16).astype(np.uint8), shape).copy()
    elif kind == "zeros":
        images.fill(0)
    elif kind == "wraparound":
        images = np.broadcast_to(np.array([255, 0, 254, 1, 128, 127], dtype=np.uint8), (2, 3, 32, 6)).copy()
    elif kind == "noncontiguous":
        images = images[..., ::-1]
        assert not images.flags.c_contiguous
    arrays = {"images": images, "slots": np.array([0, 5], np.int64),
              "positions": np.array([-.75, .75], np.float32)}
    meta = {"warnings": [{"code": "retained"}], "windows": [{"series": "example"}]}
    payload = pack(arrays, meta)
    restored = unpack(payload)
    for key in arrays:
        assert restored[key].dtype == arrays[key].dtype
        np.testing.assert_array_equal(restored[key], arrays[key])
    raw = io.BytesIO()
    np.savez_compressed(raw, **arrays,
                        metadata=np.frombuffer(json.dumps({**meta, "storage": STORAGE}).encode(), dtype=np.uint8))
    assert len(payload) <= len(raw.getvalue())
    with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
        data = json.loads(stored["metadata"].tobytes())
        assert data["storage"] == STORAGE and data["warnings"] == meta["warnings"]
        if kind == "gradient":
            assert DELTA_KEY in stored and "images" not in stored
        elif kind == "zeros":
            assert "images" in stored and DELTA_KEY not in stored


def test_legacy_npz_and_single_column_roundtrip():
    arrays = {"images": np.array([[[[255], [0], [1]]]], dtype=np.uint8),
              "slots": np.array([0]), "positions": np.array([0.], np.float32)}
    legacy = io.BytesIO()
    np.savez_compressed(legacy, **arrays)
    for payload in (legacy.getvalue(), pack(arrays, {})):
        for key, actual in unpack(payload).items():
            np.testing.assert_array_equal(actual, arrays[key])


def test_reject_ambiguous_npz_encoding():
    ambiguous = io.BytesIO()
    image = np.zeros((1, 3, 2, 2), np.uint8)
    np.savez_compressed(ambiguous, images=image, **{DELTA_KEY: image})
    with pytest.raises(ValueError, match="ambigu"):
        unpack(ambiguous.getvalue())


def test_new_storage_requires_new_probe(tmp_path):
    series = tmp_path / "series.csv"
    pd.DataFrame({UID: ["example"]}).to_csv(series, index=False)
    old = tmp_path / "old-probe.json"
    atomic_json(old, {"render": RENDER})
    with pytest.raises(ValueError, match="storage.*refaire le probe"):
        prepare(["example"], series, tmp_path, tmp_path / "cache", probe_path=old)
