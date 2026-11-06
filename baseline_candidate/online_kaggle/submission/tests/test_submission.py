"""Tests de la soumission cachée sur de petites données artificielles."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from knee_submit.blend import build_blends, validate_submission
from knee_submit.constants import LABELS, READERS, SERIES, UID
from knee_submit.data import (dicom_frames, pixel_array_to_jpeg_pixels, prepare_test,
                              read_inputs, select_study, series_candidates, slot_of)
from knee_submit.io import fingerprint, write_json
from knee_submit.predict import feature_tensor


def inputs(root, ids):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({UID: ids}).to_csv(root / "test.csv", index=False)
    # Le template garde volontairement un ancien UID : le test courant doit primer.
    pd.DataFrame([{UID: "old_example", **dict.fromkeys(LABELS, .5)}]).to_csv(
        root / "sample_submission.csv", index=False)
    series = []
    for i, uid in enumerate(ids):
        series_uid = f"series{i}"
        directory = root / "test_series" / uid / series_uid
        directory.mkdir(parents=True)
        for j in range(2):
            Image.new("L", (224, 224), color=30 + i * 20 + j).save(directory / f"slice{j}.jpeg")
        series.append({UID: uid, SERIES: series_uid, "Anatomical_Plane": "Sagittal", "Fat_Suppression": 1})
    pd.DataFrame(series).to_csv(root / "test_series.csv", index=False)
    return pd.DataFrame(series)


def dicom(path, values, uid="1.2.3", series="1.2.4", syntax=None):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = uid, series
    ds.Rows, ds.Columns = values.shape[-2:]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    if values.ndim == 3:
        ds.NumberOfFrames = len(values)
    ds.PixelData = np.asarray(values, dtype=np.uint16).tobytes()
    if syntax is not None:
        ds.compress(syntax)
    ds.save_as(path, enforce_file_format=True)
    return ds


def test_fresh_test_ids_and_global_ranks(tmp_path, monkeypatch):
    import knee_submit.inference as inference
    ids = ["new-study-4", "new-study-1", "new-study-9", "new-hidden-study"]
    root, output, bundle = tmp_path / "data", tmp_path / "output", tmp_path / "bundle"
    inputs(root, ids)
    write_json(bundle / "submission_bundle.json", {"primary_blend": "candidate05", "bundle_id": "toy"})
    calls = []
    def predict(_, __, reader, current_ids, caches, device):
        assert current_ids == ids and len(caches) == 4 and device == "cpu"
        calls.append(reader)
        frame = pd.DataFrame(np.tile(np.arange(1, 5)[:, None] / 5, (1, 12)), columns=LABELS)
        frame.insert(0, UID, current_ids)
        return frame, {"seconds": 0., "studies": 4}
    monkeypatch.setattr(inference, "predict_reader", predict)
    frame, report = inference.run_inference(bundle, root, output, device="cpu", input_format="jpeg",
                                            work_dir=tmp_path / "work")
    assert calls == list(READERS) and report["studies"] == 4 and report["head_count"] == 20
    assert frame[UID].tolist() == ids
    np.testing.assert_allclose(frame["ACL"], [.25, .5, .75, 1.])
    assert pd.read_csv(output / "submission.csv")[UID].tolist() == ids
    assert not list((tmp_path / "work").glob("pixels_*"))


def test_failed_run_removes_stale_submission(tmp_path, monkeypatch):
    import knee_submit.inference as inference
    root = tmp_path / "output"
    root.mkdir()
    (root / "submission.csv").write_text("old predictions")
    bundle = tmp_path / "bundle"
    write_json(bundle / "submission_bundle.json", {"primary_blend": "candidate05"})
    def fail(*args):
        raise RuntimeError("fixture input failure")
    monkeypatch.setattr(inference, "prepare_test", fail)
    with pytest.raises(RuntimeError, match="fixture"):
        inference.run_inference(bundle, tmp_path / "data", root, device="cpu", input_format="jpeg",
                                work_dir=tmp_path / "work")
    assert not (root / "submission.csv").exists()


def test_series_choice_hash_sampling_and_missing_slots(tmp_path):
    uid = "study"
    rows = []
    for series_uid, count in [("b", 10), ("a", 10), ("c", 4)]:
        directory = tmp_path / "test_series" / uid / series_uid
        directory.mkdir(parents=True)
        for i in range(count):
            Image.new("L", (224, 224), color=i + 10).save(directory / f"{i}.jpeg")
        rows.append({UID: uid, SERIES: series_uid, "Anatomical_Plane": "Coronal", "Fat_Suppression": 0})
    errors = []
    a, counts, chosen = select_study(tmp_path, uid, pd.DataFrame(rows), "jpeg", errors)
    b, _, _ = select_study(tmp_path, uid, pd.DataFrame(rows[::-1]), "jpeg", errors)
    assert len(chosen) == 8 and {x[SERIES] for x in chosen} == {"a"}
    assert (a["slots"] == 4).all() and len(set(x["logical_path"] for x in chosen)) == 8
    np.testing.assert_array_equal(a["pixels"], b["pixels"])
    names = [f"test_series/{uid}/a/{i}.jpeg" for i in range(10)]
    expected = sorted(names, key=lambda p: fingerprint([42, uid, "a", p]))[:8]
    assert [x["logical_path"] for x in chosen] == expected
    values, mask = feature_tensor(np.ones((8, 4), np.float32), a["slots"], a["tokens"], 4)
    assert values.dtype == np.float16 and mask.sum() == 8 and not mask[:4].any()


@pytest.mark.parametrize("shape", [(8, 9), (2, 8, 9)])
def test_dicom_single_and_multi_frame(tmp_path, shape):
    values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    path = tmp_path / "slice.dcm"
    dicom(path, values)
    frames = list(dicom_frames(path, "1.2.3", "1.2.4"))
    expected = values[None] if values.ndim == 2 else values
    assert len(frames) == len(expected)
    for record, raw in zip(frames, expected):
        np.testing.assert_array_equal(record[3], raw)
        image = pixel_array_to_jpeg_pixels(record[3])
        assert image.shape == (224, 224) and image.dtype == np.uint8


def test_compressed_rle_and_constant_pixels(tmp_path):
    from pydicom.uid import RLELossless
    path = tmp_path / "compressed.dcm"
    values = np.arange(256, dtype=np.uint16).reshape(16, 16)
    dicom(path, values, syntax=RLELossless)
    actual = list(dicom_frames(path, "1.2.3", "1.2.4"))[0][3]
    np.testing.assert_array_equal(actual, values)
    assert not pixel_array_to_jpeg_pixels(np.zeros((8, 8))).any()


def test_missing_decoder_is_fatal_and_corrupt_jpeg_is_logged(tmp_path, monkeypatch):
    import knee_submit.data as data
    directory = tmp_path / "images"
    directory.mkdir()
    (directory / "bad.jpeg").write_bytes(b"corrupt")
    Image.new("L", (224, 224), color=20).save(directory / "good.jpeg")
    errors = []
    count, picked = series_candidates(directory, "study", "series", "jpeg", errors)
    assert count == 1 and len(picked) == 1 and len(errors) == 1
    (directory / "slice.dcm").write_bytes(b"fixture")
    def missing(*args):
        raise ImportError("required decoder absent")
    monkeypatch.setattr(data, "dicom_frames", missing)
    with pytest.raises(ImportError, match="decoder"):
        series_candidates(directory, "study", "series", "dicom", [])


def test_invalid_uids_and_feature_indices_refused(tmp_path):
    root = tmp_path / "data"
    inputs(root, ["one"])
    pd.DataFrame({UID: ["one", "one"]}).to_csv(root / "test.csv", index=False)
    with pytest.raises(ValueError):
        read_inputs(root)
    with pytest.raises(ValueError):
        feature_tensor(np.ones((2, 4)), np.array([0, 0]), np.array([0, 0]), 4)
    with pytest.raises(ValueError):
        slot_of("Unknown", 1)


@pytest.mark.local_assets
def test_blends_match_existing_completed_run_and_ignore_row_order():
    root = Path(__file__).resolve().parents[2] / "rsna_reports"
    predictions = {r: pd.read_csv(root / "predictions/test" / f"{r}.csv", dtype={UID: str}) for r in READERS}
    result = build_blends(predictions)
    for name, frame in result.items():
        reference = pd.read_csv(root / "submissions" / f"submission_{name}.csv", dtype={UID: str})
        assert frame[UID].tolist() == reference[UID].tolist()
        np.testing.assert_allclose(frame[LABELS], reference[LABELS], atol=1e-9, rtol=0)
    predictions["tiny"] = predictions["tiny"].iloc[::-1].reset_index(drop=True)
    permuted = build_blends(predictions)
    np.testing.assert_allclose(result["candidate05"][LABELS], permuted["candidate05"][LABELS])
    bad = result["candidate05"].copy()
    bad.loc[0, "ACL"] = np.nan
    with pytest.raises(ValueError):
        validate_submission(bad, bad[UID].tolist())
