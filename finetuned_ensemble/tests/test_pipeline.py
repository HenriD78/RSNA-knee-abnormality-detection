import copy
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from torch import nn

from rsna3.cache import Cache, pack, prepare, probe
from rsna3.common import LABELS, UID, atomic_json, digest, read_json, sha256, spec, validate_predictions
from rsna3.dicom import CodecError, RENDER, Slice, decode, ordered_slices, physical_render, render_study
from rsna3.evaluate import aucs, blend, paired_bootstrap, ranks
from rsna3.model import StudyHead, StudyModel, bce_contribution, denominators
from rsna3.pilot import projection

torch.set_num_threads(1)


def test_accumulation_matches_full_batch_and_preserves_confidence():
    generator = torch.Generator().manual_seed(42)
    initial = torch.randn(8, 12, generator=generator)
    y = torch.rand(8, 12, generator=generator)
    y[1, 2] = float("nan")
    w = torch.rand(8, 12, generator=generator)
    w[0] = .01
    w[:, 11] = 0
    full, accumulated = initial.clone().requires_grad_(), initial.clone().requires_grad_()
    denom = denominators(y, w)
    expected = bce_contribution(full, y, w, denom)
    expected.backward()
    actual = 0.
    for i in range(8):
        loss = bce_contribution(accumulated[i:i+1], y[i:i+1], w[i:i+1], denom)
        actual += float(loss.detach())
        loss.backward()
    torch.testing.assert_close(full.grad, accumulated.grad)
    assert actual == pytest.approx(float(expected.detach()))
    assert accumulated.grad[1, 2] == 0 and accumulated.grad[:, 11].abs().sum() == 0
    with pytest.raises(ValueError):
        denominators(y, torch.zeros_like(w))


@pytest.mark.parametrize("depth", [0, 2])
def test_joint_token_permutation_and_mask(depth):
    torch.manual_seed(7)
    head = StudyHead(16, {"projection_dim": 32, "dropout": .1, "transformer_layers": depth, "transformer_heads": 4}).eval()
    features = torch.randn(2, 9, 16)
    valid = torch.tensor([[True]*6+[False]*3, [True]*9])
    features[0, 6:] = float("nan")
    slots = torch.randint(6, (2, 9))
    positions = torch.rand(2, 9) * 2 - 1
    permutation = torch.randperm(9)
    with torch.no_grad():
        a = head(features, valid, slots, positions)
        b = head(features[:, permutation], valid[:, permutation], slots[:, permutation], positions[:, permutation])
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="masquée"):
        head(features, torch.zeros_like(valid), slots, positions)


@pytest.mark.parametrize("model_id", ["coatnet2", "convnext_small", "dinov2_small"])
@pytest.mark.local_assets
def test_real_generic_weights_forward_and_tail_backward(model_id):
    from rsna3.common import ROOT
    cfg = spec(model_id)
    torch.manual_seed(42)
    model = StudyModel(cfg, ROOT.parent / cfg["initialization"]["weight_path"])
    size = cfg["input"]["input_size"]
    x = torch.rand(1, 1, 3, size, size)
    inputs = {"images": x, "valid": torch.ones(1, 1, dtype=torch.bool),
              "slots": torch.zeros(1, 1, dtype=torch.long), "positions": torch.zeros(1, 1)}
    model.set_phase(False)
    model.train()
    head_logits = model(**inputs)
    head_logits.sum().backward()
    assert head_logits.shape == (1, 12) and torch.isfinite(head_logits).all()
    assert all(p.grad is None for p in model.backbone.parameters())
    model.zero_grad(set_to_none=True)
    model.set_phase(True)
    model.train()
    bn = [(module, module.running_mean.clone()) for module in model.backbone.modules() if isinstance(module, nn.BatchNorm2d)]
    model(**inputs).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
    for module, before in bn:
        assert not module.training
        torch.testing.assert_close(module.running_mean, before, rtol=0, atol=0)
    if model_id == "dinov2_small":
        assert model.backbone.pos_embed.shape == (1, 257, 384)


def write_dicom(path, ipp, iop, mono="MONOCHROME2", frames=None):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.Rows, ds.Columns = 24, 32
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, mono
    ds.ImagePositionPatient, ds.ImageOrientationPatient, ds.PixelSpacing = list(map(float, ipp)), list(map(float, iop)), [1., 2.]
    ds.RescaleSlope, ds.RescaleIntercept = 2., -50.
    array = np.arange(24*32, dtype=np.uint16).reshape(24, 32)
    if frames:
        ds.NumberOfFrames = frames
        sequence = []
        for j in range(frames):
            group = pydicom.dataset.Dataset()
            position = pydicom.dataset.Dataset()
            position.ImagePositionPatient = (np.array(ipp) + [j, 0, 0]).tolist()
            group.PlanePositionSequence = pydicom.sequence.Sequence([position])
            sequence.append(group)
        ds.PerFrameFunctionalGroupsSequence = pydicom.sequence.Sequence(sequence)
        array = np.stack([array+j for j in range(frames)])
    ds.PixelData = array.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return ds


def test_multiframe_rescale_monochrome1(tmp_path):
    path = tmp_path / "frames.dcm"
    write_dicom(path, [0, 0, 0], [0, 1, 0, 0, 0, -1], mono="MONOCHROME1", frames=5)
    slices = decode(path)
    assert len(slices) == 5
    assert slices[2].ipp[0] == 2
    assert slices[0].pixels[0, 0] == 50
    assert slices[0].pixels[0, 1] == 48
    ordered, coords, centres, axis, _ = ordered_slices(slices[::-1])
    np.testing.assert_array_equal(coords, np.arange(5))
    assert axis == 0 and list(centres) == [1, 2, 3]


def test_projection_sort_gap_and_duplicate():
    iop = np.array([0., 1., 0., 0., 0., -1.])  # normal -x, canonical +x
    slices = [Slice(np.ones((4, 4)), np.array([x, 0., 100.-x]), iop, np.ones(2), str(i), 0, "L", "")
              for i, x in enumerate([0, 1, 2, 20, 21, 22])]
    ordered, coords, good, axis, duplicates = ordered_slices(slices[::-1] + [slices[0]])
    assert axis == 0 and duplicates == 1
    np.testing.assert_array_equal(coords, [0, 1, 2, 20, 21, 22])
    assert list(good) == [1, 4]  # no triplet spanning the gap
    bad = copy.deepcopy(slices)
    bad[-1].iop = np.array([1., 0, 0, 0, 1, 0])
    with pytest.raises(ValueError, match="orientation"):
        ordered_slices(bad)
    bad = copy.deepcopy(slices)
    bad[-1].side = "R"
    with pytest.raises(ValueError, match="dans une série"):
        ordered_slices(bad)


def test_laterality_sources_blank_fallback_and_multiframe(tmp_path):
    path = tmp_path / "frames.dcm"
    ds = write_dicom(path, [0, 0, 0], [0, 1, 0, 0, 0, -1], frames=5)
    ds.Laterality, ds.ImageLaterality = "R", ""
    ds.save_as(path, enforce_file_format=True)
    slices = decode(path)
    assert all(s.side == "R" and s.side_source == "Laterality" for s in slices)
    assert slices[0].laterality_tags == {"ImageLaterality": "", "Laterality": "R"}
    ds.ImageLaterality = "L"
    for group in ds.PerFrameFunctionalGroupsSequence:
        anatomy = pydicom.dataset.Dataset()
        anatomy.FrameLaterality = "R"
        group.FrameAnatomySequence = pydicom.sequence.Sequence([anatomy])
    ds.save_as(path, enforce_file_format=True)
    assert all(s.side == "R" and s.side_source == "FrameLaterality" for s in decode(path))
    ds.PerFrameFunctionalGroupsSequence[-1].FrameAnatomySequence[0].FrameLaterality = "L"
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match="dans une série"):
        ordered_slices(decode(path))


def test_between_series_laterality_warning_preserves_images_and_cache(tmp_path):
    uid, rows = "study", []
    for sid, plane, side, iop, direction in [
            ("sag", "Sagittal", "R", [0, 1, 0, 0, 0, -1], [1, 0, 0]),
            ("cor", "Coronal", "L", [1, 0, 0, 0, 0, -1], [0, 1, 0])]:
        rows.append({UID: uid, "SeriesInstanceUID": sid, "Anatomical_Plane": plane, "Fat_Suppression": 1})
        frame_uid = generate_uid()  # Different frames must not be treated as registered.
        for j in range(5):
            path = tmp_path / "dicom" / uid / sid / f"{j}.dcm"
            ds = write_dicom(path, np.array(direction) * j, iop)
            ds.Laterality, ds.FrameOfReferenceUID = side, frame_uid
            ds.save_as(path, enforce_file_format=True)
    series = pd.DataFrame(rows)
    arrays, metadata = render_study(uid, series, tmp_path / "dicom")
    assert set(arrays["slots"]) == {0, 1} and not metadata["errors"]
    warning, = metadata["warnings"]
    assert warning["code"] == "cross_series_laterality_conflict" and warning["study"] == uid
    assert {entry["series"] for entry in warning["series"]} == {"sag", "cor"}
    assert len({entry["frame_of_reference_uids"][0] for entry in warning["series"]}) == 2
    assert {w["laterality"] for w in metadata["windows"]} == {"L", "R"}
    assert all(w["laterality_source"] == "Laterality" for w in metadata["windows"])
    assert all(w["laterality_tags"]["Laterality"] == w["laterality"] for w in metadata["windows"])
    series_path = tmp_path / "series.csv"
    series.to_csv(series_path, index=False)
    manifest = prepare([uid], series_path, tmp_path / "dicom", tmp_path / "cache", fixture=True)
    assert manifest["laterality_conflict_studies"] == [uid]
    assert manifest["entries"][uid]["warnings"] == 1
    entry = manifest["entries"][uid]
    with zipfile.ZipFile(tmp_path / "cache" / entry["shard"]) as z:
        with np.load(io.BytesIO(z.read(entry["member"]))) as archive:
            assert json.loads(archive["metadata"].tobytes())["warnings"] == [warning]
    for path in (tmp_path / "dicom" / uid / "cor").glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.Laterality = "R"
        ds.save_as(path, enforce_file_format=True)
    consistent, info = render_study(uid, series, tmp_path / "dicom")
    assert info["warnings"] == []
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], consistent[key])


def test_probe_reports_laterality_conflicts(tmp_path, monkeypatch):
    studies = [str(i) for i in range(50)]
    series_path = tmp_path / "series.csv"
    pd.DataFrame({UID: studies}).to_csv(series_path, index=False)
    def render(uid, *_):
        return {"images": np.zeros((1, 3, 4, 4), dtype=np.uint8)}, {
            "errors": [], "syntaxes": [], "warnings": [
                {"code": "cross_series_laterality_conflict", "study": uid}] if uid == "7" else []}
    monkeypatch.setattr("rsna3.cache.render_study", render)
    report = probe(studies, series_path, tmp_path, tmp_path / "probe.json", n=50)
    assert report["laterality_conflict_studies"] == ["7"]
    assert len(report["sample"]) == 50
    assert next(row for row in report["sample"] if row[UID] == "7")["warnings"]


def test_physical_render_uses_spacing_and_canonical_axes():
    yy, xx = np.mgrid[:300, :300]
    s = Slice(xx.astype(np.float32), np.zeros(3), np.array([1., 0, 0, 0, 1, 0]), np.array([1., 2.]), "s", 0, "R", "")
    image, centre = physical_render(s, 2, 0, 300)
    assert image.shape == (384, 384) and image.dtype == np.uint8
    assert centre == [299., 149.5, 0.]
    # 153.6 mm covers 76.8 source columns at spacing 2 mm.
    expected_span = 76.6 / 300 * 255
    assert int(image[192, -1]) - int(image[192, 0]) == pytest.approx(expected_span, abs=2)


@pytest.mark.local_assets
def test_real_fixture_files_are_decodable():
    from rsna3.common import ROOT
    files = sorted((ROOT.parent / "sampled_rsna_data/dicoms").rglob("*.dcm"))
    assert files
    frames = [s for p in files for s in decode(p)]
    assert all(np.isfinite(s.pixels).all() for s in frames)


def test_cache_resume_series_slots_and_missing_slots(tmp_path):
    uid = "study"
    rows = []
    for sid, count, fs in [("seriesA", 7, 1), ("seriesB", 5, 1), ("seriesC", 5, 0)]:
        rows.append({UID: uid, "SeriesInstanceUID": sid, "Anatomical_Plane": "Sagittal", "Fat_Suppression": fs})
        for j in range(count):
            write_dicom(tmp_path / "dicom" / uid / sid / f"{count-j}.dcm", [j*2, 0, 0], [0, 1, 0, 0, 0, -1])
    series = tmp_path / "series.csv"
    pd.DataFrame(rows).to_csv(series, index=False)
    output = tmp_path / "cache"
    result = prepare([uid], series, tmp_path / "dicom", output, fixture=True)
    reader = Cache(output, verify=True, allow_fixture=True)
    arrays = reader.read(uid)
    assert set(arrays["slots"]) == {0, 3}  # no non-FS-to-FS substitution
    assert len(arrays["images"]) == 7  # 4 + 3 distinct valid centres
    assert (np.diff(arrays["positions"][:4]) > 0).all()
    reader.close()
    before = sha256(output / "shard-0000.zip")
    prepare([uid], series, tmp_path / "dicom", output, fixture=True)
    assert sha256(output / "shard-0000.zip") == before
    info = result["entries"][uid]
    with zipfile.ZipFile(output / info["shard"]) as z:
        with np.load(io.BytesIO(z.read(info["member"]))) as npz:
            meta = json.loads(npz["metadata"].tobytes())
    assert {w["series"] for w in meta["windows"]} == {"seriesA", "seriesC"}
    for window in meta["windows"]:
        assert len(window["sops_frames"]) == 3
    with pytest.raises(ValueError, match="fixture"):
        Cache(output)


def test_empty_study_and_codec_failure_stop(tmp_path, monkeypatch):
    series = pd.DataFrame([{UID: "a", "SeriesInstanceUID": "s", "Fat_Suppression": 1, "Anatomical_Plane": "Sagittal"}])
    with pytest.raises(ValueError, match="inexploitable"):
        render_study("a", series, tmp_path)
    path = tmp_path / "a/s/1.dcm"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"placeholder")
    def broken(*_):
        raise CodecError("decoder unavailable")
    monkeypatch.setattr("rsna3.dicom.decode", broken)
    with pytest.raises(CodecError):
        render_study("a", series, tmp_path)


def test_ranks_auc_alignment_and_budget():
    x = np.repeat(np.array([[.1], [.1], [.8]]), 12, axis=1)
    assert ranks(x)[0, 0] == .25 and ranks(x)[2, 0] == 1
    assert ranks(x[:1])[0, 0] == .5
    np.testing.assert_array_equal(blend({"coatnet2": x, "convnext_small": x, "dinov2_small": x}), ranks(x))
    assert aucs(np.zeros((3, 12)), x) == [None]*12
    frame = pd.DataFrame({UID: ["b", "a", "c"], **dict(zip(LABELS, x.T))})
    assert validate_predictions(frame, ["a", "b", "c"])[UID].tolist() == ["a", "b", "c"]
    with pytest.raises(ValueError, match="Couverture"):
        validate_predictions(frame.iloc[:2], ["a", "b", "c"])
    p = {"head_seconds_per_study": 1., "tail_seconds_per_study": 1., "eval_seconds_per_study": 1.,
         "reload_seconds": 0., "checkpoint_seconds": 0.}
    # 5 tasks, 2 GPUs = 3 waves. Each task = 40 train*2 + 10 heldout = 90s.
    assert projection(p, [10]*5, 2, quota_factor=2, gpus=2) == pytest.approx(270 * 1.2 * 2 / 3600)


def test_bootstrap_is_paired_by_group():
    truth = np.repeat(np.array([[0], [1], [0], [1], [0], [1]], float), 12, axis=1)
    prediction = truth * .8 + .1
    result = paired_bootstrap(truth, np.array(["a", "a", "b", "b", "c", "c"]),
                              {"baseline_frozen": prediction, "same": prediction}, repetitions=25)
    assert result["valid"] == 25 and result["differences"]["same"]["ci95"] == [0., 0.]


class TinyModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.backbone = nn.Linear(3, 8)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(8, 12))
        self.set_phase(False)

    def set_phase(self, adapt):
        self.backbone.requires_grad_(adapt)

    def forward(self, images, valid, slots, positions):
        x = images.mean((-1, -2)).mean(1)
        return self.head(torch.tanh(self.backbone(x)))


def tiny_training_data(tmp_path, n=11, five_folds=False):
    from rsna3.dicom import RENDER
    cache = tmp_path / "cache"
    cache.mkdir()
    entries, rows = {}, []
    with zipfile.ZipFile(cache / "shard.zip", "w") as z:
        for i in range(n):
            uid = f"study-{i}"
            arrays = {"images": np.random.default_rng(i).integers(0, 256, (2, 3, 16, 16), dtype=np.uint8),
                      "slots": np.array([0, 3]), "positions": np.array([-.5, .5], np.float32)}
            payload = pack(arrays, {"test": True})
            z.writestr(uid+".npz", payload)
            import hashlib
            entries[uid] = {"shard": "shard.zip", "member": uid+".npz", "sha256": hashlib.sha256(payload).hexdigest()}
            row = {UID: uid, "fold": i % 5 if five_folds else (0 if i < 2 else 1),
                   "group_id": str(i), "is_gold": i < n//2 if five_folds else i % 2 == 0}
            for j, label in enumerate(LABELS):
                row[label], row[label+"__conf"] = ((i+j) % 2), (.1 if i % 2 else 1.)
            rows.append(row)
    atomic_json(cache / "manifest.json", {"complete": True, "contract": {"render": RENDER, "fixture": True},
                "shards": [{"file": "shard.zip", "sha256": sha256(cache / "shard.zip")}], "entries": entries})
    return cache, pd.DataFrame(rows)


def test_optimizer_boundary_resume_is_exact(tmp_path, monkeypatch):
    from rsna3 import train
    monkeypatch.setattr(train, "StudyModel", TinyModel)
    cache, table = tiny_training_data(tmp_path)
    cfg = spec("convnext_small")
    cfg["input"]["input_size"] = 16
    cfg["training"].update(epochs_target=2, loader_workers_per_gpu=int(os.environ.get("RSNA3_TEST_WORKERS", "0")))
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    kwargs = dict(experiment=cfg, project=tmp_path, cache=cache, fold=0, device="cpu", fixture_table=table, fixture_weight="unused")
    train.train_fold(output=full, **kwargs)
    result = train.train_fold(output=interrupted, stop_after=3, **kwargs)
    assert result["status"] == "paused_at_step" and result["epoch"] == 1
    state = torch.load(interrupted / "convnext_small/fold0/active.pt", weights_only=False)
    assert state["accumulator_empty"] and state["offset"] == 8
    train.train_fold(output=interrupted, **kwargs)
    from safetensors.torch import load_file
    a = load_file(str(full / "convnext_small/fold0/final_fold0.safetensors"))
    b = load_file(str(interrupted / "convnext_small/fold0/final_fold0.safetensors"))
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    p1, p2 = pd.read_csv(full / "convnext_small/fold0/oof.csv"), pd.read_csv(interrupted / "convnext_small/fold0/oof.csv")
    np.testing.assert_array_equal(p1[LABELS], p2[LABELS])
    assert not (interrupted / "convnext_small/fold0/active.pt").exists()


def test_no_old_submission_after_failure(tmp_path):
    from rsna3.inference import submit
    out = tmp_path / "submission.csv"
    out.write_text("old csv")
    pd.DataFrame({UID: ["a"]}).to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(FileNotFoundError):
        submit(tmp_path / "absent", tmp_path / "test.csv", tmp_path / "series.csv", tmp_path,
               tmp_path / "work", out)
    assert not out.exists()


def test_fifteen_synthetic_folds_oof_evaluation_and_export(tmp_path, monkeypatch):
    from rsna3 import train, evaluate, inference
    from rsna3.common import SPECS
    monkeypatch.setattr(train, "StudyModel", TinyModel)
    cache, table = tiny_training_data(tmp_path, n=20, five_folds=True)
    for model_id in SPECS:
        cfg = spec(model_id)
        cfg["input"]["input_size"] = 16
        cfg["training"].update(epochs_target=2, loader_workers_per_gpu=0)
        cfg["cache_manifest_sha256"] = sha256(cache / "manifest.json")
        cfg["profile"] = {"cache_centres": 2, "cache_manifest_sha256": cfg["cache_manifest_sha256"], "config": copy.deepcopy(cfg)}
        for fold in range(5):
            train.train_fold(cfg, tmp_path, cache, tmp_path / "runs", fold, device="cpu", fixture_table=table, fixture_weight="unused")
    monkeypatch.setattr(evaluate, "input_tables", lambda *_: (table, {}))
    monkeypatch.setattr(inference, "input_tables", lambda *_: (table, {}))
    baseline = tmp_path / "baseline"
    for cohort, mask in [("gold", table.is_gold), ("weak_proxy", ~table.is_gold)]:
        (baseline / cohort).mkdir(parents=True)
        frame = table.loc[mask, [UID]+LABELS].copy()
        frame[LABELS] = frame[LABELS] * .8 + .1
        frame.to_csv(baseline / cohort / "candidate05.csv", index=False)
    summary = evaluate.evaluate(tmp_path, tmp_path / "runs", tmp_path / "reports", baseline, repetitions=25)
    assert summary["bootstrap"]["valid"] == 25
    assert summary["cohorts"]["gold"]["ensemble_equal"]["studies"] == 10
    assert not list((tmp_path / "runs").glob("*/oof_*.csv"))
    assert (tmp_path / "reports/oof_convnext_small.csv").is_file()
    metadata = inference.export(tmp_path, tmp_path / "runs", tmp_path / "export")
    assert len(metadata["files"]) == 15 and metadata["centres"] == 2
    assert not list((tmp_path / "export").rglob("active.pt"))
    # Partial coverage must stop collection; no synthetic substitute checkpoint.
    (tmp_path / "runs/coatnet2/fold4/completed.json").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate.collect_model(tmp_path / "runs", "coatnet2", table)


def test_global_public_fallback_and_partial_base_rejected(tmp_path, monkeypatch):
    from rsna3 import inference
    ids = ["b", "a", "c"]
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({UID: ids}).to_csv(test_csv, index=False)
    base = pd.DataFrame({UID: ids, **{label: [.2, .8, .5] for label in LABELS}})
    monkeypatch.setattr(inference, "run_public_recipe", lambda *_: (base, {"recipe": "fixture"}))
    result = inference.submit(tmp_path / "missing", test_csv, tmp_path / "series.csv", tmp_path,
                              tmp_path / "work", tmp_path / "submission.csv", public_recipe="fixture")
    assert result["recipe"] == "public_base_global_fallback"
    final = pd.read_csv(tmp_path / "submission.csv")
    np.testing.assert_array_equal(final[LABELS], base[LABELS])
    assert final[UID].tolist() == ids
    monkeypatch.setattr(inference, "run_public_recipe", lambda *_: (base.iloc[:2], {}))
    with pytest.raises(ValueError, match="Couverture"):
        inference.submit(tmp_path / "missing", test_csv, tmp_path / "series.csv", tmp_path,
                         tmp_path / "work", tmp_path / "submission.csv", public_recipe="fixture")
    assert not (tmp_path / "submission.csv").exists()


def test_final_rank_uses_mean_fold_probabilities():
    # Averaging ranks of folds gives a different answer here: contract is probability first.
    f1 = np.array([[.01], [.6], [.7]])
    f2 = np.array([[.99], [.5], [.1]])
    mean = np.repeat((f1+f2)/2, 12, axis=1)
    actual = blend({"coatnet2": mean, "convnext_small": mean, "dinov2_small": mean})
    assert list(actual[:, 0]) == [.5, 1., 0.]


def test_two_gpu_queue_has_no_collision_and_respects_priority(tmp_path, monkeypatch):
    from rsna3 import queue
    records, alive = [], set()
    maximum = [0]
    class FakeProcess:
        serial = 100
        def __init__(self, command, env, **_):
            gpu = env["CUDA_VISIBLE_DEVICES"]
            assert gpu not in alive
            alive.add(gpu)
            maximum[0] = max(maximum[0], len(alive))
            self.gpu, self.polls, self.pid = gpu, 0, FakeProcess.serial
            FakeProcess.serial += 1
            cfg = read_json(command[command.index("--experiment")+1])
            fold = int(command[command.index("--fold")+1])
            records.append((cfg["model_id"], fold, gpu))
            output = Path(command[command.index("--output")+1]) / cfg["model_id"] / f"fold{fold}"
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output / "completed.json", {"test_only": True})
        def poll(self):
            self.polls += 1
            if self.polls >= 2:
                alive.discard(self.gpu)
                return 0
            return None
        def terminate(self):
            self.polls = 2
        def kill(self):
            self.polls = 2
        def wait(self, **_):
            alive.discard(self.gpu)
            return 0
    from pathlib import Path
    monkeypatch.setattr(queue.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(queue.time, "sleep", lambda *_: None)
    paths = []
    for model_id in ["dinov2_small", "coatnet2", "convnext_small"]:
        cfg = spec(model_id)
        cfg.update(status="frozen_after_pilot", profiled_gpus=2, quota_factor=2.)
        path = tmp_path / f"{model_id}.json"
        atomic_json(path, cfg)
        paths.append(path)
    result = queue.run_queue(paths, tmp_path, tmp_path, tmp_path / "runs", [0, 1], quota_available=24, session_hours=1)
    assert maximum[0] == 2 and not alive
    assert len(records) == len(set((m, f) for m, f, _ in records)) == 15
    assert [m for m, _, _ in records[:5]] == ["convnext_small"]*5
    assert all(job["status"] == "completed" for job in result["jobs"].values())
