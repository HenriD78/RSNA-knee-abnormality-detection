"""Execute the inference notebook and verify transfer from the trained baseline.

Run from any directory: .venv/bin/python scripts/verification/verify_mae_submission.py
Does not train, upload, or modify the deliverable notebook or model weights.
"""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[2]
(root / "local/scratch").mkdir(parents=True, exist_ok=True)
notebook = nbformat.read(root / "notebooks/mae/mae_submission.ipynb", as_version=4)
nbformat.validate(notebook)
notebook.cells.append(nbformat.v4.new_code_cell(r'''
import math
import shutil
import tempfile
from torch.nn import functional as F

run = PROJECT_ROOT / "outputs/mae_baseline" / manifest["source_run"]
verification = {"example_test_studies": len(submission), "example_test_frames": report["frames"],
                "example_inference_seconds": report["inference_seconds"]}
assert len(submission) == 3 and report["frames"] == 557
assert report["decode_failures"] == 0 and report["neutral_fallback_studies"] == 0

# Every saved fold must reproduce its own held-out predictions, not in-sample fits.
with np.load(run / "features_mae.npz") as features:
    study_ids = features["study_ids"].tolist()
    means = features["mean"].copy()
oof = pd.read_csv(run / "oof_mae_mean.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
errors = []
for uid, feature in zip(study_ids, means):
    fold = int(oof.loc[uid, "fold"])
    expected = oof.loc[uid, LABELS].to_numpy(dtype=np.float64)
    actual = fold_probabilities(feature)[fold]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    errors.append(float(np.max(np.abs(actual - expected))))
verification["oof_prediction_max_abs_error"] = max(errors)
verification["oof_studies_checked"] = len(study_ids)

# Verify the trimmed encoder against the original notebook's actual model code.
baseline = json.loads((PROJECT_ROOT / "notebooks/mae/mae_baseline.ipynb").read_text())
model_code = "".join(baseline["cells"][6]["source"]).split("torch.manual_seed(SEED)")[0]
scope = {"torch": torch, "nn": nn, "F": F, "math": math}
exec(compile(model_code, "baseline_model_cell", "exec"), scope)
reference = scope["SmallMAE"](CFG).to(DEVICE).eval()
checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
reference.load_state_dict(checkpoint["model"])
sample_jpegs = sorted((COMPETITION_DIR / "test_series" / ids[0]).glob("*/*.jpeg"))[:4]
batch = torch.stack([next(read_frames(path, "jpeg")) for path in sample_jpegs]).to(DEVICE)
with torch.inference_mode():
    expected = reference.encode_images(batch)
    actual = encoder(batch)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
verification["original_encoder_exact_match"] = True
del reference, checkpoint

# Full-study aggregation must also agree with the baseline's stored features.
uid = study_ids[0]
paths = sorted((PROJECT_ROOT / "data/train_series" / uid).glob("*/*.jpeg"))
feature, frame_count, failures = study_feature(paths, "jpeg")
assert not failures and frame_count == len(paths)
np.testing.assert_allclose(feature, means[0], rtol=1e-4, atol=1e-5)
verification["saved_study_feature_max_abs_error"] = float(np.max(np.abs(feature - means[0])))

# Paired raw-DICOM and training-JPEG audit: pixel and feature/prediction drift.
pairs, syntaxes, pair_drift = [], set(), []
test_dicoms = []
for path in sorted((PROJECT_ROOT / "sampled_rsna_data/sample_slices").glob("*.dcm")):
    dcm = pydicom.dcmread(path)
    split = "train_series" if path.name.startswith("train") else "test_series"
    jpeg = PROJECT_ROOT / "data" / split / str(dcm.StudyInstanceUID) / str(dcm.SeriesInstanceUID) / (str(dcm.SOPInstanceUID) + ".jpeg")
    if not jpeg.exists():
        continue
    raw = next(read_frames(path, "dicom"))
    stored = next(read_frames(jpeg, "jpeg"))
    mae = float((raw - stored).abs().mean() * 255)
    assert mae < 0.02, (path, mae)
    with torch.inference_mode():
        features = encoder(torch.stack([raw, stored]).to(DEVICE)).cpu().numpy()
    drift = float(np.max(np.abs(predict_feature(features[0]) - predict_feature(features[1]))))
    assert drift < 0.001, (path, drift)
    pair_drift.append(drift)
    syntaxes.add(str(dcm.file_meta.TransferSyntaxUID))
    pairs.append({"slice": str(dcm.SOPInstanceUID), "split": split, "pixel_mae_0_255": mae,
                  "max_probability_difference": drift})
    if split == "test_series":
        test_dicoms.append((path, str(dcm.StudyInstanceUID), str(dcm.SeriesInstanceUID), str(dcm.SOPInstanceUID)))
assert len(pairs) == 25
pd.DataFrame(pairs).to_csv(PROJECT_ROOT / "submission/preprocessing_audit.csv", index=False)
verification["paired_dicom_jpegs"] = len(pairs)
verification["pixel_mae_mean_0_255"] = float(np.mean([p["pixel_mae_0_255"] for p in pairs]))
verification["pixel_mae_max_0_255"] = max(p["pixel_mae_0_255"] for p in pairs)
verification["paired_prediction_max_abs_error"] = max(pair_drift)
verification["tested_transfer_syntaxes"] = sorted(syntaxes)

# Exercise real DICOM inference through the exact final notebook cell.
# This fixture contains only downloaded slices, not complete test volumes.
with tempfile.TemporaryDirectory(prefix="mae-inference-check-") as temporary:
    fixture = Path(temporary)
    for src, study, series, sop in test_dicoms:
        destination = fixture / "test_series" / study / series / (sop + ".dcm")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
    fixture_ids = sorted({p[1] for p in test_dicoms})
    pd.DataFrame({"StudyInstanceUID": fixture_ids}).to_csv(fixture / "test.csv", index=False)
    original_competition, original_output = COMPETITION_DIR, OUTPUT_DIR
    COMPETITION_DIR, INPUT_FORMAT = fixture, "dicom"
    OUTPUT_DIR = PROJECT_ROOT / "local/scratch/mae_dicom_fixture"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
    final_cell = "".join(json.loads((PROJECT_ROOT / "notebooks/mae/mae_submission.ipynb").read_text())["cells"][-1]["source"])
    exec(compile(final_cell, "dicom_fixture_inference", "exec"), globals())
    assert report["frames"] == len(test_dicoms) and report["decode_failures"] == 0
    verification["dicom_fixture_studies"] = len(fixture_ids)
    verification["dicom_fixture_frames"] = len(test_dicoms)

    # Hidden data uses compressed syntaxes: lossless compression must preserve
    # pixels through our actual reader. Test multi-frame input at the same time.
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, RLELossless, generate_uid
    raw_pixels = np.arange(2 * 64 * 64, dtype=np.uint16).reshape(2, 64, 64)
    compressed_results = []
    for syntax in [JPEG2000Lossless, RLELossless]:
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Rows, ds.Columns, ds.NumberOfFrames = 64, 64, 2
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = raw_pixels.tobytes()
        ds.compress(syntax)
        compressed = fixture / f"compressed_{syntax}.dcm"
        ds.save_as(compressed, enforce_file_format=True)
        frames = list(read_frames(compressed, "dicom"))
        assert len(frames) == 2
        for frame, original in zip(frames, raw_pixels):
            torch.testing.assert_close(frame, pixel_array_to_training_tensor(original), rtol=0, atol=0)
        compressed_results.append(syntax.name)
    verification["synthetic_compressed_multiframe_exact_match"] = compressed_results

    # Constant frames, malformed files, mixed valid/invalid data, no-data fallback.
    zero = pixel_array_to_training_tensor(np.zeros((32, 32), dtype=np.float32))
    assert zero.shape == (1, 224, 224) and torch.isfinite(zero).all()
    broken = fixture / "broken.dcm"
    broken.write_bytes(b"not a DICOM file")
    empty_feature, count, failures = study_feature([broken], "dicom")
    assert empty_feature is None and count == 0 and len(failures) == 1
    np.testing.assert_array_equal(predict_feature(empty_feature), np.full(12, 0.5))
    good_feature, count, failures = study_feature([broken, test_dicoms[0][0]], "dicom")
    assert good_feature is not None and count == 1 and len(failures) == 1
    # Nonfinite model features must not be swallowed as decode failures.
    saved_encoder = encoder
    class BadEncoder(nn.Module):
        def forward(self, x):
            return torch.full((len(x), CFG["encoder_dim"]), float("nan"), device=x.device)
    encoder = BadEncoder()
    try:
        study_feature([test_dicoms[0][0]], "dicom")
    except RuntimeError as exc:
        assert "Nonfinite encoder" in str(exc)
    else:
        raise AssertionError("Model failures must propagate")
    finally:
        encoder = saved_encoder
    verification["corrupt_slice_handling"] = True
    verification["model_failures_propagate"] = True

verification["kaggle_hidden_run_verified"] = False
(PROJECT_ROOT / "submission/verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print("VERIFICATION_RESULT=" + json.dumps(verification))
'''))

manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]


def announce(cell_index, **kwargs):
    print(f"Executing cell {cell_index}", flush=True)


client = NotebookClient(notebook, km=manager, timeout=600,
                        resources={"metadata": {"path": str(root)}}, on_cell_start=announce)
try:
    client.execute()
finally:
    nbformat.write(notebook, root / "local/scratch/mae_submission.executed.ipynb")
    for cell in notebook.cells:
        for entry in cell.get("outputs", []):
            if entry.output_type == "stream":
                print(entry.text, end="", flush=True)
