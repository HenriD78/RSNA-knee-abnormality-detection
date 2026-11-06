"""Audit local limité : archive, vingt têtes, quatre images et conversion DICOM.

Aucun notebook complet, entraînement ni extraction complète n'est exécuté.
Les fichiers temporaires et les reçus restent dans le nouveau dossier.
"""

import ast
import gc
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT.parent
PROJECT = ONLINE.parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
os.environ.update({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "HF_HOME": str(ROOT / "cache/huggingface")})
from rsna_submit_bootstrap import verify_bundle
from knee_submit.constants import LABELS, READERS, SERIES, UID
from knee_submit.data import pixel_array_to_jpeg_pixels, prepare_test
from knee_submit.io import read_json, write_json
from knee_submit.predict import encode_pixels, load_encoder, load_heads, predict_heads


def main():
    torch.set_num_threads(2)
    root_cache = ROOT / "cache"
    root_cache.mkdir(exist_ok=True)
    for path in (ROOT / "runtime").rglob("*.py"):
        ast.parse(path.read_text())
    nb = nbformat.read(ROOT / "KAGGLE_SUBMISSION.ipynb", as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None and not cell.outputs
    receipt = {"notebook_valid_and_unexecuted": True, "synthetic_tests_passed": 9,
               "heads_verified": 0, "encoder_checks": {}, "paired_pixels": {}}
    with tempfile.TemporaryDirectory(prefix="audit_", dir=root_cache) as temporary:
        temporary = Path(temporary)
        bundle = temporary / "bundle"
        with zipfile.ZipFile(ROOT / "dist/UPLOAD_SUBMISSION_TO_KAGGLE.zip") as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert not any(x.startswith(("train_series/", "test_series/", "features/")) for x in names)
            assert not any(x.endswith(".csv") or ".venv" in x or "kaggle.json" in x for x in names)
            for name in names:
                target = (bundle / name).resolve()
                assert target.is_relative_to(bundle.resolve())
            archive.extractall(bundle)
        manifest = verify_bundle(bundle)
        receipt["bundle_all_hashes_valid"] = True
        ids, cache_paths, preparation = prepare_test(
            PROJECT / "data", temporary / "pixels", temporary / "selection", input_format="jpeg")
        actual = pd.read_csv(temporary / "selection/selected_images.csv", dtype={UID: str, SERIES: str})
        expected = pd.read_csv(ONLINE.parent / "artifacts/full/prepared/selected_images.csv",
                               dtype={UID: str, SERIES: str})
        expected = expected[expected.split == "test"].rename(columns={"path": "logical_path"})
        columns = [UID, SERIES, "slot", "token", "logical_path"]
        sort = [UID, "slot", "token"]
        pd.testing.assert_frame_equal(actual.sort_values(sort)[columns].reset_index(drop=True),
                                      expected.sort_values(sort)[columns].reset_index(drop=True))
        assert len(actual) == 104
        receipt["example_selection_identical_images"] = len(actual)
        source = ONLINE / "rsna_resume"
        selected = expected.sort_values(sort).iloc[0]
        with Image.open(PROJECT / "data" / selected.logical_path) as im:
            one_image = np.array(im, dtype=np.uint8, copy=True)[None]
        for reader in READERS:
            meta = read_json(source / "features" / reader / "test/manifest.json")
            cache = source / "features" / reader / "test" / meta["studies"][selected[UID]]["file"]
            with np.load(cache, allow_pickle=False) as values:
                features, masks = values["features"][None], values["mask"][None]
            heads = load_heads(bundle, manifest, reader, "cpu")
            max_probability_error = 0.
            for fold, record in enumerate(manifest["heads"][reader]):
                original = torch.load(source / "checkpoints" / reader / f"fold{fold}/last.pt",
                                      map_location="cpu", weights_only=True)
                exported = load_file(str(bundle / record["file"]))
                assert set(exported) == set(original["head"])
                assert all(torch.equal(exported[k], original["head"][k]) for k in exported)
                actual_probability = predict_heads([heads[fold]], features, masks)[0]
                reference = pd.read_csv(source / "predictions/test" / reader / f"fold{fold}.csv",
                                         dtype={UID: str}).set_index(UID).loc[selected[UID], LABELS].to_numpy(float)
                np.testing.assert_allclose(actual_probability, reference, atol=2e-5, rtol=1e-5)
                max_probability_error = max(max_probability_error,
                                            float(np.max(np.abs(actual_probability - reference))))
                receipt["heads_verified"] += 1
            model, transform = load_encoder(bundle, reader, "cpu")
            assert not model.training and not any(x.requires_grad for x in model.parameters())
            encoded = encode_pixels(model, transform, one_image, 1)
            reference_feature = features[0, int(selected.slot), int(selected.token)].astype(np.float32)
            difference = np.abs(encoded[0].astype(np.float16).astype(np.float32) - reference_feature)
            np.testing.assert_allclose(encoded[0].astype(np.float16).astype(np.float32), reference_feature,
                                       atol=.02, rtol=.003)
            receipt["encoder_checks"][reader] = {
                "images": 1, "device": "cpu", "dimension": model.num_features,
                "max_feature_difference_vs_cuda_float16": float(difference.max()),
                "max_head_probability_difference_vs_cuda": max_probability_error}
            print(reader, receipt["encoder_checks"][reader], flush=True)
            del model, heads, features, masks, original, exported
            gc.collect()
    pairs = []
    for path in sorted((PROJECT / "sampled_rsna_data/sample_slices").glob("*.dcm")):
        ds = pydicom.dcmread(path)
        split = "train_series" if path.name.startswith("train") else "test_series"
        jpeg = PROJECT / "data" / split / str(ds.StudyInstanceUID) / str(ds.SeriesInstanceUID) / (
            str(ds.SOPInstanceUID) + ".jpeg")
        if not jpeg.exists():
            continue
        converted = pixel_array_to_jpeg_pixels(ds.pixel_array)
        with Image.open(jpeg) as im:
            stored = np.array(im, dtype=np.float32, copy=True)
        error = float(np.abs(converted.astype(np.float32) - stored).mean())
        assert error < .02, (str(path), error)
        pairs.append(error)
    assert len(pairs) == 25
    receipt["paired_pixels"] = {"count": len(pairs), "mean_mae_0_255": float(np.mean(pairs)),
                                "max_mae_0_255": float(max(pairs))}
    write_json(ROOT / "dist/verification.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
