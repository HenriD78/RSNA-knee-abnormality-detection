"""Identités, archives et inférence sur un minuscule encodeur artificiel."""

import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import bootstrap
from knee_baseline.config import deadline_reached, load_config
from knee_baseline.features import FeatureStore, cache_identity, encode_images, import_seeds, logical_details
from knee_baseline.io import fingerprint, read_json, sha256, write_json
from knee_baseline.models import StudyHead, weighted_bce


def test_feature_identity_survives_platform_change():
    old = {"asset": {"weight_sha256": "same", "timm_version": "1.0.29"}, "transform": {"size": 224},
           "feature_dim": 4, "platform": "macOS", "torch_version": "2.14"}
    new = {**old, "platform": "Linux", "torch_version": "2.6"}
    assert logical_details(old) == logical_details(new)
    assert cache_identity({"signature": "same"}, "coatnet", old) == cache_identity({"signature": "same"}, "coatnet", new)
    changed = {**new, "transform": {"size": 384}}
    assert cache_identity({"signature": "same"}, "coatnet", old) != cache_identity({"signature": "same"}, "coatnet", changed)


@pytest.mark.parametrize("kind", ["parent", "symlink"])
def test_unsafe_tar_refused(tmp_path, kind):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("../escape" if kind == "parent" else "link")
        if kind == "symlink":
            info.type, info.linkname = tarfile.SYMTYPE, "/tmp/escape"
            archive.addfile(info)
        else:
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        bootstrap.safe_tar(path, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_seed_import_keeps_arrays_without_reencoding(tmp_path, monkeypatch):
    import knee_baseline.features as features
    seed = tmp_path / "seed"
    origin = FeatureStore(seed / "coatnet/train", "cpu", 4, 8, create=True)
    x = np.arange(6 * 8 * 4).reshape(6, 8, 4) / 100
    mask = np.ones((6, 8), bool)
    origin.put("study", x, mask); origin.flush()
    details = {"asset": {"weight_sha256": "fixed"}, "transform": {"size": 384},
               "feature_dim": 4, "platform": "CPU"}
    write_json(seed / "manifest.json", {"prepared": "portable", "readers": {
        "coatnet": {"encoder": details, "studies": origin.manifest["studies"]}}})
    monkeypatch.setenv("RSNA_SEED_ROOT", str(seed))
    monkeypatch.setattr(features, "load_prepared", lambda _: ({"signature": "portable"}, {}))
    target = FeatureStore(tmp_path / "gpu", "new-logical-id", 4, 8, create=True)
    import_seeds({}, "coatnet", "train", {**details, "platform": "GPU"}, target)
    np.testing.assert_array_equal(target.get("study")[0], origin.get("study")[0])
    import_seeds({}, "coatnet", "train", details, target)
    assert len(target.manifest["studies"]) == 1


def test_encode_inference_on_toy_network_and_oom_retry(tmp_path, monkeypatch):
    paths = []
    for i in range(4):
        p = tmp_path / f"{i}.jpeg"
        Image.new("L", (8, 8), color=i * 30).save(p); paths.append(p)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)
            self.sizes = []
        def forward(self, x):
            self.sizes.append(len(x))
            if len(x) > 1:
                raise torch.cuda.OutOfMemoryError("simulated")
            return self.conv(x).mean((2, 3))
    model = Toy().eval()
    transform = lambda im: torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1) / 255)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    values = encode_images(model, transform, paths, 4)
    assert values.shape == (4, 4) and np.isfinite(values).all()
    assert model.sizes[:2] == [4, 2] and model.sizes[-1] == 1


def test_budget_and_gpu_guard(monkeypatch):
    from knee_baseline.cloud import preflight
    monkeypatch.setenv("RSNA_STOP_AT", str(time.time() - 1))
    assert deadline_reached()
    monkeypatch.setenv("RSNA_STOP_AT", str(time.time() + 60))
    assert not deadline_reached()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        preflight()


def test_snapshot_contains_state_only_and_restores(tmp_path, monkeypatch):
    from knee_baseline.session import snapshot
    root, results = tmp_path / "state", tmp_path / "results"
    monkeypatch.setenv("RSNA_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("RSNA_RESULTS_DIR", str(results))
    (root / "features/coatnet/train").mkdir(parents=True)
    write_json(root / "features/coatnet/train/manifest.json", {"studies": {"s": {}}})
    (root / "features/coatnet/train/feature.bin").write_bytes(b"completed")
    (root / ".partial_unfinished").write_bytes(b"partial")
    path = snapshot(load_config())
    restored = tmp_path / "restored"
    bootstrap.safe_tar(path, restored)
    assert (restored / "features/coatnet/train/feature.bin").read_bytes() == b"completed"
    assert not (restored / ".partial_unfinished").exists()
    assert read_json(results / "SESSION_STATUS.json")["features"]["coatnet/train"] == 1


def test_heads_remain_masked_and_order_independent():
    torch.manual_seed(42)
    model = StudyHead(4, hidden_dim=16, depth=2, heads=4).eval()
    x, mask = torch.randn(2, 6, 3, 4), torch.ones((2, 6, 3), dtype=torch.bool)
    mask[:, 5, :] = False
    with torch.no_grad():
        original = model(x, mask)
        perturbed = x.clone()
        perturbed[:, 5] = 10000
        torch.testing.assert_close(original, model(perturbed, mask))
        order = [2, 0, 1]
        torch.testing.assert_close(original, model(x[:, :, order], mask[:, :, order]), atol=2e-5, rtol=2e-5)
        loss = weighted_bce(original, torch.zeros_like(original), torch.ones_like(original))
    assert torch.isfinite(loss)


def test_bundle_verification_detects_damage(tmp_path):
    p = tmp_path / "payload.bin"; p.write_bytes(b"complete")
    marker = {"format": "rsna-kaggle-upload-v1", "files": {p.name: sha256(p)}}
    marker["bundle_id"] = fingerprint(marker)
    write_json(tmp_path / "bundle_manifest.json", marker)
    bootstrap.verify_bundle(tmp_path)
    p.write_bytes(b"damaged")
    with pytest.raises(ValueError):
        bootstrap.verify_bundle(tmp_path)


def test_setup_and_resume_on_miniature_archives(tmp_path, monkeypatch):
    """Installe trois archives de quelques octets, sans pip, GPU ou modèle."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    files = {}
    for name in ["data_payload.bin", "assets_payload.bin", "seed_payload.bin"]:
        path = bundle / name
        with tarfile.open(path, "w") as archive:
            item = tarfile.TarInfo("example.txt")
            item.size = len(b"small")
            archive.addfile(item, io.BytesIO(b"small"))
        files[name] = sha256(path)
    marker = {"format": "rsna-kaggle-upload-v1", "files": files,
              "prepared": "synthetic", "seed_studies": {}}
    marker["bundle_id"] = fingerprint(marker)
    write_json(bundle / "bundle_manifest.json", marker)
    results = tmp_path / "results"
    first = bootstrap.setup(bundle, tmp_path / "first", results, install=False, require_gpu=False)
    assert (tmp_path / "first/inputs/example.txt").read_bytes() == b"small"
    root = Path(first["output"])
    (root / "completed.txt").write_text("keep")
    from knee_baseline.session import snapshot
    resume = snapshot(load_config())
    second = bootstrap.setup(bundle, tmp_path / "second", results,
                             resume=resume, install=False, require_gpu=False)
    assert (Path(second["output"]) / "completed.txt").read_text() == "keep"
    # Une session déjà active prime sur une ancienne archive attachée.
    (Path(second["output"]) / "completed.txt").write_text("newer")
    bootstrap.setup(bundle, tmp_path / "second", results,
                    resume=resume, install=False, require_gpu=False)
    assert (Path(second["output"]) / "completed.txt").read_text() == "newer"
    marker["prepared"] = "different"
    marker["bundle_id"] = fingerprint({k: v for k, v in marker.items() if k != "bundle_id"})
    write_json(bundle / "bundle_manifest.json", marker)
    with pytest.raises(ValueError, match="autre paquet"):
        bootstrap.setup(bundle, tmp_path / "wrong", results,
                        resume=resume, install=False, require_gpu=False)
