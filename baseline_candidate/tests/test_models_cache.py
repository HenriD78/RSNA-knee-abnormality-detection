"""Passes avant et lectures de checkpoints ; aucun entraînement réel."""

import numpy as np
import pytest
import torch
from torch import nn

from knee_baseline.features import FeatureStore, freeze_encoder
from knee_baseline.io import write_checkpoint
from knee_baseline.models import StudyHead, weighted_bce
from knee_baseline.train import load_checkpoint


@pytest.mark.parametrize("depth", [0, 2])
def test_head_mask_and_permutation(depth):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = StudyHead(8, hidden_dim=16, dropout=.1, depth=depth, heads=4).eval()
    features = torch.randn(2, 6, 3, 8)
    mask = torch.ones(2, 6, 3, dtype=torch.bool)
    mask[:, 4:, :] = False
    with torch.no_grad():
        reference = model(features, mask)
        changed = features.clone()
        changed[~mask] = 1e8
        torch.testing.assert_close(reference, model(changed, mask), atol=2e-5, rtol=2e-5)
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(reference, model(features[:, :, permutation], mask[:, :, permutation]),
                                   atol=2e-5, rtol=2e-5)
    assert reference.shape == (2, 12)
    with pytest.raises(ValueError):
        model(features, torch.zeros_like(mask))


def test_encoder_really_frozen():
    model = freeze_encoder(nn.Sequential(nn.Linear(3, 8), nn.Dropout(.5)))
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_confidence_weighted_loss_and_nan():
    logits = torch.tensor([[1., -2.], [-1., 2.]])
    target = torch.tensor([[1., float("nan")], [0., 1.]])
    confidence = torch.tensor([[1., 1.], [.5, .7]])
    loss = weighted_bce(logits, target, confidence)
    first = (torch.nn.functional.softplus(-logits[0, 0]) + .5 * torch.nn.functional.softplus(logits[1, 0])) / 1.5
    second = torch.nn.functional.softplus(-logits[1, 1])
    torch.testing.assert_close(loss, (first + second) / 2)
    with pytest.raises(ValueError):
        weighted_bce(logits, target, torch.zeros_like(confidence))


def test_cache_resume_and_corruption(tmp_path):
    store = FeatureStore(tmp_path, "identity", feature_dim=4, samples_per_slot=2, create=True)
    x = np.zeros((6, 2, 4), np.float16)
    mask = np.zeros((6, 2), bool)
    mask[0, 0] = True
    store.put("first", x, mask)
    store.flush()
    resumed = FeatureStore(tmp_path, "identity", 4, 2)
    assert resumed.has("first") and not resumed.has("second")
    restored, _ = resumed.get("first")
    np.testing.assert_array_equal(restored, x)
    with pytest.raises(ValueError, match="incompatible"):
        FeatureStore(tmp_path, "different", 4, 2)
    filename = resumed.manifest["studies"]["first"]["file"]
    (tmp_path / filename).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrompu"):
        resumed.get("first")


def test_unpublished_shard_can_be_recomputed(tmp_path):
    store = FeatureStore(tmp_path, "identity", 4, 2, create=True)
    mask = np.ones((6, 2), bool)
    store.put("orphan", np.zeros((6, 2, 4)), mask)  # manifeste non publié : arrêt simulé
    resumed = FeatureStore(tmp_path, "identity", 4, 2)
    assert not resumed.has("orphan")
    resumed.put("orphan", np.ones((6, 2, 4)), mask)
    resumed.flush()
    assert FeatureStore(tmp_path, "identity", 4, 2).has("orphan")


def test_checkpoint_identity_without_training(tmp_path):
    path = tmp_path / "last.pt"
    # Aucun optimizer.step : seulement la sérialisation d'un petit état artificiel.
    write_checkpoint(path, {"identity": "fold0", "epoch": 3, "torch_rng": torch.get_rng_state(),
                            "head": nn.Linear(2, 1).state_dict(), "history": []})
    assert load_checkpoint(path, "fold0")["epoch"] == 3
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path, "fold1")
