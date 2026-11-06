"""Têtes entraînables sur des caractéristiques gelées, indépendantes de l'ordre."""

import torch
from torch import nn
from torch.nn import functional as F

from .constants import LABELS


class StudyHead(nn.Module):
    """[B,6,K,D] + masque [B,6,K] → douze logits par étude.

    Chaque finding apprend les images qu'il doit regarder. Les embeddings
    identifient les slots, pas une position anatomique inventée à partir d'un UID.
    """

    def __init__(self, feature_dim, hidden_dim=128, dropout=.1, depth=0, heads=4):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                        nn.GELU(), nn.Dropout(dropout))
        self.slots = nn.Embedding(6, hidden_dim)
        if depth:
            layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 2,
                                               dropout=dropout, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        else:
            self.transformer = None
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, len(LABELS)))
        self.classifier_weight = nn.Parameter(torch.randn(len(LABELS), hidden_dim) * .02)
        self.classifier_bias = nn.Parameter(torch.zeros(len(LABELS)))

    def forward(self, features, mask):
        if features.ndim != 4 or features.shape[1] != 6 or mask.shape != features.shape[:3]:
            raise ValueError("Entrées attendues : features [B,6,K,D], mask [B,6,K].")
        mask = mask.bool()
        if not mask.flatten(1).any(1).all():
            raise ValueError("Une étude ne contient aucune image valide.")
        b, s, k, _ = features.shape
        slot_ids = torch.arange(s, device=features.device).view(1, s, 1)
        # Les valeurs d'un token masqué ne doivent influencer aucune prédiction.
        features = features.masked_fill(~mask.unsqueeze(-1), 0.)
        tokens = self.projection(features) + self.slots(slot_ids)
        tokens = tokens.reshape(b, s * k, -1)
        valid = mask.reshape(b, s * k)
        if self.transformer is not None:
            tokens = self.transformer(tokens, src_key_padding_mask=~valid)
        weights = self.attention(tokens).masked_fill(~valid.unsqueeze(-1), -torch.inf).softmax(dim=1)
        pooled = torch.einsum("btc,btd->bcd", weights, tokens)
        return (pooled * self.classifier_weight).sum(-1) + self.classifier_bias


def weighted_bce(logits, targets, confidence):
    """BCE par finding, normalisée par la confiance disponible dans le lot.

    Une cible NaN a un poids nul ; son remplacement numérique sert uniquement
    à éviter NaN dans BCE, et ne crée jamais une annotation négative.
    """
    known = torch.isfinite(targets)
    weight = confidence.masked_fill(~known, 0.)
    if not torch.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
        raise ValueError("Confiances invalides.")
    loss = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(targets, nan=0.), reduction="none")
    denominator = weight.sum(0)
    active = denominator > 0
    if not active.any():
        raise ValueError("Ce lot ne contient aucune cible de confiance positive.")
    return ((loss * weight).sum(0)[active] / denominator[active]).mean()


def head_spec(cfg, reader, feature_dim):
    return {"feature_dim": int(feature_dim), "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"], "depth": cfg["tiny_depth"] if reader == "tiny" else 0,
            "heads": cfg["transformer_heads"]}
