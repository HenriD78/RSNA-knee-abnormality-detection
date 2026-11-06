from __future__ import annotations

import timm
import torch
from safetensors.torch import load_file
from timm.layers import resample_abs_pos_embed
from torch import nn
from torch.nn import functional as F

from .common import LABELS, require, sha256


class StudyHead(nn.Module):
    def __init__(self, feature_dim, cfg):
        super().__init__()
        d = cfg["projection_dim"]
        self.projection = nn.Sequential(nn.Linear(feature_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(cfg["dropout"]))
        self.slot_embedding = nn.Embedding(6, d)
        self.position_embedding = nn.Sequential(nn.Linear(1, d), nn.Tanh(), nn.Linear(d, d))
        if cfg["transformer_layers"]:
            layer = nn.TransformerEncoderLayer(d, cfg["transformer_heads"], 2 * d,
                                               cfg["dropout"], batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, cfg["transformer_layers"], enable_nested_tensor=False)
        else:
            self.transformer = None
        self.attention = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, len(LABELS)))
        self.classifier = nn.Parameter(torch.randn(len(LABELS), d) * .02)
        self.bias = nn.Parameter(torch.zeros(len(LABELS)))

    def forward(self, features, valid, slots, positions):
        require(features.shape[:-1] == valid.shape == slots.shape == positions.shape, "Tenseurs de tokens désalignés")
        features = features.flatten(1, -2) if features.ndim > 3 else features
        valid, slots, positions = (t.flatten(1) for t in (valid, slots, positions))
        require(valid.any(1).all().item(), "Étude entièrement masquée")
        require(torch.isfinite(positions[valid]).all().item(), "Position physique invalide")
        x = self.projection(features.masked_fill(~valid[..., None], 0))
        x = x + self.slot_embedding(slots) + self.position_embedding(positions.masked_fill(~valid, 0)[..., None])
        if self.transformer is not None:
            x = self.transformer(x, src_key_padding_mask=~valid)
        attention = self.attention(x).float().masked_fill(~valid[..., None], -torch.inf).softmax(1)
        pooled = torch.einsum("btc,btd->bcd", attention, x.float())
        return (pooled * self.classifier.float()).sum(-1) + self.bias.float()


class StudyModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg
        is_dino = cfg["model_id"] == "dinov2_small"
        kwargs = {"img_size": cfg["input"]["input_size"]} if is_dino else {}
        self.backbone = timm.create_model(cfg["initialization"]["timm_model"], pretrained=False, **kwargs)
        if weight is not None:
            require(sha256(weight) == cfg["initialization"]["weight_sha256"], "Poids génériques incompatibles")
            state = load_file(str(weight))
            if is_dino:
                # Generic weights contain the native 37x37 grid. Preserve CLS;
                # interpolate spatial embeddings once to the 16x16 training grid.
                state["pos_embed"] = resample_abs_pos_embed(state["pos_embed"], self.backbone.patch_embed.grid_size,
                                                           num_prefix_tokens=self.backbone.num_prefix_tokens)
            self.backbone.load_state_dict(state, strict=True)
        self.backbone.reset_classifier(0)
        expected = {"coatnet2": 1024, "convnext_small": 768, "dinov2_small": 384}[cfg["model_id"]]
        require(self.backbone.num_features == expected, "Dimension d'encodeur inattendue")
        if is_dino:
            require(tuple(self.backbone.patch_embed.grid_size) == (16, 16), "Grille DINOv2 non 16x16")
        self.head = StudyHead(expected, cfg["head"])
        self.adapt = False
        self.set_phase(False)

    def tail_modules(self):
        if self.cfg["model_id"] == "dinov2_small":
            return [*list(self.backbone.blocks)[-2:], self.backbone.norm]
        modules = [self.backbone.stages[-1]]
        for name in ("norm", "norm_pre"):
            if hasattr(self.backbone, name):
                modules.append(getattr(self.backbone, name))
        if hasattr(self.backbone.head, "norm"):
            modules.append(self.backbone.head.norm)
        return modules

    def set_phase(self, adapt):
        self.adapt = bool(adapt)
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(True)
        if adapt:
            for module in self.tail_modules():
                module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.adapt:
            for module in self.tail_modules():
                module.train()
        # All BN statistics stay fixed; affine terms in the tail can still adapt.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images, valid, slots, positions):
        # Accept [B,T,3,H,W] or the documented [B,6,K,3,H,W].
        require(images.shape[:-3] == valid.shape, "Images et masque désalignés")
        b = images.shape[0]
        flat_valid = valid.reshape(-1)
        require(valid.reshape(b, -1).any(1).all().item(), "Étude sans fenêtre valide")
        selected = images.reshape(-1, *images.shape[-3:])[flat_valid]
        encoded = self.backbone.forward_head(self.backbone.forward_features(selected), pre_logits=True)
        require(encoded.ndim == 2 and encoded.shape[-1] == self.backbone.num_features, "Sortie encodeur invalide")
        features = encoded.new_zeros(flat_valid.numel(), encoded.shape[-1])
        features[flat_valid] = encoded
        return self.head(features.reshape(b, -1, encoded.shape[-1]), valid.reshape(b, -1),
                         slots.reshape(b, -1), positions.reshape(b, -1))


def denominators(targets, confidence):
    require(torch.isfinite(confidence).all().item() and ((confidence >= 0) & (confidence <= 1)).all().item(),
            "Confiance invalide")
    weights = confidence.masked_fill(~torch.isfinite(targets), 0.)
    denom = weights.sum(0)
    require((denom > 0).any().item(), "Batch effectif sans cible pondérée")
    return denom


def bce_contribution(logits, targets, confidence, denom):
    weights = confidence.masked_fill(~torch.isfinite(targets), 0.)
    loss = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(targets, nan=0.), reduction="none")
    active = denom > 0
    return ((loss * weights).sum(0)[active] / denom[active]).sum() / active.sum()


def optimizer_for(model, cfg):
    # Include the future tail from the start, preserving head Adam moments at unfreeze.
    model.set_phase(True)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = cfg["training"]["backbone_learning_rate" if name.startswith("backbone.") else "head_learning_rate"]
        decay = 0. if parameter.ndim <= 1 or name.endswith(".bias") else cfg["training"]["weight_decay"]
        groups.setdefault((lr, decay), []).append(parameter)
    result = torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": decay}
                               for (lr, decay), params in groups.items()])
    model.set_phase(False)
    return result
