from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .cache import Cache
from .common import UID, require, seed_for


class StudyDataset(Dataset):
    def __init__(self, table, cache, cfg, fold, epoch=0, training=False, fixture=False):
        self.table = table.reset_index(drop=True)
        self.cache = Cache(cache, allow_fixture=fixture)
        self.cfg, self.fold, self.epoch, self.training = cfg, fold, epoch, training
        require(set(self.table[UID]).issubset(self.cache.entries), "Cache sans certaines études demandées")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        uid = row[UID]
        rng = np.random.default_rng(seed_for(42, self.fold, self.epoch, uid, 0))
        arrays = self.cache.read(uid)
        chosen = []
        for slot in range(6):
            candidates = np.flatnonzero(arrays["slots"] == slot)
            if self.training and len(candidates) > self.cfg["input"]["train_windows_per_slot"]:
                candidates = np.sort(rng.choice(candidates, self.cfg["input"]["train_windows_per_slot"], replace=False))
            chosen.extend(candidates.tolist())
        require(bool(chosen), f"Aucune fenêtre : {uid}")
        images = torch.from_numpy(arrays["images"][chosen]).float().div_(255)
        slots = torch.from_numpy(arrays["slots"][chosen])
        positions = torch.from_numpy(arrays["positions"][chosen])
        size = self.cfg["input"]["input_size"]
        if images.shape[-1] != size:
            images = F.interpolate(images, (size, size), mode="bicubic", align_corners=False, antialias=True)
        images = images.clamp_(0, 1)
        if self.training:
            # One transform per series/slot, shared by all centres and all slices.
            for slot in slots.unique():
                indices = slots == slot
                angle = math.radians(rng.uniform(-5, 5))
                scale = rng.uniform(.96, 1.04)
                tx, ty = rng.uniform(-.03, .03, 2)
                theta = images.new_tensor([[math.cos(angle) / scale, -math.sin(angle) / scale, tx],
                                           [math.sin(angle) / scale, math.cos(angle) / scale, ty]])
                x = images[indices]
                grid = F.affine_grid(theta[None].expand(len(x), -1, -1), x.shape, align_corners=False)
                x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                images[indices] = (x.clamp(0, 1).pow(float(rng.uniform(.9, 1.1))) * float(rng.uniform(.9, 1.1))).clamp(0, 1)
        mean = images.new_tensor(self.cfg["input"]["normalization_mean"])[None, :, None, None]
        std = images.new_tensor(self.cfg["input"]["normalization_std"])[None, :, None, None]
        images = (images - mean) / std
        return {"images": images, "valid": torch.ones(len(images), dtype=torch.bool),
                "slots": slots, "positions": positions, "uid": uid}


def as_batch(sample, device):
    return {key: sample[key].unsqueeze(0).to(device, non_blocking=True)
            for key in ("images", "valid", "slots", "positions")}
