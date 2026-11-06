"""Encodeurs gelés et moyenne des cinq têtes finales, sans apprentissage."""

import gc
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .constants import LABELS, READERS, UID
from .io import read_json
from .models import StudyHead


def json_compatible(value):
    return json.loads(json.dumps(value))


def load_encoder(bundle, reader, device):
    import timm
    from timm.data import create_transform, resolve_model_data_config
    if version("timm") != "1.0.29":
        raise RuntimeError("timm 1.0.29 requis ; redémarrer la session et lancer l'installation.")
    directory = Path(bundle) / "assets" / reader
    metadata = read_json(directory / "manifest.json")
    source = read_json(directory / "encoder.json")
    spec = READERS[reader]
    if metadata["model"] != spec["model"] or source["asset"] != metadata:
        raise ValueError("Poids/architecture incompatibles avec l'expérience entraînée.")
    kwargs = {"img_size": spec["size"]} if reader == "dino" else {}
    model = timm.create_model(
        spec["model"], pretrained=True, num_classes=0, pretrained_cfg=metadata["pretrained_cfg"],
        pretrained_cfg_overlay={"file": str(directory / metadata["weight_file"]),
                                "hf_hub_id": None, "url": ""}, **kwargs)
    model.eval().requires_grad_(False)
    transform_cfg = resolve_model_data_config(model)
    transform_cfg["input_size"] = (3, spec["size"], spec["size"])
    if (json_compatible(transform_cfg) != source["transform"]
            or model.num_features != source["feature_dim"]):
        raise ValueError("La transformation ou la dimension diffère de celle des têtes entraînées.")
    transform = create_transform(**transform_cfg, is_training=False)
    return model.to(device), transform


def encode_pixels(model, transform, pixels, batch_size):
    """[N,224,224] uint8 → [N,D] float32 ; l'entrée est répliquée en RGB."""
    device = next(model.parameters()).device
    values = []
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            inputs = torch.stack([transform(Image.fromarray(x).convert("RGB"))
                                  for x in pixels[start:start + batch_size]]).to(device)
            retry = False
            try:
                output = model(inputs).float()
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                retry = True
            if retry:
                del inputs
                torch.cuda.empty_cache()
                return encode_pixels(model, transform, pixels, max(1, batch_size // 2))
            if output.ndim != 2 or not torch.isfinite(output).all():
                raise ValueError("Caractéristiques d'encodeur invalides.")
            values.append(output.cpu().numpy())
    return np.concatenate(values)


def feature_tensor(encoded, slots, tokens, feature_dim):
    """Reproduit le cache float16 utilisé pour entraîner les têtes.

    Le retour est [6,8,D] float16 et [6,8] bool. Un slot absent reste masqué ;
    aucune image grise ni duplication ne lui est attribuée.
    """
    if encoded.shape != (len(slots), feature_dim) or len(tokens) != len(slots):
        raise ValueError("Alignement des images et caractéristiques incorrect.")
    pairs = list(zip(slots.tolist(), tokens.tolist()))
    if (not pairs or len(set(pairs)) != len(pairs) or any(
            not (0 <= int(s) < 6 and 0 <= int(t) < 8) for s, t in pairs)):
        raise ValueError("Indices de slots/tokens invalides ou dupliqués.")
    values = np.zeros((6, 8, feature_dim), np.float16)
    mask = np.zeros((6, 8), bool)
    values[slots, tokens] = encoded
    mask[slots, tokens] = True
    if not np.isfinite(values).all():
        raise ValueError("Caractéristiques non finies après conversion float16.")
    return values, mask


def load_heads(bundle, manifest, reader, device):
    from safetensors.torch import load_file
    records = manifest["heads"][reader]
    if len(records) != 5 or [r["fold"] for r in records] != list(range(5)):
        raise ValueError("Cinq folds exactement sont nécessaires par lecteur.")
    heads = []
    for record in records:
        if record["epoch"] != 20:
            raise ValueError("La soumission doit utiliser les checkpoints finaux à vingt époques.")
        model = StudyHead(**record["spec"])
        model.load_state_dict(load_file(str(Path(bundle) / record["file"])), strict=True)
        heads.append(model.to(device).eval().requires_grad_(False))
    return heads


def predict_heads(heads, features, masks, batch_size=32):
    """Moyenne des probabilités des folds AVANT les rangs du mélange final."""
    device = next(heads[0].parameters()).device
    predictions = []
    with torch.inference_mode():
        for model in heads:
            blocks = []
            for start in range(0, len(features), batch_size):
                x = torch.from_numpy(features[start:start + batch_size]).float().to(device)
                mask = torch.from_numpy(masks[start:start + batch_size]).to(device)
                blocks.append(model(x, mask).sigmoid().cpu().numpy())
            predictions.append(np.concatenate(blocks))
    result = np.mean(predictions, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Prédictions non finies.")
    return result


def predict_reader(bundle, manifest, reader, ids, caches, device):
    started = time.perf_counter()
    model, transform = load_encoder(bundle, reader, device)
    dim = model.num_features
    # Un seul lecteur à la fois en RAM ; environ 128 Mo float16 pour 1 300 CoAtNet.
    features = np.zeros((len(ids), 6, 8, dim), np.float16)
    masks = np.zeros((len(ids), 6, 8), bool)
    batch_size = 16 if reader == "coatnet" else 32
    try:
        for i, path in enumerate(caches):
            with np.load(path, allow_pickle=False) as arrays:
                encoded = encode_pixels(model, transform, arrays["pixels"], batch_size)
                features[i], masks[i] = feature_tensor(encoded, arrays["slots"], arrays["tokens"], dim)
            if (i + 1) % 100 == 0 or i == 0 or i + 1 == len(ids):
                print(f"{reader} : {i + 1}/{len(ids)} études", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    heads = load_heads(bundle, manifest, reader, device)
    values = predict_heads(heads, features, masks)
    del heads, features, masks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    frame = pd.DataFrame(values, columns=LABELS)
    frame.insert(0, UID, list(ids))
    return frame, {"seconds": round(time.perf_counter() - started, 3), "studies": len(ids)}
