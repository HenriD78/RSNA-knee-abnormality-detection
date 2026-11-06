"""Encodeurs gelés hors ligne et caractéristiques [6 slots, K images, D].

Les poids sont fournis par le paquet ; ce module ne les télécharge jamais.
Le masque booléen distingue chaque image présente des emplacements absents.
"""

import platform
import shutil
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .config import ROOT, configure_environment, data_root, output_root, assets_root, deadline_reached
from .constants import READERS, UID
from .data import load_prepared
from .io import fingerprint, read_json, sha256, write_json, write_npz


def asset_dir(reader):
    if reader not in READERS:
        raise ValueError(f"Lecteur inconnu : {reader}")
    return assets_root() / reader


def asset_manifest(reader):
    directory = asset_dir(reader)
    path = directory / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Poids absents pour {reader}. Relancer la cellule d'installation du paquet.")
    manifest = read_json(path)
    if manifest["model"] != READERS[reader]["model"]:
        raise ValueError("Le modèle sauvegardé ne correspond pas au lecteur.")
    if manifest["timm_version"] != version("timm"):
        raise ValueError("La version timm a changé : vérifier les poids dans un environnement verrouillé.")
    if sha256(directory / manifest["weight_file"]) != manifest["weight_sha256"]:
        raise ValueError(f"Empreinte du poids incorrecte : {reader}.")
    return manifest


def download_assets(readers):
    raise RuntimeError("Les poids sont inclus dans le paquet ; aucun téléchargement cloud nécessaire.")


def freeze_encoder(model):
    """eval() désactive dropout ; requires_grad=False interdit l'apprentissage du tronc."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_encoder(reader, cfg):
    configure_environment()
    import timm
    from timm.data import create_transform, resolve_model_data_config
    manifest = asset_manifest(reader)
    specification = READERS[reader]
    kwargs = {"img_size": specification["size"]} if reader == "dino" else {}
    # Le fichier local a priorité ; aucune résolution Hugging Face n'est utilisée.
    model = timm.create_model(
        specification["model"], pretrained=True, num_classes=0,
        pretrained_cfg=manifest["pretrained_cfg"],
        pretrained_cfg_overlay={"file": str(asset_dir(reader) / manifest["weight_file"]),
                                "hf_hub_id": None, "url": ""}, **kwargs,
    )
    freeze_encoder(model)
    transform_cfg = resolve_model_data_config(model)
    transform_cfg["input_size"] = (3, specification["size"], specification["size"])
    transform = create_transform(**transform_cfg, is_training=False)
    torch.set_num_threads(cfg["cpu_threads"])
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU absent : activer Accelerator > GPU. Aucun repli CPU implicite.")
    model.to(device)
    if device.type == "cuda":
        # Calcul float32 pour rester proche des caractéristiques CPU déjà calculées.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    details = {"asset": manifest, "transform": transform_cfg,
               "feature_dim": model.num_features,
               "torch_version": torch.__version__, "platform": platform.platform()}
    return model, transform, details


class FeatureStore:
    """Un fichier atomique par étude : reprise sans recommencer les études terminées."""

    def __init__(self, directory, identity, feature_dim, samples_per_slot, create=False):
        self.directory = Path(directory)
        self.identity = identity
        self.feature_dim = int(feature_dim)
        self.samples = int(samples_per_slot)
        self.path = self.directory / "manifest.json"
        if self.path.exists():
            self.manifest = read_json(self.path)
            if (self.manifest["identity"] != identity or self.manifest["feature_dim"] != self.feature_dim
                    or self.manifest["samples_per_slot"] != self.samples):
                raise ValueError("Cache incompatible : données, poids, transformations ou configuration modifiés.")
        elif create:
            self.manifest = {"identity": identity, "feature_dim": self.feature_dim,
                             "samples_per_slot": self.samples, "studies": {}}
            self.flush()
        else:
            raise FileNotFoundError(f"Cache absent : {directory}. Lancer extract.")

    def flush(self):
        write_json(self.path, self.manifest)

    def has(self, uid):
        record = self.manifest["studies"].get(uid)
        if record is None:
            return False
        if sha256(self.directory / record["file"]) != record["sha256"]:
            raise ValueError(f"Cache corrompu pour {uid}.")
        return True

    def put(self, uid, features, mask):
        features = np.asarray(features, dtype=np.float16)
        mask = np.asarray(mask, dtype=bool)
        if (features.shape != (6, self.samples, self.feature_dim)
                or mask.shape != (6, self.samples) or not mask.any() or not np.isfinite(features).all()):
            raise ValueError("Formes, masque ou valeurs des caractéristiques invalides.")
        filename = fingerprint(uid) + ".npz"
        write_npz(self.directory / filename, features=features, mask=mask)
        self.manifest["studies"][uid] = {"file": filename, "sha256": sha256(self.directory / filename)}

    def get(self, uid):
        if not self.has(uid):
            raise FileNotFoundError(f"Caractéristiques absentes pour {uid} : reprendre extract.")
        record = self.manifest["studies"][uid]
        with np.load(self.directory / record["file"], allow_pickle=False) as content:
            features, mask = content["features"].astype(np.float32), content["mask"].astype(bool)
        if (features.shape != (6, self.samples, self.feature_dim)
                or mask.shape != (6, self.samples) or not mask.any() or not np.isfinite(features).all()):
            raise ValueError("Cache invalide après lecture.")
        return features, mask


def cache_identity(prepared, reader, details):
    return fingerprint({"prepared": prepared["signature"], "reader": reader,
                        "model": READERS[reader], "details": logical_details(details)})


def open_store(cfg, reader, split):
    prepared, _ = load_prepared(cfg)
    path = output_root(cfg) / "features" / reader / split
    metadata = read_json(path / "manifest.json")
    details = read_json(output_root(cfg) / "features" / reader / "encoder.json")
    # La provenance du cache doit toujours correspondre aux poids locaux vérifiés.
    if details["asset"] != asset_manifest(reader):
        raise ValueError("Les poids du lecteur ont changé depuis l'extraction.")
    identity = cache_identity(prepared, reader, details)
    return FeatureStore(path, identity, metadata["feature_dim"], cfg["samples_per_slot"])


def logical_details(details):
    """Identité indépendante du chemin, du système et du périphérique CPU/CUDA.

    Les versions de production restent consignées. La réutilisation des features
    repose sur les poids, transformations et images figés, pas sur les dates Unix.
    """
    return {"asset": details["asset"], "transform": details["transform"],
            "feature_dim": details["feature_dim"]}


def encode_images(model, transform, paths, batch_size):
    parts = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    batch.append(transform(image.convert("RGB")))
            inputs = torch.stack(batch).to(device)
            retry = False
            try:
                result = model(inputs).float()
            except torch.cuda.OutOfMemoryError:
                # Évaluer de plus petits lots ne modifie pas les poids ou le split.
                if batch_size == 1:
                    raise
                retry = True
            if retry:
                # Sortir du bloc except libère les références de la traceback CUDA.
                del inputs
                torch.cuda.empty_cache()
                return encode_images(model, transform, paths, max(1, batch_size // 2))
            if result.ndim != 2 or not torch.isfinite(result).all():
                raise ValueError("L'encodeur doit produire [N,D] fini.")
            parts.append(result.cpu().numpy())
    return np.concatenate(parts)


def import_seeds(cfg, reader, split, details, store):
    """Conserve les études CPU terminées, après vérification de leur provenance."""
    seed = Path(os.environ.get("RSNA_SEED_ROOT", ROOT / "seed"))
    if split != "train" or not (seed / "manifest.json").exists():
        return
    manifest = read_json(seed / "manifest.json")
    prepared, _ = load_prepared(cfg)
    if manifest["prepared"] != prepared["signature"]:
        raise ValueError("Le cache initial appartient à d'autres images/folds.")
    entry = manifest["readers"].get(reader)
    if entry is None:
        return
    if logical_details(entry["encoder"]) != logical_details(details):
        raise ValueError("Transformations ou poids différents du cache CPU initial.")
    added = 0
    for uid, record in entry["studies"].items():
        if store.has(uid):
            continue
        path = seed / reader / "train" / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Cache initial corrompu.")
        with np.load(path, allow_pickle=False) as arrays:
            store.put(uid, arrays["features"], arrays["mask"])
        added += 1
    store.flush()
    print(f"{reader}: {added} études locales importées, sans recalcul GPU.", flush=True)


def extract(cfg, readers, split="train"):
    prepared, tables = load_prepared(cfg, check_images=True)
    rows = tables["selected_images"]
    rows = rows[rows.split == split]
    for reader in readers:
        if deadline_reached():
            return False
        model, transform, details = load_encoder(reader, cfg)
        details = read_json_compatible(details)
        directory = output_root(cfg) / "features" / reader
        if (directory / "encoder.json").exists():
            if logical_details(read_json(directory / "encoder.json")) != logical_details(details):
                raise ValueError("Provenance d'encodeur différente : reprise refusée.")
        else:
            write_json(directory / "encoder.json", details)
        store = FeatureStore(directory / split, cache_identity(prepared, reader, details),
                             model.num_features, cfg["samples_per_slot"], create=True)
        import_seeds(cfg, reader, split, details, store)
        batch_size = cfg["coatnet_batch_size"] if reader == "coatnet" else cfg["extraction_batch_size"]
        try:
            for uid, study in tqdm(rows.groupby(UID, sort=True), desc=f"GPU {reader}/{split}"):
                if deadline_reached():
                    return False
                if store.has(uid):
                    continue
                study = study.sort_values(["slot", "token"])
                encoded = encode_images(model, transform, [data_root(cfg) / p for p in study.path], batch_size)
                features = np.zeros((6, cfg["samples_per_slot"], model.num_features), np.float16)
                mask = np.zeros((6, cfg["samples_per_slot"]), bool)
                for feature, row in zip(encoded, study.itertuples(index=False)):
                    features[row.slot, row.token], mask[row.slot, row.token] = feature, True
                store.put(uid, features, mask)
                store.flush()  # Chaque étude terminée est immédiatement reprenable.
                from .session import maybe_snapshot
                maybe_snapshot(cfg)
        finally:
            store.flush()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"Cache complet : {reader}/{split}", flush=True)
    return True


def read_json_compatible(value):
    """Normalise tuples et chaînes de versions avant la comparaison des manifestes."""
    import json
    return json.loads(json.dumps(value))


def benchmark(cfg, readers, limit=16):
    """Chronomètre un PETIT lot explicite ; aucune extraction complète n'est lancée."""
    _, tables = load_prepared(cfg, check_images=True)
    rows = tables["selected_images"]
    train = rows[rows.split == "train"].sort_values([UID, "slot", "token"])
    paths = [data_root(cfg) / p for p in train.path.iloc[:limit]]
    if not paths:
        raise ValueError("Préparer des images avant le benchmark.")
    receipts = []
    for reader in readers:
        model, transform, _ = load_encoder(reader, cfg)
        bs = cfg["coatnet_batch_size"] if reader == "coatnet" else cfg["extraction_batch_size"]
        encode_images(model, transform, paths[:min(bs, len(paths))], bs)  # chauffe CUDA
        start = time.perf_counter()
        encode_images(model, transform, paths, bs)
        seconds = time.perf_counter() - start
        receipts.append({"reader": reader, "images": len(paths), "seconds": seconds,
                         "images_per_second": len(paths) / seconds,
                         "estimated_train_hours": seconds / len(paths) * len(train) / 3600})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output_root(cfg) / "reports/benchmark.json", receipts)
    print("Estimation indicative ; caches disque, température et taille des lots peuvent la modifier.")
    for receipt in receipts:
        print(receipt)
