"""Lit les folds préparés localement : aucun nouveau split ni scan des 819k JPEG."""

import pandas as pd

from .config import data_root
from .constants import UID, SERIES
from .io import fingerprint, read_json, sha256


def load_prepared(cfg, check_images=False):
    root = data_root(cfg)
    directory = root / "prepared"
    manifest = read_json(directory / "manifest.json")
    if manifest["format"] != "rsna-portable-v1":
        raise ValueError("Format de préparation portable inconnu.")
    if fingerprint({k: v for k, v in manifest.items() if k != "signature"}) != manifest["signature"]:
        raise ValueError("Manifeste portable altéré.")
    if manifest["split_contract"] != {k: cfg[k] for k in ["seed", "n_folds", "samples_per_slot"]}:
        raise ValueError("Le split ou l'échantillonnage ne correspond pas au paquet.")
    tables = {}
    for name, digest in manifest["table_hashes"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Table portable modifiée : {name}")
        tables[name[:-4]] = pd.read_csv(directory / name, dtype={UID: str, SERIES: str})
    master = tables["train_folds"]
    if master[UID].duplicated().any() or master.groupby("group_id").fold.nunique().max() != 1:
        raise ValueError("Folds invalides : UID dupliqué ou groupe partagé.")
    if check_images:
        # Les archives sont vérifiées en SHA-256 à l'installation. Ici on contrôle
        # l'existence/taille ; les dates locales n'interviennent plus dans l'identité.
        for row in tables["selected_images"].itertuples(index=False):
            if (root / row.path).stat().st_size != row.size_bytes:
                raise ValueError(f"Image absente ou taille modifiée : {row.path}")
    return manifest, tables
