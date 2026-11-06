"""Labels, groupes de rapports et index d'images : aucune modification de data/."""

import hashlib
import re
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image
from tqdm.auto import tqdm

from .config import data_root, output_root
from .constants import LABELS, PLANES, SERIES, UID
from .io import fingerprint, read_json, sha256, write_csv, write_json

SOURCE_FILES = ["train.csv", "train_series.csv", "test.csv", "test_series.csv",
                "sample_submission.csv", "derived/report_labels_v2.csv"]


def normalize_report(text):
    """Deux rapports identiques après normalisation restent dans le même fold."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def read_table(path, required):
    frame = pd.read_csv(path, dtype={UID: str, SERIES: str})
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes absentes dans {path}: {missing}")
    if frame[UID].isna().any():
        raise ValueError(f"UID absent dans {path}")
    return frame


def build_folds(train, weak, seed=42, n_folds=5):
    """Stratifie les GROUPES, puis fournit une ligne par étude, jamais par coupe.

    Les zéros de la matrice d'indicateurs servent seulement à la stratification.
    Les annotations gold inconnues restent NaN dans les données d'origine.
    """
    if train[UID].duplicated().any() or weak[UID].duplicated().any():
        raise ValueError("Une étude apparaît plusieurs fois dans une table de labels.")
    if set(train[UID]) != set(weak[UID]):
        raise ValueError("Les labels faibles doivent couvrir exactement les études train.")
    train = train.sort_values(UID).reset_index(drop=True)
    weak = weak.set_index(UID).loc[train[UID]].reset_index()
    gold = train[LABELS].apply(pd.to_numeric, errors="raise")
    known = gold.notna().sum(axis=1)
    if not known.isin([0, len(LABELS)]).all():
        raise ValueError("Annotations gold partielles : politique non prise en charge.")
    if not (gold.isna() | gold.isin([0, 1])).all().all():
        raise ValueError("Les labels gold doivent être 0, 1 ou manquants.")
    soft = weak[LABELS].apply(pd.to_numeric, errors="raise")
    confidence = weak[[c + "__conf" for c in LABELS]].apply(pd.to_numeric, errors="raise")
    confidence.columns = LABELS
    for name, values in [("scores", soft), ("confiances", confidence)]:
        valid = values.isna() | ((values >= 0) & (values <= 1))
        if not valid.all().all():
            raise ValueError(f"{name} faibles hors [0, 1].")
    if (soft.notna() & confidence.isna()).any().any():
        raise ValueError("Une annotation faible connue doit avoir une confiance.")
    is_gold = known == len(LABELS)
    text = train["Report"].map(normalize_report)
    groups = pd.Series(["report:" + t if t else "study:" + u
                        for t, u in zip(text, train[UID])])
    indicators = pd.DataFrame(np.concatenate([
        gold.fillna(0).to_numpy(),
        (soft.fillna(-1).to_numpy() > .5) * (~is_gold.to_numpy())[:, None],
        is_gold.to_numpy()[:, None],
    ], axis=1).astype(np.int8))
    indicators["group"] = groups
    grouped = indicators.groupby("group", sort=True).max()
    if len(grouped) < n_folds:
        raise ValueError("Nombre de groupes insuffisant pour cinq folds.")
    mapping = {}
    splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val) in enumerate(splitter.split(np.zeros((len(grouped), 1)), grouped.values)):
        mapping.update({g: fold for g in grouped.index[val]})
    master = train[[UID]].copy()
    master["group_id"] = groups.map(lambda g: hashlib.sha256(g.encode()).hexdigest())
    master["fold"] = groups.map(mapping).astype(int)
    master["is_gold"] = is_gold.to_numpy()
    if master.groupby("group_id")["fold"].nunique().max() != 1:
        raise AssertionError("Un groupe de rapports traverse plusieurs folds.")
    # Gold et weak ne sont jamais confondus dans les fichiers d'évaluation.
    targets = master.copy()
    for label in LABELS:
        targets[label] = np.where(is_gold, gold[label], soft[label])
        targets[label + "__conf"] = np.where(is_gold, 1., confidence[label])
        targets.loc[targets[label].isna(), label + "__conf"] = 0.
    return master, targets, train[[UID] + LABELS].copy()


def activate_studies(master, cfg):
    master = master.copy()
    master["active"] = cfg["profile"] == "full"
    if cfg["profile"] == "smoke":
        for fold in range(cfg["n_folds"]):
            for gold, key in [(True, "smoke_gold_per_fold"), (False, "smoke_weak_per_fold")]:
                candidates = master.loc[(master.fold == fold) & (master.is_gold == gold), UID]
                ids = sorted(candidates, key=lambda u: fingerprint([cfg["seed"], u]))[:cfg[key]]
                master.loc[master[UID].isin(ids), "active"] = True
    return master


def slot_of(plane, fat_suppression):
    """Slots 0..2 avec fat-sat, 3..5 sans fat-sat ; pas de fausse adjacency."""
    if plane not in PLANES or fat_suppression not in (0, 1):
        raise ValueError(f"Métadonnées de slot invalides : {plane}, {fat_suppression}")
    return PLANES.index(plane) + 3 * (1 - int(fat_suppression))


def choose_images(series_rows, root, split, active_ids, cfg):
    """Compte les JPEG lisibles de chaque série avant de choisir les six slots.

    Cette étape est potentiellement longue en mode full : elle lit les images.
    Elle n'appelle aucun réseau de neurones et ne copie jamais les JPEG.
    """
    selected, diagnostics, errors = [], [], []
    inventory = hashlib.sha256()
    rows = series_rows.loc[series_rows[UID].isin(active_ids)].sort_values([UID, SERIES])
    for uid, study_rows in tqdm(rows.groupby(UID, sort=True), desc=f"Index JPEG {split}"):
        slots = {}
        for _, row in study_rows.iterrows():
            slot = slot_of(row.Anatomical_Plane, row.Fat_Suppression)
            directory = root / f"{split}_series" / uid / row[SERIES]
            valid = []
            for path in sorted(directory.glob("*.jpeg")):
                stat = path.stat()
                relative = path.relative_to(root).as_posix()
                readable = True
                try:
                    with Image.open(path) as im:
                        im.load()
                        if im.mode != "L" or im.size != (224, 224):
                            raise ValueError(f"JPEG attendu : L 224x224 ; reçu {im.mode} {im.size}")
                except Exception as exc:
                    readable = False
                    errors.append({"split": split, UID: uid, "path": relative, "error": str(exc)})
                inventory.update(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}|{readable}\n".encode())
                if readable:
                    valid.append({"path": relative, "size_bytes": stat.st_size,
                                  "mtime_ns": stat.st_mtime_ns})
            diagnostics.append({"split": split, UID: uid, SERIES: row[SERIES],
                                "slot": slot, "readable_images": len(valid)})
            # Le départage par UID reste stable si les lignes du CSV changent d'ordre.
            candidate = (len(valid), row[SERIES], valid)
            old = slots.get(slot)
            if valid and (old is None or (-candidate[0], candidate[1]) < (-old[0], old[1])):
                slots[slot] = candidate
        if not slots:
            raise ValueError(f"Aucune image utilisable pour l'étude {uid}. Consultez les dossiers JPEG.")
        for slot, (_, series_id, images) in sorted(slots.items()):
            images = sorted(images, key=lambda r: fingerprint([cfg["seed"], uid, series_id, r["path"]]))
            for token, info in enumerate(images[:cfg["samples_per_slot"]]):
                selected.append({"split": split, UID: uid, SERIES: series_id,
                                 "slot": slot, "token": token, **info})
    return selected, diagnostics, errors, inventory.hexdigest()


def source_hashes(root):
    return {name: sha256(root / name) for name in SOURCE_FILES}


def fold_diagnostics(master, gold, cfg):
    rows = []
    joined = master.merge(gold, on=UID, validate="one_to_one")
    for fold in range(cfg["n_folds"]):
        sub = joined[joined.fold == fold]
        g = sub[sub.is_gold]
        rows.append({"fold": fold, "studies": len(sub), "groups": sub.group_id.nunique(),
                     "gold": len(g), "active": int(sub.active.sum()),
                     **{c + "__gold_positive": int(g[c].sum()) for c in LABELS}})
    return pd.DataFrame(rows)


def prepare(cfg):
    root, output = data_root(cfg), output_root(cfg)
    destination = output / "prepared"
    if (destination / "manifest.json").exists():
        load_prepared(cfg, check_images=True)
        print("Préparation existante compatible : réutilisation, sans réindexation.")
        return destination
    initial_hashes = source_hashes(root)
    train = read_table(root / "train.csv", [UID, "Report"] + LABELS)
    weak = read_table(root / "derived/report_labels_v2.csv", [UID] + LABELS + [c + "__conf" for c in LABELS])
    test = read_table(root / "test.csv", [UID])
    if test[UID].duplicated().any() or set(train[UID]) & set(test[UID]):
        raise ValueError("UID test dupliqués ou présents dans train.")
    schema = read_table(root / "sample_submission.csv", [UID] + LABELS)
    if list(schema.columns) != [UID] + LABELS or schema[UID].tolist() != test[UID].tolist():
        raise ValueError("Le sample_submission ne correspond pas à test.csv.")
    series = {s: read_table(root / f"{s}_series.csv", [UID, SERIES, "Anatomical_Plane", "Fat_Suppression"])
              for s in ["train", "test"]}
    for split, frame in series.items():
        expected = set(train[UID] if split == "train" else test[UID])
        if set(frame[UID]) != expected or frame[SERIES].duplicated().any():
            raise ValueError(f"Métadonnées de séries invalides : {split}.")
    master, targets, gold = build_folds(train, weak, cfg["seed"], cfg["n_folds"])
    master = activate_studies(master, cfg)
    active = {"train": set(master.loc[master.active, UID]), "test": set(test[UID])}
    images, counts, failures, inventories = [], [], [], {}
    for split in ["train", "test"]:
        a, b, c, inventories[split] = choose_images(series[split], root, split, active[split], cfg)
        images.extend(a); counts.extend(b); failures.extend(c)
    if source_hashes(root) != initial_hashes:
        raise ValueError("Un CSV source a changé pendant l'indexation : recommencer la préparation.")
    output.mkdir(parents=True, exist_ok=True)
    # Publier uniquement une préparation complète ; un arrêt ne laisse pas de faux succès.
    with tempfile.TemporaryDirectory(prefix=".prepare_", dir=output) as temp:
        staging = Path(temp)
        tables = {
            "train_folds.csv": master, "targets.csv": targets, "gold_labels.csv": gold,
            "selected_images.csv": pd.DataFrame(images), "series_counts.csv": pd.DataFrame(counts),
            "image_errors.csv": pd.DataFrame(failures, columns=["split", UID, "path", "error"]),
            "fold_diagnostics.csv": fold_diagnostics(master, gold, cfg), "test_ids.csv": test,
        }
        for name, frame in tables.items():
            write_csv(staging / name, frame)
        manifest = {"config": cfg, "source_hashes": initial_hashes,
                    "table_hashes": {n: sha256(staging / n) for n in tables},
                    "image_inventory_hashes": inventories, "active_studies": len(active["train"]),
                    "gold_studies": int(master.is_gold.sum()), "selected_images": len(images)}
        manifest["signature"] = fingerprint(manifest)
        write_json(staging / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError("Préparation incomplète existante : choisir un nouveau output_dir.")
        staging.rename(destination)
    print(f"Préparation créée : {destination}; {len(images)} images sélectionnées.")
    return destination


def load_prepared(cfg, check_images=False):
    directory = output_root(cfg) / "prepared"
    manifest = read_json(directory / "manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "signature"}
    if fingerprint(payload) != manifest["signature"] or manifest["config"] != cfg:
        raise ValueError("Préparation/configuration incompatible : utiliser un nouveau output_dir.")
    if source_hashes(data_root(cfg)) != manifest["source_hashes"]:
        raise ValueError("Les CSV source ont changé depuis la préparation.")
    for name, digest in manifest["table_hashes"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Table préparée modifiée : {name}.")
    tables = {name[:-4]: pd.read_csv(directory / name, dtype={UID: str, SERIES: str})
              for name in manifest["table_hashes"]}
    if check_images:
        for row in tables["selected_images"].itertuples(index=False):
            stat = (data_root(cfg) / row.path).stat()
            if stat.st_size != row.size_bytes or stat.st_mtime_ns != row.mtime_ns:
                raise ValueError(f"Image sélectionnée modifiée : {row.path}.")
    return manifest, tables
