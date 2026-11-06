"""Entraînement des seules têtes et export OOF ; aucun encodeur n'est mis à jour."""

import time
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import output_root
from .constants import LABELS, READERS, UID
from .data import load_prepared
from .features import open_store
from .io import fingerprint, read_json, sha256, write_checkpoint, write_csv, write_json
from .models import StudyHead, head_spec, weighted_bce


class StudyFeatures(Dataset):
    """Jointure explicite par UID ; jamais par la position des lignes dans un CSV."""

    def __init__(self, ids, store, targets=None):
        self.ids, self.store = list(ids), store
        self.targets = targets.set_index(UID) if targets is not None else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = self.ids[index]
        feature, mask = self.store.get(uid)
        if self.targets is None:
            target = np.zeros(len(LABELS), np.float32)
            confidence = np.zeros(len(LABELS), np.float32)
        else:
            row = self.targets.loc[uid]
            target = row[LABELS].to_numpy(dtype=np.float32)
            confidence = row[[c + "__conf" for c in LABELS]].to_numpy(dtype=np.float32)
        return (torch.from_numpy(feature), torch.from_numpy(mask),
                torch.from_numpy(target), torch.from_numpy(confidence))


def training_identity(cfg, prepared, store, reader, fold):
    return fingerprint({"config": cfg, "prepared": prepared["signature"], "reader": reader,
                        "fold": fold, "feature_identity": store.identity,
                        "feature_records": store.manifest["studies"],
                        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                                    "python": sys.version.split()[0]}})


def load_checkpoint(path, expected_identity):
    """N'accepte pas un checkpoint d'un autre fold ou d'une autre extraction."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["identity"] != expected_identity:
        raise ValueError("Checkpoint incompatible : configuration, fold ou caractéristiques modifiés.")
    return checkpoint


@torch.inference_mode()
def predict_head(model, ids, store, batch_size):
    model.eval()
    loader = DataLoader(StudyFeatures(ids, store), batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    for features, mask, _, _ in loader:
        outputs.append(model(features, mask).sigmoid().numpy())
    if not outputs:
        raise ValueError("Aucune étude à prédire.")
    predictions = np.concatenate(outputs)
    if not np.isfinite(predictions).all():
        raise ValueError("Prédictions non finies.")
    frame = pd.DataFrame(predictions, columns=LABELS)
    frame.insert(0, UID, list(ids))
    return frame


def train_readers(cfg, readers, folds):
    prepared, tables = load_prepared(cfg, check_images=True)
    master = tables["train_folds"]
    active = master[master.active].copy()
    targets = tables["targets"]
    torch.set_num_threads(cfg["cpu_threads"])
    for reader in readers:
        store = open_store(cfg, reader, "train")
        missing = set(active[UID]) - set(store.manifest["studies"])
        if missing:
            raise ValueError(f"Extraction incomplète pour {reader}: {len(missing)} études manquantes.")
        for fold in folds:
            if fold not in range(cfg["n_folds"]):
                raise ValueError("Fold attendu entre 0 et 4.")
            train_rows, val_rows = active[active.fold != fold], active[active.fold == fold]
            if train_rows.empty or val_rows.empty:
                raise ValueError("Fold vide ; vérifier le profil smoke et les tables.")
            if set(train_rows.group_id) & set(val_rows.group_id):
                raise AssertionError("Un groupe de rapports traverse train/validation.")
            train_ids, val_ids = train_rows[UID].tolist(), val_rows[UID].tolist()
            identity = training_identity(cfg, prepared, store, reader, fold)
            spec = head_spec(cfg, reader, store.feature_dim)
            directory = output_root(cfg) / "checkpoints" / reader / f"fold{fold}"
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint_path = directory / "last.pt"
            torch.manual_seed(cfg["seed"] + fold)
            model = StudyHead(**spec)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                         weight_decay=cfg["weight_decay"])
            epoch_start, history = 0, []
            if checkpoint_path.exists():
                checkpoint = load_checkpoint(checkpoint_path, identity)
                model.load_state_dict(checkpoint["head"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                torch.set_rng_state(checkpoint["torch_rng"])
                epoch_start, history = checkpoint["epoch"], checkpoint["history"]
                if epoch_start > cfg["epochs"]:
                    raise ValueError("Checkpoint plus long que l'entraînement configuré.")
                print(f"Reprise {reader}/fold{fold} après {epoch_start} époque(s).")
            dataset = StudyFeatures(train_ids, store, targets)
            # Le générateur d'ordre est déterministe par époque, y compris après reprise.
            for epoch in range(epoch_start, cfg["epochs"]):
                generator = torch.Generator().manual_seed(cfg["seed"] + fold + 1000 * epoch)
                loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                                    generator=generator, num_workers=0)
                start, total_loss, count = time.perf_counter(), 0., 0
                model.train()
                for features, mask, labels, confidence in tqdm(loader, desc=f"{reader}/f{fold}/e{epoch + 1}"):
                    optimizer.zero_grad(set_to_none=True)
                    loss = weighted_bce(model(features, mask), labels, confidence)
                    if not torch.isfinite(loss):
                        raise ValueError("Perte non finie : entraînement arrêté.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    total_loss += loss.item() * len(features)
                    count += len(features)
                history.append({"epoch": epoch + 1, "loss": total_loss / count,
                                "seconds": time.perf_counter() - start})
                # Aucun score du fold évalué n'intervient dans le choix de ce checkpoint.
                write_checkpoint(checkpoint_path, {
                    "identity": identity, "reader": reader, "fold": fold, "spec": spec,
                    "epoch": epoch + 1, "head": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "history": history,
                })
                write_csv(directory / "history.csv", pd.DataFrame(history))
            frame = predict_head(model, val_ids, store, cfg["batch_size"])
            frame = frame.merge(val_rows[[UID, "fold", "group_id", "is_gold"]], on=UID, validate="one_to_one")
            path = output_root(cfg) / "predictions/oof" / reader / f"fold{fold}.csv"
            write_csv(path, frame)
            write_json(path.with_suffix(".json"), {"identity": identity, "sha256": sha256(path),
                       "checkpoint_sha256": sha256(checkpoint_path), "rows": len(frame)})
        collect_oof(cfg, reader, allow_incomplete=True)


def collect_oof(cfg, reader, allow_incomplete=False):
    prepared, tables = load_prepared(cfg)
    active = tables["train_folds"]
    active = active[active.active]
    store = open_store(cfg, reader, "train")
    directory = output_root(cfg) / "predictions/oof" / reader
    frames = []
    for fold in range(cfg["n_folds"]):
        path = directory / f"fold{fold}.csv"
        if not path.exists():
            continue
        receipt = read_json(path.with_suffix(".json"))
        identity = training_identity(cfg, prepared, store, reader, fold)
        checkpoint_path = output_root(cfg) / "checkpoints" / reader / f"fold{fold}/last.pt"
        if (receipt["identity"] != identity or receipt["sha256"] != sha256(path)
                or receipt["checkpoint_sha256"] != sha256(checkpoint_path)):
            raise ValueError("Export OOF incompatible ou modifié.")
        checkpoint = load_checkpoint(checkpoint_path, identity)
        if checkpoint["epoch"] != cfg["epochs"]:
            raise ValueError("L'export OOF nécessite le checkpoint de la dernière époque.")
        frame = pd.read_csv(path, dtype={UID: str})
        expected = active[active.fold == fold]
        if frame[UID].duplicated().any() or set(frame[UID]) != set(expected[UID]):
            raise ValueError("Couverture OOF incorrecte pour un fold.")
        reference = expected.set_index(UID).loc[frame[UID]]
        for column in ["fold", "group_id", "is_gold"]:
            if frame[column].tolist() != reference[column].tolist():
                raise ValueError(f"Métadonnée OOF incorrecte : {column}.")
        if not np.isfinite(frame[LABELS].to_numpy()).all():
            raise ValueError("OOF non fini.")
        frames.append(frame)
    if not frames:
        raise ValueError(f"Aucun export OOF pour {reader}.")
    combined = pd.concat(frames, ignore_index=True).sort_values(UID)
    complete = set(combined[UID]) == set(active[UID]) and not combined[UID].duplicated().any()
    if not complete and not allow_incomplete:
        raise ValueError(f"OOF incomplet pour {reader} : entraîner les cinq folds.")
    if complete:
        write_csv(output_root(cfg) / "predictions/oof" / f"{reader}.csv", combined)
    else:
        print(f"OOF partiel {reader}: {len(combined)}/{len(active)} études, évaluation finale indisponible.")
    return combined


def predict_test(cfg, readers):
    prepared, tables = load_prepared(cfg, check_images=True)
    test_ids = tables["test_ids"][UID].tolist()
    torch.set_num_threads(cfg["cpu_threads"])
    for reader in readers:
        train_store, test_store = open_store(cfg, reader, "train"), open_store(cfg, reader, "test")
        predictions, receipts = [], []
        for fold in range(cfg["n_folds"]):
            path = output_root(cfg) / "checkpoints" / reader / f"fold{fold}/last.pt"
            identity = training_identity(cfg, prepared, train_store, reader, fold)
            checkpoint = load_checkpoint(path, identity)
            if checkpoint["epoch"] != cfg["epochs"]:
                raise ValueError("Prédiction test refusée : entraînement du fold incomplet.")
            model = StudyHead(**checkpoint["spec"])
            model.load_state_dict(checkpoint["head"])
            frame = predict_head(model, test_ids, test_store, cfg["batch_size"])
            write_csv(output_root(cfg) / "predictions/test" / reader / f"fold{fold}.csv", frame)
            predictions.append(frame[LABELS].to_numpy())
            receipts.append(sha256(path))
        # Les cinq probabilités sont moyennées AVANT le calcul des rangs.
        frame = pd.DataFrame(np.mean(predictions, axis=0), columns=LABELS)
        frame.insert(0, UID, test_ids)
        path = output_root(cfg) / "predictions/test" / f"{reader}.csv"
        write_csv(path, frame)
        write_json(path.with_suffix(".json"), {"prepared": prepared["signature"], "reader": reader,
                   "checkpoint_hashes": receipts, "sha256": sha256(path),
                   "test_feature_identity": test_store.identity})
