from __future__ import annotations

import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from .cache import Cache
from .common import (LABELS, UID, atomic_csv, atomic_json, digest, input_tables,
                     read_json, require, sha256, source_hash, versions)
from .data import StudyDataset, as_batch
from .model import StudyModel, bce_contribution, denominators, optimizer_for


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


class TailSchedule:
    def __init__(self, optimizer, total, final_factor=.1):
        self.optimizer, self.total = optimizer, max(1, total)
        self.base = [g["lr"] for g in optimizer.param_groups]
        self.warmup = max(1, math.ceil(self.total * .05))
        self.final_factor, self.step = final_factor, 0

    def apply(self, adapt):
        if not adapt:
            factor = 1.
        elif self.step < self.warmup:
            factor = (self.step + 1) / self.warmup
        else:
            progress = (self.step - self.warmup) / max(1, self.total - self.warmup - 1)
            factor = self.final_factor + (1 - self.final_factor) * .5 * (1 + math.cos(math.pi * min(1., progress)))
        for group, base in zip(self.optimizer.param_groups, self.base):
            group["lr"] = base * factor

    def state_dict(self):
        return {"total": self.total, "base": self.base, "warmup": self.warmup,
                "final_factor": self.final_factor, "step": self.step}

    def load_state_dict(self, state):
        require(all(state[k] == getattr(self, k) for k in ("total", "base", "warmup", "final_factor")),
                "Scheduler incompatible")
        self.step = state["step"]


def checkpoint(path, model, optimizer, scheduler, scaler, contract, epoch, order, offset, steps, history):
    require(all(p.grad is None for p in model.parameters()), "Sauvegarde hors frontière optimizer.step")
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "rng": rng_state(), "contract": contract,
             "epoch": epoch, "order": order, "offset": offset, "steps": steps, "history": history,
             "accumulator_empty": True}
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def predict_model(model, table, cache, cfg, fold, device, workers=1, fixture=False, deadline=math.inf):
    dataset = StudyDataset(table, cache, cfg, fold, training=False, fixture=fixture)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=None, num_workers=workers, generator=generator,
                        pin_memory=device.type == "cuda", timeout=120 if workers else 0)
    model.eval()
    rows = []
    with torch.inference_mode():
        for sample in loader:
            require(time.monotonic() < deadline, "Échéance atteinte pendant la prédiction")
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probability = model(**as_batch(sample, device)).float().sigmoid()[0].cpu().numpy()
            rows.append({UID: sample["uid"], **dict(zip(LABELS, map(float, probability)))})
    dataset.cache.close()
    return pd.DataFrame(rows)


def train_fold(experiment, project, cache, output, fold, device="cuda", stop_after=None,
               deadline_seconds=None, fixture_table=None, fixture_weight=None, save_every=25):
    """No external-fold evaluation until the predeclared last epoch has finished."""
    cfg = read_json(experiment) if isinstance(experiment, (str, Path)) else experiment
    fixture = fixture_table is not None
    require(fixture or cfg.get("status") == "frozen_after_pilot", "Recette non figée par le pilote")
    require(cfg["training"]["epochs_target"] >= 2, "Une époque tête puis au moins une adaptation")
    require(fixture or cfg["source_hash"] == source_hash(), "Code modifié depuis le gel du pilote")
    require(fixture or cfg["versions"] == versions(), "Environnement modifié depuis le pilote")
    table, manifest = (fixture_table, {}) if fixture else input_tables(project)
    device = torch.device(device)
    require(fixture or device.type == "cuda", "Le vrai fine-tuning exige un GPU CUDA/AMP")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(cfg["training"]["cpu_threads_per_worker"])
    seed_all(42 + fold)
    cache_reader = Cache(cache, verify=True, allow_fixture=fixture)
    require(fixture or cfg["cache_manifest_sha256"] == sha256(Path(cache) / "manifest.json"), "Cache modifié depuis le pilote")
    require(set(table[UID]) == set(cache_reader.entries), "Couverture train/cache différente")
    cache_reader.close()
    train = table[table.fold != fold].reset_index(drop=True)
    heldout = table[table.fold == fold].reset_index(drop=True)
    require(len(train) > 0 and len(heldout) > 0, "Fold vide")
    require(not (set(train.group_id) & set(heldout.group_id)), "Fuite de groupes")
    weight = fixture_weight or Path(project) / cfg["initialization"]["weight_path"]
    model = StudyModel(cfg, weight=weight).to(device)
    optimizer = optimizer_for(model, cfg)
    model.set_phase(True)
    tail_names = [n for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    model.set_phase(False)
    epochs, accumulation = cfg["training"]["epochs_target"], cfg["training"]["gradient_accumulation_steps"]
    scheduler = TailSchedule(optimizer, (epochs - 1) * math.ceil(len(train) / accumulation), cfg["training"]["final_lr_factor"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    contract = {"config_hash": digest(cfg), "source_hash": source_hash(), "versions": versions(),
                "inputs": manifest, "cache_sha256": sha256(Path(cache) / "manifest.json"),
                "labels": LABELS, "fold": fold, "tail_parameters": tail_names,
                "train_uids_hash": digest(train[UID].tolist()), "heldout_uids_hash": digest(heldout[UID].tolist())}
    out = Path(output) / cfg["model_id"] / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    active = out / "active.pt"
    done = out / "completed.json"
    if done.exists():
        metadata = read_json(done)
        require(metadata["contract"] == contract, "Fold terminé sous un autre contrat")
        require(sha256(out / f"final_fold{fold}.safetensors") == metadata["checkpoint_sha256"] and
                sha256(out / "oof.csv") == metadata["oof_sha256"], "Sorties de fold corrompues")
        return metadata
    epoch, order, offset, steps, history = 0, None, 0, 0, []
    if active.exists():
        # Only load our own trusted local checkpoint; it includes Python/NumPy RNG objects.
        state = torch.load(active, map_location="cpu", weights_only=False)
        require(state["contract"] == contract and state["accumulator_empty"], "Reprise incompatible")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch, order, offset, steps, history = (state[k] for k in ("epoch", "order", "offset", "steps", "history"))
        restore_rng(state["rng"])
        del state
    atomic_json(out / "contract.json", {"contract": contract, "config": cfg})
    stop = {"requested": False}
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: stop.update(requested=True))
    started, session_steps = time.monotonic(), 0
    deadline = started + deadline_seconds if deadline_seconds else math.inf
    try:
        while epoch < epochs:
            model.set_phase(epoch >= 1)
            model.train()
            if order is None:
                order = np.random.default_rng(42 + fold + epoch * 1009).permutation(len(train)).tolist()
                offset = 0
            require(sorted(order) == list(range(len(train))) and 0 <= offset <= len(order), "Sampler de reprise invalide")
            dataset = StudyDataset(train, cache, cfg, fold, epoch, training=True, fixture=fixture)
            generator = torch.Generator().manual_seed(42 + fold + epoch)
            loader = DataLoader(dataset, batch_size=None, sampler=order[offset:],
                                num_workers=cfg["training"]["loader_workers_per_gpu"], generator=generator,
                                pin_memory=device.type == "cuda",
                                timeout=120 if cfg["training"]["loader_workers_per_gpu"] else 0)
            iterator = iter(loader)
            while offset < len(order):
                if stop["requested"] or time.monotonic() >= deadline or (stop_after is not None and session_steps >= stop_after):
                    optimizer.zero_grad(set_to_none=True)
                    checkpoint(active, model, optimizer, scheduler, scaler, contract, epoch, order, offset, steps, history)
                    return {"status": "paused_at_step", "epoch": epoch, "offset": offset, "steps": steps}
                indices = order[offset:offset + accumulation]
                group = train.iloc[indices]
                targets = torch.tensor(group[LABELS].to_numpy(np.float32), device=device)
                confidence = torch.tensor(group[[l + "__conf" for l in LABELS]].to_numpy(np.float32), device=device)
                denom = denominators(targets, confidence)
                optimizer.zero_grad(set_to_none=True)
                scheduler.apply(epoch >= 1)
                loss_value, group_start = 0., time.monotonic()
                for j in range(len(indices)):
                    sample = next(iterator)
                    require(sample["uid"] == group.iloc[j][UID], "Sampler/cibles désalignés")
                    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(**as_batch(sample, device))
                    loss = bce_contribution(logits, targets[j:j + 1], confidence[j:j + 1], denom)
                    require(torch.isfinite(loss).item(), "Perte non finie")
                    scaler.scale(loss).backward()
                    loss_value += float(loss.detach())
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["training"]["max_grad_norm"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                success = scaler.get_scale() >= previous_scale
                if success and epoch >= 1:
                    scheduler.step += 1
                optimizer.zero_grad(set_to_none=True)
                offset += len(indices)
                steps += int(success)
                session_steps += 1
                history.append({"epoch": epoch, "offset": offset, "step": steps, "loss": loss_value,
                                "optimizer_step_applied": success, "seconds": time.monotonic() - group_start})
                if session_steps % save_every == 0:
                    checkpoint(active, model, optimizer, scheduler, scaler, contract, epoch, order, offset, steps, history)
                    print(f"{cfg['model_id']} fold {fold} époque {epoch+1}/{epochs}, {offset}/{len(train)}, loss={loss_value:.4f}", flush=True)
            dataset.cache.close()
            epoch, order, offset = epoch + 1, None, 0
            checkpoint(active, model, optimizer, scheduler, scaler, contract, epoch, order, offset, steps, history)
        require(any(r["epoch"] >= 1 and r["optimizer_step_applied"] for r in history), "Aucun pas d'adaptation réussi")
        final = out / f"final_fold{fold}.safetensors"
        temporary = final.with_name(final.name + ".tmp")
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temporary),
                  metadata={"contract": digest(contract), "labels": __import__("json").dumps(LABELS)})
        os.replace(temporary, final)
        # Only now run the external fold, using every frozen-cache window.
        oof = predict_model(model, heldout, cache, cfg, fold, device,
                            cfg["training"]["loader_workers_per_gpu"], fixture)
        from .common import validate_predictions
        oof = validate_predictions(oof, heldout[UID].tolist())
        oof = heldout[[UID, "group_id", "fold", "is_gold"]].merge(oof, on=UID, validate="one_to_one")
        oof["provenance"] = "generic_initialization_out_of_fold_final_epoch"
        oof["checkpoint_sha256"] = sha256(final)
        atomic_csv(out / "oof.csv", oof)
        atomic_csv(out / "history.csv", pd.DataFrame(history))
        metadata = {"status": "completed", "contract": contract, "config": cfg,
                    "checkpoint_sha256": sha256(final), "oof_sha256": sha256(out / "oof.csv"),
                    "epochs": epochs, "optimizer_steps": steps, "session_seconds": time.monotonic() - started,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        atomic_json(done, metadata)
        active.unlink(missing_ok=True)
        return metadata
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
