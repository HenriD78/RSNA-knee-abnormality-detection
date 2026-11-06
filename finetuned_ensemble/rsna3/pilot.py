"""Profilage sur fold != 0, puis gel d'une recette commune aux cinq folds."""
from __future__ import annotations

import heapq
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import Cache
from .common import (LABELS, UID, atomic_json, input_tables, read_json, require,
                     sha256, source_hash, spec, versions)
from .data import StudyDataset, as_batch
from .model import StudyModel, bce_contribution, denominators, optimizer_for
from .train import TailSchedule, checkpoint, restore_rng, seed_all


def profile(model_id, project, cache, output, microbatches=104, device="cuda", windows=None):
    require(microbatches >= 100 and microbatches % 8 == 0, "Au moins 100 micro-batches, multiple de huit")
    require(torch.cuda.is_available(), "Le pilote doit mesurer la véritable exécution CUDA FP16")
    device = torch.device(device)
    cfg = spec(model_id)
    if windows is not None:
        require(1 <= windows <= cfg["input"]["train_windows_per_slot"], "Couverture pilote invalide")
        cfg["input"]["train_windows_per_slot"] = windows
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_all(42)
    startup_started = time.monotonic()
    table, _ = input_tables(project)
    train = table[table.fold != 0].reset_index(drop=True)
    reader = Cache(cache, verify=True)
    require(set(reader.entries) == set(table[UID]), "Cache train incomplet")
    # Include dense studies, without selecting by labels, plus a uniform UID sample.
    dense = sorted(range(len(train)), key=lambda i: (-reader.entries[train.iloc[i][UID]]["windows"], train.iloc[i][UID]))[:32]
    random_order = np.random.default_rng(42).permutation(len(train)).tolist()
    order = list(dict.fromkeys(dense + random_order))[:microbatches + 32]
    cache_centres = reader.manifest["contract"]["centres"]
    reader.close()
    dataset = StudyDataset(train, cache, cfg, fold=0, epoch=0, training=True)
    loader = DataLoader(dataset, batch_size=None, sampler=order, num_workers=1,
                        generator=torch.Generator().manual_seed(42), pin_memory=True, timeout=120)
    measured_iterator = iter(loader)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    weight = Path(project) / cfg["initialization"]["weight_path"]
    t0 = time.monotonic()
    model = StudyModel(cfg, weight).to(device)
    torch.cuda.synchronize(device)
    reload_seconds = time.monotonic() - startup_started
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda")
    schedule = TailSchedule(optimizer, 1000)
    events = []

    def group(indices, adapt, iterator=None):
        model.set_phase(adapt)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rows = train.iloc[indices]
        y = torch.tensor(rows[LABELS].to_numpy(np.float32), device=device)
        w = torch.tensor(rows[[l + "__conf" for l in LABELS]].to_numpy(np.float32), device=device)
        denom = denominators(y, w)
        torch.cuda.synchronize(device)
        start = time.monotonic()
        load_seconds = 0.
        for j, index in enumerate(indices):
            t = time.monotonic()
            sample = next(iterator) if iterator is not None else dataset[index]
            require(sample["uid"] == train.iloc[index][UID], "Ordre pilote désaligné")
            batch = as_batch(sample, device)
            torch.cuda.synchronize(device)
            load_seconds += time.monotonic() - t
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**batch)
            loss = bce_contribution(logits, y[j:j + 1], w[j:j + 1], denom)
            require(torch.isfinite(loss).item(), "Perte pilote non finie")
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        backbone_grads = [p.grad for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
        if adapt:
            require(any(g is not None and torch.isfinite(g).all().item() for g in backbone_grads), "Encodeur sans gradient")
        require(all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad), "Gradient dans le préfixe gelé")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        require(scaler.get_scale() >= old_scale, "Overflow AMP pendant le pilote ; diagnostiquer avant le gel")
        torch.cuda.synchronize(device)
        return {"microbatches": len(indices), "seconds": time.monotonic() - start, "loading_seconds": load_seconds}

    # Head and tail are profiled separately. Warmups are excluded from throughput.
    events.append({"phase": "head_cold_first_pass", **group(order[:8], False, measured_iterator)})
    head_groups = [group(order[8:16], False, measured_iterator), group(order[16:24], False, measured_iterator)]
    head = {"seconds": sum(g["seconds"] for g in head_groups), "microbatches": 16,
            "loading_seconds": sum(g["loading_seconds"] for g in head_groups)}
    events.append({"phase": "head_measured", **head})
    group(order[24:32], True, measured_iterator)
    tail_events = []
    for start in range(32, 32 + microbatches, 8):
        event = group(order[start:start + 8], True, measured_iterator)
        tail_events.append(event)
    events.append({"phase": "tail_measured", "groups": tail_events})
    # A real optimizer-boundary interruption/reload, then compare the next update.
    cp = output / f"{model_id}_pilot_resume.pt"
    t0 = time.monotonic()
    checkpoint(cp, model, optimizer, schedule, scaler, {"pilot": True}, 1, order, 0, 0, [])
    checkpoint_seconds = time.monotonic() - t0
    next_indices = order[:8]
    group(next_indices, True)
    expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    state = torch.load(cp, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    group(next_indices, True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected[key], rtol=1e-4, atol=1e-5, msg=f"Reprise pilote : {key}")
    del state, expected
    cp.unlink()  # Pilot updates are never an initialization for real CV.
    eval_data = StudyDataset(train.iloc[order[:16]], cache, cfg, fold=0, training=False)
    model.eval()
    torch.cuda.synchronize(device)
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(len(eval_data)):
            model(**as_batch(eval_data[i], device))
    torch.cuda.synchronize(device)
    eval_seconds = (time.monotonic() - start) / len(eval_data)
    profile_data = {"model_id": model_id, "config": cfg, "source_hash": source_hash(), "versions": versions(),
                    "cache_manifest_sha256": sha256(Path(cache) / "manifest.json"), "cache_centres": cache_centres,
                    "profile_train_fold": "fold != 0", "external_fold_evaluated": False,
                    "head_seconds_per_study": head["seconds"] / head["microbatches"],
                    "tail_seconds_per_study": sum(e["seconds"] for e in tail_events) / microbatches,
                    "eval_seconds_per_study": eval_seconds, "reload_seconds": reload_seconds,
                    "checkpoint_seconds": checkpoint_seconds, "events": events,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device), "resume_verified": True,
                    "pilot_weights_discarded": True, "cache_coldness": "first pass recorded; OS cache not forcibly evicted"}
    atomic_json(output / f"profile_{model_id}.json", profile_data)
    dataset.cache.close()
    eval_data.cache.close()
    return profile_data


def projection(profile_data, counts, epochs, quota_factor, gpus=2):
    p = profile_data
    train_n = sum(counts)
    tasks = []
    for count in counts:
        n = train_n - count
        tasks.append(n * (p["head_seconds_per_study"] + (epochs - 1) * p["tail_seconds_per_study"]) +
                     count * p["eval_seconds_per_study"] + p["reload_seconds"] +
                     p["checkpoint_seconds"] * (math.ceil(math.ceil(n / 8) * epochs / 25) + epochs))
    queue = [0.] * gpus
    for task in tasks:
        heapq.heapreplace(queue, queue[0] + task)
    return max(queue) * 1.2 * quota_factor / 3600


def freeze(profile_path, project, output, quota_before, quota_after, quota_elapsed_hours,
           gpus=2, max_epochs=12):
    require(quota_before > quota_after >= 0 and quota_elapsed_hours > 0, "Mesure réelle avant/après du quota requise")
    require(gpus in (1, 2), "Un ou deux GPU indépendants")
    p = read_json(profile_path)
    require(p["source_hash"] == source_hash() and p["versions"] == versions(), "Pilote/code/environnement incompatibles")
    require(p["resume_verified"] and not p["external_fold_evaluated"], "Pilote non admissible")
    cfg = p["config"]
    require(not list(Path(output).parent.glob(f"**/{cfg['model_id']}/fold*/completed.json")),
            "Évaluation externe déjà présente ; ouvrir une nouvelle expérience")
    table, _ = input_tables(project)
    q = (quota_before - quota_after) / quota_elapsed_hours
    counts = [int((table.fold == k).sum()) for k in range(5)]
    feasible = [epoch for epoch in range(2, max_epochs + 1)
                if projection(p, counts, epoch, q, gpus) <= cfg["budget_quota_hours"]]
    if not feasible:
        atomic_json(output, {"status": "not_feasible_in_budget", "model_id": cfg["model_id"],
                             "minimum_projected_quota_hours": projection(p, counts, 2, q, gpus),
                             "advice_fr": "Refaire un pilote avec moins de fenêtres, ou conserver les piliers terminés."})
        return read_json(output)
    # Choose the longest recipe that fits, before any external predictions exist.
    epochs = max(feasible)
    cfg["status"] = "frozen_after_pilot"
    cfg["training"]["epochs_target"] = epochs
    cfg.update({"source_hash": source_hash(), "versions": versions(), "pilot_sha256": sha256(profile_path),
                "cache_manifest_sha256": p["cache_manifest_sha256"], "quota_factor": q,
                "quota_measurement": {"before": quota_before, "after": quota_after, "elapsed_hours": quota_elapsed_hours},
                "profile": p, "profiled_gpus": gpus,
                "projected_quota_hours": projection(p, counts, epochs, q, gpus)})
    output = Path(output)
    if output.exists():
        require(read_json(output) == cfg, "Recette déjà figée ; ne pas l'écraser avec une variante")
    else:
        atomic_json(output, cfg)
    return cfg
