"""Un coordinateur, deux sous-processus indépendants, budgets persistants."""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, atomic_csv, atomic_json, digest, read_json, require, sha256


def run_queue(experiments, project, cache, output, gpus, quota_available, session_hours=8, save_margin_seconds=600):
    import pandas as pd
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gpus = list(map(str, gpus))
    require(1 <= len(gpus) <= 2 and len(set(gpus)) == len(gpus), "Un ou deux GPU distincts")
    require(quota_available > 8, "Conserver 4 h export + 3 h réserve + 1 h validation")
    configs = [(Path(p).resolve(), read_json(p)) for p in experiments]
    configs.sort(key=lambda pair: pair[1].get("priority", 99))
    require(all(c.get("status") == "frozen_after_pilot" and c["profiled_gpus"] == len(gpus) for _, c in configs),
            "Recettes figées requises, même nombre de GPU qu'à la projection")
    require(len({c["model_id"] for _, c in configs}) == len(configs), "Modèles dupliqués")
    lock = (output / "queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("Un coordinateur utilise déjà ces sorties")
    path = output / "queue_state.json"
    recipes = {c["model_id"]: digest(c) for _, c in configs}
    state = read_json(path) if path.exists() else {"recipes": recipes, "quota_estimated": {}, "jobs": {}, "sessions": []}
    require(all(model not in state["recipes"] or state["recipes"][model] == signature for model, signature in recipes.items()),
            "File existante sous d'autres recettes")
    state["recipes"].update(recipes)
    pending = []
    for p, cfg in configs:
        for fold in range(5):
            key = f"{cfg['model_id']}/fold{fold}"
            done = output / key / "completed.json"
            if done.exists():
                meta = read_json(done)
                require(meta["contract"]["config_hash"] == digest(cfg), "Contrat d'un fold terminé incompatible")
                require(sha256(done.parent / f"final_fold{fold}.safetensors") == meta["checkpoint_sha256"] and
                        sha256(done.parent / "oof.csv") == meta["oof_sha256"], "Fold terminé corrompu")
                state["jobs"][key] = {"status": "completed"}
            else:
                pending.append((key, p, cfg, fold))
    active, failed = {}, []
    stop = {"requested": False}
    old = {s: signal.signal(s, lambda *_: stop.update(requested=True)) for s in (signal.SIGINT, signal.SIGTERM)}
    start = previous = time.monotonic()
    require(session_hours * 3600 > save_margin_seconds + 60, "Session trop courte pour la marge de sauvegarde")
    deadline = start + session_hours * 3600 - save_margin_seconds
    session_quota, session_cap = 0., min(15., quota_available - 8.)
    terminated_at = None
    for cfg_path, cfg in configs:
        state["quota_estimated"].setdefault(cfg["model_id"], 0.)

    def persist():
        atomic_json(path, state)
        rows = []
        for _, cfg in configs:
            for fold in range(5):
                key = f"{cfg['model_id']}/fold{fold}"
                rows.append({"model_id": cfg["model_id"], "fold": fold,
                             "statut": state["jobs"].get(key, {}).get("status", "pending"),
                             "quota_estime_pilier_h": state["quota_estimated"][cfg["model_id"]]})
        atomic_csv(output / "suivi_folds_reel.csv", pd.DataFrame(rows))

    try:
        while pending or active:
            now = time.monotonic()
            dt = now - previous
            previous = now
            families = {job["cfg"]["model_id"]: job["cfg"] for job in active.values()}
            for model_id, cfg in families.items():
                charge = dt / 3600 * cfg["quota_factor"]
                state["quota_estimated"][model_id] += charge
                session_quota += charge
            for gpu, job in list(active.items()):
                code = job["process"].poll()
                model_id = job["cfg"]["model_id"]
                exhausted = (state["quota_estimated"][model_id] + job["cfg"]["quota_factor"] * save_margin_seconds / 3600
                             >= job["cfg"]["budget_quota_hours"])
                if code is not None:
                    job["log"].close()
                    done = output / job["key"] / "completed.json"
                    status = "completed" if code == 0 and done.exists() else "paused" if code == 75 else "failed"
                    state["jobs"][job["key"]] = {"status": status, "returncode": code, "gpu": gpu}
                    if status == "failed":
                        failed.append(job["key"])
                    del active[gpu]
                elif exhausted and not job.get("signalled"):
                    job["process"].terminate()
                    job["signalled"] = now
                    pending = [task for task in pending if task[2]["model_id"] != model_id]
            save_quota = max((c["quota_factor"] for _, c in configs), default=1.) * save_margin_seconds / 3600
            if now >= deadline or session_quota + save_quota >= session_cap or stop["requested"]:
                pending.clear()
                if terminated_at is None:
                    terminated_at = now
                for job in active.values():
                    if not job.get("signalled"):
                        job["process"].terminate()
                        job["signalled"] = now
            for job in active.values():
                if job.get("signalled") and now - job["signalled"] > save_margin_seconds - 60:
                    job["process"].kill()
            for gpu in gpus:
                if gpu in active or not pending or terminated_at is not None:
                    continue
                key, p, cfg, fold = pending.pop(0)
                if state["quota_estimated"][cfg["model_id"]] + cfg["quota_factor"] * save_margin_seconds / 3600 >= cfg["budget_quota_hours"]:
                    state["jobs"][key] = {"status": "budget_exhausted"}
                    continue
                log_dir = output / "logs"
                log_dir.mkdir(exist_ok=True)
                log = (log_dir / f"{cfg['model_id']}_fold{fold}.log").open("a", buffering=1)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
                command = [sys.executable, "-m", "rsna3", "train", "--experiment", str(p), "--project", str(Path(project).resolve()),
                           "--cache", str(Path(cache).resolve()), "--output", str(output.resolve()), "--fold", str(fold),
                           "--device", "cuda:0", "--deadline-seconds", str(max(1., deadline - now))]
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = {"process": process, "log": log, "cfg": cfg, "key": key}
                state["jobs"][key] = {"status": "running", "pid": process.pid, "gpu": gpu}
                print(f"GPU {gpu}: {key}", flush=True)
            persist()
            if active:
                time.sleep(2)
        state["sessions"].append({"elapsed_seconds": time.monotonic() - start,
                                  "quota_estimated_hours": session_quota,
                                  "note": "Estimation par q mesuré ; réconcilier avec l'interface Kaggle."})
        persist()
        require(not failed, f"Folds en échec : {failed}. Consulter les logs puis relancer la même file.")
        return state
    finally:
        for job in active.values():
            job["process"].terminate()
        for job in active.values():
            try:
                job["process"].wait(timeout=save_margin_seconds)
            except subprocess.TimeoutExpired:
                job["process"].kill()
                job["process"].wait()
            job["log"].close()
        for sig, handler in old.items():
            signal.signal(sig, handler)
        lock.close()
