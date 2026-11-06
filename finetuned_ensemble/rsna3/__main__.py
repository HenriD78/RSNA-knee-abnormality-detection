from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="RSNA : cache CPU, pilotes, fine-tuning, OOF, export, soumission")
    sub = parser.add_subparsers(dest="command", required=True)

    def arg(p, name, **kw):
        return p.add_argument("--" + name.replace("_", "-"), **kw)

    check = sub.add_parser("check")
    arg(check, "project", required=True)
    for command in ("probe-cache", "prepare-cache"):
        p = sub.add_parser(command)
        for name in ("studies_csv", "series_csv", "dicom_root", "output"):
            arg(p, name, required=True)
        arg(p, "centres", type=int, choices=(2, 4), default=4)
        arg(p, "max_hours", type=float, default=8)
        if command == "probe-cache":
            arg(p, "n", type=int, default=75)
        else:
            arg(p, "probe", required=True)
    p = sub.add_parser("pilot")
    for name in ("model", "project", "cache", "output"):
        arg(p, name, required=True)
    arg(p, "microbatches", type=int, default=104)
    arg(p, "windows", type=int)
    arg(p, "device", default="cuda:0")
    p = sub.add_parser("freeze")
    for name in ("profile", "project", "output"):
        arg(p, name, required=True)
    for name in ("quota_before", "quota_after", "quota_elapsed_hours"):
        arg(p, name, type=float, required=True)
    arg(p, "gpus", type=int, default=2)
    arg(p, "max_epochs", type=int, default=12)
    p = sub.add_parser("train")
    for name in ("experiment", "project", "cache", "output"):
        arg(p, name, required=True)
    arg(p, "fold", type=int, choices=range(5), required=True)
    arg(p, "device", default="cuda:0")
    arg(p, "stop_after", type=int)
    arg(p, "deadline_seconds", type=float)
    p = sub.add_parser("queue")
    arg(p, "experiments", nargs="+", required=True)
    for name in ("project", "cache", "output"):
        arg(p, name, required=True)
    arg(p, "gpus", nargs="+", default=["0", "1"])
    arg(p, "quota_available", type=float, required=True)
    arg(p, "session_hours", type=float, default=8)
    p = sub.add_parser("evaluate")
    for name in ("project", "output", "report"):
        arg(p, name, required=True)
    arg(p, "baseline_root")
    arg(p, "repetitions", type=int, default=2000)
    p = sub.add_parser("export")
    for name in ("project", "output", "destination"):
        arg(p, name, required=True)
    p = sub.add_parser("benchmark-inference")
    for name in ("bundle", "studies_csv", "series_csv", "dicom_root", "work"):
        arg(p, name, required=True)
    arg(p, "n", type=int, default=50)
    arg(p, "quota_factor", type=float, required=True)
    arg(p, "max_hours", type=float, default=2)
    p = sub.add_parser("submit")
    for name in ("bundle", "test_csv", "series_csv", "dicom_root", "work", "output_csv"):
        arg(p, name, required=True)
    arg(p, "max_hours", type=float, default=8)
    arg(p, "workers", type=int, default=1)
    arg(p, "public_recipe")
    arg(p, "candidate_3pct", action="store_true")
    arg(p, "optional_quota_hours", type=float)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "check":
        from .common import input_tables, spec, SPECS, sha256, require
        table, _ = input_tables(args["project"])
        for model in SPECS:
            cfg = spec(model)
            require(sha256(Path(args["project"]) / cfg["initialization"]["weight_path"]) == cfg["initialization"]["weight_sha256"],
                    f"Poids génériques invalides : {model}")
        result = {"studies": len(table), "gold": int(table.is_gold.sum()), "groups": table.group_id.nunique()}
    elif command in ("probe-cache", "prepare-cache"):
        import pandas as pd
        from .common import UID
        from .cache import probe, prepare
        args["studies"] = pd.read_csv(args.pop("studies_csv"), dtype={UID: str})[UID].tolist()
        if command == "prepare-cache":
            args["probe_path"] = args.pop("probe")
        result = (probe if command == "probe-cache" else prepare)(**args)
        result = {k: v for k, v in result.items() if k not in ("entries", "sample")}
    elif command == "pilot":
        from .pilot import profile
        args["model_id"] = args.pop("model")
        result = profile(**args)
        result = {k: v for k, v in result.items() if k not in ("config", "events")}
    elif command == "freeze":
        from .pilot import freeze
        args["profile_path"] = args.pop("profile")
        cfg = freeze(**args)
        result = {k: cfg[k] for k in ("status", "model_id")}
        if "training" in cfg:
            result.update(epochs=cfg["training"]["epochs_target"], projected_quota_hours=cfg["projected_quota_hours"])
    elif command == "train":
        import fcntl
        from .common import read_json
        from .train import train_fold
        cfg = read_json(args["experiment"])
        folder = Path(args["output"]) / cfg["model_id"] / f"fold{args['fold']}"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "worker.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = train_fold(**args)
        if result.get("status") == "paused_at_step":
            print(json.dumps(result, indent=2))
            return 75
        result = {k: v for k, v in result.items() if k not in ("contract", "config")}
    elif command == "queue":
        from .queue import run_queue
        result = run_queue(**args)
    elif command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(**args)
    elif command == "export":
        from .inference import export
        result = export(**args)
        result = {"files": len(result["files"]), "destination": args["destination"]}
    elif command == "benchmark-inference":
        from .inference import benchmark
        result = benchmark(**args)
    elif command == "submit":
        from .inference import submit
        result = submit(**args)
    print(json.dumps(result, indent=2, default=int, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
