"""Configuration portable : calculs temporaires, archives de reprise persistantes."""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/gpu.json"):
    path = Path(path)
    cfg = json.loads((path if path.is_absolute() else ROOT / path).read_text())
    if cfg["device"] not in {"cpu", "cuda"}:
        raise ValueError("device attendu : cuda (calcul réel), cpu (tests synthétiques).")
    if cfg["profile"] != "full" or cfg["n_folds"] != 5 or cfg["samples_per_slot"] != 8:
        raise ValueError("Le paquet reprend les cinq folds et les huit images/slot déjà sélectionnés.")
    return cfg


def data_root(cfg):
    return Path(os.environ.get("RSNA_DATA_ROOT", ROOT / cfg["data_dir"]))


def output_root(cfg):
    return Path(os.environ.get("RSNA_OUTPUT_ROOT", ROOT / cfg["output_dir"]))


def assets_root():
    return Path(os.environ.get("RSNA_ASSET_ROOT", ROOT / "assets"))


def deadline_reached():
    value = os.environ.get("RSNA_STOP_AT")
    return bool(value and time.time() >= float(value))


def configure_environment():
    cache = Path(os.environ.get("RSNA_CACHE_ROOT", ROOT / "cache"))
    for key, child in {"HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
                       "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib",
                       "XDG_CACHE_HOME": "xdg", "PIP_CACHE_DIR": "pip"}.items():
        os.environ[key] = str(cache / child)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
