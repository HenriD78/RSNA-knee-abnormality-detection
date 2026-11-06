"""Configuration explicite ; les chemins relatifs partent du nouveau dossier."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/full.json"):
    """Charge un profil sans créer de dossier ni démarrer un traitement."""
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    cfg = json.loads(path.read_text(encoding="utf-8"))
    expected = set(json.loads((ROOT / "configs/full.json").read_text()).keys())
    if set(cfg) != expected:
        raise ValueError(f"Clés de configuration différentes : {set(cfg) ^ expected}")
    if cfg["profile"] not in {"full", "smoke"} or cfg["n_folds"] != 5:
        raise ValueError("Utiliser full/smoke et exactement cinq folds.")
    if cfg["device"] != "cpu":
        raise ValueError("Cette version locale est configurée pour le CPU.")
    for key in ["samples_per_slot", "epochs", "batch_size", "cpu_threads",
                "extraction_batch_size", "coatnet_batch_size", "hidden_dim", "transformer_heads"]:
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} doit être un entier positif.")
    if cfg["hidden_dim"] % cfg["transformer_heads"]:
        raise ValueError("hidden_dim doit être divisible par transformer_heads.")
    if not 0 <= cfg["dropout"] < 1 or cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("Paramètres d'optimisation invalides.")
    if cfg["tiny_depth"] != 2 or cfg["bootstrap_samples"] < 0:
        raise ValueError("tiny_depth=2 et bootstrap_samples>=0 sont requis.")
    output = (ROOT / cfg["output_dir"]).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("Les sorties doivent rester dans un sous-dossier de baseline_candidate/.")
    return cfg


def data_root(cfg):
    return (ROOT / cfg["data_dir"]).resolve()


def output_root(cfg):
    return (ROOT / cfg["output_dir"]).resolve()


def configure_environment():
    """Garde les caches des bibliothèques dans le dossier du livrable."""
    for key, child in {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib",
        "XDG_CACHE_HOME": "xdg", "PIP_CACHE_DIR": "pip",
    }.items():
        os.environ[key] = str(ROOT / "cache" / child)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
