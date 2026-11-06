"""Orchestre les sessions GPU courtes sans changer les folds ou les hyperparamètres."""

import os
import time

import torch

from .config import configure_environment, deadline_reached, load_config, output_root
from .constants import READERS, UID
from .data import load_prepared
from .features import benchmark, extract, open_store
from .session import snapshot


def preflight():
    configure_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("Aucun GPU NVIDIA détecté : Settings > Accelerator > GPU, puis relancer.")
    import timm, torchvision
    print("GPU :", torch.cuda.get_device_name(0))
    print("PyTorch :", torch.__version__, "| torchvision :", torchvision.__version__, "| timm :", timm.__version__)
    if timm.__version__ != "1.0.29":
        raise RuntimeError("timm 1.0.29 requis pour préserver poids et transformations.")
    return True


def cache_complete(cfg, reader, split):
    _, tables = load_prepared(cfg)
    ids = set(tables["selected_images"].loc[tables["selected_images"].split == split, UID])
    path = output_root(cfg) / "features" / reader / split / "manifest.json"
    if not path.exists():
        return False
    store = open_store(cfg, reader, split)
    return set(store.manifest["studies"]) == ids


def run_session(hours=6, readers=None):
    """Arrêt volontaire avant la limite de session ; relancer pour continuer.

    Un quota dur peut interrompre n'importe quel service : télécharger/attacher
    régulièrement rsna_resume.tar. Aucune activation d'offre payante ici.
    """
    preflight()
    cfg = load_config()
    readers = list(READERS) if readers is None else list(readers)
    if not 0 < hours <= 10 or not readers or not set(readers) <= set(READERS):
        raise ValueError("Choisir 0 < hours <= 10 et des lecteurs connus.")
    os.environ["RSNA_STOP_AT"] = str(time.time() + hours * 3600)
    from .train import train_readers, predict_test
    from .evaluate import evaluate, export_submissions
    try:
        for reader in readers:
            if deadline_reached():
                return False
            if not cache_complete(cfg, reader, "train"):
                if not extract(cfg, [reader], "train"):
                    return False
            if deadline_reached() or not train_readers(cfg, [reader], list(range(5))):
                return False
            snapshot(cfg)
        # Un passage limité à un lecteur ne prétend pas fournir une comparaison complète.
        complete = all((output_root(cfg) / "predictions/oof" / f"{r}.csv").exists() for r in READERS)
        if complete and not deadline_reached():
            evaluate(cfg)
            for reader in READERS:
                if deadline_reached():
                    return False
                if not cache_complete(cfg, reader, "test"):
                    if not extract(cfg, [reader], "test"):
                        return False
                predict_test(cfg, [reader])
            export_submissions(cfg)
            print("PARCOURS COMPLET TERMINÉ : télécharger rsna_reports.zip et rsna_resume.tar.")
            return True
        print("Session partielle terminée. Reprise sauvegardée ; continuer dans une prochaine session.")
        return False
    finally:
        snapshot(cfg)


def benchmark_gpu(images=16):
    preflight()
    os.environ.pop("RSNA_STOP_AT", None)
    benchmark(load_config(), list(READERS), images)
    snapshot(load_config())
