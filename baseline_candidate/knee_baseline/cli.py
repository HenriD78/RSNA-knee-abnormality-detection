"""Commandes indépendantes : l'utilisateur choisit explicitement chaque étape."""

import argparse
import json
import sys

from .config import configure_environment, data_root, load_config, output_root
from .constants import LABELS, READERS, UID


def parse_readers(value):
    return list(READERS) if value == "all" else [value]


def inspect(cfg):
    """Inspection légère des CSV : aucun modèle, téléchargement ou index JPEG."""
    import pandas as pd
    root = data_root(cfg)
    frame = pd.read_csv(root / "train.csv", dtype={UID: str})
    test = pd.read_csv(root / "test.csv", dtype={UID: str})
    known = frame[LABELS].notna().sum(axis=1)
    print(json.dumps({"profile": cfg["profile"], "data_dir": str(root),
                      "output_dir": str(output_root(cfg)), "train_studies": len(frame),
                      "gold_studies": int((known == len(LABELS)).sum()),
                      "weak_studies": int((known == 0).sum()), "test_studies": len(test),
                      "readers": READERS, "execution": "inspection uniquement"},
                     indent=2, ensure_ascii=False))


def verify(cfg):
    """Vérifie les données préparées et les caches existants, sans les compléter."""
    from .data import load_prepared
    from .features import asset_dir, asset_manifest, open_store
    prepared, tables = load_prepared(cfg, check_images=True)
    print("CSV, folds et images sélectionnées : empreintes vérifiées.")
    for reader in READERS:
        if (asset_dir(reader) / "manifest.json").exists():
            asset_manifest(reader)
            print(f"Poids {reader}: OK")
        else:
            print(f"Poids {reader}: pas encore téléchargés")
        for split in ["train", "test"]:
            path = output_root(cfg) / "features" / reader / split / "manifest.json"
            if path.exists():
                store = open_store(cfg, reader, split)
                for uid in store.manifest["studies"]:
                    store.get(uid)
                expected = set(tables["selected_images"].loc[tables["selected_images"].split == split, UID])
                extra = set(store.manifest["studies"]) - expected
                if extra:
                    raise ValueError("Le cache contient des études hors de la préparation.")
                print(f"Cache {reader}/{split}: {len(store.manifest['studies'])}/{len(expected)} études valides")


def make_parser():
    parser = argparse.ArgumentParser(description="Baseline JPEG locale ; aucune étape lourde implicite.")
    parser.add_argument("--config", default="configs/full.json", help="Profil relatif à baseline_candidate/")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["inspect", "prepare", "verify", "evaluate"]:
        commands.add_parser(name)
    for name in ["assets", "benchmark", "extract", "train", "predict"]:
        sub = commands.add_parser(name)
        sub.add_argument("--reader", choices=["all"] + list(READERS), default="all")
        if name == "benchmark":
            sub.add_argument("--images", type=int, default=16)
        if name == "extract":
            sub.add_argument("--split", choices=["train", "test"], default="train")
        if name == "train":
            sub.add_argument("--fold", default="all", help="all, 0, ou une liste comme 0,1")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    cfg = load_config(args.config)
    configure_environment()
    readers = parse_readers(args.reader) if hasattr(args, "reader") else []
    if cfg["profile"] == "smoke":
        print("MODE SMOKE : parcours réduit ; les scores ne mesurent pas la performance.")
    if args.command == "inspect":
        inspect(cfg)
    elif args.command == "prepare":
        from .data import prepare
        prepare(cfg)
    elif args.command == "verify":
        verify(cfg)
    elif args.command == "assets":
        from .features import download_assets
        download_assets(readers)
    elif args.command == "benchmark":
        if args.images < 1:
            raise ValueError("Le benchmark nécessite au moins une image.")
        from .features import benchmark
        benchmark(cfg, readers, args.images)
    elif args.command == "extract":
        from .features import extract
        extract(cfg, readers, args.split)
    elif args.command == "train":
        from .train import train_readers
        folds = list(range(cfg["n_folds"])) if args.fold == "all" else sorted(set(map(int, args.fold.split(","))))
        if not folds or any(f not in range(cfg["n_folds"]) for f in folds):
            raise ValueError("Folds attendus : all ou entiers entre 0 et 4.")
        train_readers(cfg, readers, folds)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg)
    elif args.command == "predict":
        from .train import predict_test
        predict_test(cfg, readers)
        if args.reader == "all":
            from .evaluate import export_submissions
            export_submissions(cfg)
        else:
            print("Prédictions d'un seul lecteur sauvegardées. predict --reader all exportera les mélanges.")


def entrypoint():
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    entrypoint()
