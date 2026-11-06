#!/usr/bin/env python3
"""Contrôle en lecture seule des spécifications initiales, sans calcul de modèle.

Ce script ne valide pas un futur entraînement. Il vérifie les fichiers réellement
présents, leurs empreintes, la cohérence des trois recettes et les folds communs.
Il ne dépend que de la bibliothèque standard et n'écrit aucun résultat.
"""

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODELS = {
    "01_coatnet2": ("coatnet2", "coatnet", 2, 5.0, 384, 3),
    "02_convnext_small": ("convnext_small", "small", 1, 6.0, 224, 6),
    "03_dinov2_small": ("dinov2_small", "dino", 3, 4.0, 224, 4),
}
LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
)


def require(condition, message):
    """Lever une erreur explicite, y compris si Python est lancé avec -O."""
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    """Lire les poids par morceaux ; ne pas les charger comme un modèle."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed(rows, name):
    index = {row["StudyInstanceUID"]: row for row in rows}
    require(len(index) == len(rows), f"UID dupliqué dans {name}")
    require(all(index), f"UID vide dans {name}")
    return index


def numeric(value):
    return float(value) if value.strip() else float("nan")


def check_inputs(manifest):
    require(manifest["status"] == "planning_only", "Statut du manifeste inattendu")
    require(manifest["seed"] == 42 and manifest["n_folds"] == 5,
            "La graine et les cinq folds doivent rester communs")
    require(manifest["patient_independence_established"] is False,
            "Ne pas prétendre établir l'indépendance des patients")
    for name, entry in manifest["files"].items():
        path = PROJECT / entry["path"]
        require(path.is_file(), f"Entrée absente : {name} ({path})")
        require(sha256(path) == entry["sha256"], f"Empreinte incompatible : {name}")

    master = manifest["files"]["master_folds"]
    require(master["path"] == manifest["master_folds_authority"],
            "Deux autorités de folds différentes")
    folds = indexed(read_csv(PROJECT / master["path"]), "folds maîtres")
    require(len(folds) == manifest["studies"] == 4407, "Couverture différente de 4 407")
    require({row["fold"] for row in folds.values()} == set("01234"), "Folds invalides")
    require(all(row["active"] == "True" for row in folds.values()), "Études inactives")
    require(all(row["is_gold"] in {"True", "False"} for row in folds.values()),
            "Statut gold ambigu")
    groups = {}
    for row in folds.values():
        require(bool(row["group_id"]), "Groupe de rapport manquant")
        previous = groups.setdefault(row["group_id"], row["fold"])
        require(previous == row["fold"], "Un groupe de rapports traverse les folds")
    require(len(groups) == manifest["report_groups"] == 4257, "Nombre de groupes différent")
    counts = Counter(row["fold"] for row in folds.values())
    gold_counts = Counter(row["fold"] for row in folds.values() if row["is_gold"] == "True")
    require(dict(counts) == manifest["fold_counts"], "Effectifs de folds incompatibles")
    require(dict(gold_counts) == manifest["gold_fold_counts"], "Effectifs gold incompatibles")
    require(sum(gold_counts.values()) == manifest["gold_studies"] == 58, "Gold différent de 58")
    require(len(folds) - sum(gold_counts.values()) == manifest["weak_studies"] == 4349,
            "Effectif faible différent de 4 349")

    targets = indexed(read_csv(PROJECT / manifest["files"]["targets"]["path"]), "cibles")
    gold = indexed(read_csv(PROJECT / manifest["files"]["gold_labels"]["path"]), "labels gold")
    train = indexed(read_csv(PROJECT / manifest["files"]["train_csv"]["path"]), "train.csv")
    require(set(folds) == set(targets) == set(gold) == set(train), "UID désalignés entre CSV")
    for uid, row in targets.items():
        require(all(row[key] == folds[uid][key] for key in ("fold", "group_id", "is_gold")),
                f"Métadonnées de cibles incompatibles : {uid}")
        for label in LABELS:
            value, confidence = numeric(row[label]), numeric(row[label + "__conf"])
            require(math.isfinite(confidence) and 0 <= confidence <= 1, "Confiance invalide")
            if not math.isfinite(value):
                require(math.isnan(value) and confidence == 0, "Cible absente de poids non nul")
            else:
                require(0 <= value <= 1, "Cible hors [0,1]")
            if row["is_gold"] == "True":
                actual = numeric(gold[uid][label])
                if math.isfinite(actual):
                    require(value == actual and confidence == 1, "Gold altéré ou confiance différente de 1")
                else:
                    require(math.isnan(value) and confidence == 0, "Gold inconnu converti en label")
    print("OK — CSV, empreintes, UID, 4 407 études / 58 gold / 4 257 groupes sans chevauchement.")


def check_recipes(manifest, budget):
    expected_dirs = set(MODELS)
    require(expected_dirs.issubset({p.name for p in ROOT.iterdir() if p.is_dir()}),
            "Les trois sous-dossiers de modèles attendus doivent être présents")
    stages = {stage["id"]: stage["quota_hours"] for stage in budget["stages"]}
    require(len(stages) == 7 and all(value > 0 for value in stages.values()), "Postes de budget invalides")
    require(sum(stages.values()) == budget["total_quota_hours"] == 24, "Le budget ne totalise pas 24 h")
    require(budget["quota_factor_must_be_measured"] is True, "Décompte de quota non mesuré")
    require(budget["execution_strategy"] == "independent_fold_workers_no_ddp", "Stratégie GPU différente")
    require(budget["max_visible_gpus"] == 2, "Plan prévu pour deux GPU")
    master = manifest["files"]["master_folds"]
    for dirname, (model, reader, priority, quota, size, epochs) in MODELS.items():
        folder = ROOT / dirname
        cfg = read_json(folder / "experience_planifiee.json")
        require(cfg["status"] == "specification_only_not_a_training_cli_config",
                "La spécification ne doit pas se faire passer pour un trainer")
        require(cfg["model_id"] == model and cfg["priority"] == priority, "Identité/priorité incompatible")
        require(cfg["budget_quota_hours"] == stages[model] == quota, "Budget pilier incompatible")
        require((folder / cfg["reference_protocol"]).resolve() == ROOT / "PROTOCOLE_COMMUN_FR.md",
                "Protocole commun différent")
        require((folder / cfg["inputs_manifest"]).resolve() == ROOT / "entrees_manifest.json",
                "Manifeste commun différent")
        cv, image, training = cfg["cv"], cfg["input"], cfg["training"]
        require(cv["master_folds"] == master["path"] and cv["folds_sha256"] == master["sha256"],
                "Folds différents entre lecteurs")
        require(cv["n_folds"] == 5 and cv["seed"] == 42 and not cv["validation_fold_for_tuning"],
                "Contrat CV invalide")
        require(cv["train_folds"] == "all_except_k" and cv["predict_fold"] == "k", "Contrat OOF invalide")
        require(image["input_size"] == size and image["channels"] == "three_adjacent_slices",
                "Résolution ou pseudo-canaux différents")
        require(image["slots"] == 6 and image["missing_slots"] == "masked" and not image["slot_fill"],
                "Contrat de masques différent")
        require(image["cache_windows_per_slot_target"] == 4 and image["cache_windows_per_slot_fallback"] == 2,
                "Cache non conforme")
        require(1 <= image["train_windows_per_slot"] <= image["cache_windows_per_slot_fallback"],
                "Échantillonnage incompatible avec le cache réduit")
        require(training["epochs_target"] == epochs >= training["epochs_minimum"] == 2,
                "Nombre d'époques initial incompatible")
        require(training["head_warmup_epochs"] == 1 and training["epochs_frozen_after_profiling"],
                "Époques non préinscrites")
        require(training["study_batch_per_gpu"] * training["gradient_accumulation_steps"] ==
                training["effective_batch_per_fold"] == 8, "Batch effectif incohérent")
        require(training["confidence_normalization_scope"] == "effective_accumulated_batch_per_finding",
                "Les confiances seraient annulées au micro-batch")
        require(training["precision"] == "amp_fp16_gradscaler" and training["missing_targets_weight"] == 0
                and training["gold_confidence"] == 1, "Précision/perte différente")
        require(not any(training[key] for key in ("ema", "swa", "focal_loss", "mixup", "training_mirror", "tta")),
                "Ablation non prévue activée")
        require(training["checkpoint_selection"] == "final_epoch_fixed_in_advance", "Sélection externe de checkpoint")
        if reader == "dino":
            require(size % 14 == 0 and cfg["head"]["transformer_layers"] == 2, "Contrat DINOv2 invalide")
            require(training["tail_policy"] == "last_2_blocks_and_final_norm", "Dégel DINOv2 différent")
        else:
            require(training["tail_policy"] == "last_stage" and cfg["head"]["transformer_layers"] == 0,
                    "Dégel CNN différent")
        init = cfg["initialization"]
        require(init["kind"] == "generic_pretrained" and init["source_reader"] == reader
                and init["rsna_public_checkpoint_allowed_for_clean_oof"] is False,
                "Initialisation publique RSNA incompatible avec la CV principale")
        weight = PROJECT / init["weight_path"]
        original = read_json(weight.parent / "manifest.json")
        require(init["timm_model"] == original["model"] and init["revision"] == original["revision"],
                "Modèle ou révision différent des assets existants")
        require(sha256(weight) == init["weight_sha256"] == original["weight_sha256"], "Poids générique incompatible")
        require(image["normalization_mean"] == original["pretrained_cfg"]["mean"]
                and image["normalization_std"] == original["pretrained_cfg"]["std"], "Normalisation incompatible")
        require(not image["extra_center_crop"] and image["resize_interpolation"] == "bicubic", "Rendu d'entrée différent")
        require(cfg["outputs"]["expected_oof_studies"] == 4407 and cfg["outputs"]["expected_gold_studies"] == 58,
                "Couverture OOF attendue incompatible")
    print("OK — Trois initialisations génériques vérifiées ; recettes, masques, perte et budget total de 24 h cohérents.")


def check_schedule_and_docs(budget):
    rows = read_csv(ROOT / "suivi_folds.csv")
    jobs = {(row["model_id"], int(row["fold"])) for row in rows}
    expected = {(spec[0], k) for spec in MODELS.values() for k in range(5)}
    require(len(rows) == len(jobs) == 15 and jobs == expected, "Quinze tâches distinctes attendues")
    for row in rows:
        require(row["statut"] == "planifie" and row["epoches_terminees"] == "0", "Le suivi initial n'est plus vierge")
        require(all(not row[key] for key in ("quota_consomme_h", "checkpoint_sha256", "oof_sha256")),
                "Un résultat est annoncé dans le suivi initial")
    deadline = datetime.fromisoformat(budget["competition_deadline_utc"])
    paris = deadline.astimezone(ZoneInfo("Europe/Paris"))
    require(paris.isoformat() == budget["competition_deadline_paris"], "Conversion de l'échéance incorrecte")
    freeze = datetime.fromisoformat(budget["internal_code_freeze_paris"])
    start = datetime.fromisoformat(budget["internal_final_submission_start_paris"])
    require(freeze < start < deadline and budget["competition_max_runtime_hours"] == 9,
            "Calendrier interne incohérent")
    require((deadline - start).total_seconds() > 9 * 3600, "Marge finale insuffisante dans le calendrier")
    docs = [PROJECT / "docs/methodology/plan_entrainement_3_piliers.md", ROOT / "README_FR.md",
            ROOT / "PROTOCOLE_COMMUN_FR.md", *(ROOT / name / "README_FR.md" for name in MODELS)]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if "://" in link or link.startswith("#"):
                continue
            target = unquote(link.split("#", 1)[0]).strip("<>")
            require((path.parent / target).exists(), f"Lien local absent dans {path.name} : {link}")
    require((ROOT / "PLAN_INITIAL_REFERENCE.md").is_file(), "Archive du plan initial absente")
    print("OK — Quinze folds planifiés à zéro époque, calendrier et liens des documents vérifiés.")


def main():
    manifest = read_json(ROOT / "entrees_manifest.json")
    budget = read_json(ROOT / "budget_24h.json")
    check_inputs(manifest)
    check_recipes(manifest, budget)
    check_schedule_and_docs(budget)
    print("Aucun téléchargement, rendu DICOM, notebook, entraînement ou appel GPU exécuté.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERREUR — {error}", file=sys.stderr)
        sys.exit(1)
