from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .common import (LABELS, SPECS, UID, atomic_csv, atomic_json, digest, input_tables,
                     read_json, require, sha256, validate_predictions)


def ranks(values):
    values = np.asarray(values, float)
    require(values.ndim == 2 and len(values) > 0 and np.isfinite(values).all(), "Rangs : tableau vide/invalide")
    # Average ties; (rank-1)/(N-1), singleton = 0.5. Entire cohort, never a GPU batch.
    return (rankdata(values, method="average", axis=0) - 1) / (len(values) - 1) if len(values) > 1 else np.full_like(values, .5)


def blend(predictions, weights=None):
    weights = weights or {"coatnet2": 1 / 3, "convnext_small": 1 / 3, "dinov2_small": 1 / 3}
    require(set(predictions) == set(weights) and abs(sum(weights.values()) - 1) < 1e-10, "Trois lecteurs complets et poids fixes requis")
    require(len({np.asarray(v).shape for v in predictions.values()}) == 1, "Tailles de prédictions différentes")
    return ranks(sum(weights[name] * ranks(values) for name, values in predictions.items()))


def aucs(truth, prediction):
    result = []
    for j in range(len(LABELS)):
        known = np.isfinite(truth[:, j])
        y, p = truth[known, j], prediction[known, j]
        positive, negative = int((y == 1).sum()), int((y == 0).sum())
        if positive and negative:
            value = (rankdata(p)[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
            result.append(float(value))
        else:
            result.append(None)
    return result


def macro(values):
    return float(np.mean(values)) if all(v is not None for v in values) else None


def paired_bootstrap(truth, groups, predictions, reference="baseline_frozen", repetitions=2000):
    require(reference in predictions, "Référence bootstrap absente")
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(42)
    deltas = {name: [] for name in predictions if name != reference}
    valid, attempts = 0, 0
    while valid < repetitions and attempts < repetitions * 100:
        attempts += 1
        index = np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)])
        baseline = macro(aucs(truth[index], predictions[reference][index]))
        if baseline is None:
            continue
        for name in deltas:
            value = macro(aucs(truth[index], predictions[name][index]))
            require(value is not None, "Masques de bootstrap incohérents")
            deltas[name].append(value - baseline)
        valid += 1
    require(valid == repetitions, "Pas assez de réplications valides ; ne pas annoncer 2 000 tirages")
    return {"valid": valid, "attempts": attempts, "reference": reference,
            "differences": {name: {"mean": float(np.mean(v)), "ci95": np.quantile(v, [.025, .975]).tolist(),
                                    "fraction_positive": float(np.mean(np.asarray(v) > 0))} for name, v in deltas.items()},
            "interpretation_fr": "Conditionnel aux OOF fixés ; aucune sélection de recette ou de poids."}


def collect_model(output, model_id, table):
    frames, recipe = [], None
    for fold in range(5):
        folder = Path(output) / model_id / f"fold{fold}"
        meta = read_json(folder / "completed.json")
        require(meta["status"] == "completed" and meta["contract"]["fold"] == fold, "Fold non terminé")
        require(meta["contract"]["labels"] == LABELS, "Ordre des findings incorrect")
        require(sha256(folder / f"final_fold{fold}.safetensors") == meta["checkpoint_sha256"] and
                sha256(folder / "oof.csv") == meta["oof_sha256"], "Sorties de fold altérées")
        if recipe is None:
            recipe = meta["contract"]["config_hash"]
        require(recipe == meta["contract"]["config_hash"] == digest(meta["config"]), "Recettes différentes entre folds")
        frame = pd.read_csv(folder / "oof.csv", dtype={UID: str, "group_id": str})
        expected = table[table.fold == fold]
        validate_predictions(frame, expected[UID].tolist())
        aligned = frame.set_index(UID).loc[expected[UID]]
        for key in ("fold", "group_id", "is_gold"):
            require(np.array_equal(aligned[key], expected[key]), f"Provenance OOF incohérente : {key}")
        require((frame.provenance == "generic_initialization_out_of_fold_final_epoch").all(), "OOF hors protocole")
        require((frame.checkpoint_sha256 == meta["checkpoint_sha256"]).all(), "Poids OOF différents")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    validate_predictions(result, table[UID].tolist())
    return result


def evaluate(project, output, report, baseline_root=None, repetitions=2000):
    table, _ = input_tables(project)
    report = Path(report)
    report.mkdir(parents=True, exist_ok=True)
    oof = {name: collect_model(output, name, table) for name in SPECS}
    for name, frame in oof.items():
        # Saved Kaggle outputs are mounted read-only under /kaggle/input.
        atomic_csv(report / f"oof_{name}.csv", frame)
    metrics, correlations, fold_metrics, errors = [], [], [], []
    summary = {"selection": "fixed_in_advance", "cohorts": {},
               "limitations_fr": ["Groupes de rapports ; indépendance patient non établie.",
                                  "Les weak ont été audités contre les gold ; proxy heuristique distinct."]}
    baseline_root = Path(baseline_root) if baseline_root else Path(project) / "baseline_candidate/online_kaggle/rsna_reports/reports/cohort_predictions"
    for cohort, mask in (("gold", table.is_gold), ("weak_proxy", ~table.is_gold)):
        subset = table[mask].reset_index(drop=True)
        ids = subset[UID].tolist()
        truth = subset[LABELS].to_numpy(float)
        if cohort == "weak_proxy":
            conf = subset[[l + "__conf" for l in LABELS]].to_numpy(float)
            truth = np.where(np.isfinite(truth) & (conf > 0), (truth > .5).astype(float), np.nan)
        raw = {name: validate_predictions(frame[frame[UID].isin(ids)], ids)[LABELS].to_numpy(float) for name, frame in oof.items()}
        predictions = {**raw, "ensemble_equal": blend(raw),
                       "ensemble_exploratory_20_60_20": blend(raw, {"coatnet2": .2, "convnext_small": .6, "dinov2_small": .2})}
        baseline = pd.read_csv(baseline_root / cohort / "candidate05.csv", dtype={UID: str})
        predictions["baseline_frozen"] = validate_predictions(baseline, ids)[LABELS].to_numpy(float)
        summary["cohorts"][cohort] = {}
        for name, values in predictions.items():
            auc = aucs(truth, values)
            summary["cohorts"][cohort][name] = {"macro_auc": macro(auc), "defined_findings": sum(v is not None for v in auc), "studies": len(ids)}
            for j, label in enumerate(LABELS):
                metrics.append({"cohort": cohort, "model": name, "finding": label, "auc": auc[j],
                                "known": int(np.isfinite(truth[:, j]).sum()), "positive": int((truth[:, j] == 1).sum())})
            atomic_csv(report / cohort / f"{name}.csv", pd.DataFrame({UID: ids, **dict(zip(LABELS, values.T))}))
            if cohort == "gold":
                for fold in range(5):
                    m = subset.fold.to_numpy() == fold
                    per_fold = aucs(truth[m], values[m])
                    fold_metrics.extend({"model": name, "fold": fold, "finding": l, "auc": a} for l, a in zip(LABELS, per_fold))
        for left, right in itertools.combinations(predictions, 2):
            for j, label in enumerate(LABELS):
                x, z = predictions[left][:, j], predictions[right][:, j]
                corr = float(spearmanr(x, z).statistic) if np.ptp(x) > 0 and np.ptp(z) > 0 else None
                correlations.append({"cohort": cohort, "left": left, "right": right, "finding": label, "spearman": corr})
                known = np.isfinite(truth[:, j])
                a, b = x[known] - truth[known, j], z[known] - truth[known, j]
                errors.append({"cohort": cohort, "left": left, "right": right, "finding": label,
                               "error_pearson": float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.ptp(a) and np.ptp(b) else None})
        if cohort == "gold":
            summary["bootstrap"] = paired_bootstrap(truth, subset.group_id.to_numpy(), predictions, repetitions=repetitions)
    atomic_csv(report / "auc_by_finding.csv", pd.DataFrame(metrics))
    atomic_csv(report / "gold_auc_by_fold.csv", pd.DataFrame(fold_metrics))
    atomic_csv(report / "rank_correlations.csv", pd.DataFrame(correlations))
    atomic_csv(report / "error_correlations.csv", pd.DataFrame(errors))
    atomic_json(report / "evaluation.json", summary)
    return summary
