"""Comparaisons prédéfinies : aucune optimisation des poids sur les gold OOF."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import output_root
from .constants import LABELS, READERS, TINY_WEIGHTS, UID
from .data import load_prepared
from .features import open_store
from .io import read_json, sha256, write_csv, write_json
from .train import collect_oof


def align_predictions(frame, ids):
    """Vérifie les UID avant tout mélange ; une permutation des lignes est sans effet."""
    if frame[UID].duplicated().any() or set(frame[UID]) != set(ids) or len(ids) != len(set(ids)):
        raise ValueError("UID dupliqués, absents ou supplémentaires dans les prédictions.")
    result = frame.set_index(UID).loc[list(ids), LABELS].astype(float)
    values = result.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Les scores doivent être finis et compris dans [0,1].")
    return result


def ranks(frame):
    """Rangs percentiles sur toute la cohorte, par colonne, avec moyenne des égalités."""
    return frame[LABELS].rank(method="average", pct=True)


def build_blends(predictions):
    """Les coefficients sont fixés ; le dernier reranking ne modifie pas l'AUC."""
    if set(predictions) != set(READERS):
        raise ValueError("Les quatre lecteurs sont requis pour comparer les mélanges.")
    ids = predictions["coatnet"][UID].tolist()
    raw = {name: align_predictions(frame, ids) for name, frame in predictions.items()}
    baseline2 = .7 * ranks(raw["coatnet"]) + .3 * ranks(raw["small"])
    weight = pd.Series(TINY_WEIGHTS).reindex(LABELS)
    baseline3 = (1 - weight) * ranks(baseline2) + weight * ranks(raw["tiny"])
    result = {"baseline2": ranks(baseline2), "baseline3": ranks(baseline3)}
    for alpha in [.05, .10, .15]:
        result[f"candidate{int(round(100 * alpha)):02d}"] = ranks(
            (1 - alpha) * ranks(baseline3) + alpha * ranks(raw["dino"]))
    return {name: values.reset_index()[[UID] + LABELS] for name, values in result.items()}


def metrics(labels, predictions, cohort):
    """AUC undefined si une colonne n'a qu'une classe ; pas de macro trompeuse."""
    ids = labels[UID].tolist()
    aligned = align_predictions(predictions, ids)
    rows = []
    for label in LABELS:
        y = labels[label].to_numpy(dtype=float)
        known = np.isfinite(y)
        truth = y[known]
        defined = len(np.unique(truth)) == 2
        auc = roc_auc_score(truth, aligned[label].to_numpy()[known]) if defined else np.nan
        rows.append({"cohort": cohort, "finding": label, "studies": int(known.sum()),
                     "positive": int((truth == 1).sum()), "negative": int((truth == 0).sum()),
                     "auc": float(auc), "defined": bool(defined)})
    defined = all(row["defined"] for row in rows)
    summary = {"macro_auc": float(np.mean([r["auc"] for r in rows])) if defined else None,
               "defined_targets": sum(r["defined"] for r in rows), "studies": len(ids)}
    return rows, summary


def paired_bootstrap(labels, groups, baseline, candidate, repetitions=2000, seed=42):
    """Rééchantillonne les groupes gold ; mêmes tirages pour les deux prédictions.

    L'intervalle mesure l'incertitude conditionnelle aux prédictions OOF fixées.
    Il ne remplace ni un test indépendant ni une répétition de l'entraînement.
    """
    ids = labels[UID].tolist()
    y = labels[LABELS].to_numpy(dtype=float)
    a = align_predictions(baseline, ids).to_numpy()
    b = align_predictions(candidate, ids).to_numpy()
    group = groups.set_index(UID).loc[ids, "group_id"].to_numpy()
    unique = np.unique(group)
    members = {g: np.flatnonzero(group == g) for g in unique}
    rng = np.random.default_rng(seed)
    delta, attempts = [], 0
    while len(delta) < repetitions and attempts < max(1, repetitions * 5):
        attempts += 1
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[g] for g in chosen])
        if any(len(np.unique(y[index, j])) != 2 for j in range(len(LABELS))):
            continue
        delta.append(float(roc_auc_score(y[index], b[index], average="macro")
                           - roc_auc_score(y[index], a[index], average="macro")))
    if not delta:
        return {"requested": repetitions, "valid": 0, "attempts": attempts, "ci95": None}
    return {"requested": repetitions, "valid": len(delta), "attempts": attempts,
            "ci95": np.quantile(delta, [.025, .975]).tolist(),
            "fraction_positive": float(np.mean(np.asarray(delta) > 0)),
            "interpretation": "Exploratoire, conditionnel aux modèles OOF ; aucun choix automatique."}


def evaluate(cfg):
    _, tables = load_prepared(cfg)
    master = tables["train_folds"]
    master = master[master.active].sort_values(UID)
    predictions = {reader: collect_oof(cfg, reader) for reader in READERS}
    gold_ids = master.loc[master.is_gold, UID].tolist()
    weak_ids = master.loc[~master.is_gold, UID].tolist()
    if not gold_ids:
        raise ValueError("Aucune étude gold active à évaluer.")
    gold = tables["gold_labels"].set_index(UID).loc[gold_ids].reset_index()
    weak = tables["targets"].set_index(UID).loc[weak_ids].reset_index()
    for label in LABELS:
        valid = weak[label].notna() & (weak[label + "__conf"] > 0)
        weak[label] = np.where(valid, (weak[label] > .5).astype(float), np.nan)
    report = output_root(cfg) / "reports"
    all_rows, summary = [], {"profile": cfg["profile"], "weight_selection": "fixed_before_gold_evaluation"}
    cohort_blends = {}
    for cohort, truth in [("gold", gold), ("weak_proxy", weak)]:
        if truth.empty:
            continue
        ids = truth[UID].tolist()
        subset = {name: align_predictions(frame[frame[UID].isin(ids)], ids).reset_index()
                  for name, frame in predictions.items()}
        # Une cohorte gold est classée séparément : les weak n'altèrent pas ses rangs.
        # Les tables OOF originales couvrent plus d'UID : filtrer explicitement avant alignement strict.
        blends = build_blends(subset)
        cohort_blends[cohort] = blends
        summary[cohort] = {}
        for name, prediction in {**subset, **blends}.items():
            rows, result = metrics(truth, prediction, cohort)
            all_rows.extend([{**row, "model": name} for row in rows])
            summary[cohort][name] = result
            write_csv(report / "cohort_predictions" / cohort / f"{name}.csv", prediction)
        # Corrélation des rangs = diagnostic de redondance, pas preuve d'un gain.
        corr = []
        for j, left in enumerate(READERS):
            for right in list(READERS)[j + 1:]:
                for label in LABELS:
                    x = subset[left].set_index(UID).loc[ids, label]
                    z = subset[right].set_index(UID).loc[ids, label]
                    corr.append({"left": left, "right": right, "finding": label,
                                 "spearman": x.corr(z, method="spearman")})
        write_csv(report / f"rank_correlations_{cohort}.csv", pd.DataFrame(corr))
    rows = pd.DataFrame(all_rows)
    write_csv(report / "auc_by_finding.csv", rows)
    matrix = rows[rows.cohort == "gold"].pivot(index="finding", columns="model", values="auc")
    differences = pd.DataFrame({"finding": matrix.index,
                               "candidate05_minus_baseline3": matrix.candidate05 - matrix.baseline3})
    write_csv(report / "primary_delta_by_finding.csv", differences)
    fold_rows = []
    for fold in range(cfg["n_folds"]):
        fold_ids = master.loc[(master.fold == fold) & master.is_gold, UID].tolist()
        summary.setdefault("gold_folds", {})[str(fold)] = {"studies": len(fold_ids)}
        if fold_ids:
            truth = gold[gold[UID].isin(fold_ids)]
            for name, prediction in cohort_blends["gold"].items():
                scored, _ = metrics(truth, prediction[prediction[UID].isin(fold_ids)], "gold")
                fold_rows.extend([{**row, "fold": fold, "model": name} for row in scored])
    write_csv(report / "gold_auc_by_fold.csv", pd.DataFrame(fold_rows))
    if cfg["bootstrap_samples"]:
        blends = cohort_blends["gold"]
        summary["bootstrap_primary"] = paired_bootstrap(
            gold, master, blends["baseline3"], blends["candidate05"],
            cfg["bootstrap_samples"], cfg["seed"])
    summary["interpretation"] = ("PIPELINE CHECK ONLY" if cfg["profile"] == "smoke"
                                 else "Gold OOF exploratoire ; weak_proxy mesure l'accord avec les heuristiques.")
    write_json(report / "evaluation.json", summary)
    print(summary)
    return summary


def export_submissions(cfg):
    prepared, tables = load_prepared(cfg)
    ids = tables["test_ids"][UID].tolist()
    predictions = {}
    for reader in READERS:
        path = output_root(cfg) / "predictions/test" / f"{reader}.csv"
        receipt = read_json(path.with_suffix(".json"))
        store = open_store(cfg, reader, "test")
        expected = [sha256(output_root(cfg) / "checkpoints" / reader / f"fold{fold}/last.pt")
                    for fold in range(cfg["n_folds"])]
        if (receipt["prepared"] != prepared["signature"] or receipt["sha256"] != sha256(path)
                or receipt["checkpoint_hashes"] != expected
                or receipt["test_feature_identity"] != store.identity):
            raise ValueError("Prédictions test incompatibles ou modifiées.")
        predictions[reader] = pd.read_csv(path, dtype={UID: str})
        predictions[reader] = align_predictions(predictions[reader], ids).reset_index()
    destination = output_root(cfg) / "submissions"
    for name, frame in build_blends(predictions).items():
        write_csv(destination / f"submission_{name}.csv", frame)
    write_json(destination / "receipt.json", {"ids": ids, "columns": [UID] + LABELS,
               "primary": "submission_candidate05.csv", "baseline": "submission_baseline3.csv",
               "note": "Trois exemples locaux : vérification de format, aucune mesure leaderboard."})
    print(f"Soumissions distinctes créées dans {destination}; aucun gagnant choisi automatiquement.")
