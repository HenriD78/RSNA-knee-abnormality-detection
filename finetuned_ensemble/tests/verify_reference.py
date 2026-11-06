"""Reproduire la métrique gold existante, sans calcul de nouveau modèle."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsna3.common import LABELS, ROOT, UID, input_tables, validate_predictions
from rsna3.evaluate import aucs, macro, paired_bootstrap


def verify():
    table, _ = input_tables(ROOT.parent)
    gold = table[table.is_gold]
    baseline_path = ROOT.parent / "baseline_candidate/online_kaggle/rsna_reports/reports/cohort_predictions/gold/candidate05.csv"
    baseline = pd.read_csv(baseline_path, dtype={UID: str})
    probability = validate_predictions(baseline, gold[UID].tolist())[LABELS].to_numpy(float)
    truth = gold[LABELS].to_numpy(float)
    score = macro(aucs(truth, probability))
    if abs(score - .7302461374891095) > 1e-12:
        raise ValueError(f"La métrique ne reproduit pas la référence : {score}")
    result = paired_bootstrap(truth, gold.group_id.to_numpy(),
                              {"baseline_frozen": probability, "identical": probability}, repetitions=2000)
    if result["differences"]["identical"]["ci95"] != [0., 0.]:
        raise ValueError("Le bootstrap apparié de deux prédictions identiques doit être nul")
    print(json.dumps({"reference_macro_auc": score, "bootstrap_valid": result["valid"],
                      "identical_ci95": [0., 0.], "new_models_evaluated": False}, indent=2))


if __name__ == "__main__":
    verify()
