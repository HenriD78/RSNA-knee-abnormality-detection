# Unified Cross-Validation & Fold Splitting Strategy

This document outlines the standard cross-validation (CV) splitting strategy for the **RSNA Knee Abnormality Detection** challenge. 

A single, unified 5-fold split shared across all model blocks (DINOv2, RadImageNet ResNet-50, CoAtNet, MAE, etc.) is **mandatory** to enable leak-free local validation, Out-Of-Fold (OOF) probability calibration, and multi-backbone ensembling.

---

## 1. Why a Unified Split is Mandatory

In complex medical imaging pipelines, models are ensembled across diverse architectures:

```
                      Master train_folds.csv (5 Folds)
                      ┌───────────────┼───────────────┐
                      ▼               ▼               ▼
                 [Block 1]        [Block 2]       [Block 3]
                  DINOv2         RadImageNet       CoAtNet
                (5 Models)       (5 Models)       (5 Models)
                      │               │               │
                      ▼               ▼               ▼
                   OOF Preds       OOF Preds       OOF Preds
                      └───────────────┼───────────────┘
                                      ▼
                            Unified OOF Blending
                    (Optimize weights without LB probing)
```

1. **Leak-Free Ensemble Evaluation**:
   When all backbones use the exact same fold assignments, **Fold $k$** of every backbone is trained on the exact same 80% of studies and evaluated on the exact same 20%. Combining their predictions on Fold $k$ yields an authentic validation score.
2. **Local Blend Weight Optimization**:
   Blend weights (e.g., $w_{\text{DINO}} \times P_{\text{DINO}} + w_{\text{CoAtNet}} \times P_{\text{CoAtNet}}$) can be optimized locally (per finding) using scipy or logistic regression without overfitting to the public leaderboard.
3. **Reproducibility**:
   Decoupling fold generation into a standalone file (`train_folds.csv`) guarantees that data pipelines, feature extractors, and downstream heads always operate on identical splits.

---

## 2. RSNA Knee Data Hierarchy & Splitting Principles

### The Data Hierarchy
The RSNA Knee dataset has a 3-level hierarchy:

$$\text{Patient} \implies \text{Study (\texttt{StudyInstanceUID})} \implies \text{Series (\texttt{SeriesInstanceUID})} \implies \text{Slices (DICOM scans)}$$

* **`train.csv`**: Contains **4,407 rows**, exactly **1 row per unique `StudyInstanceUID`**.
* **12 Multi-label Targets**:
  `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
* **Series / Slices**: Each study contains multiple imaging planes (Sagittal, Coronal, Axial) and dozens of 2D slices.

### The Two Non-Negotiable Requirements

| Requirement | Threat Addressed | Method |
| :--- | :--- | :--- |
| **1. Group by Study** | **Data Leakage**: Slices from the same knee must never appear in both train and validation sets. | Split strictly at the `StudyInstanceUID` level. All series/slices inherit the study's fold. |
| **2. Multi-label Stratification** | **Class Imbalance**: Some findings (e.g. `Fracture`, `Contusion`) are rare. Random splitting leads to fold skew. | Use **`MultilabelStratifiedKFold`** to balance all 12 binary labels simultaneously. |

---

## 3. Recommended Splitting Algorithm: `MultilabelStratifiedKFold`

Standard scikit-learn `StratifiedKFold` only supports a single target column. Since studies exhibit multiple concurrent abnormalities, we use **iterative stratification** via `iterstrat.ml_stratifiers.MultilabelStratifiedKFold`.

Iterative stratification preserves both individual label positive rates and frequent label co-occurrence patterns across all 5 folds.

---

## 4. Production Script to Generate `train_folds.csv`

Run the following script once to produce the master `train_folds.csv` file:

```python
"""
generate_folds.py
Generates a master 5-fold multilabel stratified split for RSNA Knee Abnormality Detection.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Configuration
N_FOLDS = 5
SEED = 42
DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "train_folds.csv"

# 12 Target Findings
TARGET_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def create_folds():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"Loaded {len(train_df):,} training studies.")

    # Format multi-label matrix (fill unlabeled/missing values with 0 for splitting purposes)
    Y = train_df[TARGET_COLS].fillna(0).to_numpy(dtype=np.float32)

    # Instantiate MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )

    train_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(mskf.split(train_df, Y)):
        train_df.loc[val_idx, "fold"] = fold

    # Verification Checks
    assert (train_df["fold"] >= 0).all(), "Error: unassigned studies found!"
    assert (
        train_df["fold"].nunique() == N_FOLDS
    ), f"Error: expected {N_FOLDS} folds."

    print("\n--- Fold Distribution Summary ---")
    print(train_df["fold"].value_counts().sort_index())

    print("\n--- Positive Rates per Fold (%) ---")
    stats = []
    for f in range(N_FOLDS):
        val_subset = train_df[train_df["fold"] == f]
        pos_rates = (val_subset[TARGET_COLS].mean() * 100).round(2).to_dict()
        pos_rates["Fold"] = f
        pos_rates["Studies"] = len(val_subset)
        stats.append(pos_rates)

    stat_df = pd.DataFrame(stats).set_index("Fold")
    print(stat_df[["Studies"] + TARGET_COLS[:6]])
    print(stat_df[TARGET_COLS[6:]])

    # Save minimal lookup and full annotated table
    train_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved master folds to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_folds()
```

---

## 5. Integrating `train_folds.csv` into Model Pipelines

Regardless of the input representation (individual 2D slices, 3-slice sliding windows, or full 3D series volumes), all pipelines load data by referencing the master `train_folds.csv`.

### A. Slice / Series Level DataLoaders

```python
import pandas as pd

# 1. Load series metadata and master folds
series_df = pd.read_csv("data/train_series.csv")
folds_df = pd.read_csv("data/train_folds.csv")[["StudyInstanceUID", "fold"]]

# 2. Merge fold assignment
series_df = series_df.merge(folds_df, on="StudyInstanceUID", how="left")

# 3. Create PyTorch datasets for Fold K
VAL_FOLD = 0
train_series = series_df[series_df["fold"] != VAL_FOLD]
val_series = series_df[series_df["fold"] == VAL_FOLD]
```

### B. Out-Of-Fold (OOF) Prediction Export

Each backbone should export an OOF prediction CSV with the identical row index as `train_folds.csv`:

```python
# Save backbone OOF predictions
oof_df = pd.DataFrame(oof_predictions, columns=TARGET_COLS)
oof_df["StudyInstanceUID"] = train_df["StudyInstanceUID"]
oof_df["fold"] = train_df["fold"]
oof_df.to_csv(f"outputs/{MODEL_NAME}_oof.csv", index=False)
```

---

## 6. How Backbones are Blended Using Unified OOFs

With consistent folds, blending probabilities across backbones is trivial and leak-free:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 1. Load OOF predictions from each backbone
oof_dino = pd.read_csv("outputs/dinov2_oof.csv")[TARGET_COLS].values
oof_rad = pd.read_csv("outputs/radimagenet_oof.csv")[TARGET_COLS].values
oof_coat = pd.read_csv("outputs/coatnet_oof.csv")[TARGET_COLS].values
ground_truth = pd.read_csv("data/train_folds.csv")[TARGET_COLS].values

# 2. Compute individual model AUCs
for name, preds in [
    ("DINOv2", oof_dino),
    ("RadImageNet", oof_rad),
    ("CoAtNet", oof_coat),
]:
    auc = roc_auc_score(ground_truth, preds, average="macro")
    print(f"{name} Macro AUC: {auc:.4f}")

# 3. Evaluate unified ensemble blend
blended_oof = 0.40 * oof_dino + 0.30 * oof_rad + 0.30 * oof_coat
ensemble_auc = roc_auc_score(ground_truth, blended_oof, average="macro")
print(f"Ensemble Macro AUC: {ensemble_auc:.4f}")
```

### Finding-Specific Weights Optimization
Because the RSNA metric is **Macro ROC-AUC over 12 findings**, each finding can use customized weights without risk of overfitting:

$$P_{i}^{\text{final}} = w_{1}^{(i)} P_{i}^{\text{DINO}} + w_{2}^{(i)} P_{i}^{\text{RadImageNet}} + w_{3}^{(i)} P_{i}^{\text{CoAtNet}} \quad \text{for } i \in \{1 \dots 12\}$$

Since each finding's AUC depends solely on the relative ordering of its scores, weights can be tuned per finding on the global OOF table using `scipy.optimize`.

---

## 7. Summary Checklist

- [x] **Split Level**: Split strictly on `StudyInstanceUID` (1 row per study in `train.csv`).
- [x] **Stratification**: Use `MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- [x] **Persistence**: Save once to `data/train_folds.csv` and keep static across all experiments.
- [x] **Data Loading**: Merge `fold` into series/slice loaders via `StudyInstanceUID`.
- [x] **OOF Tracking**: Save full Out-Of-Fold probability matrices for every backbone.
- [x] **Ensembling**: Optimize blend weights directly on combined OOF predictions.
