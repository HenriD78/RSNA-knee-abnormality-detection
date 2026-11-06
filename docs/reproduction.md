# Reproduire le projet

## Vérification sans données privées

Installer `requirements/dev.txt` dans un environnement Python 3.12, puis lancer :

```bash
python scripts/check_repository.py
python scripts/run_tests.py
```

Le premier contrôle examine les fichiers destinés à Git, les notebooks,
les liens Markdown, les fichiers volumineux et les formes courantes de secrets.
Il ne remplace pas une vérification humaine des données et des attributions.
Le second lance chaque suite dans un processus séparé : les runtimes Kaggle
et local exposent des modules de même nom et doivent être testés séparément.

Les tests `local_assets` nécessitent des données, des poids ou des archives
privés. Ils sont ignorés par défaut. Après avoir restauré ces éléments :

```bash
python scripts/run_tests.py --run-local-assets
```

## Baseline JPEG

1. Préparer les tables et images décrites dans [data/README.md](../data/README.md).
2. Exécuter [le notebook de supervision faible](../notebooks/report_weak_labeling.ipynb) pour produire `data/derived/report_labels_v2.csv`.
3. Suivre [GUIDE_FR.md](../baseline_candidate/GUIDE_FR.md), puis ouvrir [baseline_candidate.ipynb](../baseline_candidate/baseline_candidate.ipynb).
4. Passer d’`inspection` à `smoke`, sélectionner explicitement les étapes, puis préparer le parcours complet.

Le fichier `baseline_candidate/requirements.lock` est un relevé historique
de l’environnement macOS, pas une promesse d’installation portable. Il reste
utile pour identifier les versions des anciens artefacts ; le développement
et la CI utilisent `requirements/dev.txt`.

## MAE

Le [notebook d’entraînement](../notebooks/mae/mae_baseline.ipynb) commence par
un contrôle `smoke`. Le pilote complet nécessite `MODE = "pilot"` et
`ALLOW_PILOT = True`. Les scripts de vérification peuvent exécuter les
notebooks avec les données et checkpoints locaux :

```bash
python scripts/verification/verify_mae_notebook.py --cpu
python scripts/verification/verify_mae_submission.py
python scripts/verification/verify_offline_codecs.py
```

Ces commandes complètes ne font pas partie de la CI : elles nécessitent les
artefacts locaux. Le script de contrôle MAE lance un petit entraînement et
écrit un nouveau run ; celui de soumission vérifie le transfert du modèle existant.

## Ensemble DICOM sur Kaggle

Le [guide de l’ensemble](../finetuned_ensemble/README_FR.md) distingue les
notebooks de départ des notebooks opérationnels de reprise et d’inférence.
Les paquets se reconstruisent avec `build_kaggle_bundle.py` ; les poids,
tables et wheels doivent être fournis séparément. Ils ne sont pas dans Git.

Les contrats d’empreintes, versions, folds et recettes sont vérifiés à la
reprise. Le code `rsna3/` a été conservé tel quel pendant la réorganisation
pour préserver les contrats des entraînements existants.

## Limite de reproduction des JPEG

La conversion JPEG d’origine n’est pas disponible. L’audit de 25 paires locales
DICOM/JPEG a identifié une conversion très proche : normalisation min–max,
uint8, redimensionnement 224 × 224 avec `INTER_AREA`, encodage JPEG qualité 95.
L’erreur moyenne observée est d’environ 0,00138 sur l’échelle 0–255 ; ce n’est
pas une preuve d’identité sur tous les scanners. Voir le
[guide MAE](../submission/README.md) et le
[relevé de vérification](../submission/verification.json).

Le pipeline affiné possède son propre rendu physique DICOM ; il ne dépend
pas de la reproduction exacte de ces JPEG historiques.
