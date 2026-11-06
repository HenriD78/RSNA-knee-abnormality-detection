# Préparation et publication GitHub

Le dépôt local est initialisé sur `main`. Aucun commit, dépôt distant ou envoi
vers GitHub n’est créé par cette préparation.

Vérifications locales du 5 novembre 2026 : **82 tests réussis**, **16 contrôles
avec artefacts locaux ignorés**, **26 notebooks sans sortie d’exécution**.
Les tests passent également dans une copie limitée aux fichiers publiables,
sans données, poids ni paquets hors ligne. L’exécution de la CI sur GitHub
reste à confirmer après publication.

## Avant le premier commit

```bash
python scripts/check_repository.py
python scripts/run_tests.py
git status --short
git add .
git diff --cached --stat
git diff --cached --name-only
```

La sélection doit contenir le code, les notebooks nettoyés, les guides et
les métriques agrégées. `.gitignore` exclut les données, rapports sources,
identifiants d’études, poids, wheels, ZIP, sorties, caches et environnements.
Les originaux des notebooks modifiés sont conservés dans
`local/notebook_backups/`. Les téléchargements de référence et essais restent
dans `local/`, également exclu.

Le contrôle des fichiers utilise les règles Git et inspecte aussi les fichiers
déjà indexés : ajouter un fichier ignoré avec `git add -f` ne contourne pas
les contrôles de données, de poids et de taille.

Les notebooks de référence complets ne sont pas proposés à la publication.
La provenance des extraits adaptés est indiquée dans
[NOTICE.md](../NOTICE.md). Aucune licence globale n’a été choisie pour le projet.

## Premier commit et dépôt distant

Après revue de la sélection, les commandes suivantes permettent de créer le
premier commit et de relier un dépôt GitHub vide créé sur son compte :

```bash
git commit -m "Prepare RSNA knee MRI project for GitHub"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Remplacer `OWNER/REPOSITORY` par le dépôt choisi. Ces commandes de publication
n’ont pas été exécutées. La première exécution GitHub Actions devra confirmer
les vérifications dans l’environnement distant.

Suggestion de description : « Knee MRI multilabel classification with weak
supervision, grouped cross-validation and offline PyTorch inference. »
Thèmes possibles : `pytorch`, `medical-imaging`, `kaggle`, `dicom`,
`weak-supervision`, `multilabel-classification`.

## Correspondance avec l’ancienne organisation

| Ancien emplacement | Nouvel emplacement |
|---|---|
| `3rd Version Finetuned Ensemble/` | `finetuned_ensemble/` |
| Notebooks opérationnels dans son `dist/` | `finetuned_ensemble/notebooks/` |
| Cellules de récupération dans son `dist/` | `finetuned_ensemble/patches/` |
| `Data Exploration/` | `notebooks/exploration/` |
| `MAE/` | `notebooks/mae/` |
| `report_weak_labeling.ipynb` | `notebooks/report_weak_labeling.ipynb` |
| Documents méthodologiques à la racine | `docs/methodology/` |
| Scripts `scratch/verify_*.py` | `scripts/verification/` |
| Autres fichiers de `scratch/` | `local/scratch/` |
| Notebooks externes et schémas associés | `local/reference_notebooks/` |

Les chemins internes de `baseline_candidate/`, les données et les artefacts
d’entraînement restent à leur emplacement pour conserver leur provenance.
