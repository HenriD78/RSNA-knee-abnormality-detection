# Ensemble affiné : CoAtNet, ConvNeXt et DINOv2

Ce pipeline entraîne trois familles de modèles sur cinq folds communs, avec
rendu physique DICOM, vues 2,5D et agrégation par étude. Il comprend le
profilage, l’admission selon le budget mesuré, la reprise des entraînements,
l’évaluation hors fold, l’export et l’inférence Kaggle hors ligne.

Le code d’entraînement dans `rsna3/` est conservé sans modification depuis
les expériences. Le helper parallèle reste distinct afin de préserver les
empreintes de reprise. Les paquets et sorties Kaggle sont des artefacts
locaux exclus de Git ; un clone contient les sources et les notebooks.

## État des résultats

Le journal opérationnel rapporte une macro AUC gold de **0,68957** pour le
mélange principal, contre **0,73025** pour la baseline gelée. La baseline reste
le candidat principal. Aucun nouveau score de leaderboard n’est annoncé.

Le benchmark Kaggle sur 50 études mesure **12,491 s/étude** avec quatre
processus contre **37,038 s/étude** en série. Le scénario de 1 300 études
donne 5,42 heures avec marge de 20 % ; la durée du test caché reste inconnue.
Les journaux sont dans [verification/](verification/), les limites et la
provenance dans [les résultats du projet](../docs/results/README.md).

## Notebooks

Les huit notebooks de départ restent avec les recettes : les cinq étapes
numérotées à la racine et les trois notebooks `entrainer_*.ipynb` par modèle.
Le parcours opérationnel corrigé se trouve dans [notebooks/](notebooks/) :

| Étape | Notebook |
|---|---|
| Préparation du cache CPU | [00_preparer_cache_cpu_corrige.ipynb](notebooks/00_preparer_cache_cpu_corrige.ipynb) |
| Profilage ConvNeXt | [01_pilote_convnext_premier.ipynb](notebooks/01_pilote_convnext_premier.ipynb) |
| Profilage CoAtNet et DINOv2 | [01_pilotes_coatnet_dinov2.ipynb](notebooks/01_pilotes_coatnet_dinov2.ipynb) |
| Entraînement de l’ensemble | [02_entrainer_ensemble_pilote.ipynb](notebooks/02_entrainer_ensemble_pilote.ipynb) |
| Reprise des folds incomplets | [02_reprendre_ensemble.ipynb](notebooks/02_reprendre_ensemble.ipynb) |
| Évaluation et export | [03_evaluer_exporter_cpu.ipynb](notebooks/03_evaluer_exporter_cpu.ipynb) |
| Benchmark série | [03b_mesurer_inference_gpu.ipynb](notebooks/03b_mesurer_inference_gpu.ipynb) |
| Benchmark parallèle | [03c_mesurer_inference_parallele.ipynb](notebooks/03c_mesurer_inference_parallele.ipynb) |
| Inférence parallèle | [04_soumettre_parallele.ipynb](notebooks/04_soumettre_parallele.ipynb) |

Le nom historique `cpu` de l’export ne décrit pas l’environnement requis par
les expériences existantes : conserver les versions PyTorch de l’entraînement
sur Kaggle, telles que vérifiées par les contrats du pipeline.

## Préparer les entrées

Depuis la racine du projet, avec les données, tables préparées et poids locaux :

```bash
python finetuned_ensemble/build_kaggle_bundle.py --help
python finetuned_ensemble/tests/verify_notebooks.py
python -m pytest finetuned_ensemble/tests -m "not local_assets" -q
```

`build_kaggle_bundle.py` assemble les sources, les tables et les poids ; il ne
génère pas de notebook. Les wheels Linux sont décrites dans
`requirements-kaggle-offline.txt` et téléchargées lors de la construction
du paquet, pas pendant l’inférence. Les DICOM ne sont pas inclus dans Git.

Sur Kaggle, attacher la compétition et les paquets privés nécessaires,
choisir l’accélérateur demandé par le notebook et désactiver Internet.
Les valeurs de quota et les chemins d’entrées doivent refléter la session
réelle. Les valeurs du journal historique ne valent pas configuration actuelle.

## Contrats et documentation

- [Protocole commun](PROTOCOLE_COMMUN_FR.md) : séparation gold/weak, folds, perte et budget.
- [Plan expérimental](../docs/methodology/plan_entrainement_3_piliers.md).
- [Plan initial archivé](PLAN_INITIAL_REFERENCE.md).
- [Vérifications historiques](VERIFICATIONS_FR.md).
- [Journal opérationnel détaillé](../docs/archive/finetuned_ensemble_operations_2026-10.md).
- [Correctifs autonomes](patches/) pour les anciens Inputs Kaggle immuables.

Les tests par défaut utilisent des entrées synthétiques. Les tests qui ouvrent
les poids génériques, les tables de compétition ou les paquets hors ligne
portent le marqueur `local_assets` et nécessitent `--run-local-assets`.
