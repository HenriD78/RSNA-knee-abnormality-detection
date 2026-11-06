# Résultats et portée des mesures

Les chiffres présentés sont des résultats conservés dans le projet. La
préparation GitHub ne relance aucun entraînement ni benchmark Kaggle.

## Classification

| Expérience | Macro AUC | Population et provenance |
|---|---:|---|
| MAE, moyenne des caractéristiques | 0,574131 | 58 gold, cinq folds ; run local `pilot_20260924T083354_091484Z`, `comparison.csv` |
| Baseline gelée `candidate05` | 0,730246 | 58 gold hors fold ; [rapport agrégé conservé](frozen_baseline_evaluation.json) |
| Ensemble affiné, mélange principal | 0,68957 | Gold ; résultat rapporté dans le guide opérationnel historique, rapport brut d’évaluation absent de la sélection publique |

Le mélange `candidate05` est fixé avant l’évaluation. Les variantes `candidate10`
et `candidate15` restent exploratoires, même lorsque leur score gold est plus
élevé. L’intervalle bootstrap à 95 % de la différence entre `candidate05` et
`baseline3` est environ **[−0,00163 ; +0,00241]** : il ne démontre pas un gain net.

Les valeurs MAE évaluent les prédictions hors fold des classifieurs, pas une
mesure indépendante de leur ensemble final. La comparaison MAE/baseline/ensemble
n’est pas une ablation contrôlée : prétraitement et modèles diffèrent.

## Inférence DICOM

Deux journaux de mesures Kaggle sur les mêmes 50 études sont conservés :

- [Traitement série](../../finetuned_ensemble/verification/kaggle_serial_inference_20261021.json) : 37,038 s/étude.
- [Traitement parallèle](../../finetuned_ensemble/verification/kaggle_parallel_inference_20261021.json) : 12,491 s/étude, quatre processus et lots de quatre, cache non réutilisé.

La durée par étude diminue d’un facteur d’environ **2,97** dans cette mesure.
Le scénario de 1 300 études donne **5,42 h** avec marge de 20 %. C’est une
projection, pas une mesure du test caché. Les poids et le rendu ne changent
pas entre ces deux chemins ; l’empreinte du helper figure dans le journal.

## Limites d’interprétation

- 58 études gold donnent une estimation fragile et des cibles parfois rares.
- Les rapports identiques restent groupés, mais les CSV ne prouvent pas l’indépendance des patients.
- L’extracteur faible a été audité sur les gold ; la provenance des cibles n’est donc pas entièrement indépendante.
- L’AUC sur les 4 349 études weak mesure l’accord avec les heuristiques, pas avec une vérité annotée indépendante.
- Aucun classement final ni score public ou privé de ce projet n’est établi par les fichiers présents.

Les rapports de vérification historiques décrivent l’état au moment de leur
création. Les empreintes d’anciens notebooks exécutés ne décrivent pas les
notebooks nettoyés pour GitHub. Le contrôle actuel est
`python scripts/check_repository.py` ; le code du runtime `rsna3/` et le
helper d’inférence conservent leurs empreintes d’origine.
