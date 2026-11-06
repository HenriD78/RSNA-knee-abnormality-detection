**Implémentation disponible** : [notebook de ce pilier](entrainer_convnext_small.ipynb), moteur commun `../rsna3/` et [guide de lancement](../README_FR.md). La recette exécutable est produite par le pilote ; `experience_planifiee.json` conserve les paramètres initiaux. Aucun nouveau checkpoint n'est annoncé avant le calcul Kaggle.

# Pilier 2 — ConvNeXt-Small à 224

**Recette implémentée, priorité de calcul 1, plafond de quota 6 h.**
Voir le [plan](../../docs/methodology/plan_entrainement_3_piliers.md), le
[protocole commun](../PROTOCOLE_COMMUN_FR.md) et les
[paramètres précis](experience_planifiee.json). Aucun checkpoint fine-tuné
n'est encore présent dans ce dossier.

## Pourquoi commencer par lui

Notre Small gelé est actuellement le meilleur lecteur individuel, environ
0,7293 de macro AUC gold OOF. Il fournit un premier contrôle utile de l'adaptation
et bénéficie du plus grand budget. L'initialisation est notre poids générique
`convnext_small.fb_in22k_ft_in1k` dans `baseline_candidate/assets/small/`.
Cette nouvelle variante ne porte pas le nom R6/R7 : ces modèles publics ont
leurs propres poids, crops, entraînements et conventions.

## Recette prévue

| Élément | Choix initial |
|---|---|
| Entrée | Triplets DICOM, six slots masqués, champ physique commun de 153,6 mm |
| Résolution | Rendu 384 réduit à 224 en bicubique, sans second center crop |
| Normalisation | Moyennes 0,485/0,456/0,406 ; écarts-types 0,229/0,224/0,225 |
| Fenêtres au train | Deux par slot présent, douze au maximum par étude |
| Fenêtres à l'évaluation | Toutes celles du cache figé |
| Tête | Projection 128, slot + position physique, douze pools d'attention, dropout 0,1 |
| Dégel | Dernier stage de l'encodeur |
| Époques visées | 6 : une tête, puis cinq avec adaptation de l'encodeur |
| LR encodeur / tête | 1e-5 / 3e-4 |
| Optimisation | AdamW, weight decay 0,01, AMP FP16, clipping 1 |
| Batch | Une étude, accumulation huit ; conserver les confiances sur le batch effectif |
| CV | Cinq folds maîtres, seed 42 + fold ; aucune sélection sur le fold externe |

Le pilote peut modifier les époques pour tous les folds avant toute évaluation
externe, selon le débit mesuré. Garder un seul jeu de paramètres pour les cinq
folds. Ne pas démarrer une recherche de crops ou d'hyperparamètres en cours d'OOF.

## Livrables à produire sur Kaggle

Cinq `final_fold0.safetensors` à `final_fold4.safetensors`, `oof_convnext_small.csv`
(4 407 UID), métriques gold et weak distinctes, historique, manifestes et coût
d'inférence. Ces fichiers restent à produire. Le
[suivi initial](../suivi_folds.csv) contient uniquement les tâches planifiées.

Les joint anchors appris, state stackers, miroir anatomique et TTA ne sont pas
activés. Un vrai localiseur ne pourra être ajouté qu'avec poids et rendu train/test
compatibles, contrôlés avant le gel de l'expérience ; un simple crop centré
ne doit pas être présenté comme un joint anchor.
