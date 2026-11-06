**Implémentation disponible** : [notebook de ce pilier](entrainer_dinov2_small.ipynb), moteur commun `../rsna3/` et [guide de lancement](../README_FR.md). La recette exécutable est produite par le pilote ; `experience_planifiee.json` conserve les paramètres initiaux. Aucun nouveau checkpoint n'est annoncé avant le calcul Kaggle.

# Pilier 3 — DINOv2-Small à 224

**Recette implémentée, priorité de calcul 3, plafond de quota 4 h.**
Voir le [plan](../../docs/methodology/plan_entrainement_3_piliers.md), le
[protocole commun](../PROTOCOLE_COMMUN_FR.md) et les
[paramètres précis](experience_planifiee.json). Aucun checkpoint fine-tuné
n'est encore présent dans ce dossier.

## Ce que nous voulons tester

Adapter un ViT auto-supervisé pour compléter les deux lecteurs CNN. Le modèle
générique `vit_small_patch14_dinov2.lvd142m` existe déjà dans
`baseline_candidate/assets/dino/`. Notre lecteur gelé obtenait environ 0,7182
de macro AUC gold OOF. Une autre architecture n'assure pas à elle seule un gain
d'ensemble : les corrélations et différences appariées seront examinées.

## Recette prévue

| Élément | Choix initial |
|---|---|
| Entrée | Triplets DICOM, six slots masqués, champ physique commun |
| Résolution | 224, grille 16 x 16 de patches 14 ; interpolation des positions spatiales |
| Normalisation | Moyennes 0,485/0,456/0,406 ; écarts-types 0,229/0,224/0,225 |
| Caractéristique | Token CLS de dimension 384, vérifié au forward de contrôle |
| Fenêtres au train | Deux par slot présent ; toutes celles du cache figé en évaluation |
| Tête | Projection 128, slot + position physique, transformer d'étude 2 couches / 4 têtes |
| Sortie | Douze pools d'attention, puis douze logits ; dropout 0,1 |
| Dégel | Deux derniers blocs et norme finale de l'encodeur |
| Époques visées | 4 : une tête, puis trois avec adaptation de l'encodeur |
| LR encodeur / tête | 3e-6 / 1e-4 |
| Optimisation | AdamW, weight decay 0,01, AMP FP16, clipping 1 |
| Batch et CV | Une étude, accumulation huit ; cinq folds maîtres, seed 42 + fold |

Vérifier séparément les positions **spatiales des patches dans une image** et
la position **physique de la fenêtre dans la série**. Le transformer d'étude
n'utilise pas de position arbitraire fondée sur l'ordre du tableau. Permuter
conjointement images, masques, slots et positions doit préserver les prédictions
en mode évaluation.

Le pilote détermine les époques définitives avant les résultats externes. Si le
minimum d'une époque tête et une époque adaptée ne tient pas pour cinq folds,
déclarer le pilier non réalisable dans son budget ; ne pas inventer des poids
manquants et ne pas publier une couverture OOF partielle comme complète.

## Livrables à produire sur Kaggle

Cinq `final_fold0.safetensors` à `final_fold4.safetensors`, `oof_dinov2_small.csv`
(4 407 UID), métriques séparées gold/weak, historique, manifestes et profil
d'inférence. Le [suivi](../suivi_folds.csv) contient actuellement des tâches
planifiées. Aucun teacher RSNA public, Swin de remplacement, distillation,
miroir ou TTA n'est ajouté à la première expérience.
