**Implémentation disponible** : [notebook de ce pilier](entrainer_coatnet2.ipynb), moteur commun `../rsna3/` et [guide de lancement](../README_FR.md). La recette exécutable est produite par le pilote ; `experience_planifiee.json` conserve les paramètres initiaux. Aucun nouveau checkpoint n'est annoncé avant le calcul Kaggle.

# Pilier 1 — CoAtNet-2 à 384

**Recette implémentée, priorité de calcul 2, plafond de quota 5 h.**
Voir le [plan](../../docs/methodology/plan_entrainement_3_piliers.md), le
[protocole commun](../PROTOCOLE_COMMUN_FR.md) et les
[paramètres précis](experience_planifiee.json). Aucun checkpoint fine-tuné
n'est encore présent dans ce dossier.

## Ce que nous voulons tester

Adapter à l'IRM notre `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` générique, déjà
stocké dans `baseline_candidate/assets/coatnet/`. Ce ne sont pas les poids RSNA/OAI
Raptor du notebook public. Notre lecteur gelé actuel obtenait environ 0,7015
de macro AUC gold OOF ; le fine-tuning doit être mesuré sans promesse de score.

## Recette prévue

| Élément | Choix initial |
|---|---|
| Entrée | Triplets DICOM, six slots masqués, champ physique 153,6 mm |
| Résolution / normalisation | 384 ; moyenne 0,5 et écart-type 0,5 sur les trois canaux |
| Fenêtres au train | Une par slot présent, six au maximum par étude |
| Fenêtres à l'évaluation | Toutes les fenêtres du cache commun figé |
| Tête | Projection 128, slot + position physique, douze pools d'attention, dropout 0,1 |
| Dégel | Dernier stage de l'encodeur ; statistiques BatchNorm gelées |
| Époques visées | 3 : une tête, puis deux avec adaptation de l'encodeur |
| LR encodeur / tête | 5e-6 / 3e-4 |
| Optimisation | AdamW, weight decay 0,01, AMP FP16, clipping 1 |
| Batch | Une étude par micro-batch, accumulation huit, perte normalisée sur ces huit études |
| CV | Les cinq folds maîtres, seed 42 + fold, checkpoint final fixé avant OOF |

Le profilage doit mesurer forward **et backward**, mémoire et I/O à 384. Ajuster
les époques avant toute évaluation externe si le plafond le nécessite ou permet
une adaptation plus longue. Vérifier que le dernier stage désigne réellement
les bons modules dans la version de timm retenue ; tracer leurs noms.

## Livrables à produire sur Kaggle

Cinq `final_fold0.safetensors` à `final_fold4.safetensors`, `oof_coatnet2.csv`
(4 407 UID), rapport séparant gold et weak, historique, manifestes et profil
d'inférence. Les fichiers seront produits par le futur moteur ; ce README
ne signifie pas qu'ils existent déjà. Le suivi initial est dans
[suivi_folds.csv](../suivi_folds.csv).

La branche 224 supplémentaire, SWA et TTA ne font pas partie de cette première
expérience. Le modèle fine-tuné devra utiliser le même rendu géométrique au test ;
les anciennes têtes entraînées sur caractéristiques JPEG ne sont pas interchangeables.
