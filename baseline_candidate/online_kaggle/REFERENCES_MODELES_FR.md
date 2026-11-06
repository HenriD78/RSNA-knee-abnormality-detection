# Modèles inclus dans le paquet GPU

Les quatre encodeurs sont des poids génériques préentraînés, repris sans fine-tuning.
Les têtes RSNA seront entraînées par vous sur les cinq folds déjà préparés.
Aucun checkpoint RSNA de la communauté n’est inclus dans ce paquet.

| Lecteur | Dépôt des poids | Révision figée | Entrée |
|---|---|---|---|
| `coatnet` | [timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k/tree/2c31d82376cc244c907c064332941e6c65be81f1) | `2c31d82376cc244c907c064332941e6c65be81f1` | 384 × 384 |
| `small` | [timm/convnext_small.fb_in22k_ft_in1k](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k/tree/fe6b358aa409414cdb522ae3d43a2c9922f7cc1b) | `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b` | 224 × 224 |
| `tiny` | [timm/convnext_tiny.fb_in22k_ft_in1k](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k/tree/bc48a87f119bae1b06418f5340bfa97ef75d4609) | `bc48a87f119bae1b06418f5340bfa97ef75d4609` | 224 × 224 |
| `dino` | [timm/vit_small_patch14_dinov2.lvd142m](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/tree/4610ca143709d58a633b6397a74412c2c3842454) | `4610ca143709d58a633b6397a74412c2c3842454` | 224 × 224 |

Les quatre manifestes conservent aussi les SHA-256, timm 1.0.29 et les
paramètres de transformation. DINOv2 adapte ses embeddings de position à 224.

## Origine de la méthode et limites

- [Notebook source de goodpjw2008, version 357478418](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954?scriptVersionId=357478418) : inspiration du mélange 70/30 et des poids Tiny par finding.
- [timm](https://github.com/huggingface/pytorch-image-models) : encodeurs et transformations.
- [DINOv2](https://github.com/facebookresearch/dinov2) : représentation auto-supervisée.
- [Iterative stratification](https://github.com/trent-b/iterative-stratification) : cinq folds communs déjà préparés localement.

Cette baseline JPEG ne reproduit ni les fenêtres anatomiques DICOM 2.5D, ni les
checkpoints RSNA spécialisés du notebook source. Les scores publiés ne sont donc
pas une promesse de performance pour ce paquet.

Les 58 études gold gardent leurs véritables labels et une confiance de 1.
Les autres utilisent les probabilités et confiances de report_labels_v2.csv ;
une cible inconnue ne devient pas un négatif. Les groupes de rapports normalisés
restent dans le même fold. Aucun identifiant disponible ne démontre une séparation patient.
Les annotations faibles ont déjà été auditées contre les gold dans le projet :
les scores OOF restent exploratoires et cette provenance n’est pas rendue indépendante
par le seul entraînement des têtes hors fold. Les mélanges restent fixés avant évaluation.
