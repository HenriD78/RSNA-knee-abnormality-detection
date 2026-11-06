# Références et limites de cette adaptation

- [Notebook de goodpjw2008, version 357478418](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954?scriptVersionId=357478418) : attribution des lecteurs, baseline 70/30 et poids Tiny par finding.
- [CoAtNet de nartaa](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) : lecteur publié avec checkpoint spécifique et géométrie DICOM.
- [Modèle compact S6 de heliosli](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1) : modèle fine-tuné avec ancrages anatomiques ; **notre ConvNeXt-Small n'est pas S6**.
- [Lecteur original Tiny de goodpjw2008](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) : trois coupes adjacentes, transformer et MIL ; notre Tiny emploie des JPEG indépendants.
- [timm](https://github.com/huggingface/pytorch-image-models) : encodeurs génériques, poids et transformations.
- [DINOv2](https://github.com/facebookresearch/dinov2) : représentation auto-supervisée générique.
- [Iterative stratification](https://github.com/trent-b/iterative-stratification) : stratification multilabel des groupes.

Une copie du guide CV fourni se trouve dans `../docs/methodology/cross_validation_strategy.md`.
Nous conservons ses cinq folds communs, avec trois précisions :

1. Les groupes de rapports identiques doivent rester ensemble. Aucun identifiant patient exploitable ne figure dans les CSV ; nous ne revendiquons donc pas une indépendance patient démontrée.
2. Remplacer tous les labels inconnus par zéro ne répartit pas correctement les 4 349 études à labels faibles. Les indicateurs gold et weak sont séparés pour la stratification ; les valeurs inconnues restent inconnues pour la perte.
3. Un mélange ajusté puis évalué sur les mêmes gold OOF peut surapprendre. Les coefficients du livrable sont fixés avant l'évaluation ; les variantes ne sont jamais sélectionnées automatiquement.

Les annotations faibles proviennent du fichier existant `data/derived/report_labels_v2.csv`.
L'extracteur source a déjà été audité contre les gold dans le projet. Les OOF
garantissent l'exclusion du groupe lors de l'entraînement des têtes, mais ne
rendent pas cette provenance des labels totalement indépendante. Les résultats
restent exploratoires sur seulement 58 études.

Les têtes, le cache, les splits et la CLI sont implémentés dans ce dossier.
Les poids téléchargés restent ceux des dépôts préentraînés ; leur révision exacte,
leur SHA-256 et leur provenance sont enregistrés dans `assets/<reader>/manifest.json`.
