# Références et provenance

Ce dépôt combine du code expérimental propre au projet et des adaptations
de travaux publics. Les scores contenus dans les noms des notebooks téléchargés
appartiennent à leurs auteurs ; ils ne décrivent pas les performances de ce dépôt.

| Ressource | Usage dans le projet |
|---|---|
| `pilkwang/rsna-knee-baseline-v1`, copie locale intitulée `0-941-sota-rsna-knee-dinov2-radimagenet.ipynb` | Extracteur multilingue de labels faibles, repris dans `notebooks/report_weak_labeling.ipynb` ; cellules 17–19 de la copie locale |
| Notebook de goodpjw2008, version `357478418` | Inspiration des lecteurs, des mélanges fixés et des poids Tiny par anomalie ; attribution détaillée dans le guide de la baseline |
| CoAtNet de nartaa et modèle compact de heliosli | Références étudiées ; les encodeurs génériques du projet ne reproduisent pas leurs modèles affinés |
| `timm` et DINOv2 | Architectures, transformations et poids génériques utilisés par les pipelines |
| `iterative-stratification` | Stratification multilabel des groupes de rapports |

Les URL conservées avec le projet et les différences méthodologiques sont
documentées dans [les références de la baseline](../baseline_candidate/REFERENCES_FR.md).
La provenance et les empreintes des poids téléchargés sont enregistrées par
les pipelines dans leurs manifestes locaux.

Les copies intégrales des notebooks externes et leurs figures restent dans
`local/reference_notebooks/` et `local/scratch/`. Elles sont exclues de Git pour
présenter clairement les contributions du projet. Leur disponibilité locale
n’établit pas une autorisation de redistribution.

Le document original de validation croisée a été dédupliqué ; une seule copie
est conservée dans [docs/methodology/](../docs/methodology/cross_validation_strategy.md).
Les corrections appliquées sont décrites dans les guides des pipelines.
