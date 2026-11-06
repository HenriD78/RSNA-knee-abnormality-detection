# Notebooks

Ouvrir JupyterLab depuis la racine du dépôt avec l’environnement décrit dans
le [README principal](../README.md). Les notebooks locaux retrouvent cette
racine lorsqu’ils sont lancés depuis leur propre dossier.

1. `exploration/rsna_knee_exploration.ipynb` : inspection des tables et échantillons DICOM ; certaines cellules accèdent à Kaggle avec les identifiants de l’utilisateur.
2. `exploration/mri_planes_and_3d_sequences.ipynb` : plans anatomiques et séquences 2D/3D sur les images locales.
3. `report_weak_labeling.ipynb` : extraction des cibles et confiances depuis les rapports ; écrit dans `data/derived/`.
4. `mae/` : reconstruction de patches masqués puis classification au niveau étude.

Les notebooks des pipelines supervisés restent avec leur code dans
[`baseline_candidate/`](../baseline_candidate/README.md) et
[`finetuned_ensemble/`](../finetuned_ensemble/README_FR.md).
Les sorties d’exécution sont exclues des notebooks destinés au dépôt.
