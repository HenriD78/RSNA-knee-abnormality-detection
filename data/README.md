# Données locales

Les données de compétition ne sont pas incluses. Les obtenir depuis la
compétition RSNA Knee Abnormality Detection avec son propre compte Kaggle,
puis respecter les conditions d’accès et de réutilisation associées.

Structure attendue pour les expériences JPEG :

```text
data/
├── train.csv
├── train_series.csv
├── test.csv
├── test_series.csv
├── sample_submission.csv
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.jpeg
├── test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.jpeg
└── derived/report_labels_v2.csv
```

`notebooks/report_weak_labeling.ipynb` construit le dernier fichier à partir
des rapports de `train.csv`. Les cibles inconnues et les confiances sont
traitées explicitement par les pipelines supervisés.

La conversion JPEG d’origine n’est pas fournie. Le guide de reproduction
documente la comparaison empirique effectuée avec des DICOM. Le pipeline
`finetuned_ensemble/rsna3/dicom.py` possède un rendu DICOM distinct et attend
les chemins transmis dans sa CLI ou les entrées de compétition sur Kaggle.

Seul ce guide est destiné à Git. Les images, rapports, CSV source, cibles
dérivées et identifiants d’études restent locaux, ainsi que
`sampled_rsna_data/` à la racine.
