# Nettoyage du 19 octobre 2026

Environ **4,54 Go** de fichiers redondants ou régénérables ont été retirés,
principalement dans les sous-dossiers de travail. Les fichiers scientifiques
utiles à la racine ont été conservés.

## Ce qui reste disponible

- Les notebooks de référence à la racine : comparaison DINOv2/RadImageNet,
  annotation faible des rapports et ensemble CoAtNet/R6 à 0,958.
- Les notebooks d'exploration dans `Data Exploration/`, les expériences dans
  `MAE/`, et les vrais notebooks de référence ou exécutés dans `scratch/`.
- Les documents de choix méthodologiques, les guides, les schémas et les scripts
  de vérification. Seul un README automatique de cache pytest a été retiré.
- `data/`, les échantillons DICOM, les environnements Python, le code, les poids
  autonomes dans `baseline_candidate/assets/` et tous les checkpoints.
- Les métriques, OOF, folds, historiques, figures et résultats des expériences,
  notamment `baseline_candidate/online_kaggle/rsna_resume/` et `rsna_reports/`.
- Les deux ZIP prêts à téléverser : `UPLOAD_TO_KAGGLE.zip` pour l'entraînement
  et `UPLOAD_SUBMISSION_TO_KAGGLE.zip` pour la soumission.
- L'ancien paquet MAE dans `submission/`, qui reste une expérience distincte.

## Copies et caches retirés

Les fichiers intermédiaires suivants, environ **3,27 Go**, sont déjà présents
octet pour octet dans `baseline_candidate/online_kaggle/dist/UPLOAD_TO_KAGGLE.zip` :

| Fichier retiré dans `dist/` | Membre conservé dans le ZIP |
|---|---|
| `data.tar` | `data_payload.bin` |
| `assets.tar` | `assets_payload.bin` |
| `seed_features.tar` | `seed_payload.bin` |
| `runtime_code.zip` | `code_payload.bin` |

Le ZIP et les quatre correspondances ont été vérifiés par SHA-256 avant retrait.
Le ZIP final suffit pour l'utilisation sur Kaggle.

Le cache `baseline_candidate/cache/huggingface/`, environ **700 Mo**, a été
retiré après vérification des quatre copies autonomes des poids dans `assets/`.
Le chargeur utilise ces copies locales ; il ne nécessite pas ce cache.

Ont également été retirés : le cache pytest à la racine, celui de
`baseline_candidate/cache/pytest/`, `scratch/matplotlib_cache/`, le notebook
d'essai de 189 octets `scratch/test_nb.ipynb` et quatre fichiers `.DS_Store`.
Les caches de caractéristiques et les vrais notebooks ont été conservés.

## Manifestes historiques compressés sans perte

Dans les deux dossiers `outputs/mae_baseline/pilot_20260924T*/`, chaque
`manifest.csv` de 310,45 Mo est désormais conservé comme `manifest.csv.gz`
de 25,12 Mo. Cela économise environ **571 Mo** au total.

Après décompression, chacun possède exactement l'empreinte du fichier d'origine,
également conservée dans son `manifest.sha256` :

```text
42b8939dfd58632b5ae76b5a5b22d014e85a403c37bd399597c375fc6c8c108f
```

Pandas peut les ouvrir avec `pd.read_csv(chemin_vers_manifest_csv_gz)`.
Le notebook MAE reconstruit son manifeste depuis les images et utilise
`manifest.sha256` pour vérifier une reprise ; il ne relit pas ce CSV historique.

Pour retrouver les deux fichiers non compressés, depuis la racine :

```python
from pathlib import Path
import gzip
import shutil

for archive in Path("outputs/mae_baseline").glob("pilot_*/manifest.csv.gz"):
    destination = archive.with_suffix("")
    if not destination.exists():
        with gzip.open(archive, "rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
```

## Restaurer les intermédiaires du paquet d'entraînement

Cette opération est utile uniquement pour relancer `tests/verify_payloads.py`
ou `build_bundle.py --stage finish`, qui attendent les archives intermédiaires.
Elle n'est pas nécessaire pour exécuter les notebooks sur Kaggle et recrée
environ 3,27 Go de copies.

Depuis la racine, exécuter ce code Python :

```python
from pathlib import Path
import shutil
import zipfile

dist = Path("baseline_candidate/online_kaggle/dist")
members = {
    "data.tar": "data_payload.bin",
    "assets.tar": "assets_payload.bin",
    "seed_features.tar": "seed_payload.bin",
    "runtime_code.zip": "code_payload.bin",
}
with zipfile.ZipFile(dist / "UPLOAD_TO_KAGGLE.zip") as bundle:
    for filename, member in members.items():
        destination = dist / filename
        if not destination.exists():
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
```

Aucun entraînement, extraction d'images ou notebook complet n'a été lancé
pendant ce nettoyage. Aucun téléchargement n'a été effectué.
