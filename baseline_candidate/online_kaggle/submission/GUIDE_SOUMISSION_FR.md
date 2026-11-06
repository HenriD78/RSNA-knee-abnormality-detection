# Soumettre la baseline entraînée sur Kaggle

Ce dossier utilise les **quatre encodeurs et les vingt têtes que vous avez déjà
entraînées**. Il lance seulement l'inférence. Les résultats locaux de trois
études ne suffisent pas : Kaggle remplace les exemples par le test caché au
moment de noter le notebook.

## Les deux fichiers à utiliser

1. **`dist/UPLOAD_SUBMISSION_TO_KAGGLE.zip`** : créer un **nouveau Dataset privé**
   avec ce ZIP. Il contient les poids génériques, les vingt têtes finales,
   le code et les dépendances hors ligne pour Linux.
2. **`KAGGLE_SUBMISSION.ipynb`** : importer comme **nouveau Notebook** Kaggle.

Ce nouveau paquet est différent de l'ancien `UPLOAD_TO_KAGGLE.zip` destiné
à l'entraînement. Les images, annotations, caches et anciens CSV de prédictions
ne font pas partie du paquet de soumission. La compétition fournira les DICOM.

## 1. Téléverser le paquet des poids

1. Dans Kaggle, **Create → New Dataset**, déposer `UPLOAD_SUBMISSION_TO_KAGGLE.zip`.
2. Donner un nom tel que `rsna-baseline-four-readers-inference` et garder **Private**.
3. Attendre la fin du traitement. Le contenu doit inclure `submission_bundle.json`,
   `assets/`, `heads/`, `code/` et `wheels/`.

Les têtes sont exportées en **safetensors**. L'optimiseur, l'état d'entraînement,
les rapports médicaux et les images train restent hors du paquet.

## 2. Importer le notebook et attacher les deux inputs

1. Créer un nouveau Notebook, puis **File → Import Notebook** et sélectionner
   `KAGGLE_SUBMISSION.ipynb`.
2. Avec **Add Input**, attacher le nouveau Dataset privé.
3. Avec **Add Input**, attacher la compétition **RSNA Knee Abnormality Detection**.
   C'est indispensable : le dossier officiel contient `test.csv`, `test_series.csv`,
   `sample_submission.csv` et `test_series/`.
4. Choisir **GPU T4 x2**. Ce notebook utilise le premier GPU.
5. Désactiver **Internet**.

Si plusieurs paquets sont attachés, préciser `BUNDLE_DIR` avec le chemin du dossier
contenant `submission_bundle.json`. Pour une arborescence inhabituelle, préciser
`COMPETITION_DIR` avec le dossier officiel contenant les trois CSV test.

## 3. Autoriser l'inférence puis sauvegarder une version

Le notebook est livré en inspection. Modifier la première cellule de code :

```python
RUN_INFERENCE = True
```

Les autres paramètres peuvent rester tels quels. Il n'y a plus de `MODE`,
`ACTION`, ni budget d'entraînement de six heures : la soumission doit traiter
**toutes** les études test avant d'écrire son CSV final.

Lancer **Save Version → Save & Run All**, puis attendre une fin réussie. Le
calcul s'effectue côté Kaggle. Une ancienne session interactive peut rester
active : dans le gestionnaire des sessions, arrêter seulement cette ancienne
session en laissant tourner la version batch voulue.

La première étape vérifie les SHA-256, installe les wheels incluses avec
`--no-index --no-deps`, puis vérifie les décodeurs DICOM. Elle conserve le
torch/torchvision CUDA du service. Ne pas installer le lock macOS et ne pas
importer pydicom/cv2/timm avant cette étape.

## 4. Contrôler les fichiers Output

| Sortie | Signification |
|---|---|
| **`submission.csv`** | Fichier principal à sélectionner pour la soumission |
| `inference_report.json` | Nombre d'études, durées, versions, modèle et mélange utilisés |
| `decode_errors.csv` | DICOM illisibles ignorés ; une dépendance manquante arrête le calcul |
| `selected_images.csv` | UID et slots des images retenues pour cette exécution |
| `series_diagnostics.csv` | Nombre d'images lisibles par série |
| `reader_predictions/` | Probabilités moyennées sur les cinq folds, avant les rangs |
| `blend_predictions/` | Les cinq mélanges prévus, pour inspection |

Le CSV final contient `StudyInstanceUID` et les douze colonnes dans l'ordre officiel.
Il suit l'ordre du **test.csv courant**. Trois lignes sont normales dans la version
visible ; lors du scoring, le même code traite les études cachées à la place.

Le fichier final utilise **`candidate05`**, fixé avant l'évaluation gold :
70/30 CoAtNet/Small, ajout Tiny par finding, puis 5 % DINOv2. Les probabilités des
cinq folds sont moyennées avant les rangs. Les rangs sont calculés globalement
sur la cohorte test courante, avec rang moyen des égalités. Ce sont des scores
de classement dans [0,1], pas des probabilités calibrées.

## 5. Envoyer à la compétition

Ouvrir la version terminée avec succès, choisir **Submit to Competition**, puis
sélectionner **`submission.csv`**. Kaggle relance ce notebook sur le test caché.
Téléverser un ancien CSV local de trois lignes n'effectue pas cette inférence.

Les [exigences officielles](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
imposent une soumission par notebook, Internet désactivé, un fichier nommé
`submission.csv` et une durée d'exécution maximale de neuf heures. La
[description des données](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
précise le remplacement des exemples par le test de scoring.

Le compte doit avoir rejoint la compétition et accepté ses règles. Si le bouton
Submit est absent, vérifier ce point et les dates sur la page officielle. Lors
de la vérification du 19 octobre 2026, l'inscription était annoncée close depuis
le 15 octobre ; la limite de soumission annoncée est le 22 octobre à 23:59 UTC,
soit le 23 octobre à 01:59 à Paris. Le paquet reste utilisable pour l'inférence
même si le compte n'est plus éligible à l'inscription.

## Carte des fichiers locaux

| Fichier | Rôle |
|---|---|
| `KAGGLE_SUBMISSION.ipynb` | Notebook à importer ; paramètres, installation, inférence, contrôles |
| `build_submission_bundle.py` | Conditionneur permanent : exporte les têtes et le ZIP sans entraînement |
| `runtime/rsna_submit_bootstrap.py` | Détecte les inputs, vérifie les empreintes, installe les wheels |
| `runtime/knee_submit/data.py` | Lit le test, décode les DICOM, choisit séries/images et crée le cache temporaire |
| `runtime/knee_submit/models.py` | Architecture exacte des têtes entraînées |
| `runtime/knee_submit/predict.py` | Charge les encodeurs/têtes, extrait les features float16 et moyenne les folds |
| `runtime/knee_submit/blend.py` | Alignement des UID, rangs globaux, mélanges et validation du CSV |
| `runtime/knee_submit/inference.py` | Orchestre le parcours et publie atomiquement le résultat final |
| `runtime/knee_submit/constants.py`, `io.py` | Noms des modèles/findings, coefficients et écritures/empreintes |
| `bundle/` | Vingt têtes exportées et manifeste local ; le ZIP inclut aussi les poids génériques |
| `wheels/` | Dépendances téléchargées pour Linux ; les wheels GDCM existantes sont réutilisées au conditionnement |
| `tests/` | Tests synthétiques, comparaison des pixels et contrôle des poids exportés |
| `VERIFICATION_FR.md` | Vérifications effectuées et limites restantes |

Pour reconstruire le ZIP depuis le dossier du présent guide :

```bash
PYTHONDONTWRITEBYTECODE=1 python build_submission_bundle.py
```

## Conversion des images et limites

Les JPEG d'entraînement ont été fournis déjà convertis ; leur convertisseur source
n'est pas disponible. L'audit local utilise des paires DICOM/JPEG et reproduit la
conversion observée : min–max par image, uint8, resize OpenCV INTER_AREA en
224 × 224, JPEG qualité 95 puis décodage Pillow. Aucun nouveau fenêtrage, inversion
photométrique ou recadrage anatomique n'est ajouté. Les différences sur d'autres
scanners et codecs restent possibles.

Les DICOM multi-frame reçoivent un identifiant de frame déterministe ; le nom
qu'aurait donné le convertisseur d'origine ne peut pas être garanti. Les images
illisibles sont consignées. Une étude sans image exploitable arrête l'exécution
avec un message explicite ; aucun ancien CSV ou score constant ne lui est substitué.

La macro AUC locale du candidat principal est environ **0,7302 sur 58 gold OOF**.
Ce chiffre n'est pas un score leaderboard. Aucun calcul complet ni envoi de
soumission Kaggle n'a été effectué pendant la création du paquet.

## Dépannage

| Problème | Solution |
|---|---|
| Mode inspection, aucun CSV | Passer RUN_INFERENCE à True puis sauvegarder/exécuter la version |
| Paquet introuvable | Attacher le nouveau Dataset de soumission, avec submission_bundle.json |
| Compétition introuvable | Attacher la compétition officielle ; un Dataset JPEG n'en tient pas lieu |
| Module déjà chargé dans une autre version | Redémarrer le noyau et exécuter les cellules dans l'ordre |
| Décodeur manquant | Vérifier le nouveau Dataset, redémarrer ; GDCM doit être installé avant pydicom |
| Wheel GDCM absente pour la version Python | Utiliser une image Kaggle Python 3.11 à 3.13 |
| Aucune image utilisable | Lire decode_errors.csv et vérifier le dossier DICOM officiel |
| Submit absent | Vérifier succès de la version, Internet désactivé, nom submission.csv et éligibilité du compte |
| Résultat à trois lignes | Normal pour les exemples ; soumettre la version notebook pour l'inférence cachée |
