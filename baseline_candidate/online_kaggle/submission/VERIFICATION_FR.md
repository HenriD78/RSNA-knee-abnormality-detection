# Contrôles du paquet de soumission

Vérification locale du **19 octobre 2026**. Les fichiers existants de formation
et les résultats téléchargés restent en lecture seule. Aucun compte, Dataset,
calcul ou envoi de soumission Kaggle n'est créé par cette livraison.

## Contrôles locaux

- **9 tests ciblés** sur données synthétiques : remplacement des UID test par
  quatre études nouvelles, rangs globaux, retrait d'un ancien CSV en cas d'échec,
  choix de série et hash d'échantillonnage, slots masqués, DICOM 2D et multi-frame,
  décodage RLE compressé, images corrompues, dépendance manquante et UID invalides.
- Les cinq mélanges reproduisent les CSV de la session terminée, même lorsque
  les lignes d'un lecteur sont permutées.
- Validation nbformat et syntaxe de toutes les cellules. Le notebook est livré
  sans exécution ni sortie enregistrée ; son mode initial est inspection.
- Vérification des CRC du ZIP et de tous les SHA-256 du manifeste, y compris les
  quatre encodeurs, les vingt têtes safetensors, le code et les wheels Linux.
- Les vingt têtes exportées sont comparées **tenseur par tenseur** aux checkpoints
  finaux, et leurs probabilités sur une étude déjà mise en cache sont comparées
  aux prédictions des folds produites sur Kaggle. Aucun entraînement n'est exécuté.
- Vérification de la sélection des **104 images des trois exemples JPEG** :
  mêmes séries, mêmes images, mêmes slots/tokens que lors de l'expérience.
- **Une image seulement par vrai encodeur**, sur CPU, avec les poids du ZIP.
  Dimensions, gel, transformations et proximité avec les caractéristiques CUDA
  sauvegardées sont contrôlés. Ce n'est pas une extraction complète.
- Audit de la conversion sur **25 paires DICOM/JPEG** disponibles localement.
  La conversion observée est min–max par image, uint8, INTER_AREA 224, JPEG 95.
  L'erreur moyenne est contrôlée sous 0,02 sur l'échelle 0–255 pour chaque paire.

Le relevé exact des différences numériques est dans **`dist/verification.json`**.
Le SHA-256 de l'archive finale figure séparément dans **`dist/UPLOAD_SHA256.json`**.

Les contrôles ont réussi : écart maximal des probabilités des têtes par rapport
aux sorties CUDA enregistrées **4,77 × 10⁻⁷** ; écart maximal des caractéristiques
float16 CPU/CUDA sur les images contrôlées **0,001953125**. Pour les 25 paires
de pixels, l'erreur absolue moyenne vaut environ **0,001385 / 255**, avec un
maximum de moyenne par image de **0,007693 / 255**. Ces nombres décrivent les
contrôles limités effectués et ne constituent pas des mesures de performance.

## Limites des contrôles

Le Mac local ne permet pas de lancer les wheels Linux GDCM ni un GPU NVIDIA.
Leurs empreintes sont vérifiées et l'installateur choisit la wheel correspondant
à la version Python ; le
chargement des binaires et le décodage JPEG Lossless/JPEG 2000 avec ces wheels
doivent être confirmés par votre exécution Kaggle. Le notebook vérifie les
décodeurs avant de traiter les images. Un codec absent arrête le calcul.

La conversion source des JPEG n'est pas disponible. L'accord sur les 25 paires
est une vérification empirique, pas une preuve d'identité sur tous les scanners.
La méthode de nommage du convertisseur d'origine pour les fichiers multi-frame
reste inconnue ; le notebook leur attribue des identifiants déterministes.

Le notebook complet et l'inférence complète des exemples/cas cachés n'ont pas
été exécutés pendant cette livraison. Les tests de parcours utilisent des
encodeurs artificiels ; les vrais modèles ne sont appelés que pour les contrôles
limités décrits ci-dessus. Aucun score leaderboard n'est annoncé.

## Refaire les contrôles

Depuis ce dossier, avec l'environnement Python local du projet :

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests --basetemp=cache/test_tmp
PYTHONDONTWRITEBYTECODE=1 python tests/verify_artifacts.py
```

La deuxième commande ouvre le ZIP dans un répertoire temporaire local et lit
les petits caches test existants. Elle supprime ce répertoire à la fin.
Ces scripts de vérification ne génèrent pas de notebook. Aucun script temporaire
de génération de notebook n'est conservé.
