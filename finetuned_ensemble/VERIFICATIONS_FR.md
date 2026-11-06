# Vérifications de l'implémentation — 19 octobre 2026

## Exécuté localement

- **43 tests automatisés passent dans cette vérification historique** ; rapport machine local : `verification/tests.xml`, exclu de Git. Les contrôles actuels se lancent avec `python scripts/run_tests.py` depuis la racine.
- Chargement strict des trois vrais poids génériques, forward et backward FP32
  pour CoAtNet-2 384, ConvNeXt-Small 224 et DINOv2-Small 224. Dimensions de
  sortie vérifiées ; embeddings DINOv2 interpolés de 37×37 à 16×16 avec CLS.
- Tête seule puis dégel : gradients dans la partie autorisée, aucun dans le
  préfixe gelé, statistiques BatchNorm inchangées après `model.train()`.
- Masques et permutation conjointe des caractéristiques/slots/positions : logits
  invariants en évaluation, pour MIL simple et transformer à deux couches.
- Accumulation pondérée de huit études : gradients identiques à la référence
  calculée sur le batch complet, cibles NaN sans gradient et confiances conservées.
- DICOM synthétiques : multi-frame, MONOCHROME1, rescale, IPP/IOP, tri selon une
  normale sagittale plutôt que z, doublons, grands trous et changement d'orientation.
  Rendu physique tenant compte de PixelSpacing et du champ de 153,6 mm.
- Décodage des huit DICOM réellement présents dans `sampled_rsna_data/dicoms/`.
  Ces fixtures de deux coupes par série ne valident pas les vrais triplets d'une
  étude complète. Les triplets/cache sont testés sur des DICOM synthétiques.
- Cache : choix de la série exploitable la plus longue, slots fat-sat distincts,
  centres sans duplication, métadonnées par fenêtre, SHA-256 et reprise de shard.
  Une étude vide ou un défaut de codec produit une erreur bloquante.
- Petit entraînement synthétique : interruption après un pas d'adaptation,
  reprise avec RNG/permutation/offset/AdamW/scheduler et poids finaux **strictement
  identiques** au parcours continu. Le dernier batch effectif incomplet est exercé.
- Cette vérification de reprise a aussi été exécutée séparément avec **un vrai
  worker DataLoader** (`RSNA3_TEST_WORKERS=1`) : un test passe. Le sandbox macOS
  bloquait initialement le démarrage multiprocessing ; l'exécution hors de cette
  restriction passe. Les données de ce contrôle restent synthétiques.
- Parcours complet de quinze folds **avec un minuscule modèle synthétique** :
  entraînement, OOF alignés, deux cohortes, mélanges fixes, bootstrap, export de
  quinze états complets et rejet d'une couverture partielle. Aucun de ces poids
  temporaires n'est livré comme résultat RSNA. Les résultats source restent en
  lecture seule ; les OOF agrégés sont écrits dans le dossier de rapports.
- Coordinateur simulé : deux GPU occupés au maximum, aucun GPU partagé entre
  deux folds, quinze tâches uniques et priorité Small respectée.
- Inférence : moyenne des probabilités avant rangs, égalités, singleton,
  rejet des UID absents/dupliqués, retrait global d'une branche, socle incomplet
  bloquant et suppression d'un ancien CSV avant un run en échec.
- **Huit notebooks** validés par `nbformat.validate`, toutes leurs cellules Python
  compilables, aucun output d'entraînement prétendu. Les notebooks ne sont pas
  exécutés sur le Mac : ils requièrent les entrées et services Kaggle.
- Le nouveau calcul d'AUC reproduit exactement la référence candidate05 :
  **0,7302461374891095** sur les 58 gold. Bootstrap de deux copies identiques de
  cette baseline : **2 000 réplications valides**, différence et intervalle nuls.
  Ce contrôle ne constitue pas une évaluation des nouveaux modèles.
- `verifier_plan.py` et `rsna3 check` : six CSV et trois poids génériques vérifiés,
  4 407 UID / 58 gold / 4 349 weak / 4 257 groupes, cinq folds sans groupe partagé.
  Le vérificateur historique accepte maintenant les nouveaux dossiers de code,
  tout en gardant ses assertions sur les spécifications et le suivi initial.
- Huit wheels Linux x86_64/Python 3.13 téléchargées pour le paquet corrigé, dont GDCM et les codecs
  pylibjpeg ; tags ABI contrôlés. Le paquet training contient **39 fichiers
  vérifiés par SHA-256**, dont le même code que `rsna3/`. Le bootstrap n'installe
  ni Torch ni NumPy et vérifie les décodeurs dans un processus frais.

Commandes reproductibles :

```bash
.venv/bin/python -m pytest "finetuned_ensemble/tests" -q
.venv/bin/python "finetuned_ensemble/tests/verify_notebooks.py"
.venv/bin/python "finetuned_ensemble/tests/verify_reference.py"
.venv/bin/python "finetuned_ensemble/tests/verify_bundle.py"
python3 "finetuned_ensemble/verifier_plan.py"
```

- Démarrage du notebook CPU : deux tests supplémentaires vérifient que le
  premier Run All lance uniquement le probe (avec et sans repli à deux centres),
  sans démarrer le cache complet. Archive ZIP pour upload préparée localement.

- Correction de l'incompatibilité Python 3.12/3.13 : cinq tests vérifient la
  sélection selon les tags ABI Linux, la compatibilité abi3 de safetensors,
  le rejet anticipé d'un ancien paquet incompatible, les conflits de wheels,
  l'intégrité du patch et l'idempotence de son installation. Les appels pip sont
  simulés ; les binaires Linux ne sont pas importés sur ce Mac.
- Patch hors ligne Python 3.13 d'environ 23 Mio prêt à ajouter à l'ancien upload,
  sans retéléverser les poids. Nouveau paquet complet Python 3.13 également
  disponible ; l'archive historique Python 3.12 est conservée.
- Correction de latéralité entre séries : désaccord L/R explicite dans les
  avertissements du probe, du manifest et des NPZ. Conservation du tag effectif,
  de sa source et du FrameOfReferenceUID ; aucune inférence de côté ni modification
  des pixels fondée sur le tag. Tests de priorité Frame/Image/Series, repli sur
  Series si Image est vide, conflit dans une série toujours rejeté, et rendu
  identique lorsque seuls les tags changent.
- Cellule de récupération testée contre l'ancien bundle immuable : activation
  dans le kernel et un subprocess frais, empreinte identique au code courant,
  spécifications et 4 407 cibles accessibles par les liens aux assets d'origine,
  et exécution répétée sans changement d'empreinte. La copie corrigée du notebook
  CPU est dans `notebooks/00_preparer_cache_cpu_corrige.ipynb`. Ces tests locaux ne
  valident pas les images de l'étude Kaggle dont les tags se contredisent.
- Stockage lossless adaptatif : huit nouveaux cas de tests couvrent bruit uint8,
  gradient, zéros, débordements modulo 256, vues non contiguës, largeur 1,
  anciens NPZ directs, encodage ambigu rejeté et nécessité d'un nouveau probe.
  Pixels, slots et positions sont identiques après lecture. L'archive retenue
  n'est jamais plus grande que la variante directe comparée pour la même étude.
- Sur les huit coupes DICOM locales rendues en 384, un essai de compression des
  seuls pixels passe de 892 876 à 751 632 octets (environ 15,8 %) avec prédiction
  horizontale, sans perte de pixels. Ce petit fixture ne prédit pas le ratio de
  compression du train ; le probe Kaggle sur les mêmes 75 UID reste obligatoire.
- La cellule combinée `patches/cell_fix_cache_and_probe.py` a été exécutée avec
  CLI simulée : une seule commande probe à deux centres, `BUILD_FULL_CACHE=False`,
  aucun cache complet créé. L'activation réelle de la copie de code et ses
  imports sont exercés ; le temps et la taille produits par cette CLI simulée
  ne constituent pas des résultats de performance.
- Notebook d'admission/entraînement : quatre tests exercent le gel sérialisable
  des recettes 7/9/10 avec les débits rapportés, la conservation du snapshot des
  pilotes, le refus d'un profil incompatible et la protection des huit heures
  de réserve. Les mêmes empreintes de recettes et un checkpoint témoin sont
  conservés lors d'une reprise. Le mode freeze du notebook commun est aussi
  testé, sans référence JSON circulaire. Métadonnées de profil synthétiques avec
  environnement local et coordinateur simulé : aucun fine-tuning CUDA n'est
  exécuté par ces tests.
- Le code modèle/cache utilisé sur Kaggle reste à l'empreinte
  `1f9c80ee22795f8702039ab37d64f84fd865755cd0c1f517134b2c6a6e1912a5`.
  Le générateur de recettes est dans les notebooks pour garder les profils et
  le runtime mesuré compatibles. Les premières métriques d'entraînement/OOF
  réels ne sont pas encore disponibles dans ce workspace.

## À exécuter sur Kaggle

- Décoder et inspecter visuellement des études complètes : couverture du genou,
  axes, côté et diversité des syntaxes réellement présentes. Vérifier l'installation
  des wheels contre le Python/NumPy du service ; les binaires Linux ne sont pas
  exécutables sur ce Mac.
- Probe CPU de 50–100 études, compression/temps/disque, choix de quatre ou deux
  centres sous 12 Gio, préparation de la couverture complète et persistance privée.
- Vrai pilote CUDA FP16/GradScaler avec chargement, tête et dégel, mémoire de
  chaque worker/GPU, checkpoint, reprise AMP et mesure du quota avant/après.
- Gel des recettes, puis calcul des quinze folds admissibles. Aucun epoch réel,
  checkpoint fine-tuné, OOF nouveau ou gain d'AUC n'est annoncé dans cette livraison.
- Bootstrap comparatif sur les futurs OOF, export, benchmark complet d'inférence
  sur 50 études représentatives et validation du notebook de compétition en
  Save & Run All avec Internet désactivé. Aucun lancement/soumission Kaggle n'a
  été effectué par cette session.
- Si un socle public exact est attaché : vérifier séparément ses assets, son
  environnement, son rendu, son TTA et son exécution isolée. Ses poids spécialisés
  ne sont pas les initialisations génériques livrées ici.

Le pilote décide de la faisabilité dans le quota ; les tests CPU ne la prouvent
pas. Les artefacts de la baseline et les données d'origine sont conservés.
