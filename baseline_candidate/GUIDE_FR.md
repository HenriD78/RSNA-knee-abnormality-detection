# Utiliser la baseline à quatre lecteurs

Ce guide permet de préparer les données, calculer les caractéristiques et entraîner
les modèles vous-même. Le relevé de la livraison initiale est dans
`VERIFICATION_FR.md` ; les résultats obtenus ensuite sont présentés dans
[docs/results/](../docs/results/README.md).

## 1. Ce que cette baseline apprend

Quatre encodeurs génériques préentraînés lisent chaque JPEG séparément : CoAtNet-2,
ConvNeXt-Small, ConvNeXt-Tiny et DINOv2-Small. Ils restent **gelés**. Nous entraînons
uniquement les têtes qui regroupent leurs caractéristiques en douze scores par étude.
Tiny ajoute un petit transformer avant le regroupement par attention.

Les JPEG sont déjà en 224 × 224, sans géométrie DICOM exploitable. L'entrée CoAtNet
est redimensionnée à 384 : cela ne recrée pas les détails perdus. Les fichiers ne
fournissent pas l'ordre anatomique des coupes ; aucune adjacency 2.5D n'est inventée.
Cette baseline n'est donc pas une reproduction du score Kaggle 0.954.

## 2. Carte des fichiers

| Fichier ou dossier | Rôle | Entrées → sorties |
|---|---|---|
| `baseline_candidate.ipynb` | Parcours commenté et cellules de commande | Configuration → mêmes étapes que la CLI |
| `configs/full.json` | Paramètres du calcul complet | 8 images/slot, 20 époques, cinq folds |
| `configs/smoke.json` | Parcours réduit que vous pouvez lancer d'abord | Au plus 2 gold + 2 weak/fold, 2 images/slot, 1 époque |
| `requirements.txt`, `requirements.lock` | Relevé historique de l’environnement macOS | Versions des artefacts existants ; nouvelle installation depuis les dépendances à la racine |
| `knee_baseline/constants.py` | Ordre des labels, modèles et poids de mélange | Aucun traitement |
| `knee_baseline/config.py` | Chemins, validation de configuration et caches locaux | JSON → configuration contrôlée |
| `knee_baseline/io.py` | Empreintes et écritures atomiques | Données → fichiers reprenables |
| `knee_baseline/data.py` | Labels, groupes, folds, choix des séries et JPEG | `../data` → tables préparées |
| `knee_baseline/features.py` | Téléchargement explicite, benchmark, extraction | Poids + JPEG → caractéristiques gelées |
| `knee_baseline/models.py` | Attention, transformer Tiny et perte pondérée | `[B,6,K,D]` → `[B,12]` logits |
| `knee_baseline/train.py` | Entraînement des têtes, reprise, OOF, test | Cache + folds → checkpoints et probabilités |
| `knee_baseline/evaluate.py` | Mélanges fixés, AUC, bootstrap, soumissions | Prédictions → rapports et CSV |
| `knee_baseline/cli.py`, `__main__.py` | Interface de commande | Commande → une étape explicite |
| `tests/` | Contrôles sur données artificielles | Aucun entraînement des lecteurs |
| `references/`, `REFERENCES_FR.md` | Environnement historique, attributions et limites | Guide CV dans `../docs/methodology/` |
| `assets/` | Poids téléchargés et manifestes | Créé par `assets` |
| `cache/` | Caches des bibliothèques et fichiers de tests | Aucun JPEG source copié |
| `artifacts/smoke/`, `artifacts/full/` | Tous les résultats des parcours | Créés uniquement lors de vos commandes |

**Les commandes ci-dessous partent de `baseline_candidate/`.** Aucun code
n'importe les anciens notebooks ou leurs modules. Seules les données existantes
dans `../data` restent une dépendance externe au dossier.

## 3. Installation et inspection

Dans un terminal, depuis la racine du projet :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/dev.txt
cd baseline_candidate
python -m knee_baseline --config configs/full.json inspect
```

`requirements.lock` fixe la fermeture des dépendances de l'environnement local
Python 3.12, avec les versions observées pendant la livraison. Leur installation
dans un environnement neuf n'a pas été exécutée. Si une version n'est pas disponible
sur votre index, résoudre l'environnement explicitement et conserver un nouveau
lock ; ne pas remplacer silencieusement les versions pour reprendre d'anciens caches.

L'inspection lit les CSV et annonce 4 407 études train, 58 gold et trois exemples
test pour les données actuelles. Elle ne lit pas les 819 076 JPEG et ne charge aucun
encodeur. Les chemins relatifs de configuration sont résolus à partir du dossier
du livrable, même si le notebook a été ouvert depuis un autre répertoire.

Modifiez `data_dir` si les données ont déménagé. Gardez `output_dir` à l'intérieur
de `baseline_candidate/`. Choisissez un nouveau `output_dir` si vous modifiez un
profil déjà préparé : la reprise exige une configuration identique.

## 4. Pourquoi les folds évitent les fuites étudiées

Tous les modèles partagent la même table. Une étude conserve le même fold pour
toutes ses séries et images. Les rapports identiques après NFKC, normalisation
des espaces et casse ignorée forment également un seul groupe. Les rapports vides
ne sont pas regroupés arbitrairement.

La stratification multilabel opère sur les groupes, avec des indicateurs gold,
weak et présence de gold. Le seuil **> 0,5** sert à la stratification et aux métriques
proxy, **pas à transformer les probabilités faibles en cibles dures d'entraînement**.
Les gold ont une confiance de 1 ; les autres cibles gardent leur confiance existante.
Une cible inconnue a un poids nul dans la BCE.

Pour un fold `k`, sa tête apprend seulement sur les quatre autres folds, puis
prédit les études de `k`. L'époque finale est fixée : aucun early stopping sur le
fold évalué. Les encodeurs gelés peuvent extraire les images des cinq folds une
seule fois, puisqu'ils ne s'adaptent ni aux images MRI ni aux labels de ces études.

Les CSV ne fournissent pas l'identité patient. La séparation étude/rapport est
vérifiée, mais ne prouve pas l'indépendance patient. Voir `REFERENCES_FR.md` pour
les corrections du guide CV et la provenance des annotations faibles.

## 5. Parcours réduit recommandé

Ces commandes sont destinées à **votre exécution**. Le mode smoke valide le
parcours, et ses scores ne doivent pas être utilisés pour conclure sur la qualité.

```bash
python -m knee_baseline --config configs/smoke.json inspect
python -m knee_baseline --config configs/smoke.json prepare
python -m knee_baseline --config configs/smoke.json verify
python -m knee_baseline --config configs/smoke.json assets --reader all
python -m knee_baseline --config configs/smoke.json benchmark --reader all --images 8
python -m knee_baseline --config configs/smoke.json extract --reader all --split train
python -m knee_baseline --config configs/smoke.json train --reader all --fold all
python -m knee_baseline --config configs/smoke.json evaluate
python -m knee_baseline --config configs/smoke.json extract --reader all --split test
python -m knee_baseline --config configs/smoke.json predict --reader all
```

La table master conserve les 4 407 études et leurs folds, mais seule une petite
sélection par fold est active pour smoke. Ce n'est pas un nouveau split indépendant.
Les poids génériques sont les vrais poids, partagés entre smoke et full ; aucune
substitution par un modèle aléatoire n'est faite. Leur téléchargement peut déjà
être volumineux, même si le jeu smoke est petit.

## 6. Parcours complet

Après lecture des estimations du benchmark :

```bash
python -m knee_baseline --config configs/full.json prepare
python -m knee_baseline --config configs/full.json verify
python -m knee_baseline --config configs/full.json assets --reader all
python -m knee_baseline --config configs/full.json benchmark --reader all --images 16
python -m knee_baseline --config configs/full.json extract --reader all --split train
python -m knee_baseline --config configs/full.json train --reader all --fold all
python -m knee_baseline --config configs/full.json evaluate
python -m knee_baseline --config configs/full.json extract --reader all --split test
python -m knee_baseline --config configs/full.json predict --reader all
```

`prepare` compte les images lisibles de toutes les séries actives avant de choisir
la plus grande série par slot. En full, cette lecture des 819 076 images peut être
longue. Les erreurs de lecture sont consignées ; une étude sans image utilisable
arrête la préparation. Le cache de caractéristiques représente ensuite plusieurs
centaines de Mo à quelques Go selon les dimensions des lecteurs, en plus des poids.

Le benchmark n'est pas un entraînement. Il chronomètre un petit lot et extrapole
le temps de l'extraction train. Il n'est pas une garantie de durée totale : la
préparation, les vingt têtes, les accès disque et le bootstrap sont supplémentaires.

Vous pouvez avancer lecteur par lecteur :

```bash
python -m knee_baseline --config configs/full.json assets --reader tiny
python -m knee_baseline --config configs/full.json extract --reader tiny --split train
python -m knee_baseline --config configs/full.json train --reader tiny --fold 0
python -m knee_baseline --config configs/full.json train --reader tiny --fold 1,2,3,4
```

Noms CLI : `coatnet`, `small`, `tiny`, `dino`, ou `all`. L'évaluation complète exige
les OOF des cinq folds de chacun des quatre lecteurs.

## 7. Faire la même chose dans le notebook

Lancer Jupyter après activation de l'environnement :

```bash
JUPYTER_RUNTIME_DIR="$PWD/cache/jupyter_runtime" \
JUPYTER_CONFIG_DIR="$PWD/cache/jupyter_config" \
JUPYTER_DATA_DIR="$PWD/cache/jupyter_data" \
IPYTHONDIR="$PWD/cache/ipython" python -m jupyterlab baseline_candidate.ipynb
```

Dans la cellule **configuration** :

```python
MODE = "smoke"       # puis "full" lorsque vous avez validé le parcours
STEPS_TO_RUN = {"prepare"}
ALLOW_DOWNLOAD = False
```

Exécutez cette cellule, puis la cellule `prepare`. Elle est l'équivalent exact de
la commande CLI. Pour la suite, ajoutez uniquement l'étape voulue à `STEPS_TO_RUN`
et relancez la configuration avant sa cellule :

| Étape dans le notebook | Commande / effet |
|---|---|
| `inspect` | Lecture légère des CSV, disponible en inspection |
| `prepare` | Labels, folds et index des images actives |
| `verify` | Empreintes des entrées et caches existants |
| `assets` | Téléchargement ; exige aussi `ALLOW_DOWNLOAD = True` |
| `benchmark` | Lot réduit et estimation du temps |
| `extract_train` | Cache des caractéristiques train |
| `train` | Entraînement des têtes des lecteurs/folds choisis |
| `evaluate` | Comparaison gold/weak, mélanges et bootstrap |
| `extract_test` | Cache des exemples test |
| `predict` | Moyenne des cinq folds et fichiers de soumission |

`READERS_TO_RUN` et `FOLDS_TO_RUN` permettent de limiter lecteurs et folds. En mode
`inspection`, toutes les cellules lourdes restent inactives, même avec « Run All ».
La lecture des rapports déjà créés reste disponible. Si vous sélectionnez toutes
les étapes en mode smoke/full, « Run All » exécutera effectivement tout : cette
sélection est votre décision explicite.

## 8. Interrompre et reprendre

Dans un terminal, utiliser `Ctrl+C`. Dans Jupyter, interrompre le noyau. Ensuite,
relancer **la même commande avec le même profil et le même `output_dir`**.

- **Extraction** : les études publiées dans le manifeste sont réutilisées après
  vérification SHA-256. Les dernières études non publiées peuvent être recalculées
  après un arrêt ; aucun cache partiel n'est accepté comme un succès.
- **Entraînement** : un checkpoint atomique est enregistré à chaque époque avec
  tête, optimiseur et générateur aléatoire. Une époque interrompue recommence depuis
  le checkpoint précédent. Le CPU facilite la reproductibilité de cette reprise.
- **Préparation** : elle est publiée uniquement quand elle est complète. Un arrêt
  avant cette publication nécessite de recommencer l'indexation.

La reprise compare la configuration, les CSV, les tailles/dates des images
sélectionnées, les poids et les caches. L'ajout ou la modification d'images non
sélectionnées nécessite une nouvelle préparation dans une nouvelle sortie pour
reconsidérer le choix des séries. Ne modifiez pas manuellement les manifestes.

Pour refaire une expérience, créer une copie de configuration avec un nouveau
`output_dir`. Les encodeurs génériques dans `assets/` peuvent être réutilisés,
mais les sorties full et smoke ne doivent pas être mélangées.

## 9. Lire les résultats

Sous le `output_dir` choisi :

| Sortie | Interprétation |
|---|---|
| `prepared/train_folds.csv` | UID, groupe, fold, statut gold et activation |
| `prepared/targets.csv` | Cibles d'entraînement et confiances |
| `prepared/gold_labels.csv` | Vérité gold originale ; inconnues conservées |
| `prepared/fold_diagnostics.csv` | Effectifs et positifs gold par fold |
| `prepared/selected_images.csv` | Série, slot, image et empreinte de fichier sélectionné |
| `prepared/image_errors.csv` | Images illisibles rencontrées |
| `features/<reader>/<split>/` | Caractéristiques float16 et masques ; ne pas éditer |
| `checkpoints/<reader>/foldN/last.pt` | Dernière époque ; aucun choix sur le fold évalué |
| `checkpoints/<reader>/foldN/history.csv` | Perte train et temps par époque |
| `predictions/oof/<reader>.csv` | OOF assemblés uniquement lorsque les cinq folds sont complets |
| `reports/auc_by_finding.csv` | AUC gold et accord avec les cibles weak seuillées |
| `reports/gold_auc_by_fold.csv` | Diagnostic sur de très petits folds gold |
| `reports/evaluation.json` | Macro AUC, interprétation et bootstrap principal |
| `reports/primary_delta_by_finding.csv` | Différence candidat 5 % − baseline à trois lecteurs |
| `reports/rank_correlations_*.csv` | Redondance des ordres ; désaccord ≠ amélioration |
| `submissions/submission_baseline3.csv` | Baseline à trois lecteurs |
| `submissions/submission_candidate05.csv` | Candidat principal, DINOv2 à 5 % |
| `submissions/submission_candidate10.csv`, `submission_candidate15.csv` | Variantes exploratoires |

La **macro AUC gold** est la moyenne des douze AUC sur les 58 OOF gold, pas une
moyenne naïve des minuscules folds. Une AUC est indéfinie si une cohorte ne contient
qu'une classe ; la macro est alors annoncée indisponible, sans cacher le problème.
Les différences de calibration entre têtes de folds peuvent influencer un score
OOF regroupé : il reste exploratoire.

Les weak sont des probabilités issues des rapports, pas une vérité radiologique
indépendante. Leurs AUC sont des mesures **proxy d'accord avec les heuristiques**.
Les coefficients de mélange sont fixés à l'avance ; aucune recherche ni promotion
automatique d'une variante n'utilise les gold.

Le bootstrap resample les groupes gold avec remplacement, sur les prédictions
fixées. Son intervalle à 95 % concerne la différence macro AUC du candidat 5 %
contre la baseline à trois lecteurs. Un intervalle contenant zéro ne démontre pas
de gain. Le bootstrap ne mesure pas toutes les variations d'entraînement et ne
crée pas un test indépendant. Les variantes 10/15 % sont exploratoires.

Les fichiers de soumission portent douze **rangs percentiles**, non des probabilités
calibrées. Les trois exemples test n'ont pas de labels : ils valident les identités,
le format et le parcours d'inférence, **aucune performance Kaggle**. Avec seulement
trois études, les rangs sont nécessairement grossiers.

## 10. Erreurs fréquentes et contrôles

| Message / situation | Cause et action |
|---|---|
| CSV ou `data/` absent | Corriger `data_dir`, puis recommencer l'inspection |
| Poids absents | Exécuter explicitement `assets --reader ...` avec Internet |
| Échec réseau dans assets | Réessayer assets ; aucune extraction ne télécharge automatiquement |
| Configuration / empreinte incompatible | Restaurer les entrées exactes ou choisir un nouveau `output_dir` |
| Cache corrompu | Arrêter et reconstruire dans une nouvelle sortie ; ne pas ignorer le SHA-256 |
| Aucune image utilisable | Lire les dossiers JPEG et les métadonnées de l'étude concernée |
| OOF incomplet | Terminer les cinq folds des quatre lecteurs avant evaluate |
| Prédiction test refusée | Terminer les checkpoints et extraire le split test |
| Macro indisponible en smoke | Le sous-ensemble peut n'avoir qu'une classe ; résultat de parcours uniquement |
| Extraction trop lente | Lire le benchmark ; réduire les lots ou les images/slot dans un nouveau profil |

Contrôles synthétiques et validation du notebook :

```bash
python -m pytest --basetemp=cache/test_tmp
python tests/verify_notebook.py
```

Ces commandes ne lancent ni entraînement des lecteurs, ni extraction des images
réelles, ni notebook complet. La livraison est vérifiée dans l'environnement
existant ; le comportement des véritables poids téléchargés et les calculs complets
restent à vérifier lors de votre exécution.
