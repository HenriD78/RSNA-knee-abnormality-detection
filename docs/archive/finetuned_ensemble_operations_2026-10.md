> Archive du guide opérationnel d’octobre 2026. Les dates, états de progression et empreintes concernent cette période. Les paquets binaires mentionnés sont des artefacts locaux, absents de Git. Le guide courant est [celui de l’ensemble](../../finetuned_ensemble/README_FR.md).

# Trois piliers fine-tunés : code et notebooks Kaggle

Le pipeline du [plan principal](../methodology/plan_entrainement_3_piliers.md) est implémenté
dans [rsna3/](../../finetuned_ensemble/rsna3). Aucun entraînement sur les DICOM complets n'a été lancé
sur ce Mac. Les epochs, le débit AMP/T4, le facteur de quota et la faisabilité
restent à mesurer sur Kaggle ; aucun score nouveau n'est annoncé.

## Démarrer maintenant

Le dossier `dist/rsna3-training-py313` est le paquet d'entrée
Kaggle prêt à téléverser en **Dataset privé** : code, trois poids génériques,
CSV autoritaires et huit wheels Linux/Python 3.13. Il ne contient ni DICOM
complets, ni cache d'images de la baseline, ni faux checkpoints fine-tunés.
Une `dist/rsna3-training-py313-upload.zip` est également disponible (594 Mio).
Créer un Dataset privé avec cette archive, puis importer les notebooks suivants dans l'ordre.

| Notebook | Accélérateur | Action |
|---|---|---|
| [00_preparer_cache_cpu.ipynb](../../finetuned_ensemble/00_preparer_cache_cpu.ipynb) | **None** | Probe 75 études par défaut ; cache complet ensuite avec BUILD_FULL_CACHE=True |
| [01_piloter_et_figer.ipynb](../../finetuned_ensemble/01_piloter_et_figer.ipynb) | T4 ×2 | Profilage des trois familles sur train seulement, reprise AMP, puis gel des epochs à partir du quota mesuré |
| [02_entrainer_deux_gpu.ipynb](../../finetuned_ensemble/02_entrainer_deux_gpu.ipynb) | T4 ×2 | File prioritaire Small → CoAtNet → DINOv2, cinq folds par modèle |
| [03_evaluer_exporter.ipynb](../../finetuned_ensemble/03_evaluer_exporter.ipynb) | CPU puis GPU | OOF complets, baseline candidate05, bootstrap groupé, export et benchmark complet d'inférence |
| [04_soumettre.ipynb](../../finetuned_ensemble/04_soumettre.ipynb) | GPU | Installation hors ligne, quinze poids réels, test dynamique et `submission.csv` atomique |

Chaque pilier possède aussi un notebook `entrainer_<model_id>.ipynb` pour lancer
sa propre file de cinq folds. Utiliser un seul coordinateur par répertoire de
sorties. La numérotation des dossiers est celle du plan ; Small reste prioritaire.

Tous les notebooks désactivent Internet. Ils détectent les entrées attachées et
s'arrêtent en cas d'ambiguïté : renseigner alors les chemins en tête des cellules.
Les valeurs `None` de quota sont intentionnelles : il faut les mesures affichées
par **votre compte**, pas un facteur inventé par le code.

## Corriger l'upload existant `cpu-probes` sur Python 3.13

Le premier paquet avait des codecs compilés pour Python 3.12 ; le traceback
Kaggle fourni confirme un interpréteur Python 3.13. L'archive historique
`rsna3-training-upload.zip` reste inchangée et ne doit pas être réutilisée seule
pour cet environnement.

1. Créer un nouveau Dataset privé nommé `rsna3-python313-patch` avec le
   `dist/rsna3-python313-patch.zip`, environ 23 Mio.
2. Attacher ce Dataset au notebook existant, en gardant le Dataset `cpu-probes`.
3. Remplacer toute la première cellule par cette ligne, puis l'exécuter :

```python
%run /kaggle/input/datasets/henridargent/rsna3-python313-patch/fix_first_cell.py
```

Le script utilise l'upload existant
`/kaggle/input/datasets/henridargent/cpu-probes`, installe les wheels compatibles
avec le Python réel, vérifie les codecs et recrée `PROJECT`, `WORK` et `cli` pour
la suite du notebook. Garder l'accélérateur sur **None**. Les poids n'ont pas à
être téléversés à nouveau. Le script ne télécharge rien et ne remplace ni Torch
ni NumPy. Pour un autre chemin du Dataset initial, définir `RSNA3_BUNDLE_ROOT`
dans l'environnement avant cette ligne.

Les huit notebooks locaux détectent également un patch attaché grâce à son
`wheels_manifest.json` et chargent son bootstrap. Les recherches d'entrées
antérieures n'ont pas été modifiées. Le patch ajoute les wheels 3.13 à celles
3.12 de l'upload initial ; seule la variante compatible est passée à pip.

## Parcours opérationnel

1. CPU : attacher compétition + paquet training. Exécuter 00 avec accélérateur
   désactivé. Le premier Run All effectue uniquement le probe sur 75 études :
   `BUILD_FULL_CACHE=False` reste désactivé. Lire les estimations de temps et taille
   affichées, puis passer ce drapeau à `True` dans la cellule de préparation pour
   créer le cache. Contrôler visuellement le genou et la latéralité. Sauvegarder le
   probe et publier le cache comme Dataset privé réutilisable. Les licences et
   droits de réutilisation restent ceux des données de compétition.
2. Pilote : attacher cache + training, exécuter 01 en mode `profile`. Il mesure
   séparément tête et dégel, dont 104 micro-batches après chauffe pour le dégel,
   avec accumulation huit et un worker réel. Il teste un rechargement et compare
   la mise à jour suivante ; ses poids sont jetés. Noter quota et durée de session
   avant/après. Le cache « froid » signifie premier passage observé ; le cache
   système du service n'est pas vidé artificiellement.
3. Dans une version `freeze` de 01, renseigner ces trois valeurs et le dossier de
   profils sauvegardé. Le gel simule cinq tâches sur deux GPU, marge 20 %, et
   retient le plus grand nombre d'epochs admissible jusqu'au plafond préalable
   `MAX_EPOCHS=12`. Une recette impossible est marquée telle quelle. Réduire les
   fenêtres et refaire un pilote si même deux epochs ne tiennent pas.
4. Attacher les recettes figées à 02. Renseigner le quota **actuel** et la limite
   de session de formation affichée. Save Version → **Save & Run All**. Garder
   8 h de quota pour validation, export/soumission et réserve. Les budgets des
   piliers sont aussi appliqués pendant l'exécution ; l'estimation par q est
   conservatrice quand les familles se chevauchent. Réconcilier avec le quota réel.
5. Pour reprendre, attacher les derniers outputs et renseigner `RESUME_FROM`.
   Seuls les résultats et états actifs sont copiés ; le cache reste en lecture
   seule. Même code, versions, folds, cache, initialisation et recette sont exigés.
   Conserver les mêmes identifiants de modèles ; ne pas retoucher une recette
   après ses premières évaluations externes.
6. Après les quinze folds, lancer 03. Le rapport distingue 58 gold et 4 349 weak,
   inclut AUC indéfinies, corrélations et 2 000 bootstrap appariés valides. Le
   mélange principal reste 1/3 ; 20/60/20 reste exploratoire, sans sélection.
   L'export exige les quinze poids et les 4 407 OOF de chaque lecteur.
7. Dans une version GPU de 03, activer `RUN_BENCHMARK` et renseigner q : mesurer
   **décodage + rendu + quinze modèles** sur 50 études train choisies par UID,
   sans métrique externe. Ce résultat enrichit le manifest d'admission. Publier
   le paquet final **après** ce benchmark. Il inclut les wheels, le code et les
   encodeurs+têtes complets ; aucun optimiseur/cache/DICOM n'est nécessaire.
8. Lancer 04 avec le paquet final. Fin visée avant 8 h, marge interne de deux
   minutes pour écrire un éventuel repli ; toujours au moins 45 minutes avant
   la limite de neuf heures. Tester la version visible, puis Save & Run All et
   Submit to Competition. Le CSV suit exactement l'ordre de `test.csv`.

Le benchmark complet appartient au poste export/inférence, pas au budget
profilage des trois fine-tunings. Ne pas le relancer sans changement justifiant
une nouvelle mesure. La préparation CPU a son échéance propre, hors quota GPU.

Après les pilotes Kaggle du 20 octobre, le parcours direct est
[notebooks/02_entrainer_ensemble_pilote.ipynb](../../finetuned_ensemble/notebooks/02_entrainer_ensemble_pilote.ipynb).
Il reprend le même runtime que le cache/pilotes, vérifie les trois profils complets
du dernier notebook de pilotes et fige les recettes avant de lancer la file.
L'écriture des recettes se fait dans les notebooks avec une copie indépendante
de la configuration du pilote : le snapshot mesuré reste intact et le JSON ne
contient pas de référence circulaire. Ne pas appeler le `freeze` CLI de l'ancien
Input immuable ; le gel des notebooks corrigés est la voie opérationnelle.

Les durées rapportées admettent 7/9/10 époques (Small/CoAtNet/DINOv2) dans les
plafonds de 6/5/4 h, avec marge de 20 %. Somme conservatrice des projections :
14,01 h de quota ; simulation de la file commune sur deux GPU : environ 12,21 h
écoulées, avant réconciliation avec le débit simultané réel. Le quota restant
rapporté est de 23 h 57 min. Le notebook prévoit une première session de huit
heures ; il peut ensuite reprendre automatiquement un unique `rsna3-runs`
attaché, avec les recettes originales et checkpoints sauvegardés. Actualiser
`QUOTA_AVAILABLE` avant chaque run. Les résumés `training_admission.json` et
`training_session_summary.json` distinguent admission, pause et quinze folds
effectivement terminés. Une pause de session ne vaut pas achèvement.

Après la première session de 7 h 51, douze folds sont terminés : les cinq
ConvNeXt, les cinq CoAtNet et les folds DINOv2 0/1. Le notebook
[notebooks/02_reprendre_ensemble.ipynb](../../finetuned_ensemble/notebooks/02_reprendre_ensemble.ipynb) est prêt pour
reprendre avec les **16 h 03 de quota restant rapportées**. Attacher les outputs
sauvegardés de cette session, en conservant le paquet, le patch et le cache CPU
en Inputs. Il exige les douze résultats et les checkpoints DINOv2 2/3, récupère
les recettes et budgets d'origine, puis laisse la file terminer les trois folds
incomplets. Configurer T4 ×2 et Save Version → Save & Run All ; actualiser le
quota si sa valeur a changé. La durée de huit heures est un plafond de session,
pas une durée imposée. Attendre quinze folds complets avant l'évaluation/export.

Après la reprise, le parcours prêt à importer est :

1. [notebooks/03_evaluer_exporter_cpu.ipynb](../../finetuned_ensemble/notebooks/03_evaluer_exporter_cpu.ipynb), avec
   **T4 ×2 pour garder les versions de l'entraînement**, `cpu-probes`, son patch et uniquement les derniers outputs
   d'entraînement. Le notebook vérifie les quinze folds, évalue les OOF puis
   exporte les poids complets. Il inclut le bootstrap corrigé et les wheels
   compatibles sélectionnées dans le paquet et son patch ; aucun cache ou DICOM
   n'est requis. L'environnement doit garder les versions de l'entraînement.
2. [notebooks/03b_mesurer_inference_gpu.ipynb](../../finetuned_ensemble/notebooks/03b_mesurer_inference_gpu.ipynb),
   T4 ×2, avec les outputs de l'étape CPU et la compétition. Il mesure cinquante
   études, enrichit le manifest d'admission et recalcule ses empreintes hors
   ligne. Le quota initial est réglé sur les quatorze heures restantes signalées.
3. [04_soumettre.ipynb](../../finetuned_ensemble/04_soumettre.ipynb), GPU, Internet désactivé, avec
   **les outputs de l'étape GPU 03b** et la compétition. Le paquet CPU seul
   n'a pas encore le profil d'admission requis par la soumission. Voir ci-dessous
   la correction de débit requise par la mesure réelle du 21 octobre.

Utiliser Save Version → Save & Run All à chaque étape. Un statut Kaggle
« successful » de l'entraînement ne suffit pas : les quinze marqueurs de folds
terminés sont exigés avant toute évaluation ou export. Si le montage des outputs
échoue encore, un dataset privé contenant les mêmes dossiers est accepté.

La mesure série réelle du 21 octobre donne **37,038 s/étude**, dont environ
29 minutes de préparation pour cinquante études ; le chargement des quinze
modèles ne représente que 13 secondes. Avec marge de 20 %, cette mesure ne couvre
qu'environ **644 études** en huit heures : la réussite du benchmark ne garantit
pas le passage sur le test caché. À titre de scénario, 1 300 études représenteraient
environ seize heures. Ne pas lancer la soumission série sur cette seule base.

Le parcours corrigé est
[notebooks/03c_mesurer_inference_parallele.ipynb](../../finetuned_ensemble/notebooks/03c_mesurer_inference_parallele.ipynb),
T4 ×2, avec le paquet issu de 03b et la compétition. Quatre processus appellent
le rendu et l'écriture de cache d'origine ; la fusion préserve exactement les
NPZ par étude. Le module [rsna3_parallel_inference.py](../../finetuned_ensemble/rsna3_parallel_inference.py)
reste hors de `rsna3`, donc l'empreinte d'entraînement n'est pas modifiée.
Une nouvelle mesure complète sur cinquante études doit établir le débit réel :
aucun facteur d'accélération n'est supposé. Les recherches d'Inputs de ces
notebooks ciblent les chemins Kaggle connus sans parcourir l'arbre DICOM.

Après cette mesure, utiliser
[notebooks/04_soumettre_parallele.ipynb](../../finetuned_ensemble/notebooks/04_soumettre_parallele.ipynb) avec
**le nouveau paquet de 03c**. Le helper, son empreinte, le nombre de processus et
la taille des lots sont enregistrés dans le profil. Le notebook série refuse
ce profil plus rapide. Le notebook parallèle relit le vrai `test.csv` et refuse
un temps projeté incompatible. Les poids, fenêtres, pixels et mélange sont
conservés ; la baseline reste le candidat principal après la régression gold
observée (0,68957 pour le nouveau mélange contre 0,73025).

La mesure Kaggle de 03c a ensuite confirmé **12,491 s/étude** sur les mêmes
cinquante UID, cache non réutilisé, avec quatre processus et lots de quatre.
Le scénario de 1 300 études représente **5 h 25 avec marge de 20 %** ; le plafond
projeté de huit heures couvre 1 912 études. Le helper mesuré porte l'empreinte
`b2652044efc6a20d6a69ee9a08378427e9ff468c0b42d6ce78be3651b23067f7`.
Le résultat est consigné dans
[verification/kaggle_parallel_inference_20261021.json](../../finetuned_ensemble/verification/kaggle_parallel_inference_20261021.json).
La suite est donc 04 parallèle, avec **le paquet issu de cette mesure 03c**.
Le notebook de soumission repère les seuls CSV de test : aucun fichier train
n'est nécessaire pour découvrir la compétition pendant le scoring.

Le notebook d'export garde son nom historique `cpu`, mais le runtime CPU Kaggle
utilise Torch `2.11.0+cpu` alors que l'entraînement utilisait `2.11.0+cu128` :
l'export réel doit employer T4 ×2 afin de conserver le contrat de versions.

## Contrats effectivement implémentés

Correction du 19 octobre — tags L/R entre séries : le notebook CPU conserve
maintenant ces désaccords comme avertissements, sans choisir arbitrairement un
côté ni modifier le rendu. Une cellule autonome de récupération est disponible
dans [patches/cell_fix_laterality.py](../../finetuned_ensemble/patches/cell_fix_laterality.py) : copier tout son
contenu dans une cellule après le setup, puis relancer le probe. Elle active une
copie corrigée du code dans `/kaggle/working` pour le kernel **et les subprocess**,
avec les mêmes sources que les notebooks à jour ; les gros poids et CSV restent
dans l'Input attaché. Aucun nouvel upload n'est requis. Une copie prête à importer
du notebook CPU est aussi disponible dans `notebooks/00_preparer_cache_cpu_corrige.ipynb`.
Les notebooks d'entraînement à jour activent la même correction afin de conserver
la concordance des empreintes avec le nouveau cache. La version de rendu est v2 :
refaire le probe avant tout cache définitif. Le premier Run All reste un probe
CPU uniquement.

Si le probe à deux centres dépasse encore 12 Gio, le stockage utilise désormais
une prédiction horizontale uint8 réversible, comparée à l'encodage direct pour
chaque étude. Seule la plus petite archive est retenue et le lecteur restaure
exactement les pixels d'origine. Résolution 384, champ 153,6 mm, centres et
triplets sont conservés. Le contrat distingue ce format de stockage du rendu.
Le plafond de 12 Gio et les marges de 20 % restent identiques : refaire le probe
avec le nouveau code, sans modifier les résultats de mesure précédents.

Pour une session Kaggle déjà ouverte avec l'ancien Input, copier tout le contenu
de [patches/cell_fix_cache_and_probe.py](../../finetuned_ensemble/patches/cell_fix_cache_and_probe.py) dans une
cellule **après le probe existant**, puis exécuter cette cellule. Elle active la
correction et mesure les mêmes 75 UID avec deux centres. Elle fixe
`BUILD_FULL_CACHE=False` et ne prépare pas le cache complet. Aucune installation
supplémentaire ni nouvel upload de poids n'est requis. Utiliser les notebooks à
jour pour les étapes suivantes afin de conserver la même empreinte de source.

- Folds maîtres/labels/poids vérifiés par SHA-256, jointures exclusivement par UID,
  groupes disjoints, gold confiance 1 et inconnus poids nul.
- DICOM single/multi-frame, codecs vérifiés avant décodage, rescale et
  MONOCHROME1, tri par normale IOP, dédoublonnage SOP/frame et position physique,
  trous d'espacement et changements d'orientation. Une contradiction L/R dans
  une même série exclut celle-ci. Entre séries retenues d'une étude, elle produit
  un avertissement explicite dans le probe, le manifest du cache et les NPZ ;
  aucun côté n'est déduit par vote, signe de l'IPP ou rapprochement de repères
  différents. Les tags, leur source et les FrameOfReferenceUID sont conservés.
  Ces études demandent un contrôle visuel avant l'entraînement : les tags seuls
  ne démontrent pas qu'elles montrent un même genou.
  Géométrie absente : **exclusion journalisée**, sans inventer PixelSpacing ou
  voisins depuis InstanceNumber. Étude vide : erreur bloquante.
- Repère LPS documenté dans [dicom.py](../../finetuned_ensemble/rsna3/dicom.py), obliquité conservée,
  champ 153,6 mm, cache uint8 384 par percentiles de série. Le centre de l'image
  est un crop physique déterministe, pas un localiseur entraîné. Les détails
  fenêtre/SOP/frame/IPP/IOP/PixelSpacing/latéralité/exclusions sont dans chaque NPZ.
- Shards ZIP stockés contenant des NPZ compressés, index et SHA-256 par shard et
  par étude. Transactions aux frontières de shard ; une préparation interrompue
  conserve les études finies du shard courant. Cache complet obligatoire.
- Six slots masqués, seules fenêtres valides encodées, position physique apprise,
  douze attentions MIL. DINOv2 : CLS, grille 16×16 interpolée, transformer d'étude
  2 couches/4 têtes. Normalisations des poids ; réduction bicubique directe.
- Tête seule une époque, puis dernier stage CNN ou deux blocs/norme DINOv2.
  Statistiques BatchNorm fixes ; préfixe gelé en eval. AdamW, decay excluant
  biais/normes, AMP+GradScaler, clipping, warmup 5 % puis cosinus.
- BCE normalisée par finding **sur les huit études effectives avant backward**,
  sans seconde division par huit ; dernier groupe incomplet conservé. Checkpoint
  aux frontières de pas : RNG CPU/CUDA, permutation, offset, optimiseur,
  scheduler et scaler. Augmentations déterministes par UID/epoch/fold.
- OOF uniquement après l'epoch finale préfixée. Au test : moyenne probabiliste
  des cinq folds puis rangs globaux par finding. Égalités : rang moyen ; singleton
  0,5. Aucun TTA/miroir, EMA, SWA, distillation ou optimisation de mélange.

## Repli public et solution existante

La [soumission existante](../../baseline_candidate/online_kaggle/submission/GUIDE_SOUMISSION_FR.md)
est conservée. Les poids Raptor/R6 spécialisés ne sont pas les trois poids
génériques présents : leur recette n'est donc pas prétendue reproduite ici.

L'inférence accepte un manifest `verified_external_submission_recipe`, avec
`files` (chemins relatifs → SHA-256), `argv` (liste, sans shell) et éventuellement
`env`. Les remplacements `{recipe_root}`, `{output_csv}`, `{test_csv}`,
`{series_csv}` et `{dicom_root}` transmettent les chemins au worker isolé de la
recette exacte. Celui-ci doit déjà avoir ses poids, son environnement hors ligne,
son rendu et son TTA vérifiés. Le coordinateur suit et arrête tout son groupe de
processus. Pas d'installation de notre timm dans son environnement.

Le socle est vérifié intégralement avant la branche propre. Le candidat public
97 % / propre 3 % doit être demandé explicitement et disposer de temps et quota
mesurés. Une erreur de branche retire cette branche **globalement** ; un socle
absent/incomplet arrête le run. Aucun résultat public sur nos gold n'est annoncé
comme validation indépendante.

## CLI et vérifications locales

Depuis la racine du projet :

```bash
PYTHONPATH="finetuned_ensemble" .venv/bin/python -m rsna3 --help
PYTHONPATH="finetuned_ensemble" .venv/bin/python -m rsna3 check --project .
.venv/bin/python -m pytest "finetuned_ensemble/tests" -q
.venv/bin/python "finetuned_ensemble/tests/verify_notebooks.py"
python3 "finetuned_ensemble/verifier_plan.py"
```

Pour reconstruire le paquet après une modification, choisir un nouveau dossier :

```bash
.venv/bin/python "finetuned_ensemble/build_kaggle_bundle.py" \
  --destination "finetuned_ensemble/dist/rsna3-training-v2"
```

Les wheels présentes ciblent **Linux x86_64 CPython 3.13**. Si l'image Kaggle
emploie une autre version Python, télécharger un ensemble adapté avec
`--download-wheels --python-version 3.12 --wheels <nouveau-dossier>`, puis
reconstruire avec `--wheels <nouveau-dossier>`. Le bootstrap conserve Torch CUDA
et NumPy du service ; il ne lance aucun téléchargement. Les versions réelles de
l'environnement sont figées au pilote, puis exigées à la reprise/inférence.

[VERIFICATIONS_FR.md](../../finetuned_ensemble/VERIFICATIONS_FR.md) distingue les tests exécutés sur Mac
et les contrôles restant à réaliser sur Kaggle. Les JSON `experience_planifiee`
restent les spécifications initiales ; les commandes `freeze` produisent les
**vraies recettes exécutables**, séparées. Le [protocole](../../finetuned_ensemble/PROTOCOLE_COMMUN_FR.md),
les manifestes d'origine, [budget_24h.json](../../finetuned_ensemble/budget_24h.json) et le suivi initial
sont conservés comme références. Les générateurs temporaires de notebooks ne
sont pas conservés.
