# Plan d'entraînement des trois piliers — budget Kaggle de 24 heures

Révision du **19 octobre 2026**. Objectif : entraîner trois familles sur nos données,
mesurer leur apport et livrer une soumission valide avant la clôture. Le score
0,958 du notebook public est une référence annoncée au leaderboard, pas une
promesse de résultat. Notre référence actuelle est **0,730246 de macro AUC OOF
sur les 58 gold**, mesurée sur une autre cohorte.

Le plan initial est conservé dans
[`PLAN_INITIAL_REFERENCE.md`](../../finetuned_ensemble/PLAN_INITIAL_REFERENCE.md).
Cette version corrige ses hypothèses de calcul, d'anatomie et de validation.
**La présente livraison prépare le plan et les dossiers ; elle ne lance aucun
entraînement et ne contient pas encore les futurs notebooks d'entraînement.**

## 1. Décisions retenues

Avant les entraînements, conserver une version de soumission déjà fonctionnelle
et ses assets hors ligne. Si la recette publique est utilisée, la vérifier telle
quelle avant d'y ajouter nos modèles. Les nouveaux entraînements ne doivent pas
remplacer l'unique solution de repli avant leur export et leur test d'inférence.

Trois modèles précis, chacun avec cinq folds communs :

| Pilier | Modèle générique de départ | Entrée | Adaptation initiale | Priorité de calcul |
|---|---|---:|---|---:|
| 1 | CoAtNet-2 `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | 384 | Dernier stage + tête MIL | 2 |
| 2 | ConvNeXt-Small `convnext_small.fb_in22k_ft_in1k` | 224 | Dernier stage + tête MIL | **1** |
| 3 | DINOv2-Small `vit_small_patch14_dinov2.lvd142m` | 224 | Deux derniers blocs + norme finale + tête MIL | 3 |

Les poids génériques sont déjà disponibles dans `baseline_candidate/assets/`.
Leurs révisions et SHA-256 sont repris dans chaque configuration de planification.
Small passe en premier : c'est notre meilleur lecteur gelé individuel (0,7293),
et le mélange actuel à 70 % de CoAtNet descend à 0,7229. Les coefficients d'un
modèle public spécialisé ne se transfèrent pas automatiquement à un autre modèle
portant le même nom de backbone.

Le périmètre des 24 heures ne comprend pas un nouvel apprentissage OAI, fastMRI,
un second CoAtNet à 224, un second compact R7, un Swin supplémentaire, un localiseur
articulaire entraîné de zéro ou une recherche d'hyperparamètres. Le nouveau
ConvNeXt est une adaptation de notre modèle ; on ne l'appellera pas « R6/R7 »,
qui désignent des recettes et checkpoints spécifiques d'autres auteurs.

Deux usages des poids doivent rester distincts :

- **Expérience CV principale** : initialisation générique, entraînement hors fold,
  trois familles fine-tunées, OOF comparables à notre baseline.
- **Socle public de soumission, optionnel** : recette Raptor 384 + 224 et R6 déjà
  spécialisés, avec leurs poids, code et prétraitement exacts. Le notebook indique
  que Raptor a vu toutes les études train ; ces poids ne permettent donc pas
  d'annoncer une nouvelle validation indépendante sur les mêmes 58 gold.
  Toute adaptation de ces poids serait une expérience séparée et explicitement
  identifiée. La recette publique inchangée reste le point de repli correspondant.

## 2. Échéances et budget : 24 heures au total

La page officielle consultée annonce la clôture le **22 octobre 2026 à 23:59 UTC**,
soit le **23 octobre à 01:59 à Paris**. Le compte doit déjà être éligible à la
compétition ; l'inscription annoncée était close le 15 octobre.

Objectifs internes : code et recettes figés le **21 octobre à 22:00 à Paris**,
puis lancement de la soumission finale le **22 octobre à 12:00 au plus tard**.
Une exécution peut durer jusqu'à neuf heures et attendre dans une file : garder
cette avance même si les tests visibles ne contiennent que trois études.

### 2.1 Enveloppes de quota

Les chiffres ci-dessous sont des **plafonds alloués**, pas des durées mesurées ni
une garantie d'achèvement. Ils s'appliquent au quota restant affiché par le compte.
Les 24 heures ne sont ni 24 heures par pilier, ni 24 heures par T4.

| Poste | Plafond de quota | Résultat attendu |
|---|---:|---|
| Profilage, intégrité et essai de reprise | 1 h | Débit réel, mémoire, facteur de décompte du quota |
| ConvNeXt-Small, cinq folds | 6 h | Cinq fins d'entraînement + OOF |
| CoAtNet-2, cinq folds | 5 h | Cinq fins d'entraînement + OOF |
| DINOv2-Small, cinq folds | 4 h | Cinq fins d'entraînement + OOF |
| Validation, relecture et éventuel rejeu OOF | 1 h | Couverture, métriques, incertitude |
| Export, tests d'inférence et soumission | 4 h | Version Kaggle complète et vérifiée |
| Réserve d'incident et de reprise | **3 h** | Marge conservée jusqu'à la fin |
| **Total** | **24 h** | |

Les contrôles purement CPU se font sans accélérateur. Une enveloppe non consommée
reste disponible en réserve : ne pas la dépenser en variantes non prévues.
Le cache DICOM est préparé sur CPU avec l'accélérateur désactivé, hors de cette
enveloppe GPU ; son coût en temps réel et en stockage doit tout de même être mesuré.

### 2.2 Pilote obligatoire avant les quinze entraînements

Sur des études des quatre folds train du pilote :

1. Contrôler les pixels, les six slots, les triplets et la séparation des groupes.
2. Mesurer chargement, forward, backward, accumulation et écriture de checkpoint
   avec la véritable précision et la partie d'encodeur dégelée. Notre ancien
   benchmark d'extraction gelée sur seize images n'est pas une mesure de ce coût.
3. Mesurer après chauffe au moins une centaine de micro-batches si le budget le
   permet, en incluant des études denses et un cache froid. Relever la mémoire
   maximale de chaque T4 et le temps de rechargement d'un fold.
4. Noter le quota avant/après pour mesurer `q = quota_déduit / temps_écoulé`.
   Ne pas supposer le mode de décompte de T4 x2 ou des re-runs de scoring.
5. Tester interruption et reprise, puis restaurer les poids initiaux pour le
   véritable entraînement ; ne pas conserver discrètement les mises à jour du pilote.

Pour cinq folds et deux GPU, simuler la file des cinq tâches. Si leurs durées sont
proches, prévoir environ **trois vagues**, plus les chargements, les OOF et une
marge de 20 %. Multiplier le temps écoulé estimé par `q` pour le comparer à l'enveloppe.
Le nombre d'époques par modèle est arrêté **avant toute évaluation du fold externe**.

Cibles initiales : Small 6 époques, CoAtNet 3, DINOv2 4, dont une époque de chauffe
de tête. Le minimum est deux époques : une de tête et au moins une d'adaptation
de l'encodeur. Si le pilote ne permet pas cinq folds dans l'enveloppe, réduire
les époques avant de démarrer, puis les fenêtres d'entraînement si nécessaire.
Si le pilote montre au contraire une marge importante, augmenter le nombre
d'époques **avant** les évaluations externes, dans le plafond du pilier, et inscrire
cette décision pour les cinq folds. Ne pas conserver artificiellement trois
époques si le débit mesuré permet une adaptation plus longue. Si même le minimum
ne tient pas, marquer le pilier non réalisable dans ce budget ;
conserver les modèles complets déjà acquis et le socle de repli. Ne pas présenter
une couverture partielle comme un OOF complet ou inventer cinq checkpoints.

## 3. Validation commune : reprendre la table existante

Autorité unique :
`baseline_candidate/artifacts/full/prepared/train_folds.csv`.
Les empreintes et celles des annotations sont enregistrées dans
[`entrees_manifest.json`](../../finetuned_ensemble/entrees_manifest.json).

| Fold | Études | Gold |
|---|---:|---:|
| 0 | 920 | 12 |
| 1 | 875 | 11 |
| 2 | 880 | 13 |
| 3 | 871 | 11 |
| 4 | 861 | 11 |
| **Total** | **4 407** | **58** |

- Graine 42 ; 4 257 groupes de rapports normalisés Unicode NFKC, espaces normalisés
  et casse ignorée. Toutes les séries d'une étude héritent du même fold.
- La stratification déjà faite distingue douze indicateurs positifs gold,
  douze indicateurs faibles `> 0,5` et la présence d'une étude gold, au niveau
  des groupes avec `MultilabelStratifiedKFold`.
- Ne pas relancer le générateur simplifié de `cross_validation_strategy.md` qui
  remplace tous les labels inconnus par zéro. Ne pas recopier les folds dans data/.
- Pour le fold k, entraîner seulement sur les quatre autres folds. Les gold de
  ces quatre folds peuvent être employés dès le début avec confiance 1.
  Les gold du fold k ne servent jamais au fine-tuning, à une calibration finale,
  au choix de l'époque ou à un ajustement d'hyperparamètres.
- Si un réglage a réellement besoin de validation, créer un sous-ensemble interne
  aux quatre folds train, toujours par groupes. Aucun choix sur le fold externe.
- Les métadonnées actuelles établissent la séparation des études et rapports,
  pas une indépendance démontrée entre patients. Les règles d'annotation faible
  ont déjà été auditées sur les gold dans le projet : communiquer cette provenance.

Les OOF finaux attendus couvrent **4 407 UID distincts par modèle**, avec fold,
group_id et provenance. Une AUC par fold avec une seule classe reste indéfinie ;
ne pas la remplacer par zéro. La macro AUC gold principale est calculée sur les
58 gold OOF réunis, séparément du proxy faible sur les 4 349 autres études.

## 4. Prétraitement DICOM et cache commun

### 4.1 Géométrie réelle et mêmes opérations au train et au test

Le local dispose surtout de JPEG et de petits fixtures DICOM. Lire les DICOM
complets depuis la compétition attachée sur Kaggle ; aucun téléchargement des
centaines de Go sur le Mac n'est requis par ce plan.

- Décoder les niveaux de gris avec les codecs hors ligne vérifiés. Gérer les
  transferts compressés et multi-frame ; journaliser les erreurs.
- Appliquer la convention d'intensité choisie, y compris MONOCHROME1 et
  RescaleSlope/Intercept, de façon identique au train et à l'inférence.
- Trier les coupes par `dot(ImagePositionPatient, normal)`, où la normale vient
  du produit vectoriel des deux axes d'`ImageOrientationPatient`. Le seul axe z
  de l'IPP ne convient pas à tous les plans. Ne pas trier par nom de fichier.
- Réorienter les images dans un repère canonique documenté, vérifier le côté et
  utiliser `PixelSpacing` pour le recadrage en millimètres. Tout repli sur
  InstanceNumber ou une géométrie absente est explicite et comptabilisé.
- Cible initiale commune : champ de 153,6 mm, rendu 384 px, soit 0,4 mm/px dans
  le cache. Les lecteurs 224 reçoivent une réduction de ce même champ ; leur
  résolution finale n'est donc plus 0,4 mm/px.
- Fenêtrage robuste fixé par série, par exemple percentiles 1 et 99,5, puis uint8.
  Les statistiques sont propres à la série, pas apprises sur le fold évalué.
- Construire de vrais triplets de coupes voisines dans une série. Contrôler les
  doublons SOP UID, les grands trous d'espacement et les changements d'orientation.
  Ne jamais construire un triplet à cheval sur deux séries.

Ce nouveau prétraitement diffère du min–max/JPEG 224 de la baseline. Il exige
un nouveau cache et de nouveaux poids adaptés. Les caches de caractéristiques
et les têtes entraînées précédemment ne doivent pas être réétiquetés comme
compatibles. Leur version est conservée comme référence.

### 4.2 Six slots, positions et séries absentes

Slots : sagittal/coronal/axial avec suppression de graisse, puis les trois mêmes
sans suppression de graisse. Retenir la série avec le plus de coupes exploitables,
avec départage par UID. Le modèle connaît le slot et la position physique relative
normalisée du centre de la fenêtre ; le numéro de token n'est pas une position.

Les slots absents restent masqués. Ne pas relabeller une série non fat-sat comme
fat-sat pour remplir une case. Un mécanisme de substitution serait une ablation
ultérieure avec provenance et apprentissage cohérents. Une étude entièrement
inexploitable arrête le parcours avec diagnostic ; elle ne devient pas un ancien
CSV d'exemple ou un vecteur constant silencieux.

### 4.3 Taille, temps CPU et persistance du cache

Un cache complet de volumes 384 px peut dépasser largement le stockage disponible.
Avant l'export, mesurer temps de décodage et taille compressée sur 50 à 100 études
représentatives, choisies sans examiner leurs labels. Mesurer le disque réellement
libre ; ne pas supposer qu'un gros volume temporaire sera sauvegardé.

- Cache initial proposé : quatre centres répartis par série retenue, chaque centre
  gardant ses trois coupes, en uint8 384. Les centres de bord évitent les triplets
  incomplets ; leur règle est figée avant l'entraînement.
- Cible de paquet persistant : **au plus 12 Gio**, empreintes/index compris, avec
  une marge sur le disque réellement disponible. Si la projection excède ce seuil,
  passer à deux centres par slot avant toute création de cache définitif.
- Ces quatre centres correspondent à un maximum d'environ **46,8 Go non compressés**
  pour 4 407 études à six slots ; deux centres environ 23,4 Go. La compression est
  à mesurer, pas à supposer. Si deux centres ne tiennent toujours pas, revoir la
  représentation avant de lancer le GPU et ne pas produire un cache incomplet.
- Exporter des shards indexés de taille limitée, par exemple environ 1 Gio chacun,
  avec SHA-256, UID, métadonnées géométriques et code de rendu versionné. Éviter les
  copies répétées de dizaines de milliers de fichiers pendant chaque époque.
- Publier le cache en Dataset privé réutilisable. Les entraînements le lisent en
  entrée ; leurs archives de résultats ne recopient ni les DICOM ni ce cache.
- Dédupliquer les études et réutiliser le cache entre folds et modèles. Le cache
  est sans labels ; la séparation des cibles intervient dans les DataLoaders.

Une défaillance de codec ne doit pas être absorbée comme des milliers d'images
corrompues. Tester les syntaxes de transfert avant la préparation entière. La
préparation CPU doit être relançable par shard et s'arrêter si la projection en
temps compromet l'échéance, même si elle ne consomme pas de quota GPU.

## 5. Fine-tuning des trois familles

Spécifications de référence : fichiers `experience_planifiee.json` dans chacun
des trois sous-dossiers. Elles ne sont pas encore des configurations de CLI.

### 5.1 Entraînement commun

- Une étude par micro-batch et par GPU ; accumulation sur huit études. Deux GPU
  travaillant sur deux folds indépendants ne donnent pas un batch de seize pour
  un modèle : chaque fold garde son batch effectif de huit.
- Première époque : encodeur gelé, tête neuve initialisée avec `42 + fold`.
  Époques suivantes : dégel de la partie désignée de l'encodeur et de la tête.
  Les modules gelés, notamment leurs BatchNorm, conservent leurs statistiques.
- AdamW, weight decay 0,01 hors biais/normes, clipping du gradient à 1,0.
  Learning rates distincts encodeur/tête ; AMP FP16 avec GradScaler.
- Après dégel : warm-up sur 5 % des pas d'optimiseur, puis décroissance cosinus
  jusqu'à 0,1 du learning rate initial. Un warm-up de deux époques n'est pas
  adapté à un entraînement de seulement trois époques.
- Époque et checkpoint finaux fixés avant le fold externe. Pas d'early stopping
  ou de « meilleur checkpoint gold » sur ce fold.
- Transformations géométriques et gain/gamma modestes, identiques pour les trois
  coupes d'une fenêtre et cohérents dans la série. Aucun jitter RGB indépendant.
- BCE par finding pondérée par confiance ; gold=1, weak=confiance existante,
  cible manquante=poids nul. Pas de nouveau label négatif créé par remplacement NaN.
  **Normaliser sur les huit études du batch effectif**, par finding, et non sur
  chaque micro-batch d'une étude : cette dernière opération annulerait la confiance.
  Calculer les dénominateurs avec les cibles des huit études avant leurs backward,
  sans garder huit graphes en mémoire. Le dernier groupe incomplet emploie son
  effectif réel ; aucune division supplémentaire par huit après cette normalisation.
- Après le rendu physique, réduire directement à la résolution du lecteur en
  bicubique, sans second center crop implicite. Appliquer moyenne/écart-type
  des poids génériques, enregistrés dans les configurations : CoAtNet 0,5/0,5,
  Small et DINOv2 selon leur convention ImageNet. Même contrat au train et au test.
- EMA, SWA, focal loss, stackers d'états et miroir sont désactivés dans la première
  expérience. La formule de focal symétrique du plan initial n'était pas une
  justification d'asymmetric loss. Ces ajouts attendent une ablation distincte.

### 5.2 CoAtNet-2 — cinq heures maximum

Résolution native 384 ; une fenêtre par slot présent à l'entraînement, soit au
plus six fenêtres par étude, puis toutes celles du cache figé à l'évaluation.
Cible : trois époques, une tête puis deux d'adaptation du dernier stage.
Learning rates initiaux : **5e-6 encodeur / 3e-4 tête**. Le coût haute résolution
impose cette couverture réduite et le pilote ; la branche 224 et SWA sont hors
périmètre de départ. Cette variante ne reproduit pas le modèle Raptor spécialisé.

### 5.3 ConvNeXt-Small — six heures maximum, exécuté en premier

Résolution 224 ; deux fenêtres par slot présent à l'entraînement. Cible : six
époques, une tête puis cinq d'adaptation du dernier stage. Learning rates :
**1e-5 encodeur / 3e-4 tête**. Tête MIL légère, projection 128 et douze pools.
Le recadrage physique centré est la base ; les vrais joint anchors sont une
extension uniquement si les poids publics du localiseur, le code, la latéralité
et le même rendu train/test sont disponibles et testés avant le gel des expériences.
Le nom R6/R7, les quatre folds d'un modèle public ou ses state stackers ne doivent
pas être attribués à cette nouvelle variante.

### 5.4 DINOv2-Small — quatre heures maximum

Résolution 224, divisible par 14, avec adaptation vérifiée des embeddings de
position spatiale. Deux fenêtres par slot, projection 128, transformer d'étude
à deux couches et quatre têtes, douze attentions par finding. Dégeler les deux
derniers blocs et la norme finale. Learning rates : **3e-6 encodeur / 1e-4 tête**.
Cible : quatre époques, une tête puis trois d'adaptation. La différence de famille
ne garantit pas une diversité utile : mesurer les erreurs, corrélations et gains.

Ces hyperparamètres sont des points de départ préinscrits, pas des valeurs
optimales établies. Tout changement dicté par la mémoire ou le débit est arrêté
au pilote et propagé à tous les folds avant leur lancement.

## 6. Exploiter les deux T4 et reprendre sans gaspiller le quota

Le lancement utilisera deux processus indépendants, chacun limité à un GPU via
CUDA_VISIBLE_DEVICES. Il s'agit de répartir les folds, pas de synchroniser les
gradients du même modèle par DDP. Une file prioritaire distribue les tâches Small,
puis CoAtNet, puis DINOv2 ; les dépendances restent explicites. Un GPU libre peut
prendre une tâche admissible suivante, sans modifier les folds ou la recette.

- Démarrer avec un worker de chargement et un thread CPU par processus GPU,
  puis ajuster uniquement selon le pilote pour éviter la saturation des CPU.
- Caches en lecture seule, répertoires distincts par modèle/fold ; une seule
  autorité coordonne les index, le budget et les archives. Pas de tar partagé
  écrasé simultanément par deux workers.
- Les reprises vérifient données, folds, cache, poids initiaux, architecture,
  versions et liste des paramètres dégelés. Un changement de contrat est une
  nouvelle expérience, pas une reprise forcée.
- Sauvegarder aux frontières d'optimizer.step : modèle, optimiseur, scheduler,
  scaler, RNG CPU/CUDA, permutation/sampler, position validée et accumulateur
  remis à zéro. Les augmentations doivent être reproductibles par
  `(seed, fold, époque, UID, visite)` malgré le préchargement des workers.
- Une échéance de session garde une marge pour sauvegarder et publier les sorties.
  Un arrêt dur peut empêcher la dernière sauvegarde : ne pas attendre la dernière minute.
- À la fin d'un fold, conserver les poids finaux safetensors et les OOF ; l'état
  d'optimiseur des folds terminés peut être retiré du paquet d'inférence. Conserver
  seulement les états actifs nécessaires à la reprise pour maîtriser le stockage.
- Save Version → Save & Run All pour les calculs autonomes ; arrêter les anciennes
  sessions interactives du benchmark. Conserver les outputs/version et attacher
  la dernière reprise à la session suivante. Fermer l'onglet n'est pas une sauvegarde.

Aucun réglage de performance, changement de seed ou nouveau modèle ne doit
consommer les quatre heures réservées à la soumission et les trois heures de
réserve sans une décision explicite motivée par un incident.

## 7. Mélanges et validation : distinguer expérience et soumission

### 7.1 Ensemble propre des trois nouveaux lecteurs

Pour chaque lecteur, les probabilités des cinq folds sont moyennées sur le test
**avant** le calcul des rangs. Une moyenne de probabilités n'est pas interdite par
principe ; le rank blending est le choix de cette expérience pour comparer des
échelles différentes. Les rangs sont globaux par finding, avec rang moyen des égalités,
jamais calculés séparément dans chaque lot GPU. Pour rester comparable à nos
rapports actuels, l'évaluation gold classe sa cohorte de 58 études et le proxy
faible sa cohorte de 4 349 études, séparément. L'inférence classe tout le test réel,
pas les seuls trois exemples visibles ni les études lot par lot.

Mélange principal préinscrit : **un tiers de rang CoAtNet, un tiers Small, un tiers
DINOv2**, puis classement final. Une seule variante exploratoire est prévue :
**0,20 CoAtNet / 0,60 Small / 0,20 DINOv2**. Elle n'est pas choisie automatiquement
sur les mêmes 58 gold. Les coefficients par finding et Nelder-Mead sont retirés
du parcours principal : les folds communs ne protègent pas du surapprentissage
des poids de mélange sur les OOF.

Comparer les trois lecteurs, le mélange principal et la baseline gelée avec AUC
gold par finding, macro AUC globale, corrélations et bootstrap apparié par groupes
(2 000 réplications valides). Les labels faibles restent un diagnostic distinct.
Une différence dont l'intervalle traverse zéro n'établit pas un bénéfice net.

### 7.2 Socle public et apport expérimental

Si les assets publics sont disponibles, reproduire d'abord la recette source
sans adaptation : deux Raptor 384/224 mélangés 0,65/0,35, **R6 seul** dans la recette
principale décrite le 16 octobre, TTA exact et poids publics par finding.
Le mélange R6/R7 0,60/0,40 n'est pas la configuration principale de cette version.
Les poids Raptor annoncés sont 0,8 Synovitis, 0,2 Lateral OA, 0,6 PF OA,
0,35 Baker's et 0,5 pour les autres findings ; ils n'ont pas été établis par notre CV.
Le Tiny du notebook à « un dixième » signifie un dixième de ses anciens poids
par finding (environ 0,3 à 3,5 %), pas un gain de 10 % ni une injection uniforme.

Conserver le socle public comme soumission séparée. Un candidat additionnel,
fixé avant ses résultats, peut être `rank(0,97 * rank(socle_public) + 0,03 *
rank(notre_ensemble))`. Il reste expérimental : ne pas annoncer son évaluation
locale sur les 58 gold comme indépendante si le socle les a vus. Ne pas distiller
un teacher public entraîné sur tous les folds puis qualifier l'OOF étudiant de
strictement sans fuite ; il faudrait des teachers eux-mêmes hors fold.

### 7.3 Le miroir anatomique n'est pas une règle universelle

Pour nos nouveaux modèles, miroir/TTA sont **désactivés au départ**. Une expérience
miroir exige une définition physique et sémantique testée : en sagittal, un miroir
peut inverser l'ordre des canaux de coupe ; en coronal/axial, il peut inverser l'axe
largeur après normalisation anatomique. Positions relatives et vues locales doivent
suivre la transformation. Dans le worker R6 fourni, la permutation médial/latéral
est définie par son propre contrat et MCL garde la passe non miroir. Cela ne justifie
pas d'intervertir ces labels après n'importe quel flip d'une image de genou.
Tout miroir doit être une involution testée et avoir le même sens au train et au test.

## 8. Inférence de compétition et repli

Le futur notebook de soumission charge les quinze poids **réellement terminés**
et leur code exact. Chaque poids final de cette expérience contient l'état
complet de l'encodeur et de la tête ; les poids génériques séparés ne sont nécessaires
qu'à l'initialisation des entraînements. Pour un socle public, garder
les environnements et rendus propres à chaque recette dans des sous-processus
isolés ; ne pas imposer notre timm 1.0.29 à un worker publié avec une autre version.
Installer hors ligne dans des répertoires dédiés, conserver le torch CUDA du
service, vérifier SHA-256, GDCM et codecs avant l'inférence. Libérer un processus
ne libère pas les autres processus GPU encore actifs : le coordinateur les suit.

- Relire les véritables UID et séries de test à chaque run. Les trois exemples
  visibles ne mesurent ni la vitesse du test entier ni une performance leaderboard.
- Mesurer sur un échantillon représentatif le temps complet : décodage, rendu,
  anchors éventuels, modèles, TTA, rangs et écritures. Faire une estimation avec
  marge avant d'admettre une branche optionnelle ; les latences de démarrage et
  torch.compile ne se multiplient pas aveuglément par le nombre d'études.
- Objectif de temps interne : fin vers huit heures au plus, avec au moins
  45 minutes de marge avant la limite officielle de neuf heures. La réserve de
  quota et le temps de scoring sont suivis séparément.
- Une branche optionnelle n'est lancée que si sa durée projetée rentre dans le
  temps et le quota restants. Si elle échoue, garder le socle complet déjà vérifié,
  inscrire la recette réellement exécutée et ne pas prétendre avoir utilisé le pilier.
- Ne pas fusionner une table à couverture partielle comme si elle couvrait tout
  le test. Le repli simple est global : retirer la branche optionnelle en entier.
  Un socle absent ou incomplet arrête le run ; aucune médiane ou constante n'est
  injectée silencieusement pour fabriquer une soumission complète.
- Vérifier UID uniques, couverture et ordre exacts de test.csv, douze noms de
  findings, valeurs finies dans [0,1]. Écrire `/kaggle/working/submission.csv`
  atomiquement après validation. Ne pas laisser un ancien CSV après une erreur.

La limite de neuf heures concerne la soumission de compétition ; le plafond
éventuel d'une session de formation est une contrainte distincte à lire dans
l'interface. La soumission reste Internet désactivé et doit être lancée en
Save & Run All avant Submit to Competition.

## 9. Livrables, ordre des journées et vérifications

Les trois sous-dossiers et leurs rôles sont décrits dans
[`README_FR.md`](../../finetuned_ensemble/README_FR.md).
Le fichier `suivi_folds.csv` contient quinze tâches **planifiées**, pas quinze
entraînements effectués. Aucun AUC cible artificiel ne sert de critère d'acceptation.

| Moment visé, heure de Paris | Livrable / décision |
|---|---|
| 19 soir / 20 matin | Implémenter le minimum, contrôler folds/labels/poids, préparer le cache CPU et mesurer taille/temps |
| 20 | Pilote des trois familles ; figer cache, paramètres, nombre d'époques et budget ; lancer Small puis les autres tâches admissibles |
| 21 | Terminer les folds possibles dans les plafonds, valider OOF, exporter les poids et figer le code avant 22:00 |
| 22 avant 12:00 | Version de soumission finale lancée, avec repli complet et temps de scoring conservé |
| 22 ensuite | Vérifier statut et sortie, traiter les incidents avec la réserve ; éviter les nouvelles recettes non testées |

Contrôles indispensables avant le gros calcul : splits/groupes, UID et labels
manquants, géométrie sur de vrais DICOM, triplets sans franchissement de série,
masques, permutation conjointe tokens/positions, forward+backward avec AMP,
paramètres dégelés et BatchNorm, reprise après optimizer.step, absence de collision
entre GPU, contraintes du disque et projection de quota. Le notebook final doit
être validé syntaxiquement et s'ouvrir en inspection.

À la fin : garder une soumission déjà fonctionnelle, les manifestes, les poids
finaux, les OOF complets disponibles, le relevé réel de quota/temps et les limites
des expériences non terminées. Les générateurs temporaires de notebook seront
supprimés ; les scripts de vérification pourront rester.

## Sources et références

- [Référence du notebook analysé](../../references/README.md) : recettes publiques, prétraitement, TTA et limites déclarées ; la copie intégrale reste locale.
- [Résultats de notre session](../results/frozen_baseline_evaluation.json) : baseline OOF et bootstrap.
- [Guide CV original](cross_validation_strategy.md) : conserver les principes de séparation avec les corrections explicites ci-dessus.
- [Compétition officielle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) : échéance, nom du CSV, Internet désactivé et limite de neuf heures.
- [Conseils GPU Kaggle](https://www.kaggle.com/page/GPU-tips-and-tricks) : quota, sessions batch et arrêt des sessions interactives inutilisées.
- [Transfert et fine-tuning PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) : distinguer encodeur gelé et adaptation des poids.
