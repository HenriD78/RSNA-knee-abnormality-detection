# Protocole commun

Cette spécification complète le [plan principal](../docs/methodology/plan_entrainement_3_piliers.md)
et les trois configurations. L'implémentation se trouve maintenant dans `rsna3/`, avec huit notebooks Kaggle
et des tests locaux. Les mesures CUDA et le contrôle visuel du rendu sur des
études complètes restent obligatoires avant le gros calcul.

## Entrées, folds et provenance

Lire les chemins et empreintes dans [entrees_manifest.json](entrees_manifest.json).
Une seule table de folds fait autorité : celle de `baseline_candidate/artifacts/full/prepared/`.
Les 4 407 UID et les 4 257 groupes de rapports sont conservés. Chaque lecteur
entraîne sur `fold != k`, puis prédit `fold == k`. Aucun ajustement, choix d'époque,
calibration ou sélection de poids de mélange sur ce fold externe.

Les 58 gold ont leurs vrais labels disponibles et confiance 1. Les autres études
gardent les cibles probabilistes et confiances existantes. Une cible manquante
reste inconnue, avec poids nul. Ne pas confondre les NaN des gold non annotés
avec des zéros, ni les faibles probabilités avec des négatifs certains.

L'ordre technique des douze findings reste celui des CSV préparés :

```text
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

Transmettre explicitement cet ordre à la tête, au checkpoint et aux CSV. Relier
toute table par UID, jamais par la position implicite des lignes. Chaque étude
et chacune de ses séries héritent de leur fold maître. Les groupes sont fondés
sur les rapports normalisés ; l'indépendance entre patients n'est pas démontrée.
L'audit antérieur des weak sur les gold reste une limite de provenance.

Une initialisation RSNA publique ayant vu toutes les études ne devient pas
indépendante simplement parce que le nouvel entraînement est hors fold. Elle
est exclue de l'expérience CV principale, de même qu'une distillation depuis
un tel teacher. Les poids génériques autorisés sont référencés dans les JSON.

## Contrat du cache DICOM

Le cache est commun, sans labels et indépendant des folds. Préparer sur CPU :
géométrie réelle, intensités, repère canonique documenté et triplets de coupes
voisines d'une même série. Toute dégradation de géométrie est tracée. Les noms
des JPEG existants ne constituent pas une géométrie 2.5D.

Conserver six slots : sagittal/coronal/axial fat-sat puis non fat-sat. Choisir
la série ayant le plus de coupes exploitables, départager par UID. Un slot absent
reste absent. Choisir quatre centres distincts par slot, ou deux si la projection
du cache dépasse le plafond mesuré de 12 Gio. Si une série est trop courte,
garder seulement ses triplets réellement disponibles, sans duplication silencieuse.

Rendu de référence : champ de 153,6 mm en 384 pixels et intensités uint8 par
percentiles de série 1/99,5. Le centre physique de l'image est le repli initial
déterministe ; il faut contrôler visuellement la couverture du genou. Un vrai
localiseur serait un composant supplémentaire, avec ses poids et sa validation.
Ne pas présenter un crop centré comme un joint anchor appris.

Enregistrer pour chaque fenêtre : UID étude/série, trois SOP UID et indices de
frame éventuels, slot, coordonnées physiques, position relative, orientation,
pixel spacing, paramètres de rendu, validité et motif d'exclusion. Fixer la
convention de position relative après orientation canonique ; la même au test.
Tester les séries de moins de trois coupes, les trous d'espacement, doublons,
multi-frame, MONOCHROME1 et syntaxes de transfert compressées.

Écrire des shards indexés reprenables, avec version et SHA-256. Publier ce cache
privé comme entrée Kaggle, sans l'ajouter aux archives de chaque fold. Le pilote
50–100 études mesure taille et temps CPU avant la préparation de tout le train.

Stockage NPZ : comparer par étude la compression des pixels uint8 directs à
celle de leurs différences horizontales modulo 256, conserver l'archive la plus
petite et reconstruire les pixels par somme cumulative modulo 256 à la lecture.
Cette opération est lossless et ne modifie ni champ, ni résolution, ni fenêtres.
Le format `npz-adaptive-uint8-delta-x-v1` figure dans les contrats probe/cache et
les métadonnées NPZ ; un ancien probe ne permet pas d'admettre ce nouveau format.
Le lecteur accepte aussi les anciens NPZ directs. Garder la marge de 20 % et le
plafond de 12 Gio : seule une nouvelle mesure complète des 75 UID décide de la
faisabilité après cette optimisation de stockage.

## Tenseurs, masques et architecture

- Images : `x[B, 6, K, 3, H, W]`, où les trois canaux sont les coupes précédente,
  centrale et suivante. `B=1` au micro-batch ; `K=1` pour CoAtNet au train, `K=2`
  pour Small et DINOv2. En évaluation, utiliser tous les centres du cache figé.
- Validité : `valid[B, 6, K]`, booléen, **True signifie fenêtre présente**. La
  convention de `src_key_padding_mask` de PyTorch est inverse : passer `~valid`.
- Position : `position[B, 6, K]` et indices de slots. Au train, l'échantillonnage
  des centres est reproductible par étude/époque ; au test, il est déterministe.
- Cibles et poids : `target[B, 12]`, `confidence[B, 12]`. Les UID restent associés
  aux exemples jusqu'à la sortie `logits[B, 12]`.

Encoder uniquement les fenêtres valides, puis replacer leurs caractéristiques
dans les emplacements correspondants. Dimensions attendues avant projection :
CoAtNet 1 024, Small 768, DINOv2 384, à vérifier au forward de contrôle. Retirer
la classification générique ; pour DINOv2, fixer l'utilisation du token CLS et
interpoler les embeddings spatiaux vers la grille 16 x 16 de l'entrée 224.

Projeter à 128, ajouter un embedding de slot et un encodage appris de la position
physique relative. Pour DINOv2 uniquement, appliquer le transformer d'étude
à deux couches et quatre têtes. Il n'y a pas de position fondée sur le numéro
arbitraire du token. Douze attentions indépendantes agrègent ensuite les fenêtres,
une par finding, avant leurs classifications. Appliquer le masque avant chaque
softmax ; une étude entièrement masquée est une erreur explicite.

Une permutation conjointe des tokens, masques, slots et positions doit laisser
les logits inchangés en mode évaluation, à la tolérance numérique près. Un test
qui ne permute que les pixels n'est pas le test de ce contrat.

Après le crop physique, réduction bicubique directe vers H=W=384 ou 224, sans
center crop supplémentaire hérité de la transformation ImageNet. Diviser par 255
puis appliquer les `normalization_mean/std` de chaque JSON. Augmentations cohérentes
entre coupes ; pas de jitter RGB indépendant. Miroir et TTA sont désactivés.

## Perte et accumulation : conserver l'effet des confiances

Pour le groupe effectif de huit études, poser pour chaque finding j :

```text
w_ij = confiance_ij si cible_ij est connue, sinon 0
D_j = somme_i w_ij
L = moyenne_sur_j_avec_D_j>0 [ somme_i w_ij * BCEWithLogits(z_ij, y_ij) / D_j ]
```

Le remplacement numérique d'un NaN avant BCE n'est autorisé qu'avec poids nul.
Un groupe sans aucune cible pondérée doit être diagnostiqué.

**Ne pas normaliser séparément chaque micro-batch d'une étude** : `w * BCE / w`
annulerait la confiance. Charger d'abord les cibles du groupe effectif, calculer
ses douze dénominateurs, puis faire chaque forward/backward avec sa contribution
pondérée aux mêmes dénominateurs. Cela conserve huit études d'accumulation sans
retenir huit graphes. Ne pas diviser de nouveau par huit. Le dernier groupe
incomplet utilise ses vrais dénominateurs ; le scheduler compte les optimizer.step.

AdamW et LR sont dans chaque JSON. Première époque : tête seule. Ensuite, ne
dégeler que le dernier stage CNN ou les deux derniers blocs et la norme DINOv2.
Tracer les noms et nombres de paramètres entraînables ; tester qu'ils reçoivent
des gradients et que les autres n'en reçoivent pas. Maintenir les BatchNorm gelées
en évaluation, y compris après `model.train()`. Le préfixe gelé peut être exécuté
sans gradients ; la partie dégelée ne doit pas être enfermée dans `no_grad()`.

Avec AMP : scaler les contributions, puis unscale, clipping, optimizer.step et
mise à jour du scaler aux frontières des groupes. Avancer le scheduler seulement
si le pas d'optimiseur a réellement eu lieu. Vérifier que les gradients du groupe
effectif pondéré correspondent à une référence en FP32 sur un petit cas synthétique.

## Deux GPU et reprise

Deux processus, un fold par GPU ; le batch effectif reste huit par fold. Commencer
avec un thread CPU et un worker de chargement par processus. Les caches sont
en lecture seule et les sorties des folds distinctes. Un seul coordinateur écrit
le suivi et l'archive de reprise. Pas d'archive écrasée par deux workers.

Sauvegarder atomiquement aux frontières des pas d'optimiseur : modèle, optimiseur,
scheduler, GradScaler, RNG, époque, pas, permutation et offset du sampler. Utiliser
des graines d'augmentation dérivées de `(seed, fold, époque, UID, visite)` pour
rester reproductible malgré le préchargement. Ne pas réutiliser un état préparé
avec d'autres folds, cache, architecture, versions ou paramètres dégelés.

Conserver l'état actif de reprise jusqu'à vérification de l'export final.
Les checkpoints finaux safetensors contiennent l'état complet encodeur + tête,
sans optimiseur ; aucune reconstruction à partir d'un préfixe générique implicite. Inclure version du rendu, modèle exact, ordre des labels et empreintes.
Tester interruption/reprise sur quelques pas avant le gros calcul ; aucun vrai
OOF ne sert à choisir le nombre d'époques. Les époques sont figées au profilage.

## Sorties et critères d'acceptation

Pour chaque pilier : cinq `final_fold{k}.safetensors`, un `oof_<model_id>.csv`
de 4 407 UID uniques, un rapport gold sur 58 études, un proxy faible distinct
sur les 4 349 autres, manifestes, historique réel et coût de quota mesuré.
Pour le proxy AUC faible, seuiller les probabilités de labels à `> 0,5` uniquement
lorsque la cible est connue et sa confiance positive ; les autres sont exclues.
Cela mesure l'accord avec les heuristiques, pas une performance sur vérité clinique.
Une prédiction OOF est produite exclusivement par le modèle n'ayant pas vu son
fold. L'historique externe ne sert pas à retenir une meilleure époque.

Au test, moyenner les probabilités des cinq folds avant le rank blending global
par finding. En OOF, classer les 58 gold séparément des 4 349 weak pour garder
la convention des rapports existants ; au test, classer toute la cohorte cachée.
Rang moyen en cas d'égalité ; normaliser les rangs dans [0,1] avec
une convention explicite pour une cohorte d'une seule étude. Le mélange principal
est 1/3 par famille ; la variante 0,20/0,60/0,20 reste exploratoire. Bootstrap
apparié par groupes pour les différences, sans optimiser les poids sur les 58 gold.

Avant soumission : dynamique des UID cachés, couverture exacte, valeurs finies,
douze colonnes, installation hors ligne, temps complet sous la limite officielle.
En cas de branche optionnelle incomplète, la retirer globalement ; jamais combler
des UID par une constante ou un CSV ancien. Évaluer les trois exemples visibles
comme un contrôle d'exécution, sans AUC de performance.
