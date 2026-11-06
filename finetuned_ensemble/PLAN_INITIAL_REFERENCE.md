> Plan initial conservé à titre historique. Les objectifs et scores cités concernent les références externes ; voir les résultats actuels dans le README du projet. Les liens ont été rendus relatifs lors de la préparation GitHub.

> Archive du plan initial. Pour les décisions actives, lire ../plan_entrainement_3_piliers.md.
> Les estimations et affirmations non vérifiées ci-dessous ne sont pas des consignes d’exécution.

# Plan d'Entraînement des 3 Piliers & Assemblage pour Battre le Score LB 0.958

Ce document constitue le **plan directeur d'entraînement et d'intégration** pour surpasser la baseline d'ensemble à **0.958 Public LB** ([`rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb`](../references/README.md)).

Il est articulé en **4 grandes étapes techniques** :
1. **Étape 1 : Entraînement du Pilier 1 — CoAtNet-2 Raptor Multi-Échelle (384 px + 224 px)**
2. **Étape 2 : Entraînement du Pilier 2 — Modèle Compact R6/R7 (ConvNeXt-Small & Ancrage Articulaire)**
3. **Étape 3 : Entraînement du Pilier 3 — Lecteur 2.5D MIL Orthogonal (DINOv2 / ConvNeXt-Tiny)**
4. **Étape 4 : Protocole d'Ensemblage & Consigne d'Exécution Complète pour Kaggle**

```
                                      ARCHITECTURE DE L'ENSEMBLE CIBLE (> 0.960 LB)
                                                            │
                 ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
                 ▼                                          ▼                                          ▼
   [PILIER 1 : RAPTOR DUAL-SCALE]             [PILIER 2 : COMPACT R6 / R7]               [PILIER 3 : 2.5D MIL ORTHOGONAL]
   • CoAtNet-2 (384 px + 224 px SWA)          • ConvNeXt-Small (352 px joint-anchors)    • Slabs 3 coupes 2.5D (DINOv2 / CNXt)
   • Pré-entraîné OAI + RSNA                  • Pré-entraîné fastMRI + RSNA              • 16 fenêtres glissantes / série
   • Spécialiste : Ligaments, PF OA           • Spécialiste : Ménisques, Lateral OA      • Spécialiste : Continuité 3D & Variance
                 │                                          │                                          │
                 ▼                                          ▼                                          ▼
        Rangs Percentiles                          Rangs Percentiles                          Rangs Percentiles
        [0.65 (384) + 0.35 (224)]                  [0.60 (R6) + 0.40 (R7)]                    (Moyenne des folds)
                 │                                          │                                          │
                 └──────────────────┬───────────────────────┘                                          │
                                    ▼                                                                  │
                      [MÉLANGE DE RANGS PAR PATHOLOGIE]                                                │
                      Optimisation Nelder-Mead par anomalie                                            │
                                    │                                                                  │
                                    └──────────────────────────────────┬───────────────────────────────┘
                                                                       ▼
                                                    [INJECTION ORTHOGONALE (10 - 15 %)]
                                                                       │
                                                                       ▼
                                                        [SUBMISSION FINALE SÉCURISÉE]
```

---

## User Review Required

> [!IMPORTANT]
> **Ressources GPU & Données Externes Requises**
> - **Données externes indispensables** :
>   - Le jeu de données **OAI (Osteoarthritis Initiative)** pour le Pilier 1 (accessible sur NDA / Kaggle).
>   - Le jeu de données **fastMRI Knee** pour le Pilier 2.
> - **Budget calcul estimé pour l'entraînement complet** :
>   - Pilier 1 (CoAtNet 384/224) : ~24 à 36 heures GPU (sur 1× A100 / RTX 4090 ou équivalent multi-T4).
>   - Pilier 2 (ConvNeXt-S 352) : ~18 à 24 heures GPU.
>   - Pilier 3 (Lecteur 2.5D MIL) : ~10 à 14 heures GPU.
> - **Alternative pragmatique** : Utiliser les checkpoints pré-entraînés publics existants (nartaa SWA, heliosli R6/R7) et concentrer l'entraînement complet sur le **Pilier 3 orthogonal** ainsi que sur le **ré-étalonnage fin du Blending**.

> [!WARNING]
> **Règle d'or anatomique sur le Flip TTA (Bilatéralité)**
> Lors de l'inférence miroir horizontale (Flip TTA) sur les coupes coronales et axiales, le compartiment interne (médial) devient externe (latéral). Il est impératif d'intervertir les prédictions `Medial` $\leftrightarrow$ `Lateral` pour les ménisques et l'arthrose (`Medial Meniscus` $\leftrightarrow$ `Lateral Meniscus` et `Medial OA` $\leftrightarrow$ `Lateral OA`). Sans cette permutation, le Flip TTA dégrade le score sur 4 des 12 cibles.

---

## Étape 1 : Entraînement du Pilier 1 — CoAtNet-2 Raptor Multi-Échelle

### 1.1 Objectif et Caractéristiques Clés
Le Pilier 1 capture les détails anatomiques haute résolution grâce à une architecture hybride convolution + auto-attention relative ([`coatnet_rmlp_2_rw_384`](../baseline_candidate/knee_baseline/constants.py)). Il domine les lésions ligamentaires fines (`ACL`, `MCL`) et les atteintes fémoro-patellaires (`PF OA`, `Synovitis`).

### 1.2 Données et Pré-entraînement
1. **Source OAI (Osteoarthritis Initiative)** :
   - 4 796 patients avec IRM multi-séquences longitudinales.
   - Pré-entraînement initial en classification du grade de Kellgren-Lawrence (KLG 0 à 4) ou pré-entraînement contrastif sur les 3 plans (Sagittal, Coronal, Axial).
2. **Jeu de données RSNA Knee** :
   - 4 407 études d'entraînement réparties sur le split unifié 5-folds (groupement par rapport textuel strict documenté dans [`cross_validation_strategy.md`](../docs/methodology/cross_validation_strategy.md)).
   - Exploitation des **58 gold labels** (priorité absolue) et des **4 349 report soft-labels** avec pondération de confiance.

### 1.3 Prétraitement DICOM & Construction des Volumes
- **Résolutions cibles** :
  - **Branche A (384 px)** : Volume redimensionné à $384 \times 384$, couvrant la totalité du champ de vue (FOV) articulaire.
  - **Branche B (224 px)** : Même volume rééchantillonné en $224 \times 224$ avec un recadrage central à 80 % (zoom sur la zone intercondylienne et les ménisques).
- **Organisation en 6 Slots Anatomiques** :
  - Slot 0 : Sagittal Fat-Suppression (PDFS/T2FS)
  - Slot 1 : Coronal Fat-Suppression
  - Slot 2 : Axial Fat-Suppression
  - Slot 3 : Sagittal non Fat-Suppression (T1/PD)
  - Slot 4 : Coronal non Fat-Suppression
  - Slot 5 : Axial non Fat-Suppression
- **Orientation canonique 3D** :
  - Tri des coupes par coordonnée axiale via `ImagePositionPatient`.
  - Normalisation d'intensité par fenêtrage robuste (clipping percentiles 0.5 % et 99.5 % puis scaling $[0, 1]$).

### 1.4 Protocole d'Entraînement & Hyperparamètres
L'entraînement s'effectue en deux sous-modèles distincts (384 px et 224 px) pour former le slot Raptor :
- **Optimiseur** : AdamW ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$).
- **Weight Decay** : $10^{-2}$.
- **Learning Rate Schedule** :
  - Warm-up linéaire sur 2 époques jusqu'à $lr_{\max} = 2 \times 10^{-4}$.
  - Décroissance en cosinus (Cosine Annealing) jusqu'à $10^{-6}$ sur 20 époques.
  - Learning rate différentiel : $0.1 \times lr_{\max}$ sur les blocs convolutifs du backbone, $1.0 \times lr_{\max}$ sur la tête d'attention MIL.
- **Batch Size** : 8 études par GPU (avec accumulation de gradient à 16 études effectives).
- **Fonction de Perte** : BCE pondérée par anomalie avec masque de validité :
  $$\mathcal{L} = \frac{1}{\sum c_i} \sum_{i=1}^{12} c_i \cdot \text{BCE}(p_i, y_i)$$
  où $c_i = 1.0$ pour les gold studies et $c_i = \text{conf}_i$ pour les weak labels.
- **Régularisation & SWA (Stochastic Weight Averaging)** :
  - DropPath / Stochastic Depth : $0.2$.
  - SWA activé sur les 5 dernières époques (taux de rafraîchissement à chaque fin d'époque) pour lisser les minima de la surface d'optimisation.
- **Extrants générés** :
  - Checkpoints 384 px : `raptor_r384_fold[0-4]_swa.pt`
  - Checkpoints 224 px : `raptor_r224_c08_fold[0-4]_swa.pt`

---

## Étape 2 : Entraînement du Pilier 2 — Modèle Compact R6/R7 (ConvNeXt-Small & Joint Anchors)

### 2.1 Objectif et Caractéristiques Clés
Le Pilier 2 (famille R6 / R7 de heliosli) est spécialisé dans les pathologies fémoro-tibiales et poplitées (`Lateral OA`, `Baker's Cyst`, `Medial Meniscus`, `Lateral Meniscus`). Il repose sur un recadrage dynamique centré sur l'interligne fémoro-tibial et des empileurs d'états dédiés.

### 2.2 Pré-entraînement Auxiliaire (fastMRI)
- Utilisation du dataset **fastMRI Knee** (milliers de séquences axiales, sagittales et coronales).
- Tâche auxiliaire : classification de coupes normales vs pathologiques et régression des coordonnées des plateaux tibiaux.

### 2.3 Prétraitement avec Ancrage Articulaire (Knee-Joint Anchors)
1. **Localisation de l'Interligne (Joint Anchor)** :
   - Un modèle léger pré-localise le centre articulaire $(x_c, y_c, z_c)$ entre les condyles fémoraux et les épines tibiales.
2. **Crop Isotrope 352×352** :
   - Découpe d'une boîte englobante de taille $352 \times 352$ centrée sur l'interligne, éliminant les portions inutiles du fût fémoral et de la métaphyse tibiale distale.
3. **Fenêtres 2.5D de 3 Coupes** :
   - Regroupement des coupes en triplets consécutifs $[z-1, z, z+1]$ formant des tenseurs à 3 canaux.

### 2.4 Architecture & Modules Dédiés
- **Backbone** : `convnext_small.fb_in22k_ft_in1k` (convolutions 7×7 modernes à large champ récepteur).
- **Stackers d'États Spécialisés (State Stackers)** :
  - Deux sous-réseaux MLP dédiés sont branchés en parallèle sur les caractéristiques globales pour prédire spécifiquement :
    - `Effusion` (épanchement liquidien intra-articulaire)
    - `Baker's` (kyste poplité rétro-tibial)
  - Ces deux anomalies dépendent de volumes liquidiens volumineux mieux capturés par un pooling global moyen/max que par une attention locale.

### 2.5 Protocole d'Entraînement
- **Validation** : 4 folds (schéma R6) ou 5 folds (schéma R7 unifié).
- **Optimiseur** : AdamW ($lr = 1 \times 10^{-4}$, weight decay $5 \times 10^{-2}$).
- **Scheduler** : OneCycleLR ou Cosine Annealing sur 25 époques.
- **Augmentations de données** :
  - Rotations faibles ($\pm 10^\circ$), translation ($\pm 8\%$), zoom ($\pm 10\%$).
  - Random Erasing / Cutout léger simulant des artéfacts métalliques ou de mouvement.
- **Entraînement avec Flip TTA natif** :
  - Pendant la validation, chaque volume est évalué sous sa forme standard et sous sa forme miroir horizontale.
  - Application immédiate de la correction d'inversion des labels latéraux et médiaux.
- **Extrants générés** :
  - Checkpoints R6 : `compact_r6_fold[0-3].pt`
  - Checkpoints R7 : `compact_r7_fold[0-4].pt`

---

## Étape 3 : Entraînement du Pilier 3 — Lecteur 2.5D MIL Orthogonal

### 3.1 Objectif et Caractéristiques Clés
Le Pilier 3 a pour rôle d'**injecter de l'orthogonalité** dans un ensemble dominé par ConvNeXt et CoAtNet. L'analyse du notebook 0.958 a démontré que le lecteur ConvNeXt-Tiny original apportait un gain modeste (10 %) parce qu'il partageait la même famille d'architecture que R6.
Pour battre 0.958, le Pilier 3 doit utiliser :
- Soit **DINOv2-Small** (`vit_small_patch14_dinov2`) basé sur des représentations auto-supervisées de vision-transformer (déjà exploré dans [`0-941-sota-rsna-knee-dinov2-radimagenet.ipynb`](../references/README.md)).
- Soit un **Swin Transformer V2** (`swinv2_tiny_window16_256`).

### 3.2 Prétraitement des Volumes Canoniques
- **Standardisation Physique à 0.4 mm/pixel** :
  - Chaque coupe est rééchantillonnée via interpolation bicubique pour qu'un pixel corresponde exactement à 0.4 mm dans le plan anatomique.
- **Tri et Fenêtrage Glissant (Sliding Windows)** :
  - 16 fenêtres régulières de 3 coupes consécutives réparties sur l'épaisseur complète de chaque série.
- **Gestion des Séries Manquantes (Slot-Fill)** :
  - Si une série manque pour un slot (ex: absence de Coronal Fat-Sat), elle est automatiquement remplacée par la série Coronal non-Fat-Sat avec un embedding d'indicateur de substitution.

### 3.3 Architecture Multi-Instance (MIL)
```
  [16 fenêtres × 6 slots = 96 tokens d'images]
                         │
                         ▼
             [Backbone DINOv2 / Swin]  (Projection d = 256)
                         │
                         ▼
        [Transformer Encoder Étude (2 couches)]
                         │
                         ▼
        [12 Têtes d'Attention Pooling Dédiées]
           (Une tête par anomalie clinique)
                         │
                         ▼
              [12 Logits par Étude]
```

### 3.4 Protocole d'Entraînement
- **Holdout des Gold Studies** :
  - Les 58 études gold ne sont **jamais** utilisées pendant les 15 premières époques d'entraînement pour éviter tout surapprentissage des têtes MIL.
  - L'apprentissage repose sur les pseudo-labels extraits des rapports textuels par NLP.
- **Fine-Tuning de Calibration (Époques 16 à 20)** :
  - Déblocage d'un taux d'apprentissage très faible ($5 \times 10^{-6}$) intégrant les gold studies OOF avec un poids de perte unitaire ($c=1.0$).
- **Asymmetric Loss / Focal Loss** :
  - Pour compenser la rareté des classes `Fracture` et `Contusion` :
    $$\mathcal{L}_{\text{asym}} = - y (1-p)^\gamma \log(p) - (1-y) p^\gamma \log(1-p)$$
    avec $\gamma=2$ sur les classes minoritaires.
- **Extrants générés** :
  - Checkpoints Pilier 3 : `orthogonal_reader_fold[0-4].pt`
  - Prédictions OOF complètes : `oof_orthogonal_reader.csv`

---

## Étape 4 : Protocole d'Ensemblage & Consigne d'Exécution sur Kaggle

Cette étape détaille comment interconnecter les modèles et exécuter l'inférence complète dans l'environnement contraint de Kaggle (GPU unique T4/P100, 9 heures maximum, pas d'accès Internet).

### 4.1 Stratégie de Fusion par Rangs Percentiles (Rank Blending)

Les probabilités brutes des différents modèles ne doivent **jamais** être moyennées arithmétiquement : elles proviennent d'architectures différentes avec des calibrations sigmoïdes discordantes.
On utilise une conversion en rangs percentiles par colonne sur les études d'inférence :

$$\text{rank}(x_i) = \frac{\text{Rang Moyen}(x_i)}{N}$$

#### Algorithme de Fusion en 3 Étages :
1. **Étage 1 — Rangs intra-piliers** :
   - Slot Raptor :
     $$\mathcal{R}_{\text{Raptor}} = 0.65 \times \text{rank}(\text{CoAtNet}_{384}) + 0.35 \times \text{rank}(\text{CoAtNet}_{224})$$
   - Slot Compact :
     $$\mathcal{R}_{\text{Compact}} = 0.60 \times \text{rank}(\text{R6}) + 0.40 \times \text{rank}(\text{R7})$$
2. **Étage 2 — Fusion bipolaire avec Poids par Pathologie** :
   Pour chaque anomalie $c \in \{1, \dots, 12\}$ :
   $$\mathcal{R}_{\text{Base}}(c) = w_c \cdot \mathcal{R}_{\text{Raptor}}(c) + (1 - w_c) \cdot \mathcal{R}_{\text{Compact}}(c)$$
   *Matrice de poids $w_c$ étalonnée sur les OOF* :
   - `Synovitis` : $0.80$ (avantage Raptor)
   - `PF OA` : $0.60$ (avantage Raptor)
   - `Medial OA` : $0.50$ (égalité)
   - `ACL` / `MCL` : $0.55$ (léger avantage Raptor 384)
   - `Baker's` : $0.35$ (avantage Compact)
   - `Lateral OA` : $0.20$ (avantage Compact)
   - `Medial Meniscus` / `Lateral Meniscus` : $0.40$ (avantage Compact 352)
   - `Contusion` / `Fracture` / `Effusion` : $0.50$
3. **Étage 3 — Injection Orthogonale du Pilier 3** :
   $$\mathcal{R}_{\text{Final}}(c) = \text{rank}\Big( 0.88 \cdot \mathcal{R}_{\text{Base}}(c) + 0.12 \cdot \text{rank}(\text{Pilier 3}(c)) \Big)$$

---

### 4.2 Consignes Opérationnelles pour le Run Kaggle (Inférence en < 9h)

```
                                CHRONOGRAMME DU RUN KAGGLE (BUDGET MAX : 8.2 HEURES)
0h00                               3h00                               6h15               7h30     8h00     8h15
 ┌──────────────────────────────────┬──────────────────────────────────┬──────────────────┬────────┬────────┬───────┐
 │ Étape 1 : Raptor 384 + 224       │ Étape 2 : Compact R6 + R7        │ Étape 3 : Pil. 3 │ Blend  │ Vérif. │ Fin   │
 │ • DICOM parsing                  │ • Joint-anchor cropping          │ • Volumes 0.4 mm │ & TTA  │ CSV    │ Prêt  │
 │ • 2 checkpoints SWA              │ • Inférence R6 + R7              │ • 16 fenêtres    │ Rangs  │ test   │ pour  │
 │ • Export raptor_raw.csv          │ • Flip TTA corrigé               │ • Export pil3.csv│ par c. │ intègre│ score │
 └──────────────────────────────────┴──────────────────────────────────┴──────────────────┴────────┴────────┴───────┘
                                                                       ▲
                                            Garde-Fou (Deadline Guard) : Si T > 7h15,
                                            le Pilier 3 est ignoré et le blend A3 est écrit.
```

#### A. Préparation des Datasets Kaggle Hors-Ligne
- Créez un dataset privé Kaggle contenant :
  1. Les poids des 3 piliers (`.pt` et `.safetensors`).
  2. Les archives de code source nécessaires (`own_src/`, `raptor_run.py`, modules d'ancrage).
  3. Les wheels Python hors-ligne : `timm-1.0.29-py3-none-any.whl`, `iterative_stratification-0.1.9-py3-none-any.whl`, `pylibjpeg`, `python-gdcm`.
- Internet désactivé (`Internet off`) dans les paramètres du notebook.

#### B. Gestion Séquentielle des Processus & Garde-Fou Temporel
- Chaque pilier s'exécute dans un **sous-processus Python isolé** (`subprocess.run`) pour libérer intégralement la mémoire VRAM GPU et la RAM CPU entre chaque étape :
  - `subprocess.run([sys.executable, 'raptor_run.py', ...])` $\rightarrow$ libère 100 % de la VRAM.
  - `subprocess.run([sys.executable, 'compact_run.py', ...])` $\rightarrow$ libère 100 % de la VRAM.
  - `subprocess.run([sys.executable, 'orthogonal_run.py', ...])` $\rightarrow$ libère 100 % de la VRAM.
- **Deadline Guard** :
  ```python
  T0 = time.time()
  DEADLINE_HOURS = 8.15 # Seuil de sécurité strict

  if time.time() - T0 > DEADLINE_HOURS * 3600:
      print("Alerte Budget Temps : Étape 3 court-circuitée, soumission Part A3 préservée.")
  ```

#### C. Gestion des Cas d'Échec et Robustesse (Fallback)
- Si une étude DICOM échoue ou est corrompue pour un des modèles :
  - **Interdiction formelle de laisser des NaN ou de remplir naïvement par 0.5**.
  - Remplacement de la valeur manquante par le rang percentile obtenu par le modèle jumeau sur cette même étude.
  - Si tous les modèles échouent sur une étude (cas d'image entièrement illisible) : imputation par la médiane de la colonne sur le jeu test.

#### D. Contrôles Finaux d'Intégrité de `submission.csv`
Avant de terminer le run, le notebook doit exécuter les assertions strictes suivantes :
```python
sub = pd.read_csv("submission.csv")
assert len(sub) == len(test_ids), f"Nombre de lignes incorrect: {len(sub)} vs {len(test_ids)}"
assert list(sub.columns) == ["StudyInstanceUID"] + LABELS, "Colonnes non conformes"
assert sub[LABELS].isna().sum().sum() == 0, "Présence de NaN dans la soumission"
assert (sub[LABELS].values >= 0.0).all() and (sub[LABELS].values <= 1.0).all(), "Valeurs hors [0, 1]"
assert np.isfinite(sub[LABELS].values).all(), "Valeurs non finies détectées"
print("SUCCÈS: submission.csv vérifié et prêt pour scoring.")
```

---

## Plan de Vérification & Métriques Cibles

### Métriques d'Étape par Pilier (Validation OOF sur les 58 Gold Studies)
| Pilier / Modèle | Résolution & Entrées | Macro AUC Gold Seul | Rôle dans l'Ensemble |
| :--- | :--- | :---: | :--- |
| **Pilier 1 : Raptor CoAtNet** | 384 px + 224 px SWA (OAI) | **$\ge 0.949$** | Pilier lourd, vision globale haute fidélité |
| **Pilier 2 : Compact R6/R7** | 352 px Joint-Anchors (fastMRI) | **$\ge 0.956$** | Spécialiste interligne fémoro-tibial & kystes |
| **Pilier 3 : Reader 2.5D** | 3 coupes consécutives (DINOv2/Swin) | **$\ge 0.935$** | Orthogonalité de représentation & continuité 3D |
| **Ensemble Fusionné (Étape 4)** | Rank Blend pondéré par finding | **$\ge 0.960$** | **Objectif de score pour battre 0.958** |

### Protocole de Test Local (Sans GPU Cloud)
Pour valider le pipeline avant envoi sur Kaggle :
1. Exécuter le profil smoke test localement :
   `python -m pytest tests/`
2. Valider la syntaxe et les assertions du script de blend sur des prédictions synthétiques :
   `python tests/test_blends.py`
3. Vérifier que l'archive finale `UPLOAD_TO_KAGGLE.zip` contient tous les manifests et poids requis.
