# Lancer la baseline sur un GPU en ligne — guide débutant

Le paquet prépare la même expérience : 4 407 études, cinq folds identiques,
huit JPEG maximum par slot et vingt époques pour chaque tête. Il conserve aussi
les **513 études CoAtNet déjà calculées sur votre Mac**.

Les encodeurs restent gelés. Le GPU accélère surtout l'extraction, qui prenait
plusieurs jours sur CPU. Aucun compte n'est créé, aucun paiement ni calcul distant
n'a été lancé lors de la préparation de ces fichiers.

## Les trois fichiers à repérer

Dans ce dossier :

1. **`dist/UPLOAD_TO_KAGGLE.zip`** : le Dataset à téléverser une fois. Il contient
   les JPEG sélectionnés, les folds/labels, les quatre poids, le code, deux petites
   dépendances hors ligne et les features CPU terminées. Environ 3 à 4 Go.
2. **`KAGGLE_GPU.ipynb`** : le notebook à importer dans Kaggle.
3. **`COLAB_GPU.ipynb`** : une alternative utilisant le même ZIP sur Google Drive.

Vous n'avez pas besoin de téléverser `data/` entier, la `.venv`, le lock macOS,
vos anciens notebooks ou vos identifiants Kaggle. Le paquet comporte seulement
170 776 JPEG sélectionnés (170 672 train, 104 test), environ 2,3 Go d'images.
Les rapports textuels originaux ne sont pas nécessaires : les groupes sont déjà figés.

Les fichiers `.bin` dans le ZIP sont des archives, **pas des exécutables** : leur
extension empêche Kaggle de les décompresser automatiquement une deuxième fois.

## A. Créer votre compte et téléverser le Dataset

1. Aller sur [kaggle.com](https://www.kaggle.com/) et créer/se connecter à votre compte.
2. Si Kaggle demande une vérification du compte ou du téléphone pour le GPU, la terminer.
3. Choisir **Create → New Dataset** (ou Datasets → New Dataset).
4. Déposer **uniquement `UPLOAD_TO_KAGGLE.zip`** dans la zone de téléversement.
5. Donner un nom, par exemple `rsna-baseline-gpu-inputs`, et garder le Dataset **Private**.
6. Attendre la fin du téléversement et du traitement. Les fichiers accessibles doivent
   inclure `bundle_manifest.json`, `data_payload.bin`, `assets_payload.bin`,
   `seed_payload.bin`, `code_payload.bin` et le dossier `wheels/`.

La vitesse de cet envoi dépend de votre connexion. Garder le ZIP original pour
une utilisation ultérieure sur Colab ou un autre service.

## B. Importer le notebook et activer le GPU

1. Choisir **Create → New Notebook**.
2. Dans le menu **File → Import Notebook**, sélectionner `KAGGLE_GPU.ipynb`.
3. Dans **Input / Add Input**, rechercher le nom de votre Dataset privé et l'ajouter.
4. Dans **Settings / Session options → Accelerator**, choisir un **GPU NVIDIA** disponible.
   Le code utilise le premier GPU ; il ne fait pas de calcul distribué si deux T4 sont fournis.
5. Garder **Internet désactivé** : les poids et les deux wheels nécessaires sont inclus.
6. Vérifier le quota GPU restant affiché par Kaggle. Il dépend du compte et peut évoluer.

Ne pas sélectionner TPU : ce code utilise CUDA. Le notebook s'arrête immédiatement
s'il ne détecte pas de GPU ; il ne recommence pas silencieusement sur CPU.

## C. Commencer par un vrai benchmark GPU

Dans la première cellule de configuration, garder :

```python
ACTION = "benchmark"
SESSION_HOURS = 6
READERS = ["coatnet", "small", "tiny", "dino"]
BUNDLE_DIR = ""
RESUME_FILE = ""
```

Exécuter les cellules dans l'ordre. Elles vont :

- Localiser votre Dataset et vérifier les SHA-256 du paquet.
- Installer **seulement timm 1.0.29 et iterative-stratification 0.1.9** hors ligne,
  sans remplacer le PyTorch/torchvision CUDA fourni par Kaggle.
- Décompresser les JPEG/poids dans le stockage **temporaire local** de la machine.
- Mesurer les quatre encodeurs sur seize images chacun et afficher une estimation.

La décompression et la première lecture peuvent durer quelques minutes. Le
benchmark ne fait pas l'extraction entière et n'entraîne aucune tête. Ses chiffres
permettent de décider si votre quota disponible suffit ; le temps CPU de votre Mac
ne permettait pas d'estimer ce quota GPU.

## D. Lancer le calcul réel et sauvegarder

Après le benchmark, changer seulement :

```python
ACTION = "run"
```

Puis utiliser **Save Version → Save & Run All** pour le calcul réel. Vérifier que
le GPU et le Dataset sont bien associés à la version sauvegardée. Ce mode exécute
le notebook côté Kaggle ; vous pouvez suivre son statut sans maintenir un calcul
sur votre Mac. L'interface peut changer : l'important est une exécution complète
côté serveur avec sauvegarde des fichiers de sortie.

Le programme traite un lecteur à la fois : extraction train, vingt époques sur ses
cinq folds, puis lecteur suivant. Une fois les quatre lecteurs terminés, il produit
les métriques gold, les mélanges, le bootstrap et les prédictions des trois exemples test.

`SESSION_HOURS = 6` est un **budget volontaire par session**, pas une promesse de
durée totale ni une limite officielle Kaggle. Choisir un budget inférieur au quota
restant et à la limite de session affichés. Le code s'arrête proprement sur ce budget,
conserve les études/époques terminées et écrit une archive. Une époque interrompue
avant sa fin sera recommencée au prochain lancement.

Les principales sorties dans **Output** sont :

| Fichier | À quoi sert-il ? |
|---|---|
| `rsna_resume.tar` | **À conserver** : features, checkpoints, état des folds et exports permettant de continuer |
| `rsna_reports.zip` | Métriques, pertes d'entraînement, OOF et soumissions disponibles |
| `SESSION_STATUS.json` | Nombre d'études extraites et d'époques terminées |

Une archive de reprise est renouvelée environ toutes les vingt minutes, après
les unités de travail terminées, et à la sortie normale ou à une interruption gérée.
Une coupure forcée du service ne peut pas exécuter le code de sauvegarde : la
dernière archive existante est alors le dernier point récupérable.

**Une archive sur le disque d'une session interactive n'est pas encore une copie
durable hors de cette session.** Sauvegarder la version et/ou télécharger les
sorties avant d'arrêter. Le notebook ne contourne aucune limite de service.

## E. Continuer dans une nouvelle session sans tout refaire

1. Ouvrir la dernière version sauvegardée et vérifier que `rsna_resume.tar` est
   présent dans ses fichiers Output. Le télécharger aussi comme sauvegarde locale.
2. Dans le notebook de la session suivante, garder le Dataset d'entrée original attaché.
3. Avec **Add Input**, ajouter les **outputs de votre précédente version de notebook**.
   Si cette option n'est pas visible, créer un deuxième Dataset privé contenant
   uniquement le fichier `rsna_resume.tar` téléchargé et l'attacher.
4. Garder `ACTION = "run"`. La reprise est détectée automatiquement si une seule
   archive `rsna_resume.tar` est attachée. S'il y en a plusieurs, mettre son chemin
   `/kaggle/input/.../rsna_resume.tar` dans `RESUME_FILE`.
5. Relancer en conservant le même environnement/image Kaggle. Les poids, folds et
   fichiers sont vérifiés ; les études finies ne sont pas recalculées.

La première session importe les 513 études CPU. Les sessions suivantes reprennent
l'état plus récent de l'archive GPU, puis complètent ce qui manque. Les différences
numériques CPU/CUDA ne sont pas promises identiques bit à bit ; tous les encodeurs
restent en float32 pour rester proches des features CPU initiales.

Changer les versions Python/PyTorch/NumPy entre sessions peut rendre les checkpoints
des têtes incompatibles. Garder l'image de la même expérience ; ne pas remplacer
le paquet d'entrée ou modifier les manifestes pour forcer une reprise.

## F. Ce qui reste identique et ce qui change

**Identique** : études, labels, groupes/folds, JPEG sélectionnés, poids génériques,
normalisation, huit images maximum/slot, architecture des têtes, vingt époques et
coefficients des mélanges. Les en-têtes des JPEG ne permettent toujours pas une
vraie adjacence 2.5D ; cette adaptation ne reproduit pas le modèle Kaggle original.

**Adapté** : modèle et lots déplacés sur CUDA, lots d'extraction plus grands avec
réduction automatique en cas de manque de VRAM, caractéristiques d'un lecteur
chargées en RAM pour éviter des millions de petites lectures répétées. Les identités
des images sont basées sur leur contenu plutôt que sur les dates de fichiers macOS.

`runtime/` contient la copie du code en ligne ; le code et les résultats locaux
dans `baseline_candidate/knee_baseline` et `artifacts/full` ne sont pas modifiés.
`build_bundle.py` peut reconstruire les archives sans exécuter de modèle.

## G. Alternative Colab et autres services

Ouvrir `COLAB_GPU.ipynb` sur [Colab](https://colab.research.google.com/).
Téléverser le ZIP dans **Google Drive → MyDrive → RSNA**, puis activer un GPU via
**Runtime → Change runtime type**. Le notebook demande votre autorisation pour
monter Drive, copie/décompresse les données sur le disque local et sauvegarde les
archives de reprise dans `MyDrive/RSNA/results`. Il ne lit pas 170 000 JPEG directement
depuis Drive. Le temps d'écriture des grosses archives Drive dépend de la connexion.

Un autre fournisseur Linux/CUDA peut utiliser le même paquet : décompresser le
ZIP, extraire `code_payload.bin` avec Python `zipfile`, puis appeler `bootstrap.setup`
avec ses chemins et lancer `knee_baseline.cloud.run_session`. Pour débuter, préférer
le notebook Kaggle plutôt qu'une machine cloud à administrer.

Voir **`OFFRES_ETUDIANT_FR.md`** pour les offres actuelles en France. Les limites
et la disponibilité des GPU restent contrôlées par chaque fournisseur.

## H. Dépannage simple

| Message | Action |
|---|---|
| Aucun GPU détecté | Activer Accelerator GPU et relancer la session |
| Attacher le Dataset du paquet | Ajouter votre Dataset privé dans Input |
| Plusieurs paquets/reprises trouvés | Renseigner BUNDLE_DIR / RESUME_FILE avec le bon chemin |
| Fichier de paquet altéré | Vérifier le téléversement, recréer le Dataset depuis le ZIP original |
| Version timm différente | Redémarrer le noyau puis exécuter l'installation hors ligne du notebook |
| torch/torchvision incompatible | Utiliser une image GPU standard du service, sans installer le lock macOS |
| Reprise issue d'un autre paquet | Attacher la reprise correspondant au Dataset d'entrée exact |
| Quota/session interrompu | Conserver la dernière archive, puis reprendre lorsque le quota est disponible |
| Métriques absentes | Le parcours n'est pas fini ; consulter SESSION_STATUS.json et reprendre |

Les archives ont été construites et vérifiées localement. Les vérifications et
limites exactes figurent dans `VERIFICATION_EN_LIGNE_FR.md`. Aucun entraînement
complet ni benchmark CUDA réel n'a été lancé pendant cette livraison.

## I. Repères dans le code et les résultats

Le code est consultable dans `runtime/` sur votre Mac. Dans la machine en ligne,
le notebook le décompresse depuis `code_payload.bin` : il s'agit du même code.

| Fichier | Rôle et relations |
|---|---|
| `runtime/bootstrap.py` | Vérifie les archives, installe les deux wheels, restaure la reprise et configure les chemins |
| `runtime/configs/gpu.json` | Expérience fixée : 20 époques, LR 0,001, batch 32, seed 42, cinq folds |
| `runtime/knee_baseline/config.py` | Lit cette configuration et les chemins temporaires/persistants |
| `runtime/knee_baseline/data.py` | Vérifie et lit les tables figées, sans refaire la stratification |
| `runtime/knee_baseline/features.py` | Charge un encodeur gelé, importe les 513 études et complète les caches sur GPU |
| `runtime/knee_baseline/models.py` | Définit les têtes, les masques et la BCE pondérée par confiance |
| `runtime/knee_baseline/train.py` | Entraîne les têtes fold par fold ; conserve la dernière époque et produit les OOF |
| `runtime/knee_baseline/evaluate.py` | Assemble les rangs avec les poids fixés, calcule les AUC et exporte les soumissions |
| `runtime/knee_baseline/cloud.py` | Enchaîne ces étapes, en respectant le budget volontaire de session |
| `runtime/knee_baseline/session.py` | Fabrique les archives de reprise et le relevé de progression |
| `runtime/knee_baseline/io.py`, `constants.py` | Écritures atomiques, empreintes, noms techniques des lecteurs et findings |
| `runtime/requirements-cloud.txt` | Dépendances attendues ; le PyTorch CUDA du service reste en place |
| `build_bundle.py` | Reconstruit le paquet local, sans lancer de modèle ; inutile pour utiliser le ZIP fourni |
| `tests/` | Petits contrôles locaux de portabilité ; aucun calcul complet |

Pour calculer uniquement un lecteur, mettre par exemple `READERS = ["coatnet"]`.
Remettre les quatre lecteurs pour terminer les comparaisons. Une reprise conserve
les features finies et reprend chaque tête à sa dernière époque entière ; un fold
de vingt époques est simplement rechargé pour vérifier/régénérer son export OOF.

Dans `rsna_reports.zip`, `reports/evaluation.json` résume les résultats et
`reports/auc_by_finding.csv` détaille chaque finding. Les OOF doivent couvrir
**4 407 études par lecteur**, avec **58 gold** pour l'évaluation principale.
Une AUC de 0,5 correspond à un classement aléatoire en espérance ; une valeur
plus élevée indique une meilleure séparation des positifs/négatifs. Une AUC
indéfinie signale notamment un sous-ensemble sans les deux classes. Les scores
`weak_proxy` sont une comparaison aux annotations faibles, pas une validation
indépendante comparable aux gold.

L'intervalle bootstrap compare `candidate05` et `baseline3` en rééchantillonnant
les groupes de rapports gold. Avec seulement 58 gold, l'incertitude peut être
forte ; un intervalle qui traverse zéro ne permet pas de conclure à un gain net.
Il reste exploratoire et conditionnel aux modèles OOF. Les métadonnées disponibles
séparent les études et groupes de rapports, sans démontrer l'indépendance patient.
Ne pas choisir de nouveaux poids puis annoncer une validation indépendante sur
ces mêmes gold. Les CSV dans `submissions/` portent sur les **trois exemples test**
disponibles : ils contrôlent le format et ne mesurent pas une performance.

Les références des modèles, leurs révisions et SHA-256 figurent dans
`REFERENCES_MODELES_FR.md` et les manifestes des poids.
