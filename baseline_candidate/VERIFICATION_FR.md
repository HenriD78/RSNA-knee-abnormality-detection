# Contrôles réalisés avant livraison

Contrôles effectués le **12 octobre 2026**, dans l'environnement Python 3.12 existant.
Le dossier est livré pour que vous lanciez vous-même les calculs complets.

## Réalisé

- **24 tests réussis** en 3,93 secondes lors du dernier passage. Ils utilisent des
  tables, JPEG, caractéristiques et états de têtes artificiels dans le cache des tests.
- Séparation des groupes de rapports, reproductibilité des folds après permutation
  des lignes, priorité des gold et traitement des cibles inconnues.
- Choix déterministe des séries/images, slots absents et signalement d'un JPEG illisible.
- Passes avant de petites têtes synthétiques : masques et invariance à l'ordre,
  y compris avec deux couches de transformer. Vérification de la BCE pondérée.
- Gel des paramètres d'un encodeur artificiel ; aucun encodeur réel téléchargé.
- Reprise des caractéristiques, refus des empreintes incompatibles et corruption
  détectée. Sérialisation/lecture de petits checkpoints sans étape d'optimisation.
- Préparation miniature, réutilisation et refus d'une image modifiée.
- Alignement des UID, égalités dans les rangs, coefficients de mélange, AUC indéfinies
  et reproductibilité d'un petit bootstrap par groupe.
- Évaluation séparée des cohortes, assemblage OOF et exports des cinq soumissions
  avec têtes synthétiques **non entraînées**. Ces fixtures n'ont aucune valeur de performance.
- Import des modules, des interfaces timm/Hugging Face et reconnaissance des neuf commandes CLI.
- Validation nbformat et syntaxe des **12 cellules de code** du notebook. Toutes
  les cellules restent sans compteur d'exécution ni sortie enregistrée.
- Exécution isolée de la cellule de configuration, avec sous-processus interdits :
  le mode `inspection`, la sélection d'étapes vide et le téléchargement désactivé
  empêchent les traitements lourds. Aucune autre cellule n'a été exécutée.
- Lecture légère des véritables CSV via `inspect` : 4 407 train, 58 gold,
  4 349 weak et trois exemples test.
- Construction des folds réels **en mémoire seulement**, sans indexer les images
  ni écrire de master de production. Résultat : 4 257 groupes de rapports et aucune
  séparation d'un même groupe entre folds.
- Vérification des **124 versions verrouillées** et de leurs contraintes actives
  dans l'environnement existant : aucune incompatibilité relevée.

### Diagnostic du split réel

| Fold | Études | Gold | Les douze findings ont deux classes parmi les gold |
|---|---:|---:|---|
| 0 | 920 | 12 | Oui |
| 1 | 875 | 11 | Oui |
| 2 | 880 | 13 | Oui |
| 3 | 871 | 11 | Oui |
| 4 | 861 | 11 | Oui |

Ces chiffres sont des contrôles de préparation, pas des scores de modèle.

## Non exécuté, à votre charge

- Installation du lock dans un environnement neuf.
- Téléchargement des quatre poids et chargement/transformations de leurs véritables checkpoints.
- Préparation/indexation complète des 819 076 images JPEG.
- Benchmark des encodeurs et extraction de caractéristiques, même en smoke sur les données réelles.
- Entraînement des vingt têtes, ou de toute autre tête sur les données réelles.
- Évaluation réelle, bootstrap des 58 gold, prédiction réelle et score Kaggle.
- Exécution complète du notebook.

Les états de têtes utilisés dans les tests sont uniquement initialisés puis lus
pour vérifier les interfaces ; **aucun `optimizer.step` n'a été exécuté** lors des contrôles.
Les données existantes dans `data/` et les autres notebooks n'ont pas été modifiés.
Les fichiers générés par les tests sont nettoyés ; aucun script générateur de notebook
n'est conservé. Le script `tests/verify_notebook.py` est un vérificateur.

## Refaire les vérifications

Depuis `baseline_candidate/`, dans l'environnement installé :

```bash
python -m pytest --basetemp=cache/test_tmp
python tests/verify_notebook.py
python -m knee_baseline --config configs/full.json inspect
```

Consultez `GUIDE_FR.md` pour démarrer ensuite votre parcours smoke, puis full.
