# Contrôles de la livraison GPU

Contrôles locaux du **16 octobre 2026**. Aucun compte cloud, téléversement,
achat ou calcul distant n'a été effectué.

## Contrôles réalisés

- **10 tests ciblés réussis** : identité portable des caches, rejet de chemins
  d'archives sortants/liens, import des features CPU sans recalcul, découpage des
  lots après une erreur mémoire simulée, garde GPU, budget de session, archives
  de reprise, masques et invariance à l'ordre, empreintes et restauration d'une
  miniature complète. Aucun entraînement ni pas d'optimiseur dans ces tests.
- Syntaxe Python et imports des modules du runtime. Validation nbformat des
  deux notebooks ; cellules de code analysées, sans exécution. Les notebooks
  livrés n'ont ni compteur d'exécution ni sorties enregistrées.
- Chargement hors ligne des **quatre vrais poids**, puis **une seule image par
  encodeur sur CPU**. Sorties finies de formes `[1, 1024]` pour CoAtNet,
  `[1, 768]` pour Small/Tiny et `[1, 384]` pour DINOv2. La transformation CoAtNet
  est compatible avec le cache déjà présent sur le Mac. Ces quatre passages
  limités ne créent aucun nouveau cache d'extraction réel.
- Lecture des archives sans décompression globale : SHA-256 de chacun des
  **170 776 JPEG**, des quatre poids et des **513 features CoAtNet** importées.
  Formes, masques et valeurs finies des features importées vérifiés.
- Les tables de folds, cibles et séries sont conservées **octet pour octet**.
  Seule la table des images est rendue portable : chemin interne court et
  empreinte du contenu, sans dépendance à la date du fichier macOS.
- Couverture vérifiée : **4 407 études**, **58 gold**, **4 257 groupes de rapports**,
  cinq folds de 920 / 875 / 880 / 871 / 861 études et 12 / 11 / 13 / 11 / 11 gold.
  Aucun groupe de rapports ne traverse plusieurs folds. Les confiances gold
  valent 1. Huit images distinctes maximum par slot.
- Les **104 JPEG test** couvrent les trois exemples disponibles. Les autres
  **170 672 JPEG** couvrent toutes les études train sélectionnées.
- Vérification du format et des CRC de l'archive ZIP finale ; calcul de son
  SHA-256. Le paquet exclut les identifiants, `.venv`, rapports textuels bruts et
  fichiers locaux non nécessaires au calcul.

Les reçus locaux sont dans `dist/encoder_check.json`, `dist/payload_check.json`
et `dist/UPLOAD_SHA256.json`. Le SHA-256 du ZIP est séparé du ZIP qu'il décrit.

## Vérifications laissées à votre exécution

Le Mac utilisé pour cette livraison ne fournit pas de GPU NVIDIA/CUDA. La vitesse,
la mémoire GPU et la compatibilité de l'image logicielle Kaggle/Colab doivent donc
être confirmées par **ACTION = "benchmark"**. La gestion OOM a été testée par
simulation ; aucun test mémoire GPU réel n'a été fait.

L'extraction complète, les vingt entraînements de vingt époques, les métriques
OOF et les soumissions finales **n'ont pas été exécutés**. Le paquet ne présente
aucun nouveau score ni aucune promesse de gain. Les sorties futures doivent
couvrir les 4 407 OOF, évaluer les 58 gold et produire les CSV des trois exemples
test ; ces trois exemples ne mesurent pas la performance.

Le calcul float32 peut différer légèrement entre CPU et CUDA. La reprise de
features conserve les poids et transformations ; elle ne promet pas une égalité
bit à bit. Les checkpoints des têtes conservent des contrôles de versions stricts :
garder la même image logicielle entre sessions d'une expérience.

## Refaire uniquement les contrôles locaux

Depuis `baseline_candidate/online_kaggle/`, avec l'environnement local existant :

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests --basetemp=cache/test_tmp
PYTHONDONTWRITEBYTECODE=1 python tests/verify_payloads.py
PYTHONDONTWRITEBYTECODE=1 python tests/verify_deliverables.py
```

Ces commandes ne lancent ni notebook ni entraînement. La vérification des archives
lit environ 3 Go de données ; elle ne refait pas l'extraction des caractéristiques.
`build_bundle.py` est un outil de conditionnement permanent, pas un générateur de
notebook. Aucun script temporaire de génération de notebook n'est conservé.
