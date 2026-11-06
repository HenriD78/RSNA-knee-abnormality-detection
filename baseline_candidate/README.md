# Baseline candidate — quatre lecteurs JPEG

Commencer par [GUIDE_FR.md](GUIDE_FR.md), puis ouvrir `baseline_candidate.ipynb`.

Le notebook est livré en **inspection**. Aucun téléchargement, benchmark,
extraction ou entraînement ne démarre automatiquement. Les CSV et JPEG dans
`../data` sont des entrées en lecture seule.

Les quatre encodeurs génériques restent gelés ; leurs têtes sont entraînées
avec cinq folds communs. Cette adaptation JPEG ne reproduit pas les modèles
Kaggle fine-tunés ni leur score. Voir [REFERENCES_FR.md](REFERENCES_FR.md).

Les contrôles de livraison sont consignés dans [VERIFICATION_FR.md](VERIFICATION_FR.md).
