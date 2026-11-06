# Expérience MAE

Le masked autoencoder apprend à reconstruire des patches d’IRM sans utiliser
les diagnostics. Son encodeur gelé produit ensuite des caractéristiques par
étude, évaluées avec cinq classifieurs linéaires hors fold.

- [mae_baseline.ipynb](mae_baseline.ipynb) : entraînement et comparaison aux caractéristiques d’un encodeur aléatoire ; mode initial `smoke`.
- [mae_submission.ipynb](mae_submission.ipynb) : inférence avec l’encodeur et les cinq classifieurs sauvegardés.
- [Guide de soumission et de vérification](../../submission/README.md).

Le pilote retenu comporte 30 époques et obtient une macro AUC hors fold de
0,5741 sur 58 études annotées. Ce résultat appartient aux classifieurs évalués
hors fold, pas à une évaluation indépendante de leur ensemble final.

Les entrées JPEG se trouvent dans `data/`, les entraînements dans
`outputs/mae_baseline/`, les poids d’inférence dans `submission/mae_assets/`.
Ces données et poids ne sont pas versionnés. Les scripts de vérification
se trouvent dans `scripts/verification/` et écrivent leurs notebooks exécutés
dans `local/scratch/`.
