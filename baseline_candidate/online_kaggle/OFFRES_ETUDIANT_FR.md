# Calcul gratuit / étudiant en France

Vérification des pages officielles : **16 octobre 2026**. Une adresse `@edu.fr`
peut aider à prouver votre inscription ; elle ne remplace pas la vérification
d'éligibilité demandée par chaque service.

| Option | Ce qui est vérifié | Conseil pour ce projet |
|---|---|---|
| Kaggle Notebooks GPU | Environnement GPU avec quotas affichés dans le compte, variables selon disponibilité | **Commencer ici**, faire le benchmark GPU puis continuer par sessions avec reprise |
| GitHub Education → Camber | L'offre du Pack annonce un plan étudiant gratuit avec **5 h GPU/mois**, 40 h CPU et 50 Go de stockage pendant l'inscription | Complément possible après validation GitHub Education ; ce n'est pas forcément plus que Kaggle |
| Lightning AI | Le plan gratuit annonce **jusqu'à 30 crédits initiaux** ; la page étudiants décrit une remise académique, pas un accès GPU illimité gratuit | Réserve ponctuelle ; vérifier les crédits accordés au compte avant un calcul |
| Colab gratuit | GPU soumis à disponibilité et limites dynamiques ; les VM et leur disque sont temporaires | Possible avec le notebook Colab et les sauvegardes Drive |
| Google AI Plus étudiant en France | Offre affichée : **un an sans frais**, vérification étudiante et conditions, dont moyen de paiement et renouvellement ultérieur | Ne pas confondre cette offre Gemini/stockage avec une promesse de quota GPU Colab |

## Activer l'offre GitHub Education / Camber

1. Aller sur [GitHub Education](https://education.github.com/pack).
2. Demander la vérification du statut étudiant ; fournir l'adresse universitaire
   et, si demandé, un justificatif d'inscription valide.
3. Une fois approuvé, ouvrir l'offre **Camber** depuis le Pack et suivre le parcours
   du partenaire. Vérifier les conditions et le compteur GPU dans votre compte.
4. Garder le paquet Kaggle : le code est portable vers un environnement Linux/CUDA,
   mais l'interface et le dépôt des fichiers dépendent du fournisseur.

L'offre officielle du [GitHub Student Developer Pack](https://education.github.com/pack)
mentionne cinq heures GPU mensuelles pour Camber ; ce crédit est un complément,
pas une garantie que notre expérience entière tiendra dans une seule allocation.

## Google : attention aux anciennes offres

La page française [Gemini pour les étudiants](https://gemini.google/fr/students/?hl=fr)
annonce actuellement Google AI Plus sans frais pendant un an pour les étudiants
éligibles. La page décrit surtout Gemini et le stockage, sans promettre de quota
Colab déterminé. Vérifier l'avantage Colab dans votre compte avant de compter dessus.
La [FAQ Colab](https://research.google.com/colaboratory/faq.html) précise aussi que
les avantages Colab des plans Google AI ne sont pas inclus pendant un essai
gratuit promotionnel : ne pas déduire un crédit GPU de cette offre étudiante.

L'ancien **Colab Pro for Education**, destiné aux établissements américains,
**n'accepte plus de nouvelles inscriptions** selon la [FAQ officielle Colab](https://research.google.com/colaboratory/faq.html).
Les allocations des offres Colab disponibles peuvent changer et ne sont pas illimitées.

Les crédits de l'essai Google Cloud ne donnent pas automatiquement accès aux GPU :
la [documentation de l'essai](https://docs.cloud.google.com/free/docs/free-cloud-features)
indique qu'un compte restant en essai gratuit ne peut pas ajouter de GPU aux VM.
Une université peut disposer d'un autre accès pédagogique/recherche : demander au
service informatique ou à l'enseignant plutôt que supposer les crédits compatibles GPU.

## Sources complémentaires

- [Documentation Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) : activation GPU et sauvegarde des outputs.
- [Lightning AI — tarifs](https://lightning.ai/pricing) : crédits initiaux et stockage du plan gratuit.
- [Lightning AI — étudiants](https://lightning.ai/docs/platform/overview/setups/academia/students) : remise académique et accès institutionnels.

Après le benchmark, comparer la durée estimée au quota réellement affiché. Aucun
achat, création de compte, transmission de mail ou activation d'abonnement n'a été effectué.
