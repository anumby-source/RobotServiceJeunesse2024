# Tests pour différents modèles

Cet exemple utilise deux couches de convolution avec activation ReLU, suivies de couches de max-pooling.
Ensuite, les caractéristiques extraites sont aplaties et envoyées à travers des couches entièrement connectées.
On pourra ajuster les hyperparamètres, comme

- le nombre de filtres,
- la taille du noyau,
- le nombre de neurones dans les couches entièrement connectées

# Il existe plusieurs méthodes pour améliorer encore les performances d un modèle de réseau de neurones convolutionnel.

## Augmentation des données :

Utilisez des techniques d augmentation des données pour créer des variations des images d entraînement,
telles que des rotations, des retournements horizontaux, des translations, etc.
Cela augmente la diversité des données d entraînement, permettant au modèle de généraliser mieux.

## Architecture du réseau :

- Vous pouvez ajouter des couches de convolution supplémentaires,
- ajuster la taille des filtres,
- ou essayer des architectures plus avancées comme ResNet, Inception, ou DenseNet.

## Normalisation du lot (Batch Normalization) :

Ajoutez des couches de normalisation du lot (Batch Normalization) entre les couches de convolution et les activations.
Cela peut accélérer l entraînement et améliorer la convergence.

## Régularisation :

Utilisez des techniques de régularisation telles que le dropout pour réduire le surajustement (overfitting)
en désactivant aléatoirement certains neurones pendant l'entraînement.

## Taux d apprentissage adaptatif :

Utilisez un taux d apprentissage adaptatif, comme Adam, qui peut ajuster automatiquement le taux d apprentissage pendant l entraînement.

## Évolution des hyperparamètres :

Utilisez des méthodes d optimisation bayésienne ou des algorithmes d évolution pour rechercher efficacement les hyperparamètres optimaux.
