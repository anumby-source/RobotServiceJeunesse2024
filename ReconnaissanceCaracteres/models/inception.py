import torch.nn as nn
from torchvision import models
from torchvision.models.inception import Inception_V3_Weights

# Utiliser InceptionV3 pré-entraîné
class InceptionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionModel, self).__init__()
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        # Remplacer la dernière couche linéaire pour s'adapter à 10 classes (MNIST)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, num_classes)

    def forward(self, x):
        x = self.inception(x)
        logits = x.logits if hasattr(x, 'logits') else x.fc.in_features
        return logits

