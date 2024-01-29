import torch.nn as nn
from torchvision import models

# Utiliser InceptionV3 pré-entraîné
class InceptionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionModel, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        # Remplacer la dernière couche linéaire pour s'adapter à 10 classes (MNIST)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, num_classes)

    def forward(self, x):
        # Utiliser seulement la sortie principale "logits"
        return self.inception(x).logits

