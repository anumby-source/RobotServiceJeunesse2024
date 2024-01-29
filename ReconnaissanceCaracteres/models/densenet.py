import torch.nn as nn
from torchvision import models

# Utiliser DenseNet pré-entraîné
class DenseNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)  # Vous pouvez changer le modèle ici
        # Remplacer la dernière couche linéaire pour s'adapter à 10 classes (MNIST)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)


