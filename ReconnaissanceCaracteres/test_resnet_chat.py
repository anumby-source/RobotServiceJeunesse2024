#-------------------
# Cet exemple utilise deux couches de convolution avec activation ReLU, suivies de couches de max-pooling.
# Ensuite, les caractéristiques extraites sont aplaties et envoyées à travers des couches entièrement connectées.
# On pourra ajuster les hyperparamètres, comme
# le nombre de filtres,
# la taille du noyau,
# le nombre de neurones dans les couches entièrement connectées
#
# Il existe plusieurs méthodes pour améliorer encore les performances d un modèle de réseau de neurones convolutionnel.
# Voici quelques idées que vous pouvez explorer :
#
# Augmentation des données :
# Utilisez des techniques d augmentation des données pour créer des variations des images d entraînement,
# telles que des rotations, des retournements horizontaux, des translations, etc.
# Cela augmente la diversité des données d entraînement, permettant au modèle de généraliser mieux.
#
# Architecture du réseau :
# Expérimentez avec différentes architectures de réseaux.
# Vous pouvez ajouter des couches de convolution supplémentaires,
# ajuster la taille des filtres,
# ou essayer des architectures plus avancées comme ResNet, Inception, ou DenseNet.
#
# Normalisation du lot (Batch Normalization) :
# Ajoutez des couches de normalisation du lot (Batch Normalization) entre les couches de convolution et les activations.
# Cela peut accélérer l entraînement et améliorer la convergence.
#
# Régularisation :
# Utilisez des techniques de régularisation telles que le dropout pour réduire le surajustement (overfitting)
# en désactivant aléatoirement certains neurones pendant l'entraînement.
#
# Taux d apprentissage adaptatif :
# Utilisez un taux d apprentissage adaptatif, comme Adam, qui peut ajuster automatiquement le taux d apprentissage pendant l entraînement.
#
# Évolution des hyperparamètres :
# Utilisez des méthodes d optimisation bayésienne ou des algorithmes d évolution pour rechercher efficacement les hyperparamètres optimaux.
#
#---------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('../fidle')
import fidle.pwk as ooo


# Définir la transformation des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_data():
    # Télécharger les données MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    x_train=train_dataset.data.type(torch.DoubleTensor)
    y_train=train_dataset.targets
    x_test=test_dataset.data.type(torch.DoubleTensor)
    y_test=test_dataset.targets

    np_x_train=x_train.numpy().astype(np.float64)
    np_y_train=y_train.numpy().astype(np.uint8)

    # display some images from the train set
    ooo.plot_images(np_x_train,np_y_train , [27],  x_size=5,y_size=5, colorbar=True)
    ooo.plot_images(np_x_train,np_y_train, range(5,41), columns=12)

    return train_dataset, test_dataset

train_dataset, test_dataset = get_data()

exit()

# Définir les chargeurs de données
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#
# Instancier le modèle, la fonction de coût et l'optimiseur
# Ce modèle ResNet est configuré pour MNIST avec une résolution d'image réduite.
# Vous pouvez ajuster le nombre de blocs dans chaque couche (layers) en fonction de vos besoins.
# Expérimentez avec ces paramètres pour voir comment ils affectent les performances de votre modèle.
#
model = ResNet(ResidualBlock, [2, 2, 2])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
epochs = 20

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

# Évaluer le modèle
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = correct / total
print(f'Accuracy on the test set: {accuracy * 100:.2f}%')
