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

# Définir la transformation des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Télécharger les données MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Définir les chargeurs de données
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Définir le modèle CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Instancier le modèle, la fonction de coût et l'optimiseur
model = CNN()
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
