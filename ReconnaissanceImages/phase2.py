import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../ReconnaissanceCaracteres/models')

import resnet

# Créer un Dataset personnalisé
class CustomDataset(Dataset):
    def __init__(self, images, image_folder, transform=None):
        self.image_folder = image_folder
        self.n = int(images/8)
        self.image_paths = [os.path.join(image_folder, "image{:02d}.jpg").format(i % 8 + 1) for i in range(8 * self.n)]
        self.labels = [i % 8 + 1 for i in range(8 * self.n)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Transformer pour l'augmentation des données
data_transform = v2.Compose([
    v2.Pad(padding=100, fill=(255,)),
    # Rotation aléatoire avec remplissage de pixels blancs    v2.Resize((100, 100)),
    v2.RandomRotation(30, fill=255),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=255),
    v2.RandomPerspective(fill=255),
    v2.RandomPhotometricDistort(),
    v2.GaussianBlur(kernel_size=9),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
])

n = 1000
train_dataset = torch.load(f"dataset/train_{n}.pt")
test_dataset = torch.load(f"dataset/test_{n}.pt")

print(train_dataset.__len__(), test_dataset.__len__())

# Créer des DataLoader pour charger les données en lots
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Exemple d'utilisation : itérer sur les données d'entraînement
for images, labels in train_loader:
    # Les images sont dans le format [batch_size, channels, height, width]
    # Les labels sont dans le format [batch_size]
    # Par exemple, pour accéder au premier lot d'images et de labels :
    first_batch_images = images
    first_batch_labels = labels
    print(first_batch_images.shape)  # Afficher la forme des images
    print(first_batch_labels.shape)  # Afficher la forme des labels
    break  # Quitter la boucle après le premier lot

model = resnet.ResNet(resnet.ResidualBlock, [2, 2, 2])  # Resnet

# Définir la fonction de coût et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

