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

# Créer un dataset augmenté avec N images
N = 100000
dataset = CustomDataset(images=N, image_folder="data", transform=data_transform)
print(dataset.__len__())

# Créer une fonction pour afficher les données
def show_images_with_labels(dataset, num_images=20):
    # Créer une grille pour afficher les images
    fig, axs = plt.subplots(4, 5, figsize=(15, 10))
    for i in range(num_images):
        # Obtenir une image et son étiquette associée
        image, label = dataset[np.random.randint(len(dataset))]
        ax = axs[i // 5, i % 5]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")  # Ajouter le label comme titre de l'image
        ax.axis('off')
    plt.show()

show_images_with_labels(dataset)

# Sauvegarder le dataset augmenté
# torch.save(dataset, f"dataset/images_{dataset.__len__()}.pt")

# Diviser le dataset en parties d'entraînement et de test
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
torch.save(train_dataset, f"dataset/train_{dataset.__len__()}.pt")
torch.save(test_dataset, f"dataset/test_{dataset.__len__()}.pt")

print("train", train_dataset.__len__(), "test", test_dataset.__len__())

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


