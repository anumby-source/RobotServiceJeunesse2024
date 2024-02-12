import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../ReconnaissanceCaracteres/models')
sys.path.append('datasets')

import os

import config
import custom

print("config.version=", config.version, "set=", config.image_path)

# Créer un dataset augmenté avec N images
dataset = custom.CustomDataset(images=config.N, image_folder=f"{config.image_folder}", transform=custom.data_transform)
print("Size of dataset=", dataset.__len__())

# Créer une fonction pour afficher les données avec leur label
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
    # plt.show()
    plt.savefig(f'{config.image_folder}/images_augmented.jpg')

show_images_with_labels(dataset)

if not os.path.exists(config.train_folder): os.makedirs(config.train_folder)

# Diviser le dataset en parties d'entraînement 80% et de test 20%
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
torch.save(train_dataset, config.train_path)
torch.save(test_dataset, config.test_path)

print("train=", train_dataset.__len__(), "test=", test_dataset.__len__())

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


