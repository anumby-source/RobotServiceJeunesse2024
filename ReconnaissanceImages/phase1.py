import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../ReconnaissanceCaracteres/models')
sys.path.append('datasets')

import custom

# Créer un dataset augmenté avec N images
N = 1250*8
dataset = custom.CustomDataset(images=N, image_folder="data", transform=custom.data_transform)
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


