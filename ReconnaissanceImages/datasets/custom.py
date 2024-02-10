from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision.transforms import v2
import torch

import config

# Créer un Dataset personnalisé
class CustomDataset(Dataset):
    def __init__(self, images, image_folder, transform=None):
        self.image_folder = image_folder      # location des images
        self.n = int(images/8)                # Nombre de packets d'images à installer dans le dataset
        self.labels = [i % 8 + 1 for i in range(8 * self.n)]   # préparation des labels (sert d'itérateur pour le dataset)
        self.transform = transform            # opérateur de transformation/augmentation
        self.i = 0                            # compteur local

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # les images sont toujours prises dans le set d'origine (8 images)
        image_path = os.path.join(self.image_folder, "image{:02d}.jpg").format(idx % 8 + 1)
        image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        if self.i % 1000 == 0: print("i", self.i, "image_path", image_path, "label", label)
        self.i += 1
        return image, label

# Transformer pour l'augmentation des données
data_transform = v2.Compose([
    # padding des images
    v2.Pad(padding=config.padding, fill=(255,)),
    # Rotation aléatoire avec remplissage de pixels blancs
    v2.RandomRotation(30, fill=255),
    # transformation affine
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=255),
    # ajout de la perspective
    v2.RandomPerspective(fill=255),
    # ajout de la distortion
    v2.RandomPhotometricDistort(),
    # flou
    v2.GaussianBlur(kernel_size=9),
    # création du tenseur de sortie
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
])

