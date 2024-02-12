import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append('datasets')

import config

print("config.version=", config.version)

if not os.path.exists(config.image_folder): os.makedirs(config.image_folder)

# Charger et redimensionner une image en une taille carrée de image_size x image_size pixels
# Passage en niveau de gris
# filtrage des images pour s'approcher du noir et blanc
# sauvegarder l'image sous le nom image<nn>.jpg
def normalize_and_save_image(image_path, output_folder):
    image_cv = cv2.imread(image_path)

    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Appliquer une opération de seuillage pour binariser l'image
    _, thresholded_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Convertir l'image binarisée en une image PIL
    image = Image.fromarray(thresholded_image)

    # Extraire le numéro de l'image
    image_number = os.path.splitext(os.path.basename(image_path))[0]
    new_name = f"image{image_number}.jpg"

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),  # Redimensionner à config.image_sizexconfig.image_size pixels
        transforms.CenterCrop((config.image_size, config.image_size)),  # Recadrer au centre pour obtenir une image carrée
    ])
    normalized_image = transform(image)

    # Sauvegarder l'image transformée
    normalized_image.save(f'{config.image_folder}/{new_name}')

    return image, normalized_image

# Visualiser les images avant et après la transformation et sauvegarder les images transformées
fig, axs = plt.subplots(2, len(config.original_image_paths), figsize=(20, 5))

for i, image_path in enumerate(config.original_image_paths):
    original_image, normalized_image = normalize_and_save_image(image_path, config.image_folder)
    axs[0, i].imshow(original_image, cmap='gray')
    axs[0, i].set_title('Avant')
    axs[0, i].axis('off')
    axs[1, i].imshow(normalized_image, cmap='gray')
    axs[1, i].set_title('Après')
    axs[1, i].axis('off')

# plt.show()
plt.savefig(f'{config.image_folder}/images.jpg')
