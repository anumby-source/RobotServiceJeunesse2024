import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2


size = 200

# Charger et redimensionner une image en une taille carrée de sizexsize pixels
def resize_and_save_image(image_path, output_folder):
    # image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
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
        transforms.Resize((size, size)),  # Redimensionner à sizexsize pixels
        transforms.CenterCrop((size, size)),  # Recadrer au centre pour obtenir une image carrée
    ])
    resized_image = transform(image)

    # Sauvegarder l'image transformée
    resized_image.save(os.path.join(output_folder, new_name))

    return image, resized_image


# Exemple d'utilisation
image_folder = "data"
output_folder = "data"

# Liste des chemins d'accès aux images d'origine
image_paths = [os.path.join(image_folder, f"{i:02d}.jpg") for i in range(1, 9)]

# Créer le dossier de sortie s'il n'existe pas déjà
os.makedirs(output_folder, exist_ok=True)

# Visualiser les images avant et après la transformation et sauvegarder les images transformées
fig, axs = plt.subplots(2, len(image_paths), figsize=(20, 5))

for i, image_path in enumerate(image_paths):
    original_image, resized_image = resize_and_save_image(image_path, output_folder)
    axs[0, i].imshow(original_image, cmap='gray')
    axs[0, i].set_title('Avant')
    axs[0, i].axis('off')
    axs[1, i].imshow(resized_image, cmap='gray')
    axs[1, i].set_title('Après')
    axs[1, i].axis('off')

plt.show()