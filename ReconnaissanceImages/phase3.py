import cv2
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('../ReconnaissanceCaracteres/models')
sys.path.append('datasets')

import resnet
import config

# Création du modèle choisi
# model = resnet.ResNet(resnet.ResidualBlock, [2, 2, 2])  # Resnet

# Charger le modèle pré-entraîné
# model.load_state_dict(torch.load(f'parameters/images_{config.version}-{config.N}-{config.epochs}.pth'))
# model.eval()

size = config.image_size*5
# Définir une transformation pour adapter l'image capturée au format d'entrée du modèle
transform = v2.Compose([
    v2.ToPILImage(),                # Convertir l'image en PIL Image
    v2.Pad(padding=config.padding, fill=(255,)),
    v2.Grayscale(num_output_channels=1),  # Convertir l'image en niveaux de gris (si elle n'est pas déjà en niveaux de gris)
    v2.Resize((size, size)),          # Redimensionner l'image à la taille d'entrée du modèle Inception (299x299)
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliser l'image
    v2.Normalize(mean=[0.485], std=[0.229])  # Normaliser l'image (si nécessaire)
])

while True:
    # Capturer une image avec la caméra
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite("captured_image.jpg", frame)
    cap.release()

    # Charger et adapter l'image capturée
    image_cv = cv2.imread("captured_image.jpg")

    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Appliquer une opération de seuillage pour binariser l'image
    _, thresholded_image = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)

    # Convertir l'image binarisée en une image PIL
    # image = Image.fromarray(thresholded_image)
    image = thresholded_image

    # Visualiser l'image capturée avant la transformation
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image_cv)
    plt.title('Image capturée')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(gray_image)
    plt.title('Image grise')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(image, cmap='gray')
    plt.title('Image capturée (avant transformation)')
    plt.axis('off')

    # Appliquer la transformation
    input_image = transform(image).unsqueeze(0)    # Ajouter une dimension pour le batch

    # Visualiser l'image après la transformation
    plt.subplot(1, 4, 4)
    plt.imshow(input_image.squeeze().numpy(), cmap='gray')
    plt.title('Image transformée (après transformation)')
    plt.axis('off')

    plt.show()

exit()

# Faire une prédiction avec le modèle
with torch.no_grad():
    output = model(input_image)

# Obtenir l'indice de la classe prédite
_, predicted_class = torch.max(output, 1)

print("Predicted class:", predicted_class.item())
