from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# InceptionV3 prend en charge une taille d'entrée de 299x299

class SafeResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if min(img.size) < min(self.size):
            return img
        return transforms.functional.resize(img, self.size)

# Transformation pour les images augmentées
transform_inception = transforms.Compose([
    SafeResize((299, 299)),  # Redimensionner à la taille attendue par InceptionV3
    # transforms.RandomResizedCrop((299, 299)),  # Redimensionner à la taille attendue par InceptionV3
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((299, 299)),  # Redimensionner à la taille finale (28x28)
    transforms.Grayscale(num_output_channels=3),  # Convertir en 3 canaux (RGB)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_data_inception():
    # Charger les données MNIST avec la transformation d'augmentation
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_inception)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        # SafeResize((299, 299)),  # Redimensionner à la taille attendue par InceptionV3
        transforms.Resize((299, 299)),  # Redimensionner à la taille finale (299x299)
        transforms.Grayscale(num_output_channels=3),  # Convertir en 3 canaux (RGB)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    # Utiliser un DataLoader pour charger les données
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

