from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transformation pour les images augmentées
transform = transforms.Compose([
    transforms.RandomResizedCrop((28, 28), scale=(1.0, 2.0)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((28, 28)),  # Redimensionner à la taille finale (28x28)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_data():
    # Charger les données MNIST avec la transformation d'augmentation
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))

    # Utiliser un DataLoader pour charger les données
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


