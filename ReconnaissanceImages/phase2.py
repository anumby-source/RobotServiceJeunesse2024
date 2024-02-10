import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

import sys
sys.path.append('../ReconnaissanceCaracteres/models')
sys.path.append('datasets')

import resnet
import config

# chargement des datasets
train_dataset = torch.load(f"dataset/train_{config.N}.pt")
test_dataset = torch.load(f"dataset/test_{config.N}.pt")

print(train_dataset.__len__(), test_dataset.__len__())

# Créer des DataLoader pour charger les données en lots
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Création du modèle choisi
model = resnet.ResNet(resnet.ResidualBlock, [2, 2, 2])  # Resnet

# Définir la fonction de coût et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle et enregistrer les valeurs de perte pour visualisation ultérieure
epochs = config.epochs
loss_values = []
all_preds = []
all_labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

t0 = time.time()

for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            t = time.time() - t0
            secondes = int(t) % 60
            minutes = int(int(t)/60) % 60
            heures = int(int(t)/3600) % 24

            print(f'{heures}h{minutes}m{secondes}s> Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

        loss_values.append(loss.item())

    # Évaluer le modèle sur le jeu de test et enregistrer les prédictions et les étiquettes
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(target.cpu().numpy())

    all_preds.extend(preds)
    all_labels.extend(labels)

# sauvegarde des paramètres du modèle entraîné
torch.save(model.state_dict(), f'images{config.N}.pth')

def show_loss(loss_values):
    # Visualiser graphiquement la valeur de la perte
    plt.plot(loss_values)
    plt.title('Training Loss Over Batches')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.show()

show_loss(loss_values)

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Afficher la matrice de confusion
class_names = [str(i) for i in range(10)]  # Les classes MNIST sont les chiffres de 0 à 9
plot_confusion_matrix(all_labels, all_preds, classes=class_names)

