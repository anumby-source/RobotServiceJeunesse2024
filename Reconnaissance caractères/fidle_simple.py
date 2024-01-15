import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Définir la transformation des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Télécharger les données MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("train_dataset", train_dataset)
print("test_dataset", test_dataset)

# Définir les chargeurs de données
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Définir le modèle
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class basicCNN(nn.Module):
    def __init__(self):
        super(basicCNN, self).__init__()

        #Convolution/RelU/MaxPooling layers definition
        self.conv1 = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=1) # 1 to 2 channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 to 16x16 pixels
        self.conv2 = nn.Conv2d(2, 4, kernel_size=2, stride=1, padding=1)  # 2 to 4 channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 to 8x8 pixels
        self.conv3 = nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=1)  # 4 to 8 channels
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 to 4x4 pixels

        # dense layers definition
        self.fc1 = nn.Linear(8 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 8 * 4 * 4) # flatten the data
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Instancier le modèle, la fonction de coût et l'optimiseur
# model = SimpleNN()

model = basicCNN()

# pred = basicCNN_model(X)
# print(pred)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entraîner le modèle
epochs = 10

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

# Évaluer le modèle
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = correct / total
print(f'Accuracy on the test set: {accuracy * 100:.2f}%')







