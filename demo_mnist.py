import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Transformaciones: convierte las imágenes a tensores y normaliza
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Cargar los datos
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# 3. Definir el modelo (red neuronal)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NeuralNet()

# 4. Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Entrenamiento
for epoch in range(1):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f"Entrenamiento finalizado - Última pérdida: {loss.item():.4f}")

# 6. Probar el modelo
correct = 0
total = 0
all_images = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Guardamos algunas imágenes para visualizarlas
        all_images.extend(images[:6])
        all_preds.extend(predicted[:6])
        all_labels.extend(labels[:6])
        break  # solo mostramos un batch

print(f'Precisión: {100 * correct / total:.2f}%')

# 7. Mostrar imágenes y predicciones
def imshow(img):
    img = img / 2 + 0.5  # Desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.show()

# Mostrar imágenes individuales
for i in range(6):
    imshow(all_images[i])
    print(f'Predicción: {all_preds[i].item()} - Real: {all_labels[i].item()}')
