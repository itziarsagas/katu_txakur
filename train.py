import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm  # Import tqdm for progress tracking
import os

# 1. Configuración de hiperparámetros
batch_size = 32
num_epochs = 10
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# 2. Transformaciones de datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño de entrada para ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Carga de datos
if os.path.exists('train_dataset.pt') and os.path.exists('val_dataset.pt'):
    print("Cargando datos preprocesados...")
    train_dataset = torch.load('train_dataset.pt')
    val_dataset = torch.load('val_dataset.pt')
else:
    print("No se han encontrado datos preprocesados. Creando y guardando los datos...")
    train_dataset = datasets.ImageFolder('./pet_images/train/', transform=transform)
    torch.save(train_dataset, 'train_dataset.pt')
    val_dataset = datasets.ImageFolder('./pet_images/val/', transform=transform)
    torch.save(val_dataset, 'val_dataset.pt')
#Se cargan los directorios en orden alfabético, por lo que se asume que la carpeta "gato" es la 0 y "perro" es la 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 4. Modelo preentrenado
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Para 2 clases: perros y gatos
model = model.to(device)

# 5. Configuración del optimizador y función de pérdida
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Entrenamiento
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validación
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

# 7. Guardar el modelo
torch.save(model.state_dict(), 'modelo_perros_gatos.pth')
print("Modelo guardado exitosamente.")
