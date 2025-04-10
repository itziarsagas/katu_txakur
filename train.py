import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm  # Import tqdm for progress tracking
import os

# Hiperparametroen hasierako baloreak
batch_size = 32
num_epochs = 10
learning_rate = 0.001
# GPU badaukagu, hori erabiliko da, bestela CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Irudiak AAko modelo batean sartzeko, irudiak preprozesatu behar dira
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño de entrada para ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datuak prozesatu baditugu, kargatu 
if os.path.exists('train_dataset.pt') and os.path.exists('val_dataset.pt'):
    print("Aurretik prozesatutako datuak kargatzen...")
    train_dataset = torch.load('train_dataset.pt')
    val_dataset = torch.load('val_dataset.pt')
else: # Bestela, datuak kargatu eta gorde
    print("Ez dira aurretik prozesatutako datuak aurkitu. Datuak prozesatzen...")
    train_dataset = datasets.ImageFolder('./pet_images/train/', transform=transform)
    torch.save(train_dataset, 'train_dataset.pt')
    val_dataset = datasets.ImageFolder('./pet_images/val/', transform=transform)
    torch.save(val_dataset, 'val_dataset.pt')
    print("Datuak prozesatu eta gorde dira.")

#Direktorioak ordena alfabetikoan kargatzen dira, eta, beraz, onartzen da "katua" karpeta 0 dela eta "txakurra" 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Normalean ez da hutsetik entrenatzen, aurretik entrenatutako modeloak erabiltzen dira
# Erabiliko dugun modeloa ResNet18 izango da eta pisu generikoak erabiliko dira (ImageNet-en entrenatutakoak)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Azkenengo fully connected geruza bi klaseetarako (txakurra eta katuak)
model = model.to(device) # modeloa GPU-ra edo CPU-ra bidaltzen da

# Loss funtzioa eta optimizatzailea definitzen dira eta learning rate hasieran definitu duguna
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamendua
for epoch in range(num_epochs):
    # Modeloa train moduan badago, pisuak egneratzen dira
    model.train()
    running_loss = 0.0
    # Hemen hasten gara entrenatzen
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        #back propagation loss funtzioaren bidez
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    # Modeloa ebaluatzeko, pisuak ez dira eguneratzen
    model.eval()
    # Accuracy kalkulatzeko aldagaiak
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

# Modeloa gordetzen dugu
torch.save(model.state_dict(), 'modelo_perros_gatos.pth')
print("Modeloa entrenatu eta gorde da.")
