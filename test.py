import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformaciones para las imágenes de prueba
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar el modelo entrenado
model = models.resnet18(pretrained=False)  # Mismo modelo que se usó para entrenar
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Dos clases: perro y gato
model.load_state_dict(torch.load('modelo_perros_gatos.pth'))  # Cargar los pesos entrenados
model = model.to(device)
model.eval()

# Diccionario de clases
class_names = {0: 'gato', 1: 'perro'}

# Carpeta de imágenes de prueba
test_folder = 'test_imgs/'  # Cambia esto por la ruta de tu carpeta de imágenes de prueba

# Procesar cada imagen en el conjunto de prueba
for image_name in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_name)
    
    # Cargar y preprocesar la imagen
    image = Image.open(image_path).convert('RGB')
    input_tensor = test_transform(image).unsqueeze(0).to(device)  # Añadir dimensión batch
    
    # Realizar predicción
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    
    # Mostrar resultado
    print(f"Imagen: {image_name}, Clase predicha: {predicted_class}")