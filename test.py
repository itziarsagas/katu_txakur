import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# CPU edo GPU erabiliko den zehaztu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Irudiak AAko modelo batean sartzeko, irudiak preprozesatu behar dira
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Entrenatutako modeloa kargatu
model = models.resnet18(pretrained=False)  # Entrenatutako modeloaren arkitektura erabili
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Bi klase: katua eta txakurra
model.load_state_dict(torch.load('modelo_perros_gatos.pth'))  # entrenatutako modeloaren pisuak kargatu
model = model.to(device)
model.eval()

# Klaseen hiztegia
class_names = {0: 'katua', 1: 'txakurra'}

# Test egiteko irudiak daukan karpeta
test_folder = 'test_imgs/' 

# Irudi bakoitza prozesatu
for image_name in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_name)
    
    # Irudia kargatu eta preprozesatu
    image = Image.open(image_path).convert('RGB')
    input_tensor = test_transform(image).unsqueeze(0).to(device)  
    
    # predikzioa egin
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        # predikzioa zenbaki bat da, hiztegiaren arabera txakurra edo katua den badakigu
        predicted_class = class_names[predicted.item()]
    
    # Emaitza
    print(f"Irudia: {image_name}, Gure modeloaren emaitza: {predicted_class}")