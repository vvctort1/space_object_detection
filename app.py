import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Definindo classes
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Estrutura arquitetura vencedora
class CNNAvancada(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. Configurando device como cpu (suficiente para o delpoy e salvando o modelo
device = torch.device("cpu") 
modelo = CNNAvancada(num_classes=10)
modelo.load_state_dict(torch.load('melhor_modelo_eurosat.pth', map_location=device))
modelo.to(device)
modelo.eval() 

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def prever_imagem(imagem):
    if not isinstance(imagem, Image.Image):
        imagem = Image.fromarray(imagem).convert('RGB')
    else:
        imagem = imagem.convert('RGB')
        
    img_tensor = transform(imagem).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = modelo(img_tensor)
        probabilidades = F.softmax(outputs[0], dim=0) 
        
    
    resultado = {CLASSES[i]: float(probabilidades[i]) for i in range(len(CLASSES))}
    return resultado

# Criando interface gradio
interface = gr.Interface(
    fn=prever_imagem,
    inputs=gr.Image(type="pil", label="Carregar Imagem de Satélite"),
    outputs=gr.Label(num_top_classes=3, label="Previsão do Modelo"),
    title="🌌 Classificador de Imagens Orbitais (EuroSAT)",
    description="Faça o upload de uma imagem de satélite (64x64) para classificar o tipo de uso do solo. Projeto para a Global Solution - Indústria Espacial.",
    theme="default"
)

# Deploy local
if __name__ == "__main__":
    interface.launch(share=True)
