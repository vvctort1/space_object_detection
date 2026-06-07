import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 1. Definir as classes exatas do EuroSAT (ordem alfabética, padrão do ImageFolder)
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# 2. Recriar a arquitetura da sua CNN Avançada
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

# 3. Configurar o dispositivo e carregar o modelo
device = torch.device("cpu") # Para o Gradio, CPU é suficiente e mais fácil de hospedar
modelo = CNNAvancada(num_classes=10)
modelo.load_state_dict(torch.load('melhor_modelo_eurosat.pth', map_location=device))
modelo.to(device)
modelo.eval() # Modo de avaliação (desliga o Dropout)

# 4. Definir as transformações (MESMAS do treinamento)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Função de predição chamada pelo Gradio
def prever_imagem(imagem):
    # O Gradio envia um array/PIL, garantimos que é PIL RGB
    if not isinstance(imagem, Image.Image):
        imagem = Image.fromarray(imagem).convert('RGB')
    else:
        imagem = imagem.convert('RGB')
        
    # Pre-processamento
    img_tensor = transform(imagem).unsqueeze(0).to(device) # Adiciona a dimensão do batch
    
    # Inferência
    with torch.no_grad():
        outputs = modelo(img_tensor)
        probabilidades = F.softmax(outputs[0], dim=0) # Converte para percentagens
        
    # Monta o dicionário de saída {Nome da Classe: Probabilidade} para o Gradio criar o gráfico
    resultado = {CLASSES[i]: float(probabilidades[i]) for i in range(len(CLASSES))}
    return resultado

# 6. Criar a interface visual do Gradio
interface = gr.Interface(
    fn=prever_imagem,
    inputs=gr.Image(type="pil", label="Carregar Imagem de Satélite"),
    outputs=gr.Label(num_top_classes=3, label="Previsão do Modelo"),
    title="🌌 Classificador de Imagens Orbitais (EuroSAT)",
    description="Faça o upload de uma imagem de satélite (64x64) para classificar o tipo de uso do solo. Projeto para a Global Solution - Indústria Espacial.",
    theme="default"
)

# 7. Lançar o aplicativo
if __name__ == "__main__":
    # share=True gera um link público temporário (ótimo para testar ou apresentar)
    interface.launch(share=True)