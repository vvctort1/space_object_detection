import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from arquiteturas.model2 import CNNAvancada

# Definindo classes
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]


# Configurando device como cpu (suficiente para o delpoy e salvando o modelo vencedor
device = torch.device("cpu") 
modelo = CNNAvancada(num_classes=10)
modelo.load_state_dict(torch.load('best_model2_weights.pth', map_location=device))
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
    title="🌌 Classificador de Imagens Orbitais - OrbitGuard",
    description="Faça o upload de uma imagem de satélite (64x64) para classificar o tipo de uso do solo. Projeto para a Global Solution - Indústria Espacial.",
    theme="default"
)

# Deploy local
if __name__ == "__main__":
    interface.launch(share=True)
