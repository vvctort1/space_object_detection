# ComputerVision-GS126
Repositório utilizado para realizar a global solution do primeiro semestre da matéria de visão computacional

## Sobre o Projeto
A solução tem como objetivo o treinamento de um modelo de deep learning capaz de distinguir diferentes tipos de objetos e superfícies identificados a partir de imagens de satélite. O intuito é classificar regiões com suas devidas características e auxiliar na comunicação com instrumentos ativos na órbita terrestre, ajudando a evitar possíveis impactos e suas consequências.
O dataset utilizado é o EuroSAT RGB, composto por imagens de satélite de 64×64 pixels em 10 categorias de uso do solo.

### As 10 Categorias:
- "AnnualCrop" (Lavoura Anual)
- "Forest" (Floresta)
- "HebaceousVegetation" (Vegetação Herbácea)
- "Highway" (Rodovias)
- "Industrial" (Área Industrial)
- "Pasture" (Pastagem)
- "PermanentCrop" (Lavoura Permanente)
- "Residential" (Área Residencial)
- "River" (Rios)
- "SeaLake" (Mar ou Lago)

## Instruções para Execução do Projeto
### Pré-requisitos

- Python 3.8 ou superior
- pip
- (Opcional, para treinamento) GPU com suporte a CUDA

### Instalação
#### 1. Clone o repositório:

- git clone https://github.com/vvctort1/space_object_detection.git

#### 2. Criar um ambiente virtual:

- python -m venv venv

- Windows:
  
venv\Scripts\activate

- Linux / macOS:
  
source venv/bin/activate   

#### 3. Instale as dependências

pip install -r requirements.txt

### Como Usar
Rodando a interface no Gradio

Certifique-se de que o arquivo melhor_modelo_eurosat.pth está na raiz do projeto antes de executar.

Rode o arquivo app.py, a aplicação abrirá automaticamente no seu navegador ou aparecerá o link do localhost no terminal. Faça o upload de uma imagem de satélite, e o modelo retornará as 3 classes mais prováveis para a imagem escolhida.

###  Treinamento do Modelo 
Caso queira re-treinar o modelo do zero: 

#### 1. Baixe o Data-set EuroSAT RGB

Extraia na pasta dataset/EuroSAT_RGB/

#### 2. Execute o notebook

Abra e execute o notebook modelo_detecção_objeto.ipynb célula por célula (mude os parâmetros caso achar necessário). Ao final, o arquivo melhor_modelo_eurosat.pth será gerado automaticamente na raiz do projeto.

Divisão dos dados:

- Treino: 80%
- Validação: 10%
- Teste: 10%

## Integrantes do grupo

- Victor Kenzo Toma RM551649
- Arthur Baldissera Claumann Marcos RM550219
- Ricardo Ramos Vergani RM550166


## Link da apresentação 
(link aqui)
