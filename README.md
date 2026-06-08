# Indústria Espacial: O Código que Move o Universo

## Sobre o OrbitGuard

A solução tem como objetivo o treinamento de um modelo de deep learning capaz de distinguir diferentes tipos de objetos e superfícies identificados a partir de imagens de satélite. O intuito é classificar regiões com suas devidas características e auxiliar na comunicação com instrumentos ativos na órbita terrestre, ajudando a evitar possíveis impactos e suas consequências.

O dataset utilizado é o EuroSAT RGB, composto por imagens de satélite de 64×64 pixels em 10 categorias de uso do solo.

### As 10 Categorias

- `AnnualCrop` — Lavoura Anual
- `Forest` — Floresta
- `HerbaceousVegetation` — Vegetação Herbácea
- `Highway` — Rodovias
- `Industrial` — Área Industrial
- `Pasture` — Pastagem
- `PermanentCrop` — Lavoura Permanente
- `Residential` — Área Residencial
- `River` — Rios
- `SeaLake` — Mar ou Lago

---

## Instruções para Execução do Projeto

### Pré-requisitos

- Python 3.8 ou superior
- pip
- (Opcional, para treinamento) GPU com suporte a CUDA

### Instalação

#### 1. Clone o repositório

```bash
git clone https://github.com/vvctort1/space_object_detection.git
```

#### 2. Crie um ambiente virtual

```bash
python -m venv venv
```

Ative o ambiente:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

#### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

### Como Usar

#### Rodando a interface no Gradio

Certifique-se de que o arquivo `melhor_modelo_eurosat.pth` está na raiz do projeto antes de executar.

Rode o arquivo `app.py` — a aplicação abrirá automaticamente no seu navegador ou aparecerá o link do localhost no terminal. Faça o upload de uma imagem de satélite, e o modelo retornará as 3 classes mais prováveis para a imagem escolhida.

---

### Treinamento do Modelo

Caso queira re-treinar o modelo do zero:

#### 1. Baixe o dataset EuroSAT RGB

Extraia na pasta `dataset/EuroSAT_RGB/`.

#### 2. Execute o notebook

Abra e execute o notebook `modelo_detecção_objeto.ipynb` célula por célula (mude os parâmetros caso achar necessário). Ao final, o arquivo `melhor_modelo_eurosat.pth` será gerado automaticamente na raiz do projeto.

#### Divisão dos dados

| Conjunto   | Proporção |
|------------|-----------|
| Treino     | 80%       |
| Validação  | 10%       |
| Teste      | 10%       |

---

## Integrantes do Grupo

| Nome | RM |
|------|----|
| Victor Kenzo Toma | RM551649 |
| Arthur Baldissera Claumann Marcos | RM550219 |
| Ricardo Ramos Vergani | RM550166 |

---

## Apresentação

> **Link da apresentação:** (link aqui)