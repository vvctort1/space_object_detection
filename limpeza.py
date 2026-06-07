import os
import cv2

def limpar_imagens_corrompidas(diretorio_base):
    imagens_removidas = 0
    
    # Percorre todas as pastas e subpastas do dataset
    for root, dirs, files in os.walk(diretorio_base):
        for file in files:
            caminho_arquivo = os.path.join(root, file)
            
            try:
                # Tenta ler a imagem com OpenCV
                img = cv2.imread(caminho_arquivo)
                
                # Se o retorno for None, a imagem é inválida ou está corrompida
                if img is None:
                    os.remove(caminho_arquivo)
                    print(f"🗑️ Imagem inválida removida: {caminho_arquivo}")
                    imagens_removidas += 1
            except Exception as e:
                os.remove(caminho_arquivo)
                print(f"⚠️ Erro ao processar, removendo: {caminho_arquivo} | Erro: {e}")
                imagens_removidas += 1
                
    print(f"\nLimpeza concluída! Total de imagens defeituosas removidas: {imagens_removidas}")

# Limpa o diretório que acabamos de criar
limpar_imagens_corrompidas("./dataset_imagens")