import os
import logging
from icrawler.builtin import GoogleImageCrawler

# Ativa os logs para vermos se há bloqueios (ERRO ou AVISO)
logging.basicConfig(level=logging.INFO)

def baixar_imagens_por_classe(classes_de_busca, max_imagens=10):
    if GoogleImageCrawler is None:
        print("icrawler não está instalado. Instale com `pip install icrawler` e execute novamente.")
        return

    # Usando um caminho relativo simples para garantir que pastas sejam criadas
    diretorio_base = "./" 
    
    if not os.path.exists(diretorio_base):
        os.makedirs(diretorio_base)

    for nome_classe, termos_de_busca in classes_de_busca.items():
        print(f"\n🚀 Iniciando download para a classe: {nome_classe}")
        
        caminho_classe = os.path.join(diretorio_base, nome_classe)
        
        # O Google costuma ter menos bloqueios que o Bing para scraping
        google_crawler = GoogleImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4, 
            storage={'root_dir': caminho_classe}
        )
        
        # Removi os filtros momentaneamente para garantir que algo seja retornado
        google_crawler.crawl(
            keyword=termos_de_busca, 
            max_num=max_imagens, 
            file_idx_offset=0
        )
        print(f"✅ Download finalizado para {nome_classe}.")

minhas_classes = {
    "sat_imgs": "images of satellite",
}

# Reduzi para 10 imagens apenas para validar se resolve o problema
baixar_imagens_por_classe(minhas_classes, max_imagens=20)