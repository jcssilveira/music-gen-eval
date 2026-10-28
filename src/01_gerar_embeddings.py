import os
import re
import warnings
import logging
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# Configurações e Logs
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def limpar_textos(caminho_csv):
    """Lê o CSV bruto e extrai as perguntas combinadas com a escala."""
    df_original = pd.read_csv(caminho_csv)
    coluna_texto = df_original.iloc[:, 0].dropna().astype(str)
    
    padrao_construct = r"Construct\s*/\s*Question:\s*([^\r\n]+)"
    padrao_scale = r"Response\s*Scale:\s*([^\r\n]+)"
    
    dados_extraidos = []
    for texto in coluna_texto:
        texto_limpo = texto.replace('\xa0', ' ')
        constructs = re.findall(padrao_construct, texto_limpo, flags=re.IGNORECASE)
        scales = re.findall(padrao_scale, texto_limpo, flags=re.IGNORECASE)
        
        for c, s in zip(constructs, scales):
            c_clean = c.strip().strip('"').strip("'")
            s_clean = s.strip().strip('"').strip("'")
            dados_extraidos.append(f"Construct/Question: {c_clean} | Response Scale: {s_clean}")
            
    # Cria uma tabela auxiliar com os textos e o número do item (ex: V1)
    df_textos = pd.DataFrame(dados_extraidos, columns=["Texto"])
    df_textos["Item"] = [f"V{i+1}" for i in range(len(df_textos))]
    df_textos.to_csv("../data/tabela_textos_itens.csv", index=False)
    
    return dados_extraidos

def extrair_openai(textos):
    """Extrai embeddings de 1536 dimensões usando a API da OpenAI."""
    # A chave da API deve estar configurada no terminal: export OPENAI_API_KEY="sk-..."
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente OPENAI_API_KEY não foi encontrada.")
        
    client = OpenAI(api_key=api_key)
    print("Iniciando extração via OpenAI (text-embedding-3-small)...")
    
    resposta = client.embeddings.create(input=textos, model="text-embedding-3-small")
    embeddings = [item.embedding for item in resposta.data]
    
    df = pd.DataFrame(np.array(embeddings))
    df.to_csv("../data/questions_embeddings_openai.csv", index=False)
    print(f"-> Concluído! OpenAI Shape: {df.shape}")

def extrair_modelo_local(textos, nome_modelo, arquivo_saida):
    """Extrai embeddings localmente usando sentence-transformers."""
    print(f"Carregando modelo local: {nome_modelo}...")
    model = SentenceTransformer(nome_modelo)
    
    print(f"Extraindo vetores...")
    embeddings = model.encode(textos, show_progress_bar=False, convert_to_numpy=True)
    
    df = pd.DataFrame(embeddings)
    df.to_csv(f"../data/{arquivo_saida}", index=False)
    print(f"-> Concluído! Shape: {df.shape}")

if __name__ == "__main__":
    print("="*50)
    print(" Pipeline de Embeddings AI-GENIE ")
    print("="*50)
    
    # 1. Ajuste este nome para o CSV original que você colocou na pasta data/
    textos_processados = limpar_textos("../data/perguntas_originais.csv")
    print(f"\n{len(textos_processados)} itens processados e prontos para vetorização.")
    
    # 2. Roda a extração pela API da OpenAI (1536d)
    try:
        extrair_openai(textos_processados)
    except Exception as e:
        print(f"Erro na OpenAI: {e}")
        
    # 3. Roda a extração local do MPNet (768d)
    extrair_modelo_local(
        textos_processados, 
        "all-mpnet-base-v2", 
        "questions_embeddings_mpnet.csv"
    )
    
    # 4. Roda a extração local do MiniLM (384d)
    extrair_modelo_local(
        textos_processados, 
        "paraphrase-MiniLM-L6-v2", 
        "questions_embeddings_minilm.csv"
    )
    
    print("\nEtapa 1 finalizada! Rode o script 02_concatenar_embeddings.py a seguir.")