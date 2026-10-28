from pathlib import Path
import pandas as pd

# Define o caminho base da pasta data de forma dinâmica
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Carrega os arquivos deixando o Pandas criar um índice numérico limpo
df_openai = pd.read_csv(DATA_DIR / "questions_embeddings_openai.csv")
df_mpnet = pd.read_csv(DATA_DIR / "questions_embeddings_mpnet.csv")
df_minilm = pd.read_csv(DATA_DIR / "questions_embeddings_minilm.csv")

# Caso algum dos CSVs tenha sido salvo com aquela coluna de índice inútil do Pandas ("Unnamed: 0"),
# nós a removemos aqui para não poluir as 2688 dimensões do super embedding
for df in [df_openai, df_mpnet, df_minilm]:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

# Concatena os DataFrames horizontalmente (axis=1) usando a posição da linha
super_embedding = pd.concat([df_openai, df_mpnet, df_minilm], axis=1)

# Exporta o resultado (usamos index=False para não gerar uma coluna de índice nova 
# que possa quebrar o script em R depois)
caminho_saida = DATA_DIR / "super_embeddings_concatenados.csv"
super_embedding.to_csv(caminho_saida, index=False)
print(f"Super embedding gerado com sucesso em {caminho_saida}! Shape: {super_embedding.shape}")

