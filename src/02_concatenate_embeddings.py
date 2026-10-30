from pathlib import Path
import pandas as pd

# Dynamic path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMB_DIR = DATA_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

source_dir = EMB_DIR if (EMB_DIR / "questions_embeddings_openai.csv").exists() else DATA_DIR

# Load individual embedding matrices
df_openai = pd.read_csv(source_dir / "questions_embeddings_openai.csv")
df_mpnet  = pd.read_csv(source_dir / "questions_embeddings_mpnet.csv")
df_minilm = pd.read_csv(source_dir / "questions_embeddings_minilm.csv")

# Drop any residual unnamed index columns
for df in [df_openai, df_mpnet, df_minilm]:
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

# Concatenate horizontally (1536 + 768 + 384 = 2688 dimensions)
super_embedding = pd.concat([df_openai, df_mpnet, df_minilm], axis=1)

output_path = EMB_DIR / "super_embeddings_concatenated.csv"
super_embedding.to_csv(output_path, index=False)
print(f"Concatenated super-embeddings saved to {output_path} | Shape: {super_embedding.shape}")

