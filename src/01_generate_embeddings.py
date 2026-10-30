import os
import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# Environment & Logging Configuration
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Resolve base paths dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EMB_DIR = DATA_DIR / "embeddings"

RAW_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)


def load_questions(filepath: Path) -> list[str]:
    """Load items from file, remove duplicates while preserving order, and create item mapping."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Questions file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip().strip('"') for line in f if line.strip()]

    # Deduplicate while preserving order of first appearance
    questions = list(dict.fromkeys(lines))
    removed = len(lines) - len(questions)
    if removed > 0:
        print(f"-> Deduplication: removed {removed} duplicates ({len(lines)} -> {len(questions)} unique items)")

    # Save item ID to text mapping (V1..Vn)
    df_items = pd.DataFrame({
        "Item": [f"V{i + 1}" for i in range(len(questions))],
        "Text": questions
    })

    mapping_path = RAW_DIR / "item_texts.csv"
    df_items.to_csv(mapping_path, index=False)
    print(f"-> Item mapping saved to: {mapping_path} ({len(df_items)} items)")

    return questions


def extract_openai_embeddings(texts: list[str]) -> None:
    """Extract 1536-dimensional embeddings using OpenAI text-embedding-3-small."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Set it in your terminal prior to execution: export OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=api_key)
    print("Extracting OpenAI embeddings (text-embedding-3-small)...")

    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = [item.embedding for item in response.data]

    df = pd.DataFrame(np.array(embeddings))
    output_path = EMB_DIR / "questions_embeddings_openai.csv"
    df.to_csv(output_path, index=False)
    print(f"-> OpenAI embeddings saved to {output_path} | Shape: {df.shape}")


def extract_local_embeddings(texts: list[str], model_name: str, output_filename: str) -> None:
    """Extract embeddings locally using sentence-transformers."""
    print(f"Loading local transformer: {model_name}...")
    model = SentenceTransformer(model_name)

    print("Encoding texts...")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    df = pd.DataFrame(embeddings)
    output_path = EMB_DIR / output_filename
    df.to_csv(output_path, index=False)
    print(f"-> Local embeddings saved to {output_path} | Shape: {df.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("AI-GENIE Embeddings Extraction Pipeline")
    print("=" * 60)

    # Locate input questions file
    questions_file = RAW_DIR / "questions_no_duplicates.csv"
    if not questions_file.is_file():
        questions_file = DATA_DIR / "questions_no_duplicates.csv"
    if not questions_file.is_file():
        questions_file = DATA_DIR / "perguntas_originais.csv"

    questions = load_questions(questions_file)
    print(f"\nLoaded {len(questions)} items ready for vectorization.\n")

    # 1. OpenAI embeddings (1536d)
    try:
        extract_openai_embeddings(questions)
    except Exception as e:
        print(f"[Warning] OpenAI extraction skipped: {e}\n")

    # 2. Local all-mpnet-base-v2 embeddings (768d)
    extract_local_embeddings(
        questions,
        "all-mpnet-base-v2",
        "questions_embeddings_mpnet.csv"
    )

    # 3. Local paraphrase-MiniLM-L6-v2 embeddings (384d)
    extract_local_embeddings(
        questions,
        "paraphrase-MiniLM-L6-v2",
        "questions_embeddings_minilm.csv"
    )

    print("\nEmbeddings extraction completed successfully.")

