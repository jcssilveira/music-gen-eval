# Evaluating the Evaluation of Music Generation Models with Generative Psychometrics

Reproducibility pipeline for the paper *"Evaluating the Evaluation of Music Generation Models with Generative Psychometrics"* (LAMIR 2026).

---

## Literature Curation & Dataset Provenance

The evaluation items analyzed in this framework were curated through a systematic review of the International Society for Music Information Retrieval (ISMIR) conference proceedings (2020–2025). The full curated dataset, including Zenodo DOIs, author metadata, abstracts, and evaluation excerpts, is available at [`data/raw/ismir_paper_curation.xlsx`](data/raw/ismir_paper_curation.xlsx).

---

## Project Structure

```text
music-gen-eval/
├── archive/                                 # Historical versions and legacy runs (ignored in git)
│   ├── v1/
│   ├── v2/
│   └── v3/
├── data/
│   ├── raw/                                 # Raw input questions & item mapping
│   │   ├── ismir_paper_curation.xlsx        # Curated ISMIR (2020-2025) papers
│   │   ├── questions_no_duplicates.csv      # Deduplicated survey items
│   │   └── item_texts.csv                   # Item ID to text mapping
│   ├── embeddings/                          # Extracted vector embeddings
│   │   ├── questions_embeddings_openai.csv   # OpenAI (1536d)
│   │   ├── questions_embeddings_mpnet.csv    # MPNet (768d)
│   │   ├── questions_embeddings_minilm.csv   # MiniLM (384d)
│   │   └── super_embeddings_concatenated.csv # Concatenated (2688d)
│   ├── results/                             # Final purified items tables (CSV)
│   │   ├── purified_items_super_embedding.csv
│   │   ├── purified_items_minilm.csv
│   │   ├── purified_items_mpnet.csv
│   │   └── purified_items_openai.csv
│   ├── plots/                               # Network topology plots (PNG)
│   │   ├── final_network_super_embedding.png
│   │   ├── final_network_minilm.png
│   │   ├── final_network_mpnet.png
│   │   └── final_network_openai.png
│   └── checkpoints/                         # bootEGA checkpoint files (*.rds)
├── src/
│   ├── 01_generate_embeddings.py            # Extract embeddings (OpenAI, MPNet, MiniLM)
│   ├── 02_concatenate_embeddings.py         # Concatenate embeddings into super-embedding
│   └── 03_final_ega_analysis.R              # Iterative purification loop (UVA + bootEGA)
├── Dockerfile                               # Container definition (Ubuntu 24.04, R 4.3, Python 3.12)
├── docker-compose.yml                       # Compose orchestration with mounted volume
├── renv.lock                                # R package dependencies
├── requirements.txt                         # Python dependencies (pip)
└── pyproject.toml                           # Python project configuration (uv)
```

---

## Quickstart

### Option 1: Docker (Recommended)

Run the full pipeline inside the container:

```bash
docker compose run --rm analysis
```

Or using native Docker:

```bash
docker build -t music-gen-eval .
docker run --rm -v $(pwd)/data:/app/data music-gen-eval
```

---

### Option 2: Local Setup

#### 1. System Dependencies (Debian/Ubuntu)

```bash
sudo apt update && sudo apt install -y r-base r-base-dev python3 python3-pip python3-venv curl libuv1-dev libglpk-dev
```

#### 2. Environment Setup

**Python:**
```bash
uv sync
# or using standard venv:
# python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**R:**
```bash
Rscript -e "install.packages('renv', repos='https://cloud.r-project.org'); renv::restore()"
```

#### 3. Run Pipeline

```bash
# 1. (Optional) Extract embeddings from raw questions
python src/01_generate_embeddings.py

# 2. Concatenate embeddings into super-embedding
python src/02_concatenate_embeddings.py

# 3. Run EGA iterative item purification
Rscript src/03_final_ega_analysis.R [super|minilm|mpnet|openai]
```
