# Evaluating the Evaluation of Music Generation Models with Generative Psychometrics

Reproducibility pipeline for the paper *"Evaluating the Evaluation of Music Generation Models with Generative Psychometrics"* (LAMIR 2026).

---

## Project Structure

```text
music-gen-eval/
├── data/
│   ├── questions_embeddings_openai.csv      # OpenAI embeddings (1536d)
│   ├── questions_embeddings_mpnet.csv       # MPNet embeddings (768d)
│   ├── questions_embeddings_minilm.csv      # MiniLM embeddings (384d)
│   ├── super_embeddings_concatenados.csv    # Concatenated super-embeddings (2688d)
│   ├── tabela_textos_itens.csv              # Item texts and IDs
│   ├── rede_final_super_embedding.png       # Final EGA network plot
│   ├── rede_final_network_only.png          # Network topology plot
│   └── itens_finais.csv                     # Final purified items
├── src/
│   ├── 01_gerar_embeddings.py               # Extract embeddings (OpenAI, MPNet, MiniLM)
│   ├── 02_concatenar_embeddings.py          # Concatenate embeddings into super-embedding
│   └── 03_analise_ega_final.R               # Iterative purification loop (UVA + bootEGA)
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
# 1. Concatenate embeddings
python src/02_concatenar_embeddings.py

# 2. Run EGA iterative purification
Rscript src/03_analise_ega_final.R
```
