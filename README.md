# Evaluating the Evaluation of Music Generation Models with Generative Psychometrics

[![LAMIR 2026](https://img.shields.io/badge/Accepted-LAMIR%202026-blue.svg)](https://lamir.edu)
[![R 4.3.3](https://img.shields.io/badge/R-4.3.3-blue?logo=R)](https://www.r-project.org/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![renv](https://img.shields.io/badge/renv-reproducible-purple)](https://rstudio.github.io/renv/)

Pipeline oficial de reprodutibilidade do artigo:
> **"Evaluating the Evaluation of Music Generation Models with Generative Psychometrics"**  
> *Proc. of the 2nd Latin American Music Information Retrieval Workshop (LAMIR 2026)*, Belo Horizonte, Brasil.

---

## 📌 Contexto Científico e Objetivo

A avaliação subjetiva (testes de escuta humana, como MOS e MUSHRA) é o padrão-ouro para validar a qualidade musical e o realismo perceptual de modelos generativos de áudio e música simbólica. No entanto, os protocolos são amplamente heterogêneos, subespecificados e difíceis de comparar entre artigos.

Este projeto aplica o paradigma da **Psicometria Generativa** através do framework **AI-GENIE** (*Automatic Item Generation and Validation with Network-Integrated Evaluation*), utilizando o pacote `EGAnet` em R para:
1. Analisar **279 itens padronizados** extraídos de **150 artigos sobre geração de música** publicados nas conferências **ISMIR (2020–2025)**.
2. Projetar esses itens em um espaço semântico de alta dimensionalidade (*super embedding* de 2688 dimensões) combinando três modelos:
   * **OpenAI** `text-embedding-3-small` (1536d)
   * **MPNet** `all-mpnet-base-v2` (768d)
   * **MiniLM** `paraphrase-MiniLM-L6-v2` (384d)
3. Executar o **Loop de Purificação Iterativa (UVA + bootEGA)** com poda de redundâncias e instabilidade topológica (corte em estabilidade empírica < 0.75).
4. Derivar empiricamente uma **taxonomia robusta de 6 dimensões estáveis** para avaliação de música gerada por IA:
   * 🎧 **Acoustic Quality** (ex: V231, V89)
   * 🤖 **Human-Likeness** (ex: V269, V141)
   * ⏱️ **Rhythmic Fluency / Accuracy** (ex: V130, V77)
   * 🥁 **Groove** (ex: V68, V72)
   * 🧠 **Memorability** (ex: V54, V107)
   * 🎤 **Vocal & Lyrical Quality** (ex: V93, V92)

---

## 🔬 Metodologia do Pipeline

```
  [Coleta de Artigos ISMIR 2020-2025] ──► [Extração e Padronização LLM (N = 279 itens)]
                                                          │
  ┌───────────────────────────────────────────────────────┘
  ▼
  [Embeddings Textuais: OpenAI (1536d) + MPNet (768d) + MiniLM (384d)]
  │
  ▼
  [02_concatenar_embeddings.py] ──► Gera Super Embedding (279 x 2688)
  │
  ▼
  [03_analise_ega_final.R (Loop de Purificação AI-GENIE)]
  │   ├── 1. Transposição: 2688 dimensões (obs) x N itens (variáveis)
  │   ├── 2. Unique Variable Analysis (UVA): elimina redundâncias semânticas
  │   ├── 3. Bootstrap Exploratory Graph Analysis (bootEGA, ncores = 18)
  │   ├── 4. Item Stability: taxa de replicação empírica em cada dimensão
  │   ├── 5. Poda de itens instáveis (< 0.75)
  │   └── 6. Itera até todos os itens terem estabilidade >= 0.75
  │
  ▼
  [Resultados Finais]
      ├── 47 itens únicos com 100% de estabilidade bootstrap
      ├── 6 dimensões latentes (taxonomia unificada)
      ├── rede_final_super_embedding.png (Plot bootEGA completo)
      ├── rede_final_network_only.png (Estrutura da rede - Figura 2 do artigo)
      └── itens_finais_purificados.csv (Itens, dimensões, estabilidade e textos)
```

---

## 📁 Estrutura de Arquivos

```
music-gen-eval/
├── data/
│   ├── questions_embeddings_openai.csv      # Embeddings OpenAI text-embedding-3-small (1536d)
│   ├── questions_embeddings_mpnet.csv       # Embeddings MPNet all-mpnet-base-v2 (768d)
│   ├── questions_embeddings_minilm.csv      # Embeddings MiniLM paraphrase-MiniLM-L6-v2 (384d)
│   ├── super_embeddings_concatenados.csv    # Super embedding concatenado (279 x 2688)
│   ├── tabela_textos_itens.csv              # Mapeamento com textos e IDs dos 279 itens
│   ├── rede_final_super_embedding.png       # Gráfico final bootEGA (Rede + Estabilidade)
│   ├── rede_final_network_only.png          # Gráfico da rede de 6 dimensões (Figura 2 do artigo)
│   └── itens_finais_purificados.csv         # Tabela final dos 47 itens após convergência
├── src/
│   ├── 02_concatenar_embeddings.py          # Concatena os 3 embeddings em um super embedding
│   └── 03_analise_ega_final.R               # Loop iterativo de purificação (UVA + bootEGA)
├── Dockerfile                               # Ambiente reproduzível (Ubuntu 24.04 + R 4.3.3 + uv)
├── docker-compose.yml                       # Orquestração do container com volume mapeado
├── renv.lock                                # Versões exatas e imutáveis dos pacotes R
├── requirements.txt                         # Dependências Python pinadas
├── pyproject.toml                           # Gerenciamento Python moderno com uv
└── README.md                                # Documentação científica e guia de uso
```

---

## 🚀 Como Executar

O pipeline pode ser reproduzido via **Docker** (100% garantido e independente de SO) ou **localmente**.

### Opção 1: Via Docker (Recomendada)

Garante as mesmas bibliotecas C/C++ (`libuv1-dev`, `libglpk-dev`), locale (`LANG=C.UTF-8`) e versões exatas de compiladores.

```bash
# Executa todo o pipeline com volume montado (os resultados salvam em ./data)
docker compose run --rm analysis
```

Ou usando Docker nativo:
```bash
docker build -t music-gen-eval .
docker run --rm -v $(pwd)/data:/app/data music-gen-eval
```

---

### Opção 2: Execução Local (Linux / macOS)

#### 1. Dependências de Sistema
No Ubuntu / Debian / Pop!_OS:
```bash
sudo apt update
sudo apt install -y r-base r-base-dev python3 python3-pip python3-venv curl libuv1-dev libglpk-dev
```

#### 2. Ambiente Python (via `uv`)
```bash
uv sync
# ou via pip clássico:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Ambiente R (via `renv`)
```bash
Rscript -e "install.packages('renv', repos='https://cloud.r-project.org'); renv::restore()"
```

#### 4. Executando o Pipeline Completo

Passo 1 — Concatenação do Super Embedding:
```bash
uv run python src/02_concatenar_embeddings.py
# ou: python3 src/02_concatenar_embeddings.py
```

Passo 2 — Loop de Purificação Iterativa (UVA + bootEGA):
```bash
Rscript src/03_analise_ega_final.R
```

---

## 📊 Resultados Esperados (Artigo LAMIR 2026)

| Dimensão | Item Âncora Mais Representativo | Carga na Rede (NL) |
|---|---|---|
| **Acoustic Quality** | *"Rate the audio quality of these segments."* (V231) | 0.56 |
| **Human-Likeness** | *"Determine whether the sample was composed by a human or a machine."* (V269) | 0.54 |
| **Rhythmic Fluency** | *"Are there notes played at the wrong time?"* (V130) | 0.63 |
| **Groove** | *"Does the sample contain drum sounds?"* (V68) | 0.50 |
| **Memorability** | *"What is your overall perceptual impression of the sample?"* (V54) | 0.73 |
| **Lyrics and Vocals** | *"How singable are the generated lyrics in the sample?"* (V93) | 0.50 |

---

## 📜 Citação

Se você utilizar este pipeline ou os embeddings em sua pesquisa, por favor cite:

```bibtex
@inproceedings{lamir2026musicpsychometrics,
  title     = {Evaluating the Evaluation of Music Generation Models with Generative Psychometrics},
  author    = {Author, F. and Author, S. and Author, T.},
  booktitle = {Proceedings of the 2nd Latin American Music Information Retrieval Workshop (LAMIR 2026)},
  year      = {2026},
  address   = {Belo Horizonte, Brazil}
}
```
