# Usa Ubuntu 24.04 LTS (Noble) para compatibilidade nativa com binários do Posit PPM
FROM ubuntu:24.04

# Evita prompts interativos durante instalação do apt e padroniza locale
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Instala R, Python e bibliotecas essenciais de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    libuv1-dev \
    libglpk-dev \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instala o uv (gerenciador de pacotes Python de alta performance)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1. Cria virtualenv e instala dependências Python via uv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY requirements.txt pyproject.toml /app/
RUN uv venv /app/.venv && uv pip install -r requirements.txt

# 2. Instala renv e restaura as versões exatas dos pacotes R do renv.lock
COPY renv.lock /app/
RUN R -e 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest")); \
          options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os))); \
          install.packages("renv"); \
          renv::restore(clean = TRUE, prompt = FALSE)'

# 3. Copia o restante dos arquivos do projeto
COPY . /app/

# Comando padrão: executa concatenação de embeddings e análise EGA
CMD ["bash", "-c", "python3 src/02_concatenar_embeddings.py && Rscript src/03_analise_ega_final.R"]
