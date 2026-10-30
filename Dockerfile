# Ubuntu 24.04 LTS (Noble) for native compatibility with Posit PPM binaries
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System dependencies for R and Python
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

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Setup Python virtualenv and dependencies
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY requirements.txt pyproject.toml /app/
RUN uv venv /app/.venv && uv pip install -r requirements.txt

# Restore exact R package dependencies via renv
COPY renv.lock /app/
RUN R -e 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest")); \
          options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os))); \
          install.packages("renv"); \
          renv::restore(clean = TRUE, prompt = FALSE)'

COPY . /app/

# Default pipeline entrypoint
CMD ["bash", "-c", "python3 src/02_concatenate_embeddings.py && Rscript src/03_final_ega_analysis.R"]
