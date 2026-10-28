# ==============================================================================
# Pipeline de Psicometria Generativa (AI-GENIE) - LAMIR 2026
# Artigo: "Evaluating the Evaluation of Music Generation Models with Generative Psychometrics"
# Etapa: Loop de Purificação Iterativa de Itens (UVA + bootEGA)
# ==============================================================================

# 1. Configura o locale do sistema para padronizar comandos do Linux (ex: 'free')
# O EGAnet internamente executa 'free' e busca por 'available'. Em sistemas em Português
# o comando retornava 'disponível', causando 'argumento tem comprimento zero'.
Sys.setenv(LANG = "C", LC_ALL = "C")

# 2. Define e cria pasta local para pacotes do usuário
pasta_local <- Sys.getenv("R_LIBS_USER", unset = "~/R/x86_64-pc-linux-gnu-library/4.3")
pasta_local <- path.expand(pasta_local)
dir.create(pasta_local, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(pasta_local, .libPaths()))

# 3. Garante que os pacotes necessários estejam carregados/instalados
pacotes <- c("EGAnet", "ggplot2")
novos_pacotes <- pacotes[!(pacotes %in% installed.packages()[, "Package"])]
if (length(novos_pacotes)) {
  cat("Instalando pacotes em:", pasta_local, "\n")
  options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
  install.packages(novos_pacotes, lib = pasta_local, repos = "https://packagemanager.posit.co/cran/__linux__/noble/latest")
}

library(EGAnet)
library(ggplot2)

# 4. Resolução de caminhos (executando da raiz ou de dentro de src/)
caminho_dados <- if (file.exists("data/super_embeddings_concatenados.csv")) {
  "data/super_embeddings_concatenados.csv"
} else if (file.exists("../data/super_embeddings_concatenados.csv")) {
  "../data/super_embeddings_concatenados.csv"
} else {
  stop("Arquivo super_embeddings_concatenados.csv não encontrado!")
}

pasta_saida <- if (dir.exists("data")) "data" else "../data"
caminho_saida_rede <- file.path(pasta_saida, "rede_final_super_embedding.png")
caminho_saida_network <- file.path(pasta_saida, "rede_final_network_only.png")
caminho_saida_itens <- file.path(pasta_saida, "itens_finais_purificados.csv")
caminho_textos <- file.path(pasta_saida, "tabela_textos_itens.csv")

# 5. Carregamento da Matriz de Super Embeddings
cat("Carregando o Super Embedding concatenado...\n")
matriz_emb <- read.csv(caminho_dados, check.names = FALSE)
rownames(matriz_emb) <- paste0("V", 1:nrow(matriz_emb))

# Transpõe para a análise psicométrica:
# Linhas = 2688 dimensões semânticas (observações/espaço latente)
# Colunas = 279 itens de avaliação (nós da rede)
dados_atuais <- t(matriz_emb)

cat(sprintf("Dimensão inicial dos dados: %d observações (dimensões) x %d variáveis (itens)\n", 
            nrow(dados_atuais), ncol(dados_atuais)))

# ==============================================================================
# 6. Loop de Purificação Iterativa (UVA + bootEGA)
# ==============================================================================
limiar_estabilidade <- 0.75
iter_boot <- 500       # Padrão do bootEGA para estimativa bootstrap robusta
seed_val <- 123
ncores_val <- 18       # Paralelização lógica no Pop!_OS
rodada <- 1
convergido <- FALSE
boot_final <- NULL

cat("\n====================================================================\n")
cat("Iniciando o Loop de Purificação Iterativa (AI-GENIE)\n")
cat(sprintf("Critério de corte: Estabilidade de replicação empírica >= %.2f\n", limiar_estabilidade))
cat("====================================================================\n")

while (!convergido) {
  cat(sprintf("\n--- [RODADA %d] Candidatos atuais: %d itens ---\n", rodada, ncol(dados_atuais)))
  
  # Trava de segurança: necessário ao menos 2 itens para estimar uma rede
  if (ncol(dados_atuais) < 2) {
    warning("Trava de segurança ativada: restam menos de 2 itens na matriz. Interrompendo purificação.")
    break
  }
  
  # Passo A: Análise de Variáveis Únicas (UVA) para eliminar redundâncias semânticas
  cat("Aplicando Unique Variable Analysis (UVA)...\n")
  uva_atual <- UVA(dados_atuais)
  dados_para_boot <- as.data.frame(uva_atual$reduced_data)
  
  cat(sprintf("Itens após poda de redundância UVA: %d itens restantes\n", ncol(dados_para_boot)))
  
  # Trava de segurança pós-UVA
  if (ncol(dados_para_boot) < 2) {
    warning("Trava de segurança ativada: restam menos de 2 itens após UVA. Interrompendo purificação.")
    break
  }
  
  # Passo B: Estimativa da rede com Bootstrap Exploratory Graph Analysis (bootEGA)
  cat(sprintf("Executando bootEGA (%d iterações, seed = %d, ncores = %d)...\n", 
              iter_boot, seed_val, ncores_val))
  boot_atual <- bootEGA(dados_para_boot, iter = iter_boot, seed = seed_val, ncores = ncores_val)
  
  # Passo C: Extração das taxas de replicação empírica (item stability)
  is_res <- itemStability(boot_atual, IS.plot = FALSE)
  estabilidade <- is_res$item.stability$empirical.dimensions
  
  n_dimensoes <- boot_atual$EGA$n.dim
  cat(sprintf("Dimensões latentes detectadas nesta rodada: %d\n", n_dimensoes))
  
  # Passo D: Aplicação do critério de corte (< 0.75)
  itens_instaveis <- names(estabilidade[is.na(estabilidade) | estabilidade < limiar_estabilidade])
  
  cat(sprintf("Itens avaliados: %d | Itens estáveis: %d | Itens instáveis (< %.2f): %d\n",
              length(estabilidade), length(estabilidade) - length(itens_instaveis),
              limiar_estabilidade, length(itens_instaveis)))
  
  if (length(itens_instaveis) == 0) {
    cat("\n>>> CONVERGÊNCIA DECLARADA! <<<\n")
    cat(sprintf("Todos os %d itens sobreviventes possuem estabilidade >= %.2f!\n", 
                length(estabilidade), limiar_estabilidade))
    convergido <- TRUE
    boot_final <- boot_atual
    break
  } else {
    cat("Itens instáveis identificados que serão podados para a próxima rodada:\n")
    print(estabilidade[itens_instaveis])
    
    # Remove os itens instáveis de dados_para_boot para iniciar a próxima rodada
    dados_atuais <- dados_para_boot[, !(colnames(dados_para_boot) %in% itens_instaveis), drop = FALSE]
    rodada <- rodada + 1
  }
}

# ==============================================================================
# 7. Finalização e Exportação dos Resultados
# ==============================================================================
if (!is.null(boot_final)) {
  cat("\n====================================================================\n")
  cat("RESULTADOS DA PURIFICAÇÃO FINAL (LAMIR 2026)\n")
  cat("====================================================================\n")
  cat(sprintf("Total de rodadas de purificação: %d\n", rodada))
  cat(sprintf("Número de itens finais estáveis: %d\n", ncol(boot_final$EGA$network)))
  cat(sprintf("Número de dimensões latentes finais: %d\n", boot_final$EGA$n.dim))
  
  # Salva o plot completo do bootEGA (Rede + Estabilidade dos Itens)
  cat("Salvando gráfico do bootEGA final em PNG...\n")
  grafico_boot <- plot(boot_final)
  ggsave(caminho_saida_rede, plot = grafico_boot, width = 12, height = 7, dpi = 300)
  cat("Gráfico salvo em:", caminho_saida_rede, "\n")
  
  # Salva também o gráfico da rede isolada (como na Figura 2 do artigo)
  tryCatch({
    grafico_rede_only <- plot(boot_final$EGA)
    ggsave(caminho_saida_network, plot = grafico_rede_only, width = 9, height = 7, dpi = 300)
    cat("Gráfico da estrutura da rede (Fig. 2) salvo em:", caminho_saida_network, "\n")
  }, error = function(e) {
    cat("Aviso ao salvar plot isolado da rede:", e$message, "\n")
  })
  
  # Tabela detalhada dos itens purificados
  tryCatch({
    wc_final <- boot_final$EGA$wc
    itens_df <- data.frame(
      Item = names(wc_final),
      Dimensao = as.numeric(wc_final),
      Estabilidade = as.numeric(is_res$item.stability$empirical.dimensions[names(wc_final)]),
      stringsAsFactors = FALSE
    )
    
    # Se houver a tabela original de textos, junta os textos dos itens
    if (file.exists(caminho_textos)) {
      tabela_textos <- read.csv(caminho_textos, stringsAsFactors = FALSE)
      if ("Item" %in% colnames(tabela_textos)) {
        itens_df <- merge(itens_df, tabela_textos, by = "Item", all.x = TRUE)
      }
    }
    
    # Ordena por Dimensão e Estabilidade
    itens_df <- itens_df[order(itens_df$Dimensao, -itens_df$Estabilidade), ]
    write.csv(itens_df, caminho_saida_itens, row.names = FALSE)
    cat("Tabela de itens purificados salva em:", caminho_saida_itens, "\n")
  }, error = function(e) {
    cat("Aviso ao gerar tabela de itens:", e$message, "\n")
  })
  
  cat("\nPipeline de purificação concluído com sucesso!\n")
} else {
  cat("\nA purificação foi interrompida sem convergência (verifique as travas de segurança).\n")
}


