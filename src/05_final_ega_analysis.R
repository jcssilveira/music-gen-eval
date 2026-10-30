# ==============================================================================
# Generative Psychometrics Pipeline (AI-GENIE) - LAMIR 2026
# "Evaluating the Evaluation of Music Generation Models with Generative Psychometrics"
# Stage: Iterative Item Purification Loop (UVA + bootEGA)
# ==============================================================================

# Standardize system locale for Linux commands called by EGAnet (e.g. 'free')
Sys.setenv(LANG = "C", LC_ALL = "C")

# User package library path
user_lib <- Sys.getenv("R_LIBS_USER", unset = "~/R/x86_64-pc-linux-gnu-library/4.3")
user_lib <- path.expand(user_lib)
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

# Ensure required packages are installed
required_pkgs <- c("EGAnet", "ggplot2")
missing_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(missing_pkgs)) {
  cat("Installing missing packages to:", user_lib, "\n")
  options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
  install.packages(missing_pkgs, lib = user_lib, repos = "https://packagemanager.posit.co/cran/__linux__/noble/latest")
}

library(EGAnet)
library(ggplot2)

# Execution parameters and model resolution
# Usage: Rscript src/05_final_ega_analysis.R [super|minilm|mpnet|openai]
args <- commandArgs(trailingOnly = TRUE)
model_arg <- if (length(args) > 0) tolower(trimws(args[1])) else "super"

models_map <- list(
  "super"   = list(filename = "super_embeddings_concatenated.csv", suffix = "super_embedding"),
  "minilm"  = list(filename = "questions_embeddings_minilm.csv",   suffix = "minilm"),
  "mpnet"   = list(filename = "questions_embeddings_mpnet.csv",    suffix = "mpnet"),
  "openai"  = list(filename = "questions_embeddings_openai.csv",   suffix = "openai")
)

# Common aliases
if (model_arg %in% c("mini", "minilm-l6")) model_arg <- "minilm"
if (model_arg %in% c("super_embedding", "super_embeddings", "concatenado", "concatenated")) model_arg <- "super"

if (!model_arg %in% names(models_map)) {
  stop(sprintf("Invalid model '%s'. Available models: %s", 
               model_arg, paste(names(models_map), collapse = ", ")))
}

model_cfg   <- models_map[[model_arg]]
data_file   <- model_cfg$filename
suffix_out  <- model_cfg$suffix

base_dir        <- if (dir.exists("data")) "data" else if (dir.exists("../data")) "../data" else "."
embeddings_dir  <- file.path(base_dir, "embeddings")
raw_dir         <- file.path(base_dir, "raw")
results_dir     <- file.path(base_dir, "results")
plots_dir       <- file.path(base_dir, "plots")
checkpoints_dir <- file.path(base_dir, "checkpoints")

# Ensure target directories exist
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(checkpoints_dir, recursive = TRUE, showWarnings = FALSE)

# Resolve input data path (check embeddings/ first, with fallback)
input_data_path <- if (file.exists(file.path(embeddings_dir, data_file))) {
  file.path(embeddings_dir, data_file)
} else if (file.exists(file.path(embeddings_dir, "super_embeddings_concatenados.csv"))) {
  file.path(embeddings_dir, "super_embeddings_concatenados.csv")
} else if (file.exists(file.path(base_dir, data_file))) {
  file.path(base_dir, data_file)
} else {
  stop(sprintf("Embeddings file not found: %s", data_file))
}

# Resolve item text mapping path (data/raw/item_texts.csv)
texts_path <- if (file.exists(file.path(raw_dir, "item_texts.csv"))) {
  file.path(raw_dir, "item_texts.csv")
} else {
  file.path(base_dir, "data", "raw", "item_texts.csv")
}

# Output file paths
out_plot_full    <- file.path(plots_dir, sprintf("final_network_%s.png", suffix_out))
out_plot_network <- file.path(plots_dir, sprintf("final_network_%s_network_only.png", suffix_out))
out_items_csv    <- file.path(results_dir, sprintf("purified_items_%s.csv", suffix_out))
checkpoint_file  <- file.path(checkpoints_dir, sprintf("checkpoint_purification_%s.rds", suffix_out))

# Load embedding matrix
cat(sprintf("\n====================================================================\n"))
cat(sprintf("SELECTED MODEL: %s\n", toupper(model_arg)))
cat(sprintf("Input embeddings: %s\n", input_data_path))
cat(sprintf("Output items CSV: %s\n", out_items_csv))
cat(sprintf("====================================================================\n"))

cat("Loading embeddings matrix...\n")
mat_emb <- read.csv(input_data_path, check.names = FALSE)
rownames(mat_emb) <- paste0("V", 1:nrow(mat_emb))

# Transpose for network psychometrics:
# Rows = semantic latent dimensions (observations)
# Columns = evaluation items (network nodes)
current_data <- t(mat_emb)

cat(sprintf("Initial data dimensions: %d observations (dimensions) x %d variables (items)\n", 
            nrow(current_data), ncol(current_data)))

# ------------------------------------------------------------------------------
# Iterative Item Purification Loop (UVA + bootEGA)
# ------------------------------------------------------------------------------
stability_threshold <- 0.75
boot_iters          <- 500
seed_val            <- 123
ncores_val          <- 10
iteration           <- 1
converged           <- FALSE
boot_final          <- NULL

# Resume from previous checkpoint if available
if (file.exists(checkpoint_file)) {
  cat(sprintf("\n[INFO] Found checkpoint for model '%s'. Resuming...\n", model_arg))
  cp <- readRDS(checkpoint_file)
  current_data <- cp$current_data
  iteration    <- cp$iteration
  cat(sprintf("Resuming from ROUND %d with %d candidate items.\n", iteration, ncol(current_data)))
}

cat("\n====================================================================\n")
cat(sprintf("Starting Iterative Purification Loop (AI-GENIE - %s)\n", toupper(model_arg)))
cat(sprintf("Cutoff criterion: Empirical item stability >= %.2f\n", stability_threshold))
cat("====================================================================\n")

while (!converged) {
  cat(sprintf("\n--- [ROUND %d] Current items: %d ---\n", iteration, ncol(current_data)))
  
  if (ncol(current_data) < 2) {
    warning("Safety stop: fewer than 2 items remain. Aborting purification.")
    break
  }
  
  # Step A: Unique Variable Analysis (UVA) for redundancy reduction
  cat("Running Unique Variable Analysis (UVA)...\n")
  uva_res <- UVA(current_data)
  
  if (!is.null(uva_res$reduced_data) && ncol(uva_res$reduced_data) > 0) {
    data_for_boot <- as.data.frame(uva_res$reduced_data)
  } else {
    data_for_boot <- as.data.frame(current_data)
  }
  
  cat(sprintf("Items after UVA redundancy pruning: %d\n", ncol(data_for_boot)))
  
  if (ncol(data_for_boot) < 2) {
    warning("Safety stop: fewer than 2 items remain after UVA. Aborting.")
    break
  }
  
  # Step B: Network estimation via Bootstrap Exploratory Graph Analysis (bootEGA)
  cat(sprintf("Running bootEGA (%d iterations, seed = %d, cores = %d)...\n", 
              boot_iters, seed_val, ncores_val))
  boot_res <- bootEGA(data_for_boot, iter = boot_iters, seed = seed_val, ncores = ncores_val, plot.itemStability = FALSE)
  
  # Step C: Extract empirical replication rates (item stability)
  stab_res    <- itemStability(boot_res, IS.plot = FALSE)
  stabilities <- stab_res$item.stability$empirical.dimensions
  
  cat(sprintf("Latent dimensions detected: %d\n", boot_res$EGA$n.dim))
  
  # Step D: Apply stability cutoff (< 0.75)
  unstable_items <- names(stabilities[is.na(stabilities) | stabilities < stability_threshold])
  
  cat(sprintf("Evaluated: %d | Stable: %d | Unstable (< %.2f): %d\n",
              length(stabilities), length(stabilities) - length(unstable_items),
              stability_threshold, length(unstable_items)))
  
  if (length(unstable_items) == 0) {
    cat("\n>>> CONVERGENCE ACHIEVED! <<<\n")
    cat(sprintf("All %d surviving items have stability >= %.2f.\n", 
                length(stabilities), stability_threshold))
    converged <- TRUE
    boot_final <- boot_res
    if (file.exists(checkpoint_file)) file.remove(checkpoint_file)
    break
  } else {
    cat("Unstable items identified for removal in next round:\n")
    print(stabilities[unstable_items])
    
    current_data <- data_for_boot[, !(colnames(data_for_boot) %in% unstable_items), drop = FALSE]
    iteration <- iteration + 1
    
    saveRDS(list(current_data = current_data, iteration = iteration), checkpoint_file)
    cat(sprintf("Checkpoint for Round %d saved to: %s\n", iteration, checkpoint_file))
  }
}

# ------------------------------------------------------------------------------
# Export Final Results
# ------------------------------------------------------------------------------
if (!is.null(boot_final)) {
  cat("\n====================================================================\n")
  cat(sprintf("FINAL PURIFICATION RESULTS - MODEL: %s (LAMIR 2026)\n", toupper(model_arg)))
  cat("====================================================================\n")
  cat(sprintf("Total purification rounds: %d\n", iteration))
  cat(sprintf("Final stable items: %d\n", ncol(boot_final$EGA$network)))
  cat(sprintf("Final latent dimensions: %d\n", boot_final$EGA$n.dim))
  
  # Save full bootEGA plot (Network + Item Stability)
  cat("Saving bootEGA visualization plot...\n")
  plot_boot <- plot(boot_final)
  ggsave(out_plot_full, plot = plot_boot, width = 12, height = 7, dpi = 300)
  cat("Plot saved to:", out_plot_full, "\n")
  
  # Save isolated network topology plot (Figure 2)
  tryCatch({
    plot_net_only <- plot(boot_final$EGA)
    ggsave(out_plot_network, plot = plot_net_only, width = 9, height = 7, dpi = 300)
    cat("Network topology plot saved to:", out_plot_network, "\n")
  }, error = function(e) {
    cat("[Warning] Failed to save isolated network plot:", e$message, "\n")
  })
  
  # Save purified items summary table
  tryCatch({
    wc_final <- boot_final$EGA$wc
    items_df <- data.frame(
      Item = names(wc_final),
      Dimension = as.numeric(wc_final),
      Stability = as.numeric(stab_res$item.stability$empirical.dimensions[names(wc_final)]),
      stringsAsFactors = FALSE
    )
    
    if (file.exists(texts_path)) {
      df_texts <- read.csv(texts_path, stringsAsFactors = FALSE)
      if ("Item" %in% colnames(df_texts)) {
        items_df <- merge(items_df, df_texts, by = "Item", all.x = TRUE)
      }
    }
    
    items_df <- items_df[order(items_df$Dimension, -items_df$Stability), ]
    write.csv(items_df, out_items_csv, row.names = FALSE)
    cat("Purified items table saved to:", out_items_csv, "\n")
  }, error = function(e) {
    cat("[Warning] Failed to export items table:", e$message, "\n")
  })
  
  cat("\nPurification pipeline completed successfully.\n")
} else {
  cat("\nPurification halted before convergence.\n")
}

