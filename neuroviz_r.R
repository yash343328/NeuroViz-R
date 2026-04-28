################################################################################
##                                                                            ##
##   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██╗   ██╗██╗███████╗         ##
##   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██║   ██║██║╚══███╔╝         ##
##   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║   ██║██║  ███╔╝          ##
##   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚██╗ ██╔╝██║ ███╔╝           ##
##   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝ ╚████╔╝ ██║███████╗         ##
##   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝   ╚═══╝  ╚═╝╚══════╝         ##
##                                                                            ##
##   NeuroViz: Neural Network Visualization & Interpretability Framework      ##
##   Version: 2.0.0  |  R Internal Libraries Project  |                       ##
##   Author: Yash Jain                                                        ##
##   Research Areas: XAI, Graph Neural Networks, Spiking Neural Networks      ##
##                                                                            ##
################################################################################
##
##  PROJECT STRUCTURE (all in one file):
##  ─────────────────────────────────────
##  [1]  SECTION A — Internal Library: neuroviz.core
##       Neural architecture builder, layer abstractions, forward pass engine
##
##  [2]  SECTION B — Internal Library: neuroviz.explain
##       SHAP-style explainability, Grad-CAM, attention rollout
##
##  [3]  SECTION C — Internal Library: neuroviz.spike
##       Spiking Neural Network (SNN) simulation engine (2026 trend)
##
##  [4]  SECTION D — Internal Library: neuroviz.graph
##       Graph Neural Network builder (GCN, GAT, GraphSAGE)
##
##  [5]  SECTION E — Internal Library: neuroviz.vis
##       ggplot2-based visualization suite for all network types
##
##  [6]  SECTION F — Demo & Research Experiments
##       Runnable experiments to explore and extend
##
################################################################################



# ==============================================================================
# DEPENDENCIES CHECK & INSTALL
# ==============================================================================

.neuroviz_deps <- c("ggplot2", "dplyr", "tidyr", "Matrix", "igraph",
                    "ggraph", "purrr", "stringr", "scales", "grDevices",
                    "methods", "stats", "utils")

.check_and_install <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) {
    message("[NeuroViz] Installing missing packages: ", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  invisible(lapply(pkgs, library, character.only = TRUE, quietly = TRUE))
  message("[NeuroViz] All dependencies loaded.")
}

.check_and_install(.neuroviz_deps)



################################################################################
##  SECTION A:  neuroviz.core  ─  Neural Architecture Engine                 ##
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
# A1. Environment & S3 Class Definitions
# ──────────────────────────────────────────────────────────────────────────────

#' Create a new NeuroViz Network object
#'
#' @param name  Character. Name of the network.
#' @param task  Character. "classification", "regression", or "generative"
#' @return      A `nvz_network` S3 object
#'
#' @examples
#' net <- nvz_network("MyNet", task = "classification")
nvz_network <- function(name = "Network", task = "classification") {
  net <- list(
    name       = name,
    task       = task,
    layers     = list(),
    weights    = list(),
    history    = list(),
    compiled   = FALSE,
    forward_fn = NULL,
    metadata   = list(
      created    = Sys.time(),
      version    = "2.0.0",
      framework  = "neuroviz.core"
    )
  )
  class(net) <- c("nvz_network", "list")
  message(sprintf("[neuroviz.core] Network '%s' created for task: %s", name, task))
  net
}


# ──────────────────────────────────────────────────────────────────────────────
# A2. Layer Factory — all standard and novel layers
# ──────────────────────────────────────────────────────────────────────────────

#' Create a Dense (fully-connected) layer
nvz_layer_dense <- function(units, activation = "relu", name = NULL,
                             dropout = 0.0, batch_norm = FALSE) {
  layer <- list(
    type        = "dense",
    units       = units,
    activation  = activation,
    dropout     = dropout,
    batch_norm  = batch_norm,
    name        = name %||% paste0("dense_", sample(1000:9999, 1)),
    trainable   = TRUE,
    params      = NULL  # initialized on compile
  )
  class(layer) <- c("nvz_layer_dense", "nvz_layer", "list")
  layer
}

#' Create a Convolutional layer (1D or 2D)
nvz_layer_conv <- function(filters, kernel_size, stride = 1, padding = "same",
                            activation = "relu", dim = "2D", name = NULL) {
  layer <- list(
    type        = "conv",
    filters     = filters,
    kernel_size = kernel_size,
    stride      = stride,
    padding     = padding,
    activation  = activation,
    dim         = dim,
    name        = name %||% paste0("conv", dim, "_", sample(1000:9999, 1)),
    trainable   = TRUE,
    params      = NULL
  )
  class(layer) <- c("nvz_layer_conv", "nvz_layer", "list")
  layer
}

#' Create a Multi-Head Attention layer (Transformer block)
nvz_layer_attention <- function(num_heads, key_dim, value_dim = NULL,
                                 dropout = 0.0, name = NULL) {
  layer <- list(
    type       = "attention",
    num_heads  = num_heads,
    key_dim    = key_dim,
    value_dim  = value_dim %||% key_dim,
    dropout    = dropout,
    name       = name %||% paste0("mha_", sample(1000:9999, 1)),
    trainable  = TRUE,
    params     = NULL
  )
  class(layer) <- c("nvz_layer_attention", "nvz_layer", "list")
  layer
}

#' Create a Liquid Time-Constant (LTC) layer — 2026 trend: neuromorphic computing
#' Based on: "Liquid Time-constant Networks" (Hasani et al., 2021)
nvz_layer_ltc <- function(units, ode_unfolds = 6, epsilon = 1e-8, name = NULL) {
  layer <- list(
    type        = "ltc",
    units       = units,
    ode_unfolds = ode_unfolds,  # ODE solver steps
    epsilon     = epsilon,
    name        = name %||% paste0("ltc_", sample(1000:9999, 1)),
    trainable   = TRUE,
    params      = NULL,
    description = "Liquid Time-Constant: continuous-time RNN with ODE dynamics"
  )
  class(layer) <- c("nvz_layer_ltc", "nvz_layer", "list")
  layer
}

#' Create a Mixture of Experts (MoE) layer — 2026 trend: sparse models
nvz_layer_moe <- function(num_experts, expert_units, top_k = 2, name = NULL) {
  layer <- list(
    type         = "moe",
    num_experts  = num_experts,
    expert_units = expert_units,
    top_k        = top_k,  # sparse gating: activate only top-k experts
    name         = name %||% paste0("moe_", sample(1000:9999, 1)),
    trainable    = TRUE,
    params       = NULL,
    description  = paste0("Sparse MoE: ", num_experts, " experts, top-", top_k, " gating")
  )
  class(layer) <- c("nvz_layer_moe", "nvz_layer", "list")
  layer
}

# Null-coalescing operator (internal)
`%||%` <- function(a, b) if (!is.null(a)) a else b


# ──────────────────────────────────────────────────────────────────────────────
# A3. Network Builder — add layers, compile, summary
# ──────────────────────────────────────────────────────────────────────────────

#' Add a layer to the network
#' @param net   nvz_network object
#' @param layer nvz_layer object
#' @return      Updated nvz_network
nvz_add_layer <- function(net, layer) {
  stopifnot(inherits(net, "nvz_network"), inherits(layer, "nvz_layer"))
  idx <- length(net$layers) + 1
  layer$index <- idx
  net$layers[[idx]] <- layer
  net$compiled <- FALSE
  net
}

#' Pipe operator for adding layers
#' @examples
#' net <- nvz_network("Transformer") %+% nvz_layer_dense(128)
`%+%` <- function(net, layer) nvz_add_layer(net, layer)


#' Compile the network — initialize weights, count parameters
#' @param net        nvz_network
#' @param input_dim  Integer. Input dimensionality
#' @param optimizer  Character. "adam", "sgd", "rmsprop"
#' @param loss       Character. "cross_entropy", "mse", "huber"
nvz_compile <- function(net, input_dim, optimizer = "adam",
                        loss = "cross_entropy", learning_rate = 0.001) {
  stopifnot(inherits(net, "nvz_network"))

  net$input_dim    <- input_dim
  net$optimizer    <- optimizer
  net$loss         <- loss
  net$learning_rate <- learning_rate
  net$total_params <- 0

  prev_dim <- input_dim

  for (i in seq_along(net$layers)) {
    layer <- net$layers[[i]]

    # Parameter initialization (Xavier/Glorot uniform)
    if (layer$type == "dense") {
      fan_in  <- prev_dim
      fan_out <- layer$units
      limit   <- sqrt(6 / (fan_in + fan_out))
      net$weights[[layer$name]] <- list(
        W = matrix(runif(fan_in * fan_out, -limit, limit), fan_in, fan_out),
        b = rep(0, fan_out)
      )
      net$layers[[i]]$params <- fan_in * fan_out + fan_out
      net$total_params        <- net$total_params + net$layers[[i]]$params
      prev_dim <- fan_out

    } else if (layer$type == "conv") {
      # Simplified: treat as dense equivalent for parameter counting
      k    <- if (layer$dim == "2D") layer$kernel_size^2 else layer$kernel_size
      p    <- k * layer$filters + layer$filters
      net$layers[[i]]$params <- p
      net$total_params        <- net$total_params + p
      prev_dim <- layer$filters

    } else if (layer$type == "attention") {
      # Q, K, V projection params
      p <- 4 * layer$key_dim * prev_dim * layer$num_heads
      net$layers[[i]]$params <- p
      net$total_params        <- net$total_params + p

    } else if (layer$type == "ltc") {
      # LTC has W_tau, W_A, W_sigma matrices + biases
      p <- 4 * prev_dim * layer$units + 3 * layer$units
      net$layers[[i]]$params <- p
      net$total_params        <- net$total_params + p
      prev_dim <- layer$units

    } else if (layer$type == "moe") {
      # Gate + N experts
      p <- prev_dim * layer$num_experts +
           layer$num_experts * prev_dim * layer$expert_units
      net$layers[[i]]$params <- p
      net$total_params        <- net$total_params + p
      prev_dim <- layer$expert_units
    }
  }

  net$compiled <- TRUE
  message(sprintf("[neuroviz.core] Compiled '%s' | Total params: %s | Optimizer: %s",
                  net$name,
                  formatC(net$total_params, format = "d", big.mark = ","),
                  optimizer))
  net
}


#' Print network summary (like Keras model.summary())
print.nvz_network <- function(x, ...) {
  cat("\n")
  cat("╔══════════════════════════════════════════════════════════╗\n")
  cat(sprintf("║  NeuroViz Network: %-38s║\n", x$name))
  cat(sprintf("║  Task: %-50s║\n", x$task))
  cat("╠══════════════════════════════════════════════════════════╣\n")
  cat(sprintf("║  %-20s %-15s %-18s║\n", "Layer (type)", "Output Shape", "Param #"))
  cat("╠══════════════════════════════════════════════════════════╣\n")

  for (lyr in x$layers) {
    pcount <- if (!is.null(lyr$params)) formatC(lyr$params, format = "d", big.mark = ",") else "—"
    shape  <- if (!is.null(lyr$units)) paste0("(None, ", lyr$units, ")") else "variable"
    cat(sprintf("║  %-20s %-15s %-18s║\n",
                substr(paste0(lyr$name, " (", lyr$type, ")"), 1, 20),
                substr(shape, 1, 15),
                substr(pcount, 1, 18)))
  }

  cat("╠══════════════════════════════════════════════════════════╣\n")
  total <- if (x$compiled) formatC(x$total_params, format = "d", big.mark = ",") else "not compiled"
  cat(sprintf("║  Total params: %-42s║\n", total))
  cat(sprintf("║  Compiled: %-46s║\n", if (x$compiled) "YES" else "NO"))
  cat("╚══════════════════════════════════════════════════════════╝\n")
  invisible(x)
}


# ──────────────────────────────────────────────────────────────────────────────
# A4. Activation Functions (pure R, vectorized)
# ──────────────────────────────────────────────────────────────────────────────

.activations <- list(
  relu       = function(x) pmax(0, x),
  leaky_relu = function(x, alpha = 0.01) ifelse(x >= 0, x, alpha * x),
  sigmoid    = function(x) 1 / (1 + exp(-x)),
  tanh       = function(x) tanh(x),
  gelu       = function(x) 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))),
  swish      = function(x) x * (1 / (1 + exp(-x))),
  softmax    = function(x) {
    e <- exp(x - max(x))
    e / sum(e)
  },
  linear     = function(x) x,
  # Mish activation (2026 trend: outperforms ReLU in many benchmarks)
  mish       = function(x) x * tanh(log(1 + exp(x)))
)

#' Apply an activation function by name
#' @param x    Numeric vector or matrix
#' @param name Character. Activation name
nvz_activate <- function(x, name = "relu") {
  fn <- .activations[[name]]
  if (is.null(fn)) stop(sprintf("Unknown activation: '%s'", name))
  fn(x)
}

#' Derivative of activation (for backprop research)
nvz_activate_grad <- function(x, name = "relu") {
  switch(name,
    relu    = as.numeric(x > 0),
    sigmoid = { s <- .activations$sigmoid(x); s * (1 - s) },
    tanh    = 1 - tanh(x)^2,
    gelu    = {
      c1 <- sqrt(2/pi); c2 <- 0.044715
      tanh_val <- tanh(c1 * (x + c2 * x^3))
      0.5 * (1 + tanh_val) + 0.5 * x * (1 - tanh_val^2) * c1 * (1 + 3*c2*x^2)
    },
    swish   = {
      s <- .activations$sigmoid(x)
      s + x * s * (1 - s)
    },
    rep(1, length(x))  # linear fallback
  )
}


# ──────────────────────────────────────────────────────────────────────────────
# A5. Forward Pass Engine
# ──────────────────────────────────────────────────────────────────────────────

#' Run a forward pass through the network
#' @param net   Compiled nvz_network
#' @param X     Numeric matrix (samples × features)
#' @param store_activations Logical. Store per-layer activations for XAI
#' @return      List with $output and optionally $activations
nvz_forward <- function(net, X, store_activations = FALSE) {
  stopifnot(net$compiled, is.matrix(X) || is.numeric(X))
  if (is.numeric(X) && !is.matrix(X)) X <- matrix(X, nrow = 1)

  activations <- list()
  h <- X

  for (lyr in net$layers) {
    if (lyr$type == "dense") {
      W <- net$weights[[lyr$name]]$W
      b <- net$weights[[lyr$name]]$b
      h <- sweep(h %*% W, 2, b, "+")
      h <- t(apply(h, 1, nvz_activate, name = lyr$activation))

      if (lyr$dropout > 0 && !is.null(net$training) && net$training) {
        mask <- matrix(rbinom(length(h), 1, 1 - lyr$dropout), nrow(h), ncol(h))
        h <- h * mask / (1 - lyr$dropout)
      }

    } else if (lyr$type == "attention") {
      # Simplified self-attention (for research scaffolding)
      d <- ncol(h)
      scale <- 1 / sqrt(lyr$key_dim)
      scores <- h %*% t(h) * scale
      weights_attn <- t(apply(scores, 1, .activations$softmax))
      h <- weights_attn %*% h

    } else if (lyr$type == "ltc") {
      # Simplified LTC dynamics simulation
      h <- .ltc_forward(h, lyr)

    } else if (lyr$type == "moe") {
      # Sparse Mixture of Experts forward
      h <- .moe_forward(h, lyr)
    }

    if (store_activations) activations[[lyr$name]] <- h
  }

  result <- list(output = h)
  if (store_activations) result$activations <- activations
  result
}

# Internal: LTC forward dynamics
.ltc_forward <- function(h, layer) {
  n  <- ncol(h)
  u  <- layer$units
  dt <- 1.0 / layer$ode_unfolds

  # Initialize cell state
  x <- matrix(0, nrow(h), u)

  for (t in seq_len(layer$ode_unfolds)) {
    # Simplified Euler integration of LTC ODE:
    # dx/dt = -x/tau + sigma(W*input + b)
    tau   <- pmax(layer$epsilon, abs(sin(seq_len(u) * pi / u)) + 0.1)
    input <- h[, seq_len(min(n, u)), drop = FALSE]
    if (ncol(input) < u) {
      input <- cbind(input, matrix(0, nrow(h), u - ncol(input)))
    }
    sigma_val <- .activations$sigmoid(input)
    dx <- (-x / tau + sigma_val) * dt
    x  <- x + dx
  }
  x
}

# Internal: MoE forward pass
.moe_forward <- function(h, layer) {
  n_in   <- ncol(h)
  n_exp  <- layer$num_experts
  top_k  <- layer$top_k
  eu     <- layer$expert_units

  # Gate: softmax over expert logits
  gate_w     <- matrix(rnorm(n_in * n_exp, 0, 0.1), n_in, n_exp)
  gate_logits <- h %*% gate_w
  gate_probs  <- t(apply(gate_logits, 1, .activations$softmax))

  # Top-K sparse selection
  out <- matrix(0, nrow(h), eu)
  for (i in seq_len(nrow(h))) {
    top_idx  <- order(gate_probs[i, ], decreasing = TRUE)[seq_len(top_k)]
    for (k in top_idx) {
      exp_w  <- matrix(rnorm(n_in * eu, 0, 0.1), n_in, eu)
      expert <- nvz_activate(h[i, ] %*% exp_w, "gelu")
      out[i, ] <- out[i, ] + gate_probs[i, k] * expert
    }
  }
  out
}



################################################################################
##  SECTION B:  neuroviz.explain  ─  Explainability Engine (XAI)             ##
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
# B1. SHAP-style Feature Importance (Kernel SHAP approximation)
# ──────────────────────────────────────────────────────────────────────────────

#' Compute Kernel SHAP values for a network prediction
#'
#' Approximates SHAP values using the Shapley kernel weighting method.
#' This is a model-agnostic approach — works with any nvz_network.
#'
#' @param net         Compiled nvz_network
#' @param X           Single sample (numeric vector) — the instance to explain
#' @param X_background Matrix of background samples (reference distribution)
#' @param n_samples   Integer. Number of coalition samples
#' @return Named numeric vector of SHAP values
nvz_shap <- function(net, X, X_background, n_samples = 100) {
  stopifnot(net$compiled)
  p <- length(X)

  # Baseline: expected output over background
  bg_preds <- sapply(seq_len(nrow(X_background)), function(i) {
    r <- nvz_forward(net, matrix(X_background[i, ], nrow = 1))
    mean(r$output)
  })
  baseline <- mean(bg_preds)

  # Model prediction on the instance
  fx <- mean(nvz_forward(net, matrix(X, nrow = 1))$output)

  shap_vals <- numeric(p)

  # Monte Carlo Shapley estimation
  for (s in seq_len(n_samples)) {
    # Random feature subset (coalition)
    coalition <- sample(c(TRUE, FALSE), p, replace = TRUE)
    n_coal    <- sum(coalition)

    # Shapley kernel weight
    w <- if (n_coal == 0 || n_coal == p) 1e6 else
      (p - 1) / (choose(p, n_coal) * n_coal * (p - n_coal))

    # Masked input: use X for coalition features, background sample otherwise
    bg_row <- X_background[sample(nrow(X_background), 1), ]
    x_masked      <- ifelse(coalition, X, bg_row)
    x_masked_excl <- x_masked

    for (j in which(coalition)) {
      x_with    <- x_masked
      x_without <- x_masked; x_without[j] <- bg_row[j]

      pred_with    <- mean(nvz_forward(net, matrix(x_with,    nrow = 1))$output)
      pred_without <- mean(nvz_forward(net, matrix(x_without, nrow = 1))$output)

      shap_vals[j] <- shap_vals[j] + w * (pred_with - pred_without)
    }
  }

  shap_vals <- shap_vals / n_samples
  names(shap_vals) <- paste0("X", seq_len(p))

  attr(shap_vals, "baseline") <- baseline
  attr(shap_vals, "fx")       <- fx
  class(shap_vals) <- c("nvz_shap", "numeric")
  shap_vals
}


# ──────────────────────────────────────────────────────────────────────────────
# B2. Gradient-based Attribution (Saliency Maps / Integrated Gradients)
# ──────────────────────────────────────────────────────────────────────────────

#' Compute Integrated Gradients attribution
#'
#' Axiomatic attribution method that satisfies completeness.
#' Reference: Sundararajan et al., 2017
#'
#' @param net      Compiled nvz_network
#' @param X        Numeric vector. Input instance
#' @param baseline Numeric vector. Reference baseline (default: zeros)
#' @param steps    Integer. Riemann sum steps
#' @return Named numeric vector of attribution scores
nvz_integrated_gradients <- function(net, X, baseline = NULL, steps = 50) {
  if (is.null(baseline)) baseline <- rep(0, length(X))

  # Interpolate between baseline and input
  alphas <- seq(0, 1, length.out = steps)

  grads <- sapply(alphas, function(alpha) {
    x_interp  <- baseline + alpha * (X - baseline)

    # Numerical gradient via finite differences
    epsilon <- 1e-4
    grad <- numeric(length(X))
    f0 <- mean(nvz_forward(net, matrix(x_interp, nrow = 1))$output)

    for (i in seq_along(X)) {
      x_plus <- x_interp; x_plus[i] <- x_plus[i] + epsilon
      fi     <- mean(nvz_forward(net, matrix(x_plus, nrow = 1))$output)
      grad[i] <- (fi - f0) / epsilon
    }
    grad
  })

  # Average gradients × (input - baseline)
  avg_grads  <- rowMeans(grads)
  attributions <- (X - baseline) * avg_grads
  names(attributions) <- paste0("X", seq_along(X))

  attr(attributions, "method") <- "integrated_gradients"
  class(attributions) <- c("nvz_attribution", "numeric")
  attributions
}


# ──────────────────────────────────────────────────────────────────────────────
# B3. Attention Rollout (for Transformer networks)
# ──────────────────────────────────────────────────────────────────────────────

#' Compute Attention Rollout for transformers
#' Propagates attention through all layers for global token importance
#'
#' @param attention_maps List of matrices (one per attention layer)
#' @return Rollout matrix
nvz_attention_rollout <- function(attention_maps) {
  n <- nrow(attention_maps[[1]])

  # Add identity (residual) to each attention map
  rollout <- diag(n)

  for (A in attention_maps) {
    A_hat <- A + diag(n)
    A_hat <- A_hat / rowSums(A_hat)  # normalize
    rollout <- rollout %*% A_hat
  }

  rollout
}


# ──────────────────────────────────────────────────────────────────────────────
# B4. Concept Activation Vectors (TCAV)
# ──────────────────────────────────────────────────────────────────────────────

#' Test with Concept Activation Vectors (TCAV)
#' Measures sensitivity of a layer to a human-defined concept
#'
#' @param net            nvz_network
#' @param concept_data   Matrix of concept-positive samples
#' @param random_data    Matrix of random (negative) samples
#' @param layer_name     Character. Target layer name
#' @return TCAV score (0 to 1)
nvz_tcav <- function(net, concept_data, random_data, layer_name) {

  # Get activations at target layer for both sets
  .get_layer_activation <- function(X_mat) {
    apply(X_mat, 1, function(row) {
      res <- nvz_forward(net, matrix(row, nrow = 1), store_activations = TRUE)
      if (!is.null(res$activations[[layer_name]])) {
        as.vector(res$activations[[layer_name]])
      } else {
        as.vector(res$output)
      }
    })
  }

  act_concept <- t(.get_layer_activation(concept_data))
  act_random  <- t(.get_layer_activation(random_data))

  # Train linear classifier (logistic regression) to get CAV direction
  labels <- c(rep(1, nrow(act_concept)), rep(0, nrow(act_random)))
  all_acts <- rbind(act_concept, act_random)

  # Simplified CAV via mean difference direction
  cav_direction <- colMeans(act_concept) - colMeans(act_random)
  cav_direction <- cav_direction / (norm(matrix(cav_direction), "F") + 1e-8)

  # TCAV score: fraction of inputs with positive directional derivative
  scores <- act_concept %*% cav_direction
  tcav_score <- mean(scores > 0)

  list(
    tcav_score    = tcav_score,
    cav_direction = cav_direction,
    layer         = layer_name,
    interpretation = ifelse(tcav_score > 0.5,
                           "Model relies on this concept (positive)",
                           "Concept not important for predictions")
  )
}



################################################################################
##  SECTION C:  neuroviz.spike  ─  Spiking Neural Network Engine             ##
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
# C1. Leaky Integrate-and-Fire (LIF) Neuron Model
# ──────────────────────────────────────────────────────────────────────────────

#' Create a Spiking Neural Network using LIF neurons
#'
#' Trend 2026: SNNs for energy-efficient neuromorphic hardware (Intel Loihi 3)
#'
#' @param n_input   Integer. Number of input neurons
#' @param n_hidden  Integer. Number of hidden LIF neurons
#' @param n_output  Integer. Number of output neurons
#' @param tau_mem   Float. Membrane time constant (ms)
#' @param threshold Float. Firing threshold voltage
nvz_snn <- function(n_input, n_hidden, n_output,
                    tau_mem   = 10.0,
                    threshold = 1.0,
                    tau_syn   = 5.0) {
  snn <- list(
    n_input    = n_input,
    n_hidden   = n_hidden,
    n_output   = n_output,
    tau_mem    = tau_mem,
    threshold  = threshold,
    tau_syn    = tau_syn,
    # Random synaptic weights (could be trained via STDP or surrogate gradient)
    W_in  = matrix(rnorm(n_input * n_hidden,  0, 1/sqrt(n_input)),  n_input,  n_hidden),
    W_out = matrix(rnorm(n_hidden * n_output, 0, 1/sqrt(n_hidden)), n_hidden, n_output),
    metadata = list(
      model    = "Leaky Integrate-and-Fire",
      encoding = "rate_coding",
      created  = Sys.time()
    )
  )
  class(snn) <- c("nvz_snn", "list")
  message(sprintf("[neuroviz.spike] SNN created: %d→%d→%d | tau_mem=%.1fms | threshold=%.2f",
                  n_input, n_hidden, n_output, tau_mem, threshold))
  snn
}


#' Run SNN simulation over time
#'
#' @param snn      nvz_snn object
#' @param input_spikes Matrix (T × n_input). Binary spike trains
#' @param dt       Float. Time step (ms)
#' @return List with spike trains, membrane potentials, firing rates
nvz_snn_simulate <- function(snn, input_spikes, dt = 1.0) {
  T      <- nrow(input_spikes)  # time steps
  n_h    <- snn$n_hidden
  n_o    <- snn$n_output

  # State variables
  V_h    <- matrix(0, T + 1, n_h)  # membrane potential hidden
  V_o    <- matrix(0, T + 1, n_o)  # membrane potential output
  S_h    <- matrix(0, T,     n_h)  # spike train hidden
  S_o    <- matrix(0, T,     n_o)  # spike train output
  I_syn  <- matrix(0, T + 1, n_h)  # synaptic current

  alpha_mem <- exp(-dt / snn$tau_mem)
  alpha_syn <- exp(-dt / snn$tau_syn)

  for (t in seq_len(T)) {
    # Synaptic current update (low-pass filter of input spikes)
    I_syn[t+1, ] <- alpha_syn * I_syn[t, ] + (input_spikes[t, ] %*% snn$W_in)

    # Membrane potential update (LIF dynamics)
    V_h[t+1, ] <- alpha_mem * V_h[t, ] * (1 - S_h[pmax(1,t), ]) + I_syn[t+1, ]

    # Spike generation (threshold crossing)
    S_h[t, ]   <- as.numeric(V_h[t+1, ] >= snn$threshold)

    # Output layer
    V_o[t+1, ] <- alpha_mem * V_o[t, ] * (1 - S_o[pmax(1,t), ]) +
                  (S_h[t, ] %*% snn$W_out)
    S_o[t, ]   <- as.numeric(V_o[t+1, ] >= snn$threshold)
  }

  # Compute firing rates (spikes per second)
  firing_rates_hidden <- colMeans(S_h) / (T * dt / 1000)
  firing_rates_output <- colMeans(S_o) / (T * dt / 1000)

  list(
    spike_train_hidden  = S_h,
    spike_train_output  = S_o,
    membrane_hidden     = V_h[-1, ],
    membrane_output     = V_o[-1, ],
    firing_rate_hidden  = firing_rates_hidden,
    firing_rate_output  = firing_rates_output,
    time_steps          = T,
    dt                  = dt
  )
}


#' Convert continuous input to Poisson spike trains (rate coding)
#' @param X       Numeric vector or matrix. Values in [0, 1]
#' @param T       Integer. Number of time steps
#' @param max_rate Float. Maximum firing rate (Hz)
nvz_rate_encode <- function(X, T = 100, max_rate = 100) {
  X <- pmax(0, pmin(1, X))  # clamp to [0,1]
  if (is.vector(X)) X <- matrix(X, nrow = 1)

  # For each neuron, generate Poisson spikes with rate proportional to input
  n <- ncol(X)
  spikes <- matrix(0, T, n)
  for (i in seq_len(n)) {
    rate   <- X[1, i] * max_rate
    p_fire <- rate / 1000  # probability of firing per ms
    spikes[, i] <- rbinom(T, 1, p_fire)
  }
  spikes
}


#' STDP (Spike-Timing Dependent Plasticity) weight update
#' Unsupervised Hebbian learning rule for SNNs
#'
#' @param W       Weight matrix to update
#' @param S_pre   Pre-synaptic spike train (T × n_pre)
#' @param S_post  Post-synaptic spike train (T × n_post)
#' @param A_plus  Float. LTP magnitude
#' @param A_minus Float. LTD magnitude
#' @param tau_plus  Float. LTP time window (ms)
#' @param tau_minus Float. LTD time window (ms)
nvz_stdp_update <- function(W, S_pre, S_post,
                             A_plus  = 0.01, A_minus = 0.012,
                             tau_plus = 20,  tau_minus = 20) {
  T <- nrow(S_pre)
  dW <- matrix(0, nrow(W), ncol(W))

  for (t in seq_len(T)) {
    post_fire <- which(S_post[t, ] > 0)
    pre_fire  <- which(S_pre[t, ]  > 0)

    if (length(post_fire) > 0 && t > 1) {
      # LTP: post fires after pre — strengthen connection
      for (j in post_fire) {
        for (t_pre in max(1, t-50):t) {
          dt_val <- t - t_pre
          dW[, j] <- dW[, j] + S_pre[t_pre, ] * A_plus * exp(-dt_val / tau_plus)
        }
      }
    }

    if (length(pre_fire) > 0 && t > 1) {
      # LTD: pre fires after post — weaken connection
      for (i in pre_fire) {
        for (t_post in max(1, t-50):t) {
          dt_val <- t - t_post
          dW[i, ] <- dW[i, ] - S_post[t_post, ] * A_minus * exp(-dt_val / tau_minus)
        }
      }
    }
  }

  # Clip weights to [0, 1] range
  pmin(pmax(W + dW, 0), 1)
}



################################################################################
##  SECTION D:  neuroviz.graph  ─  Graph Neural Network Engine               ##
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
# D1. Graph Object & Constructor
# ──────────────────────────────────────────────────────────────────────────────

#' Create a NeuroViz graph object
#'
#' @param adj_matrix  Matrix or sparse Matrix. Adjacency matrix (N×N)
#' @param node_features Matrix. Node feature matrix (N×F)
#' @param edge_weights  Numeric vector. Optional edge weights
#' @param labels        Integer vector. Node labels (for classification)
nvz_graph <- function(adj_matrix, node_features = NULL, edge_weights = NULL, labels = NULL) {
  N <- nrow(adj_matrix)

  if (is.null(node_features)) {
    node_features <- diag(N)  # identity features (one-hot)
    message("[neuroviz.graph] No node features — using identity matrix")
  }

  # Compute degree matrix and normalized Laplacian
  D      <- rowSums(adj_matrix)
  D_inv  <- ifelse(D > 0, 1 / sqrt(D), 0)
  D_mat  <- diag(D_inv)

  # Normalized adjacency: D^{-1/2} A D^{-1/2}
  A_norm <- D_mat %*% adj_matrix %*% D_mat

  # Add self-loops: A_hat = A + I
  A_hat  <- A_norm + diag(N)

  graph <- list(
    N             = N,
    adj           = adj_matrix,
    A_norm        = A_norm,
    A_hat         = A_hat,
    X             = node_features,
    edge_weights  = edge_weights,
    labels        = labels,
    degree        = D,
    F_dim         = ncol(node_features)
  )
  class(graph) <- c("nvz_graph", "list")
  message(sprintf("[neuroviz.graph] Graph: %d nodes | %d edges | %d features",
                  N, sum(adj_matrix > 0) / 2, ncol(node_features)))
  graph
}


# ──────────────────────────────────────────────────────────────────────────────
# D2. Graph Convolutional Network (GCN) — Kipf & Welling 2017
# ──────────────────────────────────────────────────────────────────────────────

#' GCN Layer Forward Pass
#' H_out = σ(A_hat × H × W)
nvz_gcn_layer <- function(A_hat, H, W, activation = "relu") {
  Z <- A_hat %*% H %*% W
  nvz_activate(Z, activation)
}

#' Build and run a full GCN model
#'
#' @param graph    nvz_graph object
#' @param hidden   Integer vector. Hidden layer dimensions
#' @param n_class  Integer. Number of output classes
#' @return List with embeddings and predictions
nvz_gcn <- function(graph, hidden = c(64, 32), n_class = 2) {
  dims <- c(graph$F_dim, hidden, n_class)
  H    <- graph$X

  layers <- list()
  for (i in seq_len(length(dims) - 1)) {
    fan_in  <- dims[i]
    fan_out <- dims[i + 1]
    limit   <- sqrt(6 / (fan_in + fan_out))
    W       <- matrix(runif(fan_in * fan_out, -limit, limit), fan_in, fan_out)
    act     <- if (i < length(dims) - 1) "relu" else "softmax"

    H <- if (act == "softmax") {
      t(apply(graph$A_hat %*% H %*% W, 1, .activations$softmax))
    } else {
      nvz_gcn_layer(graph$A_hat, H, W, act)
    }

    layers[[i]] <- list(W = W, output = H)
  }

  list(
    embeddings  = layers[[length(layers) - 1]]$output,
    predictions = H,
    pred_class  = apply(H, 1, which.max),
    layers      = layers,
    model_type  = "GCN"
  )
}


# ──────────────────────────────────────────────────────────────────────────────
# D3. Graph Attention Network (GAT) — Veličković et al., 2018
# ──────────────────────────────────────────────────────────────────────────────

#' GAT Layer with multi-head attention
#' @param adj    Adjacency matrix
#' @param H      Node feature matrix
#' @param W      Weight matrix
#' @param a      Attention vector (2F' × 1)
#' @param n_heads Number of attention heads
nvz_gat_layer <- function(adj, H, W, n_heads = 4, activation = "elu") {
  N  <- nrow(H)
  F_in <- ncol(H)
  F_out <- ncol(W)

  # For simplicity, implement single-head attention
  # Multi-head: concatenate n_heads single heads
  elu <- function(x, alpha = 1) ifelse(x >= 0, x, alpha * (exp(x) - 1))

  all_heads <- lapply(seq_len(n_heads), function(h) {
    # Transform features
    Wh    <- H %*% W  # N × F'

    # Compute attention coefficients
    # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    a_vec <- rnorm(2 * F_out)

    e_mat <- matrix(0, N, N)
    for (i in seq_len(N)) {
      for (j in which(adj[i, ] > 0)) {
        e_mat[i, j] <- sum(a_vec * c(Wh[i, ], Wh[j, ])) # dot product
        e_mat[i, j] <- pmax(-0.2 * e_mat[i, j], e_mat[i, j])  # LeakyReLU
      }
    }

    # Masked softmax attention (only over neighbors)
    alpha_mat <- matrix(-Inf, N, N)
    for (i in seq_len(N)) {
      nbrs <- which(adj[i, ] > 0)
      if (length(nbrs) > 0) {
        e_i <- e_mat[i, nbrs]
        e_i_soft <- exp(e_i - max(e_i)) / sum(exp(e_i - max(e_i)))
        alpha_mat[i, nbrs] <- e_i_soft
      }
    }
    alpha_mat[is.infinite(alpha_mat)] <- 0

    # Aggregate
    elu(alpha_mat %*% Wh)
  })

  # Concatenate heads (or average for output layer)
  do.call(cbind, all_heads)
}


# ──────────────────────────────────────────────────────────────────────────────
# D4. Graph-level Readout & Pooling
# ──────────────────────────────────────────────────────────────────────────────

#' Graph-level readout functions
#' @param H    Node embedding matrix (N × F)
#' @param method Character. "mean", "sum", "max", "attention"
nvz_graph_readout <- function(H, method = "mean") {
  switch(method,
    mean      = colMeans(H),
    sum       = colSums(H),
    max       = apply(H, 2, max),
    attention = {
      # Attention-weighted sum
      scores <- apply(H, 1, function(h) sum(h^2))  # L2 norm as attention score
      weights <- .activations$softmax(scores)
      colSums(weights * H)
    },
    colMeans(H)
  )
}

#' Generate a random Erdős–Rényi graph for testing
#' @param n  Integer. Number of nodes
#' @param p  Float. Edge probability
nvz_random_graph <- function(n = 20, p = 0.2, n_features = 8, n_classes = 3) {
  adj <- matrix(0, n, n)
  for (i in seq_len(n - 1)) {
    for (j in (i+1):n) {
      if (runif(1) < p) {
        adj[i, j] <- 1
        adj[j, i] <- 1
      }
    }
  }
  X      <- matrix(rnorm(n * n_features), n, n_features)
  labels <- sample(seq_len(n_classes), n, replace = TRUE)
  nvz_graph(adj, X, labels = labels)
}



################################################################################
##  SECTION E:  neuroviz.vis  ─  Visualization Suite                         ##
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
# E1. Network Architecture Diagram
# ──────────────────────────────────────────────────────────────────────────────

#' Plot neural network architecture as a layered graph
#' @param net  nvz_network object
nvz_plot_architecture <- function(net) {
  if (!net$compiled) stop("Compile the network first with nvz_compile()")

  n_layers <- length(net$layers) + 2  # +input +output

  # Build node data
  nodes <- data.frame()
  edges <- data.frame()

  layer_names <- c("Input", sapply(net$layers, `[[`, "name"), "Output")
  layer_types <- c("input", sapply(net$layers, `[[`, "type"), "output")
  layer_units <- c(
    net$input_dim,
    sapply(net$layers, function(l) l$units %||% 0),
    tail(sapply(net$layers, function(l) l$units %||% 0), 1)
  )

  # Neurons per layer (cap at 8 for display)
  display_units <- pmin(8, pmax(1, round(layer_units / max(layer_units, 1) * 8)))

  node_list <- list()
  edge_list <- list()
  node_id <- 0

  for (li in seq_along(layer_names)) {
    n_vis <- display_units[li]
    for (ni in seq_len(n_vis)) {
      node_id <- node_id + 1
      node_list[[node_id]] <- data.frame(
        id    = node_id,
        layer = li,
        pos_y = ni - (n_vis + 1) / 2,
        type  = layer_types[li],
        lname = layer_names[li]
      )
    }
  }

  nodes <- do.call(rbind, node_list)
  # Edges between consecutive layers
  for (li in seq_len(length(layer_names) - 1)) {
    from_nodes <- nodes$id[nodes$layer == li]
    to_nodes   <- nodes$id[nodes$layer == li + 1]
    for (f in from_nodes) {
      for (t in to_nodes) {
        edge_list[[length(edge_list) + 1]] <- data.frame(
          x    = li,     y    = nodes$pos_y[nodes$id == f],
          xend = li + 1, yend = nodes$pos_y[nodes$id == t]
        )
      }
    }
  }
  edges <- do.call(rbind, edge_list)

  type_colors <- c(
    input     = "#4ECDC4",
    dense     = "#45B7D1",
    conv      = "#96CEB4",
    attention = "#FFEAA7",
    ltc       = "#DDA0DD",
    moe       = "#F0A500",
    output    = "#FF6B6B"
  )

  p <- ggplot() +
    # Edges
    geom_segment(data = edges, aes(x=x, y=y, xend=xend, yend=yend),
                 alpha = 0.08, color = "#AAAAAA", linewidth = 0.3) +
    # Nodes
    geom_point(data = nodes,
               aes(x = layer, y = pos_y, color = type, fill = type),
               size = 7, shape = 21, stroke = 1.5) +
    scale_color_manual(values = type_colors, name = "Layer Type") +
    scale_fill_manual(values = type_colors, guide = "none") +
    # Layer labels
    geom_text(data = data.frame(
      x = seq_along(layer_names),
      y = -max(display_units)/2 - 1.2,
      label = layer_names),
      aes(x=x, y=y, label=label),
      size = 2.5, color = "#CCCCCC", angle = 30, hjust = 1) +
    theme_void() +
    theme(
      plot.background  = element_rect(fill = "#1A1A2E", color = NA),
      panel.background = element_rect(fill = "#1A1A2E", color = NA),
      legend.position  = "bottom",
      legend.text      = element_text(color = "#EEEEEE", size = 9),
      legend.title     = element_text(color = "#EEEEEE", size = 10),
      plot.title       = element_text(color = "#E0E0E0", size = 14,
                                      face = "bold", hjust = 0.5),
      plot.subtitle    = element_text(color = "#AAAAAA", size = 9, hjust = 0.5)
    ) +
    labs(
      title    = paste0("[ ", net$name, " ]  Architecture"),
      subtitle = sprintf("Total Parameters: %s  |  %d Layers",
                         formatC(net$total_params, format = "d", big.mark = ","),
                         length(net$layers))
    )
  print(p)
  invisible(p)
}


# ──────────────────────────────────────────────────────────────────────────────
# E2. SHAP Beeswarm / Waterfall Plot
# ──────────────────────────────────────────────────────────────────────────────

#' Plot SHAP values as a waterfall chart
#' @param shap_vals  nvz_shap object
#' @param feature_names Character vector of feature names
nvz_plot_shap <- function(shap_vals, feature_names = NULL) {
  n <- length(shap_vals)
  if (is.null(feature_names)) feature_names <- names(shap_vals)
  if (is.null(feature_names)) feature_names <- paste0("Feature ", seq_len(n))

  baseline <- attr(shap_vals, "baseline") %||% 0
  fx       <- attr(shap_vals, "fx") %||% (baseline + sum(shap_vals))

  # Sort by absolute value
  ord <- order(abs(shap_vals), decreasing = TRUE)
  sv  <- shap_vals[ord]
  fn  <- feature_names[ord]

  df <- data.frame(
    feature = factor(fn, levels = rev(fn)),
    shap    = sv,
    color   = ifelse(sv >= 0, "positive", "negative")
  )

  ggplot(df, aes(x = shap, y = feature, fill = color)) +
    geom_col(width = 0.65, color = NA) +
    geom_vline(xintercept = 0, color = "white", linewidth = 0.8) +
    scale_fill_manual(
      values  = c(positive = "#FF6B6B", negative = "#4ECDC4"),
      name    = "Effect",
      labels  = c("Increases prediction", "Decreases prediction")
    ) +
    geom_text(aes(
      label = sprintf("%+.4f", shap),
      x     = shap + ifelse(shap >= 0, 0.001, -0.001),
      hjust = ifelse(shap >= 0, 0, 1)
    ), size = 3, color = "white") +
    theme_minimal() +
    theme(
      plot.background  = element_rect(fill = "#16213E", color = NA),
      panel.background = element_rect(fill = "#16213E", color = NA),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(color = "#2A2A4A"),
      panel.grid.minor   = element_blank(),
      axis.text    = element_text(color = "#CCCCCC"),
      axis.title   = element_text(color = "#AAAAAA"),
      plot.title   = element_text(color = "white", size = 14, face = "bold"),
      plot.subtitle = element_text(color = "#AAAAAA", size = 10),
      legend.text  = element_text(color = "#CCCCCC"),
      legend.title = element_text(color = "#AAAAAA")
    ) +
    labs(
      title    = "SHAP Feature Attribution",
      subtitle = sprintf("Baseline: %.4f  →  Prediction: %.4f  |  ΔSHAP: %+.4f",
                         baseline, fx, fx - baseline),
      x = "SHAP Value",
      y = ""
    )
}


# ──────────────────────────────────────────────────────────────────────────────
# E3. Spike Raster Plot
# ──────────────────────────────────────────────────────────────────────────────

#' Plot spike raster from SNN simulation
#' @param sim_result  Output from nvz_snn_simulate()
#' @param layer       "hidden" or "output"
nvz_plot_raster <- function(sim_result, layer = "hidden") {
  S <- if (layer == "hidden") sim_result$spike_train_hidden else sim_result$spike_train_output
  T <- nrow(S)
  N <- ncol(S)

  # Convert to long format
  df <- which(S > 0, arr.ind = TRUE)
  df <- data.frame(time = df[, 1], neuron = df[, 2])

  if (nrow(df) == 0) {
    message("[neuroviz.spike] No spikes detected — try increasing input rate or lowering threshold")
    return(invisible(NULL))
  }

  # Firing rate over time (binned)
  bin_size <- max(1, T %/% 20)
  rate_df  <- data.frame(
    time  = seq(1, T, by = bin_size),
    rate  = sapply(seq(1, T, by = bin_size), function(t) {
      mean(S[t:min(t + bin_size - 1, T), ]) * 1000
    })
  )

  p_raster <- ggplot(df, aes(x = time, y = neuron)) +
    geom_point(size = 0.6, color = "#00F5D4", alpha = 0.7, shape = 16) +
    theme_minimal() +
    theme(
      plot.background  = element_rect(fill = "#0D0D1A", color = NA),
      panel.background = element_rect(fill = "#0D0D1A", color = NA),
      panel.grid       = element_line(color = "#1A1A2A"),
      axis.text        = element_text(color = "#AAAAAA", size = 8),
      axis.title       = element_text(color = "#CCCCCC", size = 10),
      plot.title       = element_text(color = "white", size = 13, face = "bold"),
      plot.subtitle    = element_text(color = "#888888", size = 9)
    ) +
    labs(
      title    = paste("Spike Raster —", toupper(layer), "layer"),
      subtitle = sprintf("%d neurons × %d ms  |  Mean firing rate: %.1f Hz",
                         N, T, mean(sim_result[[paste0("firing_rate_", layer)]])),
      x = "Time (ms)", y = "Neuron ID"
    )

  print(p_raster)
  invisible(p_raster)
}


# ──────────────────────────────────────────────────────────────────────────────
# E4. Graph Visualization
# ──────────────────────────────────────────────────────────────────────────────

#' Plot a graph with GNN prediction overlay
#' @param graph   nvz_graph object
#' @param gcn_out Output from nvz_gcn()
nvz_plot_graph <- function(graph, gcn_out = NULL) {
  ig <- igraph::graph_from_adjacency_matrix(graph$adj, mode = "undirected")

  # Node attributes
  V(ig)$degree <- graph$degree

  if (!is.null(gcn_out)) {
    V(ig)$pred_class <- factor(gcn_out$pred_class)
  } else if (!is.null(graph$labels)) {
    V(ig)$pred_class <- factor(graph$labels)
  } else {
    V(ig)$pred_class <- factor(rep(1, graph$N))
  }

  # Use ggraph for layout
  set.seed(42)
  p <- ggraph(ig, layout = "fr") +
    geom_edge_link(alpha = 0.25, color = "#AAAAAA", linewidth = 0.4) +
    geom_node_point(aes(color = pred_class, size = degree + 1)) +
    geom_node_text(aes(label = seq_len(graph$N)), size = 2.5,
                   color = "white", fontface = "bold", repel = FALSE) +
    scale_color_brewer(palette = "Set2", name = if (!is.null(gcn_out)) "GCN Class" else "Label") +
    scale_size_continuous(range = c(3, 10), guide = "none") +
    theme_graph(background = "#12121F") +
    theme(
      plot.title   = element_text(color = "white", size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(color = "#888888", size = 9, hjust = 0.5),
      legend.text  = element_text(color = "#CCCCCC"),
      legend.title = element_text(color = "#AAAAAA")
    ) +
    labs(
      title    = "Graph Neural Network — Node Classification",
      subtitle = sprintf("%d nodes | %d edges | GCN: %d→%s",
                         graph$N,
                         sum(graph$adj > 0) / 2,
                         graph$F_dim,
                         "64→32→class")
    )

  print(p)
  invisible(p)
}


# ──────────────────────────────────────────────────────────────────────────────
# E5. Training Loss & Metrics Dashboard
# ──────────────────────────────────────────────────────────────────────────────

#' Simulate training history and plot learning curves
#' @param n_epochs Integer. Number of epochs to simulate
#' @param model_names Character vector. Names of models to compare
nvz_plot_training <- function(n_epochs = 50, model_names = c("GCN", "GAT", "MoE-LTC")) {
  set.seed(123)

  # Simulate realistic loss curves
  epochs <- seq_len(n_epochs)
  histories <- lapply(model_names, function(mn) {
    noise   <- rnorm(n_epochs, 0, 0.015)
    speed   <- runif(1, 0.05, 0.15)
    floor   <- runif(1, 0.05, 0.20)
    train_l <- floor + (1.2 - floor) * exp(-speed * epochs) + noise
    val_l   <- train_l + abs(rnorm(n_epochs, 0, 0.03)) + 0.02

    data.frame(
      epoch  = epochs,
      train  = pmax(0.01, train_l),
      val    = pmax(0.01, val_l),
      model  = mn
    )
  })

  df <- do.call(rbind, histories)
  df_long <- tidyr::pivot_longer(df, c("train", "val"),
                                  names_to = "split", values_to = "loss")
  df_long$model_split <- paste0(df_long$model, " (", df_long$split, ")")

  ggplot(df_long, aes(x = epoch, y = loss, color = model, linetype = split)) +
    geom_line(linewidth = 0.9, alpha = 0.9) +
    geom_smooth(se = FALSE, linewidth = 0.3, linetype = "dotted", span = 0.3) +
    scale_color_manual(values = c("#FF6B6B", "#4ECDC4", "#FFD93D"), name = "Model") +
    scale_linetype_manual(values = c(train = "solid", val = "dashed"), name = "Split") +
    theme_minimal() +
    theme(
      plot.background  = element_rect(fill = "#1A1A2E", color = NA),
      panel.background = element_rect(fill = "#1A1A2E", color = NA),
      panel.grid.major = element_line(color = "#2A2A3E"),
      panel.grid.minor = element_blank(),
      axis.text  = element_text(color = "#AAAAAA"),
      axis.title = element_text(color = "#CCCCCC"),
      plot.title = element_text(color = "white", size = 14, face = "bold"),
      plot.subtitle = element_text(color = "#888888", size = 9),
      legend.background = element_rect(fill = "#1A1A2E"),
      legend.text  = element_text(color = "#CCCCCC"),
      legend.title = element_text(color = "#AAAAAA")
    ) +
    labs(
      title    = "Model Training Comparison — 2026 Architecture Trends",
      subtitle = "GCN vs Graph Attention vs Mixture-of-Experts with LTC layers",
      x = "Epoch", y = "Loss"
    )
}



################################################################################
##  SECTION F:  Research Experiments — Run These!                             ##
################################################################################

message("\n", strrep("═", 65))
message("  NeuroViz 2026 — Research Experiments Ready")
message(strrep("═", 65))

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Build a Hybrid Transformer + MoE + LTC Network
# ══════════════════════════════════════════════════════════════════

experiment_1_hybrid_network <- function() {
  cat("\n[EXP 1] Building Hybrid Transformer-MoE-LTC Network...\n")

  net <- nvz_network("HybridNet-2026", task = "classification") %+%
    nvz_layer_dense(128, activation = "gelu", name = "embed") %+%
    nvz_layer_attention(num_heads = 4, key_dim = 32, name = "attn1") %+%
    nvz_layer_moe(num_experts = 8, expert_units = 64, top_k = 2, name = "moe1") %+%
    nvz_layer_ltc(units = 32, ode_unfolds = 8, name = "ltc1") %+%
    nvz_layer_dense(10, activation = "softmax", name = "classify")

  net <- nvz_compile(net, input_dim = 64, optimizer = "adam",
                     loss = "cross_entropy", learning_rate = 3e-4)

  print(net)

  # Forward pass
  X    <- matrix(rnorm(5 * 64), 5, 64)  # batch of 5 samples
  out  <- nvz_forward(net, X, store_activations = TRUE)

  cat(sprintf("\n  Input shape:  (%d, %d)\n", 5, 64))
  cat(sprintf("  Output shape: (%d, %d)\n", nrow(out$output), ncol(out$output)))
  cat(sprintf("  Stored activation layers: %d\n\n", length(out$activations)))

  # Architecture plot
  nvz_plot_architecture(net)

  list(network = net, forward_result = out)
}


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: XAI — SHAP + Integrated Gradients
# ══════════════════════════════════════════════════════════════════

experiment_2_xai <- function() {
  cat("\n[EXP 2] Running XAI Experiments (SHAP + Integrated Gradients)...\n")

  # Build simple explainable network
  net <- nvz_network("XAI-Net", task = "regression") %+%
    nvz_layer_dense(32, activation = "relu") %+%
    nvz_layer_dense(16, activation = "swish") %+%
    nvz_layer_dense(1,  activation = "linear")

  net <- nvz_compile(net, input_dim = 10, optimizer = "adam", loss = "mse")

  # Generate data
  set.seed(42)
  X_bg      <- matrix(rnorm(50 * 10), 50, 10)
  X_explain <- rnorm(10)  # single instance to explain
  feature_names <- paste0("Feature_", LETTERS[1:10])

  cat("  Computing SHAP values (Monte Carlo approximation)...\n")
  shap_vals <- nvz_shap(net, X_explain, X_bg, n_samples = 60)

  cat(sprintf("  Baseline: %.4f  |  Prediction: %.4f\n",
              attr(shap_vals, "baseline"), attr(shap_vals, "fx")))
  cat("  Top 3 SHAP features:\n")
  top3 <- sort(abs(shap_vals), decreasing = TRUE)[1:3]
  for (nm in names(top3)) {
    cat(sprintf("    %s: SHAP = %+.4f\n", nm, shap_vals[nm]))
  }

  # Plot SHAP
  p_shap <- nvz_plot_shap(shap_vals, feature_names)
  print(p_shap)

  # Integrated Gradients
  cat("\n  Computing Integrated Gradients...\n")
  ig_attr <- nvz_integrated_gradients(net, X_explain, steps = 30)
  cat("  Top attributions (Integrated Gradients):\n")
  print(sort(ig_attr, decreasing = TRUE)[1:5])

  list(shap = shap_vals, ig = ig_attr)
}


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Spiking Neural Network Simulation
# ══════════════════════════════════════════════════════════════════

experiment_3_snn <- function() {
  cat("\n[EXP 3] Spiking Neural Network Simulation (Neuromorphic AI)...\n")

  # Create SNN
  snn <- nvz_snn(
    n_input    = 20,
    n_hidden   = 50,
    n_output   = 5,
    tau_mem    = 15.0,
    threshold  = 0.8,
    tau_syn    = 8.0
  )

  # Generate input — encode image-like pattern as spike trains
  set.seed(99)
  X_input  <- runif(20, 0.1, 0.9)
  spikes   <- nvz_rate_encode(X_input, T = 200, max_rate = 80)

  cat(sprintf("  Input neurons: %d | Simulation: %d ms | Input rate: 80Hz\n",
              20, 200))

  # Run simulation
  sim <- nvz_snn_simulate(snn, spikes, dt = 1.0)

  cat(sprintf("  Hidden layer mean firing rate: %.2f Hz\n",
              mean(sim$firing_rate_hidden)))
  cat(sprintf("  Output layer mean firing rate: %.2f Hz\n",
              mean(sim$firing_rate_output)))
  cat(sprintf("  Total hidden spikes: %d / %d possible\n",
              sum(sim$spike_train_hidden),
              nrow(sim$spike_train_hidden) * ncol(sim$spike_train_hidden)))

  # Plot raster
  nvz_plot_raster(sim, "hidden")

  # STDP learning (research hook)
  cat("\n  Applying STDP weight update...\n")
  W_new <- nvz_stdp_update(snn$W_in, spikes, sim$spike_train_hidden,
                            A_plus = 0.005, A_minus = 0.006)
  delta_W <- mean(abs(W_new - snn$W_in))
  cat(sprintf("  Mean |ΔW| after STDP: %.6f\n", delta_W))

  list(snn = snn, simulation = sim, W_updated = W_new)
}


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Graph Neural Network on Molecular-like Graph
# ══════════════════════════════════════════════════════════════════

experiment_4_gnn <- function() {
  cat("\n[EXP 4] Graph Neural Network — Node Classification...\n")

  set.seed(7)
  g <- nvz_random_graph(n = 30, p = 0.15, n_features = 16, n_classes = 4)

  cat(sprintf("  Graph stats: %d nodes | %d edges | %d features | %d classes\n",
              g$N,
              sum(g$adj > 0) / 2,
              g$F_dim,
              length(unique(g$labels))))

  # Run GCN
  cat("  Running GCN forward pass (2 hidden layers: 64→32→4)...\n")
  gcn_out <- nvz_gcn(g, hidden = c(64, 32), n_class = 4)

  # Accuracy (against random labels — for structure test)
  acc <- mean(gcn_out$pred_class == g$labels)
  cat(sprintf("  GCN accuracy (vs random labels): %.1f%%\n", acc * 100))
  cat(sprintf("  Class distribution: %s\n",
              paste(table(gcn_out$pred_class), collapse = " | ")))

  # GAT layer test
  cat("\n  Running GAT layer (4 attention heads)...\n")
  W_gat   <- matrix(rnorm(g$F_dim * 16, 0, 0.1), g$F_dim, 16)
  H_gat   <- nvz_gat_layer(g$adj, g$X, W_gat, n_heads = 4)
  cat(sprintf("  GAT output shape: (%d, %d)\n", nrow(H_gat), ncol(H_gat)))

  # Graph-level embedding
  embedding <- nvz_graph_readout(gcn_out$embeddings, method = "attention")
  cat(sprintf("  Graph embedding (attention readout) dim: %d\n", length(embedding)))

  # Plot
  nvz_plot_graph(g, gcn_out)

  list(graph = g, gcn = gcn_out, embedding = embedding)
}


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Multi-Architecture Training Comparison
# ══════════════════════════════════════════════════════════════════

experiment_5_comparison <- function() {
  cat("\n[EXP 5] 2026 Architecture Comparison — Training Curves...\n")
  cat("  Comparing: GCN vs GAT vs Mixture-of-Experts+LTC\n\n")
  p <- nvz_plot_training(n_epochs = 80,
                         model_names = c("GCN", "GAT (4-head)", "MoE-LTC (2026)"))
  print(p)
  cat("  Training comparison complete.\n")
  invisible(p)
}


# ══════════════════════════════════════════════════════════════════
# MASTER RUN — Execute all experiments
# ══════════════════════════════════════════════════════════════════

nvz_run_all <- function() {
  cat(paste0(
    "\n", strrep("─", 65), "\n",
    "  NeuroViz 2026 — Running All Research Experiments\n",
    strrep("─", 65), "\n\n"
  ))

  results <- list()

  tryCatch({
    results$exp1 <- experiment_1_hybrid_network()
    cat("[EXP 1] ✓ Hybrid Network — COMPLETE\n")
  }, error = function(e) message("[EXP 1] Error: ", e$message))

  tryCatch({
    results$exp2 <- experiment_2_xai()
    cat("[EXP 2] ✓ XAI (SHAP + IG) — COMPLETE\n")
  }, error = function(e) message("[EXP 2] Error: ", e$message))

  tryCatch({
    results$exp3 <- experiment_3_snn()
    cat("[EXP 3] ✓ Spiking Neural Network — COMPLETE\n")
  }, error = function(e) message("[EXP 3] Error: ", e$message))

  tryCatch({
    results$exp4 <- experiment_4_gnn()
    cat("[EXP 4] ✓ Graph Neural Network — COMPLETE\n")
  }, error = function(e) message("[EXP 4] Error: ", e$message))

  tryCatch({
    results$exp5 <- experiment_5_comparison()
    cat("[EXP 5] ✓ Architecture Comparison — COMPLETE\n")
  }, error = function(e) message("[EXP 5] Error: ", e$message))

  cat(paste0(
    "\n", strrep("─", 65), "\n",
    "  All experiments complete. Results in 'nvz_results'.\n",
    "  Explore: results$exp1, $exp2, $exp3, $exp4, $exp5\n",
    strrep("─", 65), "\n"
  ))

  invisible(results)
}


# ══════════════════════════════════════════════════════════════════
#  RESEARCH EXTENSIONS — Ideas for your own work
# ══════════════════════════════════════════════════════════════════
#
#  1. SURROGATE GRADIENT TRAINING for SNNs
#     → Implement straight-through estimator for SNN backprop
#     → Compare STDP vs surrogate gradient on MNIST spike encoding
#
#  2. SPARSE ATTENTION with TOPK MASKING
#     → Modify nvz_layer_attention to use top-k sparse attention
#     → Benchmark memory/speed vs full attention
#
#  3. NEURAL ARCHITECTURE SEARCH (NAS)
#     → Use nvz_compile() total_params as fitness function
#     → Evolutionary search over layer types and sizes
#
#  4. TEMPORAL GRAPH NEURAL NETWORKS
#     → Extend nvz_graph to include time-varying adjacency
#     → Combine LTC layers with GCN for dynamic graphs
#
#  5. FEDERATED NEUROVIZ
#     → Simulate multiple nvz_network clients
#     → Implement FedAvg weight averaging
#
#  6. ENERGY ESTIMATION MODULE
#     → Estimate power consumption of SNN vs ANN
#     → Model Intel Loihi 3 / IBM NorthPole efficiency metrics
#
#  7. PHYSICS-INFORMED LOSS
#     → Add constraint terms to loss for physical systems
#     → Use LTC dynamics for ODE-based physics simulation
#
#  8. CONTINUAL LEARNING (CATASTROPHIC FORGETTING)
#     → Implement Elastic Weight Consolidation (EWC)
#     → Test on sequential task learning with nvz_network
#
# ══════════════════════════════════════════════════════════════════

cat("\n")
cat("┌─────────────────────────────────────────────────────────────┐\n")
cat("│  NeuroViz 2026 loaded successfully!                         │\n")
cat("│                                                             │\n")
cat("│  Quick Start:                                               │\n")
cat("│  > nvz_results <- nvz_run_all()   # Run all experiments     │\n")
cat("│  > experiment_1_hybrid_network()  # Hybrid Transformer+MoE  │\n")
cat("│  > experiment_2_xai()             # SHAP + Integ. Gradients │\n")
cat("│  > experiment_3_snn()             # Spiking Neural Nets     │\n")
cat("│  > experiment_4_gnn()             # Graph Neural Nets       │\n")
cat("│  > experiment_5_comparison()      # Architecture comparison │\n")
cat("└─────────────────────────────────────────────────────────────┘\n\n")
