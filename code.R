# ==============================================================================
# Simulation Study: Extreme Quantile Estimation via Functional Single Index Models
# Reference: "Extreme Quantile Estimation via Functional Single Index Models"
#            by Yeonjoo Park and Yaeji Lim
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(fda)          # Functional data analysis & B-splines
  library(fields)       # Quantile smoothing spline (qsreg)
  library(caTools)      # Numerical integration (trapz)
  library(splines)      # B-spline basis functions (bs)
  library(fdapace)      # Functional Principal Component Analysis (FPCA)
  library(quantreg)     # Linear quantile regression (rq)
  library(extRemes)     # Extreme Value Theory (fevd)
  library(evmix)        # Generalized Pareto Distribution (rgpd, qgpd)
  library(ismev)        # GPD modeling and initial parameter fitting
  library(dplyr)        # Data manipulation
  library(purrr)        # Functional programming tools (map_dfr)
  library(tibble)       # Tibble data structures
  library(Rcpp)         # Rcpp integration
  library(RcppArmadillo)# High-performance linear algebra
})

# ------------------------------------------------------------------------------
# 2. Source Helper Functions & C++ Implementations
# ------------------------------------------------------------------------------
# Resolve directory of current script for portable execution
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    context_path <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(context_path)) return(dirname(context_path))
  }
  return(getwd())
}

script_dir <- get_script_dir()
qsreg_path <- file.path(script_dir, "qsreg.R")
cpp_path   <- file.path(script_dir, "weighted_llh.cpp")

# Source qsreg.R (with fallback search in parent directory)
if (file.exists(qsreg_path)) {
  source(qsreg_path)
} else if (file.exists(file.path(script_dir, "..", "qsreg.R"))) {
  source(file.path(script_dir, "..", "qsreg.R"))
} else {
  stop("qsreg.R not found. Please ensure qsreg.R is in the same directory.")
}

# Compile weighted GPD log-likelihood (C++ via RcppArmadillo)
if (file.exists(cpp_path)) {
  sourceCpp(cpp_path)
} else {
  # Inline compilation fallback if weighted_llh.cpp is not located on disk
  sourceCpp(code = '
  #include <RcppArmadillo.h>
  // [[Rcpp::depends(RcppArmadillo)]]

  // [[Rcpp::export(rng = FALSE)]]
  double weighted_llh(const arma::vec & par,
                      const arma::vec & data,
                      const arma::vec & weights,
                      double lambda,
                      double xi_prior) {
    if (par[0] <= 0) {
      return 1e6;
    } else {
      if (arma::min(1 + data / (par[0] / par[1])) <= 0) {
        return 1e6;
      } else {
        return sum(weights %
                   (log(par[0]) + log(1 + data / (par[0] / par[1]))
                      / (1 / (1 + 1 / par[1])))) / data.size()
               + lambda * pow(par[1] - xi_prior, 2);
      }
    }
  }
  ')
}

# Wrapper for weighted GPD log-likelihood optimization (Proposed-LL)
optim_wrapper <- function(wi_x, init_pars, Z, lambda, xi_prior) {
  res <- stats::optim(
    par = init_pars,
    fn = weighted_llh,
    data = Z,
    weights = wi_x,
    lambda = lambda,
    xi_prior = xi_prior
  )$par
  names(res) <- c("sigma", "xi")
  return(res)
}

# ------------------------------------------------------------------------------
# 3. Simulation Configuration
# ------------------------------------------------------------------------------
# Model link functions (Section 3.1):
#   - "M1": Linear link, f(u) = 5 * u
#   - "M2": Nonlinear link, f(u) = 0.66 * exp(u^2)
#   - "M3": Linear + Nonlinear link, f(u) = 0.66 * sin(u) + 0.2 * u
if (!exists("model", inherits = FALSE)) model <- "M2"

# Intermediate quantile level (tau_n)
if (!exists("tau_interm", inherits = FALSE)) tau_interm <- 0.8

# Number of Monte Carlo simulation repetitions (100 in paper)
if (!exists("n_sim", inherits = FALSE)) n_sim <- 100

# Directory to save simulation output .RData files
if (!exists("out_dir", inherits = FALSE)) out_dir <- file.path(script_dir, "results")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Simulation scenarios evaluated in Section 3 of the paper:
#   - Distributions:
#       * "t2": Student-t with 2 degrees of freedom
#       * "pareto25": Constant GPD with gamma = 0.25, sigma = 1
#       * "pareto_var": Predictor-dependent GPD with varying gamma(X_i), sigma = 1
#   - Sample sizes: n in {300, 500}
#   - Extreme quantile levels: tau_q in {0.99, 0.995} (optional: 0.90)
if (!exists("scenarios", inherits = FALSE)) {
  scenarios <- expand.grid(
    tau_q = c(0.99, 0.995),
    n = c(300, 500),
    dist = c("t2", "pareto25", "pareto_var"),
    stringsAsFactors = FALSE
  )
}

# ------------------------------------------------------------------------------
# 4. Simulation Execution
# ------------------------------------------------------------------------------
for (case_idx in seq_len(nrow(scenarios))) {
  dis   <- scenarios$dist[case_idx]
  n     <- scenarios$n[case_idx]
  tau_q <- scenarios$tau_q[case_idx]

  cat("\n==============================================================================\n")
  cat(sprintf("Running Scenario %d/%d: Model=%s, Dist=%s, n=%d, tau=%.3f\n",
              case_idx, nrow(scenarios), model, dis, n, tau_q))
  cat("==============================================================================\n")

  # Performance metrics across Monte Carlo repetitions:
  #   - RMSE: Root Mean Squared Error (sqrt(mean((Q_hat - Q_true)^2)))
  #   - MAE:  Mean Absolute Error (mean(abs(Q_hat - Q_true)))
  #   - MLRE: Mean Log Relative Error (mean(log(Q_hat / Q_true)))
  #   - EER:  Empirical Exceedance Rate (mean(Y > Q_hat))
  rmse_prop_fsim <- rmse_prop_ll <- rmse_fsiqm <- rmse_efqr <- rmse_const_gpd <- rep(NA_real_, n_sim)
  mae_prop_fsim  <- mae_prop_ll  <- mae_fsiqm  <- mae_efqr  <- mae_const_gpd  <- rep(NA_real_, n_sim)
  mlre_prop_fsim <- mlre_prop_ll <- mlre_fsiqm <- mlre_efqr <- mlre_const_gpd <- rep(NA_real_, n_sim)
  eer_prop_fsim  <- eer_prop_ll  <- eer_fsiqm  <- eer_efqr  <- eer_const_gpd  <- rep(NA_real_, n_sim)

  for (sim in seq_len(n_sim)) {
    if (sim %% 10 == 0 || sim == 1) {
      cat(sprintf("[%s | %s | n=%d | tau=%.3f] Repetition %d / %d\n",
                  model, dis, n, tau_q, sim, n_sim))
    }
    set.seed(sim)

    # --------------------------------------------------------------------------
    # Step A: Generate Functional Predictor X_i(t) and Single-Index u_i
    # --------------------------------------------------------------------------
    T_pts <- 100
    t_grid <- seq(0, 1, length.out = T_pts)

    # Orthogonal basis functions (Section 3.1)
    phi1 <- sin(2 * pi * t_grid) / sqrt(2)
    phi2 <- cos(2 * pi * t_grid) / sqrt(2)
    phi3 <- sin(4 * pi * t_grid) / sqrt(2)
    phi4 <- cos(4 * pi * t_grid) / sqrt(2)
    phi_mat <- rbind(phi1, phi2, phi3, phi4)

    # Eigenvalues / variances lambda_k = 0.5^(k-1)
    lambda_k <- 0.5^(0:3)

    # Generate functional predictor X_i(t) = t + colSums(phi_mat * xi)
    X <- matrix(0, nrow = n, ncol = T_pts)
    for (i in seq_len(n)) {
      xi_k <- rnorm(4, mean = 0, sd = sqrt(lambda_k))
      X[i, ] <- t_grid + colSums(phi_mat * xi_k)
    }

    # True index function beta(t) = 2*t^2 + 0.25*t + 1
    beta_t <- 2 * t_grid^2 + 0.25 * t_grid + 1

    # Single-index projection u_i = \int_0^1 X_i(t) beta(t) dt
    integrals <- apply(X, 1, function(x) sum(x * beta_t) * (1 / (T_pts - 1)))

    # Link function f(u) for models M1, M2, M3
    if (model == "M1") {
      # M1 (Linear): f(u) = 5 * u
      f_x <- 5 * integrals
    } else if (model == "M2") {
      # M2 (Nonlinear): f(u) = 0.66 * exp(u^2)
      f_x <- 0.66 * exp(integrals^2)
    } else if (model == "M3") {
      # M3 (Linear + Nonlinear): f(u) = 0.66 * sin(u) + 0.2 * u
      f_x <- 0.66 * sin(integrals) + 0.2 * integrals
    }

    # --------------------------------------------------------------------------
    # Step B: Generate Heavy-Tailed Errors and Response Variable Y
    # --------------------------------------------------------------------------
    if (dis == "t2") {
      # Student-t with 2 degrees of freedom
      errors <- rt(n, df = 2)
      Y <- f_x + errors
      true_quantile <- f_x + qt(tau_q, df = 2)

    } else if (dis == "pareto25") {
      # GPD with constant shape gamma = 0.25 and scale sigma = 1
      errors <- evmix::rgpd(n, xi = 0.25, sigmau = 1)
      Y <- f_x + errors
      true_quantile <- f_x + evmix::qgpd(tau_q, u = 0, sigmau = 1, xi = 0.25, phiu = 1, lower.tail = TRUE)

    } else if (dis == "pareto_var") {
      # GPD with predictor-dependent shape gamma(X_i) = 0.2 + 0.5 * sin(2 * z_true)
      beta_true <- function(t) sin(2 * pi * t)
      z_true <- apply(X, 1, function(row) trapz(t_grid, row * beta_true(t_grid)))
      gamma_X <- 0.2 + 0.5 * sin(2 * z_true)

      errors <- sapply(gamma_X, function(g) evmix::rgpd(1, xi = g, sigmau = 1))
      Y <- f_x + errors
      true_quantile <- f_x + sapply(gamma_X, function(g) evmix::qgpd(tau_q, u = 0, sigmau = 1, xi = g))
    }

    # ==========================================================================
    # Method 1: EFQR (Zhu et al., 2022) - FPCA + Median QR + Weissman Extrapolation
    # ==========================================================================
    fpca_res <- FPCA(
      Ly = split(X, row(X)),
      Lt = split(rep(t_grid, each = n), row(X))
    )
    q_pc <- fpca_res$selectK
    fpca_scores <- fpca_res$xiEst[, 1:q_pc, drop = FALSE]

    # Median regression on FPC scores (tau = 0.5)
    qreg_model <- rq(Y ~ fpca_scores, tau = 0.5)
    res_error  <- residuals(qreg_model)

    # Weissman extrapolation via Hill estimator on upper residuals
    k_n <- floor(4 * n^(1 / 4))
    sorted_er <- sort(res_error, decreasing = TRUE)
    er_k <- sorted_er[k_n]
    hill_estimator <- mean(log(sorted_er[1:k_n]) - log(er_k))
    weissman_quantile <- er_k * ((k_n / n / (1 - tau_q))^(hill_estimator))

    pred_efqr <- weissman_quantile + qreg_model$fitted.values

    rmse_efqr[sim] <- sqrt(mean((true_quantile - pred_efqr)^2))
    mae_efqr[sim]  <- mean(abs(true_quantile - pred_efqr))
    mlre_efqr[sim] <- mean(log(pred_efqr / true_quantile))
    eer_efqr[sim]  <- mean(Y > pred_efqr)

    # ==========================================================================
    # Method 2: FSIQM (Sang & Cao, 2020) - Direct FSIM at Target Quantile Level
    # ==========================================================================
    n_bs <- 5:7
    lamb <- c(0.0001, 0.001, 0.01, 0.1)

    # GACV grid search for tuning parameters (lambda and num_bs)
    GACV_mat <- matrix(NA, nrow = length(lamb), ncol = length(n_bs))
    for (i in seq_len(nrow(GACV_mat))) {
      for (j in seq_len(ncol(GACV_mat))) {
        res_fsiqm_cv <- try(nlm(nllk.qsreg, p = seq(1, 3, length.out = n_bs[j] - 1),
                                num_bs = n_bs[j], X = X, timepts = t_grid,
                                Y = Y, tau = tau_q, lambda = lamb[i]), silent = TRUE)
        if (!inherits(res_fsiqm_cv, "try-error")) {
          GACV_mat[i, j] <- GACV.qsreg(X = X, timepts = t_grid, Y = Y,
                                      theta = res_fsiqm_cv$estimate,
                                      num_bs = n_bs[j], tau = tau_q, lambda = lamb[i])
        }
      }
    }

    colmin <- apply(GACV_mat, 2, min, na.rm = TRUE)
    bs_opt_fsiqm  <- n_bs[which.min(colmin)]
    lam_opt_fsiqm <- lamb[which.min(GACV_mat[, which.min(colmin)])]

    res_fsiqm <- nlm(nllk.qsreg, p = seq(1, 3, length.out = bs_opt_fsiqm - 1),
                     num_bs = bs_opt_fsiqm, X = X, timepts = t_grid,
                     Y = Y, tau = tau_q, lambda = lam_opt_fsiqm)

    bsbasis_fsiqm <- create.bspline.basis(rangeval = c(0, 1), nbasis = bs_opt_fsiqm, norder = 4)
    basismat_fsiqm <- eval.basis(t_grid, bsbasis_fsiqm)
    v_fsiqm <- matrix(NA, nrow = n, ncol = bs_opt_fsiqm)
    for (i in seq_len(n)) {
      v_fsiqm[i, ] <- apply(X[i, ] * basismat_fsiqm, 2, trapz, x = t_grid)
    }
    z_fsiqm <- v_fsiqm %*% c(1, res_fsiqm$estimate)

    fit_fsiqm <- qsreg(z_fsiqm, Y, alpha = tau_q, lam = lam_opt_fsiqm)
    pred_fsiqm <- predict(fit_fsiqm, z_fsiqm)

    rmse_fsiqm[sim] <- sqrt(mean((true_quantile - pred_fsiqm)^2))
    mae_fsiqm[sim]  <- mean(abs(true_quantile - pred_fsiqm))
    mlre_fsiqm[sim] <- mean(log(pred_fsiqm / true_quantile))
    eer_fsiqm[sim]  <- mean(Y > pred_fsiqm)

    # ==========================================================================
    # Intermediate Quantile Estimation via FSIM (Shared First Stage for EVT)
    # ==========================================================================
    GACV_interm <- matrix(NA, nrow = length(lamb), ncol = length(n_bs))
    for (i in seq_len(nrow(GACV_interm))) {
      for (j in seq_len(ncol(GACV_interm))) {
        res_cv <- try(nlm(nllk.qsreg, p = seq(1, 3, length.out = n_bs[j] - 1),
                          num_bs = n_bs[j], X = X, timepts = t_grid,
                          Y = Y, tau = tau_interm, lambda = lamb[i]), silent = TRUE)
        if (!inherits(res_cv, "try-error")) {
          GACV_interm[i, j] <- GACV.qsreg(X = X, timepts = t_grid, Y = Y,
                                          theta = res_cv$estimate,
                                          num_bs = n_bs[j], tau = tau_interm, lambda = lamb[i])
        }
      }
    }

    colmin_interm <- apply(GACV_interm, 2, min, na.rm = TRUE)
    bs_opt_interm  <- n_bs[which.min(colmin_interm)]
    lam_opt_interm <- lamb[which.min(GACV_interm[, which.min(colmin_interm)])]

    res_interm <- nlm(nllk.qsreg, p = seq(1, 3, length.out = bs_opt_interm - 1),
                      num_bs = bs_opt_interm, X = X, timepts = t_grid,
                      Y = Y, tau = tau_interm, lambda = lam_opt_interm)

    bsbasis_interm <- create.bspline.basis(rangeval = c(0, 1), nbasis = bs_opt_interm, norder = 4)
    basismat_interm <- eval.basis(t_grid, bsbasis_interm)
    v_interm <- matrix(NA, nrow = n, ncol = bs_opt_interm)
    for (i in seq_len(n)) {
      v_interm[i, ] <- apply(X[i, ] * basismat_interm, 2, trapz, x = t_grid)
    }
    z_interm <- v_interm %*% c(1, res_interm$estimate)

    fit_interm <- qsreg(z_interm, Y, alpha = tau_interm, lam = lam_opt_interm)
    Q_0 <- predict(fit_interm, z_interm)

    # Exceedances above intermediate threshold Q_0
    Zfull <- pmax(Y - Q_0, 0)
    I_exc <- which(Zfull > 0)
    Z_exc <- Zfull[I_exc]

    # Global constant GPD fit on exceedances (baseline & initial parameters)
    gpd0 <- fevd(x = Z_exc, type = "GP", threshold = 0)
    sigma0_fixed <- unname(gpd0$results$par["scale"])
    gamma0_init  <- unname(gpd0$results$par["shape"])
    gamma0_init  <- max(min(gamma0_init, 0.7), 1e-4)

    # ==========================================================================
    # Method 3: Proposed-FSIM (Partially Varying GPD with Soft-Cap Penalty)
    # ==========================================================================
    df_gam <- 4
    degree_value <- 3

    b_gamma <- c(1, res_interm$estimate)
    b_gamma <- b_gamma / sqrt(sum(b_gamma^2) + 1e-12)

    eta_all_init <- as.vector(v_interm %*% b_gamma)
    Boundary_knots_fixed <- as.numeric(quantile(eta_all_init, c(0.001, 0.999)))
    buf <- 0.1 * diff(Boundary_knots_fixed)
    Boundary_knots_fixed <- Boundary_knots_fixed + c(-buf, buf)

    n_int <- max(df_gam - degree_value - 1, 0)
    knots_fixed <- if (n_int > 0) {
      as.numeric(quantile(eta_all_init, probs = seq(0, 1, length.out = n_int + 2)[-c(1, n_int + 2)]))
    } else {
      NULL
    }

    # B-spline basis constructor clamped to boundary knots
    build_B_fixed <- function(eta, I_sub, knots, B_knots) {
      eta_clamped <- pmin(pmax(eta, B_knots[1]), B_knots[2])
      B_all <- bs(eta_clamped, degree = 3, knots = knots, Boundary.knots = B_knots, intercept = FALSE)
      list(B_exc = B_all[I_sub, , drop = FALSE], B_all = B_all)
    }

    K_spline <- ncol(build_B_fixed(eta_all_init, I_exc, knots_fixed, Boundary_knots_fixed)$B_all)
    D2 <- diff(diag(K_spline), differences = 2)
    softplus <- function(x) log1p(exp(pmin(x, 30)))

    # Penalized GPD negative log-likelihood for shape parameter spline coefficients
    penal_llh_a_fixed_sig <- function(a, Z, B, sigma_val, D2_mat, lam_pen, gamma_max = 0.8) {
      gamma_vec <- softplus(B %*% a) + 1e-6
      t1 <- 1 + (gamma_vec * Z) / sigma_val
      if (any(t1 <= 0) || any(!is.finite(t1))) return(1e12)
      nll <- sum(log(sigma_val) + (1 + 1 / gamma_vec) * log(t1))
      pen <- lam_pen * sum((D2_mat %*% a)^2) + 100 * sum(pmax(gamma_vec - gamma_max, 0)^2)
      return(nll + pen)
    }

    penal_llh_b_fixed_sig <- function(u, v_mat, I_sub, Z, a, knots, B_knots, sigma_val, lam_pen) {
      b_norm <- u / sqrt(sum(u^2) + 1e-12)
      B_stuff <- build_B_fixed(as.vector(v_mat %*% b_norm), I_sub, knots, B_knots)
      penal_llh_a_fixed_sig(a, Z, B_stuff$B_exc, sigma_val,
                            diff(diag(ncol(B_stuff$B_all)), differences = 2), lam_pen)
    }

    # Alternating optimization for (a, b)
    a_hat <- rep(log(exp(gamma0_init) - 1), K_spline)
    n_alt <- 5
    lam_pen <- 1

    for (it in seq_len(n_alt)) {
      # Update a | b
      eta_current <- as.vector(v_interm %*% b_gamma)
      B_stuff <- build_B_fixed(eta_current, I_exc, knots_fixed, Boundary_knots_fixed)

      opt_a <- optim(a_hat, penal_llh_a_fixed_sig,
                     Z = Z_exc, B = B_stuff$B_exc,
                     sigma_val = sigma0_fixed, D2_mat = D2, lam_pen = lam_pen, method = "BFGS")
      a_hat <- opt_a$par

      # Update b | a
      opt_b <- optim(b_gamma, penal_llh_b_fixed_sig,
                     v_mat = v_interm, I_sub = I_exc, Z = Z_exc, a = a_hat,
                     knots = knots_fixed, B_knots = Boundary_knots_fixed,
                     sigma_val = sigma0_fixed, lam_pen = lam_pen, method = "BFGS")
      b_gamma <- opt_b$par / sqrt(sum(opt_b$par^2) + 1e-12)
    }

    # Predict extreme quantiles via Equation (2.1)
    final_B <- build_B_fixed(as.vector(v_interm %*% b_gamma), I_exc, knots_fixed, Boundary_knots_fixed)
    gamma_hat_fsim <- as.vector(softplus(final_B$B_all %*% a_hat) + 1e-6)
    gamma_hat_fsim <- pmin(pmax(gamma_hat_fsim, 0.01), 0.8)
    sigma_hat_fsim <- rep(sigma0_fixed, length(gamma_hat_fsim))

    pred_prop_fsim <- Q_0 + (sigma_hat_fsim / gamma_hat_fsim) *
      (((1 - tau_q) / (1 - tau_interm))^(-gamma_hat_fsim) - 1)

    rmse_prop_fsim[sim] <- sqrt(mean((true_quantile - pred_prop_fsim)^2))
    mae_prop_fsim[sim]  <- mean(abs(true_quantile - pred_prop_fsim))
    mlre_prop_fsim[sim] <- mean(log(pred_prop_fsim / true_quantile))
    eer_prop_fsim[sim]  <- mean(Y > pred_prop_fsim)

    # ==========================================================================
    # Method 4: Proposed-LL (Localized Log-Likelihood with FPCA Kernel Weights)
    # ==========================================================================
    W_dist <- as.matrix(dist(fpca_scores, method = "euclidean"))
    W_exc  <- W_dist[, I_exc, drop = FALSE]
    n_test <- nrow(W_exc)

    init_pars_gpd <- ismev::gpd.fit(Z_exc, 0, show = FALSE)$mle

    result_param <- tibble::tibble(
      "sigma" = numeric(0),
      "xi"    = numeric(0)
    ) %>%
      dplyr::bind_rows(
        purrr::map_dfr(seq_len(n_test), function(obs_i) {
          # Gaussian kernel weights with bandwidth h = 1 (Remark 3)
          h <- 1.0
          wi_x <- exp(-(W_exc[obs_i, ]^2) / (h^2))
          optim_wrapper(wi_x, init_pars_gpd, Z_exc, lambda = 0.001, init_pars_gpd[2])
        })
      )

    sigma_hat_ll <- result_param$sigma
    gamma_hat_ll <- result_param$xi

    pred_prop_ll <- Q_0 + (sigma_hat_ll / gamma_hat_ll) *
      (((1 - tau_q) / (1 - tau_interm))^(-gamma_hat_ll) - 1)

    rmse_prop_ll[sim] <- sqrt(mean((true_quantile - pred_prop_ll)^2))
    mae_prop_ll[sim]  <- mean(abs(true_quantile - pred_prop_ll))
    mlre_prop_ll[sim] <- mean(log(pred_prop_ll / true_quantile))
    eer_prop_ll[sim]  <- mean(Y > pred_prop_ll)

    # ==========================================================================
    # Method 5: Constant-GPD (Baseline: FSIM Intermediate + Constant GPD)
    # ==========================================================================
    pred_const_gpd <- Q_0 + (sigma0_fixed / gamma0_init) *
      (((1 - tau_q) / (1 - tau_interm))^(-gamma0_init) - 1)

    rmse_const_gpd[sim] <- sqrt(mean((true_quantile - pred_const_gpd)^2))
    mae_const_gpd[sim]  <- mean(abs(true_quantile - pred_const_gpd))
    mlre_const_gpd[sim] <- mean(log(pred_const_gpd / true_quantile))
    eer_const_gpd[sim]  <- mean(Y > pred_const_gpd)

    # Interim summary printed during simulation
    if (sim %% 10 == 0 || sim == n_sim) {
      cat(sprintf("\n--- Summary up to Repetition %d ---\n", sim))
      res_interim <- rbind(
        RMSE = round(c(mean(rmse_prop_fsim[1:sim], na.rm = TRUE),
                       mean(rmse_prop_ll[1:sim], na.rm = TRUE),
                       mean(rmse_fsiqm[1:sim], na.rm = TRUE),
                       mean(rmse_efqr[1:sim], na.rm = TRUE),
                       mean(rmse_const_gpd[1:sim], na.rm = TRUE)), 2),
        MAE  = round(c(mean(mae_prop_fsim[1:sim], na.rm = TRUE),
                       mean(mae_prop_ll[1:sim], na.rm = TRUE),
                       mean(mae_fsiqm[1:sim], na.rm = TRUE),
                       mean(mae_efqr[1:sim], na.rm = TRUE),
                       mean(mae_const_gpd[1:sim], na.rm = TRUE)), 2),
        MLRE = round(c(mean(mlre_prop_fsim[1:sim], na.rm = TRUE),
                       mean(mlre_prop_ll[1:sim], na.rm = TRUE),
                       mean(mlre_fsiqm[1:sim], na.rm = TRUE),
                       mean(mlre_efqr[1:sim], na.rm = TRUE),
                       mean(mlre_const_gpd[1:sim], na.rm = TRUE)), 2)
      )
      colnames(res_interim) <- c("Proposed-FSIM", "Proposed-LL", "FSIQM", "EFQR", "Constant-GPD")
      print(res_interim)
    }
  }

  # ----------------------------------------------------------------------------
  # Save Simulation Results for Current Scenario
  # ----------------------------------------------------------------------------
  # Aliases for backward compatibility with existing visualization scripts
  mse_prop  <- rmse_prop_fsim;  mse_prop2  <- rmse_prop_ll;  mse_fsim  <- rmse_fsiqm;  mse_EFQR  <- rmse_efqr;  mse_c_gpd  <- rmse_const_gpd
  adis_prop <- mae_prop_fsim;   adis_prop2 <- mae_prop_ll;   adis_fsim <- mae_fsiqm;   adis_EFQR <- mae_efqr;  adis_c_gpd <- mae_const_gpd
  bias_prop <- mlre_prop_fsim;  bias_prop2 <- mlre_prop_ll;  bias_fsim <- mlre_fsiqm;  bias_EFQR <- mlre_efqr; bias_c_gpd <- mlre_const_gpd
  eer_prop  <- eer_prop_fsim;   eer_prop2  <- eer_prop_ll;   eer_fsim  <- eer_fsiqm;   eer_EFQR  <- eer_efqr;  eer_c_gpd  <- eer_const_gpd

  out_filename <- paste0(model, "_", dis, "_n", n, "_tauq", tau_q, ".RData")
  save(
    rmse_prop_fsim, rmse_prop_ll, rmse_fsiqm, rmse_efqr, rmse_const_gpd,
    mae_prop_fsim,  mae_prop_ll,  mae_fsiqm,  mae_efqr,  mae_const_gpd,
    mlre_prop_fsim, mlre_prop_ll, mlre_fsiqm, mlre_efqr, mlre_const_gpd,
    eer_prop_fsim,  eer_prop_ll,  eer_fsiqm,  eer_efqr,  eer_const_gpd,
    mse_prop, mse_prop2, mse_fsim, mse_EFQR, mse_c_gpd,
    adis_prop, adis_prop2, adis_fsim, adis_EFQR, adis_c_gpd,
    bias_prop, bias_prop2, bias_fsim, bias_EFQR, bias_c_gpd,
    eer_prop, eer_prop2, eer_fsim, eer_EFQR, eer_c_gpd,
    file = file.path(out_dir, out_filename)
  )
  cat(sprintf("\nResults saved to: %s\n", file.path(out_dir, out_filename)))

  # ----------------------------------------------------------------------------
  # Format Results as LaTeX Table Rows (Matching Paper Tables 1 and 2)
  # ----------------------------------------------------------------------------
  r1 <- round(c(mean(rmse_prop_fsim, na.rm = TRUE), mean(rmse_prop_ll, na.rm = TRUE),
                mean(rmse_fsiqm, na.rm = TRUE), mean(rmse_efqr, na.rm = TRUE),
                mean(rmse_const_gpd, na.rm = TRUE)), 2)
  r2 <- round(c(sd(rmse_prop_fsim, na.rm = TRUE), sd(rmse_prop_ll, na.rm = TRUE),
                sd(rmse_fsiqm, na.rm = TRUE), sd(rmse_efqr, na.rm = TRUE),
                sd(rmse_const_gpd, na.rm = TRUE)), 2)

  r3 <- round(c(mean(mae_prop_fsim, na.rm = TRUE), mean(mae_prop_ll, na.rm = TRUE),
                mean(mae_fsiqm, na.rm = TRUE), mean(mae_efqr, na.rm = TRUE),
                mean(mae_const_gpd, na.rm = TRUE)), 2)
  r4 <- round(c(sd(mae_prop_fsim, na.rm = TRUE), sd(mae_prop_ll, na.rm = TRUE),
                sd(mae_fsiqm, na.rm = TRUE), sd(mae_efqr, na.rm = TRUE),
                sd(mae_const_gpd, na.rm = TRUE)), 2)

  r5 <- round(c(mean(mlre_prop_fsim, na.rm = TRUE), mean(mlre_prop_ll, na.rm = TRUE),
                mean(mlre_fsiqm, na.rm = TRUE), mean(mlre_efqr, na.rm = TRUE),
                mean(mlre_const_gpd, na.rm = TRUE)), 2)
  r6 <- round(c(sd(mlre_prop_fsim, na.rm = TRUE), sd(mlre_prop_ll, na.rm = TRUE),
                sd(mlre_fsiqm, na.rm = TRUE), sd(mlre_efqr, na.rm = TRUE),
                sd(mlre_const_gpd, na.rm = TRUE)), 2)

  fmt_bold_min <- function(mean_vec, sd_vec, criterion = c("min", "abs_min")) {
    criterion <- match.arg(criterion)
    min_idx <- if (criterion == "abs_min") which.min(abs(mean_vec)) else which.min(mean_vec)
    
    sapply(seq_along(mean_vec), function(i) {
      txt <- paste0(sprintf("%.2f", mean_vec[i]), "(", sprintf("%.2f", sd_vec[i]), ")")
      if (i == min_idx) {
        paste0("\\textbf{", txt, "}")
      } else {
        txt
      }
    })
  }
  
  rmse_row <- fmt_bold_min(r1, r2, criterion = "min")
  mae_row  <- fmt_bold_min(r3, r4, criterion = "min")
  mlre_row <- fmt_bold_min(r5, r6, criterion = "abs_min")
  

  fmt_plain <- function(mean_vec, sd_vec) {
    sapply(seq_along(mean_vec), function(i)
      formatC(paste0(sprintf("%.2f", mean_vec[i]), "(", sprintf("%.2f", sd_vec[i]), ")"),
              width = col_width, flag = "-"))
  }
  
  col_width    <- 28
  method_names <- c("Proposed-FSIM", "Proposed-LL", "FSIQM", "EFQR", "Constant-GPD")
  header       <- paste(formatC(method_names, width = col_width, flag = "-"), collapse = " ")
  separator    <- strrep("-", nchar(paste(formatC("Metric", width = 8, flag = "-"), header)))
  
  lines <- c(
    separator,
    paste(formatC("Metric", width = 8, flag = "-"), header),
    separator,
    paste(formatC("RMSE",  width = 8, flag = "-"), paste(fmt_plain(r1, r2), collapse = " ")),
    paste(formatC("MAE",   width = 8, flag = "-"), paste(fmt_plain(r3, r4), collapse = " ")),
    paste(formatC("MLRE",  width = 8, flag = "-"), paste(fmt_plain(r5, r6), collapse = " ")),
    separator
  )
  
  cat(paste(lines, collapse = "\n"), "\n")

  }
