# ==============================================================================
# Single-Index Quantile Regression Helper Functions (Sang & Cao, 2020)
# ==============================================================================

suppressPackageStartupMessages({
  library(fda)
  library(caTools)
  library(fields)
})

#' Pinball / Check Loss Function for Quantile Regression
#'
#' @param r Residual value (numeric vector)
#' @param alpha Quantile level tau in (0, 1)
#' @return Value of rho_tau(r)
check.rho <- function(r, alpha) {
  ifelse(r < 0, alpha * r - r, alpha * r)
}

#' Helper function to test for even numbers
even <- function(x) x %% 2 == 0

#' Profile Quantile Loss for Single-Index Quantile Regression (B-spline Basis)
#'
#' @param X Functional covariate matrix (n x m)
#' @param timepts Evaluation grid points in [0, 1]
#' @param Y Response vector (length n)
#' @param theta Spline coefficient parameters (length num_bs - 1)
#' @param num_bs Number of B-spline basis functions
#' @param tau Quantile level
#' @param lambda Smoothing parameter for link function estimation
#' @return Mean pinball loss for the given parameters
nllk.qsreg <- function(X, timepts, Y, theta, num_bs, tau, lambda) {
  # Fix the first coefficient to 1 for model identifiability: beta(0) = 1
  theta_int <- c(1, theta[1:(num_bs - 1)])
  Tmax <- max(timepts)

  bsbasis <- create.bspline.basis(rangeval = c(0, Tmax), nbasis = num_bs, norder = 4)
  basismat <- eval.basis(timepts, bsbasis)

  u <- matrix(NA, nrow = nrow(X), ncol = num_bs)
  for (i in 1:nrow(X)) {
    u[i, ] <- apply(X[i, ] * basismat, 2, trapz, x = timepts)
  }
  z <- u %*% theta_int

  fit <- qsreg(as.vector(z), as.vector(Y), alpha = tau, lam = lambda)
  if (any(abs(theta_int) > 50)) return(1e08)

  mean(check.rho(fit$residuals, alpha = tau))
}

#' Profile Quantile Loss for Single-Index Quantile Regression (Fourier Basis)
nllk.qsreg.F <- function(X, timepts, Y, theta, num_bs, tau, lambda) {
  evensum <- sum(theta[even(1:(num_bs - 1))])
  theta_int <- c(1 - evensum, theta[1:(num_bs - 1)])
  Tmax <- max(timepts)

  bsbasis <- create.fourier.basis(rangeval = c(0, Tmax), nbasis = num_bs)
  basismat <- eval.basis(timepts, bsbasis)

  u <- matrix(NA, nrow = nrow(X), ncol = num_bs)
  for (i in 1:nrow(X)) {
    u[i, ] <- apply(X[i, ] * basismat, 2, trapz, x = timepts)
  }
  z <- u %*% theta_int

  fit <- qsreg(as.vector(z), as.vector(Y), alpha = tau, lam = lambda)
  if (any(abs(theta_int) > 50)) return(1e08)

  mean(check.rho(fit$residuals, alpha = tau))
}

#' Generalized Approximate Cross-Validation (GACV) Criterion (B-spline Basis)
#'
#' Used to select tuning parameters: lambda and num_bs
GACV.qsreg <- function(X, timepts, Y, theta, num_bs, tau, lambda) {
  theta_int <- c(1, theta[1:(num_bs - 1)])
  Tmax <- max(timepts)

  bsbasis <- create.bspline.basis(rangeval = c(0, Tmax), nbasis = num_bs, norder = 4)
  basismat <- eval.basis(timepts, bsbasis)
  n <- nrow(X)

  u <- matrix(NA, nrow = n, ncol = num_bs)
  for (i in 1:nrow(X)) {
    u[i, ] <- apply(X[i, ] * basismat, 2, trapz, x = timepts)
  }
  z <- u %*% theta_int

  fit <- qsreg(as.vector(z), as.vector(Y), alpha = tau, lam = lambda)
  (sum(check.rho(fit$residuals, alpha = tau))) / (n - fit$trace)
}

#' Generalized Approximate Cross-Validation (GACV) Criterion (Fourier Basis)
GACV.qsreg.F <- function(X, timepts, Y, theta, num_bs, tau, lambda) {
  evensum <- sum(theta[even(1:(num_bs - 1))])
  theta_int <- c(1 - evensum, theta[1:(num_bs - 1)])
  Tmax <- max(timepts)

  bsbasis <- create.fourier.basis(rangeval = c(0, Tmax), nbasis = num_bs)
  basismat <- eval.basis(timepts, bsbasis)
  n <- nrow(X)

  u <- matrix(NA, nrow = n, ncol = num_bs)
  for (i in 1:nrow(X)) {
    u[i, ] <- apply(X[i, ] * basismat, 2, trapz, x = timepts)
  }
  z <- u %*% theta_int

  fit <- qsreg(z, Y, alpha = tau, lam = lambda)
  (sum(check.rho(fit$residuals, alpha = tau))) / (n - fit$trace)
}
