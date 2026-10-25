# Extreme Quantile Estimation via Functional Single Index Models

This repository provides the simulation code for the paper:

> **Extreme Quantile Estimation via Functional Single Index Models**  
> Yeonjoo Park and Yaeji Lim

---

## Overview

This repository implements the proposed extreme conditional quantile estimation framework under scalar-on-function regression (SoFR), combining **Functional Single-Index Models (FSIM)** with **Extreme Value Theory (EVT)** using Generalized Pareto Distributions (GPD) under the Peaks-Over-Threshold (POT) framework.

The code evaluates and compares the following five methods:
1. **`Proposed-FSIM`**: The proposed approach that estimates GPD parameters under the FSIM framework with a partially varying structure $\boldsymbol{\theta}(x) = \{\gamma(x), \sigma\}$ and a soft-cap regularization penalty.
2. **`Proposed-LL`**: The proposed approach that estimates GPD parameters under a localized log-likelihood framework with FPCA score kernel weights $\boldsymbol{\theta}(x) = \{\gamma(x), \sigma(x)\}$.
3. **`FSIQM`**: Functional Single-Index Quantile Model (Sang & Cao, 2020) directly fitted at the target quantile level without EVT extrapolation.
4. **`EFQR`**: Extreme Functional Quantile Regression (Zhu et al., 2022) using FPCA, median regression ($\tau = 0.5$), and Weissman extrapolation.
5. **`Constant-GPD`**: Baseline model using the FSIM intermediate quantile threshold $\widehat{Q}_{\tau_n}(x)$, followed by constant (covariate-independent) GPD parameters $(\gamma, \sigma)$ estimated globally.

---

## Directory Structure

```text
├── code.R    # Main Monte Carlo simulation script
├── qsreg.R             # FSIQM fitting and GACV tuning helper functions
├── weighted_llh.cpp    # C++ implementation of weighted GPD log-likelihood (RcppArmadillo)
├── results/            # Directory where output .RData files are automatically saved
└── README.md           # Documentation
```

---

## Prerequisites & Installation

The code is implemented in **R** and requires the following packages:

```R
install.packages(c(
  "fda", "fields", "caTools", "splines",
  "fdapace", "quantreg", "extRemes", "evmix",
  "ismev", "dplyr", "purrr", "tibble",
  "Rcpp", "RcppArmadillo"
))
```

---

## How to Run

### 1. Running the Full Simulation
To run the simulations from the terminal or RStudio:

```bash
Rscript code.R
```

### 2. Customizing Configurations
You can adjust the simulation parameters at the beginning of `code.R` (or set them before sourcing the script in R):

```R
# Model link functions: "M1" (Linear), "M2" (Nonlinear), "M3" (Linear + Nonlinear)
model <- "M3"

# Number of Monte Carlo repetitions (default in paper: 100)
n_sim <- 100

# Intermediate quantile level
tau_interm <- 0.8

# Run simulation
source("code.R")
```

---

## Evaluation Metrics

## 📊 Evaluation Metrics

As described in Section 3.2 of the paper:

* **RMSE** (Root Mean Squared Error):  
  `sqrt( mean( (Q_hat - Q_true)^2 ) )`
* **MAE** (Mean Absolute Error):  
  `mean( abs( Q_hat - Q_true ) )`
* **MLRE** (Mean Log Relative Error):  
  `mean( log( Q_hat / Q_true ) )`
* **EER** (Empirical Exceedance Rate):  
  `mean( Y > Q_hat )`

Outputs are saved in the `results/` folder as `.RData` files for downstream analysis and visualization.
