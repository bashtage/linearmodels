# Reference values for linearmodels.iv.IVJIVE, the jackknife IV estimator of
# Angrist, Imbens and Krueger (1999), written in base R so that nothing is
# shared with linearmodels.
#
# Run from this directory:   Rscript jive-reference.R
# Output:                    jive-reference.csv  (read by ../test_jive.py)
#
# The estimator is computed from its definition, not from the leverage
# shortcut: the first stage of every endogenous regressor on Z is re-estimated n
# times, leaving out observation i, and the prediction at z_i is x~_i. An
# exogenous regressor is in Z and so is its own prediction. Then
#
#     b = (X~'X)^{-1} X~'y
#
# and, since JIVE is an IV estimator with the instruments X~, the covariance
# is the usual sandwich with X~ as the instrument matrix
#
#     V = (X~'X)^{-1} S (X'X~)^{-1},   S an estimate of Var(x~_i e_i)
#
# where S is one of (e is the residual y - X b, which uses X and not X~)
#   robust      S = (1/n) sum e_i^2 x~_i x~_i'
#   unadjusted  S = (e'e/n) X~'X~/n
#   kernel      S = (1/n) [G_0 + sum_j k_j (G_j + G_j')], G_j = sum_t s_t s_{t-j}'
#               with s_t = x~_t e_t and Bartlett, Parzen or quadratic spectral
#               weights k_j
#   clustered   S = (1/n) sum_g (sum_{i in g} x~_i e_i)(sum_{i in g} x~_i e_i)'
# debiased multiplies S by n/(n-k), and for clustered by
# (n-1)/(n-k) * G/(G-1). Weighted estimation is estimation on data multiplied
# by the square root of the weights, which here are scaled to have mean one.
#
# The kernel weights, for the lag j, bandwidth m and z = j / (m + 1), are
#   Bartlett  1 - z
#   Parzen    1 - 6 z^2 + 6 z^3 if z <= 1/2, 2 (1 - z)^3 otherwise
# for j <= m, and for the quadratic spectral kernel, with z = 6 pi j / (5 m),
#   3 (sin(z)/z - cos(z)) / z^2
# for all lags (Andrews, 1991).
#
# The covariance code is also used for two-stage least squares, using the
# fitted values X^ = P_Z X as the instrument matrix, to check it against the
# Stata results in stata-iv-simulated-results.txt and stata-iv-housing-results.txt,
# which cover the unadjusted, robust, clustered and Bartlett kernel covariance
# estimators, with and without weights and the small sample adjustment. The
# script stops if it does not reproduce them. It also checks that the
# leave-one-out first stage equals the shortcut (P_Z X - h X) / (1 - h), where
# h are the leverages.
#
# JIVE has no results from Stata or R in this repository, so the JIVE values
# below are from this script alone, and not a comparison with an independent
# implementation of JIVE. The Parzen and quadratic spectral kernels have no
# Stata results either.
#
# Also reported are the R-squared (centered if there is a constant) and the
# Wald statistic that all parameters other than the constant are zero, which is
# divided by the number of parameters tested if debiased.

suppressMessages(library(foreign))

## ---------------------------------------------------------- estimators ----
# Prediction of every endogenous column of X from Z excluding observation i
loo_first_stage <- function(X, Z, w, endog) {
  n <- nrow(X)
  out <- X
  for (i in seq_len(n)) {
    fit <- lm.wfit(Z[-i, , drop = FALSE], X[-i, endog, drop = FALSE], w[-i])
    out[i, endog] <- drop(Z[i, , drop = FALSE] %*% as.matrix(fit$coefficients))
  }
  out
}

kernel_weights <- function(kernel, bandwidth, n) {
  lags <- seq_len(n - 1)
  if (kernel == "bartlett") {
    z <- lags / (bandwidth + 1)
    ifelse(lags <= bandwidth, 1 - z, 0)
  } else if (kernel == "parzen") {
    z <- lags / (bandwidth + 1)
    ifelse(lags <= bandwidth, ifelse(z <= 0.5, 1 - 6 * z^2 + 6 * z^3, 2 * (1 - z)^3), 0)
  } else if (kernel == "qs") {
    z <- 6 * pi * lags / (5 * bandwidth)
    3 * (sin(z) / z - cos(z)) / z^2
  } else {
    stop("unknown kernel")
  }
}

# Covariance of an IV estimator with instruments Zs, in the scaled data
iv_cov <- function(Zs, Xs, es, cov_type, debiased = FALSE, kernel = "bartlett",
                   bandwidth = 0, clusters = NULL) {
  n <- nrow(Xs)
  k <- ncol(Xs)
  scores <- Zs * drop(es)
  if (cov_type == "unadjusted") {
    S <- mean(es^2) * crossprod(Zs) / n
  } else if (cov_type == "robust") {
    S <- crossprod(scores) / n
  } else if (cov_type == "kernel") {
    k_j <- kernel_weights(kernel, bandwidth, n)
    S <- crossprod(scores)
    for (j in seq_along(k_j)) {
      if (k_j[j] == 0) next
      op <- crossprod(scores[(j + 1):n, , drop = FALSE], scores[1:(n - j), , drop = FALSE])
      S <- S + k_j[j] * (op + t(op))
    }
    S <- S / n
  } else if (cov_type == "clustered") {
    S <- matrix(0, k, k)
    for (g in unique(clusters)) {
      sb <- colSums(scores[clusters == g, , drop = FALSE])
      S <- S + tcrossprod(sb)
    }
    S <- S / n
  } else {
    stop("unknown cov_type")
  }
  if (debiased) {
    if (cov_type == "clustered") {
      G <- length(unique(clusters))
      S <- S * (n - 1) / (n - k) * G / (G - 1)
    } else {
      S <- S * n / (n - k)
    }
  }
  A <- solve(crossprod(Zs, Xs) / n)
  A %*% S %*% t(A) / n
}

# method is "jive" or "2sls". endog gives the columns of X that are endogenous
iv_fit <- function(y, X, Z, w = NULL, method = "jive", endog = integer(0)) {
  n <- nrow(X)
  w <- if (is.null(w)) rep(1, n) else w / mean(w)
  sw <- sqrt(w)
  if (method == "jive") {
    inst <- sw * loo_first_stage(X, Z, w, endog)
  } else {
    Zs <- Z * sw
    inst <- Zs %*% solve(crossprod(Zs), crossprod(Zs, X * sw))
  }
  Xs <- X * sw
  ys <- y * sw
  b <- drop(solve(crossprod(inst, Xs), crossprod(inst, ys)))
  list(b = b, inst = inst, Xs = Xs, ys = ys, es = drop(ys - Xs %*% b), w = w)
}

# R-squared and the Wald statistic of all parameters but the constant
fit_stats <- function(fit, V, X, debiased) {
  const <- which(apply(X, 2, function(c) all(c == 1)))
  if (length(const) == 1) {
    y <- fit$ys / sqrt(fit$w)
    ybar <- sum(fit$w * y) / sum(fit$w)
    tss <- sum(fit$w * (y - ybar)^2)
  } else {
    tss <- sum(fit$ys^2)
  }
  r2 <- 1 - sum(fit$es^2) / tss
  nc <- setdiff(seq_along(fit$b), const)
  wald <- drop(t(fit$b[nc]) %*% solve(V[nc, nc, drop = FALSE]) %*% fit$b[nc])
  c(r2 = r2, fstat = if (debiased) wald / length(nc) else wald)
}

## ------------------------------------------------------------- data -------
sim <- read.dta("simulated-data.dta")
sim$const <- 1
housing <- read.csv("housing.csv")
housing$const <- 1
hreg <- model.matrix(~ region, housing)[, -1]
housing <- cbind(housing, hreg)
card <- read.csv(bzfile("../../../datasets/card/card.csv.bz2"))
card$const <- 1

mat <- function(d, cols) as.matrix(d[, cols, drop = FALSE])
near <- function(a, b, rtol, what) {
  if (any(abs(a - b) > rtol * abs(b))) {
    stop(sprintf("calibration failed for %s", what))
  }
}

## ------------------------------------------------------ calibration ------
cat("Calibration of the covariance code against Stata ivregress 2sls\n")
stata_block <- function(lines, name) {
  start <- which(lines == sprintf("########## !%s! ##########", name))
  stopifnot(length(start) == 1)
  i <- start + 3
  names_ <- b <- t <- c()
  while (!grepl("^r2\t", lines[i])) {
    parts <- strsplit(lines[i], "\t")[[1]]
    names_ <- c(names_, parts[1])
    b <- c(b, as.numeric(parts[2]))
    t <- c(t, as.numeric(strsplit(lines[i + 1], "\t")[[1]][2]))
    i <- i + 2
  }
  list(names = names_, b = b, se = abs(b / t))
}
sim_lines <- readLines("stata-iv-simulated-results.txt")
cal <- list(
  c(vce = "unadjusted", stata = "vce(unadjusted)", dep = "y_unadjusted"),
  c(vce = "robust", stata = "vce(robust)", dep = "y_robust"),
  c(vce = "clustered", stata = "vce(cluster cluster_id)", dep = "y_clustered"),
  c(vce = "kernel", stata = "vce(hac bartlett 12)", dep = "y_kernel")
)
ncal <- 0
for (spec in cal) {
  for (weighted in c(FALSE, TRUE)) {
    for (small in c(FALSE, TRUE)) {
      name <- sprintf("2sls-num_endog_1-num_exog_3-num_instr_2-weighted_%s-%s-%s",
                      if (weighted) "True" else "False", spec[["stata"]],
                      if (small) "small" else "")
      st <- stata_block(sim_lines, name)
      X <- mat(sim, c("x1", "x3", "x4", "x5", "const"))
      Z <- mat(sim, c("x3", "x4", "x5", "const", "z1", "z2"))
      w <- if (weighted) sim$weights else NULL
      fit <- iv_fit(sim[[spec[["dep"]]]], X, Z, w, "2sls")
      V <- iv_cov(fit$inst, fit$Xs, fit$es, spec[["vce"]], small, "bartlett", 12, sim$cluster_id)
      stopifnot(identical(st$names, c("x1", "x3", "x4", "x5", "_cons")))
      near(fit$b, st$b, 1e-6, paste(name, "params"))
      near(sqrt(diag(V)), st$se, 1e-6, paste(name, "std errors"))
      ncal <- ncal + 1
    }
  }
}
cat(sprintf("  ok  2SLS parameters and standard errors for %d Stata results\n", ncal))
cat("      (simulated data; unadjusted, robust, clustered, kernel; weighted or not; small or not)\n")

# The housing data (Stata's hsng2), where the instruments have a high leverage.
# The clusters are the division
house_lines <- readLines("stata-iv-housing-results.txt")
hX <- mat(housing, c("hsngval", "pcturban", "const"))
hZ <- mat(housing, c("pcturban", "const", "faminc", colnames(hreg)))
nhouse <- 0
for (spec in list(c("unadjusted", "unadjusted"), c("robust", "robust"), c("clustered", "cluster"))) {
  for (small in c(FALSE, TRUE)) {
    name <- sprintf("2sls-%s-%s", spec[2], if (small) "small" else "asymptotic")
    st <- stata_block(house_lines, name)
    fit <- iv_fit(housing$rent, hX, hZ, NULL, "2sls")
    V <- iv_cov(fit$inst, fit$Xs, fit$es, spec[1], small, "bartlett", 0, as.integer(factor(housing$division)))
    near(fit$b, st$b, 1e-6, paste(name, "params"))
    near(sqrt(diag(V)), st$se, 1e-6, paste(name, "std errors"))
    nhouse <- nhouse + 1
  }
}
cat(sprintf("  ok  2SLS parameters and standard errors for %d Stata results for the housing data\n", nhouse))

# The leave-one-out first stage equals its leverage shortcut, and an exogenous
# regressor is its own leave-one-out prediction
X <- mat(sim, c("x1", "x3", "const"))
Z <- mat(sim, c("x3", "const", "z1", "z2", "x4", "x5"))
loo_all <- loo_first_stage(X, Z, rep(1, nrow(X)), 1:3)
H <- Z %*% solve(crossprod(Z), t(Z))
h <- diag(H)
shortcut <- (H %*% X - h * X) / (1 - h)
near(loo_all[, 1], shortcut[, 1], 1e-8, "leave-one-out first stage equals the shortcut")
near(loo_all[, 2:3], X[, 2:3], 1e-8, "exogenous regressors are their own leave-one-out predictions")
w_test <- sim$weights / mean(sim$weights)
loo_w <- loo_first_stage(X, Z, w_test, 1)
Zw <- Z * sqrt(w_test)
Hw <- Zw %*% solve(crossprod(Zw), t(Zw))
hw <- diag(Hw)
shortcut_w <- (Hw %*% (X * sqrt(w_test)) - hw * (X * sqrt(w_test))) / (1 - hw)
near(sqrt(w_test) * loo_w[, 1], shortcut_w[, 1], 1e-8, "weighted leave-one-out first stage equals the shortcut")
cat("  ok  the leave-one-out first stage equals (P_Z X - h X) / (1 - h), weighted or not\n")

## ---------------------------------------------------------- scenarios -----
# exog, endog and instruments are space separated names, with the dummy variables
# g1, ..., g24 (cluster_id modulo 25, the first is dropped) for the instruments
# that have a high leverage. weights is the name of the column with the
# weights, or empty. kernel and bandwidth are used for cov_type kernel, where
# the bandwidth is the maximum lag, or for the quadratic spectral kernel the
# bandwidth parameter. clusters is the name of the column with the clusters.
group <- factor(sim$cluster_id %% 25)
dummies <- model.matrix(~ group)[, -1]
colnames(dummies) <- paste0("g", seq_len(ncol(dummies)))
sim <- cbind(sim, dummies)
datasets <- list(sim = sim, housing = housing, card = card)
group_cols <- paste(colnames(dummies), collapse = " ")
house_instr <- paste(c("faminc", colnames(hreg)), collapse = " ")
card_exog <- paste("const exper expersq black smsa south smsa66",
                   paste0("reg66", 1:8, collapse = " "))

scen <- function(id, dataset, dep, exog, endog, instruments, weights, cov_type,
                 debiased, kernel = "", bandwidth = "", clusters = "") {
  data.frame(id = id, dataset = dataset, dep = dep, exog = exog, endog = endog,
             instruments = instruments, weights = weights, cov_type = cov_type,
             debiased = debiased, kernel = kernel, bandwidth = bandwidth,
             clusters = clusters, stringsAsFactors = FALSE)
}
sim_s <- function(id, exog, endog, instruments, weights, cov_type, debiased, ...) {
  scen(id, "sim", "y_robust", exog, endog, instruments, weights, cov_type, debiased, ...)
}
one <- function(...) sim_s("one_endog", "const x3", "x1", "z1 z2 x4 x5", "", ...)
sc <- rbind(
  # Every covariance estimator, with and without the small sample adjustment
  one("unadjusted", FALSE),
  one("robust", FALSE),
  one("kernel", FALSE, "bartlett", 4),
  one("kernel", FALSE, "parzen", 4),
  one("kernel", FALSE, "qs", 3.5),
  one("clustered", FALSE, clusters = "cluster_id"),
  one("unadjusted", TRUE),
  one("robust", TRUE),
  one("kernel", TRUE, "bartlett", 4),
  one("kernel", TRUE, "parzen", 4),
  one("kernel", TRUE, "qs", 3.5),
  one("clustered", TRUE, clusters = "cluster_id"),
  # More than one endogenous regressor
  sim_s("two_endog", "const x3", "x1 x2", "z1 z2 x4 x5", "", "robust", FALSE),
  sim_s("two_endog", "const x3", "x1 x2", "z1 z2 x4 x5", "", "unadjusted", FALSE),
  sim_s("two_endog", "const x3", "x1 x2", "z1 z2 x4 x5", "", "clustered", TRUE,
        clusters = "cluster_id"),
  sim_s("three_endog", "const", "x1 x2 x3", "z1 z2 x4 x5", "", "robust", FALSE),
  sim_s("three_endog", "const", "x1 x2 x3", "z1 z2 x4 x5", "", "kernel", TRUE, "parzen", 3),
  # Weights
  sim_s("weighted", "const x3", "x1", "z1 z2 x4 x5", "weights", "unadjusted", FALSE),
  sim_s("weighted", "const x3", "x1", "z1 z2 x4 x5", "weights", "robust", FALSE),
  sim_s("weighted", "const x3", "x1", "z1 z2 x4 x5", "weights", "kernel", FALSE, "bartlett", 4),
  sim_s("weighted", "const x3", "x1", "z1 z2 x4 x5", "weights", "kernel", FALSE, "qs", 3.5),
  sim_s("weighted", "const x3", "x1", "z1 z2 x4 x5", "weights", "clustered", TRUE,
        clusters = "cluster_id"),
  sim_s("weighted_two_endog", "const x3", "x1 x2", "z1 z2 x4 x5", "weights", "robust", TRUE),
  sim_s("weighted_two_endog", "const x3", "x1 x2", "z1 z2 x4 x5", "weights", "clustered", FALSE,
        clusters = "cluster_id"),
  # Exactly identified, several exogenous regressors, no exogenous regressors
  sim_s("just_identified", "const x3", "x1", "z1", "", "robust", FALSE),
  sim_s("just_identified", "const x3", "x1", "z1", "", "unadjusted", TRUE),
  sim_s("multi_exog", "const x3 x4 x5", "x1", "z1 z2", "", "robust", FALSE),
  sim_s("multi_exog", "const x3 x4 x5", "x1", "z1 z2", "", "clustered", TRUE, clusters = "cluster_id"),
  sim_s("no_exog", "", "x1", "z1 z2 x4", "", "robust", FALSE),
  sim_s("no_exog", "", "x1", "z1 z2 x4", "", "unadjusted", FALSE),
  # Many instruments with a high leverage
  sim_s("many_dummies", "const x3", "x1", group_cols, "", "robust", FALSE),
  sim_s("many_dummies", "const x3", "x1", group_cols, "", "unadjusted", TRUE),
  sim_s("many_dummies", "const x3", "x1", group_cols, "", "clustered", FALSE, clusters = "cluster_id"),
  sim_s("many_dummies", "const x3", "x1", group_cols, "weights", "robust", FALSE),
  # The housing data: 50 observations and instruments with a high leverage
  scen("housing", "housing", "rent", "const pcturban", "hsngval", house_instr, "", "unadjusted", FALSE),
  scen("housing", "housing", "rent", "const pcturban", "hsngval", house_instr, "", "robust", FALSE),
  scen("housing", "housing", "rent", "const pcturban", "hsngval", house_instr, "", "robust", TRUE),
  scen("housing", "housing", "rent", "const pcturban", "hsngval", house_instr, "", "clustered", FALSE,
       clusters = "division"),
  # Card (1995), 3010 observations, the returns to education using college
  # proximity as the instruments
  scen("card", "card", "lwage", card_exog, "educ", "nearc2 nearc4", "", "unadjusted", FALSE),
  scen("card", "card", "lwage", card_exog, "educ", "nearc2 nearc4", "", "robust", FALSE),
  scen("card", "card", "lwage", card_exog, "educ", "nearc2 nearc4", "", "robust", TRUE),
  scen("card", "card", "lwage", card_exog, "educ", "nearc2 nearc4", "weight", "robust", FALSE)
)

words <- function(s) strsplit(s, " ", fixed = TRUE)[[1]]
cache <- list()
rows <- list()
for (i in seq_len(nrow(sc))) {
  r <- sc[i, ]
  d <- datasets[[r$dataset]]
  exog <- words(r$exog)
  endog <- words(r$endog)
  X <- mat(d, c(exog, endog))
  Z <- mat(d, c(exog, words(r$instruments)))
  w <- if (r$weights == "") NULL else d[[r$weights]]
  key <- paste(r$id, r$dataset, r$weights)
  if (is.null(cache[[key]])) {
    cache[[key]] <- iv_fit(d[[r$dep]], X, Z, w, "jive", length(exog) + seq_along(endog))
  }
  fit <- cache[[key]]
  cl <- if (r$clusters == "") NULL else as.integer(factor(d[[r$clusters]]))
  V <- iv_cov(fit$inst, fit$Xs, fit$es, r$cov_type, r$debiased,
              if (r$kernel == "") "bartlett" else r$kernel,
              if (r$bandwidth == "") 0 else as.numeric(r$bandwidth), cl)
  st <- fit_stats(fit, V, X, r$debiased)
  rows[[i]] <- data.frame(r, term = colnames(X), coef = fit$b, se = sqrt(diag(V)),
                          r2 = st[["r2"]], fstat = st[["fstat"]],
                          stringsAsFactors = FALSE, row.names = NULL)
  cat(sprintf("%-19s %-10s %-9s debiased=%-5s b[%s]=%.8f se=%.8f\n", r$id, r$cov_type,
              r$kernel, r$debiased, colnames(X)[ncol(X)], fit$b[ncol(X)], sqrt(diag(V))[ncol(X)]))
}
res <- do.call(rbind, rows)
write.csv(res, "jive-reference.csv", row.names = FALSE, quote = TRUE)
cat("wrote jive-reference.csv\n")
