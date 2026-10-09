# Cross-check of centered-j-reference.csv against the R packages gmm and AER.
#
# Run from this directory:   Rscript centered-j-package-check.R
# Requires:                  gmm (tested with 1.9.1) and AER (tested with 1.2.17)
# Output:                    a table of the agreement, and an error if the
#                            values in the file do not agree with the packages
#
# centered-j-reference.R, which produces the reference values, is written in
# base R so that nothing is shared with linearmodels. This script is not part
# of the tests and does not write anything. It checks that the reference values
# are what established R implementations compute, to the precision that they
# have, which is limited by their numerical optimizers and not by the
# statistics.
#
#   * gmm::gmm with type="iterative" is the iterated GMM estimator. Its
#     centeredVcov argument selects whether the covariance of the moment
#     conditions is centered, and J is n times the objective. The J of a model
#     whose estimator is not centered is gmm with centeredVcov=FALSE. The
#     centered statistic is gmm::evalGmm with centeredVcov=TRUE evaluated at the
#     estimates, with the weight matrix evaluated at the same estimates.
#   * gmm::gmm with type="cue" is the continuously updating estimator.
#   * vcov="iid" is the covariance of the moment conditions without any
#     assumption on their variance, i.e., the heteroskedasticity robust
#     estimator (evalGmm does not have vcov="MDS"), and vcov="HAC" with
#     kernel="Bartlett" and prewhite=0 the kernel estimator. The
#     bandwidth bw of gmm is the number of lags plus one, so that bw=5 is
#     the 4 lags of linearmodels with weights 1 - j / 5.
#   * AER::ivreg Sargan test is the J statistic with the unadjusted estimator
#     of the covariance of the moments. It is the 2SLS J of the model, which
#     is the iterated GMM estimator with that weight.
#
# What is not checked is what the packages do not offer: gmm does not have a
# one-step weight matrix that is the same as linearmodels' (so the two-step
# estimates are not compared, their first step is identity weighting), clustered
# covariances, or the small-sample scale of debiased estimators. Those are
# checked against theory by centered-j-reference.R.

suppressMessages({
  library(foreign)
  library(gmm)
  library(AER)
})

ref <- read.csv("centered-j-reference.csv", stringsAsFactors = FALSE)
sim <- read.dta("simulated-data.dta")
sim$const <- 1
hs <- read.csv("housing.csv")
hs$const <- 1
reg <- model.matrix(~ region, hs)[, -1]
hs <- cbind(hs, reg)
mis <- read.csv("misspecified-data.csv")
mis$const <- 1

words <- function(s) strsplit(s, " ", fixed = TRUE)[[1]]

# Data of a row of the reference table, weighted as linearmodels weights it
model_data <- function(r) {
  d <- switch(r$dataset, sim = sim, housing = hs, mis = mis)
  sw <- if (r$weighted) sqrt(d$weights / mean(d$weights)) else rep(1, nrow(d))
  exog <- words(r$exog)
  endog <- words(r$endog)
  instr <- words(r$instruments)
  list(y = d[[r$dependent]] * sw,
       X = as.matrix(d[, c(exog, endog), drop = FALSE]) * sw,
       Z = as.matrix(d[, c(exog, instr), drop = FALSE]) * sw)
}

# The moment function of gmm, with the data in a single matrix [y, X, Z]
moments <- function(k) {
  function(theta, dat) {
    e <- dat[, 1] - dat[, 2:(k + 1)] %*% theta
    dat[, -(1:(k + 1)), drop = FALSE] * drop(e)
  }
}

gmm_args <- function(r) {
  if (r$weight_type == "kernel") {
    list(vcov = "HAC", kernel = "Bartlett", bw = as.integer(r$bandwidth) + 1,
         prewhite = 0)
  } else {
    list(vcov = "iid")
  }
}

# Fit by gmm, returning J and the estimates
fit_gmm <- function(r, md, centered, type) {
  k <- ncol(md$X)
  dat <- cbind(md$y, md$X, md$Z)
  n <- nrow(dat)
  g <- moments(k)
  # 2SLS starting values
  start <- drop(solve(crossprod(md$X, md$Z) %*% solve(crossprod(md$Z)) %*%
                        crossprod(md$Z, md$X),
                      crossprod(md$X, md$Z) %*% solve(crossprod(md$Z)) %*%
                        crossprod(md$Z, md$y)))
  run <- function(t0, type) {
    args <- c(list(g = g, x = dat, t0 = t0, type = type, wmatrix = "optimal",
                   centeredVcov = centered, crit = 1e-12),
              gmm_args(r))
    if (type == "iterative") args$itermax <- 1000
    do.call(gmm, args)
  }
  fit <- run(start, type)
  if (type == "cue") {
    # The CUE objective is hard to minimize. Use the best of two starts, the
    # 2SLS estimates and the iterated GMM estimates
    other <- run(coef(run(start, "iterative")), "cue")
    if (other$objective < fit$objective) fit <- other
  }
  list(b = unname(coef(fit)), J = unname(fit$objective * n), n = n, dat = dat, g = g)
}

# Objective of gmm times n at the parameters theta, with the weight matrix
# evaluated at theta_w, and with or without centering of the covariance of the
# moment conditions
eval_at <- function(r, md, theta, theta_w, centered) {
  k <- ncol(md$X)
  dat <- cbind(md$y, md$X, md$Z)
  args <- c(list(g = moments(k), x = dat, t0 = theta, tetw = theta_w,
                 wmatrix = "optimal", centeredVcov = centered), gmm_args(r))
  unname(do.call(evalGmm, args)$objective * nrow(dat))
}

# Sargan statistic of the 2SLS fit, which is the J of the unadjusted estimator
sargan <- function(r, md) {
  dat <- data.frame(y = md$y)
  xcols <- colnames(md$X)
  zcols <- colnames(md$Z)
  d <- cbind(dat, as.data.frame(md$X), as.data.frame(md$Z[, setdiff(zcols, xcols), drop = FALSE]))
  exog <- words(r$exog)
  form <- as.formula(sprintf("y ~ 0 + %s | 0 + %s",
                             paste(xcols, collapse = " + "),
                             paste(zcols, collapse = " + ")))
  fit <- ivreg(form, data = d)
  unname(summary(fit, diagnostics = TRUE)$diagnostics["Sargan", "statistic"])
}

check <- function(label, got, want, rtol) {
  err <- abs(got - want) / abs(want)
  if (!is.finite(err) || err > rtol) {
    stop(sprintf("%s: %.10g (package) vs %.10g (reference), relative difference %.2e",
                 label, got, want, err))
  }
  err
}

part_a <- list()
part_b <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i, ]
  if (r$weight_type == "clustered" || r$debiased || is.na(r$centered_j)) next
  md <- model_data(r)
  label <- sprintf("%s [%s]", r$id, r$method)

  if (r$weight_type == "unadjusted") {
    # 2SLS is the iterated GMM estimator with this weight, and its J is Sargan's
    if (r$method == "two_step") next
    s <- sargan(r, md)
    part_a[[label]] <- c(j = check(paste(label, "J vs Sargan"), s, r$j, 1e-6),
                         centered_j = check(paste(label, "centered J vs Sargan"), s,
                                            r$centered_j, 1e-6))
    next
  }

  # A. The statistics at the estimates of the reference, which are the same
  # parameters for both, so that only the statistics are compared. J uses the
  # weight matrix that was used to estimate the parameters
  theta <- as.numeric(words(r$params))
  theta_w <- as.numeric(words(r$weight_params))
  part_a[[label]] <- c(
    j = check(paste(label, "J"), eval_at(r, md, theta, theta_w, r$center), r$j, 1e-8),
    centered_j = check(paste(label, "centered J"),
                       eval_at(r, md, theta, theta, TRUE), r$centered_j, 1e-8)
  )

  # B. The estimators of the package. gmm does not have a two-step estimator
  # that starts as linearmodels does, and its CUE is only compared for the
  # robust estimator
  if (r$method == "two_step" || (r$method == "cue" && r$weight_type != "robust")) next
  fit <- fit_gmm(r, md, r$center, c(iterated = "iterative", cue = "cue")[[r$method]])
  if (r$method == "cue") {
    # The reference value is a minimum, so that the package cannot be lower.
    # The numerical optimizer of the package does not always find the minimum,
    # in which case there is nothing to compare
    if (fit$J < r$j * (1 - 1e-6)) {
      stop(sprintf("%s: the package finds J=%.8g, below the reference minimum %.8g",
                   label, fit$J, r$j))
    }
    if (fit$J > r$j * (1 + 5e-4)) {
      cat(sprintf("note: %s: gmm stops at J=%.6g, above the minimum %.6g; not compared
",
                  label, fit$J, r$j))
      next
    }
  }
  part_b[[label]] <- c(
    j = check(paste(label, "J of the estimator"), fit$J, r$j, 5e-4),
    params = max(abs(fit$b - theta) / pmax(abs(theta), 1e-2))
  )
  # The objective is flat in some directions, so that the estimates of the
  # package are less precise than its J
  if (part_b[[label]]["params"] >= 2e-2) {
    stop(sprintf("%s: the estimates of the package differ by %.2e (relative) from the reference",
                 label, part_b[[label]]["params"]))
  }
}

cat("
A. Statistics at the parameters of the reference (relative differences)
")
res_a <- do.call(rbind, part_a)
print(signif(res_a, 3))
cat("
B. Estimators of the package (relative differences in J and in the parameters)
")
res_b <- do.call(rbind, part_b)
print(signif(res_b, 3))
cat(sprintf(paste0("
A: %d models, largest relative difference %.2e in J and %.2e in ",
                   "the centered J
B: %d models, largest relative difference %.2e in J and ",
                   "%.2e in the parameters
"),
            nrow(res_a), max(res_a[, "j"]), max(res_a[, "centered_j"]),
            nrow(res_b), max(res_b[, "j"]), max(res_b[, "params"])))
