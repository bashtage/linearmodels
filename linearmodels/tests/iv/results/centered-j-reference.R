# Reference values for IVGMMResults.centered_j_stat, the J statistic that uses
# a centered estimate of the covariance of the moment conditions. Written in
# base R so that nothing is shared with linearmodels.
#
# Run from this directory:   Rscript centered-j-reference.R
# Output:                    centered-j-reference.csv (read by ../test_centered_j_stat.py)
#
# Definition (Hall 2000, Econometrica 68(6), 1517-1528; Hansen and Lee 2021,
# Econometrica 89(3), 1419-1447, Theorem 1 and Example 1): with moment
# conditions g_i = z_i e_i(b), where e_i(b) = y_i - x_i b, and gbar their mean
#
#   J  = n gbar' S^{-1}   gbar        S   = covariance of g_i, not centered
#   Jc = n gbar' S_c^{-1} gbar        S_c = the same estimator applied to g_i - gbar
#
# Both are evaluated at the same parameters b. The standard J of linearmodels
# (and of Stata's ivregress gmm) uses the weight matrix that was used in the
# last estimation step, which is evaluated at the previous iterate, while Jc
# evaluates S_c at the final estimate. They are the same at convergence of the
# iterated estimator when the estimator of S is the same.
#
# GMM conventions follow linearmodels.iv.IVGMM and Stata's ivregress gmm. The
# first step uses W = (Z'Z/n)^{-1}, later steps use W = S(b_prev)^{-1} where S
# is an estimate of the covariance of the moment conditions based on the
# residuals of the previous step. "two_step" is the estimator after one update
# of W (iter_limit=2 in linearmodels), "iterated" repeats this until the
# estimates converge, and "cue" minimizes n gbar(b)' S(b)^{-1} gbar(b) over b.
# Four estimators of S are used, which are the same as Stata's wmatrix()
# options:
#   robust      S = (1/n) sum (z_i e_i)(z_i e_i)'
#   unadjusted  S = s2 Z'Z/n, with s2 the variance of e about its mean
#   kernel      Bartlett, S = (1/n) [G_0 + sum_j (1 - j/(m+1)) (G_j + G_j')]
#               with G_j = sum_t (z_t e_t)(z_{t-j} e_{t-j})' and bandwidth m
#   clustered   S = (1/n) sum_g (sum_{i in g} z_i e_i)(sum_{i in g} z_i e_i)'
# "center" subtracts the mean of z_i e_i first, which is what the model's own
# estimator does when center=TRUE and has no effect on unadjusted, where the
# estimator is always centered. "debiased" uses the small-sample scale of
# linearmodels: n/(n-k) and (n-1)/(n-k) G/(G-1) for the clustered estimator,
# with k the number of regressors. Weighted models are estimated on data
# multiplied by the square root of the weights.
#
# Stata has no centered J, so there are no values from Stata for it. Instead
# the script
#   1. checks that its GMM and S estimators reproduce numbers produced by
#      Stata that are already used in the tests, ivregress gmm on the housing
#      and simulated data (stata-iv-housing-results.txt and
#      stata-iv-simulated-results.txt), for all four estimators of S, and
#   2. checks properties of the centered statistic that follow from theory
#      for every scenario that they apply to before writing it:
#      * Hansen and Lee (2021) Theorem 1: the iterated GMM estimator is the
#        same if S is centered, for the robust estimator,
#      * Jc = J / (1 - s J / n) for the iterated estimator with the robust
#        uncentered estimator, which follows from S = s (S_c + gbar gbar') and
#        the Sherman-Morrison formula and implies that J < n / s. s = 1
#        unless the estimator is debiased, when s = n / (n - k),
#      * Jc = J for unadjusted, whose estimator is always centered,
#      * the CUE estimates are minima of the objective function.
# The values written were also compared with the R packages gmm and AER by
# centered-j-package-check.R, which needs those packages and is not needed to
# run this script or the tests. At the estimates that this script computes the
# packages' statistics agree with the values to about 1e-14.
# The scenarios include strongly misspecified models, where J is far from the
# chi2 distribution and close to its upper bound n.

suppressMessages(library(foreign))

## ---------------------------------------------------------------- GMM ----
wt_spec <- function(type = "robust", center = FALSE, debiased = FALSE,
                    bandwidth = 0, clusters = NULL) {
  list(type = type, center = center, debiased = debiased,
       bandwidth = bandwidth, clusters = clusters)
}

# Estimate of the covariance of the moment conditions z_i e_i. nvar is the
# number of regressors
S_hat <- function(Z, e, wt, nvar) {
  n <- nrow(Z)
  if (wt$type == "unadjusted") {
    S <- mean((e - mean(e))^2) * crossprod(Z) / n
    return(if (wt$debiased) S * n / (n - nvar) else S)
  }
  ze <- Z * e
  if (wt$center) ze <- sweep(ze, 2, colMeans(ze))
  if (wt$type == "robust") {
    S <- crossprod(ze) / n
    if (wt$debiased) S <- S * n / (n - nvar)
  } else if (wt$type == "kernel") {
    S <- crossprod(ze)
    for (j in seq_len(wt$bandwidth)) {
      op <- crossprod(ze[(j + 1):n, , drop = FALSE], ze[1:(n - j), , drop = FALSE])
      S <- S + (1 - j / (wt$bandwidth + 1)) * (op + t(op))
    }
    S <- S / n
    if (wt$debiased) S <- S * n / (n - nvar)
  } else if (wt$type == "clustered") {
    S <- matrix(0, ncol(ze), ncol(ze))
    for (g in unique(wt$clusters)) {
      zb <- colSums(ze[wt$clusters == g, , drop = FALSE])
      S <- S + tcrossprod(zb)
    }
    S <- S / n
    if (wt$debiased) {
      ng <- length(unique(wt$clusters))
      S <- S * (n - 1) / (n - nvar) * ng / (ng - 1)
    }
  } else {
    stop("unknown weight type")
  }
  S
}

gmm_beta <- function(y, X, Z, W) {
  A <- crossprod(X, Z)
  drop(solve(A %*% W %*% t(A), A %*% W %*% crossprod(Z, y)))
}

gmm_J <- function(y, X, Z, b, W) {
  n <- nrow(Z)
  g <- crossprod(Z, y - X %*% b) / n
  drop(n * crossprod(g, W %*% g))
}

# Estimator after one update of the weight matrix
two_step <- function(y, X, Z, wt) {
  n <- nrow(Z)
  b1 <- gmm_beta(y, X, Z, solve(crossprod(Z) / n))
  W <- solve(S_hat(Z, drop(y - X %*% b1), wt, ncol(X)))
  list(b = gmm_beta(y, X, Z, W), W = W, b1 = b1)
}

# Iterate until the largest change in the parameters is tiny. The weight
# matrix returned was evaluated at the previous iterate, so that is is
# equal to the weight matrix of the final estimates up to the tolerance
iterated <- function(y, X, Z, wt, tol = 1e-13, itermax = 20000) {
  n <- nrow(Z)
  b <- gmm_beta(y, X, Z, solve(crossprod(Z) / n))
  for (it in seq_len(itermax)) {
    W <- solve(S_hat(Z, drop(y - X %*% b), wt, ncol(X)))
    b_new <- gmm_beta(y, X, Z, W)
    converged <- max(abs(b_new - b)) <= tol * max(1, abs(b))
    b <- b_new
    if (converged) break
  }
  stopifnot(converged)
  list(b = b, W = W, iterations = it)
}

# Objective function of the continuously updating estimator
cue_objective <- function(b, y, X, Z, wt) {
  n <- nrow(Z)
  e <- drop(y - X %*% b)
  g <- colMeans(Z * e)
  n * drop(crossprod(g, solve(S_hat(Z, e, wt, ncol(X)), g)))
}

# The objective is flat near its minimum, which optim's BFGS with numerical
# gradients does not locate precisely enough, so use Newton's method with
# central difference derivatives. The Hessian is made positive definite and
# the step is shortened until the objective decreases
cue <- function(y, X, Z, wt, start, tol = 1e-12) {
  f <- function(b) cue_objective(b, y, X, Z, wt)
  k <- length(start)
  unit <- diag(k)
  grad <- function(b, h) {
    sapply(seq_len(k), function(j) (f(b + h[j] * unit[, j]) - f(b - h[j] * unit[, j])) / (2 * h[j]))
  }
  b <- start
  for (it in seq_len(200)) {
    h <- 1e-5 * pmax(abs(b), 1e-2)
    g <- grad(b, h)
    H <- sapply(seq_len(k), function(j) {
      (grad(b + h[j] * unit[, j], h) - grad(b - h[j] * unit[, j], h)) / (2 * h[j])
    })
    H <- (H + t(H)) / 2
    lambda <- 0
    while (inherits(try(chol(H + lambda * unit), silent = TRUE), "try-error")) {
      lambda <- max(2 * lambda, 1e-6 * max(abs(diag(H))))
    }
    step <- solve(H + lambda * unit, g)
    size <- 1
    while (f(b - size * step) > f(b) && size > 1e-10) size <- size / 2
    b_new <- b - size * step
    done <- max(abs(b_new - b)) <= tol * max(1, abs(b))
    b <- b_new
    if (done) break
  }
  stopifnot(done)
  list(b = b, J = f(b))
}

# J of the model, using S from the weight matrix that was used for estimation
# (W), and J that uses the centered estimator of S evaluated at b. The
# centered statistic is NA if its S is singular, which uses the rank
# tolerance of numpy.linalg.matrix_rank
centered_J <- function(y, X, Z, b, wt) {
  n <- nrow(Z)
  e <- drop(y - X %*% b)
  g <- colMeans(Z * e)
  wc <- wt
  wc$center <- TRUE
  S <- S_hat(Z, e, wc, ncol(X))
  d <- svd(S, nu = 0, nv = 0)$d
  if (sum(d > max(d) * max(dim(S)) * .Machine$double.eps) < ncol(Z)) {
    return(NA_real_)
  }
  n * drop(crossprod(g, solve(S, g)))
}

near <- function(a, b, rtol, what) {
  if (abs(a - b) > rtol * abs(b)) {
    stop(sprintf("check failed for %s: %.10g vs %.10g", what, a, b))
  }
  cat(sprintf("  ok  %-62s %.10g (expected %.10g)\n", what, a, b))
}

## -------------------------------------------------------------- data -----
sim <- read.dta("simulated-data.dta")
sim$const <- 1
# Six blocks of 100 observations, which is as many clusters as moments in the
# scenario with the clusters "block6"
sim$block6 <- (seq_len(nrow(sim)) - 1) %/% 100
hs <- read.csv("housing.csv")
hs$const <- 1
reg <- model.matrix(~ region, hs)[, -1]
hs <- cbind(hs, reg)
reg_cols <- colnames(reg)
mis <- read.csv("misspecified-data.csv")
mis$const <- 1

mat <- function(d, cols) as.matrix(d[, cols, drop = FALSE])

## ------------------------------------------------------- calibration -----
cat("Calibration against Stata results already used in the tests\n")
# ivregress gmm rent pcturban (hsngval = faminc i.region), wmatrix(...) from
# stata-iv-housing-results.txt: gmm-{robust,unadjusted,cluster}-asymptotic
Xh <- mat(hs, c("hsngval", "pcturban", "const"))
Zh <- mat(hs, c("pcturban", "const", "faminc", reg_cols))
fit <- two_step(hs$rent, Xh, Zh, wt_spec())
near(fit$b[1], 0.00146432787, 1e-6, "housing GMM b[hsngval]")
near(fit$b[2], 0.76154815601, 1e-6, "housing GMM b[pcturban]")
near(fit$b[3], 112.12271295, 1e-6, "housing GMM b[_cons]")
near(gmm_J(hs$rent, Xh, Zh, fit$b, fit$W), 6.8364006463, 1e-6,
     "housing Hansen J, robust, chi2(3)")
hj <- function(wt) {
  f <- two_step(hs$rent, Xh, Zh, wt)
  gmm_J(hs$rent, Xh, Zh, f$b, f$W)
}
near(hj(wt_spec("unadjusted")), 11.287665072, 1e-6, "housing Hansen J, unadjusted")
near(hj(wt_spec("clustered", clusters = as.integer(factor(hs$division)))),
     3.6677640329, 1e-6, "housing Hansen J, clustered by division")

# ivregress gmm y x3 x4 x5 (x1 = z1 z2), wmatrix(...) on the simulated data. The
# dependent variable differs by weight matrix, as in the Stata run
Xs <- mat(sim, c("x1", "const", "x3", "x4", "x5"))
Zs <- mat(sim, c("const", "x3", "x4", "x5", "z1", "z2"))
sj <- function(dep, wt) {
  f <- two_step(sim[[dep]], Xs, Zs, wt)
  gmm_J(sim[[dep]], Xs, Zs, f$b, f$W)
}
cl <- sim$cluster_id
near(sj("y_unadjusted", wt_spec("unadjusted")), 0.38076944209, 1e-6,
     "simulated Hansen J, unadjusted")
near(sj("y_robust", wt_spec("robust")), 0.22164820274, 1e-6,
     "simulated Hansen J, robust")
near(sj("y_robust", wt_spec("robust", center = TRUE)), 0.22173011287, 1e-6,
     "simulated Hansen J, robust, centered")
near(sj("y_clustered", wt_spec("clustered", clusters = cl)), 0.40598636292, 1e-6,
     "simulated Hansen J, clustered")
near(sj("y_clustered", wt_spec("clustered", center = TRUE, clusters = cl)),
     0.40736456674, 1e-6, "simulated Hansen J, clustered, centered")
near(sj("y_kernel", wt_spec("kernel", bandwidth = 12)), 0.42317006975, 1e-6,
     "simulated Hansen J, Bartlett kernel with 12 lags")
near(sj("y_kernel", wt_spec("kernel", center = TRUE, bandwidth = 12)),
     0.42592978028, 1e-6, "simulated Hansen J, Bartlett kernel, centered")

## ----------------------------------------------------------- scenarios ---
# dataset, dependent, exog, endog and instruments are space separated name
# lists. weighted uses the column "weights" of the dataset. bandwidth is the
# number of lags of a kernel weight matrix, and clusters the name of the
# column that defines the clusters of a clustered weight matrix. iterated and
# cue select the estimators that are computed in addition to two_step.
scen <- function(id, dataset, dependent, exog, endog, instruments,
                 weight_type = "robust", center = FALSE, debiased = FALSE,
                 weighted = FALSE, bandwidth = "", clusters = "",
                 iterated = TRUE, cue = FALSE) {
  data.frame(id = id, dataset = dataset, dependent = dependent, exog = exog,
             endog = endog, instruments = instruments, weighted = weighted,
             weight_type = weight_type, center = center, debiased = debiased,
             bandwidth = bandwidth, clusters = clusters, iterated = iterated,
             cue = cue, stringsAsFactors = FALSE)
}
hinstr <- paste(c("faminc", reg_cols), collapse = " ")
e2 <- list(exog = "const x3", endog = "x1 x2", instruments = "z1 z2 x4 x5")
e1 <- list(exog = "const x3 x5", endog = "x1", instruments = "z1 z2 x4")
sc <- rbind(
  # Heteroskedasticity robust
  scen("sim_e1_i3", "sim", "y_robust", e1$exog, e1$endog, e1$instruments),
  scen("sim_e2_i4", "sim", "y_robust", e2$exog, e2$endog, e2$instruments, cue = TRUE),
  scen("sim_e2_i4_weighted", "sim", "y_robust", e2$exog, e2$endog, e2$instruments,
       weighted = TRUE),
  scen("sim_e2_i4_center", "sim", "y_robust", e2$exog, e2$endog, e2$instruments,
       center = TRUE, cue = TRUE),
  scen("sim_e2_i4_debiased", "sim", "y_robust", e2$exog, e2$endog, e2$instruments,
       debiased = TRUE),
  scen("housing", "housing", "rent", "const pcturban", "hsngval", hinstr),
  scen("mis_robust", "mis", "y", "const", "x", "z1 z2 zb", cue = TRUE),
  scen("mis_robust_weighted", "mis", "y", "const", "x", "z1 z2 zb", weighted = TRUE),
  scen("mis_robust_center", "mis", "y", "const", "x", "z1 z2 zb", center = TRUE,
       cue = TRUE),
  scen("mis_robust_debiased", "mis", "y", "const", "x", "z1 z2 zb", debiased = TRUE),
  # Unadjusted
  scen("sim_e2_i4_unadjusted", "sim", "y_unadjusted", e2$exog, e2$endog,
       e2$instruments, weight_type = "unadjusted"),
  scen("housing_unadjusted", "housing", "rent", "const pcturban", "hsngval", hinstr,
       weight_type = "unadjusted"),
  scen("mis_unadjusted", "mis", "y", "const", "x", "z1 z2 zb",
       weight_type = "unadjusted"),
  # Kernel
  scen("sim_e2_i4_kernel6", "sim", "y_kernel", e2$exog, e2$endog, e2$instruments,
       weight_type = "kernel", bandwidth = 6),
  scen("sim_e1_i3_kernel3_center", "sim", "y_kernel", e1$exog, e1$endog,
       e1$instruments, weight_type = "kernel", center = TRUE, bandwidth = 3),
  scen("sim_e2_i4_kernel12_weighted", "sim", "y_kernel", e2$exog, e2$endog,
       e2$instruments, weight_type = "kernel", weighted = TRUE, bandwidth = 12),
  scen("mis_kernel4", "mis", "y", "const", "x", "z1 z2 zb", weight_type = "kernel",
       bandwidth = 4),
  # Clustered
  scen("sim_e2_i4_clustered", "sim", "y_clustered", e2$exog, e2$endog,
       e2$instruments, weight_type = "clustered", clusters = "cluster_id"),
  scen("sim_e1_i3_clustered_center", "sim", "y_clustered", e1$exog, e1$endog,
       e1$instruments, weight_type = "clustered", center = TRUE,
       clusters = "cluster_id"),
  scen("housing_clustered", "housing", "rent", "const pcturban", "hsngval", hinstr,
       weight_type = "clustered", clusters = "division"),
  scen("mis_clustered", "mis", "y", "const", "x", "z1 z2 zb",
       weight_type = "clustered", clusters = "group", cue = TRUE),
  scen("mis_clustered_debiased", "mis", "y", "const", "x", "z1 z2 zb",
       weight_type = "clustered", debiased = TRUE, clusters = "group"),
  # As many clusters as moment conditions, so that the centered covariance has
  # rank 5 and not 6 and the centered statistic is not defined
  scen("sim_e1_i3_six_clusters", "sim", "y_clustered", e1$exog, e1$endog,
       e1$instruments, weight_type = "clustered", clusters = "block6",
       iterated = FALSE)
)

words <- function(s) strsplit(s, " ", fixed = TRUE)[[1]]

run_scenario <- function(r) {
  d <- switch(r$dataset, sim = sim, housing = hs, mis = mis)
  sw <- if (r$weighted) sqrt(d$weights / mean(d$weights)) else rep(1, nrow(d))
  y <- d[[r$dependent]] * sw
  X <- mat(d, c(words(r$exog), words(r$endog))) * sw
  Z <- mat(d, c(words(r$exog), words(r$instruments))) * sw
  n <- nrow(Z)
  wt <- wt_spec(r$weight_type, r$center, r$debiased,
                if (r$bandwidth == "") 0 else as.integer(r$bandwidth),
                if (r$clusters == "") NULL else as.integer(factor(d[[r$clusters]])))
  df <- ncol(Z) - ncol(X)
  # b is the estimate, and wb the parameters that the weight matrix of J was
  # evaluated at, which are those of the previous step of a two step estimator
  fmt <- function(b) paste(sprintf("%.17g", b), collapse = " ")
  row <- function(method, b, j, wb = b) {
    data.frame(r[, c("id", "dataset", "dependent", "exog", "endog", "instruments",
                     "weighted", "weight_type", "center", "debiased", "bandwidth",
                     "clusters")],
               method = method, nobs = n, df = df, j = j,
               centered_j = centered_J(y, X, Z, b, wt), params = fmt(b),
               weight_params = fmt(wb), stringsAsFactors = FALSE)
  }
  out <- list()

  ts <- two_step(y, X, Z, wt)
  out$two_step <- row("two_step", ts$b, gmm_J(y, X, Z, ts$b, ts$W), wb = ts$b1)

  if (r$iterated) {
    it <- iterated(y, X, Z, wt)
    j <- gmm_J(y, X, Z, it$b, it$W)
    res <- row("iterated", it$b, j)
    out$iterated <- res
    jc <- res$centered_j
    if (r$weight_type == "robust") {
      # Hansen and Lee (2021) Theorem 1: the iterated estimator does not
      # change if S is centered
      other <- wt
      other$center <- !wt$center
      b_other <- iterated(y, X, Z, other)$b
      stopifnot(max(abs(b_other - it$b)) <= 1e-8 * max(1, abs(it$b)))
      if (!r$center) {
        # S = s (S_c + gbar gbar') with the small-sample scale s of the
        # estimator, and Sherman-Morrison, which implies J < n / s
        s <- if (r$debiased) n / (n - ncol(X)) else 1
        stopifnot(j < n / s)
        near(jc, j / (1 - s * j / n), 1e-8,
             paste(r$id, "Jc = J / (1 - s J / n)"))
      }
    }
    if (r$weight_type == "unadjusted") {
      near(jc, j, 1e-8, paste(r$id, "Jc = J for unadjusted"))
    }
  }

  if (r$cue) {
    fit <- cue(y, X, Z, wt, ts$b)
    # Another start finds the same minimum
    again <- cue(y, X, Z, wt, ts$b * (1 + 0.05 * c(1, -1, 1, -1)[seq_along(ts$b)]))
    near(again$J, fit$J, 1e-7, paste(r$id, "CUE minimum is the same from other start"))
    stopifnot(fit$J <= gmm_J(y, X, Z, ts$b, ts$W) + 1e-8)
    out$cue <- row("cue", fit$b, fit$J)
  }
  do.call(rbind, out)
}

cat("Scenarios and checks of the properties of the statistic\n")
res <- do.call(rbind, lapply(seq_len(nrow(sc)), function(i) run_scenario(sc[i, ])))
rownames(res) <- NULL
for (i in seq_len(nrow(res))) {
  cat(sprintf("%-30s %-9s J=%-12.8g Jc=%-12.8g df=%d\n", res$id[i], res$method[i],
              res$j[i], res$centered_j[i], res$df[i]))
}
# The scenarios are useful only if the statistics differ in some of them
stopifnot(any(res$centered_j > 1.1 * res$j, na.rm = TRUE))
stopifnot(any(is.na(res$centered_j)))
write.csv(res, "centered-j-reference.csv", row.names = FALSE, quote = TRUE)
cat("wrote centered-j-reference.csv\n")
