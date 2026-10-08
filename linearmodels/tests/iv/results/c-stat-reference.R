# Reference values for IVGMMResults.c_stat (the C or "difference-in-Hansen"
# statistic), written in base R so that nothing is shared with linearmodels.
#
# Run from this directory:   Rscript c-stat-reference.R
# Output:                    c-stat-reference.csv  (read by ../test_c_stat.py)
#
# Definition (Hayashi 2000, Econometrics, pp. 218-221 and 232-234; Baum,
# Schaffer and Stillman 2003, Stata Journal 3(1), section 4.4, which is the
# definition used by ivreg2's orthog() option):
#
#   Model E treats the tested regressors as exogenous. Its moment conditions
#   are z_e = [exog, tested, instruments]. Estimate it by two-step efficient
#   GMM, giving J_e and the moment covariance S_e that was used for it.
#   Model C is the original model with moments z = [exog, instruments], which
#   are a subset of the moments in E. Re-estimate it by GMM using the weight
#   matrix W_c = (S_e[z, z])^{-1} formed from the sub-matrix of S_e whose rows
#   and columns correspond to z, and compute J_c with the same weights.
#
#       C = J_e - J_c  ~  chi2(number of tested variables)
#
# Because the same S_e is used for both statistics C >= 0 by construction.
# If the original model is just identified J_c = 0 and C = J_e.
#
# GMM conventions follow linearmodels.iv.IVGMM and Stata's ivregress gmm: the
# first step uses W = (Z'Z/n)^{-1}, the second step uses W = S^{-1} where S is
# an estimate of the covariance of the moment conditions built from first-step
# residuals, and J = n g'Wg uses that same W. Four estimators of S are used,
# all of which are the same as Stata's wmatrix() options:
#   robust      S = (1/n) sum (z_i e_i)(z_i e_i)'
#   unadjusted  S = (e'e/n) Z'Z/n
#   kernel      Bartlett, S = (1/n) [G_0 + sum_j (1 - j/(m+1)) (G_j + G_j')]
#               with G_j = sum_t (z_t e_t)(z_{t-j} e_{t-j})' and bandwidth m
#   clustered   S = (1/n) sum_g (sum_{i in g} z_i e_i)(sum_{i in g} z_i e_i)'
# "center" subtracts the mean of z_i e_i first (not used by unadjusted, where
# the residuals have mean zero because the model has a constant). Weighted
# models are estimated on data multiplied by the square root of the weights.
#
# Before it writes anything the script checks that this implementation
# reproduces numbers produced by Stata that are already used in the tests:
#   * ivregress gmm on the housing data (stata-iv-housing-results.txt), which
#     is an overidentified model, for the robust, unadjusted and clustered
#     weight matrices,
#   * ivregress gmm on the simulated data (stata-iv-simulated-results.txt)
#     for all four estimators, with and without centering, and
#   * estat endogenous / estat overid on the simulated data
#     (see ../test_postestimation.py).
# The repository's tests skip weighted GMM because Stata's treatment of weights
# differs slightly, so the weighted scenarios below use linearmodels' convention
# of scaling the data by the square root of the weights and are not Stata values.
# Equivalent Stata code for a scenario in the grid is, e.g.,
#   ivreg2 y x3 (x1 x2 = z1 z2 x4 x5), gmm2s robust orthog(x1)   [not run here]

suppressMessages(library(foreign))

## ---------------------------------------------------------------- GMM ----
wt_spec <- function(type = "robust", center = FALSE, bandwidth = 0,
                    clusters = NULL) {
  list(type = type, center = center, bandwidth = bandwidth, clusters = clusters)
}

# Estimate of the covariance of the moment conditions z_i e_i
S_hat <- function(Z, e, wt) {
  n <- nrow(Z)
  if (wt$type == "unadjusted") {
    # The mean of the residuals is zero when there is a constant, which makes
    # the estimator's treatment of centering irrelevant
    stopifnot(abs(mean(e)) <= 1e-8 * sd(e))
    return(mean(e^2) * crossprod(Z) / n)
  }
  ze <- Z * e
  if (wt$center) ze <- sweep(ze, 2, colMeans(ze))
  if (wt$type == "robust") {
    S <- crossprod(ze) / n
  } else if (wt$type == "kernel") {
    S <- crossprod(ze)
    for (j in seq_len(wt$bandwidth)) {
      op <- crossprod(ze[(j + 1):n, , drop = FALSE], ze[1:(n - j), , drop = FALSE])
      S <- S + (1 - j / (wt$bandwidth + 1)) * (op + t(op))
    }
    S <- S / n
  } else if (wt$type == "clustered") {
    S <- matrix(0, ncol(ze), ncol(ze))
    for (g in unique(wt$clusters)) {
      zb <- colSums(ze[wt$clusters == g, , drop = FALSE])
      S <- S + tcrossprod(zb)
    }
    S <- S / n
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

two_step <- function(y, X, Z, wt = wt_spec()) {
  n <- nrow(Z)
  b1 <- gmm_beta(y, X, Z, solve(crossprod(Z) / n))
  e1 <- drop(y - X %*% b1)
  S <- S_hat(Z, e1, wt)
  W <- solve(S)
  b2 <- gmm_beta(y, X, Z, W)
  list(b = b2, S = S, W = W, J = gmm_J(y, X, Z, b2, W))
}

## Returns C, J_e, J_c, the Hansen J of the original model and the df.
c_stat <- function(y, exog, endog, instr, tested, sw = NULL, wt = wt_spec(),
                   check = TRUE) {
  if (is.null(sw)) sw <- rep(1, length(y))
  y <- y * sw
  exog <- exog * sw
  endog <- endog * sw
  instr <- instr * sw
  if (length(tested) == 0) tested <- colnames(endog)
  rest <- setdiff(colnames(endog), tested)
  X <- cbind(exog, endog)
  Z <- cbind(exog, instr)
  Xe <- cbind(exog, endog[, tested, drop = FALSE], endog[, rest, drop = FALSE])
  Ze <- cbind(exog, endog[, tested, drop = FALSE], instr)
  full <- two_step(y, Xe, Ze, wt)
  ke <- ncol(exog)
  nt <- length(tested)
  keep <- c(seq_len(ke), ke + nt + seq_len(ncol(instr)))
  Wc <- solve(full$S[keep, keep])
  bc <- gmm_beta(y, X, Z, Wc)
  Jc <- gmm_J(y, X, Z, bc, Wc)
  out <- list(c = full$J - Jc, j_e = full$J, j_c = Jc,
              j_orig = two_step(y, X, Z, wt)$J, df = nt)
  if (check) {
    n <- nrow(Z)
    # (1) the closed form estimate minimises the restricted objective
    Q <- function(b) {
      g <- crossprod(Z, y - X %*% b) / n
      n * drop(crossprod(g, Wc %*% g))
    }
    ps <- pmax(abs(bc), 1e-3)
    o <- optim(bc + 0.05 * ps, Q, method = "BFGS",
               control = list(parscale = ps, reltol = 1e-15, maxit = 5000))
    stopifnot(abs(o$value - Jc) <= 1e-6 * max(1, abs(Jc)))
    # (2) inverse of a sub-block of S equals the Schur complement of W
    W <- full$W
    t_ <- setdiff(seq_len(ncol(Ze)), keep)
    sch <- W[keep, keep] - W[keep, t_, drop = FALSE] %*%
      solve(W[t_, t_, drop = FALSE]) %*% W[t_, keep, drop = FALSE]
    stopifnot(max(abs(sch - Wc)) <= 1e-8 * max(abs(Wc)))
    # (3) C >= 0 by construction
    stopifnot(out$c >= -1e-9)
    # (4) the order of the instruments and of the tested variables is
    # irrelevant. Kernel estimators depend on the order of the observations,
    # not of the variables
    perm <- rev(seq_len(ncol(instr)))
    again <- c_stat(y, exog, endog[, rev(colnames(endog)), drop = FALSE],
                    instr[, perm, drop = FALSE], rev(tested), wt = wt,
                    check = FALSE)
    stopifnot(abs(again$c - out$c) <= 1e-8 * max(1, abs(out$c)))
  }
  out
}

near <- function(a, b, rtol, what) {
  if (abs(a - b) > rtol * abs(b)) {
    stop(sprintf("calibration failed for %s: %.10g vs %.10g", what, a, b))
  }
  cat(sprintf("  ok  %-56s %.10g (Stata %.10g)\n", what, a, b))
}

## -------------------------------------------------------------- data -----
sim <- read.dta("simulated-data.dta")
sim$const <- 1
hs <- read.csv("housing.csv")
hs$const <- 1
reg <- model.matrix(~ region, hs)[, -1]
hs <- cbind(hs, reg)
reg_cols <- colnames(reg)

mat <- function(d, cols) as.matrix(d[, cols, drop = FALSE])

## ------------------------------------------------------- calibration -----
cat("Calibration against Stata results already used in the tests\n")
# ivregress gmm rent pcturban (hsngval = faminc i.region), wmatrix(...)
# from stata-iv-housing-results.txt: gmm-{robust,unadjusted,cluster}-asymptotic
Xh <- mat(hs, c("hsngval", "pcturban", "const"))
Zh <- mat(hs, c("pcturban", "const", "faminc", reg_cols))
fit <- two_step(hs$rent, Xh, Zh)
near(fit$b[1], 0.00146432787, 1e-6, "housing GMM b[hsngval]")
near(fit$b[2], 0.76154815601, 1e-6, "housing GMM b[pcturban]")
near(fit$b[3], 112.12271295, 1e-6, "housing GMM b[_cons]")
near(fit$J, 6.8364006463, 1e-6, "housing Hansen J, robust, chi2(3)")
near(fit$W[1, 1], 0.00001443637, 1e-5, "housing weight matrix W[pcturban, pcturban]")
near(fit$W[3, 3], 2.531069e-10, 1e-5, "housing weight matrix W[faminc, faminc]")
near(two_step(hs$rent, Xh, Zh, wt_spec("unadjusted"))$J, 11.287665072, 1e-6,
     "housing Hansen J, unadjusted")
near(two_step(hs$rent, Xh, Zh,
              wt_spec("clustered", clusters = as.integer(factor(hs$division))))$J,
     3.6677640329, 1e-6, "housing Hansen J, clustered by division")

# ivregress gmm y x3 x4 x5 (x1 = z1 z2), wmatrix(...) on the simulated data. The
# dependent variable differs by weight matrix, as in the Stata run
Xs <- mat(sim, c("x1", "const", "x3", "x4", "x5"))
Zs <- mat(sim, c("const", "x3", "x4", "x5", "z1", "z2"))
sj <- function(dep, wt) two_step(sim[[dep]], Xs, Zs, wt)$J
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

# ../test_postestimation.py: Stata estat endogenous / estat overid
sx <- c("const", "x3", "x4", "x5")
a <- c_stat(sim$y_robust, mat(sim, sx), mat(sim, c("x1", "x2")),
            mat(sim, c("z1", "z2")), character(0))
near(a$c, 22.684, 1e-4, "simulated C, all endogenous tested (Stata 22.684)")
a <- c_stat(sim$y_robust, mat(sim, sx), mat(sim, c("x1", "x2")),
            mat(sim, c("z1", "z2")), "x1")
near(a$c, 0.158525, 1e-3, "simulated C, x1 tested (Stata 0.158525)")
a <- c_stat(sim$y_robust, mat(sim, sx), mat(sim, "x1"),
            mat(sim, c("z1", "z2")), "x1")
near(a$j_orig, 0.221648, 1e-4, "simulated Hansen J, x1 endogenous (Stata 0.221648)")

## ----------------------------------------------------------- scenarios ---
# dataset, exog, endog, instruments and tested are space separated name lists.
# tested == "" means every endogenous variable. Names in `instruments` that
# are x*/z* columns of the simulated data are used as extra instruments.
# bandwidth is the number of lags of a kernel weight matrix, and clusters the
# name of the column that defines the clusters of a clustered weight matrix.
scen <- function(id, dataset, exog, endog, instruments, tested, weighted,
                 weight_type = "robust", center = FALSE, bandwidth = "",
                 clusters = "") {
  data.frame(id = id, dataset = dataset, exog = exog, endog = endog,
             instruments = instruments, tested = tested, weighted = weighted,
             weight_type = weight_type, center = center, bandwidth = bandwidth,
             clusters = clusters, stringsAsFactors = FALSE)
}
hinstr <- paste(c("faminc", reg_cols), collapse = " ")
sc <- rbind(
  scen("sim_e1_i3", "sim", "const x3 x5", "x1", "z1 z2 x4", "", FALSE),
  scen("sim_e1_i3_w", "sim", "const x3 x5", "x1", "z1 z2 x4", "", TRUE),
  scen("sim_e1_i2_const_w", "sim", "const", "x1", "z1 z2", "x1", TRUE),
  scen("sim_e2_i4_x1", "sim", "const x3", "x1 x2", "z1 z2 x4 x5", "x1", FALSE),
  scen("sim_e2_i4_x2", "sim", "const x3", "x1 x2", "z1 z2 x4 x5", "x2", FALSE),
  scen("sim_e2_i4_all_rev", "sim", "const x3", "x1 x2", "z1 z2 x4 x5", "x2 x1", FALSE),
  scen("sim_e2_i4_x1_w", "sim", "const x3", "x1 x2", "z1 z2 x4 x5", "x1", TRUE),
  scen("sim_e3_i4_x2x3", "sim", "const", "x1 x2 x3", "z1 z2 x4 x5", "x2 x3", FALSE),
  scen("sim_e3_i4_x3_w", "sim", "const", "x1 x2 x3", "z1 z2 x4 x5", "x3", TRUE),
  scen("sim_e3_i4_x1x3_w", "sim", "const", "x1 x2 x3", "z1 z2 x4 x5", "x1 x3", TRUE),
  scen("sim_e2_i2_just_identified", "sim", "const x3 x4 x5", "x1 x2", "z1 z2", "x1", FALSE),
  scen("housing_hsng2", "housing", "const pcturban", "hsngval", hinstr, "", FALSE),
  # Other estimators of the covariance of the moment conditions
  scen("sim_e2_i4_x1_unadjusted", "sim", "const x3", "x1 x2", "z1 z2 x4 x5", "x1", FALSE,
       weight_type = "unadjusted"),
  scen("sim_e3_i4_x2x3_unadjusted", "sim", "const", "x1 x2 x3", "z1 z2 x4 x5", "x2 x3", FALSE,
       weight_type = "unadjusted"),
  scen("housing_hsng2_unadjusted", "housing", "const pcturban", "hsngval", hinstr, "", FALSE,
       weight_type = "unadjusted"),
  scen("sim_e2_i4_x1_center", "sim", "const x3", "x1 x2", "z1 z2 x4 x5", "x1", FALSE,
       center = TRUE),
  scen("sim_e3_i4_x3_center_w", "sim", "const", "x1 x2 x3", "z1 z2 x4 x5", "x3", TRUE,
       center = TRUE),
  scen("sim_e2_i4_x2_clustered", "sim", "const x3", "x1 x2", "z1 z2 x4 x5", "x2", FALSE,
       weight_type = "clustered", clusters = "cluster_id"),
  scen("sim_e1_i3_clustered_center", "sim", "const x3 x5", "x1", "z1 z2 x4", "", FALSE,
       weight_type = "clustered", center = TRUE, clusters = "cluster_id"),
  scen("housing_hsng2_clustered", "housing", "const pcturban", "hsngval", hinstr, "", FALSE,
       weight_type = "clustered", clusters = "division"),
  scen("sim_e2_i4_x1_kernel6", "sim", "const x3", "x1 x2", "z1 z2 x4 x5", "x1", FALSE,
       weight_type = "kernel", bandwidth = 6),
  scen("sim_e3_i4_x1x3_kernel3_center", "sim", "const", "x1 x2 x3", "z1 z2 x4 x5", "x1 x3", FALSE,
       weight_type = "kernel", center = TRUE, bandwidth = 3),
  scen("sim_e1_i3_kernel12_w", "sim", "const x3 x5", "x1", "z1 z2 x4", "", TRUE,
       weight_type = "kernel", bandwidth = 12)
)

words <- function(s) strsplit(s, " ", fixed = TRUE)[[1]]
res <- do.call(rbind, lapply(seq_len(nrow(sc)), function(i) {
  r <- sc[i, ]
  d <- if (r$dataset == "sim") sim else hs
  y <- if (r$dataset == "sim") d$y_robust else d$rent
  sw <- if (r$weighted) sqrt(d$weights / mean(d$weights)) else NULL
  wt <- wt_spec(r$weight_type, r$center,
                if (r$bandwidth == "") 0 else as.integer(r$bandwidth),
                if (r$clusters == "") NULL else as.integer(factor(d[[r$clusters]])))
  o <- c_stat(y, mat(d, words(r$exog)), mat(d, words(r$endog)),
              mat(d, words(r$instruments)), words(r$tested), sw, wt)
  cat(sprintf("%-30s C=%-12.8g J_e=%-12.8g J_c=%-12.8g J=%-12.8g df=%d\n",
              r$id, o$c, o$j_e, o$j_c, o$j_orig, o$df))
  data.frame(r, c_stat = o$c, j_e = o$j_e, j_c = o$j_c, hansen_j = o$j_orig,
             df = o$df, pvalue = pchisq(o$c, o$df, lower.tail = FALSE),
             stringsAsFactors = FALSE)
}))
write.csv(res, "c-stat-reference.csv", row.names = FALSE, quote = TRUE)
cat("wrote c-stat-reference.csv\n")
