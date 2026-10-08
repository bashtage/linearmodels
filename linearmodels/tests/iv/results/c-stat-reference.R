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
# GMM conventions follow linearmodels.iv.IVGMM and Stata's ivregress gmm
# (wmatrix(robust), two-step): the first step uses W = (Z'Z/n)^{-1}, the
# second step uses W = S^{-1} where S = (1/n) sum (z_i e_i)(z_i e_i)' is the
# uncentred heteroskedasticity-robust estimate built from first-step residuals,
# and J = n g'Wg uses that same W. Weighted models are estimated on data
# multiplied by the square root of the weights.
#
# Before it writes anything the script checks that this implementation
# reproduces numbers produced by Stata that are already used in the tests:
#   * ivregress gmm on the housing data (stata-iv-housing-results.txt),
#     which is an overidentified model, and
#   * estat endogenous / estat overid on the simulated data
#     (see ../test_postestimation.py).
# Equivalent Stata code for a scenario in the grid is, e.g.,
#   ivreg2 y x3 x5 (x1 = z1 z2 x4), gmm2s robust orthog(...)   [not run here]

suppressMessages(library(foreign))

## ---------------------------------------------------------------- GMM ----
gmm_beta <- function(y, X, Z, W) {
  A <- crossprod(X, Z)
  drop(solve(A %*% W %*% t(A), A %*% W %*% crossprod(Z, y)))
}

gmm_J <- function(y, X, Z, b, W) {
  n <- nrow(Z)
  g <- crossprod(Z, y - X %*% b) / n
  drop(n * crossprod(g, W %*% g))
}

two_step <- function(y, X, Z) {
  n <- nrow(Z)
  b1 <- gmm_beta(y, X, Z, solve(crossprod(Z) / n))
  e1 <- drop(y - X %*% b1)
  S <- crossprod(Z * e1) / n
  W <- solve(S)
  b2 <- gmm_beta(y, X, Z, W)
  list(b = b2, S = S, W = W, J = gmm_J(y, X, Z, b2, W))
}

## Returns C, J_e, J_c, the Hansen J of the original model and the df.
c_stat <- function(y, exog, endog, instr, tested, sw = NULL, check = TRUE) {
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
  full <- two_step(y, Xe, Ze)
  ke <- ncol(exog)
  nt <- length(tested)
  keep <- c(seq_len(ke), ke + nt + seq_len(ncol(instr)))
  Wc <- solve(full$S[keep, keep])
  bc <- gmm_beta(y, X, Z, Wc)
  Jc <- gmm_J(y, X, Z, bc, Wc)
  out <- list(c = full$J - Jc, j_e = full$J, j_c = Jc,
              j_orig = two_step(y, X, Z)$J, df = nt)
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
    # (4) the order of the instruments and of the tested variables is irrelevant
    perm <- rev(seq_len(ncol(instr)))
    again <- c_stat(y, exog, endog[, rev(colnames(endog)), drop = FALSE],
                    instr[, perm, drop = FALSE], rev(tested), check = FALSE)
    stopifnot(abs(again$c - out$c) <= 1e-8 * max(1, abs(out$c)))
  }
  out
}

near <- function(a, b, rtol, what) {
  if (abs(a - b) > rtol * abs(b)) {
    stop(sprintf("calibration failed for %s: %.10g vs %.10g", what, a, b))
  }
  cat(sprintf("  ok  %-52s %.10g (Stata %.10g)\n", what, a, b))
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
# ivregress gmm rent pcturban (hsngval = faminc i.region), wmatrix(robust)
# from stata-iv-housing-results.txt: gmm-robust-asymptotic
Xh <- mat(hs, c("hsngval", "pcturban", "const"))
Zh <- mat(hs, c("pcturban", "const", "faminc", reg_cols))
fit <- two_step(hs$rent, Xh, Zh)
near(fit$b[1], 0.00146432787, 1e-6, "housing GMM b[hsngval]")
near(fit$b[2], 0.76154815601, 1e-6, "housing GMM b[pcturban]")
near(fit$b[3], 112.12271295, 1e-6, "housing GMM b[_cons]")
near(fit$J, 6.8364006463, 1e-6, "housing Hansen J, chi2(3)")
near(fit$W[1, 1], 0.00001443637, 1e-5, "housing weight matrix W[pcturban, pcturban]")
near(fit$W[3, 3], 2.531069e-10, 1e-5, "housing weight matrix W[faminc, faminc]")

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
sc <- data.frame(
  id = c("sim_e1_i3", "sim_e1_i3_w", "sim_e1_i2_const_w",
         "sim_e2_i4_x1", "sim_e2_i4_x2", "sim_e2_i4_all_rev", "sim_e2_i4_x1_w",
         "sim_e3_i4_x2x3", "sim_e3_i4_x3_w", "sim_e3_i4_x1x3_w",
         "sim_e2_i2_just_identified", "housing_hsng2"),
  dataset = c(rep("sim", 11), "housing"),
  exog = c("const x3 x5", "const x3 x5", "const",
           "const x3", "const x3", "const x3", "const x3",
           "const", "const", "const",
           "const x3 x4 x5", "const pcturban"),
  endog = c("x1", "x1", "x1",
            "x1 x2", "x1 x2", "x1 x2", "x1 x2",
            "x1 x2 x3", "x1 x2 x3", "x1 x2 x3",
            "x1 x2", "hsngval"),
  instruments = c("z1 z2 x4", "z1 z2 x4", "z1 z2",
                  "z1 z2 x4 x5", "z1 z2 x4 x5", "z1 z2 x4 x5", "z1 z2 x4 x5",
                  "z1 z2 x4 x5", "z1 z2 x4 x5", "z1 z2 x4 x5",
                  "z1 z2", paste(c("faminc", reg_cols), collapse = " ")),
  tested = c("", "", "x1",
             "x1", "x2", "x2 x1", "x1",
             "x2 x3", "x3", "x1 x3",
             "x1", ""),
  weighted = c(FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE,
               FALSE, TRUE, TRUE, FALSE, FALSE),
  stringsAsFactors = FALSE
)

words <- function(s) strsplit(s, " ", fixed = TRUE)[[1]]
res <- do.call(rbind, lapply(seq_len(nrow(sc)), function(i) {
  r <- sc[i, ]
  d <- if (r$dataset == "sim") sim else hs
  y <- if (r$dataset == "sim") d$y_robust else d$rent
  sw <- if (r$weighted) sqrt(d$weights / mean(d$weights)) else NULL
  o <- c_stat(y, mat(d, words(r$exog)), mat(d, words(r$endog)),
              mat(d, words(r$instruments)), words(r$tested), sw)
  cat(sprintf("%-28s C=%-12.8g J_e=%-12.8g J_c=%-12.8g J=%-12.8g df=%d\n",
              r$id, o$c, o$j_e, o$j_c, o$j_orig, o$df))
  data.frame(r, c_stat = o$c, j_e = o$j_e, j_c = o$j_c, hansen_j = o$j_orig,
             df = o$df, pvalue = pchisq(o$c, o$df, lower.tail = FALSE),
             stringsAsFactors = FALSE)
}))
write.csv(res, "c-stat-reference.csv", row.names = FALSE, quote = TRUE)
cat("wrote c-stat-reference.csv\n")
