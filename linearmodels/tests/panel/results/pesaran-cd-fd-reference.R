# Reference values for PanelResults.pesaran_cd of FirstDifferenceOLS, which
# computes the Pesaran CD statistic from the residuals of the differenced
# equation. The script simulates the panels, so that nothing is shared with
# linearmodels, and uses the R package plm.
#
# Run from this directory:   Rscript pesaran-cd-fd-reference.R
# Requires:                  plm (tested with 2.6.7 on R 4.6.1)
# Output:                    pesaran-cd-fd-panels.csv     the simulated panels
#                            pesaran-cd-fd-reference.csv  the reference values
#                            (both are read by ../test_results_pesaran_cd.py)
#
# Definition (Pesaran 2021, Empirical Economics 60(1), 13-50):
#
#   e_it   residual of the estimated equation, for the periods in which the
#          entity has a residual
#   T_ij   number of periods in which e_it and e_jt are both observed
#   rho_ij sample correlation of e_it and e_jt over those T_ij periods
#   CD   = sum_{i<j} sqrt(T_ij) rho_ij / sqrt(M)
#
# where the sum, and M the number of its terms, are over the pairs with T_ij >= 2
# and a defined correlation. CD is asymptotically standard normal under the
# null of cross-sectional independence, so the p-value is 2 (1 - Phi(|CD|)).
# This is what plm::pcdtest(test = "cd") computes for the residuals of a panel
# model. The script computes it from the definition, in two ways, and checks
# that both equal pcdtest.
#
# The residuals are those of the first-difference estimator: the regression
# of the first difference of y on the first difference of x, without a
# constant. A difference exists for the period t if the entity is observed in
# the periods t and t - 1, and its residual is labelled t.
#
# The model = "fd" of plm differences the observations of an entity that follow
# each other in the data, so that it also differences across a period that is
# missing for the entity. FirstDifferenceOLS differences adjacent periods only.
# The differences here are therefore those of diff() of plm for panel series,
# which respects the time index, followed by a pooled regression of the
# differences. The pcdtest of plm cannot be used on an fd model ("argument 1 is
# not a vector"), but it can be used on this regression. If no entity is
# missing a period inside its observed range (case "unbalanced-window") the
# differences are those of model = "fd", and the script checks that the
# estimates agree.
#
# Cases. There are N entities and T periods (0, ..., T - 1) and
#
#   y = 1 + 0.5 x + a_i + e_it,   x = standard normal + 0.3 f_t,
#
# where a_i, f_t and the shocks e_it are standard normal, except where noted:
#   balanced-null          independent shocks
#   balanced-factor        e_it = u_it + 0.3 l_i f_t, with l_i = 1 + 0.5 N(0, 1)
#   balanced-random-walk   the shocks are independent random walks; their
#                          first differences, which this estimator uses, are
#                          independent
#   unbalanced-window      each entity is observed from a first period in 0-3
#                          to a last period in T - 4 to T - 1
#   unbalanced-gaps        like balanced-factor, with each observation missing
#                          with probability 0.3 (an entity needs at least 3
#                          observations to be kept)
#   unbalanced-sparse      each observation is missing with probability 0.5, so
#                          that most pairs of entities have fewer than two
#                          differences in common
# cd_levels is the statistic of the series in levels, y - x b, where b is
# the estimate. It is what pesaran_cd used for FirstDifferenceOLS before it used
# the residuals of the differenced equation, and is only here to show how much
# the choice matters.

suppressMessages(library(plm))

cat(R.version.string, "and plm", as.character(packageVersion("plm")), "\n")
set.seed(20261012)

# CSV files with "\n" line endings on every platform
write_csv <- function(x, path, ...) {
  con <- file(path, open = "wb")
  on.exit(close(con))
  write.csv(x, con, row.names = FALSE, eol = "\n", ...)
}

simulate <- function(case, n, nt, errors = "iid", loading = 0, observed = "all") {
  common <- rnorm(nt)
  load <- 1 + 0.5 * rnorm(n)
  x <- matrix(rnorm(n * nt), n, nt) + 0.3 * matrix(rep(common, each = n), n, nt)
  shock <- matrix(rnorm(n * nt), n, nt)
  if (errors == "random-walk") shock <- t(apply(shock, 1, cumsum))
  shock <- shock + loading * outer(load, common)
  effect <- matrix(rep(rnorm(n), nt), n, nt)
  y <- 1 + 0.5 * x + effect + shock
  keep <- matrix(TRUE, n, nt)
  if (observed == "window") {
    first <- 1 + floor(runif(n) * 4)
    last <- nt - floor(runif(n) * 4)
    keep <- outer(seq_len(n), seq_len(nt), function(i, t) t >= first[i] & t <= last[i])
  } else if (is.numeric(observed)) {
    keep <- matrix(runif(n * nt) >= observed, n, nt)
    keep <- keep & rowSums(keep) >= 3
  }
  d <- data.frame(case = case,
                  entity = rep(seq_len(n) - 1L, times = nt),
                  time = rep(seq_len(nt) - 1L, each = n),
                  y = signif(as.vector(y), 8), x = signif(as.vector(x), 8))
  d <- d[as.vector(keep), ]
  d[order(d$entity, d$time), ]
}

panels <- rbind(
  simulate("balanced-null", 40, 8),
  simulate("balanced-factor", 40, 8, loading = 0.3),
  simulate("balanced-random-walk", 40, 8, errors = "random-walk"),
  simulate("unbalanced-window", 40, 10, observed = "window"),
  simulate("unbalanced-gaps", 40, 10, loading = 0.3, observed = 0.3),
  simulate("unbalanced-sparse", 40, 8, observed = 0.5)
)
write_csv(panels, "pesaran-cd-fd-panels.csv")
# Everything below uses the file, so that the reference is that of the data
# that the tests read
panels <- read.csv("pesaran-cd-fd-panels.csv", stringsAsFactors = FALSE)

## ------------------------------------------------------------- CD test ---
# The CD statistic of the residuals e of the entities and periods (entity, time),
# from the N x N matrices of the correlations and of the numbers of common periods
cd_matrix <- function(entity, time, e) {
  ent <- sort(unique(entity))
  tim <- sort(unique(time))
  m <- matrix(NA_real_, length(tim), length(ent))
  m[cbind(match(time, tim), match(entity, ent))] <- e
  tij <- crossprod(!is.na(m))
  rho <- suppressWarnings(cor(m, use = "pairwise.complete.obs"))
  pair <- lower.tri(rho) & tij >= 2 & is.finite(rho)
  list(stat = sum(sqrt(tij[pair]) * rho[pair]) / sqrt(sum(pair)),
       npairs = sum(pair))
}

# The same with a loop over the pairs, which uses the common periods of each pair
cd_loop <- function(entity, time, e) {
  series <- split(setNames(e, time), entity)
  total <- 0
  npairs <- 0
  for (i in seq_len(length(series) - 1)) {
    for (j in (i + 1):length(series)) {
      both <- intersect(names(series[[i]]), names(series[[j]]))
      if (length(both) < 2) next
      rho <- suppressWarnings(cor(series[[i]][both], series[[j]][both]))
      if (!is.finite(rho)) next
      total <- total + sqrt(length(both)) * rho
      npairs <- npairs + 1
    }
  }
  list(stat = total / sqrt(npairs), npairs = npairs)
}

agree <- function(a, b, what, rtol = 1e-10) {
  if (abs(a - b) > rtol * max(1, abs(b))) {
    stop(sprintf("%s: %.15g vs %.15g", what, a, b))
  }
}

## --------------------------------------------------------- the cases ----
rows <- list()
for (case in unique(panels$case)) {
  d <- panels[panels$case == case, c("entity", "time", "y", "x")]
  pd <- pdata.frame(d, index = c("entity", "time"))

  # Differences of adjacent periods, and the regression of the differences
  fd <- data.frame(entity = d$entity, time = d$time,
                   dy = as.numeric(diff(pd$y)), dx = as.numeric(diff(pd$x)))
  fd <- fd[!is.na(fd$dy), ]
  fit <- plm(dy ~ dx - 1, data = pdata.frame(fd, index = c("entity", "time")),
             model = "pooling")
  e <- as.numeric(residuals(fit))
  agree(sum(e * fd$dx), 0, "the residuals are orthogonal to dx")

  by_hand <- cd_matrix(fd$entity, fd$time, e)
  looped <- cd_loop(fd$entity, fd$time, e)
  agree(by_hand$stat, looped$stat, paste(case, "matrix and loop"), 1e-12)
  stopifnot(by_hand$npairs == looped$npairs)
  # plm warns that it omits the pairs without two common periods, as the
  # definition does
  pcd <- suppressWarnings(pcdtest(fit, test = "cd"))
  agree(by_hand$stat, unname(pcd$statistic), paste(case, "pcdtest of plm"), 1e-10)

  # The levels, with the estimate of the differenced equation
  levels <- cd_matrix(d$entity, d$time, d$y - unname(coef(fit)) * d$x)

  consecutive <- all(tapply(d$time, d$entity, function(t) all(diff(t) == 1)))
  if (consecutive) {
    plm_fd <- plm(y ~ x - 1, data = pd, model = "fd")
    agree(unname(coef(plm_fd)), unname(coef(fit)), paste(case, "model = fd of plm"), 1e-10)
    stopifnot(length(residuals(plm_fd)) == nrow(fd))
    cat(sprintf("%-22s model = 'fd' of plm has this coefficient and %d residuals\n",
                case, nrow(fd)))
  }

  rows[[case]] <- data.frame(case = case, nobs = nrow(fd),
                             coef = unname(coef(fit)), npairs = by_hand$npairs,
                             cd_stat = by_hand$stat,
                             cd_pvalue = 2 * pnorm(-abs(by_hand$stat)),
                             cd_levels = levels$stat)
  cat(sprintf("%-22s nobs %4d  pairs %4d  CD %9.6f  p %.6g  levels %9.6f\n", case,
              nrow(fd), by_hand$npairs, by_hand$stat,
              2 * pnorm(-abs(by_hand$stat)), levels$stat))
}
write_csv(do.call(rbind, rows), "pesaran-cd-fd-reference.csv", quote = FALSE)
cat("wrote pesaran-cd-fd-panels.csv and pesaran-cd-fd-reference.csv\n")
