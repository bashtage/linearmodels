# Reference values for the autocorrelation robust covariance (cov_type="ac") of
# the panel models, from the R package plm. They are read by
# ../test_ac_covariance.py.
#
# Run from this directory:   Rscript ac-covariance-reference.R
# Requires:                  plm (tested with 2.6.7 on R 4.6.1)
# Output:                    ac-covariance-panel.csv       the simulated panel
#                            ac-covariance-reference.csv   the reference values
#
# The estimator is the sandwich
#
#   (X'X)^-1 [ sum_i sum_{j=-L}^{L} w_j sum_t xi_it xi_i,t-j' ] (X'X)^-1,
#
# where xi_it = x_it e_it, w_j = 1 - |j| / (L + 1) are the weights of the Bartlett
# kernel and the sums of an entity are over the pairs of its observations that are j
# periods apart. This is plm::vcovNW(type = "HC0", maxlag = L), whose sums over the
# entities are not normalized by the number of observations of each entity. With
# L = 0 it is White's heteroskedasticity robust covariance. The covariance of the
# pooled regression and of the within regression (entity effects) are computed.
#
# The panel is unbalanced: each entity enters at a different period and leaves at a
# different period, and has no gaps. With gaps, plm (and sandwich::vcovPL) pairs the
# observations that are j calendar periods apart, while linearmodels pairs the
# observations that are j positions apart in the entity's observations. The two
# differ only if an entity is missing a period between two of its observations.
#
# The model is y = 1 + 0.5 x1 - 0.3 x2 + a_i + u_it, where a_i is a random effect
# and u_it is an AR(1) process with coefficient 0.5 and a variance that depends on
# x1.

suppressMessages(library(plm))

cat(R.version.string, "and plm", as.character(packageVersion("plm")), "\n")
set.seed(20261013)

# CSV files with "\n" line endings on every platform
write_csv <- function(x, path, ...) {
  con <- file(path, open = "wb")
  on.exit(close(con))
  write.csv(x, con, row.names = FALSE, eol = "\n", ...)
}

n <- 30
nt <- 10
rows <- list()
for (i in seq_len(n)) {
  first <- sample(0:3, 1)
  last <- nt - 1 - sample(0:3, 1)
  periods <- first:last
  m <- length(periods)
  x1 <- rnorm(m) + 0.4 * (i %% 5)
  x2 <- rnorm(m)
  u <- as.numeric(arima.sim(list(ar = 0.5), n = m)) * (0.5 + abs(x1) / 2)
  y <- 1 + 0.5 * x1 - 0.3 * x2 + 0.5 * rnorm(1) + u
  rows[[i]] <- data.frame(entity = i - 1L, time = periods, y = signif(y, 8),
                          x1 = signif(x1, 8), x2 = signif(x2, 8))
}
panel <- do.call(rbind, rows)
write_csv(panel, "ac-covariance-panel.csv")
# The reference is that of the data that the tests read
panel <- read.csv("ac-covariance-panel.csv")
cat("observations", nrow(panel), "entities", n, "\n")

pd <- pdata.frame(panel, index = c("entity", "time"))
models <- list(
  pooling = plm(y ~ x1 + x2, data = pd, model = "pooling"),
  within = plm(y ~ 0 + x1 + x2, data = pd, model = "within")
)
out <- list()
for (name in names(models)) {
  fit <- models[[name]]
  for (maxlag in 0:3) {
    v <- vcovNW(fit, type = "HC0", maxlag = maxlag)
    if (maxlag == 0) {
      # without lags it is White's heteroskedasticity robust covariance
      stopifnot(isTRUE(all.equal(v, vcovHC(fit, method = "white1", type = "HC0"),
                                 check.attributes = FALSE)))
    }
    terms <- rownames(v)
    for (a in seq_along(terms)) {
      for (b in seq_along(terms)) {
        out[[length(out) + 1]] <- data.frame(model = name, maxlag = maxlag,
                                             row = terms[a], column = terms[b],
                                             value = v[a, b])
      }
    }
  }
}
write_csv(do.call(rbind, out), "ac-covariance-reference.csv")
cat("wrote ac-covariance-panel.csv and ac-covariance-reference.csv\n")
