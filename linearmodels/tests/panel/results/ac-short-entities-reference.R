# Reference values for the autocorrelation robust covariance (cov_type="ac") of
# the panel models when many entities have fewer observations than the bandwidth,
# from the R package plm. They are read by ../test_ac_covariance.py.
#
# Run from this directory:   Rscript ac-short-entities-reference.R
# Requires:                  plm (tested with 2.6.7 on R 4.6.1)
# Output:                    ac-short-entities-panel.csv       the simulated panel
#                            ac-short-entities-reference.csv   the reference values
#
# The estimator is the sandwich
#
#   (X'X)^-1 [ sum_i sum_{j=-L}^{L} K(j) sum_t xi_it xi_i,t-j' ] (X'X)^-1,
#
# where xi_it = x_it e_it and K(j) are the weights of the kernel. plm::vcovNW(type =
# "HC0", maxlag = L, wj = K) sums over the lags j = 1, ..., L, and over the pairs of
# observations of an entity that are j periods apart. An entity with T_i observations
# has no pairs beyond the lag T_i - 1, so the weights of its lags are those of the
# nominal bandwidth: the bandwidth is not reduced for the entities that have fewer
# observations than the bandwidth. The kernels are
#
#   bartlett  K(j) = 1 - j / (m + 1)                         (m is the bandwidth)
#   parzen    K(j) = 1 - 6 z^2 + 6 z^3, z <= 1/2             z = j / (m + 1)
#             K(j) = 2 (1 - z)^3, 1/2 < z <= 1
#   qs        K(j) = 3 (sin(a) / a - cos(a)) / a^2           a = 6 pi j / (5 m)
#
# The quadratic spectral kernel has no truncation, and L is the largest lag that any
# entity has, max(T_i) - 1. The pooled regression, the regression with entity effects
# (within) and the regression of the first differences (fd) are computed. The panel
# has no gaps, so that the positions of the observations of an entity and its calendar
# periods agree, which is where plm and linearmodels differ.
#
# The panel is unbalanced: 60 entities, with between 2 and 8 consecutive periods out
# of 12, most of them with 2 to 4. The model is y = 1 + 0.5 x1 - 0.3 x2 + a_i + u_it,
# where a_i is a random effect and u_it is an AR(1) process with coefficient 0.5 and a
# variance that depends on x1.

suppressMessages(library(plm))

cat(R.version.string, "and plm", as.character(packageVersion("plm")), "\n")
set.seed(20261013)

# CSV files with "\n" line endings on every platform
write_csv <- function(x, path, ...) {
  con <- file(path, open = "wb")
  on.exit(close(con))
  write.csv(x, con, row.names = FALSE, eol = "\n", ...)
}

n <- 60
nt <- 12
lengths <- sample(2:8, n, replace = TRUE, prob = c(5, 5, 4, 2, 1, 1, 1))
rows <- list()
for (i in seq_len(n)) {
  m <- lengths[i]
  first <- sample(0:(nt - m), 1)
  periods <- first:(first + m - 1)
  x1 <- rnorm(m) + 0.4 * (i %% 5)
  x2 <- rnorm(m)
  u <- as.numeric(arima.sim(list(ar = 0.5), n = m)) * (0.5 + abs(x1) / 2)
  y <- 1 + 0.5 * x1 - 0.3 * x2 + 0.5 * rnorm(1) + u
  rows[[i]] <- data.frame(entity = i - 1L, time = periods, y = signif(y, 8),
                          x1 = signif(x1, 8), x2 = signif(x2, 8))
}
panel <- do.call(rbind, rows)
write_csv(panel, "ac-short-entities-panel.csv")
# The reference is that of the data that the tests read
panel <- read.csv("ac-short-entities-panel.csv")
cat("observations", nrow(panel), "entities", n, "\n")
print(table(table(panel$entity)))

pd <- pdata.frame(panel, index = c("entity", "time"))
models <- list(
  pooling = plm(y ~ x1 + x2, data = pd, model = "pooling"),
  within = plm(y ~ 0 + x1 + x2, data = pd, model = "within"),
  fd = plm(y ~ 0 + x1 + x2, data = pd, model = "fd")
)

kernels <- list(
  bartlett = function(m) function(j, maxlag) 1 - j / (m + 1),
  parzen = function(m) function(j, maxlag) {
    z <- j / (m + 1)
    ifelse(z <= 0.5, 1 - 6 * z^2 + 6 * z^3, 2 * (1 - z)^3)
  },
  qs = function(m) function(j, maxlag) {
    a <- 6 * pi * j / (5 * m)
    3 * (sin(a) / a - cos(a)) / a^2
  }
)

max_length <- max(table(panel$entity))
out <- list()
for (name in names(models)) {
  fit <- models[[name]]
  for (kernel in names(kernels)) {
    # The largest bandwidth is above the lengths of all of the entities (7)
    for (bandwidth in c(1, 2, 3, 5, 8, 10)) {
      wj <- kernels[[kernel]](bandwidth)
      # The quadratic spectral kernel uses every lag that the data have
      maxlag <- if (kernel == "qs") max_length - 1 else bandwidth
      # plm needs a maximum lag that is below the number of time periods, which is
      # one less for the first differences
      stopifnot(maxlag < length(unique(panel$time)) - (name == "fd"))
      v <- vcovNW(fit, type = "HC0", maxlag = maxlag, wj = wj)
      terms <- rownames(v)
      for (a in seq_along(terms)) {
        for (b in seq_along(terms)) {
          out[[length(out) + 1]] <- data.frame(
            model = name, kernel = kernel, bandwidth = bandwidth,
            row = terms[a], column = terms[b], value = v[a, b]
          )
        }
      }
    }
  }
}
write_csv(do.call(rbind, out), "ac-short-entities-reference.csv")
cat("wrote ac-short-entities-panel.csv and ac-short-entities-reference.csv\n")
