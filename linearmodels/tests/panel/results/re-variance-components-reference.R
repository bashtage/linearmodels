# Reference values for the variance components of RandomEffects, from the R
# package plm, which are read by ../test_random_effects_variance.py.
#
# Run from this directory:   Rscript re-variance-components-reference.R
# Requires:                  plm (tested with 2.6.7 on R 4.6.1)
# Input:                     ../../../datasets/wage_panel/wage_panel.csv.bz2
# Output:                    re-variance-components-reference.csv
#
# The model is the Swamy-Arora random effects model of plm,
# plm(..., model = "random"), whose estimator of the variance components uses the
# degrees of freedom of the regressions that estimate them:
#
#   sigma2_idios = RSS(within) / (n - N - K_w)
#   sigma2_id    = RSS(between) / (N - K_b) - sigma2_idios / Tbar     (balanced)
#
# where K_w is the number of coefficients of the within regression, which are the
# regressors that vary within the entities and are not collinear after the
# entities' means are removed, and K_b is the number of coefficients of the
# between regression, including the constant. Time-invariant regressors (black,
# hisp, educ) are not in the within regression, and the dummy variables of the
# years are collinear with the constant in the between regression of a balanced
# panel. The degrees of freedom do not count them. For an unbalanced panel plm
# uses the estimator of Baltagi and Chang (1994) (dfcor = 3), which is the
# small_sample option of RandomEffects.fit. That branch of ercomp counts the
# columns of the model matrix of the within regression for K_w, so that it counts
# a regressor that is aliased (exper with the year dummies), and this is not the
# df.residual of the within model of plm. The unbalanced cases have no aliased
# regressor.
#
# Stata's xtreg, re uses the same degrees of freedom: for the Cornwell and Rupert
# wage panel (A. C. Cameron's log of psidextract.dta), xtreg lwage exp exp2 wks ed,
# re reports sigma_e = .15220316, which is also the sigma_e of xtreg lwage exp exp2
# wks, fe, where ed is omitted because of collinearity (n - N - 3 degrees of
# freedom). The same model has sigma_e = 0.1522032 in plm.
#
# Cases (the unbalanced cases drop observations by a fixed rule):
#   balanced-notebook         the model of the panel_examples notebook: year dummies,
#                             time-invariant regressors, and exper, which is
#                             collinear with the year dummies in the within regression
#   balanced-time-invariant   time-invariant regressors without year dummies
#   balanced-time-varying     regressors that all vary within the entities
#   unbalanced-year-dummies   year dummies and time-invariant regressors, without exper
#   unbalanced-time-invariant as balanced-time-invariant

suppressMessages(library(plm))

cat(R.version.string, "and plm", as.character(packageVersion("plm")), "\n")

# CSV files with "\n" line endings on every platform
write_csv <- function(x, path, ...) {
  con <- file(path, open = "wb")
  on.exit(close(con))
  write.csv(x, con, row.names = FALSE, eol = "\n", ...)
}

wages <- read.csv(bzfile("../../../datasets/wage_panel/wage_panel.csv.bz2"))

# The observations that the unbalanced cases drop: 588 of 4,360, so that each
# entity has between 6 and 8 observations and the means of the year dummies differ
# between the entities
drop <- ((wages$nr * 7 + wages$year * 13) %% 5 == 0) & ((wages$nr + wages$year) %% 3 != 0)
cat("unbalanced cases drop", sum(drop), "of", nrow(wages), "observations\n")

formulas <- list(
  notebook = lwage ~ black + hisp + exper + expersq + married + educ + union +
    factor(year),
  `year-dummies` = lwage ~ black + hisp + expersq + married + educ + union +
    factor(year),
  `time-invariant` = lwage ~ black + hisp + exper + expersq + married + educ + union,
  `time-varying` = lwage ~ expersq + married + union
)
cases <- list(
  list(name = "balanced-notebook", data = wages, formula = "notebook"),
  list(name = "balanced-time-invariant", data = wages, formula = "time-invariant"),
  list(name = "balanced-time-varying", data = wages, formula = "time-varying"),
  list(name = "unbalanced-year-dummies", data = wages[!drop, ],
       formula = "year-dummies"),
  list(name = "unbalanced-time-invariant", data = wages[!drop, ],
       formula = "time-invariant")
)

rows <- list()
for (case in cases) {
  pd <- pdata.frame(case$data, index = c("nr", "year"))
  fit <- plm(formulas[[case$formula]], data = pd, model = "random")
  components <- ercomp(fit)
  theta <- as.numeric(components$theta)
  est <- coef(fit)
  se <- sqrt(diag(vcov(fit)))
  within <- plm(formulas[[case$formula]], data = pd, model = "within")
  between <- plm(formulas[[case$formula]], data = pd, model = "between")
  rows[[case$name]] <- rbind(
    data.frame(case = case$name, quantity = "nobs", value = nobs(fit)),
    data.frame(case = case$name, quantity = "df_within", value = df.residual(within)),
    data.frame(case = case$name, quantity = "df_between", value = df.residual(between)),
    data.frame(case = case$name, quantity = "sigma2_idios",
               value = unname(components$sigma2["idios"])),
    data.frame(case = case$name, quantity = "sigma2_id",
               value = unname(components$sigma2["id"])),
    data.frame(case = case$name, quantity = "theta_mean", value = mean(theta)),
    data.frame(case = case$name, quantity = paste0("coef", seq_along(est) - 1),
               value = unname(est)),
    data.frame(case = case$name, quantity = paste0("se", seq_along(est) - 1),
               value = unname(se))
  )
  cat(sprintf("%-27s nobs %5d  df within %5d  df between %4d  idios %.7f  id %.7f\n",
              case$name, nobs(fit), df.residual(within), df.residual(between),
              components$sigma2["idios"], components$sigma2["id"]))
}
write_csv(do.call(rbind, rows), "re-variance-components-reference.csv")
cat("wrote re-variance-components-reference.csv\n")
