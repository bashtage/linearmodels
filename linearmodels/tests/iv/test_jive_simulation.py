"""
Simulations that IVJIVE works as expected

The tests in test_jive.py show that the estimates and the covariance estimators
are what they are defined to be. These show that they behave as they should
statistically: the confidence intervals have close to their nominal coverage
when the covariance estimator is appropriate for the errors, and do not when it
is not, and JIVE has less of the many instrument bias of 2SLS. The simulations
use a fixed seed so the results do not change.
"""

import numpy as np
import pytest
from scipy import stats

from linearmodels.iv import IV2SLS, IVJIVE

CRITICAL = stats.norm.ppf(0.975)
REPS = 400


def coverage(draw, cov_types, seed=20261009):
    """Fraction of 95% confidence intervals for the slope that contain it"""
    rng = np.random.default_rng(seed)
    covered = dict.fromkeys(cov_types, 0)
    for _ in range(REPS):
        y, exog, endog, z, clusters, beta = draw(rng)
        for cov_type in cov_types:
            options = {"clusters": clusters} if cov_type == "clustered" else {}
            res = IVJIVE(y, exog, endog, z).fit(cov_type=cov_type, **options)
            error = abs(res.params.iloc[-1] - beta)
            covered[cov_type] += error <= CRITICAL * res.std_errors.iloc[-1]
    return {cov_type: count / REPS for cov_type, count in covered.items()}


def homoskedastic(rng, nobs=300, ninstr=3, beta=1.0):
    z = rng.standard_normal((nobs, ninstr))
    v = rng.standard_normal(nobs)
    x = z @ np.full(ninstr, 0.6) + v
    y = 0.5 + beta * x + 0.5 * v + np.sqrt(0.75) * rng.standard_normal(nobs)
    return y, np.ones((nobs, 1)), x[:, None], z, None, beta


def heteroskedastic(rng, nobs=300, ninstr=3, beta=1.0):
    z = rng.standard_normal((nobs, ninstr))
    v = rng.standard_normal(nobs)
    x = z @ np.full(ninstr, 0.6) + v
    # The variance of the error depends on the instruments
    sd = 0.3 + 1.5 * np.abs(z[:, 0]) ** 1.5
    y = 0.5 + beta * x + 0.5 * v + sd * rng.standard_normal(nobs)
    return y, np.ones((nobs, 1)), x[:, None], z, None, beta


def cluster_correlated(rng, nobs=300, ninstr=3, beta=1.0, ngroups=30):
    clusters = np.repeat(np.arange(ngroups), nobs // ngroups)
    # The instruments and the error both have a component that is common to
    # all of the observations in a cluster
    z = (
        rng.standard_normal((nobs, ninstr))
        + rng.standard_normal((ngroups, ninstr))[clusters]
    )
    v = rng.standard_normal(nobs)
    x = z @ np.full(ninstr, 0.6) + v
    common = rng.standard_normal(ngroups)[clusters]
    y = 0.5 + beta * x + 0.5 * v + 1.5 * common + rng.standard_normal(nobs)
    return y, np.ones((nobs, 1)), x[:, None], z, clusters, beta


def test_coverage_with_homoskedastic_errors():
    # Both the covariance estimators that assume homoskedasticity and those
    # that do not have the right coverage
    rates = coverage(homoskedastic, ["unadjusted", "robust"])
    assert 0.91 < rates["unadjusted"] < 0.98
    assert 0.91 < rates["robust"] < 0.98


def test_coverage_with_heteroskedastic_errors():
    # The robust covariance is right and the unadjusted one is not
    rates = coverage(heteroskedastic, ["unadjusted", "robust"])
    assert 0.91 < rates["robust"] < 0.98
    assert rates["unadjusted"] < 0.90


def test_coverage_with_cluster_correlated_errors():
    # The clustered covariance is about right, with only 30 clusters, and the
    # robust one, which ignores the correlation, is far from it
    rates = coverage(cluster_correlated, ["robust", "clustered"])
    assert 0.88 < rates["clustered"] < 0.98
    assert rates["robust"] < 0.80
    assert rates["clustered"] - rates["robust"] > 0.12


def test_many_weak_instruments():
    # 2SLS is biased towards OLS when there are many weak instruments. JIVE has
    # little bias but a larger spread, as in the literature on JIVE
    rng = np.random.default_rng(20261008)
    nobs, ninstr, pi, beta, rho = 200, 30, 0.1, 1.0, 0.5
    tsls, jive = [], []
    for _ in range(300):
        z = rng.standard_normal((nobs, ninstr))
        v = rng.standard_normal(nobs)
        x = z @ np.full(ninstr, pi) + v
        y = beta * x + rho * v + np.sqrt(1 - rho**2) * rng.standard_normal(nobs)
        exog = np.ones((nobs, 1))
        tsls.append(IV2SLS(y, exog, x[:, None], z).fit().params.iloc[-1] - beta)
        jive.append(IVJIVE(y, exog, x[:, None], z).fit().params.iloc[-1] - beta)

    def iqr(values):
        return np.subtract(*np.percentile(values, [75, 25]))

    assert np.median(tsls) > 0.1
    assert abs(np.median(jive)) < 0.05
    assert iqr(jive) > iqr(tsls)


@pytest.mark.parametrize("ninstr", [2, 10])
def test_jive_and_2sls_are_close_with_few_instruments(ninstr):
    # With a small number of strong instruments the leverages are small, and so
    # JIVE is close to 2SLS
    rng = np.random.default_rng(20261010)
    nobs = 2000
    z = rng.standard_normal((nobs, ninstr))
    v = rng.standard_normal(nobs)
    x = z @ np.full(ninstr, 0.8) + v
    y = 1.0 + x + 0.5 * v + rng.standard_normal(nobs)
    jive = IVJIVE(y, np.ones((nobs, 1)), x[:, None], z).fit()
    tsls = IV2SLS(y, np.ones((nobs, 1)), x[:, None], z).fit()
    assert abs(jive.params.iloc[-1] - tsls.params.iloc[-1]) < 0.02
    assert abs(jive.std_errors.iloc[-1] / tsls.std_errors.iloc[-1] - 1) < 0.05
