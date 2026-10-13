"""
The bandwidth of the Fama-MacBeth kernel covariance (cov_type="kernel")

The T cross-sectional estimates have autocovariances up to lag T - 1, so the bandwidth
has to be below T, the number of periods that are used. A bandwidth of T or more
raised "Length of w (10) is larger than the number of elements in z (9)" with the
Bartlett and Parzen kernels, when the covariance was first used and not when the model
was fit. The quadratic spectral kernel has no truncation. It returned a covariance
without an error, which was close to zero since the weights of all of the lags were
close to one and the deviations of the estimates from their average sum to zero.

Within the allowed bandwidths, the covariance is compared with the long-run covariance
of the cross-sectional estimates that is written out here, and with the covariance of
the average of each estimate in statsmodels (cov_hac), with the weights of each kernel.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats import sandwich_covariance as sw

from linearmodels.panel import FamaMacBeth
from linearmodels.panel.covariance import FamaMacBethCovariance
from linearmodels.shared.exceptions import MissingValueWarning

NENTITY = 25
NPERIOD = 9
BARTLETT = ("bartlett", "newey-west")
PARZEN = ("parzen", "gallant")
QS = ("qs", "quadratic-spectral", "andrews")
KERNELS = [*BARTLETT, *PARZEN, *QS]
DROPPED_PERIODS = [2, 5, 9]


def kernel_weights(kernel, bandwidth, nlags):
    """The weights of the lags 0, ..., nlags (Andrews, 1991)"""
    lag = np.arange(nlags + 1, dtype=float)
    if kernel in BARTLETT + PARZEN:
        # These kernels weight the lags up to the integer part of the bandwidth
        z = lag / (int(bandwidth) + 1)
        if kernel in BARTLETT:
            return np.where(z <= 1, 1 - z, 0.0)
        inner = 1 - 6 * z**2 + 6 * z**3
        return np.where(z <= 0.5, inner, np.where(z <= 1, 2 * (1 - z) ** 3, 0.0))
    if bandwidth == 0:
        return np.r_[1.0, np.zeros(nlags)]
    a = 6 * np.pi * lag[1:] / (5 * bandwidth)
    return np.r_[1.0, 3 * (np.sin(a) / a - np.cos(a)) / a**2]


def make_panel(nentity=NENTITY, nperiod=NPERIOD, seed=12):
    """Panel where the slopes are a random walk, so that the estimates are persistent"""
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["id", "year"]
    )
    nobs = index.shape[0]
    x1 = rg.standard_normal(nobs)
    x2 = rg.standard_normal(nobs)
    slopes = np.cumsum(0.3 * rg.standard_normal((nperiod, 2)), axis=0)
    period = index.codes[1]
    signal = 1 + slopes[period, 0] * x1 + slopes[period, 1] * x2
    y = pd.Series(signal + rg.standard_normal(nobs), index=index, name="y")
    x = pd.DataFrame({"const": 1.0, "x1": x1, "x2": x2}, index=index)
    return y, x


def cross_sections(y, x):
    """The estimates of the cross-sections that can be estimated, by period"""
    period = x.index.get_level_values(1)
    estimates = {}
    for value in period.unique():
        sel = period == value
        exog, dep = x.to_numpy()[sel], y.to_numpy()[sel]
        if exog.shape[0] < exog.shape[1] or np.linalg.matrix_rank(exog) < exog.shape[1]:
            continue
        estimates[value] = np.linalg.lstsq(exog, dep, rcond=None)[0]
    return pd.DataFrame(estimates, index=x.columns).T


def by_hand(estimates, kernel, bandwidth, debiased=True):
    """The kernel covariance of the T estimates, with the weights of the kernel"""
    values = estimates.to_numpy()
    nperiod = values.shape[0]
    dev = values - values.mean(axis=0)
    weights = kernel_weights(kernel, bandwidth, nperiod - 1)
    long_run = dev.T @ dev
    for lag in range(1, nperiod):
        gamma = dev[lag:].T @ dev[:-lag]
        long_run += weights[lag] * (gamma + gamma.T)
    return long_run / nperiod / (nperiod - int(debiased))


def statsmodels_variances(estimates, kernel, bandwidth):
    """The variance of the average of each estimate: a HAC regression on a constant"""
    nperiod = estimates.shape[0]
    out = []
    for name in estimates:
        ols = sm.OLS(estimates[name].to_numpy(), np.ones(nperiod)).fit()
        cov = sw.cov_hac(
            ols,
            nlags=nperiod - 1,
            weights_func=lambda n: kernel_weights(kernel, bandwidth, n),
            use_correction=False,
        )
        out.append(cov[0, 0])
    return np.array(out)


def dropped_periods_panel():
    """Panel where 3 of the 10 periods cannot be estimated"""
    y, x = make_panel(nentity=30, nperiod=10, seed=3)
    period = x.index.get_level_values(1)
    # x1 does not vary across the entities in these periods, and is then collinear
    # with the constant
    x.loc[np.isin(period, DROPPED_PERIODS), "x1"] = 1.5
    return y, x


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.mark.parametrize("bandwidth", [NPERIOD, NPERIOD + 1, 50, np.inf])
@pytest.mark.parametrize("kernel", KERNELS)
def test_bandwidth_not_below_the_number_of_periods_raises(panel, kernel, bandwidth):
    y, x = panel
    mod = FamaMacBeth(y, x)
    match = r"The bandwidth of the Fama-MacBeth kernel covariance, .*must be below the "
    match += rf"number of time periods that are used, {NPERIOD}\."
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="kernel", kernel=kernel, bandwidth=bandwidth)


@pytest.mark.parametrize("debiased", [True, False])
def test_error_is_raised_by_fit_for_the_options_of_fit(panel, debiased):
    y, x = panel
    with pytest.raises(ValueError, match=r"must be below the number of time periods"):
        FamaMacBeth(y, x).fit(cov_type="kernel", bandwidth=NPERIOD, debiased=debiased)


def test_error_message_names_the_bandwidth_and_the_largest_lag(panel):
    y, x = panel
    with pytest.raises(ValueError, match="must be below") as exc:
        FamaMacBeth(y, x).fit(cov_type="kernel", kernel="qs", bandwidth=50)
    message = str(exc.value)
    assert "Fama-MacBeth" in message
    assert ", 50, must be below" in message
    assert f"time periods that are used, {NPERIOD}." in message
    assert f"up to lag {NPERIOD - 1}" in message


def test_weighted_model_and_formula_raise(panel):
    y, x = panel
    rg = np.random.default_rng(5)
    weights = pd.Series(rg.uniform(0.5, 2.0, y.shape[0]), index=y.index)
    weighted = FamaMacBeth(y, x, weights=weights)
    with pytest.raises(ValueError, match=rf"time periods that are used, {NPERIOD}\."):
        weighted.fit(cov_type="kernel", bandwidth=NPERIOD)
    data = x.join(y)
    formula = FamaMacBeth.from_formula("y ~ x1 + x2", data)
    with pytest.raises(ValueError, match=rf"time periods that are used, {NPERIOD}\."):
        formula.fit(cov_type="kernel", bandwidth=NPERIOD)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
def test_periods_that_are_used_are_counted(kernel):
    # 7 of the 10 periods are used, so that a bandwidth of 7 to 9 is not allowed
    y, x = dropped_periods_panel()
    with pytest.warns(
        MissingValueWarning, match=r"3 of the cross-sectional regressions"
    ):
        mod = FamaMacBeth(y, x)
    for bandwidth in (7, 8, 9):
        with pytest.raises(ValueError, match=r"time periods that are used, 7\."):
            mod.fit(cov_type="kernel", kernel=kernel, bandwidth=bandwidth)
    res = mod.fit(cov_type="kernel", kernel=kernel, bandwidth=6)
    estimates = cross_sections(y, x)
    assert estimates.shape[0] == 7
    assert res.all_params.shape[0] == 10
    assert_allclose(res.cov, by_hand(estimates, kernel, 6), rtol=1e-8)


def test_estimator_checks_the_bandwidth_when_it_is_created(panel):
    y, x = panel
    res = FamaMacBeth(y, x).fit()
    args = (
        y.to_numpy()[:, None],
        x.to_numpy(),
        res.params.to_numpy()[:, None],
        res.all_params,
    )
    with pytest.raises(ValueError, match=r"must be below the number of time periods"):
        FamaMacBethCovariance(*args, bandwidth=NPERIOD)
    cov = FamaMacBethCovariance(*args, bandwidth=NPERIOD - 1)
    # The estimator is not debiased unless it is told to be
    expected = by_hand(cross_sections(y, x), "bartlett", NPERIOD - 1, debiased=False)
    assert_allclose(cov.cov, expected, rtol=1e-8)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
@pytest.mark.parametrize("bandwidth", [0, 1, 2, 4, NPERIOD - 1, NPERIOD - 0.5])
def test_allowed_bandwidths_against_by_hand_and_statsmodels(panel, kernel, bandwidth):
    y, x = panel
    estimates = cross_sections(y, x)
    mod = FamaMacBeth(y, x)
    res = mod.fit(cov_type="kernel", kernel=kernel, bandwidth=bandwidth, debiased=False)
    assert_allclose(res.cov, by_hand(estimates, kernel, bandwidth, False), rtol=1e-8)
    variances = statsmodels_variances(estimates, kernel, bandwidth)
    assert_allclose(np.diag(res.cov), variances, rtol=1e-8)
    debiased = mod.fit(cov_type="kernel", kernel=kernel, bandwidth=bandwidth)
    expected = by_hand(estimates, kernel, bandwidth, True)
    assert_allclose(debiased.cov, expected, rtol=1e-8)


@pytest.mark.parametrize("kernel", KERNELS)
def test_largest_allowed_bandwidth_with_every_kernel_name(panel, kernel):
    y, x = panel
    res = FamaMacBeth(y, x).fit(cov_type="kernel", kernel=kernel, bandwidth=NPERIOD - 1)
    expected = by_hand(cross_sections(y, x), kernel, NPERIOD - 1)
    assert_allclose(res.cov, expected, rtol=1e-8)
    std_errors = np.asarray(res.std_errors)
    positive = std_errors > 0
    n_positive = int(positive.sum())
    assert n_positive == std_errors.shape[0]


@pytest.mark.parametrize("nperiod", [6, 12])
@pytest.mark.parametrize("kernel", KERNELS)
def test_automatic_bandwidth_is_below_the_number_of_periods(kernel, nperiod):
    # With few periods the automatic bandwidth is often the largest allowed one, T - 1
    y, x = make_panel(nentity=30, nperiod=nperiod, seed=nperiod)
    res = FamaMacBeth(y, x).fit(cov_type="kernel", kernel=kernel)
    args = (
        y.to_numpy()[:, None],
        x.to_numpy(),
        res.params.to_numpy()[:, None],
        res.all_params,
    )
    bandwidth = FamaMacBethCovariance(*args, kernel=kernel).bandwidth
    assert bandwidth < nperiod
    expected = by_hand(cross_sections(y, x), kernel, bandwidth)
    assert_allclose(res.cov, expected, rtol=1e-8)


@pytest.mark.parametrize("cov_type", ["unadjusted", "homoskedastic", "robust"])
def test_other_cov_types_do_not_use_the_bandwidth(panel, cov_type):
    y, x = panel
    mod = FamaMacBeth(y, x)
    res = mod.fit(cov_type=cov_type, bandwidth=50)
    expected = by_hand(cross_sections(y, x), "bartlett", 0)
    assert_allclose(res.cov, expected, rtol=1e-8)
