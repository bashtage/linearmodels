"""
The bandwidth of the Driscoll-Kraay covariance (cov_type="kernel")

The sums of the scores of the T time periods have autocovariances up to lag T - 1,
so the bandwidth has to be below T. A bandwidth of T or more raised "Length of w (9)
is larger than the number of elements in z (8)" with the Bartlett and Parzen kernels,
when the covariance was first used. The quadratic spectral kernel has no truncation. It
returned a covariance without an error, which was close to zero since the weights of
all of the lags were close to one and the scores sum to zero.

Within the allowed bandwidths, the covariance is compared with the Driscoll-Kraay
covariance of statsmodels (cov_nw_groupsum) with the weights of each kernel, which are
written out here.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats import sandwich_covariance as sw

from linearmodels.panel import (
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.covariance import DriscollKraay

NENTITY = 25
NPERIOD = 9
KERNELS = ["bartlett", "parzen", "qs"]


def kernel_weights(kernel, bandwidth, nlags):
    """The weights of the lags 0, ..., nlags (Andrews, 1991)"""
    lag = np.arange(nlags + 1, dtype=float)
    if kernel in ("bartlett", "parzen"):
        # These kernels weight the lags up to the integer part of the bandwidth
        z = lag / (int(bandwidth) + 1)
        if kernel == "bartlett":
            return np.where(z <= 1, 1 - z, 0.0)
        inner = 1 - 6 * z**2 + 6 * z**3
        return np.where(z <= 0.5, inner, np.where(z <= 1, 2 * (1 - z) ** 3, 0.0))
    if bandwidth == 0:
        return np.r_[1.0, np.zeros(nlags)]
    a = 6 * np.pi * lag[1:] / (5 * bandwidth)
    return np.r_[1.0, 3 * (np.sin(a) / a - np.cos(a)) / a**2]


@pytest.fixture(scope="module")
def frame():
    rg = np.random.default_rng(12)
    index = pd.MultiIndex.from_product(
        [np.arange(NENTITY), np.arange(NPERIOD)], names=["id", "year"]
    )
    nobs = index.shape[0]
    common = rg.standard_normal(NPERIOD)
    x1 = rg.standard_normal(nobs) + 0.5 * common[index.codes[1]]
    x2 = rg.standard_normal(nobs)
    effect = rg.standard_normal(NENTITY)[index.codes[0]]
    shock = rg.standard_normal(nobs) + common[index.codes[1]]
    y = 1 + 0.5 * x1 - 0.3 * x2 + effect + shock
    frame = pd.DataFrame({"y": y, "const": 1.0, "x1": x1, "x2": x2}, index=index)
    return frame


def pooled(frame, **options):
    return PooledOLS(frame["y"], frame[["const", "x1", "x2"]]).fit(**options)


MODELS = {
    "PooledOLS": lambda f: PooledOLS(f["y"], f[["const", "x1", "x2"]]),
    "PanelOLS": lambda f: PanelOLS(f["y"], f[["x1", "x2"]], entity_effects=True),
    "RandomEffects": lambda f: RandomEffects(f["y"], f[["const", "x1", "x2"]]),
}


@pytest.mark.parametrize("bandwidth", [NPERIOD, NPERIOD + 1, 50, np.inf])
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("model", MODELS)
def test_bandwidth_not_below_the_number_of_periods_raises(
    frame, model, kernel, bandwidth
):
    mod = MODELS[model](frame)
    match = r"The bandwidth of the Driscoll-Kraay covariance, .*must be below the "
    match += rf"number of time periods, {NPERIOD}\."
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="kernel", kernel=kernel, bandwidth=bandwidth)


@pytest.mark.parametrize("cov_type", ["kernel", "driscoll-kraay", "dk"])
def test_cov_type_aliases_raise(frame, cov_type):
    with pytest.raises(ValueError, match=r"must be below the number of time periods"):
        pooled(frame, cov_type=cov_type, bandwidth=50)


def test_error_message_names_the_number_of_periods_and_the_largest_lag(frame):
    with pytest.raises(ValueError, match="must be below") as exc:
        pooled(frame, cov_type="kernel", kernel="qs", bandwidth=50)
    message = str(exc.value)
    assert "50" in message
    assert f"the number of time periods, {NPERIOD}." in message
    assert f"up to lag {NPERIOD - 1}" in message


def test_estimator_checks_the_bandwidth_when_it_is_created(frame):
    mod = PooledOLS(frame["y"], frame[["const", "x1", "x2"]])
    res = mod.fit()
    args = (
        mod.dependent.values2d,
        mod.exog.values2d,
        np.asarray(res.params)[:, None],
        mod.dependent.entity_ids,
        mod.dependent.time_ids,
    )
    with pytest.raises(ValueError, match=r"must be below the number of time periods"):
        DriscollKraay(*args, bandwidth=NPERIOD)
    cov = DriscollKraay(*args, bandwidth=NPERIOD - 1)
    assert np.all(np.isfinite(cov.cov))


def test_first_difference_has_one_period_less(frame):
    # The differences are observed in NPERIOD - 1 periods
    mod = FirstDifferenceOLS(frame["y"], frame[["x1", "x2"]])
    with pytest.raises(ValueError, match=rf"number of time periods, {NPERIOD - 1}\."):
        mod.fit(cov_type="kernel", bandwidth=NPERIOD - 1)
    res = mod.fit(cov_type="kernel", bandwidth=NPERIOD - 2)
    assert np.all(np.isfinite(res.std_errors))


def test_single_period_default_bandwidth_raises():
    # The rule-of-thumb bandwidth of one period is 1
    index = pd.MultiIndex.from_product([np.arange(30), [0]], names=["id", "year"])
    rg = np.random.default_rng(1)
    x = rg.standard_normal(30)
    frame = pd.DataFrame(
        {"y": x + rg.standard_normal(30), "const": 1.0, "x": x}, index=index
    )
    mod = PooledOLS(frame["y"], frame[["const", "x"]])
    with pytest.raises(ValueError, match=r"The default bandwidth of the .* 1, must be"):
        mod.fit(cov_type="kernel")


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("bandwidth", [0, 1, 2, 4, NPERIOD - 1, NPERIOD - 0.5])
def test_allowed_bandwidths_against_statsmodels(frame, kernel, bandwidth):
    res = pooled(
        frame, cov_type="kernel", kernel=kernel, bandwidth=bandwidth, debiased=False
    )
    ols = sm.OLS(frame["y"], frame[["const", "x1", "x2"]]).fit()
    time = frame.index.codes[1]
    # The Bartlett and Parzen kernels use the integer part of the bandwidth
    nlags = NPERIOD - 1 if kernel == "qs" else int(bandwidth)
    cov = sw.cov_nw_groupsum(
        ols,
        nlags,
        time,
        weights_func=lambda n: kernel_weights(kernel, bandwidth, n),
        use_correction=False,
    )
    assert_allclose(res.cov.to_numpy(), cov, rtol=1e-8)
    assert np.all(np.isfinite(res.std_errors))


@pytest.mark.parametrize("kernel", KERNELS)
def test_largest_allowed_bandwidth_with_effects_and_the_first_differences(
    frame, kernel
):
    # The covariances of the other models use the same estimator
    mod = PanelOLS(frame["y"], frame[["x1", "x2"]], entity_effects=True)
    res = mod.fit(cov_type="kernel", kernel=kernel, bandwidth=NPERIOD - 1)
    assert np.all(np.isfinite(res.std_errors))
    assert np.all(res.std_errors > 0)
