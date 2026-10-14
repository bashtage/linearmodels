"""
The automatic bandwidth of the kernel estimators needs enough observations

The rule of Newey and West (1994) estimates the long-run variance of the series with
its autocovariances up to a pilot lag n = ceil(4 (t / 100) ** a), where a is 2/9 for
the Bartlett kernel, 4/25 for the Parzen kernel and 2/25 for the quadratic spectral
kernel. If n >= t - 1 these are all of the autocovariances, whose sum is
(sum x) ** 2 / t. It is zero for a series whose mean is zero, such as the scores of a
regression or the deviations of the Fama-MacBeth estimates from their average, and the
rule divides by it. ``kernel_optimal_bandwidth`` raised "cannot convert float infinity
to integer" (an OverflowError) or "cannot convert float NaN to integer" for some
series, and returned t - 1, or another bandwidth that was only rounding error, for the
others. The estimators that use it failed in the same way, for example
``FamaMacBeth.fit(cov_type="kernel")`` with 3 to 5 periods.

It now raises a ValueError that gives the number of observations and says that the
bandwidth has to be passed, with fewer than 4 observations for the Bartlett kernel, 5
for the Parzen kernel and 6 for the quadratic spectral kernel. It also raises a
ValueError if the bandwidth is not finite although there are enough observations: the
series is zero, the estimate of its long-run variance is exactly zero or it has values
that are not finite.

The boundary is computed here from the closed form of the pilot lag, and the
bandwidth above the boundary is compared with the rule that is written out with plain
loops. The data of the boundary tests are integer series whose sum is exactly zero in
floating point, so that the old code fails with them.
"""

import math
import re
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.asset_pricing.covariance import (
    KernelCovariance as FactorKernelCovariance,
    KernelWeight as FactorKernelWeight,
)
from linearmodels.iv import (
    IV2SLS,
    IVGMM,
    IVGMMCUE,
    IVJIVE,
    IVLIML,
    kleibergen_paap,
    kleibergen_paap_f,
)
from linearmodels.iv.absorbing import AbsorbingLS
from linearmodels.iv.covariance import kernel_optimal_bandwidth
from linearmodels.iv.gmm import KernelWeightMatrix as IVKernelWeightMatrix
from linearmodels.panel import FamaMacBeth, PooledOLS
from linearmodels.panel.covariance import FamaMacBethCovariance
from linearmodels.shared.exceptions import DefaultBandwidthWarning, MissingValueWarning
from linearmodels.system import IV3SLS, SUR, IVSystemGMM
from linearmodels.system.gmm import KernelWeightMatrix as SystemKernelWeightMatrix

BARTLETT = ("bartlett", "newey-west")
PARZEN = ("parzen", "gallant")
QS = ("qs", "quadratic-spectral", "andrews")
KERNELS = [*BARTLETT, *PARZEN, *QS]
CANONICAL = ["bartlett", "parzen", "qs"]
# q, the constant c and the exponent a of the pilot lag, from Newey and West (1994)
RULE = {
    **dict.fromkeys(BARTLETT, (1, 1.1447, 2 / 9)),
    **dict.fromkeys(PARZEN, (2, 2.6614, 4 / 25)),
    **dict.fromkeys(QS, (2, 1.3221, 2 / 25)),
}
# The fewest observations that the documentation gives: the pilot lags 2, 3 and 4 are
# below t - 1 from t = 4, 5 and 6
FEWEST = {
    **dict.fromkeys(BARTLETT, 4),
    **dict.fromkeys(PARZEN, 5),
    **dict.fromkeys(QS, 6),
}


def pilot_lag(nobs, kernel):
    """The number of lags in the pilot estimates, from the closed form"""
    return math.ceil(4 * (nobs / 100) ** RULE[kernel][2])


def fewest_observations(kernel):
    """The smallest number of observations with a pilot lag below nobs - 1"""
    nobs = 1
    while pilot_lag(nobs, kernel) >= nobs - 1:
        nobs += 1
    return nobs


def pilot_sums(series, kernel):
    """s_0 and s_q of the pilot estimates of Newey and West (1994), with loops"""
    x = [float(value) for value in np.asarray(series).ravel()]
    nobs = len(x)
    q = RULE[kernel][0]
    lags = pilot_lag(nobs, kernel)
    # sigma_j = sum_{i=j}^{t-1} x_i x_{i-j} / t, s_0 = sigma_0 + 2 sum_j sigma_j and
    # s_q = 2 sum_j j ** q sigma_j
    sigma = [
        sum(x[i] * x[i - j] for i in range(j, nobs)) / nobs for j in range(lags + 1)
    ]
    s0 = sigma[0] + 2 * sum(sigma[1:])
    sq = 2 * sum(j**q * sigma[j] for j in range(1, lags + 1))
    return s0, sq


def newey_west_rule(series, kernel):
    """The rule of Newey and West (1994) before it is rounded and capped"""
    nobs = np.asarray(series).size
    q, c, _ = RULE[kernel]
    s0, sq = pilot_sums(series, kernel)
    gamma = c * ((sq / s0) ** 2) ** (1 / (2 * q + 1))
    return gamma * nobs ** (1 / (2 * q + 1))


def newey_west(series, kernel):
    """The bandwidth of Newey and West (1994): the ceiling of the rule, at most t - 1"""
    m = newey_west_rule(series, kernel)
    # Away from an integer, so that rounding cannot change the ceiling
    distance = abs(m - round(m))
    assert m == 0 or distance > 1e-6
    return min(math.ceil(m), np.asarray(series).size - 1)


def too_few_message(kernel, nobs):
    """The message of kernel_optimal_bandwidth for too few observations"""
    return (
        f"The automatic bandwidth of the {kernel!r} kernel needs at least "
        f"{FEWEST[kernel]} observations, but there are {nobs}. Pass the bandwidth "
        "explicitly."
    )


def zero_sum_series(nobs):
    """Integers whose sum is exactly zero, so the old rule divides by zero or noise"""
    x = np.array([(-1) ** i * (i + 1) for i in range(nobs)], dtype=float)
    x[-1] = -x[:-1].sum()
    return x


def white_noise(nobs):
    return np.random.default_rng(3).standard_normal(nobs)


def autoregression(nobs, rho, seed):
    e = np.random.default_rng(seed).standard_normal(nobs)
    x = np.empty(nobs)
    x[0] = e[0]
    for i in range(1, nobs):
        x[i] = rho * x[i - 1] + e[i]
    return x


SERIES = {
    "white noise": white_noise,
    "positive autocorrelation": lambda nobs: autoregression(nobs, 0.7, 5),
    "negative autocorrelation": lambda nobs: autoregression(nobs, -0.5, 8),
}


# ---------------------------------------------------------------------------------
# kernel_optimal_bandwidth
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", KERNELS)
def test_too_few_observations_raise(kernel):
    first = fewest_observations(kernel)
    assert first == FEWEST[kernel]
    for nobs in range(1, first):
        x = zero_sum_series(nobs)
        total = x.sum()
        assert total == 0
        with warnings.catch_warnings():
            # The error is all that is reported, not the warnings of the arithmetic
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError, match=re.escape(too_few_message(kernel, nobs))
            ):
                kernel_optimal_bandwidth(x, kernel)


@pytest.mark.parametrize("kernel", KERNELS)
def test_no_observations_raise(kernel):
    with pytest.raises(
        ValueError, match=r"needs at least \d observations, but there are 0"
    ):
        kernel_optimal_bandwidth(np.empty(0), kernel)


@pytest.mark.parametrize("kernel", KERNELS)
def test_message_names_the_kernel_the_observations_and_the_bandwidth(kernel):
    first = fewest_observations(kernel)
    with pytest.raises(ValueError, match="bandwidth") as exc:
        kernel_optimal_bandwidth(zero_sum_series(first - 1), kernel)
    message = str(exc.value)
    quoted = repr(kernel)
    assert quoted in message
    assert f"needs at least {first} observations" in message
    assert f"there are {first - 1}" in message
    assert "Pass the bandwidth explicitly" in message


@pytest.mark.parametrize("extra", [0, 1, 2, 26, 116])
@pytest.mark.parametrize("series", list(SERIES))
@pytest.mark.parametrize("kernel", KERNELS)
def test_bandwidth_above_the_boundary_is_the_newey_west_rule(kernel, series, extra):
    nobs = fewest_observations(kernel) + extra
    x = SERIES[series](nobs)
    result = kernel_optimal_bandwidth(x, kernel)
    is_integer = isinstance(result, int)
    expected = newey_west(x, kernel)
    assert is_integer
    assert result == expected


# Series with the fewest observations whose rule is much larger than nobs - 1
CAPPED = {
    "bartlett": [-1.0, -1.0, 1.0, 0.5],
    "parzen": [1.0, 1.0, -1.0, 1.0, 0.5],
    "qs": [-1.0, -1.0, -1.0, 1.0, 0.5, 1.0],
}


@pytest.mark.parametrize("kernel", CANONICAL)
def test_bandwidth_is_capped_at_the_number_of_observations_less_one(kernel):
    x = np.array(CAPPED[kernel])
    nobs = x.shape[0]
    first = fewest_observations(kernel)
    rule = newey_west_rule(x, kernel)
    result = kernel_optimal_bandwidth(x, kernel)
    assert nobs == first
    assert rule > 1.4 * (nobs - 1)
    assert result == nobs - 1


# The digits of pi with signs, which have a long-run variance that is not 0
INTEGERS = np.array([3, -1, 4, 1, -5, 9, 2, -6, 5, 3, -5, 8])


@pytest.mark.parametrize("kernel", KERNELS)
def test_column_vector_and_integers_give_the_same_bandwidth(kernel):
    nobs = fewest_observations(kernel) + 3
    x = INTEGERS[:nobs]
    expected = newey_west(x, kernel)
    result = kernel_optimal_bandwidth(x, kernel)
    column = kernel_optimal_bandwidth(x[:, None].astype(float), kernel)
    assert result == expected
    assert column == expected


# The rule of sandwich::bwNeweyWest (sandwich 3.1.3 in R 4.6.1) for the series of
# sandwich_series, which has 100 observations. It is the same rule and the same
# constants. The pilot lag of sandwich is floor(4 (T / 100) ** a), the one here is the
# ceiling, and they are the same if that number is an integer, as it is for every kernel
# with 100 observations (it is 4):
#
#   e <- sin((1:100)^2)
#   x <- as.numeric(stats::filter(e, 0.6, method = "recursive"))
#   bwNeweyWest(matrix(x), kernel = "Bartlett", prewhite = 0, weights = 1)
SANDWICH_RULE = {
    **dict.fromkeys(BARTLETT, 6.1644899939539499),
    **dict.fromkeys(PARZEN, 10.174290345382101),
    **dict.fromkeys(QS, 5.0542681542157037),
}


def sandwich_series():
    e = np.sin(np.arange(1, 101, dtype=float) ** 2)
    x = np.empty(100)
    x[0] = e[0]
    for i in range(1, 100):
        x[i] = 0.6 * x[i - 1] + e[i]
    return x


@pytest.mark.parametrize("kernel", KERNELS)
def test_rule_is_that_of_r_sandwich(kernel):
    x = sandwich_series()
    expected = SANDWICH_RULE[kernel]
    rule = newey_west_rule(x, kernel)
    result = kernel_optimal_bandwidth(x, kernel)
    ceiling = math.ceil(expected)
    assert_allclose(rule, expected, rtol=1e-12)
    assert result == ceiling


@pytest.mark.parametrize("nobs", [1, 3, 20])
def test_unknown_kernel_is_reported_before_the_number_of_observations(nobs):
    with pytest.raises(ValueError, match="Unknown kernel: 'unknown'"):
        kernel_optimal_bandwidth(zero_sum_series(nobs), "unknown")


# Series with enough observations whose bandwidth is not finite. The second has an
# estimate of the long-run variance that is exactly zero: with 8 observations the
# autocovariances are multiples of 1/8, so that their sum is exactly 0, and the
# weighted sum of the autocovariances is not zero
EXACTLY_ZERO = "exactly zero long-run variance"
NOT_FINITE = {
    "zeros": np.zeros(8),
    EXACTLY_ZERO: np.array([1.0, -1.0, 0, 0, 0, 0, 0, 0]),
    "nan": np.array([1.0, np.nan, 2.0, 0.5, 1.5, -1.0, 0.25, 3.0]),
    "infinity": np.array([1.0, np.inf, 2.0, 0.5, 1.5, -1.0, 0.25, 3.0]),
    "overflow of the squares": np.array(
        [1e200, -3e200, 2e200, 1e200, -1e200, 5e199, 1e200, 1.0]
    ),
    "underflow of the squares": np.array(
        [1e-200, -3e-200, 2e-200, 1e-200, -1e-200, 5e-201, 1e-200, 1e-200]
    ),
}


@pytest.mark.parametrize("case", list(NOT_FINITE))
@pytest.mark.parametrize("kernel", KERNELS)
def test_bandwidth_that_is_not_finite_raises(kernel, case):
    x = NOT_FINITE[case]
    first = fewest_observations(kernel)
    assert x.shape[0] >= first
    if case == EXACTLY_ZERO:
        # The rule divides a number that is not zero by zero, which is exact here
        s0, sq = pilot_sums(x, kernel)
        assert s0 == 0
        assert sq != 0
    match = rf"The automatic bandwidth of the {kernel!r} kernel is not finite\."
    with warnings.catch_warnings():
        # The error is all that is reported, not the warnings of the arithmetic
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=match):
            kernel_optimal_bandwidth(x, kernel)


def test_message_of_a_bandwidth_that_is_not_finite():
    with pytest.raises(ValueError, match="is not finite") as exc:
        kernel_optimal_bandwidth(np.zeros(8), "bartlett")
    message = str(exc.value)
    assert "the 'bartlett' kernel" in message
    assert "long-run variance" in message
    assert "Pass the bandwidth explicitly" in message


# ---------------------------------------------------------------------------------
# FamaMacBeth
# ---------------------------------------------------------------------------------
def fama_macbeth_panel(nperiod, nentity=25):
    """Panel where the intercept and the slope are random walks, so that the
    cross-sectional estimates are autocorrelated"""
    rg = np.random.default_rng(100 + nperiod)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    nobs = index.shape[0]
    x = rg.standard_normal(nobs)
    period = index.codes[1]
    intercept = np.cumsum(0.3 * rg.standard_normal(nperiod))
    slope = 1 + np.cumsum(0.5 * rg.standard_normal(nperiod))
    y = intercept[period] + slope[period] * x + rg.standard_normal(nobs)
    exog = pd.DataFrame({"const": 1.0, "x": x}, index=index)
    return pd.Series(y, index=index, name="y"), exog


def cross_sections(y, exog):
    """The estimates of the cross-sections that can be estimated, one row each"""
    time = exog.index.get_level_values(1)
    rows = []
    for period in time.unique():
        selected = time == period
        regressors = exog.to_numpy()[selected]
        if np.linalg.matrix_rank(regressors) < regressors.shape[1]:
            continue
        rows.append(np.linalg.lstsq(regressors, y.to_numpy()[selected], rcond=None)[0])
    return np.array(rows)


def fama_macbeth_bandwidth(estimates, kernel):
    """The automatic bandwidth: the rule for the sum of the standardized deviations"""
    deviations = estimates - estimates.mean(axis=0)
    return newey_west((deviations / deviations.std(axis=0)).sum(axis=1), kernel)


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


def fama_macbeth_cov(estimates, kernel, bandwidth):
    """The debiased kernel covariance of the average of the estimates"""
    deviations = estimates - estimates.mean(axis=0)
    nperiod = deviations.shape[0]
    weights = kernel_weights(kernel, bandwidth, nperiod - 1)
    long_run = deviations.T @ deviations
    for lag in range(1, nperiod):
        gamma = deviations[lag:].T @ deviations[:-lag]
        long_run += weights[lag] * (gamma + gamma.T)
    return long_run / nperiod / (nperiod - 1)


def fama_macbeth_message(kernel, nperiod):
    return (
        "The automatic bandwidth of the Fama-MacBeth kernel covariance with the "
        f"{kernel!r} kernel needs at least {FEWEST[kernel]} time periods, but the "
        f"number of time periods that are used is {nperiod}. Pass the bandwidth to "
        f"fit, which must be below {nperiod}."
    )


FEW_PERIODS = [
    (kernel, nperiod)
    for kernel in KERNELS
    for nperiod in range(2, fewest_observations(kernel))
]
EXPLICIT_BANDWIDTHS = [
    (kernel, fewest_observations(kernel) - 1, bandwidth)
    for kernel in KERNELS
    for bandwidth in range(fewest_observations(kernel) - 1)
]


@pytest.mark.parametrize(("kernel", "nperiod"), FEW_PERIODS)
def test_fama_macbeth_with_too_few_periods_raises_when_the_model_is_fit(
    kernel, nperiod
):
    first = fewest_observations(kernel)
    assert FEWEST[kernel] == first
    y, x = fama_macbeth_panel(nperiod)
    mod = FamaMacBeth(y, x)
    match = re.escape(fama_macbeth_message(kernel, nperiod))
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="kernel", kernel=kernel)


def test_fama_macbeth_names_the_default_kernel():
    y, x = fama_macbeth_panel(3)
    match = r"with the 'newey-west' kernel needs at least 4 time periods"
    with pytest.raises(ValueError, match=match):
        FamaMacBeth(y, x).fit(cov_type="kernel")


@pytest.mark.parametrize("extra", [0, 1, 2])
@pytest.mark.parametrize("kernel", KERNELS)
def test_fama_macbeth_automatic_bandwidth_with_the_fewest_periods(kernel, extra):
    nperiod = fewest_observations(kernel) + extra
    y, x = fama_macbeth_panel(nperiod)
    estimates = cross_sections(y, x)
    bandwidth = fama_macbeth_bandwidth(estimates, kernel)
    res = FamaMacBeth(y, x).fit(cov_type="kernel", kernel=kernel)
    assert_allclose(res.cov, fama_macbeth_cov(estimates, kernel, bandwidth), rtol=1e-8)
    explicit = FamaMacBeth(y, x).fit(
        cov_type="kernel", kernel=kernel, bandwidth=bandwidth
    )
    assert_allclose(res.cov, explicit.cov, rtol=1e-12)


@pytest.mark.parametrize(("kernel", "nperiod", "bandwidth"), EXPLICIT_BANDWIDTHS)
def test_fama_macbeth_explicit_bandwidth_works_with_the_periods_that_raise(
    kernel, nperiod, bandwidth
):
    y, x = fama_macbeth_panel(nperiod)
    res = FamaMacBeth(y, x).fit(cov_type="kernel", kernel=kernel, bandwidth=bandwidth)
    expected = fama_macbeth_cov(cross_sections(y, x), kernel, bandwidth)
    assert_allclose(res.cov, expected, rtol=1e-8)


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust"])
def test_fama_macbeth_other_covariances_do_not_need_the_bandwidth(cov_type):
    y, x = fama_macbeth_panel(3)
    res = FamaMacBeth(y, x).fit(cov_type=cov_type)
    expected = fama_macbeth_cov(cross_sections(y, x), "bartlett", 0)
    assert_allclose(res.cov, expected, rtol=1e-8)


def test_fama_macbeth_counts_the_periods_that_are_used():
    # 4 of the 8 periods cannot be estimated: x does not vary across the entities
    # in them, and is collinear with the constant
    y, x = fama_macbeth_panel(8, nentity=30)
    period = x.index.get_level_values(1)
    x.loc[np.isin(period, [1, 3, 4, 6]), "x"] = 1.5
    with pytest.warns(MissingValueWarning, match="4 of the cross-sectional regress"):
        mod = FamaMacBeth(y, x)
    estimates = cross_sections(y, x)
    assert estimates.shape[0] == 4
    # The Bartlett kernel needs 4 periods
    bandwidth = fama_macbeth_bandwidth(estimates, "bartlett")
    res = mod.fit(cov_type="kernel", kernel="bartlett")
    assert_allclose(
        res.cov, fama_macbeth_cov(estimates, "bartlett", bandwidth), rtol=1e-8
    )
    for kernel in (*PARZEN, *QS):
        with pytest.raises(
            ValueError, match=re.escape(fama_macbeth_message(kernel, 4))
        ):
            mod.fit(cov_type="kernel", kernel=kernel)


def test_fama_macbeth_unknown_kernel_raises_when_the_model_is_fit():
    y, x = fama_macbeth_panel(12)
    mod = FamaMacBeth(y, x)
    with pytest.raises(
        ValueError, match=r"Unknown kernel: 'parzan'\. Did you mean 'parzen'"
    ):
        mod.fit(cov_type="kernel", kernel="parzan")


def estimates_and_params(estimates):
    all_params = pd.DataFrame(estimates, columns=["const", "x"])
    return all_params, all_params.mean(axis=0).to_numpy()[:, None]


@pytest.mark.parametrize(("kernel", "nperiod"), FEW_PERIODS)
def test_fama_macbeth_covariance_raises_when_it_is_created(kernel, nperiod):
    estimates = np.random.default_rng(nperiod).standard_normal((nperiod, 2))
    all_params, params = estimates_and_params(estimates)
    y, x = np.zeros((10, 1)), np.ones((10, 2))
    match = re.escape(fama_macbeth_message(kernel, nperiod))
    with pytest.raises(ValueError, match=match):
        FamaMacBethCovariance(y, x, params, all_params, kernel=kernel)
    # The bandwidth is not needed if it is given
    cov = FamaMacBethCovariance(y, x, params, all_params, kernel=kernel, bandwidth=1)
    expected = fama_macbeth_cov(estimates, kernel, 1) * (nperiod - 1) / nperiod
    assert_allclose(cov.cov, expected, rtol=1e-10)


@pytest.mark.parametrize("nperiod", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("kernel", CANONICAL)
def test_driscoll_kraay_default_bandwidth_is_another_rule_and_does_not_change(
    kernel, nperiod
):
    # The default of the kernel covariance of the other panel models is the integer
    # part of 4 (T / 100) ** (2 / 9), which is below T whatever T is, and it does not
    # need the periods that the rule of the other estimators needs
    y, x = fama_macbeth_panel(nperiod)
    default = PooledOLS(y, x).fit(cov_type="kernel", kernel=kernel)
    bandwidth = math.floor(4 * (nperiod / 100) ** (2 / 9))
    explicit = PooledOLS(y, x).fit(
        cov_type="kernel", kernel=kernel, bandwidth=bandwidth
    )
    assert_allclose(default.cov, explicit.cov, rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("kernel", KERNELS)
def test_fama_macbeth_estimates_that_do_not_vary_have_no_finite_bandwidth(kernel):
    # The standard deviation of the estimates is 0, and the deviations are divided by it
    all_params, params = estimates_and_params(np.tile([1.0, 2.0], (8, 1)))
    y, x = np.zeros((10, 1)), np.ones((10, 2))
    match = rf"The automatic bandwidth of the {kernel!r} kernel is not finite\."
    with pytest.raises(ValueError, match=match):
        FamaMacBethCovariance(y, x, params, all_params, kernel=kernel)


# ---------------------------------------------------------------------------------
# IV models and the first stage rank tests
# ---------------------------------------------------------------------------------
def iv_data(nobs, seed=11):
    """Data of a model with a constant, an endogenous regressor and two instruments"""
    rg = np.random.default_rng(seed)
    z1, z2, exogenous, noise_x, noise_y = rg.standard_normal((5, nobs))
    endogenous = 0.8 * z1 + 0.3 * z2 + noise_x
    y = 1 + endogenous + exogenous + noise_y
    return pd.DataFrame(
        {
            "y": y,
            "xn": endogenous,
            "xe": exogenous,
            "z1": z1,
            "z2": z2,
            "const": np.ones(nobs),
        }
    )


def iv_args(data, instruments=("z1",), constant=True):
    exog = data[["const"]] if constant else None
    return data.y, exog, data[["xn"]], data[list(instruments)]


def moment_series(z, eps, center=False):
    """The series that the automatic bandwidth of the moment conditions uses"""
    ze = z * eps[:, None]
    if center:
        ze = ze - ze.mean(axis=0)
    return (ze / ze.std(axis=0)).sum(axis=1)


def two_stage_least_squares(data):
    """The 2SLS estimates and the weighted data of the model with the instrument z1"""
    x = data[["const", "xn"]].to_numpy()
    z = data[["const", "z1"]].to_numpy()
    y = data.y.to_numpy()
    xhat = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    params = np.linalg.solve(xhat.T @ x, xhat.T @ y)
    return x, z, y, xhat, params


def gmm_step_one(data):
    """The first step estimates of GMM, with the weight matrix inv(z'z / n)"""
    x = data[["const", "xn"]].to_numpy()
    z = data[["const", "z1", "z2"]].to_numpy()
    y = data.y.to_numpy()
    weight = np.linalg.inv(z.T @ z / z.shape[0])
    xpz = x.T @ z
    params = np.linalg.solve(xpz @ weight @ xpz.T, xpz @ weight @ (z.T @ y))
    return x, z, y, params


def gmm_step_two(data):
    """The second step estimates with the robust weight matrix"""
    x, z, y, params = gmm_step_one(data)
    ze = z * (y - x @ params)[:, None]
    weight = np.linalg.inv(ze.T @ ze / z.shape[0])
    xpz = x.T @ z
    return x, z, y, np.linalg.solve(xpz @ weight @ xpz.T, xpz @ weight @ (z.T @ y))


def check_covariance_is_finite(res):
    values = np.asarray(res.cov)
    finite = bool(np.isfinite(values).all())
    assert finite


IV_MODELS = [IV2SLS, IVLIML, IVJIVE]


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", IV_MODELS)
def test_iv_models_raise_when_fit_with_too_few_observations(model, kernel):
    nobs = fewest_observations(kernel) - 1
    # Without the constant, see the explicit bandwidth test below
    mod = model(*iv_args(iv_data(nobs), constant=False))
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit(cov_type="kernel", kernel=kernel)


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", IV_MODELS)
def test_iv_models_with_the_fewest_observations(model, kernel):
    nobs = fewest_observations(kernel)
    res = model(*iv_args(iv_data(nobs))).fit(cov_type="kernel", kernel=kernel)
    bandwidth = res.cov_config["bandwidth"]
    assert 0 <= bandwidth <= nobs - 1
    check_covariance_is_finite(res)


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", IV_MODELS)
def test_iv_models_explicit_bandwidth_works_with_the_observations_that_raise(
    model, kernel
):
    nobs = fewest_observations(kernel) - 1
    # The kappa of LIML, which the IV models compute, needs at least 1 + the number
    # of endogenous variables observations more than instruments. With the constant,
    # the 3 observations of the Bartlett kernel leave 1 for 2, and a matrix with an
    # eigenvalue that is rounding error is inverted: kappa was NaN on macOS arm64 and
    # rounding error elsewhere. Without the constant, the model is well posed.
    mod = model(*iv_args(iv_data(nobs), constant=False))
    res = mod.fit(cov_type="kernel", kernel=kernel, bandwidth=1)
    assert res.cov_config["bandwidth"] == 1
    check_covariance_is_finite(res)


@pytest.mark.parametrize("extra", [0, 1, 2, 10])
@pytest.mark.parametrize("kernel", CANONICAL)
def test_iv2sls_automatic_bandwidth_is_the_rule_for_the_scores(kernel, extra):
    nobs = fewest_observations(kernel) + extra
    data = iv_data(nobs)
    res = IV2SLS(*iv_args(data)).fit(cov_type="kernel", kernel=kernel)
    x, _, y, xhat, params = two_stage_least_squares(data)
    assert_allclose(res.params, params, rtol=1e-8)
    # The scores of the regressors that are not the constant: xhat * eps
    scores = xhat[:, 1] * (y - x @ params)
    expected = newey_west(scores, kernel)
    assert res.cov_config["bandwidth"] == expected


@pytest.mark.parametrize("kernel", CANONICAL)
def test_absorbing_ls_raises_when_fit_with_too_few_observations(kernel):
    nobs = fewest_observations(kernel) - 1
    data = iv_data(nobs)
    absorb = pd.DataFrame({"group": pd.Series(np.arange(nobs) % 2).astype("category")})
    mod = AbsorbingLS(data.y, data[["xe"]], absorb=absorb)
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit(cov_type="kernel", kernel=kernel)


@pytest.mark.parametrize("kernel", CANONICAL)
def test_absorbing_ls_with_the_fewest_observations(kernel):
    nobs = fewest_observations(kernel)
    data = iv_data(nobs)
    absorb = pd.DataFrame({"group": pd.Series(np.arange(nobs) % 2).astype("category")})
    mod = AbsorbingLS(data.y, data[["xe"]], absorb=absorb)
    res = mod.fit(cov_type="kernel", kernel=kernel)
    assert 0 <= res.cov_config["bandwidth"] <= nobs - 1
    explicit = mod.fit(cov_type="kernel", kernel=kernel, bandwidth=1)
    assert explicit.cov_config["bandwidth"] == 1


# IVGMM and IVGMMCUE ---------------------------------------------------------------
@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
def test_gmm_weights_raise_when_fit_with_too_few_observations(model, kernel):
    nobs = fewest_observations(kernel) - 1
    mod = model(
        *iv_args(iv_data(nobs)), weight_type="kernel", kernel=kernel, optimal_bw=True
    )
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit()


@pytest.mark.parametrize("kernel", CANONICAL)
def test_gmm_kernel_covariance_raises_when_fit_with_too_few_observations(kernel):
    nobs = fewest_observations(kernel) - 1
    mod = IVGMM(*iv_args(iv_data(nobs)))
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit(cov_type="kernel", kernel=kernel, optimal_bw=True)


@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
def test_gmm_default_bandwidth_raises_when_fit_with_too_few_observations(model):
    # The default is to estimate the bandwidth, and fit warns about the change
    mod = model(*iv_args(iv_data(3)), weight_type="kernel")
    with pytest.warns(DefaultBandwidthWarning):
        with pytest.raises(ValueError, match=re.escape(too_few_message("bartlett", 3))):
            mod.fit()


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
def test_gmm_explicit_bandwidth_works_with_the_observations_that_raise(model, kernel):
    nobs = fewest_observations(kernel) - 1
    args = iv_args(iv_data(nobs))
    explicit = model(*args, weight_type="kernel", kernel=kernel, bandwidth=1).fit()
    assert explicit.weight_config["bandwidth"] == 1
    # The bandwidth of 7.0, which the weight matrix can be restored to
    restored = model(*args, weight_type="kernel", kernel=kernel, optimal_bw=False).fit()
    assert restored.weight_config["bandwidth"] == nobs - 2


@pytest.mark.parametrize("extra", [0, 2, 8])
@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
def test_gmm_weight_bandwidth_is_the_rule_for_the_first_step_residuals(
    model, kernel, extra
):
    # Three moment conditions, the constant and two instruments, need one more
    # observation than the kernel does
    nobs = fewest_observations(kernel) + extra + 1
    data = iv_data(nobs)
    mod = model(
        *iv_args(data, ("z1", "z2")),
        weight_type="kernel",
        kernel=kernel,
        optimal_bw=True,
    )
    res = mod.fit()
    x, z, y, params = gmm_step_one(data)
    # IVGMMCUE centers the moment conditions by default
    series = moment_series(z, y - x @ params, center=model is IVGMMCUE)
    expected = newey_west(series, kernel)
    assert res.weight_config["bandwidth"] == expected


@pytest.mark.parametrize("extra", [0, 2, 8])
@pytest.mark.parametrize("kernel", CANONICAL)
def test_gmm_covariance_bandwidth_is_the_rule_for_the_final_residuals(kernel, extra):
    nobs = fewest_observations(kernel) + extra + 1
    data = iv_data(nobs)
    mod = IVGMM(*iv_args(data, ("z1", "z2")))
    res = mod.fit(cov_type="kernel", kernel=kernel, optimal_bw=True)
    x, z, y, params = gmm_step_two(data)
    assert_allclose(res.params, params, rtol=1e-8)
    expected = newey_west(moment_series(z, y - x @ params), kernel)
    assert res.cov_config["bandwidth"] == expected


@pytest.mark.parametrize("kernel", CANONICAL)
def test_kernel_weight_matrix_raises_with_too_few_observations(kernel):
    nobs = fewest_observations(kernel) - 1
    rg = np.random.default_rng(nobs)
    x, z, eps = (
        rg.standard_normal((nobs, 2)),
        rg.standard_normal((nobs, 3)),
        rg.standard_normal((nobs, 1)),
    )
    weight = IVKernelWeightMatrix(kernel=kernel, optimal_bw=True)
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        weight.weight_matrix(x, z, eps)
    # The bandwidth of 7.0 and an explicit bandwidth do not need the observations
    for options, expected in (({"optimal_bw": False}, nobs - 2), ({"bandwidth": 1}, 1)):
        restored = IVKernelWeightMatrix(kernel=kernel, **options)
        restored.weight_matrix(x, z, eps)
        bandwidth = restored.bandwidth
        assert bandwidth == expected


# The first stage rank tests --------------------------------------------------------
def rank_test_arrays(nobs):
    data = iv_data(nobs)
    return (
        data[["xn"]].to_numpy(),
        data[["z1", "z2"]].to_numpy(),
        data[["const"]].to_numpy(),
    )


# The rank test needs more observations than the other estimators before it gets to
# the bandwidth: with the Bartlett kernel and 3 observations it is not defined
RANK_TEST_FEW = [("parzen", 4), ("qs", 4), ("qs", 5)]
RANK_TEST_ENOUGH = [("bartlett", 4), ("parzen", 5), ("qs", 6)]


@pytest.mark.parametrize("statistic", [kleibergen_paap, kleibergen_paap_f])
@pytest.mark.parametrize(("kernel", "nobs"), RANK_TEST_FEW)
def test_rank_tests_raise_with_too_few_observations(statistic, kernel, nobs):
    endog, instr, exog = rank_test_arrays(nobs)
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        statistic(endog, instr, exog, "kernel", {"kernel": kernel})


@pytest.mark.parametrize("statistic", [kleibergen_paap, kleibergen_paap_f])
@pytest.mark.parametrize(("kernel", "nobs"), RANK_TEST_FEW)
def test_rank_tests_with_an_explicit_bandwidth_work_with_too_few_observations(
    statistic, kernel, nobs
):
    endog, instr, exog = rank_test_arrays(nobs)
    config = {"kernel": kernel, "bandwidth": 1}
    result = statistic(endog, instr, exog, "kernel", config)
    # kleibergen_paap returns a test statistic and kleibergen_paap_f a float
    value = float(getattr(result, "stat", result))
    finite = np.isfinite(value)
    assert finite


@pytest.mark.parametrize(("kernel", "nobs"), RANK_TEST_ENOUGH)
def test_rank_tests_with_enough_observations(kernel, nobs):
    endog, instr, exog = rank_test_arrays(nobs)
    stat = kleibergen_paap_f(endog, instr, exog, "kernel", {"kernel": kernel})
    finite = np.isfinite(stat)
    assert finite


# ---------------------------------------------------------------------------------
# System models
# ---------------------------------------------------------------------------------
def system_equations(nobs, endogenous=False):
    first, second = iv_data(nobs, seed=11), iv_data(nobs, seed=111)
    if endogenous:
        return {
            f"eq{i}": {
                "dependent": data.y,
                "exog": data[["const"]],
                "endog": data[["xn"]],
                "instruments": data[["z1"]],
            }
            for i, data in enumerate((first, second))
        }
    return {
        f"eq{i}": {"dependent": data.y, "exog": data[["const", "xe"]]}
        for i, data in enumerate((first, second))
    }


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize(("model", "endogenous"), [(SUR, False), (IV3SLS, True)])
def test_system_models_raise_when_fit_with_too_few_observations(
    model, endogenous, kernel
):
    nobs = fewest_observations(kernel) - 1
    mod = model(system_equations(nobs, endogenous))
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit(cov_type="kernel", kernel=kernel)


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize(("model", "endogenous"), [(SUR, False), (IV3SLS, True)])
def test_system_models_with_the_fewest_observations(model, endogenous, kernel):
    nobs = fewest_observations(kernel)
    mod = model(system_equations(nobs, endogenous))
    res = mod.fit(cov_type="kernel", kernel=kernel)
    assert 0 <= res.cov_config["bandwidth"] <= nobs - 1
    check_covariance_is_finite(res)
    explicit = mod.fit(cov_type="kernel", kernel=kernel, bandwidth=1)
    assert explicit.cov_config["bandwidth"] == 1


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize(("model", "endogenous"), [(SUR, False), (IV3SLS, True)])
def test_system_models_explicit_bandwidth_works_with_the_observations_that_raise(
    model, endogenous, kernel
):
    nobs = fewest_observations(kernel) - 1
    res = model(system_equations(nobs, endogenous)).fit(
        cov_type="kernel", kernel=kernel, bandwidth=1
    )
    assert res.cov_config["bandwidth"] == 1
    check_covariance_is_finite(res)


@pytest.mark.parametrize("kernel", CANONICAL)
def test_system_gmm_weights_raise_when_fit_with_too_few_observations(kernel):
    nobs = fewest_observations(kernel) - 1
    mod = IVSystemGMM(
        system_equations(nobs, True),
        weight_type="kernel",
        kernel=kernel,
        optimal_bw=True,
    )
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit()


# The robust weights of the two equations need 4 observations before the covariance
# gets to the bandwidth: the Bartlett kernel is reached with 4
SYSTEM_GMM_COVARIANCE_FEW = [("parzen", 4), ("qs", 4), ("qs", 5)]


@pytest.mark.parametrize(("kernel", "nobs"), SYSTEM_GMM_COVARIANCE_FEW)
def test_system_gmm_covariance_raises_when_fit_with_too_few_observations(kernel, nobs):
    mod = IVSystemGMM(system_equations(nobs, True))
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit(cov_type="kernel", kernel=kernel)


def test_system_gmm_default_bandwidth_raises_when_fit_with_too_few_observations():
    mod = IVSystemGMM(system_equations(3, True), weight_type="kernel")
    with pytest.warns(DefaultBandwidthWarning):
        with pytest.raises(ValueError, match=re.escape(too_few_message("bartlett", 3))):
            mod.fit()


@pytest.mark.parametrize("kernel", CANONICAL)
def test_system_gmm_explicit_bandwidth_works_with_the_observations_that_raise(kernel):
    nobs = fewest_observations(kernel) - 1
    equations = system_equations(nobs, True)
    explicit = IVSystemGMM(
        equations, weight_type="kernel", kernel=kernel, bandwidth=1
    ).fit()
    assert explicit.weight_config["bandwidth"] == 1
    restored = IVSystemGMM(
        equations, weight_type="kernel", kernel=kernel, optimal_bw=False
    ).fit()
    assert restored.weight_config["bandwidth"] == nobs - 2


@pytest.mark.parametrize("kernel", CANONICAL)
def test_system_gmm_with_the_fewest_observations(kernel):
    nobs = fewest_observations(kernel)
    equations = system_equations(nobs, True)
    res = IVSystemGMM(
        equations, weight_type="kernel", kernel=kernel, optimal_bw=True
    ).fit(cov_type="kernel", kernel=kernel)
    assert 0 <= res.weight_config["bandwidth"] <= nobs - 1
    assert 0 <= res.cov_config["bandwidth"] <= nobs - 1
    check_covariance_is_finite(res)


@pytest.mark.parametrize("extra", [0, 3])
@pytest.mark.parametrize("kernel", CANONICAL)
def test_system_kernel_weight_matrix_bandwidth_is_the_rule(kernel, extra):
    nobs = fewest_observations(kernel) + extra
    rg = np.random.default_rng(5)
    x = [rg.standard_normal((nobs, 2)) for _ in range(2)]
    z = [rg.standard_normal((nobs, 3)) for _ in range(2)]
    eps = rg.standard_normal((nobs, 2))
    weight = SystemKernelWeightMatrix(kernel=kernel, optimal_bw=True)
    weight.weight_matrix(x, z, eps)
    ze = np.hstack([z[i] * eps[:, [i]] for i in range(2)])
    series = (ze / ze.std(axis=0)).sum(axis=1)
    expected = newey_west(series, kernel)
    bandwidth = weight.bandwidth
    assert bandwidth == expected


@pytest.mark.parametrize("kernel", CANONICAL)
def test_system_kernel_weight_matrix_raises_with_too_few_observations(kernel):
    nobs = fewest_observations(kernel) - 1
    rg = np.random.default_rng(nobs)
    x = [rg.standard_normal((nobs, 2))]
    z = [rg.standard_normal((nobs, 3))]
    eps = rg.standard_normal((nobs, 1))
    weight = SystemKernelWeightMatrix(kernel=kernel, optimal_bw=True)
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        weight.weight_matrix(x, z, eps)
    explicit = SystemKernelWeightMatrix(kernel=kernel, bandwidth=1)
    explicit.weight_matrix(x, z, eps)
    bandwidth = explicit.bandwidth
    assert bandwidth == 1


# ---------------------------------------------------------------------------------
# Asset pricing models
# ---------------------------------------------------------------------------------
def factor_data(nobs, nportfolio=2):
    rg = np.random.default_rng(11)
    return rg.standard_normal((nobs, nportfolio)), rg.standard_normal((nobs, 1))


def bandwidth_of(res):
    """The bandwidth in the description of the covariance estimator"""
    found = re.search(r"Bandwidth: (\d+)", res.cov_estimator)
    assert found is not None
    return int(found.group(1))


def traded_factor_moments(portfolios, factors):
    """The moment conditions of TradedFactorModel: the time-series regressions"""
    nobs = portfolios.shape[0]
    regressors = np.c_[np.ones(nobs), factors]
    coefs = np.linalg.lstsq(regressors, portfolios, rcond=None)[0]
    eps = portfolios - regressors @ coefs
    columns = [
        regressors[:, k] * eps[:, i]
        for i in range(portfolios.shape[1])
        for k in range(regressors.shape[1])
    ]
    columns += [factors[:, k] - factors[:, k].mean() for k in range(factors.shape[1])]
    return np.column_stack(columns)


FACTOR_MODELS = [TradedFactorModel, LinearFactorModel]


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", FACTOR_MODELS)
def test_factor_models_raise_when_fit_with_too_few_observations(model, kernel):
    nobs = fewest_observations(kernel) - 1
    mod = model(*factor_data(nobs))
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit(cov_type="kernel", kernel=kernel)


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", FACTOR_MODELS)
def test_factor_models_explicit_bandwidth_works_with_the_observations_that_raise(
    model, kernel
):
    nobs = fewest_observations(kernel) - 1
    res = model(*factor_data(nobs)).fit(cov_type="kernel", kernel=kernel, bandwidth=1)
    bandwidth = bandwidth_of(res)
    assert bandwidth == 1
    check_covariance_is_finite(res)


@pytest.mark.parametrize("kernel", CANONICAL)
@pytest.mark.parametrize("model", FACTOR_MODELS)
def test_factor_models_with_the_fewest_observations(model, kernel):
    nobs = fewest_observations(kernel)
    res = model(*factor_data(nobs)).fit(cov_type="kernel", kernel=kernel)
    bandwidth = bandwidth_of(res)
    assert 0 <= bandwidth <= nobs - 1
    check_covariance_is_finite(res)


@pytest.mark.parametrize("extra", [0, 1, 2, 10])
@pytest.mark.parametrize("kernel", CANONICAL)
def test_traded_factor_model_bandwidth_is_the_rule_for_the_moments(kernel, extra):
    nobs = fewest_observations(kernel) + extra
    portfolios, factors = factor_data(nobs)
    res = TradedFactorModel(portfolios, factors).fit(cov_type="kernel", kernel=kernel)
    moments = traded_factor_moments(portfolios, factors)
    series = (moments / moments.std(axis=0)).sum(axis=1)
    expected = newey_west(series, kernel)
    bandwidth = bandwidth_of(res)
    assert bandwidth == expected


# One portfolio and one factor have 3 moment conditions, and GMM needs more
# observations than that: the Bartlett kernel is reached with 4
GMM_FACTOR_FEW = [("parzen", 4), ("qs", 4), ("qs", 5)]


@pytest.mark.parametrize(("kernel", "nobs"), GMM_FACTOR_FEW)
def test_gmm_factor_model_raises_when_fit_with_too_few_observations(kernel, nobs):
    mod = LinearFactorModelGMM(*factor_data(nobs, nportfolio=1))
    with pytest.raises(ValueError, match=re.escape(too_few_message(kernel, nobs))):
        mod.fit(cov_type="kernel", kernel=kernel, disp=0)


@pytest.mark.parametrize(("kernel", "nobs"), GMM_FACTOR_FEW)
def test_gmm_factor_model_explicit_bandwidth_works_with_the_observations_that_raise(
    kernel, nobs
):
    mod = LinearFactorModelGMM(*factor_data(nobs, nportfolio=1))
    res = mod.fit(cov_type="kernel", kernel=kernel, bandwidth=1, disp=0)
    bandwidth = bandwidth_of(res)
    assert bandwidth == 1


@pytest.mark.parametrize("extra", [0, 3])
@pytest.mark.parametrize("kernel", CANONICAL)
def test_factor_kernel_covariance_and_weight_bandwidth_is_the_rule(kernel, extra):
    nobs = fewest_observations(kernel) + extra
    moments = np.random.default_rng(7).standard_normal((nobs, 3))
    series = (moments / moments.std(axis=0)).sum(axis=1)
    expected = newey_west(series, kernel)
    covariance = FactorKernelCovariance(moments, jacobian=np.eye(3), kernel=kernel)
    covariance_bandwidth = covariance.bandwidth
    weight = FactorKernelWeight(moments, kernel=kernel)
    weight_bandwidth = weight.bandwidth
    assert covariance_bandwidth == expected
    assert weight_bandwidth == expected


@pytest.mark.parametrize("kernel", CANONICAL)
def test_factor_kernel_covariance_and_weight_raise_with_too_few_observations(kernel):
    nobs = fewest_observations(kernel) - 1
    moments = np.random.default_rng(7).standard_normal((nobs, 3))
    match = re.escape(too_few_message(kernel, nobs))
    covariance = FactorKernelCovariance(moments, jacobian=np.eye(3), kernel=kernel)
    with pytest.raises(ValueError, match=match):
        _ = covariance.cov
    weight = FactorKernelWeight(moments, kernel=kernel)
    with pytest.raises(ValueError, match=match):
        weight.w(moments)
    # An explicit bandwidth does not need the observations
    explicit = FactorKernelWeight(moments, kernel=kernel, bandwidth=1)
    bandwidth = explicit.bandwidth
    assert bandwidth == 1


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("kernel", CANONICAL)
def test_moments_that_do_not_vary_have_no_finite_bandwidth(kernel):
    # The standard deviation of the moments is 0, and they are divided by it
    nobs = fewest_observations(kernel) + 5
    zeros = np.zeros((nobs, 2))
    match = rf"The automatic bandwidth of the {kernel!r} kernel is not finite\."
    iv_weight = IVKernelWeightMatrix(kernel=kernel, optimal_bw=True)
    with pytest.raises(ValueError, match=match):
        iv_weight._estimate_bandwidth(zeros)
    system_weight = SystemKernelWeightMatrix(kernel=kernel, optimal_bw=True)
    with pytest.raises(ValueError, match=match):
        system_weight._optimal_bandwidth(zeros)
    covariance = FactorKernelCovariance(zeros, jacobian=np.eye(2), kernel=kernel)
    with pytest.raises(ValueError, match=match):
        _ = covariance.bandwidth
