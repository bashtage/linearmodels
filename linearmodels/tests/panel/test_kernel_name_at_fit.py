"""
A name that is not a kernel is reported when the panel models are fit

The kernel weights were looked up when the covariance was first used, so that
``fit`` returned and the error was raised later, for example by ``res.std_errors``.
It happened with the Driscoll-Kraay ("kernel") and "ac" covariances of PooledOLS,
PanelOLS, RandomEffects and FirstDifferenceOLS, with the kernel covariance of
FamaMacBeth if the bandwidth is given, and with every covariance of FamaMacBeth, since
it passes the kernel to its covariance whatever cov_type is. The kernel is checked when
the covariance is created now, so that ``fit`` raises, whether or not the bandwidth is
given.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import (
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.covariance import (
    ACCovariance,
    DriscollKraay,
    FamaMacBethCovariance,
)
from linearmodels.shared.exceptions import UnknownOptionError

NENTITY = 12
NPERIOD = 8
# Written out, so that the tests fail if a name is added or removed, or if the message
# lists a name that is not accepted
SUPPORTED = (
    "'andrews', 'bartlett', 'gallant', 'newey-west', 'parzen', 'qs', "
    "'quadratic-spectral'"
)
ALIASES = [
    ("bartlett", "newey-west"),
    ("parzen", "gallant"),
    ("qs", "quadratic-spectral", "andrews"),
]
# A name that is not close to any kernel and a name that is close to one, for which
# the message makes a suggestion. The names are not case sensitive
UNKNOWN = [("foo", None), ("parzan", "parzen"), ("bartlet", "bartlett")]
UNKNOWN_IDS = [name for name, _ in UNKNOWN]
KERNEL_COV_TYPES = ["kernel", "driscoll-kraay", "dk"]
AC_COV_TYPES = ["ac", "autocorrelated"]
BANDWIDTHS = [pytest.param(None, id="default"), pytest.param(2, id="given")]


def message(name, suggestion):
    """The message, written out in full and not made by the function that makes it"""
    hint = f" Did you mean {suggestion!r}?" if suggestion else ""
    start = f"Unknown kernel: {name!r}.{hint}"
    return f"{start} The supported values of kernel are: {SUPPORTED}."


@pytest.fixture(scope="module")
def frame():
    rg = np.random.default_rng(20261013)
    index = pd.MultiIndex.from_product(
        [np.arange(NENTITY), np.arange(NPERIOD)], names=["id", "year"]
    )
    nobs = index.shape[0]
    common = rg.standard_normal(NPERIOD)
    x = rg.standard_normal(nobs) + 0.5 * common[index.codes[1]]
    effect = rg.standard_normal(NENTITY)[index.codes[0]]
    shock = rg.standard_normal(nobs) + common[index.codes[1]]
    y = 1 + 0.5 * x + effect + shock
    return pd.DataFrame({"y": y, "const": 1.0, "x": x}, index=index)


MODELS = {
    "PooledOLS": lambda f: PooledOLS(f["y"], f[["const", "x"]]),
    "PanelOLS-entity": lambda f: PanelOLS(f["y"], f[["x"]], entity_effects=True),
    "PanelOLS-time": lambda f: PanelOLS(f["y"], f[["x"]], time_effects=True),
    "RandomEffects": lambda f: RandomEffects(f["y"], f[["const", "x"]]),
    "FirstDifferenceOLS": lambda f: FirstDifferenceOLS(f["y"], f[["x"]]),
}


def options(bandwidth):
    return {} if bandwidth is None else {"bandwidth": bandwidth}


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
@pytest.mark.parametrize(("name", "suggestion"), UNKNOWN, ids=UNKNOWN_IDS)
@pytest.mark.parametrize("cov_type", [*KERNEL_COV_TYPES, *AC_COV_TYPES])
@pytest.mark.parametrize("model", MODELS)
def test_unknown_kernel_raises_when_the_model_is_fit(
    frame, model, cov_type, name, suggestion, bandwidth
):
    mod = MODELS[model](frame)
    with pytest.raises(UnknownOptionError) as info:
        mod.fit(cov_type=cov_type, kernel=name, **options(bandwidth))
    error = info.value
    # Code that caught the KeyError or the ValueError of earlier versions still works
    is_value_error = isinstance(error, ValueError)
    is_key_error = isinstance(error, KeyError)
    assert is_value_error
    assert is_key_error
    text = str(error)
    expected = message(name, suggestion)
    assert text == expected


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
@pytest.mark.parametrize(("name", "suggestion"), UNKNOWN, ids=UNKNOWN_IDS)
@pytest.mark.parametrize(
    "cov_type", ["kernel", "unadjusted", "homoskedastic", "robust", "heteroskedastic"]
)
def test_fama_macbeth_unknown_kernel_raises_when_the_model_is_fit(
    frame, cov_type, name, suggestion, bandwidth
):
    # The kernel is passed to the covariance whatever the covariance is, and the
    # covariances other than the kernel covariance used the weights of the kernel
    # with a bandwidth of 0 when they were first used
    mod = FamaMacBeth(frame["y"], frame[["const", "x"]])
    with pytest.raises(UnknownOptionError) as info:
        mod.fit(cov_type=cov_type, kernel=name, **options(bandwidth))
    text = str(info.value)
    expected = message(name, suggestion)
    assert text == expected


def test_the_name_is_checked_before_the_bandwidth(frame):
    # A bandwidth that is not below the number of periods is another ValueError. The
    # name that is not a kernel is reported first, as the name is wrong whatever the
    # bandwidth is
    for cov_type in ("kernel", "ac"):
        mod = PooledOLS(frame["y"], frame[["const", "x"]])
        with pytest.raises(UnknownOptionError, match="Unknown kernel: 'foo'"):
            mod.fit(cov_type=cov_type, kernel="foo", bandwidth=50)
    fm = FamaMacBeth(frame["y"], frame[["const", "x"]])
    with pytest.raises(UnknownOptionError, match="Unknown kernel: 'foo'"):
        fm.fit(cov_type="kernel", kernel="foo", bandwidth=50)
    # The bandwidth is still reported with a kernel that is one
    with pytest.raises(ValueError, match="must be below the number of time periods"):
        PooledOLS(frame["y"], frame[["const", "x"]]).fit(
            cov_type="kernel", kernel="bartlett", bandwidth=50
        )


def test_model_is_not_changed_by_the_failed_fit(frame):
    mod = PooledOLS(frame["y"], frame[["const", "x"]])
    expected = mod.fit(cov_type="kernel", kernel="parzen", bandwidth=2).std_errors
    with pytest.raises(UnknownOptionError):
        mod.fit(cov_type="kernel", kernel="foo", bandwidth=2)
    after = mod.fit(cov_type="kernel", kernel="parzen", bandwidth=2).std_errors
    assert_allclose(after, expected, rtol=1e-12)


def covariance_args(frame):
    mod = PooledOLS(frame["y"], frame[["const", "x"]])
    res = mod.fit()
    return (
        mod.dependent.values2d,
        mod.exog.values2d,
        np.asarray(res.params)[:, None],
        mod.dependent.entity_ids,
        mod.dependent.time_ids,
    )


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
@pytest.mark.parametrize("cov", [DriscollKraay, ACCovariance])
def test_estimators_check_the_kernel_when_they_are_created(frame, cov, bandwidth):
    args = covariance_args(frame)
    with pytest.raises(UnknownOptionError, match="Unknown kernel: 'foo'"):
        cov(*args, kernel="foo", **options(bandwidth))
    estimator = cov(*args, kernel="parzen", **options(bandwidth))
    finite = np.all(np.isfinite(estimator.cov))
    assert finite


@pytest.mark.parametrize("bandwidth", [*BANDWIDTHS, pytest.param(0, id="zero")])
def test_fama_macbeth_estimator_checks_the_kernel_when_it_is_created(frame, bandwidth):
    mod = FamaMacBeth(frame["y"], frame[["const", "x"]])
    res = mod.fit()
    args = (
        mod.dependent.values2d,
        mod.exog.values2d,
        np.asarray(res.params)[:, None],
        res.all_params,
    )
    with pytest.raises(UnknownOptionError, match="Unknown kernel: 'foo'"):
        FamaMacBethCovariance(*args, kernel="foo", **options(bandwidth))
    estimator = FamaMacBethCovariance(*args, kernel="parzen", **options(bandwidth))
    finite = np.all(np.isfinite(estimator.cov))
    assert finite


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
@pytest.mark.parametrize("cov_type", ["kernel", "ac"])
@pytest.mark.parametrize("names", ALIASES, ids=[names[0] for names in ALIASES])
def test_every_name_of_a_kernel_is_accepted_and_gives_the_same_covariance(
    frame, names, cov_type, bandwidth
):
    covariances = []
    for name in names:
        mod = PooledOLS(frame["y"], frame[["const", "x"]])
        res = mod.fit(cov_type=cov_type, kernel=name, **options(bandwidth))
        covariances.append(np.asarray(res.cov))
    for other in covariances[1:]:
        assert_allclose(other, covariances[0], rtol=1e-12)


@pytest.mark.parametrize("cov_type", ["kernel", "ac"])
def test_default_kernel_is_bartlett(frame, cov_type):
    mod = PooledOLS(frame["y"], frame[["const", "x"]])
    default = mod.fit(cov_type=cov_type, bandwidth=2)
    named = mod.fit(cov_type=cov_type, kernel="bartlett", bandwidth=2)
    assert_allclose(default.cov, named.cov, rtol=1e-12)
