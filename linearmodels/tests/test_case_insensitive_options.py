"""
Tests that the names of options are not case sensitive

The cov_type, weight_type, kernel and group are compared in lower case, so that a
spelling such as "Robust" gives the results of "robust".
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML, AbsorbingLS
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.data import PanelData
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

NOBS = 120
KERNEL = {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 3}
KERNEL_MIXED = {"cov_type": "Kernel", "kernel": "Bartlett", "bandwidth": 3}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261014)
    z = rng.standard_normal((NOBS, 2))
    x = (z @ [1.0, 0.5] + rng.standard_normal(NOBS))[:, None]
    y = x[:, 0] + rng.standard_normal(NOBS)
    index = pd.MultiIndex.from_product([range(12), range(10)])
    factors = rng.standard_normal((NOBS, 2))
    return {
        "iv": (y, None, x, z),
        "clusters": np.arange(NOBS) // 6,
        "cats": pd.DataFrame({"a": pd.Categorical(rng.integers(0, 5, NOBS))}),
        "equations": {"a": {"dependent": y, "exog": x}},
        "iv_equations": {
            "a": {"dependent": y, "exog": None, "endog": x, "instruments": z}
        },
        "panel": (pd.Series(y, index=index), pd.DataFrame(x, index=index)),
        "factors": factors,
        "portfolios": factors @ rng.standard_normal((2, 4))
        + rng.standard_normal((NOBS, 4)),
    }


# Each case has a function that makes the model and the arguments of fit with the
# names in lower case and with the names in mixed case
CASES = {
    "IV2SLS": (lambda d: IV2SLS(*d["iv"]), KERNEL, KERNEL_MIXED),
    "IVLIML": (
        lambda d: IVLIML(*d["iv"]),
        {"cov_type": "unadjusted"},
        {"cov_type": "Unadjusted"},
    ),
    "AbsorbingLS": (
        lambda d: AbsorbingLS(d["iv"][0], absorb=d["cats"]),
        KERNEL,
        KERNEL_MIXED,
    ),
    "IVGMM": (lambda d: IVGMM(*d["iv"]), KERNEL, KERNEL_MIXED),
    "IVGMMCUE": (lambda d: IVGMMCUE(*d["iv"]), KERNEL, KERNEL_MIXED),
    "IVJIVE": (lambda d: IVJIVE(*d["iv"]), KERNEL, KERNEL_MIXED),
    "SUR": (lambda d: SUR(d["equations"]), KERNEL, KERNEL_MIXED),
    "IV3SLS": (lambda d: IV3SLS(d["iv_equations"]), KERNEL, KERNEL_MIXED),
    "IVSystemGMM": (lambda d: IVSystemGMM(d["iv_equations"]), KERNEL, KERNEL_MIXED),
    "PooledOLS": (lambda d: PooledOLS(*d["panel"]), KERNEL, KERNEL_MIXED),
    "PanelOLS": (
        lambda d: PanelOLS(*d["panel"], entity_effects=True),
        {"cov_type": "clustered", "cluster_entity": True},
        {"cov_type": "Clustered", "cluster_entity": True},
    ),
    "BetweenOLS": (
        lambda d: BetweenOLS(*d["panel"]),
        {"cov_type": "robust"},
        {"cov_type": "Robust"},
    ),
    "FirstDifferenceOLS": (
        lambda d: FirstDifferenceOLS(*d["panel"]),
        KERNEL,
        KERNEL_MIXED,
    ),
    "RandomEffects": (lambda d: RandomEffects(*d["panel"]), KERNEL, KERNEL_MIXED),
    "FamaMacBeth": (lambda d: FamaMacBeth(*d["panel"]), KERNEL, KERNEL_MIXED),
    "TradedFactorModel": (
        lambda d: TradedFactorModel(d["portfolios"], d["factors"]),
        KERNEL,
        KERNEL_MIXED,
    ),
    "LinearFactorModel": (
        lambda d: LinearFactorModel(d["portfolios"], d["factors"]),
        KERNEL,
        KERNEL_MIXED,
    ),
    "LinearFactorModelGMM": (
        lambda d: LinearFactorModelGMM(d["portfolios"], d["factors"]),
        KERNEL,
        KERNEL_MIXED,
    ),
}


def fit(model, options):
    extra = {"disp": 0} if isinstance(model, LinearFactorModelGMM) else {}
    return model.fit(**options, **extra)


# LinearFactorModelGMM minimizes the GMM objective with BFGS, whose gradient is a finite
# difference. Fits of the same data that differ by rounding, in the last bit, stop at
# estimates that differ by 4e-8 (the parameters) and 2e-7 (the covariance, whose
# smallest elements have the largest relative differences) and not by 1e-16.
RTOL = {"LinearFactorModelGMM": 1e-5}


@pytest.mark.parametrize("name", list(CASES))
def test_mixed_case_names_give_the_results_of_lower_case_names(data, name):
    make, lower, mixed = CASES[name]
    expected = fit(make(data), lower)
    result = fit(make(data), mixed)
    rtol = RTOL.get(name, 1e-7)
    assert_allclose(result.params, expected.params, rtol=rtol)
    assert_allclose(result.cov, expected.cov, rtol=rtol)


def as_array(value):
    return value.values2d if isinstance(value, PanelData) else np.asarray(value)


@pytest.mark.parametrize("method", ["count", "mean", "demean", "dummies"])
def test_group_is_not_case_sensitive(data, method):
    panel = PanelData(data["panel"][1])
    expected = getattr(panel, method)("time")
    result = getattr(panel, method)("Time")
    assert_allclose(as_array(result), as_array(expected))


WEIGHT_MODELS = {
    "IVGMM": lambda d, **kw: IVGMM(*d["iv"], **kw),
    "IVGMMCUE": lambda d, **kw: IVGMMCUE(*d["iv"], **kw),
    "IVSystemGMM": lambda d, **kw: IVSystemGMM(d["iv_equations"], **kw),
}


@pytest.mark.parametrize("weight_type", ["robust", "kernel"])
@pytest.mark.parametrize("name", list(WEIGHT_MODELS))
def test_mixed_case_weight_type_gives_the_results_of_lower_case(
    data, name, weight_type
):
    options = {"kernel": "bartlett", "bandwidth": 3} if weight_type == "kernel" else {}
    mixed_options = {"kernel": "Bartlett", "bandwidth": 3} if options else {}
    make = WEIGHT_MODELS[name]
    expected = make(data, weight_type=weight_type, **options).fit()
    result = make(data, weight_type=weight_type.capitalize(), **mixed_options).fit()
    assert_allclose(result.params, expected.params)
    assert_allclose(result.cov, expected.cov)
