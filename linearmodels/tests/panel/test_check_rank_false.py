"""
The constant is detected whether or not the rank of the regressors is checked

check_rank=False skips the computation of the rank of the regressors, and the models
then use the number of regressors, which is the rank if they are not collinear. They
used the number of periods of the panel instead. The detection of a constant that is
not a column, e.g., dummy variables that add to one, compares the rank of the
regressors with the rank of the regressors and a constant, so that it decided that
there was no constant in a panel with 5 periods and 2 dummy variables, and that there
was a constant in a panel with 2 periods and 3 regressors. The R-squared and the
F-statistic then treated the models as if they had, or did not have, a constant.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)

MODELS = ["pooled", "between", "random-effects", "fama-macbeth", "panel-ols"]

# Two cross-sections cannot give the standard errors of 3 parameters
pytestmark = pytest.mark.filterwarnings(
    "ignore:The number of cross-sectional regressions:"
    "linearmodels.shared.exceptions.InferenceUnavailableWarning"
)


def build(model, y, x, check_rank):
    if model == "panel-ols":
        return PanelOLS(y, x, entity_effects=True, check_rank=check_rank)
    models = {
        "pooled": PooledOLS,
        "between": BetweenOLS,
        "random-effects": RandomEffects,
        "fama-macbeth": FamaMacBeth,
        "first-difference": FirstDifferenceOLS,
    }
    return models[model](y, x, check_rank=check_rank)


def make_index(nentity, nperiod):
    return pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )


def implicit_constant_panel():
    """Two dummy variables that add to one, and 5 periods"""
    rg = np.random.default_rng(0)
    index = make_index(40, 5)
    group = rg.integers(0, 2, len(index))
    x = pd.DataFrame(
        {"a": (group == 0) * 1.0, "b": (group == 1) * 1.0},
        index=index,
    )
    y = pd.Series(rg.standard_normal(len(index)) + group, index=index, name="y")
    return y, x


def no_constant_panel():
    """Three regressors with a mean of zero, and 2 periods"""
    rg = np.random.default_rng(1)
    index = make_index(40, 2)
    x = pd.DataFrame(
        rg.standard_normal((len(index), 3)), index=index, columns=["a", "b", "c"]
    )
    y = pd.Series(rg.standard_normal(len(index)) + 1, index=index, name="y")
    return y, x


def r_squared(y, x, centered):
    yv, xv = y.to_numpy(), x.to_numpy()
    resid = yv - xv @ np.linalg.lstsq(xv, yv, rcond=None)[0]
    total = yv - yv.mean() if centered else yv
    return 1 - resid @ resid / (total @ total)


@pytest.mark.parametrize("model", MODELS)
def test_implicit_constant(model):
    y, x = implicit_constant_panel()
    checked = build(model, y, x, True)
    skipped = build(model, y, x, False)
    assert checked.has_constant
    assert skipped.has_constant
    res_checked, res_skipped = checked.fit(), skipped.fit()
    assert_allclose(res_skipped.params, res_checked.params)
    assert_allclose(res_skipped.rsquared, res_checked.rsquared)
    assert_allclose(res_skipped.f_statistic.stat, res_checked.f_statistic.stat)


@pytest.mark.parametrize("model", MODELS + ["first-difference"])
def test_no_constant_with_more_regressors_than_periods(model):
    y, x = no_constant_panel()
    checked = build(model, y, x, True)
    skipped = build(model, y, x, False)
    assert not checked.has_constant
    assert not skipped.has_constant
    res_checked, res_skipped = checked.fit(), skipped.fit()
    assert_allclose(res_skipped.params, res_checked.params)
    assert_allclose(res_skipped.rsquared, res_checked.rsquared)
    assert_allclose(res_skipped.f_statistic.stat, res_checked.f_statistic.stat)


@pytest.mark.parametrize("check_rank", [True, False])
def test_pooled_r_squared_of_an_implicit_constant_is_centered(check_rank):
    y, x = implicit_constant_panel()
    res = PooledOLS(y, x, check_rank=check_rank).fit()
    assert_allclose(res.rsquared, r_squared(y, x, True))


@pytest.mark.parametrize("check_rank", [True, False])
def test_pooled_r_squared_without_a_constant_is_not_centered(check_rank):
    y, x = no_constant_panel()
    res = PooledOLS(y, x, check_rank=check_rank).fit()
    assert_allclose(res.rsquared, r_squared(y, x, False))
