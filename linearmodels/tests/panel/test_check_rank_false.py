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

from contextlib import nullcontext

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
from linearmodels.shared.exceptions import InferenceUnavailableWarning

MODELS = ["pooled", "between", "random-effects", "fama-macbeth", "panel-ols"]


def warns_for_too_few_periods(model):
    """Two cross-sections cannot give the standard errors of 3 parameters"""
    if model == "fama-macbeth":
        return pytest.warns(
            InferenceUnavailableWarning, match="cross-sectional regressions"
        )
    return nullcontext()


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
    with warns_for_too_few_periods(model):
        checked = build(model, y, x, True)
    with warns_for_too_few_periods(model):
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


def rank_deficient_panel(kind):
    """
    Regressors that do not have full column rank, and the regressors without the excess

    A duplicated regressor and a column of zeros have no constant. A complete set of dummy
    variables, which adds to one, has a constant, whatever else is in the regressors.
    """
    index = make_index(40, 5)
    rg = np.random.default_rng(2)
    reg = pd.DataFrame(
        rg.standard_normal((len(index), 2)), index=index, columns=["a", "b"]
    )
    group = rg.integers(0, 3, len(index))
    dummies = pd.DataFrame({f"d{i}": (group == i) * 1.0 for i in range(3)}, index=index)
    y = pd.Series(1 + rg.standard_normal(len(index)) + group, index=index, name="y")
    cases = {
        "duplicate": (reg.assign(dup=reg["a"]), reg, False),
        "zeros": (reg.assign(zero=0.0), reg, False),
        "combination": (reg.assign(combo=reg["a"] - 2 * reg["b"]), reg, False),
        "dummies and a duplicate": (dummies.assign(dup=dummies["d0"]), dummies, True),
        "dummies and zeros": (dummies.assign(zero=0.0), dummies, True),
        "dummies, regressors and a combination": (
            pd.concat([dummies, reg], axis=1).assign(combo=reg["a"] + dummies["d1"]),
            pd.concat([dummies, reg], axis=1),
            True,
        ),
    }
    x_deficient, x_full, constant = cases[kind]
    return y, x_deficient, x_full, constant


RANK_DEFICIENT = [
    "duplicate",
    "zeros",
    "combination",
    "dummies and a duplicate",
    "dummies and zeros",
    "dummies, regressors and a combination",
]


@pytest.mark.parametrize("kind", RANK_DEFICIENT)
@pytest.mark.parametrize("model", ["pooled", "between", "random-effects"])
def test_rank_deficient_regressors_have_a_constant_only_if_they_span_one(model, kind):
    # Regressors without full column rank were read as having a constant. The constant
    # decides whether R-squared is centered, so that it was wrong without one, and
    # it was not centered for dummy variables that add to one if the number of
    # regressors was used for the rank
    y, x_deficient, x_full, constant = rank_deficient_panel(kind)
    skipped = build(model, y, x_deficient, False)
    reference = build(model, y, x_full, True)
    assert skipped.has_constant is constant
    assert reference.has_constant is constant
    res_skipped, res_reference = skipped.fit(), reference.fit()
    assert_allclose(res_skipped.rsquared, res_reference.rsquared)
    assert_allclose(res_skipped.fitted_values, res_reference.fitted_values)


@pytest.mark.parametrize("kind", ["duplicate", "zeros", "combination"])
def test_first_difference_of_rank_deficient_regressors_without_a_constant(kind):
    # The regressors have no constant, which the first difference rejects, whether or
    # not they have full column rank
    y, x_deficient, x_full, _ = rank_deficient_panel(kind)
    res = build("first-difference", y, x_deficient, False).fit()
    reference = build("first-difference", y, x_full, True).fit()
    assert_allclose(res.rsquared, reference.rsquared)
    assert_allclose(res.fitted_values, reference.fitted_values)


def test_first_difference_rejects_dummies_that_add_to_one():
    y, x_deficient, _, _ = rank_deficient_panel("dummies and a duplicate")
    with pytest.raises(ValueError, match=r"Constants are not allowed"):
        build("first-difference", y, x_deficient, False)
