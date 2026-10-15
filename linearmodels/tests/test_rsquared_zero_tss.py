"""
R-squared is NaN, with no error and no warning, if its total sum of squares is zero

A dependent variable that is constant, or all zero, has a total sum of squares of 0 (or of
rounding error), and the R-squared 1 - RSS / TSS is not defined. The panel models raised a
bare ZeroDivisionError for PooledOLS, BetweenOLS, FirstDifferenceOLS, FamaMacBeth and
PanelOLS, returned 0.0 for PanelOLS in other cases, and RandomEffects returned rounding
noise (0.0042). The IV and system models returned -inf, NaN, or rounding noise with
RuntimeWarnings. Every model family uses one rule now. The values of an R-squared that is
defined do not change.
"""

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
from linearmodels.iv import IV2SLS, IVGMM, IVJIVE, IVLIML, AbsorbingLS
from linearmodels.iv.model import _OLS
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared.hypotheses import InvalidTestStatistic
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

CONSTANTS = [5.0, 0.1, 1 / 3, np.pi, -2.5, 1e-10, 1e10, 123456.789]
PANEL_MEASURES = [
    "rsquared",
    "rsquared_within",
    "rsquared_between",
    "rsquared_overall",
    "corr_squared_within",
    "corr_squared_between",
    "corr_squared_overall",
]
NENTITY, NPERIOD = 40, 6


def assert_nan(res, attributes):
    """The attributes of the results are all NaN"""
    unexpected = [name for name in attributes if not np.isnan(getattr(res, name))]
    assert unexpected == []


def assert_finite(res, attributes):
    """The attributes of the results are all finite"""
    unexpected = [name for name in attributes if not np.isfinite(getattr(res, name))]
    assert unexpected == []


def assert_value_is_nan(value):
    is_nan = np.isnan(value)
    assert is_nan


def assert_value_is_finite(value):
    is_finite = np.isfinite(value)
    assert is_finite


def assert_summary_has(res, text):
    summary = str(res.summary)
    contains = text in summary
    assert contains


def panel_data(nentity=NENTITY, nperiod=NPERIOD, seed=0):
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    x = pd.DataFrame(
        rg.standard_normal((len(index), 2)), index=index, columns=["a", "b"]
    )
    return index, x, x.assign(const=1.0)


# The models that have a constant (the last regressor of xc), and the extra measures
PANEL_WITH_CONSTANT = {
    "PooledOLS": (lambda y, x, w: PooledOLS(y, x, weights=w), []),
    "BetweenOLS": (lambda y, x, w: BetweenOLS(y, x, weights=w), []),
    "RandomEffects": (lambda y, x, w: RandomEffects(y, x, weights=w), []),
    "FamaMacBeth": (
        lambda y, x, w: FamaMacBeth(y, x, weights=w),
        ["avg_rsquared", "avg_adj_rsquared"],
    ),
    "PanelOLS entity": (
        lambda y, x, w: PanelOLS(y, x, weights=w, entity_effects=True),
        ["rsquared_inclusive"],
    ),
    "PanelOLS time": (
        lambda y, x, w: PanelOLS(y, x, weights=w, time_effects=True),
        ["rsquared_inclusive"],
    ),
    "PanelOLS both": (
        lambda y, x, w: PanelOLS(
            y, x, weights=w, entity_effects=True, time_effects=True
        ),
        ["rsquared_inclusive"],
    ),
}


@pytest.mark.parametrize("value", CONSTANTS)
@pytest.mark.parametrize("name", PANEL_WITH_CONSTANT)
def test_constant_dependent_variable_panel(name, value):
    index, _, xc = panel_data()
    y = pd.Series(value, index=index, name="y")
    build, extra = PANEL_WITH_CONSTANT[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = build(y, xc, None).fit()
    assert_nan(res, PANEL_MEASURES + extra)
    assert_summary_has(res, "R-squared")


@pytest.mark.parametrize("name", PANEL_WITH_CONSTANT)
def test_constant_dependent_variable_weighted_and_unbalanced_panel(name):
    index, _, xc = panel_data(seed=1)
    rg = np.random.default_rng(2)
    keep = rg.random(len(index)) > 0.2
    weights = pd.Series(0.5 + rg.chisquare(3, len(index)), index=index)
    y = pd.Series(0.1, index=index, name="y")
    build, extra = PANEL_WITH_CONSTANT[name]
    if name == "RandomEffects":
        # The quasi-differenced dependent variable of an unbalanced panel varies, since
        # the weight of the entity means depends on the number of observations
        keep[:] = True
    res = build(y[keep], xc[keep], weights[keep]).fit()
    assert_nan(res, PANEL_MEASURES + extra)


NO_CONSTANT = {
    "PooledOLS": PooledOLS,
    "BetweenOLS": BetweenOLS,
    "FirstDifferenceOLS": FirstDifferenceOLS,
    "FamaMacBeth": FamaMacBeth,
    "PanelOLS entity": lambda y, x: PanelOLS(y, x, entity_effects=True),
    "PanelOLS time": lambda y, x: PanelOLS(y, x, time_effects=True),
    "PanelOLS both": lambda y, x: PanelOLS(
        y, x, entity_effects=True, time_effects=True
    ),
}


@pytest.mark.parametrize("name", NO_CONSTANT)
def test_all_zero_dependent_variable_panel(name):
    index, x, xc = panel_data()
    y = pd.Series(0.0, index=index, name="y")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = [NO_CONSTANT[name](y, x).fit()]
        if name != "FirstDifferenceOLS":
            results.append(NO_CONSTANT[name](y, xc).fit())
    # Without a constant, the total sum of squares is not centered, and it is 0
    for result in results:
        measures = [
            m for m in PANEL_MEASURES + ["rsquared_inclusive"] if hasattr(result, m)
        ]
        assert_nan(result, measures)


@pytest.mark.parametrize("name", ["PooledOLS", "BetweenOLS", "FamaMacBeth"])
def test_constant_dependent_variable_without_a_constant_is_not_centered(name):
    # The total sum of squares is the sum of squares of the dependent variable, which is
    # not 0, and the mean is not removed from it
    index, x, _ = panel_data()
    y = pd.Series(5.0, index=index, name="y")
    res = NO_CONSTANT[name](y, x).fit()
    assert_finite(res, ["rsquared"])
    assert_allclose(res.rsquared, 1 - res.resid_ss / res.total_ss, rtol=1e-12)
    if name != "BetweenOLS":
        assert_allclose(res.total_ss, 5.0**2 * len(index))
    # The within and the correlation measures remove the mean
    assert_nan(res, ["rsquared_within", "corr_squared_overall"])


def test_first_difference_of_a_constant():
    index, x, _ = panel_data()
    y = pd.Series(5.0, index=index, name="y")
    res = FirstDifferenceOLS(y, x).fit()
    # The differences are 0, but the levels are not
    assert_nan(res, ["rsquared", "rsquared_within", "corr_squared_overall"])
    assert_finite(res, ["rsquared_overall", "rsquared_between"])


def test_dependent_variable_that_is_constant_within_entities():
    rg = np.random.default_rng(3)
    index, x, xc = panel_data()
    y = pd.Series(np.repeat(rg.standard_normal(NENTITY) * 3 + 1, NPERIOD), index=index)
    res = PanelOLS(y, xc, entity_effects=True).fit()
    # The effects explain the dependent variable, and nothing is left to explain
    assert_nan(res, ["rsquared", "rsquared_within", "corr_squared_within"])
    assert_allclose(res.rsquared_inclusive, 1.0, atol=1e-10)
    # The means of the entities vary
    assert_finite(res, ["rsquared_between", "rsquared_overall"])
    res = PanelOLS(y, x, entity_effects=True).fit()
    assert_nan(res, ["rsquared", "rsquared_within", "corr_squared_within"])


@pytest.mark.parametrize("model", [PooledOLS, BetweenOLS])
def test_single_period_panel_has_no_within_r_squared(model):
    # Nothing is left once the means of the entities are removed
    index, _, xc = panel_data(nperiod=1)
    y = pd.Series(np.random.default_rng(4).standard_normal(NENTITY), index=index)
    res = model(y, xc).fit()
    assert_nan(res, ["rsquared_within", "corr_squared_within"])
    assert_finite(res, ["rsquared_overall"])


@pytest.mark.parametrize("name", ["PooledOLS", "BetweenOLS", "PanelOLS entity"])
def test_constant_only_model(name):
    index, _, xc = panel_data()
    const = xc[["const"]]
    build = {
        "PooledOLS": PooledOLS,
        "BetweenOLS": BetweenOLS,
        "PanelOLS entity": lambda y, x: PanelOLS(y, x, entity_effects=True),
    }[name]
    y = pd.Series(np.random.default_rng(5).standard_normal(len(index)) + 2, index=index)
    res = build(y, const).fit()
    # The constant explains nothing, whatever the data
    assert_allclose(res.rsquared_between, 0.0, atol=1e-12)
    assert_allclose(res.rsquared_overall, 0.0, atol=1e-12)
    assert_allclose(res.rsquared_within, 0.0, atol=1e-12)
    assert_allclose(res.rsquared, 0.0, atol=1e-12)
    # A model with only a constant has no R-squared when the dependent does not vary
    y = pd.Series(5.0, index=index, name="y")
    res = build(y, const).fit()
    assert_nan(res, PANEL_MEASURES)


def test_pooled_f_statistic_of_an_exact_fit():
    # The effects explain the dependent variable exactly, so that the residual sum of
    # squares is 0 and the F-statistic is not defined. It was a ZeroDivisionError
    index, x, _ = panel_data()
    for value in (5.0, 0.0):
        y = pd.Series(value, index=index, name="y")
        for effects in ({"entity_effects": True}, {"time_effects": True}):
            res = PanelOLS(y, x, **effects).fit()
            is_invalid = isinstance(res.f_pooled, InvalidTestStatistic)
            reason = str(res.f_pooled)
            assert is_invalid
            assert "residual sum of squares" in reason


@pytest.mark.parametrize("name", PANEL_WITH_CONSTANT)
def test_ordinary_values_of_panel_models(name):
    rg = np.random.default_rng(6)
    index, x, xc = panel_data()
    y = pd.Series(1 + (x @ [0.5, -0.3]) + rg.standard_normal(len(index)), index=index)
    weights = pd.Series(0.5 + rg.chisquare(3, len(index)), index=index)
    build, _ = PANEL_WITH_CONSTANT[name]
    for w in (None, weights):
        res = build(y, xc, w).fit()
        # 1 - RSS / TSS, with the operations of the model
        assert_allclose(res.rsquared, 1 - res.resid_ss / res.total_ss, rtol=1e-12)
        assert_finite(res, PANEL_MEASURES)
        for measure in PANEL_MEASURES:
            value = getattr(res, measure)
            assert value <= 1.0


def test_ordinary_values_against_manual_computations():
    rg = np.random.default_rng(7)
    index, x, xc = panel_data()
    y = pd.Series(1 + (x @ [0.5, -0.3]) + rg.standard_normal(len(index)), index=index)
    res = PooledOLS(y, xc).fit()
    params = res.params.to_numpy()
    yv, xv = y.to_numpy(), xc.to_numpy()
    resid = yv - xv @ params
    assert_allclose(res.rsquared, 1 - resid @ resid / ((yv - yv.mean()) ** 2).sum())
    # Within: the entity means are removed from the dependent and the regressors
    yw = (y - y.groupby(level=0).transform("mean")).to_numpy()
    xw = (xc - xc.groupby(level=0).transform("mean")).to_numpy()
    resid = yw - xw @ params
    assert_allclose(res.rsquared_within, 1 - resid @ resid / (yw @ yw))
    # Between: the entity means, with a constant
    ym = y.groupby(level=0).mean().to_numpy()
    xm = xc.groupby(level=0).mean().to_numpy()
    resid = ym - xm @ params
    tss = ((ym - ym.mean()) ** 2).sum()
    assert_allclose(res.rsquared_between, 1 - resid @ resid / tss)


# ---------------------------------------------------------------- IV models
NIV = 240


def iv_data(seed=0):
    rg = np.random.default_rng(seed)
    x = rg.standard_normal((NIV, 2))
    z = rg.standard_normal((NIV, 3))
    endog = (z @ [0.5, 0.3, 0.2] + rg.standard_normal(NIV))[:, None]
    exog = np.c_[np.ones(NIV), x]
    cats = pd.DataFrame({"c": pd.Categorical(rg.integers(0, 8, NIV))})
    return x, exog, endog, z, cats


IV_MODELS = {
    "OLS": lambda y, d, w: _OLS(y, d[1], weights=w),
    "IV2SLS": lambda y, d, w: IV2SLS(y, d[1], d[2], d[3], weights=w),
    "IVLIML": lambda y, d, w: IVLIML(y, d[1], d[2], d[3], weights=w),
    "IVJIVE": lambda y, d, w: IVJIVE(y, d[1], d[2], d[3], weights=w),
    "IVGMM": lambda y, d, w: IVGMM(y, d[1], d[2], d[3], weights=w),
    "AbsorbingLS": lambda y, d, w: AbsorbingLS(y, d[1], absorb=d[4], weights=w),
}
# IVGMM needs the inverse of the covariance of its moment conditions, which is singular
# if the residuals are 0
IV_FITS_EXACTLY = [name for name in IV_MODELS if name != "IVGMM"]


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("value", CONSTANTS)
@pytest.mark.parametrize("name", IV_FITS_EXACTLY)
def test_constant_dependent_variable_iv(name, value, weighted):
    data = iv_data()
    y = np.full(NIV, value)
    w = 0.5 + np.random.default_rng(1).chisquare(3, NIV) if weighted else None
    res = IV_MODELS[name](y, data, w).fit()
    attributes = ["rsquared", "rsquared_adj"]
    if name == "AbsorbingLS":
        attributes.append("absorbed_rsquared")
    assert_nan(res, attributes)
    assert_summary_has(res, "R-squared")


@pytest.mark.parametrize("name", ["OLS", "IV2SLS", "IVJIVE", "AbsorbingLS"])
def test_all_zero_dependent_variable_iv(name):
    # The residuals are 0, and so is the covariance of the parameters, which has no
    # inverse. The F-statistic is not defined
    res = IV_MODELS[name](np.zeros(NIV), iv_data(), None).fit()
    assert_nan(res, ["rsquared", "rsquared_adj"])
    is_invalid = isinstance(res.f_statistic, InvalidTestStatistic)
    assert is_invalid
    assert_summary_has(res, "R-squared")


def test_iv_model_without_a_constant_is_not_centered():
    x, _, endog, z, _ = iv_data()
    y = np.full(NIV, 5.0)
    res = IV2SLS(y, x, endog, z).fit()
    assert_allclose(res.total_ss, 25.0 * NIV)
    assert_allclose(res.rsquared, 1 - res.resid_ss / res.total_ss, rtol=1e-12)
    assert_finite(res, ["rsquared", "rsquared_adj"])


def test_absorbing_dependent_variable_that_is_explained_by_the_effects():
    _, exog, _, _, cats = iv_data()
    effects = np.random.default_rng(8).standard_normal(8) * 3
    y = effects[cats["c"].cat.codes.to_numpy()]
    res = AbsorbingLS(y, exog, absorb=cats).fit()
    # The effects explain the dependent variable, and nothing is left to explain
    assert_nan(res, ["absorbed_rsquared"])
    assert_allclose(res.rsquared, 1.0, atol=1e-10)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("name", IV_MODELS)
def test_ordinary_values_of_iv_models(name, weighted):
    rg = np.random.default_rng(9)
    data = iv_data()
    _, exog, endog, _, _ = data
    y = exog @ [1, 0.5, -0.3] + 0.4 * endog[:, 0] + rg.standard_normal(NIV)
    w = 0.5 + rg.chisquare(3, NIV) if weighted else None
    res = IV_MODELS[name](y, data, w).fit()
    assert_allclose(res.rsquared, 1 - res.resid_ss / res.total_ss, rtol=1e-12)
    n, k, c = res.nobs, res.df_model, int(res.has_constant)
    assert_allclose(
        res.rsquared_adj, 1 - ((n - c) / (n - k)) * (1 - res.rsquared), rtol=1e-12
    )
    assert_finite(res, ["rsquared", "rsquared_adj"])


def test_adjusted_r_squared_without_residual_degrees_of_freedom():
    # The adjusted R-squared is not defined if the model has as many parameters as
    # observations. It was a ZeroDivisionError, which the summary raised as well
    rg = np.random.default_rng(10)
    exog = np.c_[np.ones(4), rg.standard_normal((4, 3))]
    res = IV2SLS(rg.standard_normal(4), exog, None, None).fit()
    assert_allclose(res.rsquared, 1.0)
    assert_nan(res, ["rsquared_adj"])
    assert_summary_has(res, "Adj. R-squared")


# ------------------------------------------------------------- system models
def system_equations(y_a, y_b, instruments=False):
    _, exog, endog, z, _ = iv_data(seed=11)
    first = {"dependent": y_a, "exog": exog}
    if instruments:
        first.update({"endog": endog, "instruments": z})
    return {"a": first, "b": {"dependent": y_b, "exog": exog}}


SYSTEM_MODELS = {
    "SUR": SUR,
    "IV3SLS": IV3SLS,
    "IVSystemGMM": IVSystemGMM,
}
# The fits that do not need the inverse of a covariance that is estimated from residuals
# of 0, which does not exist: the covariance is given, or the estimator is OLS
SYSTEM_EXACT_FITS = {
    "SUR OLS": lambda eq: SUR(eq).fit(method="ols"),
    "SUR GLS": lambda eq: SUR(eq, sigma=np.eye(len(eq))).fit(),
    "IV3SLS OLS": lambda eq: IV3SLS(eq).fit(method="ols"),
    "IV3SLS GLS": lambda eq: IV3SLS(eq, sigma=np.eye(len(eq))).fit(),
}


@pytest.mark.parametrize("value", [0.0, 5.0, 0.1, np.pi, 1e10])
@pytest.mark.parametrize("name", SYSTEM_EXACT_FITS)
def test_all_dependent_variables_constant_system(name, value):
    eq = system_equations(
        np.full(NIV, value), np.full(NIV, 2 * value), "IV3SLS" in name
    )
    res = SYSTEM_EXACT_FITS[name](eq)
    assert_nan(res, ["rsquared"])
    all_nan = res.system_rsquared.isna().all()
    assert all_nan
    for equation in res.equations.values():
        assert_nan(equation, ["rsquared", "rsquared_adj"])
    assert_summary_has(res, "Overall R-squared")


@pytest.mark.parametrize("name", SYSTEM_EXACT_FITS)
def test_one_dependent_variable_constant_system(name):
    _, exog, _, _, _ = iv_data(seed=11)
    y_b = exog @ [0.5, 0.2, 0.1] + np.random.default_rng(12).standard_normal(NIV)
    eq = system_equations(np.full(NIV, 5.0), y_b, "IV3SLS" in name)
    res = SYSTEM_EXACT_FITS[name](eq)
    assert_nan(res.equations["a"], ["rsquared", "rsquared_adj"])
    assert_finite(res.equations["b"], ["rsquared", "rsquared_adj"])
    assert_finite(res, ["rsquared"])
    # The equation without a total sum of squares has no weight in the measures of the
    # system. Berndt's measure needs the covariance of all of the dependent variables
    system = res.system_rsquared
    assert_value_is_nan(system["berndt"])
    assert_allclose(system["dhrymes"], res.equations["b"].rsquared, rtol=1e-8)
    assert_value_is_finite(system["judge"])


@pytest.mark.parametrize("name", SYSTEM_MODELS)
def test_ordinary_values_of_system_models(name):
    rg = np.random.default_rng(13)
    _, exog, endog, _, _ = iv_data(seed=11)
    y_a = exog @ [1, 0.5, -0.3] + 0.4 * endog[:, 0] + rg.standard_normal(NIV)
    y_b = exog @ [0.5, 0.2, 0.1] + rg.standard_normal(NIV)
    res = SYSTEM_MODELS[name](system_equations(y_a, y_b, name != "SUR")).fit()
    assert_allclose(res.rsquared, 1.0 - res.resid_ss / res.total_ss, rtol=1e-12)
    for equation in res.equations.values():
        assert_allclose(
            equation.rsquared, 1.0 - equation.resid_ss / equation.total_ss, rtol=1e-12
        )
        assert_finite(equation, ["rsquared_adj"])
    defined = res.system_rsquared.notna().sum()
    assert defined >= 3


# --------------------------------------------------------- asset pricing models
@pytest.mark.parametrize(
    "model", [TradedFactorModel, LinearFactorModel, LinearFactorModelGMM]
)
def test_ordinary_values_of_asset_pricing_models(model):
    # The portfolios cannot be constant, which these models reject
    rg = np.random.default_rng(14)
    factors = rg.standard_normal((200, 2))
    portfolios = factors @ rg.standard_normal((2, 6)) + rg.standard_normal((200, 6))
    res = model(portfolios, factors).fit()
    assert_allclose(res.rsquared, 1 - res.residual_ss / res.total_ss, rtol=1e-12)
    assert 0.0 < res.rsquared < 1.0
    with pytest.raises(ValueError, match="portfolios must not contain a constant"):
        model(np.c_[portfolios, np.full(200, 0.5)], factors)
