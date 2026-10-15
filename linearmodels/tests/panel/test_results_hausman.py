"""
Tests of RandomEffectsResults.hausman and HausmanTestResults: the return type and
the checks that the models are comparable.
"""

import pickle
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from scipy import stats
from statsmodels.tools.tools import add_constant

from linearmodels.datasets import wage_panel
from linearmodels.iv.results import IVResults
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
import linearmodels.panel.results
from linearmodels.panel.results import (
    HausmanTestResults,
    PanelEffectsResults,
    RandomEffectsResults,
)
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.tests._utility import assert_same_text

HAUSMAN_NAMES = ["exper", "expersq"]


@pytest.fixture(scope="module")
def wage():
    return wage_panel.load().set_index(["nr", "year"])


@pytest.fixture(scope="module")
def hausman_results(wage):
    exog = add_constant(wage[HAUSMAN_NAMES])
    re_res = RandomEffects(wage["hours"], exog).fit()
    fe_res = PanelOLS(wage["hours"], exog, entity_effects=True).fit()
    return re_res, fe_res


def hausman_by_hand(re_res, fe_res, names=HAUSMAN_NAMES):
    delta = (fe_res.params - re_res.params)[names].to_numpy()
    diff = (fe_res.cov - re_res.cov).loc[names, names].to_numpy()
    return float(delta @ np.linalg.pinv(diff) @ delta)


def negative_statistic_results():
    # In small samples Var(b0) - Var(b1) is often indefinite and the statistic
    # can be negative
    rg = np.random.default_rng(3)
    index = pd.MultiIndex.from_product([np.arange(12), np.arange(3)])
    effects = np.repeat(rg.standard_normal(12), 3)
    x = pd.DataFrame(rg.standard_normal((36, 2)), index=index, columns=["x0", "x1"])
    y = pd.Series(
        x.sum(axis=1).to_numpy() + effects + rg.standard_normal(36), index=index
    )
    fe_res = PanelOLS(y, x, entity_effects=True).fit()
    re_res = RandomEffects(y, add_constant(x)).fit()
    return re_res, fe_res


def test_hausman_replaces_wu_hausman(hausman_results):
    re_res, _ = hausman_results
    # The method was never released, so there is no alias
    assert not hasattr(RandomEffectsResults, "wu_hausman")
    assert not hasattr(re_res, "wu_hausman")
    assert callable(re_res.hausman)
    # IVResults.wu_hausman is a different test that keeps its name
    assert callable(IVResults.wu_hausman)


def test_hausman_results_exported():
    assert "HausmanTestResults" in linearmodels.panel.results.__all__


def test_hausman_named_tuple(hausman_results):
    re_res, fe_res = hausman_results
    result = re_res.hausman(fe_res)
    assert isinstance(result, HausmanTestResults)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result._fields == ("statistic", "estimates")

    # Field access and indexing
    assert isinstance(result.statistic, WaldTestStatistic)
    assert not isinstance(result.statistic, InvalidTestStatistic)
    assert isinstance(result.estimates, pd.DataFrame)
    assert result[0] is result.statistic
    assert result[1] is result.estimates
    assert_allclose(result.statistic.stat, hausman_by_hand(re_res, fe_res))
    assert list(result.estimates.columns) == ["b0", "b1", "b0-b1", "Std. Err."]
    assert list(result.estimates.index) == HAUSMAN_NAMES

    # Unpacking
    statistic, estimates = re_res.hausman(fe_res)
    assert isinstance(statistic, WaldTestStatistic)
    assert_frame_equal(estimates, result.estimates)
    assert_allclose(statistic.stat, result.statistic.stat)


def test_hausman_str_repr(hausman_results):
    re_res, fe_res = hausman_results
    result = re_res.hausman(fe_res)
    text = str(result)
    assert_same_text(repr(result), text)
    # The statistic comes first, followed by the estimates
    assert text.startswith(str(result.statistic))
    assert "Hausman specification test" in text
    assert "H0: No systematic difference in coefficients between models" in text
    assert f"Statistic: {result.statistic.stat:0.4f}" in text
    assert f"P-value: {result.statistic.pval:0.4f}" in text
    assert "Distributed: chi2(2)" in text
    assert text.index("Statistic:") < text.index("Coefficients")
    assert text.endswith(str(result.estimates))
    for label in [*result.estimates.columns, *result.estimates.index]:
        assert label in text
    # No object ids, which would make the repr change from run to run
    assert "id:" not in text
    assert "0x" not in text


def test_hausman_str_repr_invalid():
    re_res, fe_res = negative_statistic_results()
    with pytest.warns(UserWarning, match="not positive definite"):
        result = re_res.hausman(fe_res)
    assert isinstance(result.statistic, InvalidTestStatistic)
    text = str(result)
    assert_same_text(repr(result), text)
    assert text.startswith(str(result.statistic))
    assert "Invalid test statistic" in text
    assert "negative" in text
    assert text.endswith(str(result.estimates))
    assert "x0" in text
    assert "x1" in text


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_hausman_pickle(hausman_results, protocol):
    re_res, fe_res = hausman_results
    result = re_res.hausman(fe_res)
    loaded = pickle.loads(pickle.dumps(result, protocol=protocol))
    assert type(loaded) is HausmanTestResults
    assert isinstance(loaded.statistic, WaldTestStatistic)
    assert loaded.statistic.stat == result.statistic.stat
    assert loaded.statistic.pval == result.statistic.pval
    assert loaded.statistic.df == result.statistic.df
    assert loaded.statistic.dist_name == result.statistic.dist_name
    assert loaded.statistic.null == result.statistic.null
    assert_frame_equal(loaded.estimates, result.estimates)
    assert_same_text(str(loaded), str(result))


def test_hausman_pickle_invalid():
    re_res, fe_res = negative_statistic_results()
    with pytest.warns(UserWarning, match="not positive definite"):
        result = re_res.hausman(fe_res)
    loaded = pickle.loads(pickle.dumps(result))
    assert type(loaded) is HausmanTestResults
    assert isinstance(loaded.statistic, InvalidTestStatistic)
    assert np.isnan(loaded.statistic.stat)
    assert np.isnan(loaded.statistic.pval)
    assert_frame_equal(loaded.estimates, result.estimates)
    assert_same_text(str(loaded), str(result))


def test_hausman_pval(hausman_results):
    re_res, fe_res = hausman_results
    result = re_res.hausman(fe_res)
    assert_allclose(result.statistic.pval, stats.chi2.sf(result.statistic.stat, 2))


def weighted_results(wage, re_weights, fe_weights):
    n = wage.shape[0]
    base = pd.Series(1.0 + (np.arange(n) % 5), index=wage.index)
    exog = add_constant(wage[HAUSMAN_NAMES])
    re_res = RandomEffects(wage["hours"], exog, weights=re_weights(base)).fit()
    fe_res = PanelOLS(
        wage["hours"], exog, entity_effects=True, weights=fe_weights(base)
    ).fit()
    return re_res, fe_res


def unweighted(base):
    return None


def weighted(base):
    return base


@pytest.mark.parametrize(
    ("re_weights", "fe_weights"),
    [(unweighted, weighted), (weighted, unweighted)],
    ids=["fe-weighted", "re-weighted"],
)
def test_hausman_different_weights(wage, re_weights, fe_weights):
    re_res, fe_res = weighted_results(wage, re_weights, fe_weights)
    with pytest.raises(ValueError, match="same weights"):
        re_res.hausman(fe_res)


def test_hausman_different_weight_values(wage):
    re_res, fe_res = weighted_results(wage, weighted, lambda base: base[::-1].array)
    with pytest.raises(ValueError, match="same weights"):
        re_res.hausman(fe_res)


def test_hausman_same_weights(wage):
    re_res, fe_res = weighted_results(wage, weighted, weighted)
    result = re_res.hausman(fe_res)
    assert isinstance(result.statistic, WaldTestStatistic)
    assert not isinstance(result.statistic, InvalidTestStatistic)
    assert_allclose(result.statistic.stat, hausman_by_hand(re_res, fe_res))
    # The weights are only relevant up to scale
    scaled = weighted_results(wage, weighted, lambda base: 3.0 * base)
    scaled_result = scaled[0].hausman(scaled[1])
    assert_allclose(scaled_result.statistic.stat, result.statistic.stat, rtol=1e-8)


def test_hausman_different_dependent(wage):
    exog = add_constant(wage[HAUSMAN_NAMES])
    fe_res = PanelOLS(wage["lwage"], exog, entity_effects=True).fit()
    re_res = RandomEffects(wage["hours"], exog).fit()
    with pytest.raises(ValueError, match="same dependent variable"):
        re_res.hausman(fe_res)
    # A transformed dependent variable is also a different dependent variable
    re_res = RandomEffects(np.exp(wage["lwage"]), exog).fit()
    with pytest.raises(ValueError, match="same dependent variable"):
        re_res.hausman(fe_res)


def test_hausman_renamed_dependent(wage):
    exog = add_constant(wage[HAUSMAN_NAMES])
    fe_res = PanelOLS(wage["hours"], exog, entity_effects=True).fit()
    re_res = RandomEffects(wage["hours"].rename("other_name"), exog).fit()
    result = re_res.hausman(fe_res)
    assert isinstance(result.statistic, WaldTestStatistic)
    assert_allclose(result.statistic.stat, hausman_by_hand(re_res, fe_res))


def test_hausman_different_rows_same_count(wage):
    # The first entity has 8 observations, as does the last entity
    exog = add_constant(wage[HAUSMAN_NAMES])
    nobs = wage.shape[0]
    first = RandomEffects(wage["hours"].iloc[8:], exog.iloc[8:]).fit()
    last = PanelOLS(
        wage["hours"].iloc[: nobs - 8], exog.iloc[: nobs - 8], entity_effects=True
    ).fit()
    assert first.nobs == last.nobs
    with pytest.raises(ValueError, match="same observations"):
        first.hausman(last)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_hausman_non_finite_is_invalid(wage, bad):
    exog = add_constant(wage[HAUSMAN_NAMES])
    re_res = RandomEffects(wage["hours"], exog).fit()
    fe_res = PanelOLS(wage["hours"], exog, entity_effects=True).fit()
    fe_res.cov = fe_res.cov * bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = re_res.hausman(fe_res)
    assert isinstance(result.statistic, InvalidTestStatistic)
    assert np.isnan(result.statistic.stat)
    assert np.isnan(result.statistic.pval)
    assert "not finite" in str(result.statistic)
    assert list(result.estimates.index) == HAUSMAN_NAMES
    assert not np.isfinite(result.estimates["Std. Err."]).any()
    assert "Invalid test statistic" in str(result)


def test_hausman_non_finite_coefficients_is_invalid(wage, monkeypatch):
    exog = add_constant(wage[HAUSMAN_NAMES])
    re_res = RandomEffects(wage["hours"], exog).fit()
    fe_res = PanelOLS(wage["hours"], exog, entity_effects=True).fit()
    params = fe_res.params.copy()
    params["exper"] = np.nan
    monkeypatch.setattr(PanelEffectsResults, "params", property(lambda self: params))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = re_res.hausman(fe_res)
    assert isinstance(result.statistic, InvalidTestStatistic)
    assert "not finite" in str(result.statistic)


def test_hausman_pooled_is_invalid(wage):
    # The unadjusted covariance of the pooled model ignores the entity effects
    # and it is not consistent under the alternative
    exog = add_constant(wage[["expersq", "union", "married"]])
    re_res = RandomEffects(wage["lwage"], exog).fit()
    pooled = PooledOLS(wage["lwage"], exog).fit()
    with pytest.warns(UserWarning, match="not positive definite"):
        result = re_res.hausman(pooled)
    assert isinstance(result.statistic, InvalidTestStatistic)
    assert np.isnan(result.statistic.stat)
    assert np.isnan(result.statistic.pval)
    assert "negative" in str(result.statistic)
    assert "do not meet the assumptions of the Hausman test" in str(result.statistic)
    assert list(result.estimates.index) == ["expersq", "union", "married"]


def test_hausman_two_way_fixed_effects(wage):
    # A model with time effects is accepted as other. The test then also
    # examines the time effects, so the dummies should be in this model.
    years = pd.get_dummies(
        wage.index.get_level_values("year"), prefix="y", drop_first=True, dtype=float
    )
    years.index = wage.index
    names = ["expersq", "union", "married"]
    exog = add_constant(wage[names])
    fe_res = PanelOLS(wage["lwage"], exog, entity_effects=True, time_effects=True).fit()

    re_plain = RandomEffects(wage["lwage"], exog).fit()
    plain = re_plain.hausman(fe_res)
    assert isinstance(plain.statistic, WaldTestStatistic)
    assert list(plain.estimates.index) == names

    re_dummies = RandomEffects(wage["lwage"], pd.concat([exog, years], axis=1)).fit()
    with_dummies = re_dummies.hausman(fe_res)
    assert isinstance(with_dummies.statistic, WaldTestStatistic)
    # Only the coefficients that are in both models are compared
    assert list(with_dummies.estimates.index) == names
    assert with_dummies.statistic.df == len(names)
    assert_allclose(
        with_dummies.statistic.stat, hausman_by_hand(re_dummies, fe_res, names)
    )
    assert not np.isclose(with_dummies.statistic.stat, plain.statistic.stat)
