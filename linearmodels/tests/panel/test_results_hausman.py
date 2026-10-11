"""
Tests of RandomEffectsResults.hausman and HausmanTestResults: the return type and
the checks that the models are comparable.
"""

import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from scipy import stats
from statsmodels.tools.tools import add_constant

from linearmodels.datasets import wage_panel
from linearmodels.iv.results import IVResults
from linearmodels.panel import PanelOLS, RandomEffects
import linearmodels.panel.results
from linearmodels.panel.results import HausmanTestResults, RandomEffectsResults
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic

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
    assert repr(result) == text
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
    assert repr(result) == text
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
    assert str(loaded) == str(result)


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
    assert str(loaded) == str(result)


def test_hausman_pval(hausman_results):
    re_res, fe_res = hausman_results
    result = re_res.hausman(fe_res)
    assert_allclose(result.statistic.pval, stats.chi2.sf(result.statistic.stat, 2))
