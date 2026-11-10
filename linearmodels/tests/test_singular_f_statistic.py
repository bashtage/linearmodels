"""
The F-statistics of a model that fits exactly are invalid, they do not raise an error

A model fits exactly if its dependent variable is all zero, so that the parameters and the
residuals are 0, and the covariance of the parameters is 0 as well. It has no inverse,
which the Wald statistic of the model needs. It raised ``LinAlgError("Singular matrix")``
when the IV models and AbsorbingLS were estimated, and when the robust F-statistic of the
panel models was read, e.g., by the summary.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS, IVJIVE, AbsorbingLS
from linearmodels.iv.common import f_statistic
from linearmodels.iv.model import _OLS
from linearmodels.panel.model import FInfo, _deferred_f, _hotelling_f
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic

NOBS = 200
MESSAGE = "The covariance of the parameters is singular"


def assert_invalid(stat):
    is_invalid = isinstance(stat, InvalidTestStatistic)
    reason = str(stat)
    assert is_invalid
    assert MESSAGE in reason


def assert_wald(stat, expected):
    is_wald = isinstance(stat, WaldTestStatistic)
    assert is_wald
    assert_allclose(stat.stat, expected)


def iv_data():
    rg = np.random.default_rng(0)
    x = rg.standard_normal((NOBS, 2))
    z = rg.standard_normal((NOBS, 3))
    endog = (z @ [0.5, 0.3, 0.2] + rg.standard_normal(NOBS))[:, None]
    exog = np.c_[np.ones(NOBS), x]
    cats = pd.DataFrame({"c": pd.Categorical(rg.integers(0, 8, NOBS))})
    return exog, endog, z, cats


IV_MODELS = {
    "OLS": lambda y, d: _OLS(y, d[0]),
    "IV2SLS": lambda y, d: IV2SLS(y, d[0], d[1], d[2]),
    "IVJIVE": lambda y, d: IVJIVE(y, d[0], d[1], d[2]),
    "AbsorbingLS": lambda y, d: AbsorbingLS(y, d[0], absorb=d[3]),
    "AbsorbingLS without a constant": lambda y, d: AbsorbingLS(
        y, d[0][:, 1:], absorb=d[3]
    ),
}


@pytest.mark.parametrize("debiased", [True, False])
@pytest.mark.parametrize("cov", [np.zeros((3, 3)), np.ones((3, 3))], ids=["0", "1"])
def test_iv_f_statistic_with_a_singular_covariance(debiased, cov):
    params = np.array([[1.0], [2.0], [3.0]])
    stat = f_statistic(params, cov, debiased, 100, const_loc=0)
    assert_invalid(stat)
    is_nan = np.isnan(stat.stat)
    assert is_nan


def test_iv_f_statistic_with_a_covariance():
    params = np.array([[1.0], [2.0], [3.0]])
    stat = f_statistic(params, np.eye(3), False, 100, const_loc=0)
    assert_wald(stat, 13.0)


@pytest.mark.parametrize("name", IV_MODELS)
def test_iv_model_of_an_all_zero_dependent_variable(name):
    # The parameters, the residuals and the covariance are exactly 0
    res = IV_MODELS[name](np.zeros(NOBS), iv_data()).fit()
    assert_invalid(res.f_statistic)
    assert_allclose(res.params, 0.0)
    summary = str(res.summary)
    assert "F-statistic" in summary


def make_f_info(nvar):
    sel = np.ones(nvar, dtype=bool)
    return FInfo(sel, "Model F-statistic (robust)", None, False)


@pytest.mark.parametrize("debiased", [True, False])
def test_panel_f_statistic_with_a_singular_covariance(debiased):
    params = pd.Series([1.0, 2.0], index=["a", "b"])
    cov = pd.DataFrame(np.zeros((2, 2)), index=["a", "b"], columns=["a", "b"])
    assert_invalid(_deferred_f(params, cov, debiased, 100, make_f_info(2)))
    # The Wald test of FamaMacBeth, which needs more periods than parameters
    assert_invalid(_hotelling_f(params, cov, debiased, 50, make_f_info(2)))


def test_panel_f_statistic_with_a_covariance():
    params = pd.Series([1.0, 2.0], index=["a", "b"])
    cov = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    assert_wald(_deferred_f(params, cov, False, 100, make_f_info(2)), 5.0)
    assert_wald(_hotelling_f(params, cov, False, 50, make_f_info(2)), 5.0)
