"""
Tests of the t-statistics and p-values of parameters with a standard error of 0

The alphas of a model that is just identified are zero by construction, and their
standard errors are 0. The test of a parameter that is known is not defined, so that
its t-statistic and p-value are NaN, and not infinite with a p-value of 0.
"""

import re
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.tests.asset_pricing._utility import simulate_data

NFACTOR = 3
INFINITE = re.compile(r"(?<![A-Za-z])-?inf(?![A-Za-z])")


def just_identified_data(seed, risk_free):
    """As many portfolios as risk premia"""
    nportfolio = NFACTOR + int(risk_free)
    return simulate_data(250, nportfolio, NFACTOR, seed=seed, alpha=False)


def check_tests(res, zero):
    """The t-statistics and p-values are NaN where the standard error is 0"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tstats = np.asarray(res.tstats)
        pvalues = np.asarray(res.pvalues)
        text = str(res.full_summary) + str(res.summary)
    assert not np.any(np.isinf(tstats))
    assert np.all(np.isnan(tstats[zero]))
    assert np.all(np.isnan(pvalues[zero]))
    assert INFINITE.search(text) is None
    # Elsewhere they are what they were
    params = np.asarray(res.params)
    se = np.asarray(res.std_errors)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = params / se
    assert_allclose(tstats[~zero], ratio[~zero])
    assert_allclose(pvalues[~zero], 2 * stats.norm.sf(np.abs(tstats[~zero])))


@pytest.mark.parametrize("risk_free", [False, True])
def test_gmm_just_identified(risk_free):
    portfolios, factors = just_identified_data(0, risk_free)
    res = LinearFactorModelGMM(portfolios, factors, risk_free=risk_free).fit(disp=0)
    zero = np.asarray(res.std_errors == 0)
    assert zero[:, 0].all()
    assert not zero[:, 1:].any()
    check_tests(res, zero)
    assert np.all(np.isnan(res.tstats["alpha"]))
    assert np.all(np.isnan(res.pvalues["alpha"]))


@pytest.mark.parametrize("risk_free", [False, True])
def test_linear_factor_model_just_identified(risk_free):
    # The variances of the alphas are tiny numbers of either sign. The standard error
    # is 0 if the variance is negative, and a number near 0 if it is positive
    for seed in range(6):
        portfolios, factors = just_identified_data(seed, risk_free)
        res = LinearFactorModel(portfolios, factors, risk_free=risk_free).fit()
        zero = np.asarray(res.std_errors == 0)
        assert not zero[:, 1:].any()
        check_tests(res, zero)


def modified_results(known):
    portfolios, factors = just_identified_data(1, False)
    res = LinearFactorModel(portfolios, factors).fit()
    cov = np.array(res.cov)
    positions = np.arange(portfolios.shape[1]) * (NFACTOR + 1)
    sign = -1 if known else 1
    cov[positions, positions] = sign * 1e-13 * np.diag(cov).max()
    res._cov = cov
    return res


def test_alphas_with_negative_rounding_variances():
    res = modified_results(known=True)
    zero = np.asarray(res.std_errors == 0)
    assert zero[:, 0].all()
    assert not zero[:, 1:].any()
    check_tests(res, zero)
    assert np.all(np.isnan(res.tstats["alpha"]))


def test_alphas_with_positive_rounding_variances():
    # The standard error is not 0, so that the test is defined
    res = modified_results(known=False)
    zero = np.asarray(res.std_errors == 0)
    assert not zero.any()
    check_tests(res, zero)
    assert np.all(np.isfinite(res.tstats["alpha"]))
    assert np.all(np.abs(res.tstats["alpha"]) < 1e-3)
    assert_allclose(res.pvalues["alpha"], 1.0, atol=1e-3)


def test_risk_premia_with_a_standard_error_of_0():
    portfolios, factors = just_identified_data(2, False)
    res = TradedFactorModel(portfolios, factors).fit()
    rp_cov = np.array(res._rp_cov)
    rp_cov[0, :] = 0.0
    rp_cov[:, 0] = 0.0
    res._rp_cov = rp_cov
    se = np.asarray(res.risk_premia_se)
    assert_allclose(se[0], 0.0, atol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tstats = np.asarray(res.risk_premia_tstats)
        text = str(res.summary)
    assert np.isnan(tstats[0])
    expected = np.asarray(res.risk_premia)[1:] / se[1:]
    assert_allclose(tstats[1:], expected)
    assert INFINITE.search(text) is None


@pytest.mark.parametrize("model", [TradedFactorModel, LinearFactorModel])
def test_tests_are_not_changed(model):
    portfolios, factors = simulate_data(250, 7, 2, seed=6)
    res = model(portfolios, factors).fit()
    params = np.asarray(res.params)
    se = np.asarray(res.std_errors)
    assert_allclose(np.asarray(res.tstats), params / se, rtol=1e-12)
    assert_allclose(
        np.asarray(res.pvalues), 2 * stats.norm.sf(np.abs(params / se)), rtol=1e-12
    )
    rp = np.asarray(res.risk_premia)
    assert_allclose(np.asarray(res.risk_premia_tstats), rp / res.risk_premia_se)
