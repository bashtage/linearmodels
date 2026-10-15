"""
Tests of the standard errors when a variance is zero up to rounding

The alphas of a model that is just identified are zero, and so are their variances.
Rounding makes those variances tiny positive or negative numbers, and the square root
of a negative number is NaN. A variance that is negative by no more than a small
fraction of the largest variance is zero up to rounding, and its standard error is 0. A
variance that is more negative is not hidden.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.asset_pricing.results import _VARIANCE_RTOL, _std_errors
from linearmodels.tests.asset_pricing._utility import simulate_data

NFACTOR = 3
SCALES = [1e-8, 1.0, 1e8]


def just_identified_data(seed, risk_free, nobs=250):
    """As many portfolios as risk premia"""
    nportfolio = NFACTOR + int(risk_free)
    return simulate_data(nobs, nportfolio, NFACTOR, seed=seed, alpha=False)


def alpha_positions(nportfolio):
    """Positions of the alphas on the diagonal of the covariance of a linear model"""
    return np.arange(nportfolio) * (NFACTOR + 1)


@pytest.mark.parametrize("scale", SCALES)
def test_std_errors_rounding(scale):
    # Zero, and negative by much less than the tolerance, relative to the largest
    cov = scale * np.diag([4.0, 1.0, 0.0, -1e-14, -0.5 * _VARIANCE_RTOL * 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se = _std_errors(cov)
    assert_allclose(se, np.sqrt(scale) * np.array([2.0, 1.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("scale", SCALES)
def test_std_errors_tolerance_is_relative(scale):
    largest = 4.0 * scale
    inside = np.diag([largest, scale, -0.99 * _VARIANCE_RTOL * largest])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_allclose(_std_errors(inside)[2], 0.0, atol=1e-12 * np.sqrt(largest))
    outside = np.diag([largest, scale, -1.01 * _VARIANCE_RTOL * largest])
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        se = _std_errors(outside)
    assert np.isnan(se[2])
    assert_allclose(se[:2], np.sqrt([largest, scale]))


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("fraction", [1e-3, 1e-1, 1.0])
def test_std_errors_negative_variance_is_not_hidden(scale, fraction):
    # The fraction of the largest variance is a long way beyond rounding
    largest = 4.0 * scale
    cov = np.diag([largest, scale, -fraction * largest])
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        se = _std_errors(cov)
    assert np.isnan(se[2])
    assert_allclose(se[:2], np.sqrt([largest, scale]))


def test_std_errors_not_changed():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 6))
    cov = x @ x.T
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se = _std_errors(cov)
    assert_allclose(se, np.sqrt(np.diag(cov)), rtol=1e-12)
    # A matrix is not changed, and NaN is not hidden
    assert_allclose(cov, x @ x.T, rtol=1e-12, atol=1e-12)
    cov[2, 2] = np.nan
    se = _std_errors(cov)
    assert np.isnan(se[2])
    others = np.delete(se, 2)
    assert np.all(np.isfinite(others))


@pytest.mark.parametrize("risk_free", [False, True])
@pytest.mark.parametrize("cov_type", ["robust", "kernel"])
def test_just_identified_linear_factor_model(risk_free, cov_type):
    config = {"bandwidth": 3} if cov_type == "kernel" else {}
    for seed in range(12):
        portfolios, factors = just_identified_data(seed, risk_free)
        res = LinearFactorModel(portfolios, factors, risk_free=risk_free).fit(
            cov_type=cov_type, **config
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            se = res.std_errors
            assert "Coefficients" in str(res.full_summary)
            assert "Risk Premia" in str(res.summary)
        assert np.all(np.isfinite(np.asarray(se)))
        # The alphas are known: their variance is zero, up to rounding
        cov_diag = np.diag(np.asarray(res.cov))
        positions = alpha_positions(portfolios.shape[1])
        assert np.all(np.abs(cov_diag[positions]) < _VARIANCE_RTOL * cov_diag.max())
        negative = cov_diag[positions] < 0
        assert_allclose(np.asarray(se["alpha"])[negative], 0.0, atol=1e-12)


@pytest.mark.parametrize("risk_free", [False, True])
def test_rounding_in_the_variance_of_the_alphas(risk_free):
    # Deterministic version of the test above: the variances of the alphas are
    # negative by rounding
    portfolios, factors = just_identified_data(0, risk_free)
    res = LinearFactorModel(portfolios, factors, risk_free=risk_free).fit()
    with np.errstate(invalid="ignore"):
        base = np.sqrt(np.diag(np.asarray(res.cov)))
    cov = np.array(res.cov)
    positions = alpha_positions(portfolios.shape[1])
    cov[positions, positions] = -1e-13 * np.diag(cov).max()
    res._cov = cov
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se = res.std_errors
        assert "Coefficients" in str(res.full_summary)
    assert_allclose(np.asarray(se["alpha"]), 0.0, atol=1e-12)
    loadings = np.asarray(se.iloc[:, 1:])
    expected = base[: loadings.shape[0] * (NFACTOR + 1)].reshape(-1, NFACTOR + 1)
    assert_allclose(loadings, expected[:, 1:])


def test_real_negative_variance_of_an_alpha_is_not_hidden():
    portfolios, factors = just_identified_data(1, False)
    res = LinearFactorModel(portfolios, factors).fit()
    cov = np.array(res.cov)
    cov[0, 0] = -1e-3 * np.diag(cov).max()
    res._cov = cov
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        se = res.std_errors
    assert np.isnan(se.iloc[0, 0])
    assert np.all(np.isfinite(np.asarray(se)[1:]))
    assert np.all(np.isfinite(np.asarray(se)[0, 1:]))


def test_risk_premia_se_rounding():
    portfolios, factors = just_identified_data(2, False)
    res = TradedFactorModel(portfolios, factors).fit()
    base = np.asarray(res.risk_premia_se)
    rp_cov = np.array(res._rp_cov)
    rp_cov[0, 0] = -1e-13 * rp_cov.max()
    res._rp_cov = rp_cov
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se = np.asarray(res.risk_premia_se)
    assert_allclose(se[0], 0.0, atol=1e-12)
    assert_allclose(se[1:], base[1:])
    rp_cov[0, 0] = -1e-3 * rp_cov.max()
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        se = np.asarray(res.risk_premia_se)
    assert np.isnan(se[0])


def gmm_results(nportfolio):
    portfolios, factors = simulate_data(250, nportfolio, 1, seed=4, alpha=False)
    return LinearFactorModelGMM(portfolios, factors).fit(disp=0)


def test_gmm_alpha_standard_errors_rounding():
    res = gmm_results(4)
    base = res.std_errors
    alpha_vcv = np.array(res._alpha_vcv)
    # Rounding in the variance of one alpha, which is small relative to the others
    alpha_vcv[0, 0] = -1e-13 * np.diag(alpha_vcv).max()
    res._alpha_vcv = alpha_vcv
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se = res.std_errors
    assert_allclose(se.iloc[0, 0], 0.0, atol=1e-12)
    assert_allclose(np.asarray(se)[1:], np.asarray(base)[1:])
    assert_allclose(np.asarray(se)[:, 1:], np.asarray(base)[:, 1:])
    # A variance that is negative by more than rounding is not hidden
    alpha_vcv[0, 0] = -1e-2 * np.diag(alpha_vcv).max()
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        se = res.std_errors
    assert np.isnan(se.iloc[0, 0])


@pytest.mark.parametrize("model", [TradedFactorModel, LinearFactorModel])
def test_standard_errors_are_square_roots_of_the_diagonal(model):
    # Variances that are not zero are not changed
    portfolios, factors = simulate_data(250, 7, 2, seed=6)
    res = model(portfolios, factors).fit()
    expected = np.sqrt(np.diag(np.asarray(res.cov)))
    nloading = 7 * 3
    assert_allclose(np.asarray(res.std_errors).ravel(), expected[:nloading], rtol=1e-12)
    assert_allclose(
        np.asarray(res.risk_premia_se), np.sqrt(np.diag(res._rp_cov)), rtol=1e-12
    )
