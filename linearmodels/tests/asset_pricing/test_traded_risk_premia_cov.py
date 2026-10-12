"""
Tests of the covariance of the risk premia of TradedFactorModel

The risk premia are the sample means of the factors. When the covariance is debiased,
the covariance of the risk premia in cov and risk_premia_se use nobs - 1.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from linearmodels.asset_pricing.model import TradedFactorModel
from linearmodels.iv.covariance import cov_kernel, kernel_weight_bartlett
from linearmodels.tests.asset_pricing._utility import simulate_data


@pytest.mark.parametrize("cov_type", ["robust", "kernel"])
@pytest.mark.parametrize("debiased", [False, True])
def test_traded_risk_premia_covariance(cov_type, debiased):
    nobs, nportfolio, nfactor = 80, 5, 3
    portfolios, factors = simulate_data(nobs, nportfolio, nfactor, seed=7)
    config = {"bandwidth": 3} if cov_type == "kernel" else {}
    res = TradedFactorModel(portfolios, factors).fit(
        cov_type=cov_type, debiased=debiased, **config
    )
    nloading = (nfactor + 1) * nportfolio
    cov = np.asarray(res.cov)
    # The risk premia are sample means, so that their covariance has nobs - 1 when
    # debiased and nobs otherwise
    fe = factors - factors.mean(0)
    if cov_type == "robust":
        s = fe.T @ fe / nobs
    else:
        s = cov_kernel(fe, kernel_weight_bartlett(3, nobs - 1))
    dof = nobs - 1 if debiased else nobs
    expected = s / dof
    assert_allclose(cov[nloading:, nloading:], expected, rtol=1e-10)
    assert_allclose(res.risk_premia_se, np.sqrt(np.diag(expected)), rtol=1e-10)
    assert_allclose(np.diag(cov)[nloading:], res.risk_premia_se**2, rtol=1e-12)
    assert_allclose(res.risk_premia, factors.mean(0))
    if cov_type == "robust" and debiased:
        assert_allclose(cov[nloading:, nloading:], np.cov(factors.T) / nobs)

    # The loadings use nobs - nfactor - 1, and the covariance between the two is
    # scaled by the geometric mean, so that it remains a covariance matrix
    raw = np.asarray(
        TradedFactorModel(portfolios, factors)
        .fit(cov_type=cov_type, debiased=False, **config)
        .cov
    )
    scale = np.ones(nloading + nfactor)
    if debiased:
        scale[:nloading] = np.sqrt(nobs / (nobs - nfactor - 1))
        scale[nloading:] = np.sqrt(nobs / (nobs - 1))
    assert_allclose(cov, raw * np.outer(scale, scale), rtol=1e-10)
    assert_allclose(cov, cov.T, rtol=1e-14)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > -1e-12 * eigenvalues.max()


def test_traded_risk_premia_covariance_default_debiased():
    # The default is debiased, and the ratio of the covariances is the ratio of the
    # divisors
    nobs, nfactor = 60, 3
    portfolios, factors = simulate_data(nobs, 6, nfactor, seed=2)
    mod = TradedFactorModel(portfolios, factors)
    default, raw = mod.fit(), mod.fit(debiased=False)
    nloading = (nfactor + 1) * 6
    ratio = np.diag(default.cov)[nloading:] / np.diag(raw.cov)[nloading:]
    assert_allclose(ratio, nobs / (nobs - 1))
    ratio = np.diag(default.cov)[:nloading] / np.diag(raw.cov)[:nloading]
    assert_allclose(ratio, nobs / (nobs - nfactor - 1))
    assert_allclose(
        default.risk_premia_se**2 * (nobs - 1) / nobs, raw.risk_premia_se**2
    )
