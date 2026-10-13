"""
Tests of the standard errors of the alphas of LinearFactorModelGMM

The alphas are averages of moment conditions, so their standard errors use the
covariance of the average moment conditions at the estimates, which accounts for the
estimation of the parameters.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from linearmodels.asset_pricing.model import LinearFactorModelGMM
from linearmodels.iv.covariance import cov_kernel, kernel_weight_bartlett
from linearmodels.tests.asset_pricing._utility import simulate_data


@pytest.mark.parametrize("risk_free", [False, True])
@pytest.mark.parametrize("cov_type", ["robust", "kernel"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("debiased", [False, True])
def test_gmm_alpha_covariance_efficient(risk_free, cov_type, center, debiased):
    nportfolio, nfactor = 6, 2
    portfolios, factors = simulate_data(400, nportfolio, nfactor, seed=21, alpha=False)
    mod = LinearFactorModelGMM(portfolios, factors, risk_free=risk_free)
    config = {"bandwidth": 3} if cov_type == "kernel" else {}
    res = mod.fit(disp=0, cov_type=cov_type, center=center, debiased=debiased, **config)
    nobs = portfolios.shape[0]
    # The moments and the Jacobian at the estimates
    g = np.asarray(res._cov_est._xe)
    jac = np.asarray(res._cov_est.jacobian)
    assert g.shape[0] == nobs
    if cov_type == "robust":
        gc = g - g.mean(0) if center else g
        s = gc.T @ gc / nobs
    else:
        # The HAC estimate does not center the moments
        s = cov_kernel(g, kernel_weight_bartlett(3, nobs - 1))
    s = (s + s.T) / 2
    # The covariance of the average moments of efficient GMM, in the form M S M'
    s_inv = np.linalg.inv(s)
    proj = np.eye(s.shape[0]) - jac @ np.linalg.inv(jac.T @ s_inv @ jac) @ jac.T @ s_inv
    omega = proj @ s @ proj.T
    sel = slice(0, nportfolio * (nfactor + 1), nfactor + 1)
    expected = omega[sel, sel] / (nobs - int(debiased) * nfactor)

    alphas = np.asarray(res.alphas)
    assert_allclose(alphas, g.mean(0)[sel])
    assert_allclose(res.std_errors["alpha"], np.sqrt(np.diag(expected)), rtol=1e-7)
    assert_allclose(res.tstats["alpha"], alphas / np.sqrt(np.diag(expected)), rtol=1e-7)
    # The covariance has rank nportfolio - nfactor - nrf, as the J statistic has
    # degrees of freedom
    eigenvalues = np.linalg.eigvalsh(expected)
    rank = int(np.sum(eigenvalues > 1e-8 * eigenvalues.max()))
    assert rank == res.j_statistic.df == nportfolio - nfactor - int(risk_free)
    # The alphas are less variable than the average moments of which they are part
    sel_s = s[sel, sel] / (nobs - int(debiased) * nfactor)
    assert np.all(np.diag(expected) < np.diag(sel_s))


@pytest.mark.slow
def test_gmm_alpha_standard_errors_simulation():
    # One portfolio has a large beta, so that its alpha is almost pinned down by the
    # risk premium. Its sampling standard deviation is less than half of the standard
    # error that ignores the estimation of the parameters
    nobs, reps = 600, 50
    betas = np.array([[0.5, 0.7, 1.0, 3.0]])
    alphas, std_errors = [], []
    for i in range(reps):
        rng = np.random.default_rng(1000 + i)
        e = rng.standard_normal((nobs, 1))
        factors = np.sqrt(0.2**2 / 12) * (
            0.2 * rng.standard_normal((nobs, 1)) + 0.98 * e
        )
        factors = factors + 0.1 / 12
        idio = np.sqrt(0.1**2 / 12) * rng.standard_normal((nobs, 4))
        portfolios = factors @ betas + idio
        res = LinearFactorModelGMM(portfolios, factors).fit(disp=0, debiased=False)
        alphas.append(np.asarray(res.alphas))
        std_errors.append(np.asarray(res.std_errors["alpha"]))
    sampling_sd = np.std(alphas, axis=0, ddof=1)
    mean_se = np.mean(std_errors, axis=0)
    ratio = mean_se / sampling_sd
    assert np.all(ratio > 0.7)
    assert np.all(ratio < 1.4)


def test_gmm_just_identified_alphas():
    # As many portfolios as risk premia: the alphas are 0 and have no variance
    portfolios, factors = simulate_data(300, 2, 2, seed=5)
    res = LinearFactorModelGMM(portfolios, factors).fit(disp=0)
    assert res.j_statistic.df == 0
    assert_allclose(res.alphas, 0.0, atol=1e-6)
    assert np.all(np.isfinite(res.std_errors["alpha"]))
    assert_allclose(res.std_errors["alpha"], 0.0, atol=1e-5)
    # The t-statistics of the alphas are not finite, which the summary can format
    assert not np.all(np.isfinite(res.tstats["alpha"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert isinstance(str(res.full_summary), str)
