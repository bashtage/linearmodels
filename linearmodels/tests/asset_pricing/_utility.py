import numpy as np
import pandas as pd
from scipy.linalg import null_space

from linearmodels.shared.utility import AttrDict


def generate_data(
    nfactor=3, nportfolio=25, nobs=1000, premia=None, output="pandas", alpha=False
):
    np.random.seed(12345)
    if premia is None:
        premia = np.arange(1, nfactor + 1) / (10 * nfactor)
    rho = 0.2
    e = np.random.randn(nobs, nfactor)
    factors = rho * np.random.randn(nobs, 1) + np.sqrt(1 - rho**2) * e
    factors = np.sqrt(0.20**2 / 12) * factors
    factors += premia[None, :] / 12
    idio = np.sqrt(0.10**2 / 12) * np.random.randn(nobs, nportfolio)
    betas = np.random.chisquare(2, (nfactor, nportfolio)) / 2.0
    portfolios = factors @ betas + idio
    if alpha:
        portfolios += np.arange(nportfolio)[None, :] / nportfolio / 100
    index = pd.date_range("1930-1-1", periods=nobs, freq="D")
    if output == "pandas":
        cols = [f"factor_{i:0>2}" for i in range(1, nfactor + 1)]
        factors = pd.DataFrame(factors, columns=cols, index=index)
        cols = [f"port_{i:0>2}" for i in range(1, nportfolio + 1)]
        portfolios = pd.DataFrame(portfolios, columns=cols, index=index)

    return AttrDict(factors=factors, portfolios=portfolios)


def simulate_data(nobs=300, nportfolio=7, nfactor=2, seed=0, alpha=True):
    """
    Simulate returns that have a factor structure, with a local random generator.

    The portfolios have pricing errors, which increase with the portfolio number, if
    alpha is True.
    """
    rng = np.random.default_rng(seed)
    rho = 0.2
    e = rng.standard_normal((nobs, nfactor))
    factors = rho * rng.standard_normal((nobs, 1)) + np.sqrt(1 - rho**2) * e
    factors = np.sqrt(0.20**2 / 12) * factors
    factors += (np.arange(1, nfactor + 1) / (10 * nfactor) / 12)[None, :]
    idio = np.sqrt(0.10**2 / 12) * rng.standard_normal((nobs, nportfolio))
    betas = rng.chisquare(2, (nfactor, nportfolio)) / 2.0
    portfolios = factors @ betas + idio
    if alpha:
        portfolios = portfolios + (np.arange(nportfolio) / nportfolio / 100)[None, :]
    return portfolios, factors


def get_all(res):
    attrs = dir(res)
    for attr_name in attrs:
        if attr_name.startswith("_"):
            continue
        attr = getattr(res, attr_name)
        assert not callable(attr)


def textbook_j_stat(alphas, vcv, restrictions):
    """
    The J statistic of the pricing errors of a two-pass estimator.

    The pricing errors satisfy restrictions.T @ alphas = 0, so that they are in the
    space orthogonal to the columns of restrictions, which has dimension
    nportfolio - nrestriction. The statistic is the Wald statistic of the
    coordinates of the pricing errors in that space, (Q'a)'(Q'VQ)^-1(Q'a), where the
    columns of Q are a basis of that space.
    """
    q = null_space(restrictions.T)
    qa = q.T @ alphas
    return float(np.squeeze(qa.T @ np.linalg.solve(q.T @ vcv @ q, qa)))
