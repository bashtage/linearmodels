"""
Tests of the validation of sigma, the covariance used for GLS by LinearFactorModel

sigma must be a symmetric positive definite array with one row and one column for each
test portfolio.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.asset_pricing.model import LinearFactorModel

NOBS = 200
NPORTFOLIO = 6


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    factors = rng.standard_normal((NOBS, 2))
    betas = rng.standard_normal((2, NPORTFOLIO))
    portfolios = factors @ betas + rng.standard_normal((NOBS, NPORTFOLIO))
    return portfolios, factors


@pytest.mark.parametrize(
    "shape", [(5, 5), (7, 7), (6, 5), (5, 6), (6, 6, 1), (36,), ()]
)
def test_wrong_shape(data, shape):
    portfolios, factors = data
    sigma = np.ones(shape) if shape else np.array(1.0)
    with pytest.raises(ValueError, match=r"sigma must be a square array with shape"):
        LinearFactorModel(portfolios, factors, sigma=sigma)


def test_wrong_shape_message(data):
    portfolios, factors = data
    match = r"shape \(6, 6\), where 6 is the number of test portfolios.*\(5, 5\)"
    with pytest.raises(ValueError, match=match):
        LinearFactorModel(portfolios, factors, sigma=np.eye(5))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_not_finite(data, bad):
    portfolios, factors = data
    sigma = np.eye(NPORTFOLIO)
    sigma[1, 2] = sigma[2, 1] = bad
    with pytest.raises(ValueError, match=r"sigma must not contain missing or infinite"):
        LinearFactorModel(portfolios, factors, sigma=sigma)


def test_not_symmetric(data):
    portfolios, factors = data
    sigma = np.eye(NPORTFOLIO)
    sigma[1, 2] = 0.2
    with pytest.raises(ValueError, match=r"sigma must be symmetric"):
        LinearFactorModel(portfolios, factors, sigma=sigma)


@pytest.mark.parametrize("kind", ["negative", "indefinite", "singular", "zero"])
def test_not_positive_definite(data, kind):
    portfolios, factors = data
    if kind == "negative":
        sigma = -np.eye(NPORTFOLIO)
    elif kind == "indefinite":
        sigma = np.eye(NPORTFOLIO)
        sigma[0, 0] = -1.0
    elif kind == "singular":
        sigma = np.eye(NPORTFOLIO)
        sigma[0, 1] = sigma[1, 0] = 1.0
    else:
        sigma = np.zeros((NPORTFOLIO, NPORTFOLIO))
    with pytest.raises(ValueError, match=r"sigma must be positive definite"):
        LinearFactorModel(portfolios, factors, sigma=sigma)


def test_sample_covariance_with_too_few_observations():
    # The covariance of 6 portfolios from 4 observations is singular
    rng = np.random.default_rng(1)
    portfolios = rng.standard_normal((4, NPORTFOLIO))
    factors = rng.standard_normal((4, 1))
    with pytest.raises(ValueError, match=r"sigma must be positive definite"):
        LinearFactorModel(portfolios, factors, sigma=np.cov(portfolios.T))


@pytest.mark.parametrize("kind", ["array", "frame", "list", "ints"])
def test_valid_sigma(data, kind):
    portfolios, factors = data
    sigma = np.cov(portfolios.T)
    if kind == "frame":
        sigma = pd.DataFrame(sigma)
    elif kind == "list":
        sigma = sigma.tolist()
    elif kind == "ints":
        sigma = np.diag(np.arange(1, NPORTFOLIO + 1))
    mod = LinearFactorModel(portfolios, factors, sigma=sigma)
    assert "using GLS" in str(mod)
    res = mod.fit()
    sigma_arr = np.asarray(sigma, dtype=float)
    # The second stage is GLS with the sigma that was given
    betas = np.asarray(res.betas)
    lam = np.linalg.solve(
        betas.T @ np.linalg.solve(sigma_arr, betas),
        betas.T @ np.linalg.solve(sigma_arr, portfolios.mean(0)),
    )
    assert_allclose(res.risk_premia, lam)


def test_symmetric_up_to_rounding(data):
    # A product of matrices is symmetric up to rounding
    portfolios, factors = data
    rng = np.random.default_rng(2)
    a = rng.standard_normal((NPORTFOLIO, NPORTFOLIO))
    sigma = a @ np.diag(np.arange(1.0, NPORTFOLIO + 1)) @ a.T
    sigma = sigma + 1e-14 * rng.standard_normal((NPORTFOLIO, NPORTFOLIO))
    res = LinearFactorModel(portfolios, factors, sigma=sigma).fit()
    assert np.all(np.isfinite(res.cov))


def test_from_formula_validates_sigma(data):
    portfolios, factors = data
    frame = pd.DataFrame(
        np.c_[portfolios, factors],
        columns=[f"p{i}" for i in range(NPORTFOLIO)] + ["f0", "f1"],
    )
    formula = " + ".join(f"p{i}" for i in range(NPORTFOLIO)) + " ~ f0 + f1"
    with pytest.raises(ValueError, match=r"sigma must be a square array"):
        LinearFactorModel.from_formula(formula, frame, sigma=np.eye(3))
    mod = LinearFactorModel.from_formula(formula, frame, sigma=np.eye(NPORTFOLIO))
    assert mod.portfolios.shape[1] == NPORTFOLIO
