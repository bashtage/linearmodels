"""
Tests of the number of observations of LinearFactorModelGMM

There are nportfolio * (nfactor + 1) + nfactor moment conditions. Their covariance is
singular unless there are more observations than moment conditions. The estimator
raised a LinAlgError or returned a J-statistic that was NaN or arbitrary.
"""

import numpy as np
import pandas as pd
import pytest

from linearmodels.asset_pricing.model import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.shared.exceptions import MissingValueWarning

# (nportfolio, nfactor)
CASES = [(4, 1), (5, 2), (4, 3), (6, 1)]


def nmoment_of(nportfolio, nfactor):
    return nportfolio * (nfactor + 1) + nfactor


def make_data(nobs, nportfolio, nfactor, risk_free=False, seed=0):
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((nobs, nfactor)) * 0.05 + 0.01
    betas = rng.standard_normal((nfactor, nportfolio)) + 1
    portfolios = factors @ betas + 0.02 * rng.standard_normal((nobs, nportfolio))
    if risk_free:
        portfolios = portfolios + 0.004
    return portfolios, factors


@pytest.mark.parametrize("risk_free", [False, True])
@pytest.mark.parametrize(("nportfolio", "nfactor"), CASES)
def test_too_few_observations(nportfolio, nfactor, risk_free):
    nmoment = nmoment_of(nportfolio, nfactor)
    # Fewer observations than regressors, fewer than moment conditions, and as many as
    # moment conditions
    for nobs in (nfactor + 2, nmoment - 1, nmoment):
        portfolios, factors = make_data(nobs, nportfolio, nfactor, risk_free)
        match = (
            rf"observations \({nobs}\) must exceed the number of moment conditions "
            rf"\({nmoment}\)"
        )
        with pytest.raises(ValueError, match=match):
            LinearFactorModelGMM(portfolios, factors, risk_free=risk_free)


@pytest.mark.parametrize("risk_free", [False, True])
@pytest.mark.parametrize(("nportfolio", "nfactor"), CASES)
def test_one_more_observation_than_moment_conditions(nportfolio, nfactor, risk_free):
    nmoment = nmoment_of(nportfolio, nfactor)
    portfolios, factors = make_data(nmoment + 1, nportfolio, nfactor, risk_free)
    mod = LinearFactorModelGMM(portfolios, factors, risk_free=risk_free)
    assert mod.portfolios.shape[0] == nmoment + 1


@pytest.mark.parametrize("risk_free", [False, True])
def test_estimation_with_more_observations(risk_free):
    nportfolio, nfactor = 5, 2
    nmoment = nmoment_of(nportfolio, nfactor)
    portfolios, factors = make_data(3 * nmoment, nportfolio, nfactor, risk_free)
    res = LinearFactorModelGMM(portfolios, factors, risk_free=risk_free).fit(disp=0)
    assert np.all(np.isfinite(res.params))
    assert np.isfinite(res.j_statistic.stat)
    assert res.j_statistic.df == nportfolio - nfactor - int(risk_free)


def test_message_gives_the_numbers():
    portfolios, factors = make_data(10, 4, 2)
    with pytest.raises(ValueError, match=r"number of observations \(10\)") as exc:
        LinearFactorModelGMM(portfolios, factors)
    message = str(exc.value)
    assert "number of observations (10)" in message
    assert "moment conditions (14)" in message
    assert "nportfolio is 4 and nfactor is 2" in message


def test_missing_values_are_dropped_before_the_check():
    nportfolio, nfactor = 4, 1
    nmoment = nmoment_of(nportfolio, nfactor)
    portfolios, factors = make_data(nmoment + 3, nportfolio, nfactor)
    portfolios[:3, 0] = np.nan
    match = rf"observations \({nmoment}\) must exceed .* \({nmoment}\)"
    with pytest.warns(MissingValueWarning):
        with pytest.raises(ValueError, match=match):
            LinearFactorModelGMM(portfolios, factors)
    portfolios, factors = make_data(nmoment + 4, nportfolio, nfactor)
    portfolios[:3, 0] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = LinearFactorModelGMM(portfolios, factors)
    assert mod.portfolios.shape[0] == nmoment + 1


def test_from_formula():
    portfolios, factors = make_data(8, 4, 1)
    data = pd.DataFrame(np.c_[portfolios, factors], columns=list("abcde"))
    with pytest.raises(ValueError, match=r"observations \(8\) must exceed"):
        LinearFactorModelGMM.from_formula("a + b + c + d ~ e", data)
    portfolios, factors = make_data(10, 4, 1)
    data = pd.DataFrame(np.c_[portfolios, factors], columns=list("abcde"))
    mod = LinearFactorModelGMM.from_formula("a + b + c + d ~ e", data)
    assert mod.portfolios.shape == (10, 4)


def test_other_models_are_not_restricted():
    # Their covariances do not weight the moment conditions
    portfolios, factors = make_data(6, 5, 2)
    assert nmoment_of(5, 2) > 6
    LinearFactorModel(portfolios, factors)
    TradedFactorModel(portfolios, factors)
