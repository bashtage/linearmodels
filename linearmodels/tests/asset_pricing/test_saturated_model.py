"""
Tests of models with as many observations as regressors in the time-series regressions

With nobs equal to nfactor + 1 the regressions have no residual degrees of freedom,
so the covariance cannot be debiased.
"""

import numpy as np
import pytest

from linearmodels.asset_pricing.model import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)

NFACTOR = 2
NPORTFOLIO = 5


def make_data(nobs):
    rng = np.random.default_rng(0)
    factors = rng.standard_normal((nobs, NFACTOR))
    betas = rng.standard_normal((NFACTOR, NPORTFOLIO))
    portfolios = factors @ betas + 0.1 * rng.standard_normal((nobs, NPORTFOLIO))
    return portfolios, factors


@pytest.mark.parametrize("model", [TradedFactorModel, LinearFactorModel])
@pytest.mark.parametrize("fit_options", [{}, {"debiased": True}])
def test_saturated_model_cannot_be_debiased(model, fit_options):
    portfolios, factors = make_data(NFACTOR + 1)
    mod = model(portfolios, factors)
    match = r"cannot be debiased.*\(3\).*\(3\).*Use debiased=False"
    with pytest.raises(ValueError, match=match):
        mod.fit(**fit_options)
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="kernel", bandwidth=1, **fit_options)


@pytest.mark.parametrize("model", [TradedFactorModel, LinearFactorModel])
def test_saturated_model_without_debiasing(model):
    portfolios, factors = make_data(NFACTOR + 1)
    res = model(portfolios, factors).fit(debiased=False)
    assert res.nobs == NFACTOR + 1
    assert np.all(np.isfinite(res.params))


@pytest.mark.parametrize("model", [TradedFactorModel, LinearFactorModel])
def test_one_more_observation_can_be_debiased(model):
    portfolios, factors = make_data(NFACTOR + 2)
    res = model(portfolios, factors).fit()
    assert np.all(np.isfinite(res.cov))


def test_gmm_starting_values_are_not_debiased(monkeypatch):
    # The starting values come from LinearFactorModel, and do not use the covariance,
    # so that they must not fail if the first stage has no degrees of freedom
    portfolios, factors = make_data(40)
    calls = []
    fit = LinearFactorModel.fit

    def record(self, *args, **kwargs):
        calls.append(kwargs.get("debiased", True))
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(LinearFactorModel, "fit", record)
    LinearFactorModelGMM(portfolios, factors).fit(disp=0)
    assert calls == [False]
