import numpy as np
import pandas as pd
import pytest

from linearmodels.asset_pricing.model import (LinearFactorModel,
                                              LinearFactorModelGMM,
                                              TradedFactorModel)
from linearmodels.compat.pandas import assert_frame_equal
from linearmodels.tests.asset_pricing._utility import generate_data

FORMULA_FACTORS = 'factor_1 + factor_2 + factor_3'
FORMULA_PORT = 'port_1 + port_2 + port_3 + port_4 + port_5 + port_6 + port_7 + ' \
               'port_8 + port_9 + port_10'
FORMULA = ' ~ '.join((FORMULA_PORT, FORMULA_FACTORS))


@pytest.fixture(scope='module', params=[TradedFactorModel, LinearFactorModel,
                                        LinearFactorModelGMM])
def model(request):
    return request.param


@pytest.fixture(scope='module', params=[LinearFactorModel, LinearFactorModelGMM])
def non_traded_model(request):
    return request.param


@pytest.fixture(scope='module')
def data(request):
    premia = np.array([.1, .1, .1])
    out = generate_data(nportfolio=10, output='pandas', alpha=True, premia=premia)
    out['joined'] = pd.concat([out.factors, out.portfolios], 1)
    return out


def test_traded_model_formula(data, model):
    mod1 = model.from_formula(FORMULA, data.joined)
    mod2 = model(data.portfolios, data.factors)
    res1 = mod1.fit()
    res2 = mod2.fit()
    assert_frame_equal(mod1.factors.pandas, mod2.factors.pandas)
    assert_frame_equal(mod1.portfolios.pandas, mod2.portfolios.pandas)
    assert_frame_equal(res1.params, res2.params)
    assert mod1.formula == FORMULA
    assert mod2.formula is None

    mod1 = model.from_formula(FORMULA_FACTORS, data.joined, portfolios=data.portfolios)
    mod2 = model(data.portfolios, data.factors)
    res1 = mod1.fit()
    res2 = mod2.fit()
    assert_frame_equal(mod1.factors.pandas, mod2.factors.pandas)
    assert_frame_equal(mod1.portfolios.pandas, mod2.portfolios.pandas)
    assert_frame_equal(res1.params, res2.params)
    assert mod1.formula == FORMULA_FACTORS
    assert mod2.formula is None


@pytest.mark.slow
def test_non_traded_risk_free(data, non_traded_model):
    mod1 = non_traded_model.from_formula(FORMULA, data.joined, risk_free=True)
    mod2 = non_traded_model(data.portfolios, data.factors, risk_free=True)
    res1 = mod1.fit()
    res2 = mod2.fit()
    assert_frame_equal(mod1.factors.pandas, mod2.factors.pandas)
    assert_frame_equal(mod1.portfolios.pandas, mod2.portfolios.pandas)
    assert_frame_equal(res1.params, res2.params)
    assert mod1.formula == FORMULA
    assert mod2.formula is None

    mod1 = non_traded_model.from_formula(FORMULA_FACTORS, data.joined,
                                         portfolios=data.portfolios, risk_free=True)
    mod2 = non_traded_model(data.portfolios, data.factors, risk_free=True)
    res1 = mod1.fit()
    res2 = mod2.fit()
    assert_frame_equal(mod1.factors.pandas, mod2.factors.pandas)
    assert_frame_equal(mod1.portfolios.pandas, mod2.portfolios.pandas)
    assert_frame_equal(res1.params, res2.params)
    assert mod1.formula == FORMULA_FACTORS
    assert mod2.formula is None
