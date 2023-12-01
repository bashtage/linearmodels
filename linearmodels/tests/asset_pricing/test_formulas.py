import pickle

import numpy as np
from pandas import concat
from pandas.testing import assert_frame_equal
import pytest

from linearmodels.asset_pricing.model import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.tests.asset_pricing._utility import generate_data

FORMULA_FACTORS = "factor_01 + factor_02 + factor_03"
FORMULA_PORT = (
    "port_01 + port_02 + port_03 + port_04 + port_05 + port_06 + port_07 + "
    "port_08 + port_09 + port_10"
)
FORMULA = " ~ ".join((FORMULA_PORT, FORMULA_FACTORS))


@pytest.fixture(
    scope="module", params=[TradedFactorModel, LinearFactorModel, LinearFactorModelGMM]
)
def model(request):
    return request.param


@pytest.fixture(scope="module", params=[LinearFactorModel, LinearFactorModelGMM])
def non_traded_model(request):
    return request.param


@pytest.fixture(scope="module")
def data():
    premia = np.array([0.1, 0.1, 0.1])
    out = generate_data(nportfolio=10, output="pandas", alpha=True, premia=premia)
    out["joined"] = concat([out.factors, out.portfolios], axis=1, sort=False)
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

    pmod = pickle.loads(pickle.dumps(mod1))
    pres = pickle.loads(pickle.dumps(res1))
    ppres = pmod.fit()
    assert_frame_equal(mod1.factors.pandas, pmod.factors.pandas)
    assert_frame_equal(mod1.portfolios.pandas, pmod.portfolios.pandas)
    assert_frame_equal(res1.params, pres.params)
    assert_frame_equal(res1.params, ppres.params)
    assert mod1.formula == FORMULA
    assert pmod.formula == FORMULA
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

    mod1 = non_traded_model.from_formula(
        FORMULA_FACTORS, data.joined, portfolios=data.portfolios, risk_free=True
    )
    mod2 = non_traded_model(data.portfolios, data.factors, risk_free=True)
    res1 = mod1.fit()
    res2 = mod2.fit()
    assert_frame_equal(mod1.factors.pandas, mod2.factors.pandas)
    assert_frame_equal(mod1.portfolios.pandas, mod2.portfolios.pandas)
    assert_frame_equal(res1.params, res2.params)
    assert mod1.formula == FORMULA_FACTORS
    assert mod2.formula is None


def test_starting_values_options(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(steps=1, disp=0)
    res_full = mod.fit(disp=0)
    mod2 = LinearFactorModelGMM(data.portfolios, data.factors)
    oo = {"method": "L-BFGS-B"}
    sv = np.r_[np.asarray(res.betas).ravel(), np.asarray(res.risk_premia)]
    sv = np.r_[sv, data.factors.mean()]
    res2 = mod2.fit(starting=sv, opt_options=oo, disp=0)
    assert res_full.iterations == res2.iterations
    res3 = mod2.fit(starting=sv[:, None], opt_options=oo, disp=0)
    assert_frame_equal(res2.params, res3.params)

    with pytest.raises(ValueError, match="tarting values"):
        mod2.fit(starting=sv[:-3], opt_options=oo, disp=0)


def test_escaped_formula(data, model):
    def transform(fmla):
        names = [f"`{factor.strip().replace('_', ' ')}`" for factor in fmla.split("+")]
        return " + ".join(names)

    formula_factors = transform(FORMULA_FACTORS)
    formula_port = transform(FORMULA_PORT)
    formula = " ~ ".join((formula_port, formula_factors))
    data_rename = data.joined.copy()
    data_rename.columns = [col.replace("_", " ") for col in data_rename]
    mod = model.from_formula(formula, data_rename)
    res = mod.fit()
    ports = [
        val.strip().replace("`", "").strip() for val in formula.split("~")[0].split("+")
    ]
    factors = [
        val.strip().replace("`", "").strip() for val in formula.split("~")[1].split("+")
    ]
    for port in ports:
        assert port in res.params.index
    for factor in factors:
        assert factor in res.params.columns
