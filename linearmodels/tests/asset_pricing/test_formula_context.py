"""
Tests that the formulas of the asset pricing models use the variables and functions
of the code that calls from_formula, as the formulas of the system models do
"""

from formulaic.errors import FactorEvaluationError
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.asset_pricing.model import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.system import SUR

MODELS = [TradedFactorModel, LinearFactorModel, LinearFactorModelGMM]
PORTFOLIOS = "port_01 + port_02 + port_03 + port_04"


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(2)
    factors = 0.05 * rng.standard_normal((120, 2)) + 0.01
    portfolios = factors @ rng.uniform(0.5, 1.5, (2, 4))
    portfolios += 0.03 * rng.standard_normal((120, 4))
    frame = pd.DataFrame(portfolios, columns=PORTFOLIOS.split(" + "))
    frame["factor_01"] = factors[:, 0]
    frame["factor_02"] = factors[:, 1]
    return frame


def double(x):
    return 2 * x


def check(mod, data, first_factor, name):
    """The model has the transformed first factor and the other, as the arrays"""
    expected = np.c_[first_factor, data["factor_02"]]
    assert list(mod.factors.pandas.columns) == [name, "factor_02"]
    assert_allclose(mod.factors.ndarray, expected)
    assert_allclose(mod.portfolios.ndarray, data[PORTFOLIOS.split(" + ")])
    assert mod.formula is not None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("portfolios_in_formula", [True, False])
def test_function_of_the_module(model, data, portfolios_in_formula):
    factors = "double(factor_01) + factor_02"
    if portfolios_in_formula:
        mod = model.from_formula(f"{PORTFOLIOS} ~ {factors}", data)
    else:
        portfolios = data[PORTFOLIOS.split(" + ")]
        mod = model.from_formula(factors, data, portfolios=portfolios)
    check(mod, data, 2 * data["factor_01"], "double(factor_01)")


@pytest.mark.parametrize("model", MODELS)
def test_local_variable_and_function(model, data):
    scale_factor = 3.0

    def triple(x):
        return scale_factor * x

    formula = f"{PORTFOLIOS} ~ I(factor_01 * scale_factor) + factor_02"
    mod = model.from_formula(formula, data)
    check(mod, data, 3 * data["factor_01"], "I(factor_01 * scale_factor)")

    formula = f"{PORTFOLIOS} ~ triple(factor_01) + factor_02"
    mod = model.from_formula(formula, data)
    check(mod, data, 3 * data["factor_01"], "triple(factor_01)")


@pytest.mark.parametrize("model", MODELS)
def test_function_in_the_portfolios(model, data):
    formula = "double(port_01) + port_02 + port_03 + port_04 ~ factor_01 + factor_02"
    mod = model.from_formula(formula, data)
    assert mod.portfolios.pandas.columns[0] == "double(port_01)"
    assert_allclose(mod.portfolios.ndarray[:, 0], 2 * data["port_01"])
    assert_allclose(mod.portfolios.ndarray[:, 1:], data[PORTFOLIOS.split(" + ")[1:]])


@pytest.mark.parametrize("model", MODELS)
def test_built_in_transforms_do_not_need_an_import(model, data):
    # np is a transform of formulaic, and center is its own
    formula = f"{PORTFOLIOS} ~ np.exp(factor_01) + center(factor_02)"
    mod = model.from_formula(formula, data)
    assert_allclose(mod.factors.ndarray[:, 0], np.exp(data["factor_01"]))
    assert_allclose(
        mod.factors.ndarray[:, 1], data["factor_02"] - data["factor_02"].mean()
    )


@pytest.mark.parametrize("model", MODELS)
def test_data_before_context(model, data):
    # A column of the data hides a variable with the same name
    factor_01 = 12345.0
    assert factor_01 > 0
    mod = model.from_formula(f"{PORTFOLIOS} ~ factor_01 + factor_02", data)
    assert_allclose(mod.factors.ndarray[:, 0], data["factor_01"])


@pytest.mark.parametrize("model", MODELS)
def test_unknown_names_still_raise(model, data):
    with pytest.raises(FactorEvaluationError, match="not_defined_anywhere"):
        model.from_formula(f"{PORTFOLIOS} ~ not_defined_anywhere(factor_01)", data)


def test_results_use_the_transformed_factor(data):
    multiplier = 0.5
    formula = f"{PORTFOLIOS} ~ I(factor_01 * multiplier) + factor_02"
    for model in (TradedFactorModel, LinearFactorModel):
        res = model.from_formula(formula, data).fit()
        direct = model(
            data[PORTFOLIOS.split(" + ")],
            np.c_[multiplier * data["factor_01"], data["factor_02"]],
        ).fit()
        assert_allclose(res.params, direct.params)
        assert_allclose(res.cov, direct.cov)
        assert_allclose(res.risk_premia, direct.risk_premia)


def test_same_as_the_system_models(data):
    # The system models capture the context of the caller in the same way
    frame = data.rename(columns={"port_01": "y1", "port_02": "y2"})
    res = SUR.from_formula(
        {"a": "y1 ~ 1 + double(factor_01)", "b": "y2 ~ 1 + factor_02"}, frame
    ).fit()
    assert res.params.shape == (4,)
    mod = TradedFactorModel.from_formula(
        "y1 + y2 ~ double(factor_01) + factor_02", frame
    )
    assert mod.factors.shape == (120, 2)
