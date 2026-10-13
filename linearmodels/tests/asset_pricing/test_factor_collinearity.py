"""
Tests of the messages for factors that are collinear or contain a constant

has_constant is also True for a matrix that does not have full column rank, so that
collinear factors were reported as containing a constant and the message for factors
without full column rank could not be reached.
"""

import numpy as np
import pandas as pd
import pytest

from linearmodels.asset_pricing.model import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)

MODELS = [TradedFactorModel, LinearFactorModel, LinearFactorModelGMM]
RANK = r"Model cannot be estimated\. factors do not have full column rank\."
CONSTANT = r"factors must not contain a constant or equivalent\."


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    factors = rng.standard_normal((100, 3))
    portfolios = factors @ rng.standard_normal((3, 6)) + rng.standard_normal((100, 6))
    return portfolios, factors


def collinear_factors(kind, factors):
    x1, x2, x3 = factors.T
    ones = np.ones_like(x1)
    cases = {
        "duplicate": np.c_[factors, x1],
        "combination": np.c_[factors, x1 - 2 * x2],
        "combination first": np.c_[x1 + x2, factors],
        # A constant column, or a constant in the span, makes the factors contain one
        "duplicate constant": np.c_[factors, ones, ones],
        "constant in span": np.c_[x1, x2, 1 - x1 - x2],
        "constant in span, collinear": np.c_[x1, x2, 1 - x1 - x2, x1 + x2],
        "constant and duplicate": np.c_[factors, x1, ones],
        "constant not one": np.c_[factors, 3 * ones],
        "constant not one and duplicate": np.c_[factors, 3 * ones, x1],
        "scaled constant in span": np.c_[x1, x2, 5 - 2 * x1 - 2 * x2],
        "x3 only": x3[:, None],
    }
    return cases[kind]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("kind", ["duplicate", "combination", "combination first"])
@pytest.mark.parametrize("pandas", [False, True])
def test_collinear_factors(data, model, kind, pandas):
    portfolios, factors = data
    collinear = collinear_factors(kind, factors)
    if pandas:
        collinear = pd.DataFrame(collinear)
    with pytest.raises(ValueError, match=RANK):
        model(portfolios, collinear)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "kind",
    [
        "duplicate constant",
        "constant in span",
        "constant in span, collinear",
        "constant and duplicate",
        "constant not one",
        "constant not one and duplicate",
        "scaled constant in span",
    ],
)
def test_constant_is_reported_as_a_constant(data, model, kind):
    portfolios, factors = data
    with pytest.raises(ValueError, match=CONSTANT):
        model(portfolios, collinear_factors(kind, factors))


@pytest.mark.parametrize("model", MODELS)
def test_valid_factors(data, model):
    portfolios, factors = data
    mod = model(portfolios, factors)
    assert mod.factors.shape == (100, 3)
    mod = model(portfolios, collinear_factors("x3 only", factors))
    assert mod.factors.shape == (100, 1)


def test_fewer_observations_than_factors(data):
    # A wide matrix does not have full column rank, whatever it contains
    portfolios, factors = data
    with pytest.raises(ValueError, match=RANK):
        TradedFactorModel(portfolios[:2], factors[:2])
