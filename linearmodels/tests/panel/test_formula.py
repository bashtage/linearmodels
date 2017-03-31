from itertools import product

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.panel.model import BetweenOLS, FirstDifferenceOLS, PanelOLS, PooledOLS
from linearmodels.tests.panel._utility import generate_data

PERC_MISSING = [0, 0.02, 0.10, 0.33]
TYPES = ['numpy', 'pandas', 'xarray']


@pytest.fixture(params=list(product(PERC_MISSING, TYPES)),
                ids=list(map(lambda x: str(int(100 * x[0])) + '-' + str(x[1]),
                             product(PERC_MISSING, TYPES))))
def data(request):
    missing, datatype = request.param
    return generate_data(missing, datatype)


@pytest.fixture(params=['y ~ x1 + x2',
                        'y ~ x0 + x1 + x2 + x3 + x4 '],
                scope='module')
def formula(request):
    return request.param


@pytest.fixture(params=[PooledOLS, BetweenOLS, FirstDifferenceOLS])
def model(request):
    return request.param


def test_basic_formulas(data, model, formula):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    mod = model.from_formula(formula, joined)
    res = mod.fit()
    parts = formula.split('~')
    vars = parts[1].replace(' 1 ', ' const ').split('+')
    vars = list(map(lambda s: s.strip(), vars))
    x = data.x
    res2 = model(data.y, x[vars]).fit()
    assert_allclose(res.params, res2.params)
    assert mod.formula == formula

    if model is FirstDifferenceOLS:
        return

    formula = formula.split('~')
    formula[1] = ' 1 + ' + formula[1]
    formula = '~'.join(formula)
    mod = model.from_formula(formula, joined)
    res = mod.fit()
    x['Intercept'] = 1.0
    vars = ['Intercept'] + vars
    mod2 = model(data.y, x[vars])
    res2 = mod2.fit()
    assert_allclose(res.params, res2.params)
    assert mod.formula == formula


def test_panel_ols_formula(data):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    formula = 'y ~ x1 + x2'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula

    formula = 'y ~ x1 + x2 + EntityEffect'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.entity_effect is True
    assert mod.time_effect is False

    formula = 'y ~ x1 + x2 + TimeEffect'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.time_effect is True
    assert mod.entity_effect is False

    formula = 'y ~ x1 + EntityEffect + TimeEffect + x2 '
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.entity_effect is True
    assert mod.time_effect is True

    formula = 'y ~ x1 + EntityEffect + FixedEffect + x2 '
    with pytest.raises(ValueError):
        PanelOLS.from_formula(formula, joined)
