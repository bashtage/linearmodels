from itertools import product

import numpy as np
import pandas as pd
import pytest

from linearmodels.compat.pandas import assert_frame_equal
from linearmodels.formula import (between_ols, first_difference_ols, panel_ols,
                                  pooled_ols, random_effects, fama_macbeth)
from linearmodels.panel.model import (BetweenOLS, FirstDifferenceOLS, PanelOLS,
                                      PooledOLS, RandomEffects, FamaMacBeth,
                                      PanelFormulaParser)
from linearmodels.tests.panel._utility import generate_data, datatypes

pytestmark = pytest.mark.filterwarnings('ignore::linearmodels.utility.MissingValueWarning')

PERC_MISSING = [0, 0.02, 0.10, 0.33]
TYPES = datatypes


@pytest.fixture(params=list(product(PERC_MISSING, TYPES)),
                ids=list(map(lambda x: str(int(100 * x[0])) + '-' + str(x[1]),
                             product(PERC_MISSING, TYPES))))
def data(request):
    missing, datatype = request.param
    return generate_data(missing, datatype, ntk=(91, 7, 5))


@pytest.fixture(params=['y ~ x1 + x2',
                        'y ~ x0 + x1 + x2 + x3 + x4 '],
                scope='module')
def formula(request):
    return request.param


classes = [PooledOLS, BetweenOLS, FirstDifferenceOLS, RandomEffects, FamaMacBeth]
funcs = [pooled_ols, between_ols, first_difference_ols, random_effects, fama_macbeth]


@pytest.fixture(params=list(zip(classes, funcs)))
def models(request):
    return request.param


@pytest.fixture(params=[True, False])
def effects(request):
    return request.param


def sigmoid(v):
    return np.exp(v) / (1 + np.exp(v))


def test_basic_formulas(data, models, formula):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    model, formula_func = models
    mod = model.from_formula(formula, joined)
    res = mod.fit()
    wmod = model.from_formula(formula, joined, weights=data.w)
    wres = wmod.fit()

    mod2 = formula_func(formula, joined)
    res2 = mod2.fit()
    np.testing.assert_allclose(res.params, res2.params)

    parts = formula.split('~')
    vars = parts[1].replace(' 1 ', ' const ').split('+')
    vars = list(map(lambda s: s.strip(), vars))
    x = data.x
    res2 = model(data.y, x[vars]).fit()
    wres2 = model(data.y, x[vars], weights=data.w).fit()
    np.testing.assert_allclose(res.params, res2.params)
    np.testing.assert_allclose(wres.params, wres2.params)
    assert isinstance(mod, model)
    assert mod.formula == formula

    if model is FirstDifferenceOLS:
        return

    formula = formula.split('~')
    formula[1] = ' 1 + ' + formula[1]
    formula = '~'.join(formula)
    mod = model.from_formula(formula, joined)
    res = mod.fit()

    mod2 = formula_func(formula, joined)
    res2 = mod2.fit()
    np.testing.assert_allclose(res.params, res2.params)

    x['Intercept'] = 1.0
    vars = ['Intercept'] + vars
    mod2 = model(data.y, x[vars])
    res2 = mod2.fit()
    np.testing.assert_allclose(res.params, res2.params)
    assert mod.formula == formula


def test_basic_formulas_math_op(data, models, formula):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    formula = formula.replace('x0', 'np.exp(x0)')
    formula = formula.replace('x1', 'sigmoid(x1)')
    model, formula_func = models
    res = model.from_formula(formula, joined).fit()
    pred = res.predict(data=joined)
    pred = pred.reindex(res.fitted_values.index)
    np.testing.assert_allclose(pred.values, res.fitted_values.values)


def test_panel_ols_formulas_math_op(data):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    formula = 'y ~ x1 + np.exp(x2)'
    mod = PanelOLS.from_formula(formula, joined)
    mod.fit()


def test_panel_ols_formula(data):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    formula = 'y ~ x1 + x2'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula

    formula = 'y ~ x1 + x2 + EntityEffects'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.entity_effects is True
    assert mod.time_effects is False

    formula = 'y ~ x1 + x2 + TimeEffects'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.time_effects is True
    assert mod.entity_effects is False

    formula = 'y ~ x1 + EntityEffects + TimeEffects + x2 '
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.entity_effects is True
    assert mod.time_effects is True
    mod2 = panel_ols(formula, joined)
    res = mod.fit()
    res2 = mod2.fit()
    np.testing.assert_allclose(res.params, res2.params)

    formula = 'y ~ x1 + EntityEffects + FixedEffects + x2 '
    with pytest.raises(ValueError):
        PanelOLS.from_formula(formula, joined)


def test_basic_formulas_predict(data, models, formula):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    model, formula_func = models
    mod = model.from_formula(formula, joined)
    res = mod.fit()
    pred = res.predict(data=joined)

    mod2 = formula_func(formula, joined)
    res2 = mod2.fit()
    pred2 = res2.predict(data=joined)
    np.testing.assert_allclose(pred.values, pred2.values, atol=1e-8)

    parts = formula.split('~')
    vars = parts[1].replace(' 1 ', ' const ').split('+')
    vars = list(map(lambda s: s.strip(), vars))
    x = data.x
    res2 = model(data.y, x[vars]).fit()
    pred3 = res2.predict(x[vars])
    pred4 = res.predict(x[vars])
    np.testing.assert_allclose(pred.values, pred3.values, atol=1e-8)
    np.testing.assert_allclose(pred.values, pred4.values, atol=1e-8)

    if model is FirstDifferenceOLS:
        return

    formula = formula.split('~')
    formula[1] = ' 1 + ' + formula[1]
    formula = '~'.join(formula)
    mod = model.from_formula(formula, joined)
    res = mod.fit()
    pred = res.predict(data=joined)

    x['Intercept'] = 1.0
    vars = ['Intercept'] + vars
    mod2 = model(data.y, x[vars])
    res2 = mod2.fit()
    pred2 = res.predict(x[vars])
    pred3 = res2.predict(x[vars])
    np.testing.assert_allclose(pred, pred2, atol=1e-8)
    np.testing.assert_allclose(pred, pred3, atol=1e-8)


def test_formulas_predict_error(data, models, formula):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    model, formula_func = models
    mod = model.from_formula(formula, joined)
    res = mod.fit()
    with pytest.raises(ValueError):
        res.predict(joined, data=joined)
    with pytest.raises(ValueError):
        mod.predict(params=res.params, exog=joined, data=joined)

    parts = formula.split('~')
    vars = parts[1].replace(' 1 ', ' const ').split('+')
    vars = list(map(lambda s: s.strip(), vars))
    x = data.x
    res = model(data.y, x[vars]).fit()
    with pytest.raises(ValueError):
        res.predict(data=joined)


def test_parser(data, formula, effects):
    if not isinstance(data.y, pd.DataFrame):
        return
    if effects:
        formula += ' + EntityEffects + TimeEffects'
    joined = data.x
    joined['y'] = data.y
    parser = PanelFormulaParser(formula, joined)
    dep, exog = parser.data
    assert_frame_equal(parser.dependent, dep)
    assert_frame_equal(parser.exog, exog)
    parser.eval_env = 3
    assert parser.eval_env == 3
    parser.eval_env = 2
    assert parser.eval_env == 2
    assert parser.entity_effect == ('EntityEffects' in formula)
    assert parser.time_effect == ('TimeEffects' in formula)

    formula += ' + FixedEffects '
    if effects:
        with pytest.raises(ValueError):
            PanelFormulaParser(formula, joined)
    else:
        parser = PanelFormulaParser(formula, joined)
        assert parser.entity_effect
