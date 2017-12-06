from collections import OrderedDict
from itertools import product

import pytest
import statsmodels.api as sm

from linearmodels.compat.pandas import assert_series_equal
from linearmodels.datasets import wage_panel
from linearmodels.iv.model import IV2SLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PanelOLS, PooledOLS, RandomEffects
from linearmodels.panel.results import compare
from linearmodels.tests.panel._utility import generate_data, datatypes


@pytest.fixture(params=[wage_panel.load()])
def data(request):
    return request.param


missing = [0.0, 0.02, 0.20]
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def generated_data(request):
    missing, datatype, const = request.param
    return generate_data(missing, datatype, const=const, ntk=(91, 7, 5), other_effects=2)


def test_single(data):
    dependent = data.set_index(['nr', 'year']).lwage
    exog = sm.add_constant(data.set_index(['nr', 'year'])[['expersq', 'married', 'union']])
    res = PanelOLS(dependent, exog, entity_effects=True).fit()
    comp = compare([res])
    assert len(comp.rsquared) == 1
    d = dir(comp)
    for value in d:
        if value.startswith('_'):
            continue
        getattr(comp, value)


def test_multiple(data):
    dependent = data.set_index(['nr', 'year']).lwage
    exog = sm.add_constant(data.set_index(['nr', 'year'])[['expersq', 'married', 'union']])
    res = PanelOLS(dependent, exog, entity_effects=True, time_effects=True).fit()
    res2 = PanelOLS(dependent, exog, entity_effects=True).fit(cov_type='clustered',
                                                              cluster_entity=True)
    exog = sm.add_constant(data.set_index(['nr', 'year'])[['married', 'union']])
    res3 = PooledOLS(dependent, exog).fit()
    exog = data.set_index(['nr', 'year'])[['exper']]
    res4 = RandomEffects(dependent, exog).fit()
    comp = compare([res, res2, res3, res4])
    assert len(comp.rsquared) == 4
    d = dir(comp)
    for value in d:
        if value.startswith('_'):
            continue
        getattr(comp, value)


def test_multiple_no_effects(data):
    dependent = data.set_index(['nr', 'year']).lwage
    exog = sm.add_constant(data.set_index(['nr', 'year'])[['expersq', 'married', 'union']])
    res = PanelOLS(dependent, exog).fit()
    exog = sm.add_constant(data.set_index(['nr', 'year'])[['married', 'union']])
    res3 = PooledOLS(dependent, exog).fit()
    exog = data.set_index(['nr', 'year'])[['exper']]
    res4 = RandomEffects(dependent, exog).fit()
    comp = compare(dict(a=res, model2=res3, model3=res4))
    assert len(comp.rsquared) == 3
    d = dir(comp)
    for value in d:
        if value.startswith('_'):
            continue
        getattr(comp, value)
    compare(OrderedDict(a=res, model2=res3, model3=res4))


def test_incorrect_type(data):
    dependent = data.set_index(['nr', 'year']).lwage
    exog = sm.add_constant(data.set_index(['nr', 'year'])[['expersq', 'married', 'union']])
    mod = PanelOLS(dependent, exog)
    res = mod.fit()
    mod2 = IV2SLS(mod.dependent.dataframe, mod.exog.dataframe, None, None)
    res2 = mod2.fit()
    with pytest.raises(TypeError):
        compare(dict(model1=res, model2=res2))


def test_predict(generated_data):
    mod = PanelOLS(generated_data.y, generated_data.x, entity_effects=True)
    res = mod.fit()
    pred = res.predict()
    nobs = mod.dependent.dataframe.shape[0]
    assert list(pred.columns) == ['fitted_values']
    assert pred.shape == (nobs, 1)
    pred = res.predict(effects=True, idiosyncratic=True)
    assert list(pred.columns) == ['fitted_values', 'estimated_effects', 'idiosyncratic']
    assert pred.shape == (nobs, 3)
    assert_series_equal(pred.fitted_values, res.fitted_values.iloc[:, 0])
    assert_series_equal(pred.estimated_effects, res.estimated_effects.iloc[:, 0])
    assert_series_equal(pred.idiosyncratic, res.idiosyncratic.iloc[:, 0])
    pred = res.predict(effects=True, idiosyncratic=True, missing=True)
    assert list(pred.columns) == ['fitted_values', 'estimated_effects', 'idiosyncratic']
    assert pred.shape == (PanelData(generated_data.y).dataframe.shape[0], 3)

    mod = PanelOLS(generated_data.y, generated_data.x)
    res = mod.fit()
    pred = res.predict()
    assert list(pred.columns) == ['fitted_values']
    assert pred.shape == (nobs, 1)
    pred = res.predict(effects=True, idiosyncratic=True)
    assert list(pred.columns) == ['fitted_values', 'estimated_effects', 'idiosyncratic']
    assert pred.shape == (nobs, 3)
    assert_series_equal(pred.fitted_values, res.fitted_values.iloc[:, 0])
    assert_series_equal(pred.estimated_effects, res.estimated_effects.iloc[:, 0])
    assert_series_equal(pred.idiosyncratic, res.idiosyncratic.iloc[:, 0])
    pred = res.predict(effects=True, idiosyncratic=True, missing=True)
    assert list(pred.columns) == ['fitted_values', 'estimated_effects', 'idiosyncratic']
    assert pred.shape == (PanelData(generated_data.y).dataframe.shape[0], 3)


def test_predict_no_selection(generated_data):
    mod = PanelOLS(generated_data.y, generated_data.x, entity_effects=True)
    res = mod.fit()
    with pytest.raises(ValueError):
        res.predict(fitted=False)
    with pytest.raises(ValueError):
        res.predict(fitted=False, effects=False, idiosyncratic=False, missing=True)
