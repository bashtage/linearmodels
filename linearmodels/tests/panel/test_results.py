from collections import OrderedDict

import pytest
import statsmodels.api as sm

from linearmodels.datasets import wage_panel
from linearmodels.iv.model import IV2SLS
from linearmodels.panel.model import PanelOLS, PooledOLS, RandomEffects
from linearmodels.panel.results import compare


@pytest.fixture(params=[wage_panel.load()])
def data(request):
    return request.param


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
