import os

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.panel.model import BetweenOLS, PanelOLS, PooledOLS
from linearmodels.tests.panel.results import parse_stata_results
from linearmodels.utility import AttrDict

STATA_RESULTS = parse_stata_results.data()
MODELS = {'between': BetweenOLS, 'fixed_effect': PanelOLS, 'pooled': PooledOLS}
cwd = os.path.split(os.path.abspath(__file__))[0]
sim_data = pd.read_stata(os.path.join(cwd, 'results', 'simulated-panel.dta'))
sim_data = sim_data.set_index(['firm', 'time'])


@pytest.fixture(params=list(STATA_RESULTS.keys()),
                scope='module')
def data(request):
    model, vcv, missing = request.param.split('-')
    y_vars = ['y']
    x_vars = ['x1', 'x2', 'x3', 'x4', 'x5']
    vars = y_vars + x_vars
    if missing:
        for i, v in enumerate(vars):
            vars[i] = v + missing
        y_vars = vars[:1]
        x_vars = vars[1:]
    y = sim_data[y_vars]
    x = sim_data[['intercept'] + x_vars]
    mod = MODELS[model]
    mod_options = {}
    if model == 'fixed_effect':
        mod_options = {'entity_effect': True}
    fit = mod(y, x, **mod_options).fit()
    return AttrDict(fit=fit, model=mod, model_options=mod_options, y=y, x=x,
                    stata=STATA_RESULTS[request.param])


def test_params(data):
    model_params = data.fit
    stata_params = (data.stata.params.param)
    assert_allclose(stata_params.values, model_params.params.squeeze())
