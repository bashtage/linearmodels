import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import BetweenOLS, PanelOLS, PooledOLS
from linearmodels.tests.panel.results import parse_stata_results
from linearmodels.utility import AttrDict

STATA_RESULTS = parse_stata_results.data()
MODELS = {'between': BetweenOLS, 'fixed_effect': PanelOLS, 'pooled': PooledOLS}
cwd = os.path.split(os.path.abspath(__file__))[0]
sim_data = pd.read_stata(os.path.join(cwd, 'results', 'simulated-panel.dta'))
sim_data = sim_data.set_index(['firm', 'time'])

valid = sorted(list(filter(lambda x: True, list(STATA_RESULTS.keys()))))


@pytest.fixture(params=valid, scope='module')
def data(request):
    model, vcv, weights, missing = request.param.split('-')
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
    if weights == 'weighted':
        mod_options.update({'weights': sim_data['w']})
    fit_options = {'debiased': True}
    if weights == 'wls':
        fit_options.update({'reweight': True})
    if vcv == 'unadjusted':
        fit_options.update({'cov_type': 'unadjusted'})
    elif vcv == 'robust' and model != 'fixed_effect':
        fit_options.update({'cov_type': 'robust'})
    elif vcv in ('cluster', 'robust'):
        y_data = PanelData(y)
        eid = y_data.entity_ids
        entities = pd.DataFrame(eid, index=y_data.index, columns=['firm_ids'])
        fit_options.update({'cov_type': 'clustered',
                            'clusters': entities})
    
    if vcv == 'cluster':
        fit_options.update({'group_debias': True})
    spec_mod = mod(y, x, **mod_options)
    fit = spec_mod.fit(**fit_options)
    return AttrDict(fit=fit, model=spec_mod, model_options=mod_options, y=y, x=x,
                    stata=STATA_RESULTS[request.param], fit_options=fit_options,
                    model_name=model, vcv=vcv, weights=weights, missing=missing)


def test_params(data):
    model_params = data.fit
    stata_params = (data.stata.params.param)
    assert_allclose(stata_params.values, model_params.params.squeeze())


@pytest.mark.skip(reason='Not compatible yet')
def test_rsquared_between(data):
    r2_between = data.fit.rsquared_between
    if np.isnan(data.stata.r2_b):
        return
    assert_allclose(r2_between, data.stata.r2_b, rtol=1e-3)


@pytest.mark.skip(reason='Not compatible yet')
def test_rsquared_within(data):
    r2_within = data.fit.rsquared_within
    if np.isnan(data.stata.r2_w):
        return
    assert_allclose(r2_within, data.stata.r2_w, rtol=1e-3)


def test_cov(data):
    if data.model_name == 'fixed_effect' and data.vcv in ('cluster', 'robust'):
        pytest.xfail(reason='Stata does not adjust for # effects, and '
                            'so LSDV and FE disagree')
    fit = data.fit
    stata = data.stata
    repl = []
    for c in stata.variance.columns:
        if c == '_cons':
            repl.append('intercept')
        else:
            repl.append(c)
    var = stata.variance.copy()
    var.columns = repl
    var.index = repl
    var = var[fit.cov.columns].reindex(fit.cov.index)
    assert_allclose(fit.cov.values, var.values, rtol=1e-4)


# TODO: pvals, r2o, r2

def test_f_pooled(data):
    f_pool = data.fit.f_pooled.stat
    stata_f_pool = data.stata.F_f
    if np.isnan(f_pool) or np.isnan(stata_f_pool):
        pytest.skip('Reulst not available for testing')
    assert_allclose(f_pool, stata_f_pool)


def test_f_stat(data):
    if data.model_name == 'fixed_effect' and data.vcv in ('cluster', 'robust'):
        pytest.xfail(reason='Stata does not adjust for # effects, and '
                            'so LSDV and FE disagree')
    
    if data.vcv == 'conventional':
        f_stat = data.fit.f_statistic.stat
    else:
        f_stat = data.fit.f_statistic_robust.stat
    
    stata_f_stat = data.stata.F
    if np.isnan(f_stat) or np.isnan(stata_f_stat):
        pytest.skip('Reulst not available for testing')
    print(f_stat, data.fit.f_statistic_robust.stat, stata_f_stat)


def test_t_stat(data):
    if data.model_name == 'fixed_effect' and data.vcv in ('cluster', 'robust'):
        pytest.xfail(reason='Stata does not adjust for # effects, and '
                            'so LSDV and FE disagree')
    stata_t = data.stata.params.tstat
    repl = []
    for c in stata_t.index:
        if c == '_cons':
            repl.append('intercept')
        else:
            repl.append(c)
    stata_t.index = repl
    t = data.fit.tstats
    stata_t = stata_t.reindex(t.index)
    assert_allclose(t, stata_t)
