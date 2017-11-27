from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.system import IVSystemGMM
from linearmodels.tests.system._utility import generate_3sls_data_v2, simple_gmm
from linearmodels.utility import AttrDict

params = list(product([1, 2], [True, False]))


def gen_id(r):
    id = 'steps:{0}'.format(r[0])
    if r[1]:
        id += ',robust'
    else:
        id += ',unadjusted'
    return id


ids = list(map(gen_id, params))


@pytest.fixture(scope='module', params=params, ids=ids)
def data(request):
    steps, robust = request.param
    weight_type = 'robust' if robust else 'unadjusted'
    eqns = generate_3sls_data_v2(k=3)
    y = [eqns[key].dependent for key in eqns]
    x = [np.concatenate([eqns[key].exog, eqns[key].endog], 1) for key in eqns]
    z = [np.concatenate([eqns[key].exog, eqns[key].instruments], 1) for key in eqns]

    return AttrDict(eqns=eqns, x=x, y=y, z=z, steps=steps,
                    robust=robust, weight_type=weight_type)


def test_params(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust)
    if data.steps == 1:
        beta = simple.beta0
    else:
        beta = simple.beta1
    assert_allclose(res.params.values, beta, rtol=1e-4)


def test_weights(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust)
    w = simple.w0 if data.steps == 1 else simple.w1
    assert_allclose(res.w, w, rtol=1e-4)


def test_cov(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust, data.steps)
    assert_allclose(res.cov.values, simple.cov)


def test_formula_equivalence(data):
    mod = IVSystemGMM(data.eqns, weight_type='unadjusted')
    formula = []
    df = []
    for i, key in enumerate(data.eqns):
        eqn = data.eqns[key]
        dep = eqn.dependent
        ex = eqn.exog
        en = eqn.endog
        instr = eqn.instruments
        dep = pd.DataFrame(dep, columns=['dep_{0}'.format(i)])
        has_const = False
        if np.any(np.all(ex == 1, 0)):
            ex = ex[:, 1:]
            has_const = True
        ex = pd.DataFrame(ex, columns=['ex_{0}_{1}'.format(i, j) for j in range(ex.shape[1])])
        en = pd.DataFrame(en, columns=['en_{0}_{1}'.format(i, j) for j in range(en.shape[1])])
        instr = pd.DataFrame(instr, columns=['instr_{0}_{1}'.format(i, j)
                                             for j in range(ex.shape[1])])
        fmla = ''.join(dep.columns) + ' ~  '
        if has_const:
            fmla += ' 1 + '
        fmla += ' + '.join(ex.columns) + ' + ['
        fmla += ' + '.join(en.columns) + ' ~ '
        fmla += ' + '.join(instr.columns) + ' ] '
        formula.append(fmla)
        df.extend([dep, ex, en, instr])
    from collections import OrderedDict
    formulas = OrderedDict()
    for i, f in enumerate(formula):
        formulas['eq{0}'.format(i)] = f
    df = pd.concat(df, 1)
    formula_mod = IVSystemGMM.from_formula(formulas, df, weight_type='unadjusted')
    res = mod.fit(cov_type='unadjusted')
    formula_res = formula_mod.fit(cov_type='unadjusted')
    assert_allclose(res.params, formula_res.params)


def test_formula_equivalence_weights(data):
    weights = AttrDict()
    eqn_copy = AttrDict()
    for key in data.eqns:
        eqn = {k: v for k, v in data.eqns[key].items()}
        nobs = eqn['dependent'].shape[0]
        w = np.random.chisquare(2, (nobs, 1)) / 2
        weights[key] = w
        eqn['weights'] = w
        eqn_copy[key] = eqn

    mod = IVSystemGMM(eqn_copy, weight_type='unadjusted')
    df = []
    formulas = OrderedDict()
    for i, key in enumerate(data.eqns):
        eqn = data.eqns[key]
        dep = eqn.dependent
        ex = eqn.exog
        en = eqn.endog
        instr = eqn.instruments
        dep = pd.DataFrame(dep, columns=['dep_{0}'.format(i)])
        has_const = False
        if np.any(np.all(ex == 1, 0)):
            ex = ex[:, 1:]
            has_const = True
        ex = pd.DataFrame(ex, columns=['ex_{0}_{1}'.format(i, j) for j in range(ex.shape[1])])
        en = pd.DataFrame(en, columns=['en_{0}_{1}'.format(i, j) for j in range(en.shape[1])])
        instr = pd.DataFrame(instr, columns=['instr_{0}_{1}'.format(i, j)
                                             for j in range(ex.shape[1])])
        fmla = ''.join(dep.columns) + ' ~  '
        if has_const:
            fmla += ' 1 + '
        fmla += ' + '.join(ex.columns) + ' + ['
        fmla += ' + '.join(en.columns) + ' ~ '
        fmla += ' + '.join(instr.columns) + ' ] '
        formulas[key] = fmla
        df.extend([dep, ex, en, instr])
    df = pd.concat(df, 1)
    formula_mod = IVSystemGMM.from_formula(formulas, df, weights=weights, weight_type='unadjusted')
    res = mod.fit(cov_type='unadjusted')
    formula_res = formula_mod.fit(cov_type='unadjusted')
    assert_allclose(res.params, formula_res.params)


def test_weight_options(data):
    mod = IVSystemGMM(data.eqns, weight_type='unadjusted', debiased=True)
    res = mod.fit(cov_type='unadjusted')
    assert res.weight_config == {'debiased': True}
    assert res.weight_type == 'unadjusted'
    base_res = IVSystemGMM(data.eqns, weight_type='unadjusted').fit(cov_type='unadjusted')
    assert np.all(np.diag(res.w) >= np.diag(base_res.w))


def test_no_constant_smoke():
    eqns = generate_3sls_data_v2(k=3, const=False)
    mod = IVSystemGMM(eqns)
    mod.fit()


def test_unknown_weight_type(data):
    with pytest.raises(ValueError):
        IVSystemGMM(data.eqns, weight_type='unknown')


def test_unknown_cov_type(data):
    mod = IVSystemGMM(data.eqns)
    with pytest.raises(ValueError):
        mod.fit(cov_type='unknown')
    with pytest.raises(ValueError):
        mod.fit(cov_type=3)


def test_initial_weight_matrix(data):
    mod = IVSystemGMM(data.eqns)
    z = [np.concatenate([data.eqns[key].exog, data.eqns[key].instruments], 1)
         for key in data.eqns]
    z = np.concatenate(z, 1)
    ze = z + np.random.standard_normal(size=z.shape)
    w0 = ze.T @ ze / ze.shape[0]
    res0 = mod.fit(initial_weight=w0, iter_limit=1)
    res = mod.fit(iter_limit=1)
    assert np.any(res0.params != res.params)
