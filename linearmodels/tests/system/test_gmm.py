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
    simple = simple_gmm(data.y, data.x, data.z, data.robust, steps=data.steps)
    if data.steps == 1:
        beta = simple.beta0
    else:
        beta = simple.beta1
    assert_allclose(res.params.values, beta, rtol=1e-4)


def test_weights(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust, steps=data.steps)
    w = simple.w0 if data.steps == 1 else simple.w1
    assert_allclose(res.w, w, rtol=1e-4)


def test_cov(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust, steps=data.steps)
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
    mod = IVSystemGMM(data.eqns, weight_type='unadjusted', debiased=True, center=True)
    res = mod.fit(cov_type='unadjusted')
    assert res.weight_config == {'debiased': True, 'center': True}
    assert res.weight_type == 'unadjusted'
    assert 'Debiased: True' in str(res.summary)
    assert str(hex(id(res._weight_estimtor))) in res._weight_estimtor.__repr__()
    assert res._weight_estimtor.config == {'debiased': True, 'center': True}
    base_res = IVSystemGMM(data.eqns, weight_type='unadjusted').fit(cov_type='unadjusted')
    assert np.all(np.diag(res.w) >= np.diag(base_res.w))

    mod = IVSystemGMM(data.eqns, weight_type='robust', debiased=True)
    res = mod.fit(cov_type='robust')


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


def test_summary(data):
    mod = IVSystemGMM(data.eqns)
    res = mod.fit()
    assert 'Instruments' in res.summary.as_text()
    assert 'Weight Estimator' in res.summary.as_text()
    for eq in res.equations:
        assert 'Weight Estimator' in res.equations[eq].summary.as_text()
        assert 'Instruments' in res.equations[eq].summary.as_text()

    res = mod.fit(iter_limit=10)
    if res.iterations > 2:
        assert 'Iterative System GMM' in res.summary.as_text()


def test_summary_homoskedastic(data):
    mod = IVSystemGMM(data.eqns, weight_type='unadjusted', debiased=True)
    res = mod.fit(cov_type='homoskedastic', debiased=True)
    assert 'Homoskedastic (Unadjusted) Weighting' in res.summary.as_text()


def test_fixed_sigma(data):
    mod = IVSystemGMM(data.eqns, weight_type='unadjusted')
    res = mod.fit(cov_type='unadjusted')
    k = len(data.eqns)
    b = np.random.standard_normal((k, 1))
    sigma = b @ b.T + np.diag(np.ones(k))
    mod_sigma = IVSystemGMM(data.eqns, weight_type='unadjusted', sigma=sigma)
    res_sigma = mod_sigma.fit()
    assert np.any(res.params != res_sigma.params)
    assert np.any(res.sigma != res_sigma.sigma)


def test_incorrect_sigma_shape(data):
    k = len(data.eqns)
    b = np.random.standard_normal((k + 2, 1))
    sigma = b @ b.T + np.diag(np.ones(k + 2))
    with pytest.raises(ValueError):
        IVSystemGMM(data.eqns, weight_type='unadjusted', sigma=sigma)


def test_invalid_sigma_usage(data):
    k = len(data.eqns)
    b = np.random.standard_normal((k, 1))
    sigma = b @ b.T + np.diag(np.ones(k))
    with pytest.warns(UserWarning):
        IVSystemGMM(data.eqns, weight_type='robust', sigma=sigma)


def test_j_statistic_direct(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust, steps=data.steps)
    assert_allclose(res.j_stat.stat, simple.j_stat, rtol=1e-4)


def test_linear_constraint(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    p = mod.param_names
    r = pd.DataFrame(np.zeros((1, len(p))), index=[0], columns=p)
    r.iloc[0, 1::6] = 1
    q = pd.Series([6])
    mod.add_constraints(r, q)

    res = mod.fit()
    assert_allclose(res.params.iloc[1::6].sum(), 6)


def test_kernel_equiv(data):
    mod = IVSystemGMM(data.eqns, weight_type='kernel', bandwidth=0)
    res = mod.fit(cov_type='kernel', debiased=True, bandwidth=0)
    assert 'Kernel (HAC) Weighting' in res.summary.as_text()
    rob_mod = IVSystemGMM(data.eqns, weight_type='robust')
    rob_res = rob_mod.fit(cov_type='robust', debiased=True)
    assert_allclose(res.tstats, rob_res.tstats)


def test_kernel_optimal_bandwidth_smoke(data):
    mod = IVSystemGMM(data.eqns, weight_type='kernel')
    mod.fit(cov_type='kernel', debiased=True)
    mod = IVSystemGMM(data.eqns, weight_type='kernel', optimal_bw=True)
    mod.fit(cov_type='kernel', debiased=True)
