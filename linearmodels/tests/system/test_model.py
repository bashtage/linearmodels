from collections.abc import Mapping
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pandas import DataFrame

from linearmodels.iv.model import _OLS as OLS
from linearmodels.system._utility import blocked_column_product, blocked_diag_product, \
    inv_matrix_sqrt
from linearmodels.system.model import SUR
from linearmodels.tests.system._utility import generate_data
from linearmodels.utility import AttrDict

p = [3, [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]]
const = [True, False]
rho = [0.8, 0.0]
common_exog = [True, False]
included_weights = [True, False]
output_dict = [True, False]
params = list(product(p, const, rho, common_exog, included_weights, output_dict))


def gen_id(param):
    idstr = 'homo' if isinstance(param[0], list) else 'hetero'
    idstr += '-const' if param[1] else ''
    idstr += '-correl' if param[2] != 0 else ''
    idstr += '-common' if param[3] else ''
    idstr += '-weights' if param[4] else ''
    idstr += '-dist' if param[4] else '-tuple'
    return idstr


ids = list(map(gen_id, params))


def check_results(res1, res2):
    assert_allclose(res1.params, res2.params)
    assert_allclose(res1.rsquared, res2.rsquared)
    assert_allclose(res1.cov, res2.cov)
    assert_allclose(res1.pvalues, res2.pvalues, atol=1e-6)
    assert_allclose(res1.resids, res2.resids)
    assert_allclose(res1.wresids, res2.wresids)
    assert_allclose(res1.tstats, res2.tstats)
    assert_allclose(res1.std_errors, res2.std_errors)
    if hasattr(res1, 'rsquared_adj'):
        assert_allclose(res1.rsquared_adj, res2.rsquared_adj)
    if hasattr(res1, 'f_statistic'):
        assert_allclose(res1.f_statistic.stat, res2.f_statistic.stat)
        assert_allclose(res1.f_statistic.pval, res2.f_statistic.pval)


def get_res(res):
    d = filter(lambda s: not s.startswith('_'), dir(res))
    for attr in d:
        value = getattr(res, attr)
        if isinstance(value, Mapping):
            for key in value:
                get_res(value[key])


@pytest.fixture(scope='module', params=params, ids=ids)
def data(request):
    p, const, rho, common_exog, included_weights, output_dict = request.param
    if common_exog and isinstance(p, list):
        p = 3
    return generate_data(p=p, const=const, rho=rho,
                         common_exog=common_exog, included_weights=included_weights,
                         output_dict=output_dict)


params = list(product(const, rho, included_weights))


def gen_id(param):
    idstr = 'const' if param[0] else ''
    idstr += '-correl' if param[1] != 0 else ''
    idstr += '-weights' if param[2] else ''
    return idstr


ids = list(map(gen_id, params))


@pytest.fixture(scope='module', params=params, ids=ids)
def mvreg_data(request):
    const, rho, included_weights = request.param
    values = generate_data(const=const, rho=rho,
                           common_exog=True, included_weights=included_weights)
    dep = []
    for key in values:
        exog = values[key]['exog']
        dep.append(values[key]['dependent'])
    return np.hstack(dep), exog


def test_smoke(data):
    mod = SUR(data)
    res = mod.fit()
    res = mod.fit(cov_type='unadjusted')
    res = mod.fit(cov_type='unadjusted', use_gls=False)
    res = mod.fit(full_cov=False)
    
    get_res(res)


def test_errors():
    with pytest.raises(TypeError):
        SUR([])
    with pytest.raises(TypeError):
        SUR({'a': 'absde', 'b': 12345})
    
    moddata = {'a': {'dependent': np.random.standard_normal((100, 1)),
                     'exog': np.random.standard_normal((100, 5))}}
    with pytest.raises(ValueError):
        mod = SUR(moddata)
        mod.fit(cov_type='unknown')
    
    moddata = {'a': {'dependent': np.random.standard_normal((100, 1)),
                     'exog': np.random.standard_normal((101, 5))}}
    with pytest.raises(ValueError):
        SUR(moddata)
    
    moddata = {'a': {'dependent': np.random.standard_normal((10, 1)),
                     'exog': np.random.standard_normal((10, 20))}}
    with pytest.raises(ValueError):
        SUR(moddata)
    
    x = np.random.standard_normal((100, 2))
    x = np.c_[x, x]
    moddata = {'a': {'dependent': np.random.standard_normal((100, 1)),
                     'exog': x}}
    with pytest.raises(ValueError):
        SUR(moddata)


def test_mv_reg_smoke(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit()
    res = mod.fit(cov_type='unadjusted')
    res = mod.fit(cov_type='unadjusted', use_gls=False)
    assert res.method == 'OLS'
    res = mod.fit(full_cov=False)
    
    get_res(res)


def test_formula_smoke():
    data = DataFrame(np.random.standard_normal((500, 4)),
                     columns=['y1', 'y2', 'x1', 'x2'])
    formula = {'eq1': 'y1 ~ 1 + x1', 'eq2': 'y2 ~ 1 + x2'}
    mod = SUR.from_formula(formula, data)
    res = mod.fit()
    
    formula = '{y1 ~ 1 + x1} {y2 ~ 1 + x2}'
    mod = SUR.from_formula(formula, data)
    res = mod.fit(cov_type='heteroskedastic')
    
    formula = '''
    {y1 ~ 1 + x1}
    {y2 ~ 1 + x2}
    '''
    mod = SUR.from_formula(formula, data)
    res = mod.fit(cov_type='heteroskedastic')
    
    formula = '''
    {eq.a:y1 ~ 1 + x1}
    {second: y2 ~ 1 + x2}
    '''
    mod = SUR.from_formula(formula, data)
    res = mod.fit(cov_type='heteroskedastic')
    assert 'eq.a' in res.equation_labels
    assert 'second' in res.equation_labels


# TODO: Implement weights
# TODO: 1. MV OLS and OLS (weighted) homo and hetero
# TODO: Implement observation dropping and check

def test_mv_ols_equivalence(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit(cov_type='unadjusted')
    keys = res.equation_labels
    assert res.method == 'OLS'
    
    for i in range(dependent.shape[1]):
        ols_mod = OLS(dependent[:, i], exog)
        ols_res = ols_mod.fit(cov_type='unadjusted', debiased=False)
        mv_res = res.equations[keys[i]]
        assert mv_res.method == 'OLS'
        check_results(mv_res, ols_res)


def test_mv_ols_equivalence_robust(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit(cov_type='robust')
    keys = res.equation_labels
    
    for i in range(dependent.shape[1]):
        ols_mod = OLS(dependent[:, i], exog)
        ols_res = ols_mod.fit(cov_type='robust', debiased=False)
        mv_res = res.equations[keys[i]]
        check_results(mv_res, ols_res)


def test_mv_ols_equivalence_debiased(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit(cov_type='unadjusted', debiased=True)
    keys = res.equation_labels
    
    for i in range(dependent.shape[1]):
        ols_mod = OLS(dependent[:, i], exog)
        ols_res = ols_mod.fit(cov_type='unadjusted', debiased=True)
        mv_res = res.equations[keys[i]]
        check_results(mv_res, ols_res)


def test_mv_ols_equivalence_hetero_debiased(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit(cov_type='robust', debiased=True)
    keys = res.equation_labels
    
    for i in range(dependent.shape[1]):
        ols_mod = OLS(dependent[:, i], exog)
        ols_res = ols_mod.fit(cov_type='robust', debiased=True)
        mv_res = res.equations[keys[i]]
        check_results(mv_res, ols_res)


def test_gls_eye_mv_ols_equiv(mvreg_data):
    dependent, exog = mvreg_data
    mv_mod = SUR.multivariate_ls(dependent, exog)
    mv_res = mv_mod.fit()
    keys = mv_res.equation_labels
    
    ad = AttrDict()
    for i in range(dependent.shape[1]):
        key = 'dependent.{0}'.format(i)
        df = DataFrame(dependent[:, [i]], columns=[key])
        ad[key] = {'dependent': df,
                   'exog': exog.copy()}
    gls_mod = SUR(ad, sigma=np.eye(len(ad)))
    gls_res = gls_mod.fit(use_gls=True)
    check_results(mv_res, gls_res)
    
    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)
    
    mv_res = mv_mod.fit(cov_type='robust')
    gls_res = gls_mod.fit(cov_type='robust', use_gls=True)
    check_results(mv_res, gls_res)
    
    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)
    
    mv_res = mv_mod.fit(cov_type='robust', debiased=True)
    gls_res = gls_mod.fit(cov_type='robust', use_gls=True, debiased=True)
    check_results(mv_res, gls_res)
    
    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)


def test_gls_without_mv_ols_equiv(mvreg_data):
    dependent, exog = mvreg_data
    mv_mod = SUR.multivariate_ls(dependent, exog)
    mv_res = mv_mod.fit()
    keys = mv_res.equation_labels
    
    ad = AttrDict()
    for i in range(dependent.shape[1]):
        key = 'dependent.{0}'.format(i)
        df = DataFrame(dependent[:, [i]], columns=[key])
        ad[key] = {'dependent': df,
                   'exog': exog.copy()}
    gls_mod = SUR(ad)
    gls_res = gls_mod.fit(use_gls=False)
    check_results(mv_res, gls_res)
    
    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)
    
    mv_res = mv_mod.fit(cov_type='robust')
    gls_res = gls_mod.fit(cov_type='robust', use_gls=False)
    check_results(mv_res, gls_res)
    
    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)
    
    mv_res = mv_mod.fit(cov_type='robust', debiased=True)
    gls_res = gls_mod.fit(cov_type='robust', use_gls=False, debiased=True)
    check_results(mv_res, gls_res)
    
    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)


def test_ols_against_gls(data):
    mod = SUR(data)
    res = mod.fit()
    sigma = res.sigma
    sigma_m12 = inv_matrix_sqrt(sigma)
    key = list(data.keys())[0]
    if isinstance(data[key], Mapping):
        y = [data[key]['dependent'] for key in data]
        x = [data[key]['exog'] for key in data]
    else:
        y = [data[key][0] for key in data]
        x = [data[key][1] for key in data]
    
    y = blocked_column_product(y, sigma_m12)
    x = blocked_diag_product(x, sigma_m12)
    ols_res = OLS(y, x).fit(debiased=False)
    assert_allclose(res.params, ols_res.params)
    if not any(mod.has_constant) and res.method == 'GLS':
        # R2 only same w/o constant
        # Only GLS is implemented when common regressor used!
        assert_allclose(res.total_ss, ols_res.total_ss)
        assert_allclose(res.resid_ss, ols_res.resid_ss)
        assert_allclose(res.rsquared, ols_res.rsquared)
