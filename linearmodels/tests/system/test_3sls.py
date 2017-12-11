from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.compat.pandas import assert_frame_equal, assert_series_equal
from linearmodels.system.model import IV3SLS
from linearmodels.tests.system._utility import generate_3sls_data, simple_3sls, \
    generate_3sls_data_v2

nexog = [3, [1, 2, 3, 4, 5]]
nendog = [2, [1, 2, 1, 2, 1]]
ninstr = [3, 2, [2, 3, 2, 3, 2]]
const = [True, False]
rho = [0.8, 0.0]
common_exog = [True, False]
included_weights = [True, False]
output_dict = [True, False]
params = list(product(nexog, nendog, ninstr, const, rho, common_exog,
                      included_weights, output_dict))


def gen_id(param):
    idstr = 'homo' if isinstance(param[0], list) else 'hetero'
    idstr += '-homo_endog' if isinstance(param[1], list) else '-hetero_endog'
    idstr += '-homo_instr' if isinstance(param[2], list) else '-hetero_instr'
    idstr += '-const' if param[3] else ''
    idstr += '-correl' if param[4] != 0 else ''
    idstr += '-common' if param[5] else ''
    idstr += '-weights' if param[6] else ''
    idstr += '-dict' if param[7] else '-tuple'
    return idstr


ids = list(map(gen_id, params))


@pytest.fixture(params=params, ids=ids)
def data(request):
    p, en, instr, const, rho, common_exog, included_weights, output_dict = request.param
    list_like = isinstance(p, list) or isinstance(en, list) or isinstance(instr, list)
    k = 4
    if common_exog and list_like:
        p = 3
        en = 2
        instr = 3
    elif list_like:
        def safe_len(a):
            a = np.array(a)
            if a.ndim == 0:
                return 0
            return len(a)

        k = max(map(safe_len, [p, en, instr]))

    return generate_3sls_data(n=250, k=k, p=p, en=en, instr=instr, const=const, rho=rho,
                              common_exog=common_exog, included_weights=included_weights,
                              output_dict=output_dict)


def test_direct_simple(data):
    mod = IV3SLS(data)
    res = mod.fit(cov_type='unadjusted')

    y = []
    x = []
    z = []
    for key in data:
        val = data[key]
        if isinstance(val, tuple):
            y.append(val[0])
            x.append(np.concatenate([val[1], val[2]], 1))
            z.append(np.concatenate([val[1], val[3]], 1))
            if len(val) == 5:
                return  # weighted
        else:
            y.append(val['dependent'])
            x.append(np.concatenate([val['exog'], val['endog']], 1))
            z.append(np.concatenate([val['exog'], val['instruments']], 1))
            if 'weights' in val:
                return  # weighted
    out = simple_3sls(y, x, z)
    assert_allclose(res.params.values, out.beta1.squeeze())
    assert_allclose(res.sigma, out.sigma)
    assert_allclose(res.resids.values, out.eps, atol=1e-4)
    assert_allclose(np.diag(res.cov), np.diag(out.cov))


def test_single_equation(data):
    key = list(data.keys())[0]
    data = {key: data[key]}

    mod = IV3SLS(data)
    res = mod.fit(cov_type='unadjusted')

    y = []
    x = []
    z = []
    for key in data:
        val = data[key]
        if isinstance(val, tuple):
            y.append(val[0])
            x.append(np.concatenate([val[1], val[2]], 1))
            z.append(np.concatenate([val[1], val[3]], 1))
            if len(val) == 5:
                return  # weighted
        else:
            y.append(val['dependent'])
            x.append(np.concatenate([val['exog'], val['endog']], 1))
            z.append(np.concatenate([val['exog'], val['instruments']], 1))
            if 'weights' in val:
                return  # weighted
    out = simple_3sls(y, x, z)
    assert_allclose(res.params.values, out.beta1.squeeze())
    assert_allclose(res.sigma, out.sigma)
    assert_allclose(res.resids.values, out.eps)
    assert_allclose(np.diag(res.cov), np.diag(out.cov))


def test_too_few_instruments():
    n = 200
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    endog = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, 1))
    eqns = {}
    for i in range(2):
        eqns['eqn.{0}'.format(i)] = (dep[:, i], exog, endog, instr)
    with pytest.raises(ValueError):
        IV3SLS(eqns)


def test_redundant_instruments():
    n = 200
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    endog = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, 1))
    instr = np.concatenate([exog, instr], 1)
    eqns = {}
    for i in range(2):
        eqns['eqn.{0}'.format(i)] = (dep[:, i], exog, endog, instr)
    with pytest.raises(ValueError):
        IV3SLS(eqns)


def test_too_many_instruments():
    n = 50
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    endog = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, n + 1))
    eqns = {}
    for i in range(2):
        eqns['eqn.{0}'.format(i)] = (dep[:, i], exog, endog, instr)
    with pytest.raises(ValueError):
        IV3SLS(eqns)


def test_wrong_input_type():
    n = 200
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    endog = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, 1))
    instr = np.concatenate([exog, instr], 1)
    eqns = []
    for i in range(2):
        eqns.append((dep[:, i], exog, endog, instr))
    with pytest.raises(TypeError):
        IV3SLS(eqns)

    eqns = {}
    for i in range(2):
        eqns[i] = (dep[:, i], exog, endog, instr)
    with pytest.raises(ValueError):
        IV3SLS(eqns)


def test_multivariate_iv():
    n = 250
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    exog = pd.DataFrame(exog, columns=['exog.{0}'.format(i) for i in range(3)])
    endog = np.random.standard_normal((n, 2))
    endog = pd.DataFrame(endog, columns=['endog.{0}'.format(i) for i in range(2)])
    instr = np.random.standard_normal((n, 3))
    instr = pd.DataFrame(instr, columns=['instr.{0}'.format(i) for i in range(3)])
    eqns = {}
    for i in range(2):
        eqns['dependent.{0}'.format(i)] = (dep[:, i], exog, endog, instr)
    mod = IV3SLS(eqns)
    res = mod.fit()

    common_mod = IV3SLS.multivariate_ls(dep, exog, endog, instr)
    common_res = common_mod.fit()

    assert_series_equal(res.params, common_res.params)


def test_multivariate_iv_bad_data():
    n = 250
    dep = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, 3))
    instr = pd.DataFrame(instr, columns=['instr.{0}'.format(i) for i in range(3)])

    with pytest.raises(ValueError):
        IV3SLS.multivariate_ls(dep, None, None, instr)


def test_fitted(data):
    mod = IV3SLS(data)
    res = mod.fit()
    expected = []
    for i, key in enumerate(res.equations):
        eq = res.equations[key]
        fv = res.fitted_values[key].copy()
        fv.name = 'fitted_values'
        assert_series_equal(eq.fitted_values, fv)
        b = eq.params.values
        direct = mod._x[i] @ b
        expected.append(direct[:, None])
        assert_allclose(eq.fitted_values, direct, atol=1e-8)
    expected = np.concatenate(expected, 1)
    expected = pd.DataFrame(expected, index=mod._dependent[i].pandas.index,
                            columns=[key for key in res.equations])
    assert_frame_equal(expected, res.fitted_values)


def test_no_exog():
    data = generate_3sls_data_v2(nexog=0, const=False)
    mod = IV3SLS(data)
    res = mod.fit()

    data = generate_3sls_data_v2(nexog=0, const=False, omitted='drop')
    mod = IV3SLS(data)
    res2 = mod.fit()

    data = generate_3sls_data_v2(nexog=0, const=False, omitted='empty')
    mod = IV3SLS(data)
    res3 = mod.fit()

    data = generate_3sls_data_v2(nexog=0, const=False, output_dict=False)
    mod = IV3SLS(data)
    res4 = mod.fit()

    data = generate_3sls_data_v2(nexog=0, const=False, output_dict=False, omitted='empty')
    mod = IV3SLS(data)
    res5 = mod.fit()
    assert_series_equal(res.params, res2.params)
    assert_series_equal(res.params, res3.params)
    assert_series_equal(res.params, res4.params)
    assert_series_equal(res.params, res5.params)


def test_no_endog():
    data = generate_3sls_data_v2(nendog=0, ninstr=0)
    mod = IV3SLS(data)
    res = mod.fit()

    data = generate_3sls_data_v2(nendog=0, ninstr=0, omitted='drop')
    mod = IV3SLS(data)
    res2 = mod.fit()

    data = generate_3sls_data_v2(nendog=0, ninstr=0, omitted='empty')
    mod = IV3SLS(data)
    res3 = mod.fit()

    data = generate_3sls_data_v2(nendog=0, ninstr=0, output_dict=False)
    mod = IV3SLS(data)
    res4 = mod.fit()

    data = generate_3sls_data_v2(nendog=0, ninstr=0, output_dict=False, omitted='empty')
    mod = IV3SLS(data)
    res5 = mod.fit()
    assert_series_equal(res.params, res2.params)
    assert_series_equal(res.params, res3.params)
    assert_series_equal(res.params, res4.params)
    assert_series_equal(res.params, res5.params)


def test_uneven_shapes():
    data = generate_3sls_data_v2()
    eq = data[list(data.keys())[0]]
    eq['weights'] = np.ones(eq.dependent.shape[0] // 2)
    with pytest.raises(ValueError):
        IV3SLS(data)
