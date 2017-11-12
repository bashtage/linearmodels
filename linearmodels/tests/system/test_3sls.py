from itertools import product

import numpy as np
import pandas as pd
import pytest

from linearmodels.compat.pandas import assert_series_equal
from linearmodels.system.model import IV3SLS
from linearmodels.tests.system._utility import generate_3sls_data, simple_3sls

p = [3, [1, 2, 3, 4, 5]]
en = [2, [1, 2, 1, 2, 1]]
instr = [3, 2, [2, 3, 2, 3, 2]]
const = [True, False]
rho = [0.8, 0.0]
common_exog = [True, False]
included_weights = [True, False]
output_dict = [True, False]
params = list(product(p, en, instr, const, rho, common_exog, included_weights, output_dict))


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
    
    return generate_3sls_data(n=500, k=k, p=p, en=en, instr=instr, const=const, rho=rho,
                              common_exog=common_exog, included_weights=included_weights,
                              output_dict=output_dict)


def test_smoke():
    eqn = {}
    n = 500
    for i in range(3):
        key = 'eqn.{0}'.format(i)
        y = np.random.randn(n, 1)
        ex = np.random.randn(n, 3)
        en = np.random.randn(n, 2)
        instr = np.random.randn(n, 3)
        eqn[key] = (y, ex, en, instr)
    mod = IV3SLS(eqn)
    mod.fit()
    
    eqn = {}
    for i in range(3):
        key = 'eqn.{0}'.format(i)
        eqn[key] = dict(dependent=np.random.randn(n, 1), exog=np.random.randn(n, 3),
                        endog=np.random.randn(n, 2), instruments=np.random.randn(n, 3))
    IV3SLS(eqn)
    for i in range(3):
        eqn[key]['weights'] = np.random.chisquare(3, n) / 3
    
    eqn = {}
    for i in range(3):
        key = 'eqn.{0}'.format(i)
        eqn[key] = dict(dependent=np.random.randn(n, 1), exog=np.random.randn(n, 3))
    mod = IV3SLS(eqn)
    mod.fit()
    for i in range(3):
        eqn[key]['weights'] = np.random.chisquare(3, n) / 3
    mod = IV3SLS(eqn)
    
    ex = np.random.randn(n, 3)
    en = np.random.randn(n, 2)
    instr = np.random.randn(n, 3)
    for i in range(3):
        y = np.random.randn(n, 1)
        key = 'eqn.{0}'.format(i)
        eqn[key] = (y, ex, en, instr)
    mod.fit()


def test_nothing(data):
    res = IV3SLS(data).fit()
    
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
