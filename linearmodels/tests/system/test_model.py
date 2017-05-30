from collections.abc import Mapping
from itertools import product

import numpy as np
import pytest
from pandas import DataFrame

from linearmodels.system.model import SUR
from linearmodels.tests.system._utility import generate_data

p = [3, [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]]
const = [True, False]
rho = [0.8, 0.0]
common_exog = [True, False]
included_weights = [True, False]
output_dict = [True, False]
params = list(product(p, const, rho, common_exog, included_weights, output_dict))


def get_res(res):
    d = filter(lambda s: not s.startswith('_'), dir(res))
    for attr in d:
        value = getattr(res, attr)
        if isinstance(value, Mapping):
            for key in value:
                get_res(value[key])


@pytest.fixture(scope='module', params=params)
def data(request):
    p, const, rho, common_exog, included_weights, output_dict = request.param
    if common_exog and isinstance(p, list):
        p = 3
    return generate_data(p=p, const=const, rho=rho,
                         common_exog=common_exog, included_weights=included_weights,
                         output_dict=output_dict)


params = list(product(const, rho, included_weights))


@pytest.fixture(scope='module', params=params)
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
