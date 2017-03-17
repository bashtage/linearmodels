import numpy as np
import pandas as pd
import pytest

# from patsy import SyntaxError
from linearmodels.iv import IV2SLS, IVGMMCUE, IVGMM, IVLIML


@pytest.fixture(scope='module', params=[IV2SLS, IVLIML, IVGMMCUE, IVGMM])
def model(request):
    return request.param


formulas = ['y ~ 1 + x3 + x4 + x5 + [x1 + x2 ~ z1 + z2 + z3]',
            'y ~ 1 + x3 + x4 + [x1 + x2 ~ z1 + z2 + z3] + x5']


@pytest.fixture(scope='module', params=formulas)
def formula(request):
    return request.param


@pytest.fixture(scope='module')
def data():
    n, k, p = 1000, 5, 3
    np.random.seed(12345)
    rho = 0.5
    r = np.zeros((k + p + 1, k + p + 1))
    r.fill(rho)
    r[-1, 2:] = 0
    r[2:, -1] = 0
    r[-1, -1] = 0.5
    r += np.eye(9) * 0.5
    v = np.random.multivariate_normal(np.zeros(r.shape[0]), r, n)

    x = v[:, :k]
    z = v[:, k:k + p]
    e = v[:, [-1]]
    params = np.arange(1, k + 1) / k
    params = params[:, None]
    y = x @ params + e
    cols = ['y'] + ['x' + str(i) for i in range(1, 6)]
    cols += ['z' + str(i) for i in range(1, 4)]
    data = pd.DataFrame(np.c_[y, x, z], columns=cols)
    data['Intercept'] = 1.0
    return data


def test_formula(data, model, formula):
    mod = model.from_formula(formula, data)
    res = mod.fit()
    exog = data[['Intercept', 'x3', 'x4', 'x5']]
    endog = data[['x1', 'x2']]
    instr = data[['z1', 'z2', 'z3']]
    res2 = model(data.y, exog, endog, instr).fit()
    assert res.rsquared == res2.rsquared
    assert mod.formula == formula


def test_formula_kernel(data, model, formula):
    mod = model.from_formula(formula, data)
    mod.fit(cov_type='kernel')


def test_formula_ols(data, model):
    formula = 'y ~ 1 + x1 + x2 + x3 + x4 + x5'
    exog = data[['Intercept', 'x1', 'x2', 'x3', 'x4', 'x5']]
    res2 = model(data.y, exog, None, None)
    res2 = res2.fit()
    res = model.from_formula(formula, data).fit()

    assert res.rsquared == res2.rsquared


def test_invalid_formula(data, model):
    formula = 'y ~ 1 + x1 + x2 ~ x3 + [x4  x5 ~ z1 z2]'
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    formula = 'y ~ 1 + x1 + x2 + x3 + x4 + x5 ~ z1 z2'
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    formula = 'y y2 ~ 1 + x1 + x2 + x3 + [x4 + x5 ~ + z1 + z2]'
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    formula = 'y y2 ~ 1 + x1 + x2 + x3 [ + x4 + x5 ~ z1 + z2]'
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    formula = 'y y2 ~ 1 + x1 + x2 + x3 + [x4 + x5 ~ z1 + z2]'
    with pytest.raises(SyntaxError):
        model.from_formula(formula, data).fit()
