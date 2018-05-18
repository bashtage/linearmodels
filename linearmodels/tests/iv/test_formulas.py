import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal

from linearmodels.compat.pandas import assert_frame_equal
from linearmodels.formula import iv_2sls, iv_gmm, iv_gmm_cue, iv_liml
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVLIML


@pytest.fixture(scope='module',
                params=list(zip([IV2SLS, IVLIML, IVGMMCUE, IVGMM],
                                [iv_2sls, iv_liml, iv_gmm_cue, iv_gmm])))
def model_and_func(request):
    return request.param


def sigmoid(v):
    return np.exp(v) / (1 + np.exp(v))


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
    data['weights'] = np.random.chisquare(10, size=data.shape[0]) / 10
    return data


def test_formula(data, model_and_func, formula):
    model, func = model_and_func
    mod = model.from_formula(formula, data)
    res = mod.fit()
    exog = data[['Intercept', 'x3', 'x4', 'x5']]
    endog = data[['x1', 'x2']]
    instr = data[['z1', 'z2', 'z3']]
    res2 = model(data.y, exog, endog, instr).fit()
    assert res.rsquared == res2.rsquared
    assert mod.formula == formula

    mod = func(formula, data)
    res = mod.fit()
    assert res.rsquared == res2.rsquared
    assert mod.formula == formula


def test_formula_weights(data, model_and_func, formula):
    model, func = model_and_func
    mod = model.from_formula(formula, data, weights=data.weights)
    res = mod.fit()
    exog = data[['Intercept', 'x3', 'x4', 'x5']]
    endog = data[['x1', 'x2']]
    instr = data[['z1', 'z2', 'z3']]
    res2 = model(data.y, exog, endog, instr, weights=data.weights).fit()
    assert res.rsquared == res2.rsquared
    assert mod.formula == formula

    mod = func(formula, data, weights=data.weights)
    res = mod.fit()
    assert res.rsquared == res2.rsquared
    assert mod.formula == formula


def test_formula_kernel(data, model_and_func, formula):
    model, func = model_and_func
    mod = model.from_formula(formula, data)
    mod.fit(cov_type='kernel')
    func(formula, data).fit(cov_type='kernel')


def test_formula_ols(data, model_and_func):
    model, func = model_and_func
    formula = 'y ~ 1 + x1 + x2 + x3 + x4 + x5'
    exog = data[['Intercept', 'x1', 'x2', 'x3', 'x4', 'x5']]
    res2 = model(data.y, exog, None, None)
    res2 = res2.fit()
    res = model.from_formula(formula, data).fit()
    res3 = func(formula, data).fit()

    assert res.rsquared == res2.rsquared
    assert res.rsquared == res3.rsquared


def test_formula_ols_weights(data, model_and_func):
    model, func = model_and_func
    formula = 'y ~ 1 + x1 + x2 + x3 + x4 + x5'
    exog = data[['Intercept', 'x1', 'x2', 'x3', 'x4', 'x5']]
    res2 = model(data.y, exog, None, None, weights=data.weights)
    res2 = res2.fit()
    res = model.from_formula(formula, data, weights=data.weights).fit()
    res3 = func(formula, data, weights=data.weights).fit()

    assert res.rsquared == res2.rsquared
    assert res.rsquared == res3.rsquared


def test_no_exog(data, model_and_func):
    model, func = model_and_func
    formula = 'y ~ [x1 + x2 ~ z1 + z2 + z3]'
    mod = model.from_formula(formula, data)
    res = mod.fit()
    res2 = func(formula, data).fit()

    assert res.rsquared == res2.rsquared
    assert mod.formula == formula

    mod2 = model(data.y, None, data[['x1', 'x2']], data[['z1', 'z2', 'z3']])
    res3 = mod2.fit()

    assert_allclose(res.rsquared, res3.rsquared)


def test_invalid_formula(data, model_and_func):
    model, func = model_and_func
    formula = 'y ~ 1 + x1 + x2 ~ x3 + [x4  x5 ~ z1 z2]'
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    with pytest.raises(ValueError):
        func(formula, data).fit()
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


def test_categorical(model_and_func):
    formula = 'y ~ 1 + d + x1'
    y = np.random.randn(1000)
    x1 = np.random.randn(1000)
    d = np.random.randint(0, 4, 1000)
    d = pd.Categorical(d)
    data = pd.DataFrame({'y': y, 'x1': x1, 'd': d})
    data['Intercept'] = 1.0
    model, func = model_and_func
    mod = model.from_formula(formula, data)
    res3 = mod.fit()
    res2 = func(formula, data).fit()
    res = model(data.y, data[['Intercept', 'x1', 'd']], None, None).fit()

    assert_allclose(res.rsquared, res2.rsquared)
    assert_allclose(res2.rsquared, res3.rsquared)
    assert mod.formula == formula


def test_predict_formula(data, model_and_func, formula):
    model, func = model_and_func
    mod = model.from_formula(formula, data)
    res = mod.fit()
    exog = data[['Intercept', 'x3', 'x4', 'x5']]
    endog = data[['x1', 'x2']]
    pred = res.predict(exog, endog)
    pred2 = res.predict(data=data)
    assert_frame_equal(pred, pred2)
    assert_allclose(res.fitted_values, pred)


def test_formula_function(data, model_and_func):
    model, func = model_and_func
    fmla = 'y ~ 1 + sigmoid(x3) + x4 + [x1 + x2 ~ z1 + z2 + z3] + np.exp(x5)'
    mod = model.from_formula(fmla, data)
    res = mod.fit()

    dep = data.y
    exog = [data[['Intercept']], sigmoid(data[['x3']]), data[['x4']],
            np.exp(data[['x5']])]
    exog = pd.concat(exog, 1)
    endog = data[['x1', 'x2']]
    instr = data[['z1', 'z2', 'z3']]
    mod = model(dep, exog, endog, instr)
    res2 = mod.fit()
    assert_equal(res.params.values, res2.params.values)
    res3 = func(fmla, data).fit()
    assert_equal(res.params.values, res3.params.values)

    with pytest.raises(ValueError):
        res2.predict(data=data)


def test_predict_formula_function(data, model_and_func):
    model, func = model_and_func
    fmla = 'y ~ 1 + sigmoid(x3) + x4 + [x1 + x2 ~ z1 + z2 + z3] + np.exp(x5)'
    mod = model.from_formula(fmla, data)
    res = mod.fit()

    exog = [data[['Intercept']], sigmoid(data[['x3']]), data[['x4']],
            np.exp(data[['x5']])]
    exog = pd.concat(exog, 1)
    endog = data[['x1', 'x2']]
    pred = res.predict(exog, endog)
    pred2 = res.predict(data=data)
    assert_frame_equal(pred, pred2)
    assert_allclose(res.fitted_values, pred)

    res2 = func(fmla, data).fit()
    pred3 = res2.predict(exog, endog)
    pred4 = res2.predict(data=data)
    assert_frame_equal(pred, pred3)
    assert_frame_equal(pred, pred4)


def test_predict_formula_error(data, model_and_func, formula):
    model, func = model_and_func
    mod = model.from_formula(formula, data)
    res = mod.fit()
    exog = data[['Intercept', 'x3', 'x4', 'x5']]
    endog = data[['x1', 'x2']]
    with pytest.raises(ValueError):
        res.predict(exog, endog, data=data)
    with pytest.raises(ValueError):
        mod.predict(res.params, exog=exog, endog=endog, data=data)


def test_single_character_names(data, model_and_func):
    # GH 149
    data = data.copy()
    data['x'] = data['x1']
    data['v'] = data['x2']
    data['z'] = data['z1']
    data['a'] = data['z2']
    fmla = 'y ~ 1 + [x ~ z]'
    model, func = model_and_func
    mod = model.from_formula(fmla, data)
    mod.fit()

    fmla = 'y ~ 1 + [x ~ z + a]'
    model, func = model_and_func
    mod = model.from_formula(fmla, data)
    mod.fit()

    fmla = 'y ~ 1 + [x + v ~ z + a]'
    model, func = model_and_func
    mod = model.from_formula(fmla, data)
    mod.fit()
