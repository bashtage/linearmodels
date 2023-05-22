import pickle

from formulaic.errors import FormulaSyntaxError
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas import Categorical, DataFrame, concat
from pandas.testing import assert_frame_equal
import pytest

from linearmodels.formula import iv_2sls, iv_gmm, iv_gmm_cue, iv_liml
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from linearmodels.iv._utility import IVFormulaParser
from linearmodels.shared.exceptions import IndexWarning


@pytest.fixture(
    scope="module",
    params=list(
        zip([IV2SLS, IVLIML, IVGMMCUE, IVGMM], [iv_2sls, iv_liml, iv_gmm_cue, iv_gmm])
    ),
)
def model_and_func(request):
    return request.param


def sigmoid(v):
    return np.exp(v) / (1 + np.exp(v))


formulas = [
    "y ~ 1 + x3 + x4 + x5 + [x1 + x2 ~ z1 + z2 + z3]",
    "y ~ 1 + x3 + x4 + [x1 + x2 ~ z1 + z2 + z3] + x5",
]


@pytest.fixture(scope="module", params=formulas)
def formula(request):
    return request.param


@pytest.fixture(scope="module")
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
    z = v[:, k : k + p]
    e = v[:, [-1]]
    params = np.arange(1, k + 1) / k
    params = params[:, None]
    y = x @ params + e
    cols = ["y"] + ["x" + str(i) for i in range(1, 6)]
    cols += ["z" + str(i) for i in range(1, 4)]
    data = DataFrame(np.c_[y, x, z], columns=cols)
    data["Intercept"] = 1.0
    data["weights"] = np.random.chisquare(10, size=data.shape[0]) / 10
    return data


def test_formula(data, model_and_func, formula):
    model, func = model_and_func
    mod = model.from_formula(formula, data)
    res = mod.fit()
    exog = data[["Intercept", "x3", "x4", "x5"]]
    endog = data[["x1", "x2"]]
    instr = data[["z1", "z2", "z3"]]
    res2 = model(data.y, exog, endog, instr).fit()
    assert res.rsquared == res2.rsquared
    assert mod.formula == formula

    mod = func(formula, data)
    res = mod.fit()
    assert res.rsquared == res2.rsquared
    assert mod.formula == formula

    pmod = pickle.loads(pickle.dumps(mod))
    ppres = pmod.fit()
    pres = pickle.loads(pickle.dumps(res))
    assert pres.rsquared == res2.rsquared
    assert ppres.rsquared == res2.rsquared


def test_formula_weights(data, model_and_func, formula):
    model, func = model_and_func
    mod = model.from_formula(formula, data, weights=data.weights)
    res = mod.fit()
    exog = data[["Intercept", "x3", "x4", "x5"]]
    endog = data[["x1", "x2"]]
    instr = data[["z1", "z2", "z3"]]
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
    mod.fit(cov_type="kernel")
    func(formula, data).fit(cov_type="kernel")


def test_formula_ols(data, model_and_func):
    model, func = model_and_func
    formula = "y ~ 1 + x1 + x2 + x3 + x4 + x5"
    exog = data[["Intercept", "x1", "x2", "x3", "x4", "x5"]]
    res2 = model(data.y, exog, None, None)
    res2 = res2.fit()
    res = model.from_formula(formula, data).fit()
    res3 = func(formula, data).fit()

    assert res.rsquared == res2.rsquared
    assert res.rsquared == res3.rsquared


def test_formula_ols_weights(data, model_and_func):
    model, func = model_and_func
    formula = "y ~ 1 + x1 + x2 + x3 + x4 + x5"
    exog = data[["Intercept", "x1", "x2", "x3", "x4", "x5"]]
    res2 = model(data.y, exog, None, None, weights=data.weights)
    res2 = res2.fit()
    res = model.from_formula(formula, data, weights=data.weights).fit()
    res3 = func(formula, data, weights=data.weights).fit()

    assert res.rsquared == res2.rsquared
    assert res.rsquared == res3.rsquared


def test_no_exog(data, model_and_func):
    model, func = model_and_func
    formula = "y ~ [x1 + x2 ~ z1 + z2 + z3]"
    mod = model.from_formula(formula, data)
    res = mod.fit()
    res2 = func(formula, data).fit()

    assert res.rsquared == res2.rsquared
    assert mod.formula == formula

    mod2 = model(data.y, None, data[["x1", "x2"]], data[["z1", "z2", "z3"]])
    res3 = mod2.fit()

    assert_allclose(res.rsquared, res3.rsquared)


def test_invalid_formula(data, model_and_func):
    model, func = model_and_func
    formula = "y ~ 1 + x1 + x2 ~ x3 + [x4  x5 ~ z1 z2]"
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    with pytest.raises(ValueError):
        func(formula, data).fit()
    formula = "y ~ 1 + x1 + x2 + x3 + x4 + x5 ~ z1 z2"
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    formula = "y y2 ~ 1 + x1 + x2 + x3 + [x4 + x5 ~ + z1 + z2]"
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    formula = "y y2 ~ 1 + x1 + x2 + x3 [ + x4 + x5 ~ z1 + z2]"
    with pytest.raises(ValueError):
        model.from_formula(formula, data).fit()
    formula = "y y2 ~ 1 + x1 + x2 + x3 + [x4 + x5 ~ z1 + z2]"
    with pytest.raises(FormulaSyntaxError):
        model.from_formula(formula, data).fit()


def test_categorical(model_and_func):
    formula = "y ~ 1 + d + x1"
    y = np.random.randn(1000)
    x1 = np.random.randn(1000)
    d = np.random.randint(0, 4, 1000)
    d = Categorical(d)
    data = DataFrame({"y": y, "x1": x1, "d": d})
    data["Intercept"] = 1.0
    model, func = model_and_func
    mod = model.from_formula(formula, data)
    res3 = mod.fit()
    res2 = func(formula, data).fit()
    res = model(data.y, data[["Intercept", "x1", "d"]], None, None).fit()

    assert_allclose(res.rsquared, res2.rsquared)
    assert_allclose(res2.rsquared, res3.rsquared)
    assert mod.formula == formula


def test_predict_formula(data, model_and_func, formula):
    model, _ = model_and_func
    mod = model.from_formula(formula, data)
    res = mod.fit()
    exog = data[["Intercept", "x3", "x4", "x5"]]
    endog = data[["x1", "x2"]]
    pred = res.predict(exog, endog)
    pred2 = res.predict(data=data)
    assert_frame_equal(pred, pred2)
    assert_allclose(res.fitted_values, pred)

    with pytest.raises(ValueError, match="exog and endog or data must be provided"):
        mod.predict(res.params)


def test_formula_function(data, model_and_func):
    model, func = model_and_func
    fmla = "y ~ 1 + sigmoid(x3) + x4 + [x1 + x2 ~ z1 + z2 + z3] + np.exp(x5)"
    fmla_mod = model.from_formula(fmla, data)
    fmla_res = fmla_mod.fit()

    dep = data.y
    exog = [
        data[["Intercept"]],
        sigmoid(data[["x3"]]),
        data[["x4"]],
        np.exp(data[["x5"]]),
    ]
    exog = concat(exog, axis=1, sort=False)
    endog = data[["x1", "x2"]]
    instr = data[["z1", "z2", "z3"]]
    mod = model(dep, exog, endog, instr)
    array_res = mod.fit()
    func_res = func(fmla, data).fit()

    assert_allclose(fmla_res.params.values, array_res.params.values, rtol=1e-5)
    assert_allclose(fmla_res.params.values, func_res.params.values, rtol=1e-5)

    with pytest.raises(ValueError):
        array_res.predict(data=data)


def test_predict_formula_function(data, model_and_func):
    model, func = model_and_func
    fmla = "y ~ 1 + sigmoid(x3) + x4 + [x1 + x2 ~ z1 + z2 + z3] + np.exp(x5)"
    mod = model.from_formula(fmla, data)
    res = mod.fit()

    exog = [
        data[["Intercept"]],
        sigmoid(data[["x3"]]),
        data[["x4"]],
        np.exp(data[["x5"]]),
    ]
    exog = concat(exog, axis=1, sort=False)
    endog = data[["x1", "x2"]]
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
    model, _ = model_and_func
    mod = model.from_formula(formula, data)
    res = mod.fit()
    exog = data[["Intercept", "x3", "x4", "x5"]]
    endog = data[["x1", "x2"]]
    with pytest.raises(ValueError):
        res.predict(exog, endog, data=data)
    with pytest.raises(ValueError):
        mod.predict(res.params, exog=exog, endog=endog, data=data)


def test_single_character_names(data, model_and_func):
    # GH 149
    data = data.copy()
    data["x"] = data["x1"]
    data["v"] = data["x2"]
    data["z"] = data["z1"]
    data["a"] = data["z2"]
    fmla = "y ~ 1 + [x ~ z]"
    model, func = model_and_func
    mod = model.from_formula(fmla, data)
    mod.fit()

    fmla = "y ~ 1 + [x ~ z + a]"
    model, func = model_and_func
    mod = model.from_formula(fmla, data)
    mod.fit()

    fmla = "y ~ 1 + [x + v ~ z + a]"
    model, func = model_and_func
    mod = model.from_formula(fmla, data)
    mod.fit()


def test_ols_formula(data):
    # GH 185
    data = data.copy()
    fmla = "y ~ 1 + x1"
    mod = IV2SLS.from_formula(fmla, data)
    res = mod.fit()
    assert "OLS Estimation Summary" in str(res)


def test_iv_formula_parser(data, model_and_func, formula):
    parser = IVFormulaParser(formula, data)
    assert parser.eval_env == 2
    parser.eval_env = 3
    assert parser.eval_env == 3
    assert isinstance(parser.exog, DataFrame)


def test_predict_exception(data, model_and_func, formula):
    model, _ = model_and_func
    data = data.copy()
    data.index = np.arange(100000, 100000 + data.shape[0])
    mod = model.from_formula(formula, data)
    res = mod.fit()
    exog = data[["Intercept", "x3", "x4", "x5"]]
    endog = data[["x1", "x2"]]
    pred = res.predict(exog, endog)
    pred2 = res.predict(data=data)
    pred3 = res.predict(np.asarray(exog), np.asarray(endog))
    with pytest.warns(IndexWarning):
        res.predict(exog, np.asarray(endog))
    with pytest.warns(IndexWarning):
        res.predict(np.asarray(exog), endog)
    assert_frame_equal(pred, pred2)
    assert_allclose(pred, pred3)


def test_predict_no_formula_predict_data(data, model_and_func, formula):
    model, _ = model_and_func
    mod_fmla: IV2SLS = model.from_formula(formula, data)
    mod = model(mod_fmla.dependent, mod_fmla.exog, mod_fmla.endog, mod_fmla.instruments)
    res = mod.fit()
    with pytest.raises(ValueError, match="exog and endog must have"):
        x = np.asarray(mod_fmla.exog.pandas)
        w = np.asarray(mod_fmla.endog.pandas)
        w = w[: mod_fmla.endog.shape[0] // 2]
        res.predict(exog=x, endog=w)
    with pytest.raises(ValueError, match="Unable to"):
        res.predict(data=data)


def test_formula_escaped_simple():
    rs = np.random.RandomState(1232)
    data = pd.DataFrame({"y": np.arange(100), "var x": rs.standard_normal(100)})
    IV2SLS.from_formula("y ~ 1 + `var x`", data)


def test_formula_escape():
    data = DataFrame(
        np.random.standard_normal((250, 3)), columns=["y space", "x 1", "z 0"]
    )
    data.loc[:, "x 1"] *= data.loc[:, "z 0"]
    formula = "`y space` ~ 1 + [`x 1` ~ `z 0`]"
    mod = IV2SLS.from_formula(formula, data=data)
    res = mod.fit()
    summ = res.summary
    assert len(res.params) == 2
    assert "x 1" in res.params.index
    assert "y space" in str(summ)
    assert "Instruments: z 0" in str(summ)
