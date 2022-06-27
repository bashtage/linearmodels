from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from pandas.testing import assert_series_equal
import pytest
from statsmodels.tools.tools import add_constant

from linearmodels.datasets import wage_panel
from linearmodels.iv.model import IV2SLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PanelOLS, PooledOLS, RandomEffects
from linearmodels.panel.results import compare
from linearmodels.tests.panel._utility import datatypes, generate_data


@pytest.fixture(params=[wage_panel.load()])
def data(request):
    return request.param


perc_missing = [0.0, 0.02, 0.20]
has_const = [True, False]
perms = list(product(perc_missing, datatypes, has_const))
ids = ["-".join(str(param) for param in perms) for perm in perms]


@pytest.fixture(params=perms, ids=ids)
def generated_data(request):
    missing, datatype, const = request.param
    return generate_data(
        missing, datatype, const=const, ntk=(91, 7, 5), other_effects=2
    )


@pytest.mark.parametrize("precision", ("tstats", "std_errors", "pvalues"))
def test_single(data, precision):
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    res = PanelOLS(dependent, exog, entity_effects=True).fit()
    comp = compare([res])
    assert len(comp.rsquared) == 1
    d = dir(comp)
    for value in d:
        if value.startswith("_"):
            continue
        getattr(comp, value)


@pytest.mark.parametrize("stars", [False, True])
@pytest.mark.parametrize("precision", ("tstats", "std_errors", "pvalues"))
def test_multiple(data, precision, stars):
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    res = PanelOLS(dependent, exog, entity_effects=True, time_effects=True).fit()
    res2 = PanelOLS(dependent, exog, entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    exog = add_constant(data.set_index(["nr", "year"])[["married", "union"]])
    res3 = PooledOLS(dependent, exog).fit()
    exog = data.set_index(["nr", "year"])[["exper"]]
    res4 = RandomEffects(dependent, exog).fit()
    comp = compare([res, res2, res3, res4], precision=precision, stars=stars)
    assert len(comp.rsquared) == 4
    if stars:
        assert "***" in str(comp)
    d = dir(comp)
    for value in d:
        if value.startswith("_"):
            continue
        getattr(comp, value)
    with pytest.raises(ValueError):
        compare([res, res2, res3, res4], precision="unknown")


def test_multiple_no_effects(data):
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    res = PanelOLS(dependent, exog).fit()
    exog = add_constant(data.set_index(["nr", "year"])[["married", "union"]])
    res3 = PooledOLS(dependent, exog).fit()
    exog = data.set_index(["nr", "year"])[["exper"]]
    res4 = RandomEffects(dependent, exog).fit()
    comp = compare(dict(a=res, model2=res3, model3=res4))
    assert len(comp.rsquared) == 3
    d = dir(comp)
    for value in d:
        if value.startswith("_"):
            continue
        getattr(comp, value)
    compare(dict(a=res, model2=res3, model3=res4))


def test_incorrect_type(data):
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    mod = PanelOLS(dependent, exog)
    res = mod.fit()
    mod2 = IV2SLS(mod.dependent.dataframe, mod.exog.dataframe, None, None)
    res2 = mod2.fit()
    with pytest.raises(TypeError):
        compare(dict(model1=res, model2=res2))


@pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)
def test_predict(generated_data):
    mod = PanelOLS(generated_data.y, generated_data.x, entity_effects=True)
    res = mod.fit()
    pred = res.predict()
    nobs = mod.dependent.dataframe.shape[0]
    assert list(pred.columns) == ["fitted_values"]
    assert pred.shape == (nobs, 1)
    pred = res.predict(effects=True, idiosyncratic=True)
    assert list(pred.columns) == ["fitted_values", "estimated_effects", "idiosyncratic"]
    assert pred.shape == (nobs, 3)
    assert_series_equal(pred.fitted_values, res.fitted_values.iloc[:, 0])
    assert_series_equal(pred.estimated_effects, res.estimated_effects.iloc[:, 0])
    assert_series_equal(pred.idiosyncratic, res.idiosyncratic.iloc[:, 0])
    pred = res.predict(effects=True, idiosyncratic=True, missing=True)
    assert list(pred.columns) == ["fitted_values", "estimated_effects", "idiosyncratic"]
    assert pred.shape == (PanelData(generated_data.y).dataframe.shape[0], 3)

    mod = PanelOLS(generated_data.y, generated_data.x)
    res = mod.fit()
    pred = res.predict()
    assert list(pred.columns) == ["fitted_values"]
    assert pred.shape == (nobs, 1)
    pred = res.predict(effects=True, idiosyncratic=True)
    assert list(pred.columns) == ["fitted_values", "estimated_effects", "idiosyncratic"]
    assert pred.shape == (nobs, 3)
    assert_series_equal(pred.fitted_values, res.fitted_values.iloc[:, 0])
    assert_series_equal(pred.estimated_effects, res.estimated_effects.iloc[:, 0])
    assert_series_equal(pred.idiosyncratic, res.idiosyncratic.iloc[:, 0])
    pred = res.predict(effects=True, idiosyncratic=True, missing=True)
    assert list(pred.columns) == ["fitted_values", "estimated_effects", "idiosyncratic"]
    assert pred.shape == (PanelData(generated_data.y).dataframe.shape[0], 3)
    pred = res.predict(missing=True)
    assert pred.shape[0] <= np.prod(generated_data.y.shape)


def test_predict_exception(generated_data):
    if np.any(np.isnan(generated_data.x)):
        pytest.skip("Cannot test with missing values")
    mod = PanelOLS(generated_data.y, generated_data.x, entity_effects=True)
    res = mod.fit()
    pred = res.predict()
    pred2 = res.predict(generated_data.x)
    assert_allclose(pred, pred2, atol=1e-3)

    panel_data = PanelData(generated_data.x, copy=True)
    x = panel_data.dataframe
    x.index = np.arange(x.shape[0])
    with pytest.raises(ValueError, match="exog does not have the correct number"):
        res.predict(x)


@pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)
def test_predict_no_selection(generated_data):
    mod = PanelOLS(generated_data.y, generated_data.x, entity_effects=True)
    res = mod.fit()
    with pytest.raises(ValueError):
        res.predict(fitted=False)
    with pytest.raises(ValueError):
        res.predict(fitted=False, effects=False, idiosyncratic=False, missing=True)


@pytest.mark.parametrize(
    "constraint_formula",
    [
        "married = 0",
        {"married": 0},
        ["married = 0"],
    ],
)
def test_wald_single(data, constraint_formula):
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    res = PanelOLS(dependent, exog, entity_effects=True, time_effects=True).fit()
    restriction = np.zeros((1, 4))
    restriction[0, 2] = 1
    t1 = res.wald_test(restriction)
    t2 = res.wald_test(restriction, np.zeros(1))
    t3 = res.wald_test(formula=constraint_formula)
    assert_allclose(t1.stat, t2.stat)
    assert_allclose(t1.stat, t3.stat)


@pytest.mark.parametrize(
    "constraint_formula",
    [
        "married = 0, union = 0",
        "married = union = 0",
        {"married": 0, "union": 0},
        ["married = 0", "union = 0"],
    ],
)
def test_wald_test(data, constraint_formula):
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    res = PanelOLS(dependent, exog, entity_effects=True, time_effects=True).fit()

    restriction = np.zeros((2, 4))
    restriction[0, 2] = 1
    restriction[1, 3] = 1
    t1 = res.wald_test(restriction)
    t2 = res.wald_test(restriction, np.zeros(2))
    t3 = res.wald_test(formula=constraint_formula)
    p = res.params.values[:, None]
    c = np.asarray(res.cov)
    c = c[-2:, -2:]
    p = p[-2:]
    direct = p.T @ np.linalg.inv(c) @ p
    assert_allclose(direct, t1.stat)
    assert_allclose(direct, t2.stat)
    assert_allclose(direct, t3.stat)

    with pytest.raises(ValueError):
        res.wald_test(restriction, np.zeros(2), formula=constraint_formula)
