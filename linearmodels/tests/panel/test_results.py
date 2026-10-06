from itertools import product
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
from statsmodels.tools.tools import add_constant

from linearmodels.datasets import wage_panel
from linearmodels.iv.model import IV2SLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PanelOLS, PooledOLS, RandomEffects
from linearmodels.panel.results import compare
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.tests.panel._utility import datatypes, generate_data


@pytest.fixture(params=[wage_panel.load()])
def data(request):
    return request.param


perc_missing = [0.0, 0.02, 0.20]
has_const = [True, False]
perms = list(product(perc_missing, datatypes, has_const))
ids = ["-".join(str(param) for param in perm) for perm in perms]


@pytest.fixture(params=perms, ids=ids)
def generated_data(request):
    missing, datatype, const = request.param
    return generate_data(
        missing, datatype, const=const, ntk=(91, 7, 5), other_effects=2
    )


@pytest.mark.parametrize("precision", ["tstats", "std_errors", "pvalues"])
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
@pytest.mark.parametrize("precision", ["tstats", "std_errors", "pvalues"])
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
    with pytest.raises(ValueError, match=r"Unknown precision value"):
        compare([res, res2, res3, res4], precision="unknown")


def test_multiple_no_effects(data):
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    res = PanelOLS(dependent, exog).fit()
    exog = add_constant(data.set_index(["nr", "year"])[["married", "union"]])
    res3 = PooledOLS(dependent, exog).fit()
    exog = data.set_index(["nr", "year"])[["exper"]]
    res4 = RandomEffects(dependent, exog).fit()
    comp = compare({"a": res, "model2": res3, "model3": res4})
    assert len(comp.rsquared) == 3
    d = dir(comp)
    for value in d:
        if value.startswith("_"):
            continue
        getattr(comp, value)
    compare({"a": res, "model2": res3, "model3": res4})


def test_incorrect_type(data):
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    mod = PanelOLS(dependent, exog)
    res = mod.fit()
    mod2 = IV2SLS(mod.dependent.dataframe, mod.exog.dataframe, None, None)
    res2 = mod2.fit()
    with pytest.raises(TypeError, match=r"Results from unknown model"):
        compare({"model1": res, "model2": res2})


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
    with pytest.raises(ValueError, match=r"exog does not have the correct number"):
        res.predict(x)


@pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)
def test_predict_no_selection(generated_data):
    mod = PanelOLS(generated_data.y, generated_data.x, entity_effects=True)
    res = mod.fit()
    with pytest.raises(ValueError, match=r"At least one output must be"):
        res.predict(fitted=False)
    with pytest.raises(ValueError, match=r"At least one output must be"):
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

    with pytest.raises(ValueError, match=r"restriction and formula cannot"):
        res.wald_test(restriction, np.zeros(2), formula=constraint_formula)


# Values computed in Stata (hausman fe re, with the sigmamore and sigmaless
# options and with constant, using xtreg, fe and xtreg, re)
STATA_HAUSMAN = {
    "": (7.190854126884934, 0.0274489582259534),
    "include_constant": (7.190854126885962, 0.0660570908629669),
    "sigmaless": (6.953506564342694, 0.0309075965524561),
    "include_constant-sigmaless": (6.953506564340507, 0.0733945334224529),
    "sigmamore": (6.945610047252053, 0.0310298689573541),
    "include_constant-sigmamore": (6.94561004725098, 0.0736517192483979),
}
HAUSMAN_NAMES = ["exper", "expersq"]


def hausman_by_hand(re_res, fe_res, names=HAUSMAN_NAMES):
    delta = (fe_res.params - re_res.params)[names].to_numpy()
    diff = (fe_res.cov - re_res.cov).loc[names, names].to_numpy()
    return float(delta @ np.linalg.pinv(diff) @ delta)


@pytest.fixture
def hausman_data(data):
    data = data.set_index(["nr", "year"])
    return data["hours"], data[HAUSMAN_NAMES]


@pytest.fixture
def re_fe(hausman_data):
    dependent, regressors = hausman_data
    exog = add_constant(regressors)
    re_res = RandomEffects(dependent, exog).fit()
    fe_res = PanelOLS(dependent, exog, entity_effects=True).fit()
    return re_res, fe_res


@pytest.mark.parametrize("variant", ["", "sigmamore", "sigmaless"])
@pytest.mark.parametrize("constant", [False, True], ids=["", "include_constant"])
def test_wu_hausman_stata(re_fe, constant, variant):
    re_res, fe_res = re_fe
    opts = {
        "include_constant": constant,
        "sigmamore": variant == "sigmamore",
        "sigmaless": variant == "sigmaless",
    }
    if constant:
        # The covariance difference is not positive definite with the constant
        with pytest.warns(UserWarning, match="not positive definite") as record:
            wald, estimates = re_res.wu_hausman(other=fe_res, **opts)
        assert all(issubclass(warn.category, UserWarning) for warn in record)
        names = ["const", *HAUSMAN_NAMES]
        assert estimates["Std. Err."].isna().any()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wald, estimates = re_res.wu_hausman(other=fe_res, **opts)
        names = HAUSMAN_NAMES
        assert estimates["Std. Err."].notna().all()
    expected_stat, expected_pval = STATA_HAUSMAN[
        "-".join(k for k, v in opts.items() if v)
    ]
    assert isinstance(wald, WaldTestStatistic)
    assert not isinstance(wald, InvalidTestStatistic)
    assert wald.stat == pytest.approx(expected_stat, abs=1e-9)
    assert wald.pval == pytest.approx(expected_pval, abs=1e-9)
    assert wald.df == len(names)
    assert wald.dist_name == f"chi2({len(names)})"

    assert list(estimates.index) == names
    assert list(estimates.columns) == ["b0", "b1", "b0-b1", "Std. Err."]
    assert_allclose(estimates["b0"], fe_res.params[names])
    assert_allclose(estimates["b1"], re_res.params[names])
    assert_allclose(estimates["b0-b1"], estimates["b0"] - estimates["b1"])


def test_wu_hausman_formula(re_fe):
    re_res, fe_res = re_fe
    expected = hausman_by_hand(re_res, fe_res)
    wald, estimates = re_res.wu_hausman(fe_res)
    assert_allclose(wald.stat, expected)
    assert_allclose(wald.pval, stats.chi2.sf(expected, 2))
    assert wald.df == 2
    assert "No systematic difference" in wald.null
    assert "Hausman specification test" in str(wald)
    diff = (fe_res.cov - re_res.cov).loc[HAUSMAN_NAMES, HAUSMAN_NAMES]
    assert_allclose(estimates["Std. Err."], np.sqrt(np.diag(diff)))


@pytest.mark.parametrize("re_constant", [True, False], ids=["re-const", "re-noconst"])
@pytest.mark.parametrize("fe_constant", [True, False], ids=["fe-const", "fe-noconst"])
@pytest.mark.parametrize(
    "include_constant", [False, True], ids=["", "include_constant"]
)
def test_wu_hausman_constants(hausman_data, re_constant, fe_constant, include_constant):
    dependent, regressors = hausman_data
    re_exog = add_constant(regressors) if re_constant else regressors
    fe_exog = add_constant(regressors) if fe_constant else regressors
    re_res = RandomEffects(dependent, re_exog).fit()
    fe_res = PanelOLS(dependent, fe_exog, entity_effects=True).fit()
    with warnings.catch_warnings():
        # Warnings are only possible when the constant is part of the test
        warnings.simplefilter("ignore", UserWarning)
        wald, estimates = re_res.wu_hausman(fe_res, include_constant=include_constant)
    # The constant can only be compared when both models have one
    names = (
        ["const"] if include_constant and re_constant and fe_constant else []
    ) + HAUSMAN_NAMES
    assert list(estimates.index) == names
    assert wald.df == len(names)
    expected = hausman_by_hand(re_res, fe_res, names)
    assert_allclose(wald.stat, expected, atol=1e-12)


def test_wu_hausman_negative_statistic():
    # In small samples Var(b0) - Var(b1) is often indefinite, and the
    # statistic can be negative, in which case the test is not valid
    rg = np.random.default_rng(3)
    index = pd.MultiIndex.from_product([np.arange(12), np.arange(3)])
    effects = np.repeat(rg.standard_normal(12), 3)
    x = pd.DataFrame(rg.standard_normal((36, 2)), index=index, columns=["x0", "x1"])
    y = pd.Series(
        x.sum(axis=1).to_numpy() + effects + rg.standard_normal(36), index=index
    )
    fe_res = PanelOLS(y, x, entity_effects=True).fit()
    re_res = RandomEffects(y, add_constant(x)).fit()
    with pytest.warns(UserWarning, match="not positive definite"):
        test, estimates = re_res.wu_hausman(fe_res)
    assert isinstance(test, InvalidTestStatistic)
    assert np.isnan(test.stat)
    assert np.isnan(test.pval)
    assert "negative" in str(test)
    assert "Hausman specification test" in str(test)
    assert list(estimates.index) == ["x0", "x1"]
    assert hausman_by_hand(re_res, fe_res, ["x0", "x1"]) < 0


def test_wu_hausman_conflicting_options(re_fe):
    re_res, fe_res = re_fe
    with pytest.raises(ValueError, match="cannot both be True"):
        re_res.wu_hausman(fe_res, sigmamore=True, sigmaless=True)


@pytest.mark.parametrize(
    "other", [None, "fe", np.arange(3)], ids=["none", "str", "array"]
)
def test_wu_hausman_invalid_other(re_fe, other):
    re_res, _ = re_fe
    with pytest.raises(TypeError, match="other must be the results of a panel model"):
        re_res.wu_hausman(other)


@pytest.mark.parametrize("which", ["other", "self"])
@pytest.mark.parametrize(
    ("cov_type", "cov_config"),
    [
        ("robust", {}),
        ("clustered", {"cluster_entity": True}),
        ("kernel", {}),
        ("autocorrelated", {}),
    ],
)
def test_wu_hausman_requires_unadjusted_cov(hausman_data, which, cov_type, cov_config):
    dependent, regressors = hausman_data
    exog = add_constant(regressors)
    re_kwargs = fe_kwargs = {}
    if which == "self":
        re_kwargs = {"cov_type": cov_type, **cov_config}
    else:
        fe_kwargs = {"cov_type": cov_type, **cov_config}
    re_res = RandomEffects(dependent, exog).fit(**re_kwargs)
    fe_res = PanelOLS(dependent, exog, entity_effects=True).fit(**fe_kwargs)
    with pytest.raises(TypeError, match="unadjusted covariance estimator"):
        re_res.wu_hausman(fe_res)


def test_wu_hausman_conventional_covariance(hausman_data):
    # "conventional" and "homoskedastic" are aliases of the unadjusted estimator
    dependent, regressors = hausman_data
    exog = add_constant(regressors)
    re_res = RandomEffects(dependent, exog).fit(cov_type="homoskedastic")
    fe_res = PanelOLS(dependent, exog, entity_effects=True).fit(cov_type="conventional")
    wald, _ = re_res.wu_hausman(fe_res)
    assert_allclose(wald.stat, STATA_HAUSMAN[""][0], atol=1e-9)


def test_wu_hausman_different_observations(hausman_data):
    dependent, regressors = hausman_data
    exog = add_constant(regressors)
    re_res = RandomEffects(dependent, exog).fit()
    fe_res = PanelOLS(dependent.iloc[:-8], exog.iloc[:-8], entity_effects=True).fit()
    with pytest.raises(ValueError, match="same observations"):
        re_res.wu_hausman(fe_res)


def test_wu_hausman_no_common_coefficients(hausman_data, data):
    dependent, _ = hausman_data
    data = data.set_index(["nr", "year"])
    re_res = RandomEffects(dependent, add_constant(data[HAUSMAN_NAMES])).fit()
    fe_res = PanelOLS(dependent, data[["married", "union"]], entity_effects=True).fit()
    with pytest.raises(ValueError, match="coefficients in common"):
        re_res.wu_hausman(fe_res)
    # Only the constant is shared
    fe_const = PanelOLS(dependent, add_constant(data[["married"]]), entity_effects=True)
    with pytest.raises(ValueError, match="coefficients in common"):
        re_res.wu_hausman(fe_const.fit())
