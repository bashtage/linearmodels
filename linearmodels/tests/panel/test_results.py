from itertools import combinations, product
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
from linearmodels.panel.model import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.results import compare
from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    InvalidTestStatistic,
    NormalTestStatistic,
    WaldTestStatistic,
)
from linearmodels.tests.panel._utility import datatypes, generate_data


@pytest.fixture(params=[wage_panel.load()])
def data(request):
    return request.param


perc_missing = [0.0, 0.02, 0.20]
has_const = [True, False]
perms = list(product(perc_missing, datatypes, has_const))
ids = ["-".join(str(param) for param in perm) for perm in perms]


def pesaran_panel(seed, nentity, nperiod, loading=0.0, drop=0.0, min_obs=2):
    """
    Simulate a single-regressor panel with a common factor in the shocks

    ``loading`` scales the entity-specific loading on the common factor, so
    that ``loading=0`` gives cross-sectionally independent shocks. Each
    observation is dropped with probability ``drop`` and entities left with
    fewer than ``min_obs`` observations are removed.
    """
    rs = np.random.RandomState(seed)
    factor = rs.standard_normal(nperiod)
    load = rs.standard_normal(nentity)
    x = rs.standard_normal((nentity, nperiod)) + 0.3 * factor
    shock = loading * load[:, None] * factor + rs.standard_normal((nentity, nperiod))
    keep = rs.uniform(size=(nentity, nperiod)) >= drop
    keep &= keep.sum(1, keepdims=True) >= min_obs
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    frame = pd.DataFrame(
        {"y": (1.0 + 0.5 * x + shock).ravel(), "x": x.ravel()}, index=index
    )
    return frame.loc[keep.ravel()]


def brute_force_pesaran_cd(idiosyncratic):
    """Pesaran CD using explicit loops over entity pairs"""
    resid = idiosyncratic.iloc[:, 0]
    by_entity = {key: grp.droplevel(0) for key, grp in resid.groupby(level=0)}
    total = 0.0
    npairs = 0
    for first, second in combinations(by_entity, 2):
        common = by_entity[first].index.intersection(by_entity[second].index)
        if len(common) < 2:
            continue
        rho = np.corrcoef(by_entity[first][common], by_entity[second][common])[0, 1]
        if np.isfinite(rho):
            total += np.sqrt(len(common)) * rho
            npairs += 1
    return total / np.sqrt(npairs)


PESARAN_PANELS = {
    "balanced-null": {"seed": 1, "nentity": 40, "nperiod": 8},
    "balanced-dependent": {"seed": 2, "nentity": 40, "nperiod": 8, "loading": 0.6},
    "unbalanced-null": {
        "seed": 3,
        "nentity": 40,
        "nperiod": 10,
        "drop": 0.3,
        "min_obs": 3,
    },
    "unbalanced-dependent": {
        "seed": 4,
        "nentity": 40,
        "nperiod": 10,
        "loading": 0.5,
        "drop": 0.3,
        "min_obs": 3,
    },
    # Many entity pairs have fewer than 2 common time periods
    "sparse": {"seed": 5, "nentity": 40, "nperiod": 8, "drop": 0.55},
}


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


# Reference values for the Pesaran CD test, (statistic, p-value), computed with
# plm::pcdtest(test="cd") from R 4.6.1 and plm 2.6.7. The panels in
# PESARAN_PANELS were written to CSV (pesaran_panel(**kwargs).reset_index())
# and read back in R using
#
#   pd <- pdata.frame(read.csv(file), index = c("entity", "time"))
#   pcdtest(plm(y ~ x, data = pd, model = "pooling"), test = "cd")
#   pcdtest(plm(y ~ x, data = pd, model = "within"), test = "cd")
#
# The "random" model is not used since plm and linearmodels use different
# estimators of the variance components. Pairs of entities with fewer than 2
# common periods are dropped by plm, as they are here.
PLM_PESARAN_CD = {
    ("balanced-null", "pooled"): (1.65059180202, 0.098821953974),
    ("balanced-null", "within"): (1.82523389124, 0.0679657414809),
    ("balanced-dependent", "pooled"): (1.08642468176, 0.27729114669),
    ("balanced-dependent", "within"): (1.18632438201, 0.235494221437),
    ("unbalanced-null", "pooled"): (0.877751079133, 0.380078818139),
    ("unbalanced-null", "within"): (0.982498727879, 0.325854209365),
    ("unbalanced-dependent", "pooled"): (-1.25144189714, 0.210773299518),
    ("unbalanced-dependent", "within"): (-0.795027241495, 0.426597656),
    ("sparse", "pooled"): (-0.0627630427581, 0.949955195428),
    ("sparse", "within"): (-0.0831274553108, 0.933750195211),
}


def fit_pesaran_model(frame, model):
    if model == "pooled":
        return PooledOLS(frame.y, add_constant(frame[["x"]])).fit()
    return PanelOLS(frame.y, frame[["x"]], entity_effects=True).fit()


@pytest.mark.parametrize(("case", "model"), list(PLM_PESARAN_CD))
def test_pesaran_cd_plm(case, model):
    frame = pesaran_panel(**PESARAN_PANELS[case])
    cd = fit_pesaran_model(frame, model).pesaran_cd
    assert isinstance(cd, NormalTestStatistic)
    stat, pval = PLM_PESARAN_CD[(case, model)]
    assert_allclose(cd.stat, stat, rtol=1e-8)
    assert_allclose(cd.pval, pval, rtol=1e-8)


def test_pesaran_cd_plm_wage_panel(data):
    # Same plm calls as above, using plm(lwage ~ expersq + married + union,
    # model="within") on the wage panel. The tiny p-value is only accurate when
    # computed from the survival function.
    dependent = data.set_index(["nr", "year"]).lwage
    exog = add_constant(data.set_index(["nr", "year"])[["expersq", "married", "union"]])
    cd = PanelOLS(dependent, exog, entity_effects=True).fit().pesaran_cd
    assert isinstance(cd, NormalTestStatistic)
    assert_allclose(cd.stat, 7.81128099174, rtol=1e-8)
    assert_allclose(cd.pval, 5.66096490562e-15, rtol=1e-8)


@pytest.mark.parametrize(
    "estimator",
    [
        "pooled",
        "entity",
        "entity-time",
        "random",
        "between",
        "first-difference",
        "fama-macbeth",
    ],
)
def test_pesaran_cd_all_estimators(estimator):
    frame = pesaran_panel(**PESARAN_PANELS["unbalanced-dependent"])
    exog = add_constant(frame[["x"]])
    models = {
        "pooled": lambda: PooledOLS(frame.y, exog),
        "entity": lambda: PanelOLS(frame.y, frame[["x"]], entity_effects=True),
        "entity-time": lambda: PanelOLS(
            frame.y, frame[["x"]], entity_effects=True, time_effects=True
        ),
        "random": lambda: RandomEffects(frame.y, exog),
        "between": lambda: BetweenOLS(frame.y, exog),
        "first-difference": lambda: FirstDifferenceOLS(frame.y, frame[["x"]]),
        "fama-macbeth": lambda: FamaMacBeth(frame.y, exog),
    }
    res = models[estimator]().fit()
    cd = res.pesaran_cd
    assert isinstance(cd, NormalTestStatistic)
    expected = brute_force_pesaran_cd(res.idiosyncratic)
    assert_allclose(cd.stat, expected, rtol=1e-10)
    assert_allclose(cd.pval, 2 * stats.norm.sf(abs(expected)), rtol=1e-10)


def test_pesaran_cd_string_entities_and_dates():
    frame = pesaran_panel(**PESARAN_PANELS["unbalanced-null"])
    expected = fit_pesaran_model(frame, "pooled").pesaran_cd.stat
    entity = frame.index.get_level_values(0)
    time = pd.to_datetime("2000-01-01") + pd.to_timedelta(
        frame.index.get_level_values(1), unit="D"
    )
    index = pd.MultiIndex.from_arrays(
        [[f"firm{i:02d}" for i in entity], time], names=["entity", "time"]
    )
    relabeled = frame.set_axis(index)
    cd = fit_pesaran_model(relabeled, "pooled").pesaran_cd
    assert_allclose(cd.stat, expected, rtol=1e-10)


def test_pesaran_cd_skips_constant_residuals():
    # With only a constant, residuals are the demeaned outcome. The constant
    # entity has an undefined correlation with every other entity, so only the
    # pair (0, 1) enters the statistic.
    index = pd.MultiIndex.from_product([[0, 1, 2], range(5)], names=["i", "t"])
    y = pd.Series(
        np.r_[[1.0, 3.0, 2.0, 5.0, 4.0], [2.0, 2.5, 1.0, 4.0, 4.5], np.full(5, 2.0)],
        index=index,
        name="y",
    )
    res = PooledOLS(y, pd.DataFrame({"const": 1.0}, index=index)).fit()
    cd = res.pesaran_cd
    rho = np.corrcoef(y.loc[0].to_numpy(), y.loc[1].to_numpy())[0, 1]
    assert isinstance(cd, NormalTestStatistic)
    assert_allclose(cd.stat, np.sqrt(5) * rho, rtol=1e-10)


@pytest.mark.parametrize("common", [0, 1])
def test_pesaran_cd_no_overlap(common):
    # Entity 0 is observed in periods 0-3, entity 1 in periods 4-7 (common=0) or
    # 3-6 (common=1), so no pair of entities has two or more common periods.
    start = 4 - common
    time = np.r_[np.arange(4), np.arange(start, start + 4)]
    entity = np.repeat([0, 1], 4)
    index = pd.MultiIndex.from_arrays([entity, time], names=["i", "t"])
    gen = np.random.RandomState(0)
    y = pd.Series(gen.standard_normal(8), index=index, name="y")
    x = pd.DataFrame({"const": 1.0, "x": gen.standard_normal(8)}, index=index)
    cd = PooledOLS(y, x).fit().pesaran_cd
    assert isinstance(cd, InapplicableTestStatistic)
    assert np.isnan(cd.stat)
    assert np.isnan(cd.pval)
    assert "two or more overlapping observations" in str(cd)


def test_pesaran_cd_single_entity():
    index = pd.MultiIndex.from_product([["firm0"], range(8)], names=["firm", "time"])
    y = pd.Series(np.linspace(0.0, 1.0, 8), index=index, name="y")
    x = pd.DataFrame({"const": 1.0, "x1": np.linspace(-1.0, 1.0, 8)}, index=index)
    res = PooledOLS(y, x).fit()
    cd = res.pesaran_cd
    assert isinstance(cd, InapplicableTestStatistic)
    assert np.isnan(cd.pval)
    assert "at least two entities" in str(cd)


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
