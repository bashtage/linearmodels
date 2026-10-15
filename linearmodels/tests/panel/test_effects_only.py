"""
PanelOLS models that only have effects

    y_it = a_i + g_t + e_it

The model has no regressors: exog has no columns, or the formula has none, for
example "y ~ EntityEffects". The estimated effects are the means of y of the entities
and the periods (the fitted values of the regression on the dummy variables if there
are both), the residuals are y less the effects, and the fitted values of the
regressors are 0. These were errors from pandas, "No objects to concatenate".

The references are the means, the projection on the dummy variables and the statistics
that follow from them, written out with NumPy and pandas, the one-way analysis of
variance of SciPy, and the regression on the dummy variables of statsmodels.
"""

import re
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
import xarray as xr

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
    compare,
)
from linearmodels.panel.data import PanelData
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.shared.hypotheses import InvalidTestStatistic
from linearmodels.tests._utility import assert_same_text

NENTITY = 8
NPERIOD = 6


def make_data(unbalanced, seed=21):
    """The dependent variable, the weights and a third category, with the index"""
    rg = np.random.default_rng(seed)
    rows = []
    for entity in range(NENTITY):
        if unbalanced:
            keep = rg.random(NPERIOD) > 0.3
            keep[rg.choice(NPERIOD, 3, replace=False)] = True
        else:
            keep = np.ones(NPERIOD, dtype=bool)
        rows.extend((entity, int(t)) for t in np.flatnonzero(keep))
    index = pd.MultiIndex.from_tuples(rows, names=["entity", "time"])
    nobs = index.shape[0]
    entity_effect = rg.standard_normal(NENTITY)[index.codes[0]]
    time_effect = rg.standard_normal(NPERIOD)[index.get_level_values(1)]
    y = 1 + entity_effect + 0.5 * time_effect + rg.standard_normal(nobs)
    frame = pd.DataFrame(
        {
            "y": y,
            "w": rg.chisquare(4, nobs) / 4 + 0.2,
            "g": (index.codes[0] + 2 * index.get_level_values(1).to_numpy()) % 3,
        },
        index=index,
    )
    return frame


@pytest.fixture(scope="module", params=[False, True], ids=["balanced", "unbalanced"])
def frame(request):
    return make_data(request.param)


def no_regressors(frame):
    """An exog with the index of the data and no columns"""
    return pd.DataFrame(index=frame.index)


EFFECTS = {
    "entity": {"entity_effects": True},
    "time": {"time_effects": True},
    "both": {"entity_effects": True, "time_effects": True},
    "other": {"other_effects": ["g"]},
    "entity+other": {"entity_effects": True, "other_effects": ["g"]},
}
ALGORITHMS = {"means": {}, "lsdv": {"use_lsdv": True}, "lsmr": {"use_lsmr": True}}
# The effects and the residuals are exact for the dummy variables. The algorithms that
# iterate (LSMR, and the means with other effects) stop at a relative error of 1e-8, and
# their sums of squares are accurate to the square of that
EFFECT_ATOL = {"means": 1e-6, "lsdv": 1e-10, "lsmr": 1e-5}
SUM_OF_SQUARES_RTOL = 1e-8
# The tolerance of the numbers that are zero in exact arithmetic, for example the
# R-squared of a model that only has effects, which is 1 - residual_ss / total_ss and
# is -2.22e-16, 0 or 1.11e-16 depending on the rounding of the platform
ATOL = 1e-12
# The number of effects that are estimated, with no constant to absorb one of them
NEFFECTS = {
    "entity": NENTITY,
    "time": NPERIOD,
    "both": NENTITY + NPERIOD - 1,
    "other": 3,
    "entity+other": NENTITY + 3 - 1,
}


def model_options(frame, effect):
    options = dict(EFFECTS[effect])
    if "other_effects" in options:
        options["other_effects"] = frame[options["other_effects"]]
    return options


def dummies(frame, effect, drop_first=False):
    """
    The dummy variables of the groups of the effects

    If drop_first is False the design does not have full rank with more than one
    effect, which does not matter for the fitted values. With drop_first, a dummy is
    dropped from every effect so that a constant can be added.
    """
    parts = []
    if effect in ("entity", "both", "entity+other"):
        values = frame.index.get_level_values(0)
        parts.append(pd.get_dummies(values, dtype=float, drop_first=drop_first))
    if effect in ("time", "both"):
        values = frame.index.get_level_values(1)
        parts.append(pd.get_dummies(values, dtype=float, drop_first=drop_first))
    if effect in ("other", "entity+other"):
        values = frame["g"].to_numpy()
        parts.append(pd.get_dummies(values, dtype=float, drop_first=drop_first))
    return np.column_stack([p.to_numpy() for p in parts])


def projection(frame, effect, weighted):
    """The weighted least squares fit on the dummy variables of the effects"""
    d = dummies(frame, effect)
    w = frame["w"].to_numpy() / frame["w"].mean() if weighted else np.ones(len(frame))
    root_w = np.sqrt(w)[:, None]
    y = frame["y"].to_numpy()[:, None]
    coef = np.linalg.lstsq(root_w * d, root_w * y, rcond=None)[0]
    fitted = (d @ coef).ravel()
    return fitted, w


def fit_effects_only(frame, effect, weighted, algorithm="means", **fit_options):
    weights = frame["w"] if weighted else None
    mod = PanelOLS(
        frame["y"],
        no_regressors(frame),
        weights=weights,
        **model_options(frame, effect),
    )
    return mod.fit(**ALGORITHMS[algorithm], **fit_options)


# --------------------------------------------------------------------------
# Hand computations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("effect", EFFECTS)
def test_effects_and_residuals_are_the_means(frame, effect, weighted, algorithm):
    res = fit_effects_only(frame, effect, weighted, algorithm)
    fitted, _ = projection(frame, effect, weighted)
    y = frame["y"].to_numpy()
    atol = EFFECT_ATOL[algorithm]
    assert res.params.shape == (0,)
    assert_allclose(res.estimated_effects.to_numpy().ravel(), fitted, atol=atol)
    assert_allclose(res.fitted_values.to_numpy().ravel(), 0.0, atol=ATOL)
    assert_allclose(res.idiosyncratic.to_numpy().ravel(), y - fitted, atol=atol)
    assert_allclose(res.resids.to_numpy(), y - fitted, atol=atol)
    assert res.estimated_effects.index.equals(frame.index)
    # y is the sum of the parts, whichever the weights
    total = res.fitted_values.to_numpy() + res.estimated_effects.to_numpy()
    total = total + res.idiosyncratic.to_numpy()
    assert_allclose(total.ravel(), y)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_entity_and_time_effects_are_the_group_means(frame, weighted):
    # The (weighted) mean of each group is the least squares fit on its dummy variable
    w = frame["w"] if weighted else pd.Series(1.0, index=frame.index)
    weighted_y = (w * frame["y"]).groupby(level=0).transform("sum")
    by_entity = weighted_y / w.groupby(level=0).transform("sum")
    res = fit_effects_only(frame, "entity", weighted)
    assert_allclose(res.estimated_effects.to_numpy().ravel(), by_entity.to_numpy())
    weighted_y = (w * frame["y"]).groupby(level=1).transform("sum")
    by_time = weighted_y / w.groupby(level=1).transform("sum")
    res = fit_effects_only(frame, "time", weighted)
    assert_allclose(res.estimated_effects.to_numpy().ravel(), by_time.to_numpy())


def test_two_way_effects_of_a_balanced_panel():
    # The effects of the entities and the periods are additive: the mean of the
    # entity plus the mean of the period less the overall mean
    frame = make_data(False)
    y = frame["y"]
    expected = (
        y.groupby(level=0).transform("mean")
        + y.groupby(level=1).transform("mean")
        - y.mean()
    )
    res = fit_effects_only(frame, "both", False)
    assert_allclose(res.estimated_effects.to_numpy().ravel(), expected.to_numpy())


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("effect", EFFECTS)
def test_statistics_against_hand_computation(frame, effect, weighted, algorithm):
    res = fit_effects_only(frame, effect, weighted, algorithm)
    fitted, w = projection(frame, effect, weighted)
    y = frame["y"].to_numpy()
    nobs = y.shape[0]
    neffects = NEFFECTS[effect]
    resid_ss = float(np.sum(w * (y - fitted) ** 2))
    centered = float(np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2))

    rtol = SUM_OF_SQUARES_RTOL
    assert res.nobs == nobs
    assert res.df_model == neffects
    assert res.df_resid == nobs - neffects
    assert_allclose(res.resid_ss, resid_ss, rtol=rtol)
    # The effects are removed from y, and nothing else can explain it
    assert_allclose(res.total_ss, resid_ss, rtol=rtol)
    assert_allclose(res.model_ss, 0.0, atol=rtol)
    for name in ("rsquared", "rsquared_within", "rsquared_between", "rsquared_overall"):
        assert_allclose(getattr(res, name), 0.0, atol=rtol, err_msg=name)
    for name in ("within", "between", "overall"):
        corr_squared = getattr(res, f"corr_squared_{name}")
        assert_allclose(corr_squared, 0.0, atol=ATOL, err_msg=name)
    assert_allclose(res.rsquared_inclusive, 1 - resid_ss / centered, rtol=rtol)
    assert_allclose(res.s2, resid_ss / (nobs - neffects), rtol=rtol)
    # The log-likelihood of the regression on a constant and the dummy variables,
    # which PooledOLS estimates with the same weights
    design = pd.DataFrame(
        sm.add_constant(dummies(frame, effect, drop_first=True)), index=frame.index
    )
    pooled = PooledOLS(
        frame["y"], design, weights=frame["w"] if weighted else None
    ).fit()
    assert_allclose(res.loglik, pooled.loglik, rtol=rtol)
    if not weighted:
        sigma2 = resid_ss / nobs
        loglik = -0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        assert_allclose(res.loglik, loglik, rtol=rtol)
        # There is no constant, so that the variance of the effects is around zero
        total = float(y @ y) / nobs
        decomposition = res.variance_decomposition
        assert_allclose(decomposition["Residual"], sigma2, rtol=rtol)
        assert_allclose(decomposition["Effects"], total - sigma2, rtol=rtol)
        share = (total - sigma2) / total
        assert_allclose(decomposition["Percent due to Effects"], share, rtol=rtol)
    # The test that the effects are zero is the test that the constant is the only
    # coefficient of the regression on the dummy variables
    df_num, df_denom = neffects - 1, nobs - neffects
    f_stat = ((centered - resid_ss) / df_num) / (resid_ss / df_denom)
    assert_allclose(res.f_pooled.stat, f_stat, rtol=rtol)
    assert_allclose(res.f_pooled.pval, stats.f.sf(f_stat, df_num, df_denom), rtol=rtol)
    assert res.f_pooled.df == df_num
    assert res.f_pooled.df_denom == df_denom


@pytest.mark.parametrize("effect", ["entity", "time"])
def test_pooled_f_against_one_way_analysis_of_variance(frame, effect):
    level = 0 if effect == "entity" else 1
    groups = [g.to_numpy() for _, g in frame["y"].groupby(level=level)]
    anova = stats.f_oneway(*groups)
    res = fit_effects_only(frame, effect, False)
    assert_allclose(res.f_pooled.stat, anova.statistic)
    assert_allclose(res.f_pooled.pval, anova.pvalue)


@pytest.mark.parametrize("effect", list(EFFECTS))
def test_pooled_f_against_the_regression_on_dummy_variables(frame, effect):
    # The F-test of the regression on a constant and the dummy variables
    design = sm.add_constant(dummies(frame, effect, drop_first=True))
    ols = sm.OLS(frame["y"].to_numpy(), design).fit()
    res = fit_effects_only(frame, effect, False)
    assert_allclose(res.f_pooled.stat, ols.fvalue)
    assert_allclose(res.f_pooled.pval, ols.f_pvalue)
    assert res.f_pooled.df == ols.df_model


def test_robust_and_classical_f_statistics_are_invalid(frame):
    res = fit_effects_only(frame, "entity", False)
    for stat in (res.f_statistic, res.f_statistic_robust):
        assert isinstance(stat, InvalidTestStatistic)
        assert np.isnan(stat.stat)
        assert np.isnan(stat.pval)
        assert "no regressors" in str(stat)


COVARIANCE = {
    "unadjusted": {"cov_type": "unadjusted"},
    "not-debiased": {"cov_type": "unadjusted", "debiased": False},
    "robust": {"cov_type": "robust"},
    "cluster-entity": {"cov_type": "clustered", "cluster_entity": True},
    "cluster-time": {"cov_type": "clustered", "cluster_time": True},
    "cluster-both": {
        "cov_type": "clustered",
        "cluster_entity": True,
        "cluster_time": True,
    },
    "kernel": {"cov_type": "kernel", "bandwidth": 2},
    "ac": {"cov_type": "ac", "bandwidth": 2},
}


@pytest.mark.parametrize("cov", list(COVARIANCE))
def test_tables_have_no_rows(frame, cov):
    cov_options = COVARIANCE[cov]
    res = fit_effects_only(frame, "entity", False, **cov_options)
    assert res.params.shape == (0,)
    assert res.cov.shape == (0, 0)
    for table in (res.std_errors, res.tstats, res.pvalues):
        assert table.shape == (0,)
    ci = res.conf_int(0.9)
    assert ci.shape == (0, 2)
    assert list(ci.columns) == ["lower", "upper"]
    # The effects are not counted in the error variance if they are nested in the
    # clusters, which entity effects are in the entities
    resid_ss = float(res.resids.to_numpy() @ res.resids.to_numpy())
    nested = cov == "cluster-entity"
    assert_allclose(res.s2, resid_ss / (res.nobs if nested else res.df_resid))


def test_summary(frame):
    res = fit_effects_only(frame, "both", True)
    summary = res.summary
    text = summary.as_text()
    assert "PanelOLS Estimation Summary" in text
    assert re.search(rf"No. Observations:\s+{res.nobs}\b", text)
    # There are no parameters to show
    assert "Parameter Estimates" not in text
    assert "F-test for Poolability" in text
    assert "Included effects: Entity, Time" in text
    # The other formats work without the table of parameters
    for as_format in (summary.as_html, summary.as_latex, summary.as_csv):
        assert len(as_format()) > 100
    assert_same_text(str(res), text)


def test_predict_and_wald_test(frame):
    res = fit_effects_only(frame, "entity", False)
    out = res.predict(effects=True, idiosyncratic=True)
    assert list(out.columns) == ["fitted_values", "estimated_effects", "idiosyncratic"]
    assert_allclose(out["fitted_values"], 0.0, atol=ATOL)
    assert_allclose(out.sum(axis=1).to_numpy(), frame["y"].to_numpy())
    # Out of sample predictions with no regressors are 0
    new = res.predict(exog=no_regressors(frame).iloc[:5])
    assert_allclose(new.to_numpy(), 0.0, atol=ATOL)
    with pytest.raises(ValueError, match="The model has no parameters"):
        res.wald_test(formula="x = 0")
    with pytest.raises(ValueError, match="The model has no parameters"):
        res.wald_test(np.zeros((1, 0)), np.zeros(1))


def test_compare(frame):
    exog = pd.DataFrame({"x": np.arange(len(frame)) % 7 + 0.0}, index=frame.index)
    exog["x"] += frame["y"]
    with_x = PanelOLS(frame["y"], exog, entity_effects=True).fit()
    only = fit_effects_only(frame, "entity", False)
    both = compare({"effects": only, "regressor": with_x})
    text = both.summary.as_text()
    assert "effects" in text
    assert "x" in text
    # Only models without regressors have no parameters to show
    alone = compare({"effects": only})
    assert "Model Comparison" in alone.summary.as_text()
    assert alone.params.shape[0] == 0


# --------------------------------------------------------------------------
# The inputs
# --------------------------------------------------------------------------
def test_no_regressors_without_effects_is_an_error(frame):
    with pytest.raises(ValueError, match=r"exog has no columns and the model has no"):
        PanelOLS(frame["y"], no_regressors(frame))
    with pytest.raises(ValueError, match=r"exog has no columns and the model has no"):
        PanelOLS.from_formula("y ~ 0", frame)


@pytest.mark.parametrize(
    "cls",
    [PooledOLS, BetweenOLS, FirstDifferenceOLS, RandomEffects, FamaMacBeth],
)
def test_other_models_need_regressors(frame, cls):
    match = rf"{cls.__name__} needs at least one regressor, but exog has no columns"
    with pytest.raises(ValueError, match=match):
        cls(frame["y"], no_regressors(frame))
    with pytest.raises(ValueError, match=match):
        cls.from_formula("y ~ 0", frame)


def test_forms_of_exog_without_variables():
    # An array and a DataArray with 0 variables, and a PanelData, give the same results
    rg = np.random.default_rng(9)
    y = rg.standard_normal((NPERIOD, NENTITY))
    entities = [f"firm{i}" for i in range(NENTITY)]
    time = pd.date_range("2000-01-01", periods=NPERIOD, freq="MS")
    y_frame = pd.DataFrame(y, index=time, columns=entities)
    forms = {
        "ndarray": np.empty((0, NPERIOD, NENTITY)),
        "DataArray": xr.DataArray(
            np.empty((0, NPERIOD, NENTITY)),
            coords={"vars": [], "time": time, "entities": entities},
            dims=["vars", "time", "entities"],
        ),
        "PanelData": PanelData(
            pd.DataFrame(index=PanelData(y_frame).dataframe.index),
            convert_dummies=False,
        ),
        "DataFrame": pd.DataFrame(index=PanelData(y_frame).dataframe.index),
    }
    expected = PanelOLS(y_frame, forms["DataFrame"], entity_effects=True).fit()
    assert_allclose(
        expected.estimated_effects.to_numpy().ravel(),
        PanelData(y_frame)
        .dataframe.groupby(level=0)
        .transform("mean")
        .to_numpy()
        .ravel(),
    )
    for name, exog in forms.items():
        mod = PanelOLS(y_frame, exog, entity_effects=True)
        assert mod.exog.nvar == 0, name
        res = mod.fit()
        assert_allclose(res.resid_ss, expected.resid_ss, err_msg=name)
        assert_allclose(res.estimated_effects, expected.estimated_effects)


def test_y_only_arrays():
    # A 2-d array of y, time by entity, and exog without variables
    rg = np.random.default_rng(10)
    y = rg.standard_normal((NPERIOD, NENTITY))
    res = PanelOLS(y, np.empty((0, NPERIOD, NENTITY)), entity_effects=True).fit()
    assert_allclose(res.resid_ss, np.sum((y - y.mean(0)) ** 2))
    assert res.df_resid == NPERIOD * NENTITY - NENTITY


def test_missing_values_in_dependent_warn(frame):
    data = frame.copy()
    data.iloc[[2, 11], 0] = np.nan
    keep = data["y"].notna().to_numpy()
    with pytest.warns(MissingValueWarning):
        mod = PanelOLS(data["y"], no_regressors(data), entity_effects=True)
    expected = PanelOLS(
        data["y"][keep], no_regressors(data)[keep], entity_effects=True
    ).fit()
    res = mod.fit()
    assert res.nobs == keep.sum()
    assert_allclose(res.resid_ss, expected.resid_ss)
    assert_allclose(res.estimated_effects, expected.estimated_effects)


def test_no_warning_without_missing_values(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PanelOLS(frame["y"], no_regressors(frame), entity_effects=True).fit()


@pytest.mark.parametrize("effect", ["entity", "time", "both"])
def test_options_of_the_model(frame, effect):
    options = model_options(frame, effect)
    mod = PanelOLS(
        frame["y"],
        no_regressors(frame),
        weights=frame["w"],
        singletons=False,
        drop_absorbed=True,
        check_rank=False,
        **options,
    )
    res = mod.fit(low_memory=True)
    expected = fit_effects_only(frame, effect, True)
    assert_allclose(res.resid_ss, expected.resid_ss)
    assert str(mod).startswith("PanelOLS")
    assert "Num exog: 0" in str(mod)


# --------------------------------------------------------------------------
# Formulas
# --------------------------------------------------------------------------
SKIP_ATTRIBUTES = {"model", "summary", "conf_int", "predict", "wald_test", "compare"}


def assert_same_table(value, other, label):
    """The labels of two Series or DataFrames are the same, and the values are close"""
    assert type(value) is type(other), label
    assert value.index.equals(other.index), label
    assert value.shape == other.shape, label
    if isinstance(value, pd.DataFrame):
        assert value.columns.equals(other.columns), label
    assert_allclose(
        value.to_numpy(dtype=float),
        other.to_numpy(dtype=float),
        atol=ATOL,
        equal_nan=True,
        err_msg=label,
    )


def assert_same_results(res, expected):
    """Every public attribute of two results is the same, up to rounding"""
    assert type(res) is type(expected)
    for name in dir(res):
        if name.startswith("_") or name in SKIP_ATTRIBUTES:
            continue
        value, other = getattr(res, name), getattr(expected, name)
        label = f"attribute {name}"
        if isinstance(value, (pd.Series, pd.DataFrame)):
            assert_same_table(value, other, label)
        elif hasattr(value, "stat"):
            assert type(value) is type(other), label
            assert_same_text(str(value), str(other), label)
            assert_allclose(value.stat, other.stat, equal_nan=True, err_msg=label)
            assert_allclose(value.pval, other.pval, equal_nan=True, err_msg=label)
        elif isinstance(value, (int, float, np.number)):
            assert_allclose(value, other, atol=ATOL, equal_nan=True, err_msg=label)
        else:
            # Names of the effects, which are text
            assert value == other, label
    assert_same_table(res.conf_int(), expected.conf_int(), "conf_int")
    assert_same_table(res.conf_int(0.8), expected.conf_int(0.8), "conf_int(0.8)")

    def text(results):
        out = results.summary.as_text()
        out = re.sub(r"Date:\s+\w+, \w+ \d+ \d+", "Date:", out)
        return re.sub(r"Time:\s+\d+:\d+:\d+", "Time:", out)

    assert_same_text(text(res), text(expected), "summary")
    assert_allclose(
        res.predict(effects=True, idiosyncratic=True),
        expected.predict(effects=True, idiosyncratic=True),
        atol=ATOL,
    )


FORMULAS = {
    "y ~ EntityEffects": {"entity_effects": True},
    "y ~ TimeEffects": {"time_effects": True},
    "y ~ EntityEffects + TimeEffects": {"entity_effects": True, "time_effects": True},
    "y ~ TimeEffects + EntityEffects": {"entity_effects": True, "time_effects": True},
    "y ~ FixedEffects": {"entity_effects": True},
    "y ~ 0 + EntityEffects": {"entity_effects": True},
    "y ~ -1 + EntityEffects + TimeEffects": {
        "entity_effects": True,
        "time_effects": True,
    },
}


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("formula", list(FORMULAS))
def test_formula_equals_the_model_without_regressors(frame, formula, weighted):
    weights = frame["w"] if weighted else None
    direct = PanelOLS(
        frame["y"], no_regressors(frame), weights=weights, **FORMULAS[formula]
    ).fit()
    mod = PanelOLS.from_formula(formula, frame, weights=weights)
    assert mod.formula == formula
    assert mod.exog.nvar == 0
    assert_same_results(mod.fit(), direct)


@pytest.mark.parametrize("effect", ["other", "entity+other"])
def test_formula_with_other_effects(frame, effect):
    options = model_options(frame, effect)
    direct = PanelOLS(frame["y"], no_regressors(frame), **options).fit()
    formula = "y ~ EntityEffects" if "entity_effects" in options else "y ~ 0"
    mod = PanelOLS.from_formula(formula, frame, other_effects=options["other_effects"])
    assert_same_results(mod.fit(), direct)


@pytest.mark.parametrize("cov", ["robust", "ac"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_formula_equals_the_model_with_fit_options(frame, algorithm, cov):
    cov_options = COVARIANCE[cov]
    direct = fit_effects_only(frame, "both", True, algorithm, **cov_options)
    mod = PanelOLS.from_formula(
        "y ~ EntityEffects + TimeEffects", frame, weights=frame["w"]
    )
    res = mod.fit(**ALGORITHMS[algorithm], **cov_options)
    assert_same_results(res, direct)


def test_formula_prediction_with_data(frame):
    mod = PanelOLS.from_formula("y ~ EntityEffects", frame)
    res = mod.fit()
    out = res.predict(data=frame.iloc[:10])
    assert out.shape == (10, 1)
    assert_allclose(out.to_numpy(), 0.0, atol=ATOL)


def test_formula_with_a_constant_is_not_a_model_without_regressors(frame):
    # A constant is a regressor
    mod = PanelOLS.from_formula("y ~ 1 + EntityEffects", frame)
    res = mod.fit()
    assert list(res.params.index) == ["Intercept"]
    assert_allclose(res.params.iloc[0], frame["y"].mean())
