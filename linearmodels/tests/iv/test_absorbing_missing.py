"""
AbsorbingLS drops observations with missing values in any of its inputs.

The estimates must be the same as the estimates from a model that is fit to the
same observations after the rows with missing values have been dropped by hand.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import pytest

from linearmodels.iv import IV2SLS
from linearmodels.iv.absorbing import AbsorbingLS, Interaction
from linearmodels.panel.utility import AbsorbingEffectWarning
from linearmodels.shared.exceptions import MissingValueWarning

NOBS = 240
# Rows given a missing value, including a row that appears twice
MISSING_ROWS = [3, 17, 17, 101, 233]


def make_data(weighted, seed=0, labelled=True):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"obs{i}" for i in range(NOBS)]) if labelled else None
    x = pd.DataFrame(rng.standard_normal((NOBS, 2)), columns=["x0", "x1"], index=index)
    cat = pd.DataFrame(
        {
            "g0": pd.Categorical(rng.integers(0, 8, NOBS)),
            "g1": pd.Categorical(rng.integers(0, 5, NOBS)),
        },
        index=index,
    )
    cont = pd.DataFrame({"z": rng.standard_normal(NOBS)}, index=index)
    inter_cat = pd.DataFrame(
        {"ic": pd.Categorical(rng.integers(0, 3, NOBS))}, index=index
    )
    inter_cont = pd.DataFrame({"ix": rng.standard_normal(NOBS)}, index=index)
    codes = cat["g0"].cat.codes.to_numpy()
    y = (
        0.5 * x["x0"]
        - 0.25 * x["x1"]
        + 0.1 * codes
        + 0.3 * cont["z"]
        + rng.standard_normal(NOBS)
    )
    y.name = "y"
    weights = None
    if weighted:
        weights = pd.Series(
            np.exp(0.5 * rng.standard_normal(NOBS)), index=index, name="w"
        )
    return {
        "dependent": y,
        "exog": x,
        "absorb_cat": cat,
        "absorb_cont": cont,
        "interaction_cat": inter_cat,
        "interaction_cont": inter_cont,
        "weights": weights,
    }


def add_missing(data, location, rows=MISSING_ROWS):
    """Replace values in the input called location with missing values"""
    data = {key: None if v is None else v.copy() for key, v in data.items()}
    value = data[location]
    if location == "exog":
        # Both columns, in different rows
        value.iloc[rows[:2], 0] = np.nan
        value.iloc[rows[2:], 1] = np.nan
    elif location in ("absorb_cat", "interaction_cat"):
        # Categoricals with missing values need to be built from floats
        col = value.columns[0]
        values = value[col].astype(float).to_numpy().copy()
        values[rows] = np.nan
        value[col] = pd.Categorical(values)
    elif isinstance(value, pd.DataFrame):
        value.iloc[rows, 0] = np.nan
    else:
        value.iloc[rows] = np.nan
    return data


def make_model(data, absorb_type, interact, **kwargs):
    """Construct a model from the inputs, using the forms AbsorbingLS accepts"""
    if absorb_type == "categorical":
        absorb = data["absorb_cat"]
    elif absorb_type == "mixed":
        absorb = pd.concat([data["absorb_cat"], data["absorb_cont"]], axis=1)
    elif absorb_type == "interaction":
        absorb = Interaction(data["absorb_cat"], data["absorb_cont"])
    else:
        absorb = None
    interactions = None
    if interact:
        interactions = Interaction(data["interaction_cat"], data["interaction_cont"])
    return AbsorbingLS(
        data["dependent"],
        data["exog"],
        absorb=absorb,
        interactions=interactions,
        weights=data["weights"],
        **kwargs,
    )


def drop_manually(data, absorb_type, interact):
    """Drop the rows with missing values from the inputs that the model uses"""
    used = ["dependent", "exog", "weights"]
    if absorb_type != "none":
        used.append("absorb_cat")
    if absorb_type in ("mixed", "interaction"):
        used.append("absorb_cont")
    if interact:
        used.extend(["interaction_cat", "interaction_cont"])
    missing = np.zeros(NOBS, dtype=bool)
    for key in used:
        value = data[key]
        if value is not None:
            isnull = value.isnull()
            if isinstance(isnull, pd.DataFrame):
                isnull = isnull.any(axis=1)
            missing |= isnull.to_numpy()
    keep = ~missing
    cleaned = {key: None if v is None else v.loc[keep] for key, v in data.items()}
    return cleaned, keep


def assert_same_fit(res, expected):
    """Every result of res must be the result of the fit to the dropped data"""
    assert res.nobs == expected.nobs
    assert res.df_model == expected.df_model
    assert res.df_resid == expected.df_resid
    assert res.df_absorbed == expected.df_absorbed
    assert res.cov_type == expected.cov_type
    assert res.debiased == expected.debiased
    assert res.f_statistic.dist_name == expected.f_statistic.dist_name
    assert_index_equal(res.params.index, expected.params.index)
    assert_index_equal(res.resids.index, expected.resids.index)
    assert_index_equal(res.fitted_values.index, expected.fitted_values.index)
    scalars = [
        "rsquared",
        "rsquared_adj",
        "absorbed_rsquared",
        "resid_ss",
        "total_ss",
        "model_ss",
        "s2",
    ]
    for name in scalars:
        assert_allclose(getattr(res, name), getattr(expected, name), rtol=1e-9)
    assert_allclose(res.f_statistic.stat, expected.f_statistic.stat, rtol=1e-9)
    assert_allclose(res.f_statistic.pval, expected.f_statistic.pval, rtol=1e-9)
    series = ("params", "std_errors", "tstats", "pvalues", "cov")
    series += ("resids", "wresids", "fitted_values", "absorbed_effects")
    for name in series:
        assert_allclose(
            getattr(res, name).to_numpy(),
            getattr(expected, name).to_numpy(),
            rtol=1e-9,
            atol=1e-12,
            err_msg=name,
        )
    assert_allclose(res.conf_int().to_numpy(), expected.conf_int().to_numpy())


def configurations():
    """The input that has the missing values and how the model is specified"""
    configs = []
    for absorb_type in ("categorical", "mixed", "interaction", "none"):
        for interact in (False, True):
            if absorb_type == "none" and not interact:
                continue  # Nothing to absorb
            for weighted in (False, True):
                locations = ["dependent", "exog"]
                if absorb_type != "none":
                    locations.append("absorb_cat")
                if absorb_type in ("mixed", "interaction"):
                    locations.append("absorb_cont")
                if weighted:
                    locations.append("weights")
                if interact:
                    locations.extend(["interaction_cat", "interaction_cont"])
                configs.extend(
                    (absorb_type, interact, weighted, location)
                    for location in locations
                )
    return configs


CONFIGURATIONS = configurations()
CONFIGURATION_IDS = [
    f"{absorb_type}-{'interact' if interact else 'nointeract'}-"
    f"{'weighted' if weighted else 'unweighted'}-nan_in_{location}"
    for absorb_type, interact, weighted, location in CONFIGURATIONS
]


@pytest.mark.parametrize(
    ("absorb_type", "interact", "weighted", "location"),
    CONFIGURATIONS,
    ids=CONFIGURATION_IDS,
)
@pytest.mark.parametrize("debiased", [False, True])
def test_matches_manual_drop(absorb_type, interact, weighted, location, debiased):
    data = add_missing(make_data(weighted), location)
    cleaned, keep = drop_manually(data, absorb_type, interact)
    assert NOBS - 4 <= keep.sum() < NOBS

    with pytest.warns(MissingValueWarning):
        mod = make_model(data, absorb_type, interact)
    res = mod.fit(debiased=debiased)
    expected = make_model(cleaned, absorb_type, interact).fit(debiased=debiased)

    assert res.nobs == keep.sum()
    assert_same_fit(res, expected)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("absorb_type", ["categorical", "interaction"])
@pytest.mark.parametrize(
    ("cov_type", "cov_config"),
    [
        ("unadjusted", {}),
        ("robust", {}),
        ("kernel", {"bandwidth": 4}),
        ("clustered", {}),
    ],
)
@pytest.mark.parametrize("debiased", [False, True])
def test_covariance_matches_manual_drop(
    weighted, absorb_type, cov_type, cov_config, debiased
):
    # Missing values in different inputs, in different rows
    data = make_data(weighted)
    data = add_missing(data, "dependent", rows=[5])
    data = add_missing(data, "exog", rows=[70, 80, 150])
    data = add_missing(data, "absorb_cat", rows=[30])
    if weighted:
        data = add_missing(data, "weights", rows=[200])
    cleaned, keep = drop_manually(data, absorb_type, False)
    assert keep.sum() == NOBS - 5 - weighted
    if cov_type == "clustered":
        clusters = np.random.default_rng(1).integers(0, 12, NOBS)
        cov_config = {"clusters": clusters[keep]}

    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = make_model(data, absorb_type, False)
    res = mod.fit(cov_type=cov_type, debiased=debiased, **cov_config)
    expected = make_model(cleaned, absorb_type, False).fit(
        cov_type=cov_type, debiased=debiased, **cov_config
    )
    assert res.nobs == keep.sum()
    assert_same_fit(res, expected)


def test_single_missing_value():
    # The simplest model that failed: a single missing value in exog
    rng = np.random.default_rng(0)
    nobs = 120
    y = pd.Series(rng.standard_normal(nobs), name="y")
    x = pd.DataFrame({"x": rng.standard_normal(nobs)})
    x.iloc[3, 0] = np.nan
    cat = pd.DataFrame({"g": pd.Categorical(rng.integers(0, 6, nobs))})
    with pytest.warns(MissingValueWarning):
        mod = AbsorbingLS(y, x, absorb=cat)
    res = mod.fit()
    assert res.nobs == nobs - 1
    assert res.resids.shape == (nobs - 1,)
    assert res.df_resid == nobs - 1 - res.df_model
    assert isinstance(str(res.summary), str)

    weights = pd.Series(np.exp(rng.standard_normal(nobs)))
    with pytest.warns(MissingValueWarning, match="missing values"):
        res = AbsorbingLS(y, x, absorb=cat, weights=weights).fit()
    assert res.nobs == nobs - 1
    assert isinstance(str(res.summary), str)


@pytest.mark.parametrize("weighted", [False, True])
def test_ndarray_inputs(weighted):
    data = make_data(weighted, labelled=False)
    data = add_missing(data, "dependent", rows=[2, 50])
    data = add_missing(data, "exog", rows=[9, 77, 100, 101])
    cleaned, keep = drop_manually(data, "categorical", False)

    def arrays(inputs):
        weights = inputs["weights"]
        weights = None if weights is None else weights.to_numpy()
        return inputs["dependent"].to_numpy(), inputs["exog"].to_numpy(), weights

    y, x, w = arrays(data)
    with pytest.warns(MissingValueWarning, match="missing values"):
        res = AbsorbingLS(y, x, absorb=data["absorb_cat"], weights=w).fit()
    y, x, w = arrays(cleaned)
    expected = AbsorbingLS(y, x, absorb=cleaned["absorb_cat"], weights=w).fit()

    assert res.nobs == keep.sum()
    # The index of the results is the position of the retained observations
    assert list(res.resids.index) == list(np.flatnonzero(keep))
    for name in ("params", "std_errors", "cov", "resids", "wresids"):
        assert_allclose(
            getattr(res, name).to_numpy(),
            getattr(expected, name).to_numpy(),
            rtol=1e-9,
            atol=1e-12,
        )
    for name in ("rsquared", "resid_ss", "total_ss", "s2", "df_model", "df_resid"):
        assert_allclose(getattr(res, name), getattr(expected, name), rtol=1e-9)


@pytest.mark.parametrize("method", ["hdfe", "lsmr"])
def test_category_removed_by_missing_data(method):
    # The only observation in a category is dropped, which removes the category
    data = make_data(False)
    codes = np.asarray(data["absorb_cat"]["g0"].cat.codes).astype(int)
    codes[7] = 99
    data["absorb_cat"] = data["absorb_cat"].assign(g0=pd.Categorical(codes))
    data = add_missing(data, "exog", rows=[7])
    cleaned, keep = drop_manually(data, "categorical", False)
    assert keep.sum() == NOBS - 1

    with pytest.warns(MissingValueWarning, match="missing values"):
        res = make_model(data, "categorical", False).fit(method=method)
    expected = make_model(cleaned, "categorical", False).fit(method=method)
    assert_same_fit(res, expected)

    # Same as data that never had the category
    absorb = cleaned["absorb_cat"].copy()
    for col in absorb:
        absorb[col] = absorb[col].cat.remove_unused_categories()
    cleaned["absorb_cat"] = absorb
    expected = make_model(cleaned, "categorical", False).fit(method=method)
    assert res.df_model == expected.df_model
    assert_allclose(res.params, expected.params)
    assert_allclose(res.std_errors, expected.std_errors)


def test_drop_absorbed_with_missing():
    data = make_data(False)
    # A single effect, so that the dummy is exactly absorbed
    data["absorb_cat"] = data["absorb_cat"][["g1"]]
    dummy = pd.get_dummies(data["absorb_cat"]["g1"], prefix="d", dtype=float)
    data["exog"] = pd.concat([data["exog"], dummy.iloc[:, [0]]], axis=1)
    data = add_missing(data, "dependent", rows=[4, 40])
    cleaned, _ = drop_manually(data, "categorical", False)

    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = make_model(data, "categorical", False, drop_absorbed=True)
    expected = make_model(cleaned, "categorical", False, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning):
        res = mod.fit()
    with pytest.warns(AbsorbingEffectWarning):
        expected = expected.fit()
    assert list(res.params.index) == ["x0", "x1"]
    assert_same_fit(res, expected)


def test_independent_reference():
    # Dummy variables and weights in the standard linear model
    data = make_data(True)
    data = add_missing(data, "dependent", rows=[10, 99])
    data = add_missing(data, "exog", rows=[20, 120, 130])
    data = add_missing(data, "absorb_cat", rows=[50])
    data = add_missing(data, "weights", rows=[200, 201])
    cleaned, keep = drop_manually(data, "categorical", False)
    assert keep.sum() == NOBS - 8
    dummies = pd.concat(
        [
            pd.get_dummies(cleaned["absorb_cat"][col], prefix=col, drop_first=True)
            for col in cleaned["absorb_cat"]
        ],
        axis=1,
    ).astype(float)
    exog = pd.concat([cleaned["exog"], dummies], axis=1)
    exog.insert(0, "const", 1.0)
    benchmark = IV2SLS(
        cleaned["dependent"], exog, None, None, weights=cleaned["weights"]
    ).fit(cov_type="robust")

    with pytest.warns(MissingValueWarning, match="missing values"):
        res = make_model(data, "categorical", False).fit(cov_type="robust")
    cols = ["x0", "x1"]
    assert res.nobs == benchmark.nobs
    assert_allclose(res.params[cols], benchmark.params[cols], rtol=1e-8)
    assert_allclose(res.std_errors[cols], benchmark.std_errors[cols], rtol=1e-8)
    assert_allclose(res.resids.to_numpy(), benchmark.resids.to_numpy(), atol=1e-8)
    assert_allclose(res.rsquared, benchmark.rsquared, rtol=1e-8)


def test_all_observations_missing():
    data = make_data(False)
    data["dependent"] = pd.Series(np.nan, index=data["dependent"].index)
    with pytest.raises(ValueError, match="All observations contain missing data"):
        make_model(data, "categorical", False)


def test_missing_value_warning():
    data = add_missing(make_data(False), "exog", rows=[1])
    with pytest.warns(MissingValueWarning):
        make_model(data, "categorical", False)
    data = add_missing(make_data(True), "weights", rows=[1])
    with pytest.warns(MissingValueWarning):
        make_model(data, "none", True)


@pytest.mark.parametrize("weighted", [False, True])
def test_weights_have_unit_mean_after_drop(weighted):
    data = add_missing(make_data(weighted), "dependent", rows=[0, 1, 2, 3])
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = make_model(data, "categorical", False)
    weights = mod.weights.ndarray
    assert weights.shape == (NOBS - 4, 1)
    assert_allclose(weights.mean(), 1.0)
    if weighted:
        cleaned, _ = drop_manually(data, "categorical", False)
        expected = cleaned["weights"].to_numpy()[:, None]
        assert_allclose(weights, expected / expected.mean())


def test_clusters_have_estimation_sample_length():
    data = add_missing(make_data(False), "exog", rows=[8, 90])
    cleaned, keep = drop_manually(data, "categorical", False)
    clusters = np.random.default_rng(2).integers(0, 9, NOBS)
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = make_model(data, "categorical", False)

    res = mod.fit(cov_type="clustered", clusters=clusters[keep])
    expected = make_model(cleaned, "categorical", False).fit(
        cov_type="clustered", clusters=clusters[keep]
    )
    assert_allclose(res.cov, expected.cov)
    assert_allclose(res.std_errors, expected.std_errors)

    # The clusters of the dropped observations must be dropped too
    with pytest.raises(ValueError, match="clusters has the wrong nobs"):
        mod.fit(cov_type="clustered", clusters=clusters)


@pytest.mark.parametrize("as_absorb", [True, False])
@pytest.mark.parametrize("missing_in_instance", [True, False])
def test_interaction_instances_are_not_modified(as_absorb, missing_in_instance):
    # The instance belongs to the caller. A model that drops rows must use a copy.
    data = make_data(False)
    data = add_missing(data, "absorb_cont" if missing_in_instance else "exog", [8, 90])
    inter = Interaction(data["absorb_cat"], data["absorb_cont"])
    cat_before = inter.cat.copy()
    cont_before = inter.cont.copy()

    def fit():
        if as_absorb:
            with pytest.warns(MissingValueWarning, match="missing values"):
                mod = AbsorbingLS(data["dependent"], data["exog"], absorb=inter)
        else:
            with pytest.warns(MissingValueWarning, match="missing values"):
                mod = AbsorbingLS(
                    data["dependent"],
                    data["exog"],
                    absorb=data["absorb_cat"],
                    interactions=[inter],
                )
        return mod.fit()

    res = fit()
    assert res.nobs == NOBS - 2
    assert inter.nobs == NOBS
    assert_frame_equal(inter.cat, cat_before)
    assert_frame_equal(inter.cont, cont_before)

    # The instance can be used again
    assert_same_fit(fit(), res)
