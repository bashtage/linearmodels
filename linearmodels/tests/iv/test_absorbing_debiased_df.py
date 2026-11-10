"""
The small-sample adjustment of AbsorbingLS counts the degrees of freedom of the absorbed terms.

With debiased=True the covariance was scaled by n / (n - k), where k counts the
regressors of exog, and not the dummy variables of the effects. The standard errors were
too small by sqrt((n - k_total) / (n - k)), where k_total counts all of the terms,
compared with the regression on the regressors and the dummy variables of the effects
and with PanelOLS(debiased=True), although df_resid, which the t and F tests use,
counted the absorbed terms.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels import PanelOLS
from linearmodels.iv import IV2SLS
from linearmodels.iv.absorbing import AbsorbingLS, Interaction

NOBS = 240
LEVELS = (14, 6, 4)

# pyhdfe 0.2.0 calls the sparse matrix classes of SciPy, which SciPy 2.0 deprecates.
# pyhdfe 0.2.1 does not
pytestmark = pytest.mark.filterwarnings(
    "ignore:(coo|csc|csr)_matrix is being replaced by:DeprecationWarning:pyhdfe"
)


def make_data(neffects, seed=0, weighted=False):
    rng = np.random.default_rng(seed)
    codes = [rng.integers(0, level, NOBS) for level in LEVELS[:neffects]]
    cats = pd.DataFrame({f"g{i}": pd.Categorical(c) for i, c in enumerate(codes)})
    effects = sum(
        rng.standard_normal(level)[c] for level, c in zip(LEVELS, codes, strict=False)
    )
    x = pd.DataFrame(
        rng.standard_normal((NOBS, 2)) + 0.5 * effects[:, None], columns=["x0", "x1"]
    )
    y = pd.Series(
        x.to_numpy() @ [1.0, -0.5] + effects + rng.standard_normal(NOBS), name="y"
    )
    weights = None
    if weighted:
        weights = pd.Series(np.exp(0.5 * rng.standard_normal(NOBS)), name="w")
    clusters = rng.integers(0, 17, NOBS)
    return y, x, cats, weights, clusters


def dummies(cats, drop_first_all=False):
    """Dummy variables of the effects: the first effect has all of its categories"""
    return np.column_stack(
        [
            pd.get_dummies(
                cats[col], drop_first=drop_first_all or i > 0, dtype=float
            ).to_numpy()
            for i, col in enumerate(cats)
        ]
    )


COV_CONFIGS = [
    ("unadjusted", {}),
    ("robust", {}),
    ("kernel", {"kernel": "bartlett", "bandwidth": 4}),
    ("clustered", {"clusters": "clusters"}),
]
COV_IDS = ["unadjusted", "robust", "kernel", "clustered"]


def config(cov_config, clusters):
    return {
        key: (clusters if value == "clusters" else value)
        for key, value in cov_config.items()
    }


def assert_matches_dummy_regression(
    res, y, x, controls, weights, cov_type, cov_config, extra=None
):
    """res is the fit of the regression on x and the controls, for the x parameters"""
    exog = np.column_stack([x.to_numpy(), controls])
    k = x.shape[1]
    reference = IV2SLS(y, exog, None, None, weights=weights).fit(
        cov_type=cov_type, debiased=True, **cov_config
    )
    assert_allclose(res.params, reference.params.iloc[:k], rtol=1e-6)
    assert_allclose(res.cov, reference.cov.iloc[:k, :k], rtol=1e-6)
    assert_allclose(res.std_errors, reference.std_errors.iloc[:k], rtol=1e-6)
    assert res.df_resid == reference.df_resid
    assert_allclose(res.s2, reference.s2, rtol=1e-6)
    assert_allclose(res.pvalues, reference.pvalues.iloc[:k], rtol=1e-5)
    assert_allclose(res.conf_int(), reference.conf_int().iloc[:k], rtol=1e-6)


@pytest.mark.parametrize(("cov_type", "cov_config"), COV_CONFIGS, ids=COV_IDS)
@pytest.mark.parametrize("neffects", [1, 2, 3])
@pytest.mark.parametrize(
    ("method", "weighted"), [("hdfe", False), ("lsmr", False), ("lsmr", True)]
)
def test_equals_the_dummy_variable_regression(
    cov_type, cov_config, neffects, method, weighted
):
    y, x, cats, weights, clusters = make_data(neffects, weighted=weighted)
    cov_config = config(cov_config, clusters)
    # No cluster is nested in a category, and the effects are counted
    res = AbsorbingLS(y, x, absorb=cats, weights=weights).fit(
        method=method, cov_type=cov_type, debiased=True, **cov_config
    )
    assert_matches_dummy_regression(
        res, y, x, dummies(cats), weights, cov_type, cov_config
    )


@pytest.mark.parametrize(("cov_type", "cov_config"), COV_CONFIGS, ids=COV_IDS)
@pytest.mark.parametrize("neffects", [1, 2])
def test_constant_in_exog(cov_type, cov_config, neffects):
    # The constant takes one of the degrees of freedom of the dummy variables
    y, x, cats, _, clusters = make_data(neffects, seed=1)
    cov_config = config(cov_config, clusters)
    exog = x.assign(const=1.0)
    res = AbsorbingLS(y, exog, absorb=cats).fit(
        cov_type=cov_type, debiased=True, **cov_config
    )
    assert res.df_absorbed == sum(cats[c].nunique() for c in cats) - neffects
    reference = IV2SLS(y, np.column_stack([exog, dummies(cats, True)]), None, None)
    reference_res = reference.fit(cov_type=cov_type, debiased=True, **cov_config)
    assert res.df_resid == reference_res.df_resid
    # The constant is the mean level in AbsorbingLS, and the level of a category in the
    # regression, so only the slopes are comparable
    assert_allclose(res.params.iloc[:2], reference_res.params.iloc[:2], rtol=1e-6)
    assert_allclose(res.cov.iloc[:2, :2], reference_res.cov.iloc[:2, :2], rtol=1e-6)
    assert_allclose(res.s2, reference_res.s2, rtol=1e-6)


@pytest.mark.parametrize(("cov_type", "cov_config"), COV_CONFIGS, ids=COV_IDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_continuous_variables_and_interactions(cov_type, cov_config, weighted):
    y, x, cats, weights, clusters = make_data(2, seed=2, weighted=weighted)
    cov_config = config(cov_config, clusters)
    rng = np.random.default_rng(3)
    z = pd.Series(rng.standard_normal(NOBS), name="z")
    ic = pd.Series(pd.Categorical(rng.integers(0, 4, NOBS)), name="ic")
    ix = pd.Series(rng.standard_normal(NOBS), name="ix")
    y = y + 0.7 * z + 0.4 * ix * ic.cat.codes
    res = AbsorbingLS(
        y,
        x,
        absorb=pd.concat([cats, z], axis=1),
        interactions=Interaction(ic.to_frame(), ix.to_frame()),
        weights=weights,
    ).fit(cov_type=cov_type, debiased=True, **cov_config)
    slopes = np.column_stack(
        [ix.to_numpy() * (ic == k).astype(float).to_numpy() for k in range(4)]
    )
    controls = np.column_stack([dummies(cats), z.to_numpy(), slopes])
    # Dummy variables, a continuous variable and 4 slopes
    assert res.df_absorbed == controls.shape[1]
    assert_matches_dummy_regression(res, y, x, controls, weights, cov_type, cov_config)


def test_closed_form():
    # The unadjusted covariance is s2 (X'X)^-1 with s2 = e'e / (n - k_total)
    y, x, cats, _, _ = make_data(2, seed=4)
    res = AbsorbingLS(y, x, absorb=cats).fit(cov_type="unadjusted", debiased=True)
    z = np.column_stack([x.to_numpy(), dummies(cats)])
    beta = np.linalg.lstsq(z, y.to_numpy(), rcond=None)[0]
    resid = y.to_numpy() - z @ beta
    k_total = z.shape[1]
    s2 = resid @ resid / (NOBS - k_total)
    expected = s2 * np.linalg.inv(z.T @ z)[:2, :2]
    assert_allclose(res.cov, expected, rtol=1e-6)
    assert_allclose(res.s2, s2, rtol=1e-6)
    assert res.df_resid == NOBS - k_total
    # The adjustment applied is n / (n - k_total), and it was n / (n - 2)
    plain = AbsorbingLS(y, x, absorb=cats).fit(cov_type="unadjusted")
    assert_allclose(res.cov / plain.cov, NOBS / (NOBS - k_total), rtol=1e-8)


@pytest.mark.parametrize(("cov_type", "cov_config"), COV_CONFIGS, ids=COV_IDS)
def test_not_debiased_is_not_adjusted(cov_type, cov_config):
    # Nothing is adjusted if debiased is False, unlike PanelOLS
    y, x, cats, _, clusters = make_data(2, seed=5)
    cov_config = config(cov_config, clusters)
    res = AbsorbingLS(y, x, absorb=cats).fit(cov_type=cov_type, **cov_config)
    reference = IV2SLS(y, np.column_stack([x.to_numpy(), dummies(cats)]), None, None)
    expected = reference.fit(cov_type=cov_type, **cov_config)
    assert_allclose(res.cov, expected.cov.iloc[:2, :2], rtol=1e-6)
    assert not res.debiased


def test_fit_again_with_other_debiased():
    y, x, cats, _, _ = make_data(2, seed=6)
    mod = AbsorbingLS(y, x, absorb=cats)
    first = mod.fit(cov_type="robust", debiased=True)
    plain = mod.fit(cov_type="robust")
    again = mod.fit(cov_type="robust", debiased=True)
    assert_allclose(first.cov, again.cov, rtol=1e-12)
    k_total = 2 + first.df_absorbed
    assert_allclose(first.cov, plain.cov * NOBS / (NOBS - k_total), rtol=1e-10)


def panel_data(seed=0, nentity=25, ntime=12):
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(ntime)], names=["entity", "time"]
    )
    entity, time = index.codes
    x = pd.DataFrame(
        rng.standard_normal((len(index), 2)), index=index, columns=["a", "b"]
    )
    y = pd.Series(
        x.sum(axis=1).to_numpy()
        + rng.standard_normal(nentity)[entity]
        + rng.standard_normal(ntime)[time]
        + rng.standard_normal(len(index)),
        index=index,
        name="y",
    )
    cats = pd.DataFrame(
        {"entity": pd.Categorical(entity), "time": pd.Categorical(time)}, index=index
    )
    return y, x, cats, entity, time


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust"])
@pytest.mark.parametrize("effects", [("entity",), ("time",), ("entity", "time")])
def test_equals_panel_ols(cov_type, effects):
    y, x, cats, _, _ = panel_data()
    panel = PanelOLS(
        y, x, entity_effects="entity" in effects, time_effects="time" in effects
    ).fit(cov_type=cov_type, debiased=True)
    res = AbsorbingLS(y, x, absorb=cats[list(effects)]).fit(
        cov_type=cov_type, debiased=True
    )
    assert_allclose(res.std_errors, panel.std_errors, rtol=1e-8)
    assert_allclose(res.cov, panel.cov, rtol=1e-8)
    assert res.df_resid == panel.df_resid


CLUSTERS = [("entity", {"cluster_entity": True}), ("time", {"cluster_time": True})]


@pytest.mark.parametrize(("cluster_name", "panel_config"), CLUSTERS)
@pytest.mark.parametrize("effects", [("entity",), ("time",), ("entity", "time")])
def test_clustered_equals_panel_ols(cluster_name, panel_config, effects):
    # PanelOLS does not count an effect that is nested in the clusters. The entity
    # effects are nested in entity clusters. The covariances are the same, using the
    # adjustment for the number of clusters of AbsorbingLS in PanelOLS
    y, x, cats, entity, time = panel_data()
    clusters = entity if cluster_name == "entity" else time
    panel = PanelOLS(
        y, x, entity_effects="entity" in effects, time_effects="time" in effects
    ).fit(cov_type="clustered", debiased=True, group_debias=True, **panel_config)
    res = AbsorbingLS(y, x, absorb=cats[list(effects)]).fit(
        cov_type="clustered", debiased=True, clusters=clusters
    )
    assert_allclose(res.std_errors, panel.std_errors, rtol=1e-8)
    assert_allclose(res.cov, panel.cov, rtol=1e-8)


def adjustment(res, plain):
    """The ratio of the variances of the debiased and the not debiased covariance"""
    return float(res.cov.iloc[0, 0] / plain.cov.iloc[0, 0])


def test_nested_effect_is_not_counted():
    y, x, cats, entity, time = panel_data()
    n, g = len(y), 25
    absorb = cats[["entity"]]
    mod = AbsorbingLS(y, x, absorb=absorb)
    plain = mod.fit(cov_type="clustered", clusters=entity)
    # Nested in the clusters: only the 2 regressors count
    nested = mod.fit(cov_type="clustered", clusters=entity, debiased=True)
    assert_allclose(
        adjustment(nested, plain), (n - 1) / (n - 2) * g / (g - 1), rtol=1e-10
    )
    # Not nested in time clusters: the 25 dummy variables count
    other = mod.fit(cov_type="clustered", clusters=time, debiased=True)
    plain_time = mod.fit(cov_type="clustered", clusters=time)
    assert_allclose(
        adjustment(other, plain_time), (n - 1) / (n - 2 - 25) * 12 / 11, rtol=1e-10
    )
    # df_resid counts the effects whatever the covariance
    assert nested.df_resid == other.df_resid == n - 2 - 25
    # Without clusters, each observation is a cluster, and the effects count
    each = mod.fit(cov_type="clustered", debiased=True)
    plain_each = mod.fit(cov_type="clustered")
    assert_allclose(
        adjustment(each, plain_each), (n - 1) / (n - 2 - 25) * n / (n - 1), rtol=1e-10
    )


def test_nested_effect_with_a_constant():
    y, x, cats, entity, _ = panel_data()
    n, g = len(y), 25
    exog = x.assign(const=1.0)
    mod = AbsorbingLS(y, exog, absorb=cats[["entity"]])
    plain = mod.fit(cov_type="clustered", clusters=entity)
    nested = mod.fit(cov_type="clustered", clusters=entity, debiased=True)
    # The 3 columns count, and the effects, which the constant overlaps, do not
    assert_allclose(
        adjustment(nested, plain), (n - 1) / (n - 3) * g / (g - 1), rtol=1e-10
    )
    panel = PanelOLS(y, exog, entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True, debiased=True, group_debias=True
    )
    assert_allclose(nested.std_errors, panel.std_errors, rtol=1e-8)


def test_the_effects_are_counted_when_there_are_more_than_one():
    # PanelOLS counts the effects of 2-way models, even if one is nested
    y, x, cats, entity, _ = panel_data()
    n, g = len(y), 25
    mod = AbsorbingLS(y, x, absorb=cats)
    plain = mod.fit(cov_type="clustered", clusters=entity)
    res = mod.fit(cov_type="clustered", clusters=entity, debiased=True)
    assert_allclose(
        adjustment(res, plain), (n - 1) / (n - 2 - 25 - 11) * g / (g - 1), rtol=1e-10
    )


def test_continuous_variable_is_counted_with_a_nested_effect():
    y, x, cats, entity, _ = panel_data()
    n, g = len(y), 25
    z = pd.DataFrame({"z": np.arange(len(y), dtype=float) % 7}, index=y.index)
    absorb = pd.concat([cats[["entity"]], z], axis=1)
    mod = AbsorbingLS(y, x, absorb=absorb)
    plain = mod.fit(cov_type="clustered", clusters=entity)
    res = mod.fit(cov_type="clustered", clusters=entity, debiased=True)
    # The effects are nested and not counted. The continuous variable is
    assert_allclose(adjustment(res, plain), (n - 1) / (n - 3) * g / (g - 1), rtol=1e-10)


def test_the_effect_is_nested_in_every_cluster_variable():
    # s2 is scaled by n / (n - k - d), so its ratio shows the degrees of freedom that
    # were counted, whatever the number of cluster variables
    y, x, cats, entity, time = panel_data()
    n = len(y)
    mod = AbsorbingLS(y, x, absorb=cats[["entity"]])
    for clusters, extra in (
        (entity, 0),
        (time, 25),
        (np.column_stack([entity, time]), 25),  # Not nested in time
        (np.column_stack([time, entity]), 25),
        (np.column_stack([entity, entity // 5]), 0),  # Nested in both
        (np.column_stack([entity // 5, entity]), 0),
    ):
        plain = mod.fit(cov_type="clustered", clusters=clusters)
        res = mod.fit(cov_type="clustered", clusters=clusters, debiased=True)
        assert_allclose(res.s2 / plain.s2, n / (n - 2 - extra), rtol=1e-10)


def test_clusters_with_the_wrong_number_of_rows_raise_the_usual_error():
    y, x, cats, entity, _ = panel_data()
    mod = AbsorbingLS(y, x, absorb=cats[["entity"]])
    with pytest.raises(ValueError, match="clusters has the wrong nobs"):
        mod.fit(cov_type="clustered", clusters=entity[:-3], debiased=True)


def test_degrees_of_freedom_are_the_same_as_the_results_report():
    y, x, cats, _, _ = make_data(3, seed=7)
    mod = AbsorbingLS(y, x, absorb=cats)
    res = mod.fit(cov_type="robust", debiased=True)
    k_total = 2 + res.df_absorbed
    assert res.df_model == k_total
    assert res.df_resid == NOBS - k_total
    # The same scale as the one of the unadjusted covariance
    plain = mod.fit(cov_type="robust")
    assert_allclose(adjustment(res, plain), NOBS / (NOBS - k_total), rtol=1e-10)


def test_saturated_model_cannot_be_debiased():
    # 2 regressors and the 12 dummy variables of the effect use the 14 observations, so
    # there is no degree of freedom left and the small-sample adjustment is not defined
    rng = np.random.default_rng(8)
    n = 14
    cats = pd.DataFrame({"g": pd.Categorical(np.arange(n) % 12)})
    x = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"])
    y = pd.Series(rng.standard_normal(n), name="y")
    mod = AbsorbingLS(y, x, absorb=cats)
    with pytest.raises(ValueError, match="needs more observations"):
        mod.fit(debiased=True)
