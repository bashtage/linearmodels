"""
The degrees of freedom of the clustered covariance of PanelOLS with one effect

The covariance does not count the degrees of freedom of an effect that is nested in
the clusters, since an effect with the same groups as a cluster (e.g., entity effects
and clustering by entity) is purged from the scores of the cluster. With several
cluster variables, e.g., two-way clustering, the effect has to be nested in all of them,
as in AbsorbingLS. It was nested if it was in the last cluster variable only, so that
the standard errors changed with the order of the cluster variables.

The reference is the covariance of the regression on the data from which the effect is
removed, written out in NumPy.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import PanelOLS


def make_data(unbalanced, seed=3, nentity=30, nperiod=8):
    """A panel with entities that are nested in industries"""
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["id", "year"]
    )
    entity, period = index.codes
    x = rg.standard_normal((index.shape[0], 2))
    effect = rg.standard_normal(nentity)[entity]
    shock = (1 + np.abs(x[:, 0])) * rg.standard_normal(index.shape[0])
    data = pd.DataFrame(
        {
            "y": x @ [0.5, -0.3] + effect + shock,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "entity": entity,
            "time": period,
            "industry": entity // 5,
            "group": (entity + period) % 4,
        },
        index=index,
    )
    if unbalanced:
        keep = rg.random((nentity, nperiod)) > 0.3
        for i in range(nentity):
            keep[i, rg.choice(nperiod, 3, replace=False)] = True
        data = data.loc[keep.ravel()]
    return data


@pytest.fixture(params=[False, True], ids=["balanced", "unbalanced"], scope="module")
def data(request):
    return make_data(request.param)


EFFECTS = {
    "entity": {"entity_effects": True},
    "time": {"time_effects": True},
    "group": {"other_effects": ("group",)},
}
PAIRS = [
    ("entity", "time"),
    ("industry", "time"),
    ("industry", "entity"),
    ("group", "time"),
    ("entity", "group"),
]


def build(data, effect):
    options = dict(EFFECTS[effect])
    if "other_effects" in options:
        options["other_effects"] = data[list(options["other_effects"])]
    return PanelOLS(data["y"], data[["x1", "x2"]], **options)


def sum_of_scores(scores, groups):
    total = 0.0
    for key in np.unique(groups):
        score = scores[groups == key].sum(0)
        total = total + np.outer(score, score)
    return total


def clustered_covariance(data, effect, clusters, count_effects):
    """Debiased covariance of the regression on the data without the effect"""
    groups = data[effect].to_numpy()
    x = data[["x1", "x2"]] - data[["x1", "x2"]].groupby(groups).transform("mean")
    y = data["y"] - data["y"].groupby(groups).transform("mean")
    x, y = x.to_numpy(), y.to_numpy()
    resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    scores = x * resid[:, None]
    labels = [data[c].to_numpy() for c in clusters]
    meat = sum_of_scores(scores, labels[0])
    if len(labels) == 2:
        pairs = pd.factorize(pd.Series(list(zip(*labels, strict=True))))[0]
        meat = meat + sum_of_scores(scores, labels[1]) - sum_of_scores(scores, pairs)
    nobs, nvar = x.shape
    neffects = np.unique(groups).shape[0] if count_effects else 0
    bread = np.linalg.inv(x.T @ x)
    return bread @ meat @ bread * nobs / (nobs - neffects - nvar)


def is_nested(data, effect, clusters):
    """The groups of the effect are in a single group of every cluster variable"""
    by_effect = data.groupby(effect)
    return all(by_effect[c].nunique().max() == 1 for c in clusters)


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("effect", list(EFFECTS))
def test_clustered_covariance_against_hand_computation(data, effect, pair):
    mod = build(data, effect)
    expected = {}
    for clusters in (list(pair), list(reversed(pair))):
        res = mod.fit(cov_type="clustered", clusters=data[clusters])
        expected = clustered_covariance(
            data, effect, clusters, count_effects=not is_nested(data, effect, clusters)
        )
        assert_allclose(res.cov.to_numpy(), expected, rtol=1e-8)


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("effect", list(EFFECTS))
def test_order_of_the_clusters_does_not_matter(data, effect, pair):
    mod = build(data, effect)
    forward = mod.fit(cov_type="clustered", clusters=data[list(pair)])
    backward = mod.fit(cov_type="clustered", clusters=data[list(reversed(pair))])
    assert_allclose(forward.cov, backward.cov)


@pytest.mark.parametrize("effect", ["entity", "time"])
def test_cluster_entity_and_cluster_time(data, effect):
    # An effect that is nested in the clusters by entity or by time only is counted
    mod = build(data, effect)
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    expected = clustered_covariance(
        data, effect, ["entity", "time"], count_effects=True
    )
    assert_allclose(res.cov.to_numpy(), expected, rtol=1e-8)
    one_way = mod.fit(
        cov_type="clustered",
        cluster_entity=effect == "entity",
        cluster_time=effect == "time",
    )
    expected = clustered_covariance(data, effect, [effect], count_effects=False)
    assert_allclose(one_way.cov.to_numpy(), expected, rtol=1e-8)


def test_nested_in_all_clusters_does_not_count_the_effects(data):
    # Entities are nested in industries: both cluster variables nest the entity effects
    mod = build(data, "entity")
    res = mod.fit(cov_type="clustered", clusters=data[["industry", "entity"]])
    counted = mod.fit(
        cov_type="clustered",
        clusters=data[["industry", "entity"]],
        auto_df=False,
        count_effects=True,
    )
    not_counted = mod.fit(
        cov_type="clustered",
        clusters=data[["industry", "entity"]],
        auto_df=False,
        count_effects=False,
    )
    assert_allclose(res.cov, not_counted.cov)
    assert not np.allclose(res.cov, counted.cov)
