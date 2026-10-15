"""
Missing values in the clusters of the panel models

The panel models replaced the clusters by the codes of their categories, which give a
missing value the code -1 and so a cluster of its own. PooledOLS, PanelOLS and
RandomEffects kept the observations with a missing cluster as the observations of one
more cluster, and the standard errors changed without an error. BetweenOLS and
FirstDifferenceOLS ignored some missing clusters, and raised errors that did not name
the missing values for the others. The IV and system models raise a ValueError, and the
panel models raise the same error for the observations that they use. The clusters of
the observations that are dropped, since a value of the data is missing, are not used
and can be missing.
"""

import functools

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared.exceptions import MissingValueWarning

NENTITY = 20
NPERIOD = 6
ENTITIES = [f"firm{i:02d}" for i in range(NENTITY)]
PERIODS = list(range(2001, 2001 + NPERIOD))
INDEX = pd.MultiIndex.from_product([ENTITIES, PERIODS], names=["entity", "year"])
# The rows that have a missing cluster. They are in the entities 0, 6 and 12, which have
# other rows with a cluster
MISSING_ROWS = [3, 40, 77]
MESSAGE = "clusters cannot contain missing values"


def make_inputs():
    rng = np.random.default_rng(789)
    nobs = NENTITY * NPERIOD
    entity = np.repeat(np.arange(NENTITY), NPERIOD)
    x = rng.standard_normal((nobs, 2))
    y = x @ [1.0, -0.5] + rng.standard_normal(NENTITY)[entity]
    y += rng.standard_normal(nobs)
    return (
        pd.Series(y, index=INDEX, name="y"),
        pd.DataFrame(x, index=INDEX, columns=["x0", "x1"]),
        # Groups of 4 entities, so that the clusters do not vary within an entity as
        # BetweenOLS and FirstDifferenceOLS need
        (entity // 4).astype(float),
    )


Y, X, CLUSTERS = make_inputs()
MISSING_CLUSTERS = CLUSTERS.copy()
MISSING_CLUSTERS[MISSING_ROWS] = np.nan


def grid(values):
    """The time periods are the rows and the entities are the columns"""
    return values.reshape(NENTITY, NPERIOD).T


def text(values):
    labels = [None if np.isnan(v) else f"group{int(v)}" for v in values]
    return pd.DataFrame({"c": labels}, index=INDEX)


def clustered_standard_errors(y, x, clusters):
    """Standard errors of OLS with the scores summed by cluster, from the definition"""
    # The covariance is (x'x)^-1 sum_g(x_g'e_g e_g'x_g) (x'x)^-1 n / (n - k)
    xpxi = np.linalg.inv(x.T @ x)
    resid = y - x @ (xpxi @ x.T @ y)
    scores = x * resid[:, None]
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in np.unique(clusters):
        total = scores[clusters == group].sum(0)
        meat += np.outer(total, total)
    nobs, nvar = x.shape
    return np.sqrt(np.diag(xpxi @ meat @ xpxi * nobs / (nobs - nvar)))


MODELS = {
    "PooledOLS": PooledOLS,
    "PanelOLS": functools.partial(PanelOLS, entity_effects=True),
    "RandomEffects": RandomEffects,
    "BetweenOLS": BetweenOLS,
    "FirstDifferenceOLS": FirstDifferenceOLS,
}
# The inputs with a value for each row, for each time period and entity (a grid), and
# the text and categories that are replaced by codes
FORMS = {
    "frame": lambda v: pd.DataFrame({"c": v}, index=INDEX),
    "series": lambda v: pd.Series(v, index=INDEX, name="c"),
    "array": lambda v: v.copy(),
    "grid-array": grid,
    "grid-3d-array": lambda v: grid(v)[None],
    "grid-frame": lambda v: pd.DataFrame(grid(v), index=PERIODS, columns=ENTITIES),
    "text": text,
    "categorical": lambda v: text(v).astype("category"),
}
# The standard errors of the models with the clusters CLUSTERS, from the version before
# a missing cluster raised. Those of PooledOLS are also computed from the definition
EXPECTED_SE = {
    "PooledOLS": [0.0706095231, 0.1947655616],
    "PanelOLS": [0.0875362626, 0.1321212053],
    "RandomEffects": [0.0819130482, 0.1418174683],
    "BetweenOLS": [0.3439555812, 0.6240345783],
    "FirstDifferenceOLS": [0.1138596065, 0.0871024591],
}


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("model", MODELS)
def test_missing_cluster_raises(model, form):
    mod = MODELS[model](Y, X)
    clusters = FORMS[form](MISSING_CLUSTERS)
    with pytest.raises(ValueError, match=MESSAGE):
        mod.fit(cov_type="clustered", clusters=clusters)


@pytest.mark.parametrize("model", MODELS)
def test_missing_cluster_in_one_of_two_columns_raises(model):
    mod = MODELS[model](Y, X)
    clusters = pd.DataFrame({"a": CLUSTERS, "b": MISSING_CLUSTERS}, index=INDEX)
    with pytest.raises(ValueError, match=MESSAGE):
        mod.fit(cov_type="clustered", clusters=clusters)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("model", MODELS)
def test_clusters_without_missing_values_are_unchanged(model, form):
    mod = MODELS[model](Y, X)
    res = mod.fit(cov_type="clustered", clusters=FORMS[form](CLUSTERS))
    assert_allclose(res.std_errors, EXPECTED_SE[model], rtol=1e-6)


@pytest.mark.parametrize("form", FORMS)
def test_pooled_ols_clusters_without_missing_values_are_the_definition(form):
    expected = clustered_standard_errors(Y.to_numpy(), X.to_numpy(), CLUSTERS)
    res = PooledOLS(Y, X).fit(cov_type="clustered", clusters=FORMS[form](CLUSTERS))
    assert_allclose(res.std_errors, expected)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("model", MODELS)
def test_missing_cluster_of_a_dropped_row_is_not_used(model, form):
    # The model drops the rows in which the dependent variable is missing, together
    # with their clusters, which can be missing. The results do not depend on them
    y = Y.copy()
    y.iloc[MISSING_ROWS] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = MODELS[model](y, X)
    is_kept = mod.not_null[MISSING_ROWS].any()
    assert not is_kept
    expected = mod.fit(cov_type="clustered", clusters=FORMS[form](CLUSTERS))
    res = mod.fit(cov_type="clustered", clusters=FORMS[form](MISSING_CLUSTERS))
    assert res.nobs == expected.nobs
    assert_allclose(res.params, expected.params)
    assert_allclose(res.std_errors, expected.std_errors)
