"""
The two-way clustered covariance of the IV models does not depend on the cluster labels.

The intersection of the two cluster variables was computed by an arithmetic encoding
of the labels that was then truncated to int64. Labels that are not evenly spaced
integers, for example 0.0, 0.1, 0.2, ..., merged groups of the intersection, so that
the covariance was silently wrong, and string labels raised. The intersection is now
the same as in the panel and system models.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS, AbsorbingLS


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    nobs = 400
    entity = rng.integers(0, 20, nobs)
    time = rng.integers(0, 15, nobs)
    x = rng.standard_normal(nobs)
    y = 1 + 0.5 * x + rng.standard_normal(nobs)
    exog = pd.DataFrame({"const": 1.0, "x": x})
    return y, exog, entity, time


def two_way_by_hand(y, exog, entity, time):
    """(X'X)^-1 (S_entity + S_time - S_entity,time) (X'X)^-1"""
    x = np.asarray(exog)
    xpxi = np.linalg.inv(x.T @ x)
    eps = y - x @ (xpxi @ x.T @ y)
    scores = pd.DataFrame(x * eps[:, None])

    def meat(*labels):
        sums = np.asarray(scores.groupby(list(labels)).sum())
        return sums.T @ sums

    s = meat(entity) + meat(time) - meat(entity, time)
    return xpxi @ s @ xpxi


LABELS = {
    "scaled-float": lambda e, t: np.column_stack([0.1 * e, t]).astype(float),
    "string": lambda e, t: np.column_stack([e.astype(str), t.astype(str)]),
    "mixed-frame": lambda e, t: pd.DataFrame({"e": [f"f{v}" for v in e], "t": t}),
}


@pytest.mark.parametrize("labels", list(LABELS))
def test_iv_two_way_clusters_do_not_depend_on_labels(data, labels):
    y, exog, entity, time = data
    clusters = LABELS[labels](entity, time)
    res = IV2SLS(y, exog, None, None).fit(cov_type="clustered", clusters=clusters)
    expected = IV2SLS(y, exog, None, None).fit(
        cov_type="clustered", clusters=np.column_stack([entity, time])
    )
    assert_allclose(res.cov, expected.cov)
    assert_allclose(res.cov, two_way_by_hand(y, exog, entity, time))


def test_absorbing_two_way_clusters_do_not_depend_on_labels(data):
    y, exog, entity, time = data
    absorb = pd.DataFrame({"g": pd.Categorical(entity % 4)})
    mod = AbsorbingLS(y, exog, absorb=absorb)
    expected = mod.fit(cov_type="clustered", clusters=np.column_stack([entity, time]))
    res = mod.fit(cov_type="clustered", clusters=np.column_stack([0.1 * entity, time]))
    assert_allclose(res.cov, expected.cov)
