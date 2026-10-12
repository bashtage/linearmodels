"""
The weights of the panel models must be finite and strictly positive

The weights are scaled to have a mean of one. Zero and negative weights were checked, but
an infinite weight was scaled to NaN and the other weights to zero, and the models failed
with errors from NumPy, SciPy and pandas that do not say what is wrong ("array must not
contain infs or NaNs", "SVD did not converge", an AbsorbingEffectError, and "operands could not
be broadcast together" for FirstDifferenceOLS). Missing weights are dropped with their
observations, as the other missing values are, and have to give the model of the data
without them.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared.exceptions import MissingValueWarning

MODELS = {
    "pooled": lambda y, x, w: PooledOLS(y, x, weights=w),
    "panel-entity-effects": lambda y, x, w: PanelOLS(
        y, x, weights=w, entity_effects=True
    ),
    "panel-time-effects": lambda y, x, w: PanelOLS(y, x, weights=w, time_effects=True),
    "between": lambda y, x, w: BetweenOLS(y, x, weights=w),
    "first-difference": lambda y, x, w: FirstDifferenceOLS(y, x, weights=w),
    "random-effects": lambda y, x, w: RandomEffects(y, x, weights=w),
    "fama-macbeth": lambda y, x, w: FamaMacBeth(y, x, weights=w),
}


def make_panel(seed=0):
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(30), np.arange(6)], names=["entity", "time"]
    )
    x = pd.DataFrame(
        rg.standard_normal((index.shape[0], 2)), index=index, columns=["a", "b"]
    )
    y = pd.Series(
        x @ [0.5, -0.3] + rg.standard_normal(index.shape[0]), index=index, name="y"
    )
    w = pd.Series(rg.chisquare(4, index.shape[0]) / 4 + 0.2, index=index, name="w")
    return y, x, w


@pytest.mark.parametrize("bad", [np.inf, -np.inf, 0.0, -1.0])
@pytest.mark.parametrize("model", list(MODELS))
def test_weights_must_be_finite_and_strictly_positive(model, bad):
    y, x, w = make_panel()
    w.iloc[7] = bad
    with pytest.raises(
        ValueError, match="weights must be strictly positive and finite"
    ):
        MODELS[model](y, x, w)


@pytest.mark.parametrize("model", list(MODELS))
def test_missing_weights_are_dropped_with_their_observations(model):
    y, x, w = make_panel()
    missing = [3, 17, 18, 100]
    w_missing = w.copy()
    w_missing.iloc[missing] = np.nan
    with pytest.warns(MissingValueWarning):
        res = MODELS[model](y, x, w_missing).fit()
    keep = np.ones(y.shape[0], dtype=bool)
    keep[missing] = False
    expected = MODELS[model](y[keep], x[keep], w[keep]).fit()
    assert res.nobs == expected.nobs
    assert_allclose(res.params, expected.params)
    assert_allclose(res.std_errors, expected.std_errors)
