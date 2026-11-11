"""Positionally paired inputs use the entity and time groups of dependent."""

from contextlib import nullcontext

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared.exceptions import MissingValueWarning

MODELS = [
    PooledOLS,
    PanelOLS,
    BetweenOLS,
    FirstDifferenceOLS,
    RandomEffects,
    FamaMacBeth,
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("labelled_dependent", [False, True])
def test_positional_inputs_use_dependent_groups(
    model, order, missing, labelled_dependent
):
    rng = np.random.default_rng(918)
    index = pd.MultiIndex.from_product([list("abcdefgh"), range(5)])
    x = pd.DataFrame(rng.standard_normal((40, 2)), index=index, columns=["x1", "x2"])
    y = pd.Series(x.to_numpy() @ [1.5, -0.3] + rng.standard_normal(40), index=index)
    weights = pd.Series(0.5 + rng.random(40), index=index)
    positions = np.arange(40)
    if order == "descending":
        positions = positions.reshape(8, 5)[:, ::-1].ravel()
    elif order == "shuffled":
        positions = rng.permutation(40)
    y, x, weights = y.iloc[positions], x.iloc[positions], weights.iloc[positions]
    if missing:
        y.iloc[7] = np.nan
    raw_y = y.to_numpy().reshape(8, 5).T
    raw_x = x.to_numpy().reshape(8, 5, 2).transpose(2, 1, 0)
    kwargs = {"entity_effects": True} if model is PanelOLS else {}
    if labelled_dependent:
        dependent, exog, reference_exog = y, raw_x, x
        reference_weights = weights
    else:
        dependent, exog, reference_exog = raw_y, x, raw_x
        reference_weights = weights.to_numpy()
    with pytest.warns(MissingValueWarning) if missing else nullcontext():
        expected = model(
            dependent, reference_exog, weights=reference_weights, **kwargs
        ).fit()
    with pytest.warns(MissingValueWarning) if missing else nullcontext():
        actual = model(dependent, exog, weights=weights, **kwargs).fit()
    assert_allclose(actual.params, expected.params, rtol=1e-10, atol=1e-12)
    assert_allclose(actual.cov, expected.cov, rtol=1e-10, atol=1e-12)
    assert_allclose(actual.resids, expected.resids, rtol=1e-10, atol=1e-12)
    assert actual.resids.index.equals(expected.resids.index)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("weighted", [False, True])
def test_positional_column_on_unbalanced_panel(model, weighted):
    rng = np.random.default_rng(322)
    index = pd.MultiIndex.from_product([list("abcdefgh"), range(5)])
    frame = pd.DataFrame(rng.standard_normal((40, 2)), index=index, columns=["y", "x"])
    frame = frame.drop(index[[0, 7, 19, 26]]).sample(frac=1, random_state=19)
    weights = 0.5 + rng.random(len(frame)) if weighted else None
    kwargs = {"time_effects": True} if model is PanelOLS else {}
    expected = model(frame.y, frame[["x"]], weights=weights, **kwargs).fit()
    actual = model(frame.y, frame[["x"]].to_numpy(), weights=weights, **kwargs).fit()
    assert_allclose(actual.params, expected.params, rtol=1e-10, atol=1e-12)
    assert_allclose(actual.cov, expected.cov, rtol=1e-10, atol=1e-12)
    assert_allclose(actual.resids, expected.resids, rtol=1e-10, atol=1e-12)


def test_positional_first_difference_matches_direct_regression():
    index = pd.MultiIndex.from_product([list("abc"), [4, 3, 2, 1]])
    values = np.array([3.0, 1.0, 4.0, 2.0, 5.0, 2.0, 1.0, 6.0, 2.0, 7.0, 4.0, 3.0])
    noise = np.array([0.0, 0.2, -0.1, 0.3, 0.1, -0.2, 0.2, 0.0, -0.1, 0.1, 0.2, -0.2])
    y = pd.Series(2 * values + noise, index=index)
    x = values.reshape(3, 4).T
    # Both arrays run from the latest to the earliest period. Reverse them
    # before differencing so that the oracle subtracts the preceding period.
    dx = np.diff(x[::-1], axis=0).T.reshape(-1, 1)
    dy = np.diff(y.to_numpy().reshape(3, 4)[:, ::-1], axis=1).ravel()
    expected = np.linalg.lstsq(dx, dy, rcond=None)[0]
    result = FirstDifferenceOLS(y, x).fit()
    assert_allclose(result.params, expected)
