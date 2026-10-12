"""
AbsorbingLSResults.absorbed_effects is the part of the fitted values that the absorbed
terms explain, which is fitted_values - exog @ params.

The reference is the regression on the regressors and the dummy variables of the
effects, and on the other terms that are absorbed. Its fitted values less the
contribution of the regressors are the effects.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest

from linearmodels.iv.absorbing import AbsorbingLS, Interaction
from linearmodels.panel.utility import AbsorbingEffectWarning
from linearmodels.shared.exceptions import MissingValueWarning

NOBS = 360
LEVELS = (12, 7, 4)


def make_data(neffects=1, seed=0, weighted=False, labelled=False):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"obs{i}" for i in range(NOBS)]) if labelled else None
    codes = [rng.integers(0, level, NOBS) for level in LEVELS[:neffects]]
    cats = pd.DataFrame(
        {f"g{i}": pd.Categorical(c) for i, c in enumerate(codes)}, index=index
    )
    effects = sum(
        rng.standard_normal(level)[c]
        for level, c in zip(LEVELS[:neffects], codes, strict=True)
    )
    x = pd.DataFrame(
        rng.standard_normal((NOBS, 2)) + 0.5 * effects[:, None],
        columns=["x0", "x1"],
        index=index,
    )
    y = pd.Series(
        x.to_numpy() @ [1.0, -0.5] + 2.0 * effects + 3.0 + rng.standard_normal(NOBS),
        index=index,
        name="y",
    )
    weights = None
    if weighted:
        weights = pd.Series(
            np.exp(0.5 * rng.standard_normal(NOBS)), index=index, name="w"
        )
    return y, x, cats, weights


def dummies(cats):
    """Dummy variables of every effect, with the first category of all but the first"""
    return np.column_stack(
        [
            pd.get_dummies(cats[col], drop_first=i > 0, dtype=float).to_numpy()
            for i, col in enumerate(cats)
        ]
    )


def dummy_regression(y, x, controls, weights=None):
    """
    Regress y on x and the controls, which are the absorbed terms

    Returns the parameters of x, the fitted values and the part of them that the
    controls explain.
    """
    x = np.asarray(x)
    z = np.column_stack([x, controls])
    w = np.ones(len(z)) if weights is None else np.asarray(weights, dtype=float)
    root_w = np.sqrt(w / w.mean())[:, None]
    beta = np.linalg.lstsq(z * root_w, np.asarray(y)[:, None] * root_w, rcond=None)[0]
    fitted = (z @ beta)[:, 0]
    effects = (np.asarray(controls) @ beta[x.shape[1] :])[:, 0]
    return beta[: x.shape[1], 0], fitted, effects


def weighted_mean(values, weights):
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    return float(np.average(np.asarray(values), weights=w))


def test_one_value_for_each_category():
    # The setting of the issue: 600 observations and 12 categories
    rng = np.random.default_rng(366)
    n, k = 600, 12
    g = rng.integers(0, k, n)
    x = pd.DataFrame(rng.standard_normal((n, 2)), columns=["x1", "x2"])
    y = pd.Series(
        x.to_numpy() @ [1.0, -0.5] + rng.standard_normal(k)[g] + rng.standard_normal(n),
        name="y",
    )
    res = AbsorbingLS(y, x, absorb=pd.DataFrame({"g": pd.Categorical(g)})).fit()

    effects = res.absorbed_effects
    assert isinstance(effects, pd.DataFrame)
    assert list(effects.columns) == ["absorbed_effects"]
    assert len(np.unique(effects.to_numpy().round(10))) == k
    # With no constant, the effect of a category is the mean of y less x b
    resid_mean = pd.Series(y.to_numpy() - x.to_numpy() @ res.params.to_numpy())
    expected = resid_mean.groupby(g).transform("mean").to_numpy()
    assert_allclose(effects.to_numpy().squeeze(), expected, rtol=1e-10)
    # The fitted values are the sum of the regressors and the effects
    assert_allclose(
        res.fitted_values.to_numpy().squeeze(),
        x.to_numpy() @ res.params.to_numpy() + effects.to_numpy().squeeze(),
        rtol=1e-10,
    )
    level = (y.to_numpy() - x.to_numpy() @ res.params.to_numpy()).mean()
    assert_allclose(effects.to_numpy().mean(), level)


@pytest.mark.parametrize("neffects", [1, 2, 3])
@pytest.mark.parametrize(
    ("method", "weighted"), [("hdfe", False), ("lsmr", False), ("lsmr", True)]
)
def test_equals_the_effects_of_the_dummy_regression(neffects, method, weighted):
    y, x, cats, weights = make_data(neffects, weighted=weighted)
    res = AbsorbingLS(y, x, absorb=cats, weights=weights).fit(method=method)
    params, fitted, effects = dummy_regression(y, x, dummies(cats), weights)
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(res.fitted_values.to_numpy().squeeze(), fitted, rtol=1e-6)
    assert_allclose(res.absorbed_effects.to_numpy().squeeze(), effects, atol=1e-6)
    # The effects are in the units of the dependent variable, and not weighted
    assert_allclose(
        res.fitted_values.to_numpy().squeeze(),
        x.to_numpy() @ res.params.to_numpy()
        + res.absorbed_effects.to_numpy().squeeze(),
        rtol=1e-8,
    )


@pytest.mark.parametrize("neffects", [1, 2])
@pytest.mark.parametrize(
    ("method", "weighted"), [("hdfe", False), ("lsmr", False), ("lsmr", True)]
)
@pytest.mark.parametrize("position", [0, 2])
def test_constant_takes_the_mean_of_the_effects(neffects, method, weighted, position):
    y, x, cats, weights = make_data(neffects, weighted=weighted)
    exog = x.copy()
    exog.insert(position, "const", 1.0)
    res = AbsorbingLS(y, exog, absorb=cats, weights=weights).fit(method=method)
    effects = res.absorbed_effects.to_numpy().squeeze()
    assert_allclose(
        res.fitted_values.to_numpy().squeeze(),
        exog.to_numpy() @ res.params.to_numpy() + effects,
        rtol=1e-8,
    )
    # The absorbed terms have mean 0 when the constant is in exog
    assert_allclose(weighted_mean(effects, weights), 0.0, atol=1e-8)
    # They are the effects of the dummy regression less their mean
    params, _fitted, expected = dummy_regression(y, x, dummies(cats), weights)
    expected = expected - weighted_mean(expected, weights)
    assert_allclose(effects, expected, atol=1e-6)
    assert_allclose(res.params[["x0", "x1"]], params, rtol=1e-6)


@pytest.mark.parametrize("weighted", [False, True])
def test_continuous_variables_and_interactions(weighted):
    y, x, cats, weights = make_data(2, seed=3, weighted=weighted)
    rng = np.random.default_rng(4)
    z = pd.Series(rng.standard_normal(NOBS), name="z")
    ic = pd.Series(pd.Categorical(rng.integers(0, 4, NOBS)), name="ic")
    ix = pd.Series(rng.standard_normal(NOBS), name="ix")
    y = y + 0.7 * z + 0.4 * ix * ic.cat.codes
    absorb = pd.concat([cats, z], axis=1)
    inter = Interaction(ic.to_frame(), ix.to_frame())
    res = AbsorbingLS(y, x, absorb=absorb, interactions=inter, weights=weights).fit()

    slopes = np.column_stack(
        [ix.to_numpy() * (ic == k).astype(float).to_numpy() for k in range(4)]
    )
    controls = np.column_stack([dummies(cats), z.to_numpy(), slopes])
    params, _fitted, effects = dummy_regression(y, x, controls, weights)
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(res.absorbed_effects.to_numpy().squeeze(), effects, atol=1e-6)
    assert_allclose(
        res.fitted_values.to_numpy().squeeze(),
        x.to_numpy() @ res.params.to_numpy()
        + res.absorbed_effects.to_numpy().squeeze(),
        rtol=1e-8,
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_rows_with_missing_values(weighted):
    y, x, cats, weights = make_data(2, seed=5, weighted=weighted, labelled=True)
    y = y.copy()
    x = x.copy()
    y.iloc[[3, 40]] = np.nan
    x.iloc[17, 1] = np.nan
    keep = ~(y.isnull() | x.isnull().any(axis=1)).to_numpy()
    with pytest.warns(MissingValueWarning):
        res = AbsorbingLS(y, x, absorb=cats, weights=weights).fit()
    assert res.nobs == keep.sum() == NOBS - 3
    effects = res.absorbed_effects
    assert_index_equal(effects.index, y.index[keep])
    w = None if weights is None else weights[keep]
    controls = dummies(cats.iloc[keep])
    params, fitted, expected = dummy_regression(y[keep], x[keep], controls, w)
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(effects.to_numpy().squeeze(), expected, atol=1e-6)
    assert_allclose(res.fitted_values.to_numpy().squeeze(), fitted, rtol=1e-6)


def test_dropped_regressor_is_part_of_the_absorbed_effects():
    y, x, cats, _ = make_data(2, seed=6)
    exog = x.assign(dropme=lambda df: cats["g0"].cat.codes.to_numpy() * 0.5)
    mod = AbsorbingLS(y, exog, absorb=cats, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match="dropme"):
        res = mod.fit()
    assert list(res.params.index) == ["x0", "x1"]
    params, _fitted, expected = dummy_regression(y, x, dummies(cats))
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(res.absorbed_effects.to_numpy().squeeze(), expected, atol=1e-6)
    assert_allclose(
        res.fitted_values.to_numpy().squeeze(),
        x.to_numpy() @ res.params.to_numpy()
        + res.absorbed_effects.to_numpy().squeeze(),
        rtol=1e-8,
    )


def test_no_regressors():
    # All of the fitted values are explained by the absorbed terms
    y, _, cats, _ = make_data(2, seed=7)
    res = AbsorbingLS(y, absorb=cats).fit()
    assert_allclose(res.absorbed_effects.to_numpy(), res.fitted_values.to_numpy())
    _, _fitted, effects = dummy_regression(y, np.empty((NOBS, 0)), dummies(cats))
    assert_allclose(res.absorbed_effects.to_numpy().squeeze(), effects, atol=1e-6)
