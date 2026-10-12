"""
AbsorbingLS drops, or refuses to fit, regressors that the absorbed effects remove.

HDFE and LSMR absorb the effects with an error of about 1e-8 of the norm of the data.
A regressor that the effects absorb entirely is left with a residual of that size, and
not with 0. The test of absorption must find it, whatever the units of the variables, and
it must keep the regressors that the effects do not absorb. The estimates of the retained
regressors must be those of the regression on the regressors and the dummy variables of
all of the effects.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv.absorbing import AbsorbingLS, Interaction
from linearmodels.panel.utility import AbsorbingEffectError, AbsorbingEffectWarning

NOBS = 400
LEVELS = (30, 12, 6)
ABSORBED = "dropme"


def make_data(neffects, kind, seed=0, weighted=False):
    """
    Simulate a model with neffects effects and a regressor that depends on kind

    kind is "sum" for a regressor that is a sum of variables that are constant within
    each effect, "dummy" for the dummy variable of a category of the last effect,
    "collinear" for the sum of x0 and such a sum, which is not absorbed but is collinear
    with x0 once the effects are removed, and "none" for a third regressor that the
    effects do not absorb.
    """
    rng = np.random.default_rng(seed)
    codes = [rng.integers(0, level, NOBS) for level in LEVELS[:neffects]]
    cats = pd.DataFrame({f"g{i}": pd.Categorical(c) for i, c in enumerate(codes)})

    def combination():
        return sum(rng.standard_normal(c.max() + 1)[c] for c in codes)

    # The regressors are partly explained by the first effect
    x = rng.standard_normal((NOBS, 2))
    x += 0.5 * rng.standard_normal(LEVELS[0])[codes[0]][:, None]
    exog = pd.DataFrame(x, columns=["x0", "x1"])
    if kind == "sum":
        exog[ABSORBED] = combination()
    elif kind == "dummy":
        exog[ABSORBED] = (codes[-1] == 0).astype(float)
    elif kind == "collinear":
        exog[ABSORBED] = x[:, 0] + combination()
    else:
        exog["x2"] = rng.standard_normal(NOBS) + 0.5 * combination()
    y = pd.Series(x @ [1.0, -0.5] + combination() + rng.standard_normal(NOBS), name="y")
    weights = None
    if weighted:
        weights = pd.Series(np.exp(0.5 * rng.standard_normal(NOBS)), name="w")
    return y, exog, cats, weights


def dummy_regression(y, exog, cats, weights=None, extra=None):
    """
    Regress y on exog, extra and the dummy variables of every effect

    Returns the parameters of exog and the robust and unadjusted covariances of the
    parameters, all without small-sample adjustments, as AbsorbingLS computes them
    with its default settings.
    """
    dummies = [
        pd.get_dummies(cats[col], drop_first=i > 0, dtype=float).to_numpy()
        for i, col in enumerate(cats)
    ]
    columns = [exog.to_numpy()]
    if extra is not None:
        columns.append(np.asarray(extra))
    z = np.column_stack(columns + dummies)
    w = np.ones(len(y)) if weights is None else weights.to_numpy() / weights.mean()
    root_w = np.sqrt(w)[:, None]
    zw = z * root_w
    yw = y.to_numpy()[:, None] * root_w
    zwi = np.linalg.inv(zw.T @ zw)
    beta = zwi @ (zw.T @ yw)
    resid = yw - zw @ beta
    scores = zw * resid
    robust = zwi @ (scores.T @ scores) @ zwi
    unadjusted = zwi * float(np.sum(resid**2)) / len(y)
    k = exog.shape[1]
    return beta[:k, 0], robust[:k, :k], unadjusted[:k, :k]


def absorbed_messages(record):
    """The messages of the warnings about absorbed variables, without those of pyhdfe"""
    return [
        str(w.message) for w in record if issubclass(w.category, AbsorbingEffectWarning)
    ]


METHODS = [("hdfe", False), ("lsmr", False), ("lsmr", True)]
METHOD_IDS = ["hdfe", "lsmr", "lsmr-weighted"]


@pytest.mark.parametrize("kind", ["sum", "dummy", "collinear"])
@pytest.mark.parametrize("neffects", [2, 3])
@pytest.mark.parametrize(("method", "weighted"), METHODS, ids=METHOD_IDS)
def test_absorbed_regressor_is_dropped(kind, neffects, method, weighted):
    y, exog, cats, weights = make_data(neffects, kind, weighted=weighted)
    mod = AbsorbingLS(y, exog, absorb=cats, weights=weights, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match=ABSORBED) as record:
        res = mod.fit(method=method)
    messages = absorbed_messages(record)
    assert len(messages) == 1
    assert messages[0].strip().endswith(ABSORBED)
    kept = ["x0", "x1"]
    assert list(res.params.index) == kept
    assert list(res.cov.columns) == kept
    params, robust, unadjusted = dummy_regression(y, exog[kept], cats, weights)
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(res.cov, robust, rtol=1e-5)
    unadj = mod.fit(cov_type="unadjusted", method=method)
    assert_allclose(unadj.cov, unadjusted, rtol=1e-5)
    # The dummy variables of the effects use the same degrees of freedom whatever
    # regressor is dropped: all of the categories, less 1 for each effect after the first
    assert res.df_absorbed == sum(cats[c].nunique() for c in cats) - (neffects - 1)


@pytest.mark.parametrize("kind", ["sum", "dummy", "collinear"])
@pytest.mark.parametrize("neffects", [2, 3])
@pytest.mark.parametrize(("method", "weighted"), METHODS, ids=METHOD_IDS)
def test_absorbed_regressor_raises_without_drop_absorbed(
    kind, neffects, method, weighted
):
    y, exog, cats, weights = make_data(neffects, kind, weighted=weighted)
    mod = AbsorbingLS(y, exog, absorb=cats, weights=weights)
    with pytest.raises(AbsorbingEffectError, match=ABSORBED):
        mod.fit(method=method)


def test_collinear_regressor_is_the_one_that_comes_last():
    y, exog, cats, _ = make_data(2, "collinear")
    exog = exog[[ABSORBED, "x0", "x1"]]
    mod = AbsorbingLS(y, exog, absorb=cats, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match="x0"):
        res = mod.fit()
    assert list(res.params.index) == [ABSORBED, "x1"]
    assert_allclose(res.params, dummy_regression(y, exog[[ABSORBED, "x1"]], cats)[0])


def test_dummy_absorbed_by_two_effects_with_the_default_method():
    # The regressor is a dummy variable of one of two effects. The default method is
    # HDFE, which left a residual of 5e-11 that was treated as a regressor, with the
    # coefficient -42316.55 and a standard error of 1.5e10.
    rng = np.random.default_rng(0)
    n = 240
    x = pd.DataFrame(rng.standard_normal((n, 2)), columns=["x0", "x1"])
    g0 = pd.Categorical(rng.integers(0, 8, n))
    g1 = pd.Categorical(rng.integers(0, 5, n))
    y = pd.Series(rng.standard_normal(n), name="y")
    dummy = pd.get_dummies(pd.Series(g1), prefix="d", dtype=float).iloc[:, [0]]
    exog = pd.concat([x, dummy], axis=1)
    cats = pd.DataFrame({"g0": g0, "g1": g1})
    with pytest.warns(AbsorbingEffectWarning, match="d_0"):
        res = AbsorbingLS(y, exog, absorb=cats, drop_absorbed=True).fit()
    assert list(res.params.index) == ["x0", "x1"]
    params, robust, _ = dummy_regression(y, x, cats)
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(res.cov, robust, rtol=1e-5)
    assert np.all(np.abs(res.params) < 1)
    with pytest.raises(AbsorbingEffectError, match="d_0"):
        AbsorbingLS(y, exog, absorb=cats).fit()


CONSTANT_ORDERS = [
    ("const", "x0", "x1", ABSORBED),
    ("x0", "const", "x1", ABSORBED),
    (ABSORBED, "const", "x0", "x1"),
    ("x0", ABSORBED, "const", "x1"),
    ("x0", "x1", ABSORBED, "const"),
]


@pytest.mark.parametrize("order", CONSTANT_ORDERS, ids="-".join)
@pytest.mark.parametrize("method", ["hdfe", "lsmr"])
def test_constant_is_retained_when_a_regressor_is_dropped(order, method):
    # A constant in exog has the span of the effects. The constant is retained and the
    # regressor that is a combination of the effects is dropped, whatever the order of
    # the columns, since the absorbed column is a multiple of the constant once the
    # grand mean is restored. The results must be those of the model without the
    # column, including the statistics that use the location of the constant.
    y, exog, cats, _ = make_data(2, "sum")
    exog["const"] = 1.0
    exog = exog[list(order)]
    kept = [c for c in order if c != ABSORBED]
    mod = AbsorbingLS(y, exog, absorb=cats, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match=ABSORBED):
        res = mod.fit(method=method)
    assert list(res.params.index) == kept
    expected = AbsorbingLS(y, exog[kept], absorb=cats).fit(method=method)
    assert_allclose(res.params, expected.params, rtol=1e-8)
    assert_allclose(res.cov, expected.cov, rtol=1e-8)
    assert res.df_model == expected.df_model
    assert_allclose(res.absorbed_rsquared, expected.absorbed_rsquared, rtol=1e-8)
    assert_allclose(res.f_statistic.stat, expected.f_statistic.stat, rtol=1e-8)
    # The slopes are those of the dummy variable regression, and the constant is the
    # level that remains once the slopes and the effects are removed
    slopes = exog[["x0", "x1"]]
    params, robust, _ = dummy_regression(y, slopes, cats)
    assert_allclose(res.params[["x0", "x1"]], params, rtol=1e-6)
    assert_allclose(res.cov.loc[["x0", "x1"], ["x0", "x1"]], robust, rtol=1e-5)
    level = float(y.mean() - slopes.mean().to_numpy() @ params)
    assert_allclose(res.params["const"], level, rtol=1e-6)
    with pytest.raises(AbsorbingEffectError, match=ABSORBED):
        AbsorbingLS(y, exog, absorb=cats).fit(method=method)


def test_continuous_absorbers_and_interactions():
    # The regressors are absorbed by categorical effects, a continuous variable and an
    # interaction of a category and a continuous variable, which only LSMR can absorb
    rng = np.random.default_rng(3)
    cats = pd.DataFrame(
        {
            "g0": pd.Categorical(rng.integers(0, 20, NOBS)),
            "g1": pd.Categorical(rng.integers(0, 8, NOBS)),
        }
    )
    z = pd.Series(rng.standard_normal(NOBS), name="z")
    ic = pd.Series(pd.Categorical(rng.integers(0, 4, NOBS)), name="ic")
    ix = pd.Series(rng.standard_normal(NOBS), name="ix")
    x = pd.DataFrame(rng.standard_normal((NOBS, 2)), columns=["x0", "x1"])
    effects = rng.standard_normal(20)[cats["g0"].cat.codes.to_numpy()]
    exog = x.assign(
        by_effects=effects + 2.5 * z,
        by_interaction=ix * (ic == 1).astype(float),
    )
    y = pd.Series(
        x.to_numpy() @ [1.0, 0.5]
        + effects
        + z
        + 0.3 * ix * ic.cat.codes
        + rng.standard_normal(NOBS),
        name="y",
    )
    absorb = pd.concat([cats, z], axis=1)
    interactions = Interaction(ic.to_frame(), ix.to_frame())
    mod = AbsorbingLS(
        y, exog, absorb=absorb, interactions=interactions, drop_absorbed=True
    )
    with pytest.warns(AbsorbingEffectWarning, match="by_effects") as record:
        res = mod.fit()
    messages = absorbed_messages(record)
    assert len(messages) == 1
    assert "by_interaction" in messages[0]
    assert list(res.params.index) == ["x0", "x1"]
    controls = np.column_stack(
        [z.to_numpy()]
        + [ix.to_numpy() * (ic == k).astype(float).to_numpy() for k in range(4)]
    )
    params, robust, _ = dummy_regression(y, x, cats, extra=controls)
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(res.cov, robust, rtol=1e-5)
    with pytest.raises(AbsorbingEffectError, match="by_interaction"):
        AbsorbingLS(y, exog, absorb=absorb, interactions=interactions).fit()


UNITS = [(1e-8, 1.0, 1e8), (1e8, 1e-8, 1.0), (1e-3, 1e3, 1.0)]


@pytest.mark.parametrize("units", UNITS)
@pytest.mark.parametrize("neffects", [2, 3])
@pytest.mark.parametrize("method", ["hdfe", "lsmr"])
@pytest.mark.parametrize("drop_absorbed", [True, False])
def test_regressors_in_any_units_are_retained(units, neffects, method, drop_absorbed):
    y, exog, cats, _ = make_data(neffects, "none")
    scaled = exog * np.array(units)
    mod = AbsorbingLS(y, scaled, absorb=cats, drop_absorbed=drop_absorbed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AbsorbingEffectWarning)
        res = mod.fit(method=method)
    assert list(res.params.index) == ["x0", "x1", "x2"]
    if max(units) / min(units) > 1e8:
        return  # The least squares step is not accurate if the units differ by 1e16
    base = AbsorbingLS(y, exog, absorb=cats).fit(method=method)
    assert_allclose(res.params * np.array(units), base.params, rtol=1e-6)
    assert_allclose(res.std_errors * np.array(units), base.std_errors, rtol=1e-6)


@pytest.mark.parametrize("unit", [1e-8, 1e-4, 1e4, 1e8])
@pytest.mark.parametrize("neffects", [2, 3])
@pytest.mark.parametrize("method", ["hdfe", "lsmr"])
def test_absorbed_regressor_in_any_units_is_dropped(unit, neffects, method):
    y, exog, cats, _ = make_data(neffects, "sum")
    exog[ABSORBED] *= unit
    mod = AbsorbingLS(y, exog, absorb=cats, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match=ABSORBED):
        res = mod.fit(method=method)
    assert list(res.params.index) == ["x0", "x1"]
    params = dummy_regression(y, exog[["x0", "x1"]], cats)[0]
    assert_allclose(res.params, params, rtol=1e-6)
    with pytest.raises(AbsorbingEffectError, match=ABSORBED):
        AbsorbingLS(y, exog, absorb=cats).fit(method=method)


def test_all_regressors_absorbed():
    y, exog, cats, _ = make_data(2, "sum")
    only = exog[[ABSORBED]]
    with pytest.raises(AbsorbingEffectError, match="All exog variables"):
        AbsorbingLS(y, only, absorb=cats).fit()
    with pytest.raises(ValueError, match="All columns in exog"):
        AbsorbingLS(y, only, absorb=cats, drop_absorbed=True).fit()
