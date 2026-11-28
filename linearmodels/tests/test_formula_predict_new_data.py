"""
predict with data builds the regressors as the model was built.

The formula was evaluated again on the new data, so that stateful transforms, such as
``center``, used the new data and the predictions were silently wrong, and categorical
variables had the levels of the new data. New data without the base level, or without
any other level, raised an error about the number of columns. The specifications of the
model matrices of the estimation data are now used.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

IV_MODELS = [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE]
PANEL_MODELS = [
    (PooledOLS, "y ~ 1 + center(x) + C(c)"),
    (PanelOLS, "y ~ 1 + center(x) + C(c) + EntityEffects"),
    (RandomEffects, "y ~ 1 + center(x) + C(c)"),
    (BetweenOLS, "y ~ 1 + center(x) + C(c)"),
    (FamaMacBeth, "y ~ 1 + center(x) + C(c)"),
    (FirstDifferenceOLS, "y ~ center(x)"),
]
SYSTEM_MODELS = [SUR, IV3SLS, IVSystemGMM]


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    nobs = 300
    df = pd.DataFrame(
        {
            "x": 3 + rng.standard_normal(nobs),
            "z": rng.standard_normal(nobs),
            "c": rng.choice(["a", "b", "c"], nobs),
        }
    )
    df["e"] = df["z"] + rng.standard_normal(nobs)
    shift = df["c"].map({"a": 0.0, "b": 1.0, "c": 2.0})
    df["y"] = 1 + df["x"] + df["e"] + shift + rng.standard_normal(nobs)
    df["y2"] = 2 - df["x"] + shift + rng.standard_normal(nobs)
    return df


def by_hand(params, train, new, endog=True):
    """Intercept + slope * (x - mean of x in train) + level + slope * e"""
    level = {"a": 0.0}
    for lvl in ("b", "c"):
        level[lvl] = params[f"C(c)[T.{lvl}]"]
    pred = params["Intercept"] + params["center(x)"] * (new["x"] - train["x"].mean())
    pred = pred + new["c"].map(level)
    if endog:
        pred = pred + params["e"] * new["e"]
    return pred.to_numpy()


@pytest.mark.parametrize("cls", IV_MODELS)
def test_iv(frame, cls):
    # No row of the base level "a", and x has a different mean
    new = frame.loc[frame["c"] != "a"].iloc[:10]
    mod = cls.from_formula("y ~ 1 + center(x) + C(c) + [e ~ z]", frame)
    res = mod.fit()
    expected = by_hand(res.params, frame, new)
    assert_allclose(np.squeeze(res.predict(data=new)), expected, rtol=1e-10)
    by_model = mod.predict(res.params, data=new)
    assert_allclose(np.squeeze(by_model), expected, rtol=1e-10)
    assert_allclose(expected, res.fitted_values.loc[new.index].squeeze(), rtol=1e-10)

    only_c = frame.loc[frame["c"] == "c"].iloc[:5]
    assert_allclose(
        np.squeeze(res.predict(data=only_c)),
        by_hand(res.params, frame, only_c),
        rtol=1e-10,
    )


@pytest.mark.parametrize(("cls", "formula"), PANEL_MODELS)
def test_panel(frame, cls, formula):
    panel = frame.copy()
    panel.index = pd.MultiIndex.from_product(
        [np.arange(30), pd.date_range("2000-01-01", periods=10, freq="YS")]
    )
    mod = cls.from_formula(formula, panel)
    res = mod.fit()
    new = panel.loc[panel["c"] != "a"].iloc[:10]
    if "C(c)" in formula:
        expected = by_hand(res.params, panel, new, endog=False)
    else:
        expected = res.params["center(x)"] * (new["x"] - panel["x"].mean())
        expected = expected.to_numpy()
    assert_allclose(np.squeeze(res.predict(data=new)), expected, rtol=1e-10)
    by_model = mod.predict(res.params, data=new)
    assert_allclose(np.squeeze(by_model), expected, rtol=1e-10)


@pytest.mark.parametrize("cls", SYSTEM_MODELS)
def test_system(frame, cls):
    if cls is SUR:
        formula = {
            "y": "y ~ 1 + center(x) + C(c) + e",
            "y2": "y2 ~ 1 + center(x) + C(c)",
        }
    else:
        formula = {
            "y": "y ~ 1 + center(x) + C(c) + [e ~ z]",
            "y2": "y2 ~ 1 + center(x) + C(c)",
        }
    mod = cls.from_formula(formula, frame)
    res = mod.fit()
    new = frame.loc[frame["c"] != "a"].iloc[:10]
    pred = res.predict(data=new)
    for key, endog in (("y", True), ("y2", False)):
        params = res.params[[p for p in res.params.index if p.startswith(f"{key}_")]]
        params.index = [p[len(key) + 1 :] for p in params.index]
        expected = by_hand(params, frame, new, endog=endog)
        assert_allclose(np.asarray(pred[key]), expected, rtol=1e-10)
