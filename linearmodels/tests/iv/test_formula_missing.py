"""
Missing values in a formula are dropped, whatever block of the formula has them.

The dependent variable, the exogenous regressors, the endogenous regressors and the
instruments of a model made from a formula can have missing values, as the arrays that
the models use. The observations are dropped with a MissingValueWarning, and the results
are those of the model fit to the data after the rows with missing values were dropped.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from pandas.testing import assert_index_equal
import pytest

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML
from linearmodels.shared.exceptions import MissingValueWarning

MODELS = [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE]
FORMULA = "y ~ 1 + x + w + [e ~ z1 + z2]"
NOBS = 150
ROWS = [3, 40]
BLOCKS = {"dependent": "y", "exog": "x", "endog": "e", "instruments": "z1"}


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((NOBS, 6)),
        columns=["y", "x", "w", "e", "z1", "z2"],
        index=pd.Index([f"obs{i}" for i in range(NOBS)]),
    )
    df["e"] = df["e"] + df["z1"] + df["z2"]
    df["y"] = df["y"] + df["e"] + df["x"]
    df["weight"] = np.exp(0.3 * rng.standard_normal(NOBS))
    return df


def assert_same_fit(res, expected, cls):
    # The estimates of the continuously updated GMM come from an optimizer, and the
    # optimizer sees data in another memory layout
    rtol = 1e-5 if cls is IVGMMCUE else 1e-10
    assert res.nobs == expected.nobs
    assert_index_equal(res.resids.index, expected.resids.index)
    assert_allclose(res.params, expected.params, rtol=rtol)
    assert_allclose(res.cov, expected.cov, rtol=rtol * 100)
    assert_allclose(res.resids, expected.resids, rtol=rtol * 100, atol=1e-12)


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("block", list(BLOCKS))
def test_missing_value_in_each_block(frame, cls, block):
    data = frame.copy()
    data.iloc[ROWS, data.columns.get_loc(BLOCKS[block])] = np.nan
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = cls.from_formula(FORMULA, data)
    assert mod.notnull.sum() == NOBS - 2
    assert not mod.notnull[ROWS].any()
    expected = cls.from_formula(FORMULA, frame.drop(index=frame.index[ROWS])).fit()
    assert_same_fit(mod.fit(), expected, cls)


@pytest.mark.parametrize("cls", MODELS)
def test_missing_values_in_every_block_in_different_rows(frame, cls):
    data = frame.copy()
    rows = {"y": 3, "x": 10, "e": 20, "z2": 30, "w": 31}
    for col, row in rows.items():
        data.iloc[row, data.columns.get_loc(col)] = np.nan
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = cls.from_formula(FORMULA, data)
    keep = np.ones(NOBS, dtype=bool)
    keep[list(rows.values())] = False
    assert_array_equal(mod.notnull, keep)
    expected = cls.from_formula(FORMULA, frame[keep]).fit()
    assert_same_fit(mod.fit(), expected, cls)


@pytest.mark.parametrize("cls", [IV2SLS, IVLIML, IVGMM])
@pytest.mark.parametrize("block", list(BLOCKS))
def test_missing_values_with_weights(frame, cls, block):
    data = frame.copy()
    data.iloc[ROWS, data.columns.get_loc(BLOCKS[block])] = np.nan
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = cls.from_formula(FORMULA, data, weights=data["weight"])
    clean = frame.drop(index=frame.index[ROWS])
    expected = cls.from_formula(FORMULA, clean, weights=clean["weight"]).fit()
    res = mod.fit()
    assert res.nobs == NOBS - 2
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.cov, expected.cov, rtol=1e-8)


def test_no_missing_values_do_not_warn(frame, recwarn):
    IV2SLS.from_formula(FORMULA, frame).fit()
    assert not [w for w in recwarn if issubclass(w.category, MissingValueWarning)]


def test_all_values_missing(frame):
    data = frame.copy()
    data["y"] = np.nan
    with pytest.raises(ValueError, match="All observations contain missing data"):
        IV2SLS.from_formula(FORMULA, data)


@pytest.mark.parametrize("block", ["x", "e"])
def test_predict_with_missing_values(frame, block):
    res = IV2SLS.from_formula(FORMULA, frame).fit()
    new = frame.iloc[:7].copy()
    new.iloc[2, new.columns.get_loc(block)] = np.nan
    pred = res.model.predict(res.params, data=new)
    assert pred.shape == (7, 1)
    assert np.isnan(pred.iloc[2, 0])
    assert pred.drop(index=new.index[2]).notnull().all().all()
    # The results drop the predictions that are missing, unless missing is True
    assert res.predict(data=new).shape == (6, 1)
    kept = res.predict(data=new, missing=True)
    assert kept.shape == (7, 1)
    assert_allclose(kept.drop(index=new.index[2]), pred.drop(index=new.index[2]))
    complete = res.predict(data=frame.iloc[:7])
    assert_allclose(
        complete.drop(index=new.index[2]), pred.drop(index=new.index[2]), rtol=1e-12
    )
