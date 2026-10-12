"""
Missing values in the formulas of the system models are dropped in every block.

The formulas of the system models are parsed by the parser of the IV models. A missing value
in the dependent variable, the endogenous regressors or the instruments raised an error,
while a missing value in the exogenous regressors, and in the arrays that the models
use, dropped the observation and raised a MissingValueWarning.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

NOBS = 150
ROWS = [3, 40]
IV_FORMULA = {"eq1": "y1 ~ 1 + x1 + [e1 ~ z1 + z2]", "eq2": "y2 ~ 1 + x2"}
SUR_FORMULA = {"eq1": "y1 ~ 1 + x1", "eq2": "y2 ~ 1 + x2"}


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((NOBS, 7)),
        columns=["y1", "y2", "x1", "x2", "e1", "z1", "z2"],
    )
    df["e1"] = df["e1"] + df["z1"] + df["z2"]
    df["y1"] = df["y1"] + df["e1"]
    return df


@pytest.mark.parametrize("cls", [IV3SLS, IVSystemGMM])
@pytest.mark.parametrize("column", ["y1", "x1", "e1", "z1", "y2", "x2"])
def test_missing_value_in_each_block(frame, cls, column):
    data = frame.copy()
    data.iloc[ROWS, data.columns.get_loc(column)] = np.nan
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = cls.from_formula(IV_FORMULA, data)
    res = mod.fit()
    expected = cls.from_formula(IV_FORMULA, frame.drop(index=ROWS)).fit()
    assert res.nobs == expected.nobs
    assert_allclose(res.params, expected.params, rtol=1e-8)
    assert_allclose(res.cov, expected.cov, rtol=1e-6)


@pytest.mark.parametrize("column", ["y1", "x1", "y2", "x2"])
def test_sur_missing_value_in_each_block(frame, column):
    data = frame.copy()
    data.iloc[ROWS, data.columns.get_loc(column)] = np.nan
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = SUR.from_formula(SUR_FORMULA, data)
    res = mod.fit()
    expected = SUR.from_formula(SUR_FORMULA, frame.drop(index=ROWS)).fit()
    assert_allclose(res.params, expected.params, rtol=1e-8)
