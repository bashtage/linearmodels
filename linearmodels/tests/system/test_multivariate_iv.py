"""
Tests of IV3SLS.multivariate_iv with only some of the exogenous, endogenous and
instrumental variables

The documentation requires at least one of exog and endog, so endog and instruments
without exog is a valid specification.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS
from linearmodels.system import IV3SLS

NOBS = 200
NDEP = 2


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    instruments = rng.standard_normal((NOBS, 3))
    shocks = rng.standard_normal((NOBS, NDEP + 1))
    endog = (
        instruments @ np.array([[1.0, 0.5], [0.5, 1.0], [0.2, 0.2]])
        + 0.5 * shocks[:, [0]]
        + rng.standard_normal((NOBS, 2))
    )
    dependent = endog @ rng.standard_normal((2, NDEP)) + shocks[:, 1:]
    exog = rng.standard_normal((NOBS, 2))
    return dependent, exog, endog, instruments


@pytest.mark.parametrize("kind", ["array", "frame"])
def test_endog_and_instruments_without_exog(data, kind):
    dependent, _, endog, instruments = data
    if kind == "frame":
        index = pd.date_range("2000-1-1", periods=NOBS)
        dependent = pd.DataFrame(dependent, columns=["y0", "y1"], index=index)
        endog = pd.DataFrame(endog, columns=["x0", "x1"], index=index)
        instruments = pd.DataFrame(instruments, columns=["z0", "z1", "z2"], index=index)
    mod = IV3SLS.multivariate_iv(dependent, endog=endog, instruments=instruments)
    res = mod.fit()
    assert len(res.equations) == NDEP
    # The equations share the regressors and the instruments, so 3SLS is 2SLS
    # in each equation
    dep = np.asarray(dependent)
    for i, label in enumerate(res.equation_labels):
        expected = IV2SLS(dep[:, i], None, np.asarray(endog), np.asarray(instruments))
        assert_allclose(
            res.equations[label].params, expected.fit().params, rtol=1e-8, atol=1e-10
        )
        assert len(res.equations[label].params) == 2


def test_exog_only_and_with_exog_are_unchanged(data):
    dependent, exog, endog, instruments = data
    only_exog = IV3SLS.multivariate_iv(dependent, exog=exog)
    assert only_exog.fit().params.shape[0] == NDEP * exog.shape[1]
    full = IV3SLS.multivariate_iv(dependent, exog, endog, instruments).fit()
    assert full.params.shape[0] == NDEP * (exog.shape[1] + endog.shape[1])
    # endog without instruments: nothing identifies the endogenous variables
    with pytest.raises(ValueError, match=r"endogenous variables"):
        IV3SLS.multivariate_iv(dependent, endog=endog)


def test_wrong_number_of_observations(data):
    dependent, _, endog, instruments = data
    with pytest.raises(ValueError, match=r"observations"):
        IV3SLS.multivariate_iv(dependent, endog=endog[:-1], instruments=instruments)
    with pytest.raises(ValueError, match=r"observations"):
        IV3SLS.multivariate_iv(
            dependent, exog=instruments[:-5], endog=endog, instruments=instruments
        )
