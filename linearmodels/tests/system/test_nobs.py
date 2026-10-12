"""
Tests of the number of observations of the results of the system models

All equations use the same observations. nobs is the number of observations of each
equation, as the summary reports, and not of all the equations. The residual degrees
of freedom, the p-values and the confidence intervals use the observations of all
the equations.
"""

import re

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

NOBS = 120
NEQ = 3

ESTIMATORS = [
    pytest.param(SUR, False, {}, id="SUR-gls"),
    pytest.param(SUR, False, {"method": "ols"}, id="SUR-ols"),
    pytest.param(IV3SLS, True, {}, id="IV3SLS-gls"),
    pytest.param(IV3SLS, True, {"method": "ols"}, id="IV3SLS-ols"),
    pytest.param(IVSystemGMM, True, {}, id="GMM"),
]


def make_equations(instruments, nobs=NOBS, seed=0):
    """Three equations with different numbers of regressors"""
    rng = np.random.default_rng(seed)
    const = np.ones((nobs, 1))
    x = rng.standard_normal((nobs, 3))
    z = rng.standard_normal((nobs, 3))
    e = rng.standard_normal((nobs, NEQ))
    endog = z[:, [0]] + z[:, [1]] + 0.5 * e[:, [0]] + rng.standard_normal((nobs, 1))
    endog2 = z[:, [1]] - z[:, [2]] + 0.5 * e[:, [1]] + rng.standard_normal((nobs, 1))
    exog = [
        np.c_[const, x[:, [0]]],
        np.c_[const, x[:, [0, 1]]],
        np.c_[const, x[:, [2]]],
    ]
    dependent = [
        exog[0] @ [1.0, 0.5] + e[:, 0],
        exog[1] @ [0.0, 0.3, -0.2] + 0.5 * endog[:, 0] + e[:, 1],
        exog[2] @ [0.5, 0.2] - 0.3 * endog2[:, 0] + e[:, 2],
    ]
    equations = {}
    for i in range(NEQ):
        equations[f"eq{i}"] = {"dependent": dependent[i], "exog": exog[i]}
    if instruments:
        equations["eq1"].update({"endog": endog, "instruments": z[:, :2]})
        equations["eq2"].update({"endog": endog2, "instruments": z[:, 1:]})
    else:
        equations["eq1"]["exog"] = np.c_[exog[1], endog]
        equations["eq2"]["exog"] = np.c_[exog[2], endog2]
    return equations


@pytest.mark.parametrize(("model", "instruments", "options"), ESTIMATORS)
def test_nobs_is_observations_of_each_equation(model, instruments, options):
    equations = make_equations(instruments)
    res = model(equations).fit(**options)
    assert res.nobs == NOBS
    assert res.nobs == len(equations["eq0"]["dependent"])
    assert res.resids.shape == (NOBS, NEQ)
    assert res.fitted_values.shape == (NOBS, NEQ)
    assert res.wresids.shape == (NOBS, NEQ)
    for label in res.equation_labels:
        assert res.equations[label].nobs == res.nobs
    # The summary reports the number of observations of each equation
    summary = str(res.summary)
    assert re.search(rf"No. Observations:\s+{NOBS}\s", summary)
    assert str(NEQ * NOBS) not in summary


@pytest.mark.parametrize(("model", "instruments", "options"), ESTIMATORS)
@pytest.mark.parametrize("debiased", [False, True])
def test_stacked_observations_define_df_resid_and_inference(
    model, instruments, options, debiased
):
    equations = make_equations(instruments)
    res = model(equations).fit(debiased=debiased, **options)
    nparam = len(res.params)
    assert nparam == 2 + 4 + 3
    # The residual degrees of freedom count the observations of all equations
    assert res.df_resid == NEQ * NOBS - nparam
    assert res.df_model == nparam
    # and those of an equation the observations of the equation
    for label in res.equation_labels:
        eq = res.equations[label]
        assert eq.df_resid == NOBS - len(eq.params)
        assert eq.df_model == len(eq.params)

    tstats = np.asarray(res.params / res.std_errors)
    if debiased:
        pvalues = 2 * stats.t.sf(np.abs(tstats), NEQ * NOBS - nparam)
        quantile = stats.t.ppf(0.975, NEQ * NOBS - nparam)
    else:
        pvalues = 2 * stats.norm.sf(np.abs(tstats))
        quantile = stats.norm.ppf(0.975)
    assert_allclose(res.pvalues, pvalues, rtol=1e-12)
    ci = np.c_[
        res.params - quantile * res.std_errors, res.params + quantile * res.std_errors
    ]
    assert_allclose(res.conf_int(), ci, rtol=1e-12)


@pytest.mark.parametrize(("model", "instruments", "options"), ESTIMATORS)
def test_nobs_with_missing_values(model, instruments, options):
    equations = make_equations(instruments)
    # A missing value in any variable of any equation drops the row from all
    equations["eq0"]["dependent"][3] = np.nan
    equations["eq2"]["exog"][17, 1] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = model(equations)
    res = mod.fit(**options)
    assert res.nobs == NOBS - 2
    assert res.resids.shape == (NOBS - 2, NEQ)
    assert res.df_resid == NEQ * (NOBS - 2) - len(res.params)
    assert all(res.equations[label].nobs == NOBS - 2 for label in res.equation_labels)


@pytest.mark.parametrize("model", [SUR, IV3SLS, IVSystemGMM])
def test_equations_must_have_the_same_observations(model):
    equations = make_equations(instruments=False)
    equations["eq1"]["dependent"] = equations["eq1"]["dependent"][:-5]
    equations["eq1"]["exog"] = equations["eq1"]["exog"][:-5]
    with pytest.raises(ValueError, match="same number of observations"):
        model(equations)
