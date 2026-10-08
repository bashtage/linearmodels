import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.tools.tools import add_constant

from linearmodels.iv import IV2SLS, IVGMM
from linearmodels.shared.utility import AttrDict

CWD = os.path.split(os.path.abspath(__file__))[0]

HOUSING_DATA = pd.read_csv(os.path.join(CWD, "results", "housing.csv"), index_col=0)
HOUSING_DATA.region = HOUSING_DATA.region.astype("category")
HOUSING_DATA.state = HOUSING_DATA.state.astype("category")
HOUSING_DATA.division = HOUSING_DATA.division.astype("category")

SIMULATED_DATA = pd.read_stata(os.path.join(CWD, "results", "simulated-data.dta"))


@pytest.fixture(scope="module")
def data():
    return AttrDict(
        dep=SIMULATED_DATA.y_robust,
        exog=add_constant(SIMULATED_DATA[["x3", "x4", "x5"]]),
        endog=SIMULATED_DATA[["x1", "x2"]],
        instr=SIMULATED_DATA[["z1", "z2"]],
    )


def test_sargan(data):
    # Stata code:
    # ivregress 2sls y_robust x3 x4 x5 (x1=z1 z2)
    # estat overid
    res = IV2SLS(data.dep, data.exog, data.endog[["x1"]], data.instr).fit(
        cov_type="unadjusted"
    )
    assert_allclose(res.sargan.stat, 0.176535, rtol=1e-4)
    assert_allclose(res.sargan.pval, 0.6744, rtol=1e-4)


def test_basmann(data):
    # Stata code:
    # ivregress 2sls y_robust x3 x4 x5 (x1=z1 z2)
    # estat overid
    res = IV2SLS(data.dep, data.exog, data.endog[["x1"]], data.instr).fit(
        cov_type="unadjusted"
    )
    assert_allclose(res.basmann.stat, 0.174822, rtol=1e-4)
    assert_allclose(res.basmann.pval, 0.6759, rtol=1e-3)


def test_durbin(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit(cov_type="unadjusted")
    assert_allclose(res.durbin().stat, 35.1258, rtol=1e-4)
    assert_allclose(res.durbin().pval, 0.0000, atol=1e-6)

    assert_allclose(res.durbin("x1").stat, 0.156341, rtol=1e-4)
    assert_allclose(res.durbin("x1").pval, 0.6925, rtol=1e-3)


def test_wu_hausman(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit(cov_type="unadjusted")
    assert_allclose(res.wu_hausman().stat, 18.4063, rtol=1e-4)
    assert_allclose(res.wu_hausman().pval, 0.0000, atol=1e-6)

    assert_allclose(res.wu_hausman("x1").stat, 0.154557, rtol=1e-4)
    assert_allclose(res.wu_hausman("x1").pval, 0.6944, rtol=1e-3)


def test_durbin_wu_hausman_invariant_to_instrument_shift(data):
    # A constant shift of the excluded instruments does not change the
    # column space once the exogenous block contains a constant, so the
    # 2SLS coefficient and both exogeneity statistics stay put.
    stats = []
    params = []
    for shift in (0.0, 1.0, 5.0):
        instr = data.instr.copy()
        instr["z1"] = instr["z1"] + shift
        instr["z2"] = instr["z2"] + 0.5 * shift
        res = IV2SLS(data.dep, data.exog, data.endog[["x1"]], instr).fit(
            cov_type="unadjusted"
        )
        stats.append((res.durbin().stat, res.wu_hausman().stat))
        params.append(np.asarray(res.params))
    assert_allclose(params[1], params[0], rtol=0, atol=1e-10)
    assert_allclose(params[2], params[0], rtol=0, atol=1e-10)
    assert_allclose(stats[1], stats[0], rtol=0, atol=1e-8)
    assert_allclose(stats[2], stats[0], rtol=0, atol=1e-8)


def test_wooldridge_score(data):
    res = IV2SLS(data.dep, data.exog, data.endog[["x1", "x2"]], data.instr).fit(
        cov_type="robust"
    )
    assert_allclose(res.wooldridge_score.stat, 22.684, rtol=1e-4)
    assert_allclose(res.wooldridge_score.pval, 0.0000, atol=1e-4)


def test_wooldridge_regression(data):
    mod = IV2SLS(data.dep, data.exog, data.endog[["x1", "x2"]], data.instr)
    res = mod.fit(cov_type="robust", debiased=True)
    # Scale to correct for F vs Wald treatment
    assert_allclose(res.wooldridge_regression.stat, 2 * 13.3461, rtol=1e-4)
    assert_allclose(res.wooldridge_regression.pval, 0.0000, atol=1e-4)


def test_wooldridge_overid(data):
    res = IV2SLS(data.dep, data.exog, data.endog[["x1"]], data.instr).fit(
        cov_type="robust"
    )
    assert_allclose(res.wooldridge_overid.stat, 0.221648, rtol=1e-4)
    assert_allclose(res.wooldridge_overid.pval, 0.6378, rtol=1e-3)


def test_anderson_rubin(data):
    res = IV2SLS(data.dep, data.exog, data.endog[["x1"]], data.instr).fit(
        cov_type="unadjusted"
    )
    assert_allclose(res.nobs * (res._liml_kappa - 1), 0.176587, rtol=1e-4)


def test_basmann_f(data):
    res = IV2SLS(data.dep, data.exog, data.endog[["x1"]], data.instr).fit(
        cov_type="unadjusted"
    )
    assert_allclose(res.basmann_f.stat, 0.174821, rtol=1e-4)
    assert_allclose(res.basmann_f.pval, 0.6760, rtol=1e-3)


@pytest.mark.smoke
def test_c_stat_smoke(data):
    res = IVGMM(data.dep, data.exog, data.endog, data.instr).fit(cov_type="robust")
    c_stat = res.c_stat()
    assert_allclose(c_stat.stat, 22.684, rtol=1e-4)
    assert_allclose(c_stat.pval, 0.00, atol=1e-3)
    c_stat = res.c_stat(["x1"])
    assert_allclose(c_stat.stat, 0.158525, rtol=1e-3)
    assert_allclose(c_stat.pval, 0.6905, rtol=1e-3)
    # Final test
    c_stat2 = res.c_stat("x1")
    assert_allclose(c_stat.stat, c_stat2.stat)


def test_c_stat_exception(data):
    res = IVGMM(data.dep, data.exog, data.endog, data.instr).fit(cov_type="robust")
    match = "variables must be a str or a list of str"
    with pytest.raises(TypeError, match=match):
        res.c_stat(variables=1)
    with pytest.raises(TypeError, match=match):
        res.c_stat(variables=("x1", "x2"))


def test_weighted_sargan_wu_hausman(data):
    # R code (AER):
    # m <- ivreg(y_robust ~ x3 + x4 + x5 + x1 | x3 + x4 + x5 + z1 + z2,
    #            data = d, weights = weights)
    # summary(m, diagnostics = TRUE)
    res = IV2SLS(
        data.dep,
        data.exog,
        data.endog[["x1"]],
        data.instr,
        weights=SIMULATED_DATA.weights,
    ).fit(cov_type="unadjusted")
    assert_allclose(res.sargan.stat, 1.3091639082, rtol=1e-6)
    assert_allclose(res.wu_hausman().stat, 0.0126420296, rtol=1e-6)


def test_weighted_diagnostics_match_rescaled_data(data):
    # Weighted estimation is OLS/IV on data scaled by the root of the weights,
    # so the specification tests must agree with the rescaled unweighted model
    w = SIMULATED_DATA.weights
    root_w = np.sqrt(w)
    dep = data.dep * root_w
    exog = data.exog.mul(root_w, axis=0)
    endog_s = data.endog.mul(root_w, axis=0)
    instr = data.instr.mul(root_w, axis=0)
    for endog in (["x1"], ["x1", "x2"]):
        res = IV2SLS(data.dep, data.exog, data.endog[endog], data.instr, weights=w).fit(
            cov_type="robust"
        )
        expected = IV2SLS(dep, exog, endog_s[endog], instr).fit(cov_type="robust")
        names = ["wooldridge_score", "wooldridge_regression"]
        if len(endog) == 1:
            # Overidentification tests need more instruments than endogenous
            names += ["sargan", "basmann", "wooldridge_overid"]
        for name in names:
            assert_allclose(
                getattr(res, name).stat, getattr(expected, name).stat, rtol=1e-8
            )
        for name in ("durbin", "wu_hausman"):
            assert_allclose(
                getattr(res, name)().stat, getattr(expected, name)().stat, rtol=1e-8
            )
            assert_allclose(
                getattr(res, name)("x1").stat,
                getattr(expected, name)("x1").stat,
                rtol=1e-8,
            )

    # One endogenous variable is overidentified, two is just identified
    for endog in (["x1"], ["x1", "x2"]):
        res = IVGMM(data.dep, data.exog, data.endog[endog], data.instr, weights=w).fit(
            cov_type="robust"
        )
        expected = IVGMM(dep, exog, endog_s[endog], instr).fit(cov_type="robust")
        for variables in (None, "x1"):
            assert_allclose(
                res.c_stat(variables).stat,
                expected.c_stat(variables).stat,
                rtol=1e-8,
            )


def test_linear_restriction(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit(cov_type="robust")
    nvar = len(res.params)
    q = np.eye(nvar)
    ts = res.wald_test(q, np.zeros(nvar))
    p = res.params.values[:, None]
    c = np.asarray(res.cov)
    stat = float(np.squeeze(p.T @ np.linalg.inv(c) @ p))
    assert_allclose(stat, ts.stat)
    assert ts.df == nvar
    formula_dict: dict[str, float] = {f"{p}": 0 for p in res.params.index}
    ts2 = res.wald_test(formula=formula_dict)
    assert_allclose(ts.stat, ts2.stat)

    formula_list = [f"{k} = {v} " for k, v in formula_dict.items()]
    ts2 = res.wald_test(formula=formula_list)
    assert_allclose(ts.stat, ts2.stat)

    formula_str = ",".join([f"{k} = {v} " for k, v in formula_dict.items()])
    ts2 = res.wald_test(formula=formula_str)
    assert_allclose(ts.stat, ts2.stat)
    formula_str = " = ".join(formula_dict.keys()) + " = 0"
    ts2 = res.wald_test(formula=formula_str)
    assert_allclose(ts.stat, ts2.stat)
