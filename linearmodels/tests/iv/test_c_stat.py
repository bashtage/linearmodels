"""
Tests of the C (difference-in-Hansen) statistic, IVGMMResults.c_stat

The reference values in results/c-stat-reference.csv were produced by
results/c-stat-reference.R, a base R implementation of the definition in
Hayashi (2000) and Baum, Schaffer and Stillman (2003, section 4.4). Before it
writes the reference values the script checks that it reproduces results from
Stata that are used elsewhere in the tests: ivregress gmm on the housing data,
including the Hansen J statistic of an overidentified model, and estat
endogenous / estat overid on the simulated data.

Almost all of the Stata-based tests of the C statistic use a model that is
just identified. Then the restricted J statistic is identically 0 and the
choice of weighting matrix used for it is irrelevant. The scenarios here
include overidentified models, where it is not.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IVGMM, IVGMMCUE

CWD = os.path.split(os.path.abspath(__file__))[0]
RESULTS = os.path.join(CWD, "results")

SIMULATED = pd.read_stata(os.path.join(RESULTS, "simulated-data.dta"))
SIMULATED["const"] = 1.0
HOUSING = pd.read_csv(os.path.join(RESULTS, "housing.csv"), index_col=0)
HOUSING["const"] = 1.0
# Same names as the columns created by model.matrix in the R script
REGIONS = pd.get_dummies(HOUSING.region, drop_first=True).add_prefix("region")
HOUSING = pd.concat([HOUSING, REGIONS.astype(float)], axis=1)

REFERENCE = pd.read_csv(
    os.path.join(RESULTS, "c-stat-reference.csv"), keep_default_na=False
)


def build_model(row, model=IVGMM):
    """Create the model described by a row of the reference table"""
    if row.dataset == "housing":
        data, dep = HOUSING, HOUSING.rent
    else:
        data, dep = SIMULATED, SIMULATED.y_robust
    weights = data.weights if row.weighted else None
    return model(
        dep,
        data[row.exog.split()],
        data[row.endog.split()],
        data[row.instruments.split()],
        weights=weights,
    )


def reference_rows():
    return [pytest.param(row, id=row.id) for row in REFERENCE.itertuples(index=False)]


@pytest.mark.parametrize("row", reference_rows())
def test_c_stat_against_r(row):
    res = build_model(row).fit(cov_type="robust")
    c_stat = res.c_stat(row.tested.split() or None)
    assert_allclose(c_stat.stat, row.c_stat, rtol=1e-6)
    # pval is 1 - cdf, so it cannot resolve tail probabilities below about 1e-16
    assert_allclose(c_stat.pval, row.pvalue, rtol=1e-5, atol=1e-12)
    assert c_stat.df == row.df
    # The Hansen J statistic of the original model uses the same GMM steps
    assert_allclose(res.j_stat.stat, row.hansen_j, rtol=1e-6, atol=1e-8)


def test_reference_has_overidentified_models():
    # Guard against the table losing the cases where the restricted J != 0
    overidentified = REFERENCE.j_c > 1e-3
    assert overidentified.sum() >= 10
    assert not overidentified.all()
    assert (REFERENCE.c_stat >= 0).all()
    assert REFERENCE.weighted.any()
    assert not REFERENCE.weighted.all()
    # Strongly rejecting and non-rejecting examples
    assert (REFERENCE.pvalue < 1e-6).any()
    assert (REFERENCE.pvalue > 0.5).any()


def test_c_stat_just_identified_is_j_of_model_with_exogenous_tested():
    # If the original model is just identified its J is 0 and C equals the J
    # of the model that treats the tested variables as exogenous
    data = SIMULATED
    exog, endog, instr = ["const", "x3", "x4", "x5"], ["x1", "x2"], ["z1", "z2"]
    res = IVGMM(data.y_robust, data[exog], data[endog], data[instr]).fit()
    model_e = IVGMM(data.y_robust, data[exog + ["x1"]], data[["x2"]], data[instr])
    assert_allclose(res.c_stat("x1").stat, model_e.fit().j_stat.stat, rtol=1e-10)


def test_c_stat_does_not_depend_on_the_ordering_of_variables():
    # The statistic is a property of the sets of moment conditions, so it can
    # not depend on how exog, endog, the instruments and the tested variables
    # are ordered
    row = REFERENCE.set_index("id").loc["sim_e3_i4_x2x3"]
    data = SIMULATED
    exog, endog, instr = (row[c].split() for c in ("exog", "endog", "instruments"))
    base = IVGMM(data.y_robust, data[exog], data[endog], data[instr]).fit()
    expected = base.c_stat(["x2", "x3"]).stat
    assert_allclose(expected, row.c_stat, rtol=1e-6)

    rng = np.random.default_rng(0)
    for _ in range(5):
        exog_p = list(rng.permutation(exog))
        endog_p = list(rng.permutation(endog))
        instr_p = list(rng.permutation(instr))
        tested = list(rng.permutation(["x2", "x3"]))
        res = IVGMM(data.y_robust, data[exog_p], data[endog_p], data[instr_p]).fit()
        assert_allclose(res.c_stat(tested).stat, expected, rtol=1e-8)


@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
def test_c_stat_depends_on_the_data_not_the_estimator(model):
    # c_stat re-estimates the models it compares, so it is the same for
    # results from any GMM estimator of the same specification
    ref = REFERENCE.set_index("id").loc["sim_e2_i4_x2"]
    data = SIMULATED
    res = model(
        data.y_robust,
        data[ref.exog.split()],
        data[ref.endog.split()],
        data[ref.instruments.split()],
    ).fit(cov_type="robust")
    assert_allclose(res.c_stat(["x2"]).stat, ref.c_stat, rtol=1e-6)


def random_model(seed, weighted):
    """Overidentified model, one endogenous variable and three instruments"""
    rng = np.random.default_rng(seed)
    n = 200
    z = rng.standard_normal((n, 3))
    u = rng.standard_normal(n)
    v = 0.5 * u + rng.standard_normal(n)
    x1 = z @ np.array([0.6, 0.4, 0.3]) + v
    x3 = rng.standard_normal(n)
    df = pd.DataFrame(
        {
            "y": 1.0 + 0.5 * x1 + 0.3 * x3 + u,
            "const": 1.0,
            "x3": x3,
            "x1": x1,
            "z1": z[:, 0],
            "z2": z[:, 1],
            "z3": z[:, 2],
        }
    )
    w = np.exp(0.5 * rng.standard_normal(n)) if weighted else None
    return IVGMM(
        df.y, df[["const", "x3"]], df[["x1"]], df[["z1", "z2", "z3"]], weights=w
    )


# Draws where the statistic was negative, which is impossible for a C statistic
@pytest.mark.parametrize(
    ("seed", "weighted"),
    [(119, False), (193, False), (283, False), (88, True), (119, True), (341, True)],
)
def test_c_stat_is_not_negative(seed, weighted):
    c_stat = random_model(seed, weighted).fit().c_stat("x1")
    assert c_stat.stat >= 0
    assert c_stat.df == 1
