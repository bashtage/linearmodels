"""
Results report the estimator that produced them

IVGMMCUE set the name of its method before calling the constructor of its base
class, which set the name again. The results of the continuously updating
estimator then said that they came from IV-GMM: in the method property, in the
title and the Estimator row of the summary, and in the Estimator row of a
comparison of models.
"""

import os

import numpy as np
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML, compare

CWD = os.path.split(os.path.abspath(__file__))[0]
SIMULATED = pd.read_stata(os.path.join(CWD, "results", "simulated-data.dta"))
SIMULATED["const"] = 1.0

WEIGHT_TYPES = [
    ("robust", {}),
    ("unadjusted", {}),
    ("kernel", {"bandwidth": 4}),
    ("clustered", {"clusters": SIMULATED.cluster_id.to_numpy()}),
]
FORMULA = "y_robust ~ 1 + x3 + [x1 ~ z1 + z2 + x4]"


@pytest.fixture(scope="module")
def args():
    data = SIMULATED
    return (
        data.y_robust,
        data[["const", "x3"]],
        data[["x1"]],
        data[["z1", "z2", "x4"]],
    )


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(0)
    return pd.Series(np.exp(rng.standard_normal(SIMULATED.shape[0])))


def summary_estimator(res):
    """The method according to the title and the Estimator row of the summary"""
    lines = str(res.summary).splitlines()
    (title,) = [line.strip() for line in lines if "Estimation Summary" in line]
    (row,) = [line for line in lines if line.startswith("Estimator:")]
    return title.replace(" Estimation Summary", ""), row.split()[1]


def check_name(mod, name):
    assert mod._method == name
    res = mod.fit()
    assert res.method == name
    assert summary_estimator(res) == (name, name)
    return res


@pytest.mark.parametrize(
    ("cls", "options", "name"),
    [
        (IV2SLS, {}, "IV-2SLS"),
        (IVLIML, {}, "IV-LIML"),
        (IVLIML, {"fuller": 1}, "IV-LIML(fuller(alpha=1))"),
        (IVLIML, {"kappa": 0.9}, "IV-LIML(kappa=0.9)"),
        (IVGMM, {}, "IV-GMM"),
        (IVGMMCUE, {}, "IV-GMM-CUE"),
        (IVJIVE, {}, "IV-JIVE"),
    ],
)
def test_estimator_name(args, cls, options, name):
    check_name(cls(*args, **options), name)


@pytest.mark.parametrize(("weight_type", "options"), WEIGHT_TYPES)
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
def test_cue_name_with_any_options(
    args, weights, weight_type, options, center, weighted
):
    mod = IVGMMCUE(
        *args,
        weights=weights if weighted else None,
        weight_type=weight_type,
        center=center,
        **options,
    )
    check_name(mod, "IV-GMM-CUE")


@pytest.mark.parametrize(("weight_type", "options"), WEIGHT_TYPES)
def test_gmm_name_with_any_options(args, weights, weight_type, options):
    mod = IVGMM(*args, weights=weights, weight_type=weight_type, **options)
    check_name(mod, "IV-GMM")


def test_formula():
    cue = IVGMMCUE.from_formula(FORMULA, SIMULATED)
    assert check_name(cue, "IV-GMM-CUE").model is cue
    check_name(IVGMM.from_formula(FORMULA, SIMULATED), "IV-GMM")


def test_without_endogenous_regressors(args):
    # GMM estimators keep their name when the model has no endogenous regressors
    y, exog = args[0], args[1]
    check_name(IVGMMCUE(y, exog, None, None), "IV-GMM-CUE")
    check_name(IVGMM(y, exog, None, None), "IV-GMM")


def test_comparison(args):
    results = {
        "2sls": IV2SLS(*args).fit(),
        "gmm": IVGMM(*args).fit(),
        "cue": IVGMMCUE(*args).fit(),
    }
    comparison = compare(results)
    expected = {"2sls": "IV-2SLS", "gmm": "IV-GMM", "cue": "IV-GMM-CUE"}
    assert comparison.estimator_method.to_dict() == expected
    (row,) = [
        line for line in str(comparison.summary).splitlines() if "Estimator" in line
    ]
    assert row.split() == ["Estimator", "IV-2SLS", "IV-GMM", "IV-GMM-CUE"]
