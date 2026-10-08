"""
Tests that IVGMMCUE centers the moment conditions by default

The class documentation says that ``center`` should be True, and the constructor
tried to make it the default, but it set the default after the weight estimator
had been created so the default had no effect.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IVGMM, IVGMMCUE

CWD = os.path.split(os.path.abspath(__file__))[0]
SIMULATED = pd.read_stata(os.path.join(CWD, "results", "simulated-data.dta"))
SIMULATED["const"] = 1.0

WEIGHT_TYPES = [
    ("robust", {}),
    ("unadjusted", {}),
    ("kernel", {"bandwidth": 4}),
    ("clustered", {"clusters": SIMULATED.cluster_id.to_numpy()}),
]


@pytest.fixture(scope="module")
def args():
    data = SIMULATED
    return (
        data.y_robust,
        data[["const", "x3"]],
        data[["x1"]],
        data[["z1", "z2", "x4"]],
    )


def criterion(params, center):
    """CUE objective, written out without any of the package's weight classes"""
    data = SIMULATED
    x = data[["const", "x3", "x1"]].to_numpy()
    z = data[["const", "x3", "z1", "z2", "x4"]].to_numpy()
    nobs = x.shape[0]
    g = z * (data.y_robust.to_numpy() - x @ params)[:, None]
    gbar = g.mean(0)
    if center:
        g = g - gbar
    s = g.T @ g / nobs
    return nobs * gbar @ np.linalg.solve(s, gbar)


@pytest.mark.parametrize(("weight_type", "options"), WEIGHT_TYPES)
def test_center_is_the_default(args, weight_type, options):
    mod = IVGMMCUE(*args, weight_type=weight_type, **options)
    assert mod._weight.config["center"] is True
    assert (
        IVGMM(*args, weight_type=weight_type, **options)._weight.config["center"]
        is False
    )


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize(("weight_type", "options"), WEIGHT_TYPES)
def test_explicit_center_is_respected(args, weight_type, options, center):
    mod = IVGMMCUE(*args, weight_type=weight_type, center=center, **options)
    assert mod._weight.config["center"] is center


def test_from_formula_centers_by_default():
    formula = "y_robust ~ 1 + x3 + [x1 ~ z1 + z2 + x4]"
    default = IVGMMCUE.from_formula(formula, SIMULATED)
    assert default._weight.config["center"] is True
    off = IVGMMCUE.from_formula(formula, SIMULATED, center=False)
    assert off._weight.config["center"] is False


def test_default_estimates_use_the_centered_criterion(args):
    default = IVGMMCUE(*args).fit(display=False)
    centered = IVGMMCUE(*args, center=True).fit(display=False)
    uncentered = IVGMMCUE(*args, center=False).fit(display=False)
    assert_allclose(default.params, centered.params, rtol=1e-10)
    assert_allclose(default.j_stat.stat, centered.j_stat.stat, rtol=1e-10)
    assert abs(default.j_stat.stat - uncentered.j_stat.stat) > 0.1

    # The J statistic is the criterion evaluated at the estimates
    params = default.params[["const", "x3", "x1"]].to_numpy()
    assert_allclose(default.j_stat.stat, criterion(params, True), rtol=1e-8)
    params = uncentered.params[["const", "x3", "x1"]].to_numpy()
    assert_allclose(uncentered.j_stat.stat, criterion(params, False), rtol=1e-8)


def test_cue_against_r(args):
    # R, using the two-step GMM estimates as starting values:
    #
    # d <- foreign::read.dta("simulated-data.dta"); d$const <- 1
    # y <- d$y_robust
    # X <- as.matrix(d[, c("const", "x3", "x1")])
    # Z <- as.matrix(d[, c("const", "x3", "z1", "z2", "x4")])
    # n <- nrow(Z)
    # crit <- function(b, center) {
    #   g <- Z * drop(y - X %*% b)
    #   gbar <- colMeans(g)
    #   if (center) g <- sweep(g, 2, gbar)
    #   n * drop(crossprod(gbar, solve(crossprod(g) / n, gbar)))
    # }
    # optim(b_two_step, crit, center = TRUE, method = "BFGS",
    #       control = list(reltol = 1e-15, maxit = 2000))
    #
    # J is 21.0887480859 with center = TRUE and 20.3726905222 with center = FALSE
    expected = {
        True: (21.0887480859, [1.03085721, 0.07555129, 3.70351070]),
        False: (20.3726905222, [1.03085722, 0.07555129, 3.70351070]),
    }
    default = IVGMMCUE(*args).fit(display=False)
    assert_allclose(default.j_stat.stat, expected[True][0], rtol=1e-7)
    assert_allclose(default.params[["const", "x3", "x1"]], expected[True][1], rtol=1e-5)
    uncentered = IVGMMCUE(*args, center=False).fit(display=False)
    assert_allclose(uncentered.j_stat.stat, expected[False][0], rtol=1e-7)
    assert_allclose(
        uncentered.params[["const", "x3", "x1"]], expected[False][1], rtol=1e-5
    )
