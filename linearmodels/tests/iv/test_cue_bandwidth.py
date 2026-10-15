"""
Tests of the bandwidth that IVGMMCUE uses with kernel weights

The bandwidth is estimated once, from the residuals of the first step estimates
(the two-stage least squares estimates). The same bandwidth is used for the
starting values, for every evaluation of the objective function and for the
weight matrix of the estimates. It was estimated again at every evaluation of the
objective function, and it was a different integer for different parameters, so
that the objective function was a step function. BFGS, which uses numerical
derivatives, stopped before the minimum in about one fit of ten.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.optimize import minimize

from linearmodels.iv import IVGMM, IVGMMCUE
from linearmodels.iv.covariance import kernel_optimal_bandwidth
import linearmodels.iv.gmm as gmm_module
from linearmodels.shared.exceptions import DefaultBandwidthWarning
from linearmodels.tests._utility import warns_if_default_bandwidth

CWD = os.path.split(os.path.abspath(__file__))[0]
SIMULATED = pd.read_stata(os.path.join(CWD, "results", "simulated-data.dta"))
SIMULATED["const"] = 1.0
NOBS = SIMULATED.shape[0]
KERNELS = ["bartlett", "parzen", "qs"]
# IVGMMCUE minimizes its objective with BFGS. Two fits that only differ by the
# rounding of the linear algebra stop at estimates that differ by up to 1e-6, and in
# the number of iterations. The J statistic differs by 1e-13 or less
CUE_RTOL = 1e-4


@pytest.fixture(scope="module")
def args():
    data = SIMULATED
    return (
        data.y_kernel,
        data[["const", "x3"]],
        data[["x1"]],
        data[["z1", "z2", "x4"]],
    )


def two_stage_least_squares(x, y, z):
    """The first step estimates of GMM, from the formula of 2SLS"""
    xpz, zpz = x.T @ z, z.T @ z
    return np.linalg.solve(
        xpz @ np.linalg.solve(zpz, xpz.T), xpz @ np.linalg.solve(zpz, z.T @ y)
    )


def estimated_bandwidth(z, eps, kernel, center):
    """The bandwidth that is estimated from residuals, from its definition"""
    ze = z * np.asarray(eps)[:, None]
    if center:
        ze = ze - ze.mean(0)
    scores = (ze / ze.std(0)[None, :]).sum(1)
    return kernel_optimal_bandwidth(scores, kernel)


def simulated_arrays():
    data = SIMULATED
    x = data[["const", "x3", "x1"]].to_numpy()
    z = data[["const", "x3", "z1", "z2", "x4"]].to_numpy()
    return x, data.y_kernel.to_numpy(), z


def first_step_bandwidth(kernel="bartlett", center=True):
    x, y, z = simulated_arrays()
    eps = y - x @ two_stage_least_squares(x, y, z)
    return estimated_bandwidth(z, eps, kernel, center)


@pytest.fixture
def bandwidth_calls(monkeypatch):
    """The kernels for which kernel_optimal_bandwidth was called"""
    calls = []

    def counting(x, kernel="bartlett"):
        calls.append(kernel)
        return kernel_optimal_bandwidth(x, kernel)

    monkeypatch.setattr(gmm_module, "kernel_optimal_bandwidth", counting)
    return calls


def count_objective(mod):
    """Count the evaluations of the objective function that the optimizer makes"""
    calls = []
    j = mod.j

    def counting(*args, **kwargs):
        calls.append(1)
        return j(*args, **kwargs)

    mod.j = counting
    return calls


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("options", [{}, {"optimal_bw": True}], ids=["default", "True"])
def test_bandwidth_is_estimated_once_from_the_first_step(
    args, bandwidth_calls, kernel, center, options
):
    mod = IVGMMCUE(*args, weight_type="kernel", kernel=kernel, center=center, **options)
    evaluations = count_objective(mod)
    with warns_if_default_bandwidth(**options):
        res = mod.fit()
    # The objective function was evaluated many times with different parameters,
    # but the bandwidth was estimated once, from the first step residuals
    assert len(evaluations) > 10
    assert bandwidth_calls == [kernel]
    expected = first_step_bandwidth(kernel, center)
    assert res.weight_config["bandwidth"] == expected
    assert 0 < expected < NOBS // 10


@pytest.mark.parametrize("kernel", ["bartlett", "parzen"])
@pytest.mark.parametrize("center", [True, False])
def test_bandwidth_follows_the_centering_of_the_moment_conditions(kernel, center):
    # In simulated-data.dta the bandwidth is the same with and without centering
    # the first step scores. In this draw it is not: it is 1 and 2 with a Bartlett
    # kernel, and 5 and 6 with a Parzen kernel
    y, exog, endog, z = autocorrelated_iv(9, 250, 0.5, 4)
    x_all, z_all = np.column_stack([exog, endog]), np.column_stack([exog, z])
    eps = y - x_all @ two_stage_least_squares(x_all, y, z_all)
    centered = estimated_bandwidth(z_all, eps, kernel, True)
    uncentered = estimated_bandwidth(z_all, eps, kernel, False)
    assert centered != uncentered
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = IVGMMCUE(
            y, exog, endog, z, weight_type="kernel", kernel=kernel, center=center
        ).fit()
    assert res.weight_config["bandwidth"] == (centered if center else uncentered)


def test_bandwidth_of_the_covariance_is_estimated_from_the_final_estimates(
    args, bandwidth_calls
):
    # The kernel covariance of the parameters is a second estimator. It is
    # estimated from the residuals of the final estimates, and not the bandwidth
    # of the weight matrix
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = IVGMMCUE(*args, weight_type="kernel").fit(cov_type="kernel")
    assert len(bandwidth_calls) == 2
    x, y, z = simulated_arrays()
    eps = y - x @ res.params.to_numpy()
    assert res.weight_config["bandwidth"] == first_step_bandwidth(center=True)
    assert res.cov_config["bandwidth"] == estimated_bandwidth(
        z, eps, "bartlett", center=False
    )


@pytest.mark.parametrize(
    "options",
    [{"bandwidth": 7}, {"optimal_bw": False}, {"bandwidth": 7, "optimal_bw": True}],
    ids=["bandwidth", "not_optimal_bw", "bandwidth_and_optimal_bw"],
)
def test_bandwidth_that_is_not_estimated(args, bandwidth_calls, options):
    # A bandwidth that was given is used as given, and optimal_bw=False is nobs - 2
    res = IVGMMCUE(*args, weight_type="kernel", **options).fit()
    assert bandwidth_calls == []
    assert res.weight_config["bandwidth"] == options.get("bandwidth", NOBS - 2)


@pytest.mark.parametrize("weight_type", ["robust", "unadjusted"])
def test_other_weights_do_not_estimate_a_bandwidth(args, bandwidth_calls, weight_type):
    IVGMMCUE(*args, weight_type=weight_type).fit()
    assert bandwidth_calls == []


@pytest.mark.parametrize("kernel", KERNELS)
def test_cue_has_the_bandwidth_of_two_step_gmm(args, kernel):
    # Both use the residuals of the first step, so the estimators of the
    # weight matrix use the same bandwidth, and the starting values are those of
    # two-step GMM
    two_step = IVGMM(*args, weight_type="kernel", kernel=kernel, center=True)
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        two_step_res = two_step.fit()
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = IVGMMCUE(*args, weight_type="kernel", kernel=kernel).fit(
            starting=two_step_res.params
        )
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        default = IVGMMCUE(*args, weight_type="kernel", kernel=kernel).fit()
    assert res.weight_config["bandwidth"] == two_step_res.weight_config["bandwidth"]
    assert_allclose(res.params, default.params, rtol=CUE_RTOL)
    assert res.iterations == default.iterations


def test_the_objective_function_uses_one_bandwidth(args):
    # The objective function does not depend on the residuals at the parameters
    # through the bandwidth, so it is the function of a model that was given the
    # bandwidth. This is so before the model is fit as well
    x, y, z = simulated_arrays()
    bandwidth = first_step_bandwidth()
    mod = IVGMMCUE(*args, weight_type="kernel")
    given = IVGMMCUE(*args, weight_type="kernel", bandwidth=bandwidth)
    rng = np.random.default_rng(0)
    params = rng.standard_normal((20, 3)) * np.array([0.5, 0.5, 2.0]) + [1.0, 0.3, 3.0]
    others = []
    for param in params:
        eps = y - x @ param
        others.append(estimated_bandwidth(z, eps, "bartlett", center=True))
        j_estimated = mod.j(param, x, y[:, None], z)
        j_given = given.j(param, x, y[:, None], z)
        assert_allclose(j_estimated, j_given, rtol=1e-10)
    # which are not residuals that give the same bandwidth
    assert len(set(others) - {bandwidth}) > 3
    # and neither after the model is fit, which changes the parameters
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = mod.fit()
    assert res.weight_config["bandwidth"] == bandwidth
    for param in params:
        j_estimated = mod.j(param, x, y[:, None], z)
        j_given = given.j(param, x, y[:, None], z)
        assert_allclose(j_estimated, j_given, rtol=1e-10)


def autocorrelated_iv(seed, nobs, rho, nz):
    """A model with AR(1) errors and an endogenous regressor"""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((nobs, nz))
    v = rng.standard_normal(nobs)
    innovations = rng.standard_normal(nobs)
    errors = np.empty(nobs)
    errors[0] = innovations[0]
    for t in range(1, nobs):
        errors[t] = rho * errors[t - 1] + innovations[t]
    endog = z @ np.linspace(0.5, 0.2, nz) + v
    y = 1 + endog + 0.5 * v + errors
    return y, np.ones((nobs, 1)), endog[:, None], z


# Draws in which BFGS stopped before the minimum of the objective function, when
# the bandwidth was estimated at each evaluation. The objective function could be
# reduced by 2.6% to 8.9% by a Nelder-Mead search that started from the estimates
EARLY_STOPS = [
    (80, 250, 0.5, 4),
    (73, 250, 0.5, 4),
    (98, 250, 0.5, 4),
    (54, 250, 0.5, 4),
    (68, 500, 0.0, 6),
    (56, 500, 0.0, 6),
    (104, 500, 0.0, 6),
    (22, 500, 0.0, 6),
]


@pytest.mark.parametrize(("seed", "nobs", "rho", "nz"), EARLY_STOPS)
def test_the_optimizer_reaches_the_minimum(seed, nobs, rho, nz):
    y, exog, endog, z = autocorrelated_iv(seed, nobs, rho, nz)
    mod = IVGMMCUE(y, exog, endog, z, weight_type="kernel")
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = mod.fit()
    x, wy, wz = mod._wx, mod._wy, mod._wz
    start = res.params.to_numpy()
    value = mod.j(start, x, wy, wz)
    # Search for a lower value of the objective function that the fit used
    search = minimize(
        mod.j,
        start,
        args=(x, wy, wz),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
    )
    assert (value - search.fun) / value < 1e-9
    std_errors = res.std_errors.to_numpy()
    assert np.all(np.abs(search.x - start) / std_errors < 1e-4)
    # and so the estimates do not depend on the starting values
    other = np.asarray(IVGMM(y, exog, endog, z).fit().params) + 0.05
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        again = mod.fit(starting=other)
    assert np.all(np.abs(again.params.to_numpy() - start) / std_errors < 1e-3)


@pytest.mark.parametrize(("seed", "nobs", "rho", "nz"), EARLY_STOPS)
def test_estimates_are_those_of_the_bandwidth_that_was_estimated(seed, nobs, rho, nz):
    # The estimates move by less than 1e-10 (relative) from the estimates of
    # a model that is given the bandwidth, which is the same computation
    y, exog, endog, z = autocorrelated_iv(seed, nobs, rho, nz)
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = IVGMMCUE(y, exog, endog, z, weight_type="kernel").fit()
    bandwidth = res.weight_config["bandwidth"]
    assert bandwidth > 0
    given = IVGMMCUE(y, exog, endog, z, weight_type="kernel", bandwidth=bandwidth)
    given_res = given.fit()
    assert_allclose(res.params, given_res.params, rtol=1e-10)
    assert_allclose(res.std_errors, given_res.std_errors, rtol=1e-10)
    assert_allclose(res.j_stat.stat, given_res.j_stat.stat, rtol=1e-10)
    assert_allclose(res.weight_matrix, given_res.weight_matrix, rtol=1e-10)
    assert res.iterations == given_res.iterations


@pytest.mark.parametrize("kernel", KERNELS)
def test_weight_config_reproduces_the_fit(args, kernel):
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = IVGMMCUE(*args, weight_type="kernel", kernel=kernel).fit()
    again = IVGMMCUE(*args, weight_type=res.weight_type, **res.weight_config).fit()
    assert_allclose(again.params, res.params, rtol=CUE_RTOL)
    assert_allclose(again.j_stat.stat, res.j_stat.stat, rtol=1e-8)
    assert again.weight_config == res.weight_config


def test_fitting_again_gives_the_same_results(args, bandwidth_calls):
    mod = IVGMMCUE(*args, weight_type="kernel")
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        first = mod.fit()
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        second = mod.fit()
    assert_allclose(second.params, first.params, rtol=CUE_RTOL)
    assert second.weight_config == first.weight_config
    # It is a property of the data of the model, so it is not estimated again
    assert len(bandwidth_calls) == 1


def test_centered_j_statistic_has_the_bandwidth_of_the_estimates():
    # With centered moment conditions the J statistic and its centered version
    # are the same statistic. In this draw the residuals of the estimates give a
    # bandwidth of 9, and the residuals of the first step give 10, which the
    # centered statistic must use too
    y, exog, endog, z = autocorrelated_iv(10, 250, 0.5, 4)
    mod = IVGMMCUE(y, exog, endog, z, weight_type="kernel")
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = mod.fit()
    eps = (mod._wy - mod._wx @ res.params.to_numpy()[:, None])[:, 0]
    final_bandwidth = estimated_bandwidth(mod._wz, eps, "bartlett", center=True)
    assert final_bandwidth == 9
    assert res.weight_config["bandwidth"] == 10
    assert_allclose(res.centered_j_stat.stat, res.j_stat.stat, rtol=1e-8)
    # which is not the statistic that has another bandwidth
    other = IVGMMCUE(y, exog, endog, z, weight_type="kernel", bandwidth=9).fit()
    assert abs(other.j_stat.stat - res.j_stat.stat) > 1e-3


def test_weighted_model_uses_the_weighted_data():
    data = SIMULATED
    rng = np.random.default_rng(12)
    weights = np.exp(0.5 * rng.standard_normal(NOBS))
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        res = IVGMMCUE(
            data.y_kernel,
            data[["const", "x3"]],
            data[["x1"]],
            data[["z1", "z2", "x4"]],
            weight_type="kernel",
            weights=weights,
        ).fit()
    x, y, z = simulated_arrays()
    root_w = np.sqrt(weights / weights.mean())
    x, y, z = x * root_w[:, None], y * root_w, z * root_w[:, None]
    eps = y - x @ two_stage_least_squares(x, y, z)
    assert res.weight_config["bandwidth"] == estimated_bandwidth(
        z, eps, "bartlett", True
    )
