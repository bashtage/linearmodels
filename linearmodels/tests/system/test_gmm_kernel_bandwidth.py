"""
Tests of the default bandwidth of the kernel weight matrix of IVSystemGMM

The kernel weight matrix used a bandwidth of nobs - 2 when neither ``bandwidth``
nor ``optimal_bw`` was given. That is not a consistent estimator of the
covariance of the moment conditions. The bandwidth is now estimated from the
data, as in IVGMM, and a DefaultBandwidthWarning says how to restore the previous
bandwidth. The kernel covariance of the parameters always estimated its
bandwidth from the data, and it is not changed.

The reference values are from linearmodels 7.0, which is the code of main before
this change, with the data of ``system_data``. The bandwidth of 7.0 is
nobs - 2 = 198.
"""

from contextlib import contextmanager
import os
import warnings

import numpy as np
from numpy.linalg import solve
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IVGMM
from linearmodels.iv.covariance import kernel_optimal_bandwidth
from linearmodels.shared.exceptions import DefaultBandwidthWarning
from linearmodels.system import IVSystemGMM
from linearmodels.system.gmm import KernelWeightMatrix
from linearmodels.tests._utility import assert_close_matrix

NOBS = 200

# The two ways to restore the bandwidth of 7.0
RESTORE = {
    "bandwidth": {"bandwidth": NOBS - 2},
    "optimal_bw=False": {"optimal_bw": False},
}

# name: (options of the kernel weight matrix, keyword arguments of fit)
SPECS = {
    "weights_bartlett": ({}, {}),
    "weights_centered": ({"center": True}, {}),
    "weights_parzen": ({"kernel": "parzen"}, {}),
    "weights_debiased": ({"debiased": True}, {"debiased": True}),
    "weights_three_steps": ({}, {"iter_limit": 3}),
    "weights_and_kernel_covariance": ({}, {"cov_type": "kernel"}),
    "weights_and_kernel_covariance_parzen": (
        {"kernel": "parzen"},
        {"cov_type": "kernel", "kernel": "parzen"},
    ),
}

# Estimates of the two equations, their standard errors, the J statistic, the
# bandwidth of the weight matrix and of the kernel covariance, if there is one, of
# linearmodels 7.0
REFERENCE = {
    "weights_bartlett": (
        [
            [1.0286497540324866, 0.5430317563605719, 0.965637398315143],
            [0.7334179868151676, 0.4632870695416205, 0.9950478063081558],
        ],
        [
            [0.10446935855634021, 0.16028216189791028, 0.17761191761572884],
            [0.1547690433620426, 0.09796316188984708, 0.12343898619331123],
        ],
        1.776687580640766,
        198,
        None,
    ),
    "weights_centered": (
        [
            [1.0277017851635608, 0.3987728799791293, 0.8802764503631072],
            [0.7914629577546048, 0.5119394028497142, 0.9958100893134528],
        ],
        [
            [0.11020888095971208, 0.15108554570440147, 0.16250153540858303],
            [0.1317978421418398, 0.10018405344021883, 0.12433863809772873],
        ],
        9.762928015633923,
        198,
        None,
    ),
    "weights_parzen": (
        [
            [1.0143704600100065, 0.5354462639017347, 0.9571458773599809],
            [0.7248437413004225, 0.4667835881809671, 0.9899329024485781],
        ],
        [
            [0.2343669068820392, 0.3803940919860462, 0.2963306666914161],
            [0.20971834020714625, 0.15724321922187723, 0.1463066825105782],
        ],
        2.2646627085102162,
        198,
        None,
    ),
    "weights_debiased": (
        [
            [1.0286497540324866, 0.5430317563605712, 0.9656373983151406],
            [0.7334179868151662, 0.46328706954162113, 0.9950478063081565],
        ],
        [
            [0.10526180499610832, 0.16149797321626844, 0.17895918282076148],
            [0.15594303523002573, 0.09870625594097204, 0.12437532567590441],
        ],
        1.7500372669311537,
        198,
        None,
    ),
    "weights_three_steps": (
        [
            [1.0373461811210551, 0.5391132317774052, 0.9602381418845499],
            [0.738296572579834, 0.4650210429618931, 0.9997465168957521],
        ],
        [
            [0.10649450081947595, 0.15714623427632204, 0.17686537519157233],
            [0.14980844829910284, 0.09730627309855935, 0.12307736614360494],
        ],
        1.775092238526025,
        198,
        None,
    ),
    "weights_and_kernel_covariance": (
        [
            [1.0286497540324866, 0.5430317563605719, 0.965637398315143],
            [0.7334179868151676, 0.4632870695416205, 0.9950478063081558],
        ],
        [
            [0.1733925380790484, 0.1399948701599772, 0.1670137218388977],
            [0.15909313158064597, 0.08286530206437336, 0.10590806225748757],
        ],
        1.776687580640766,
        198,
        8,
    ),
    "weights_and_kernel_covariance_parzen": (
        [
            [1.0143704600100065, 0.5354462639017347, 0.9571458773599809],
            [0.7248437413004225, 0.4667835881809671, 0.9899329024485781],
        ],
        [
            [0.2957357973100857, 0.3221275459499461, 0.26286037941253876],
            [0.1820655681220517, 0.13151621706496508, 0.1258306265572413],
        ],
        2.2646627085102162,
        198,
        14,
    ),
}
assert set(REFERENCE) == set(SPECS)


def system_data(nobs=NOBS, rho=0.6, seed=20241012):
    """Two equations with AR(1) errors, an endogenous regressor and 3 instruments"""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    innovations = rng.standard_normal((nobs, 2)) @ chol.T
    errors = np.empty_like(innovations)
    errors[0] = innovations[0]
    for t in range(1, nobs):
        errors[t] = rho * errors[t - 1] + innovations[t]
    eqns = {}
    for i, label in enumerate(("eq1", "eq2")):
        instruments = rng.standard_normal((nobs, 3))
        x = rng.standard_normal(nobs)
        endog = (
            instruments @ np.array([0.6, 0.4, 0.3])
            + 0.5 * errors[:, i]
            + rng.standard_normal(nobs)
        )
        dependent = 1.0 + 0.5 * x + endog + errors[:, i]
        eqns[label] = {
            "dependent": dependent,
            "exog": np.column_stack([np.ones(nobs), x]),
            "endog": endog[:, None],
            "instruments": instruments,
        }
    return eqns


@pytest.fixture(scope="module")
def eqns():
    return system_data()


@contextmanager
def no_default_bandwidth_warning():
    """Fail if a DefaultBandwidthWarning is issued"""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DefaultBandwidthWarning)
        yield


def default_bandwidth_warnings(func):
    """Result of func() and the DefaultBandwidthWarnings that it issued"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func()
    return result, [w for w in caught if w.category is DefaultBandwidthWarning]


def two_stage_least_squares(eqns):
    """Regressors, instruments and residuals of 2SLS for each equation"""
    x_arrays, z_arrays, residuals = [], [], []
    for eqn in eqns.values():
        x = np.column_stack([eqn["exog"], eqn["endog"]])
        z = np.column_stack([eqn["exog"], eqn["instruments"]])
        y = eqn["dependent"]
        xpz, zpz = x.T @ z, z.T @ z
        beta = solve(xpz @ solve(zpz, xpz.T), xpz @ solve(zpz, z.T @ y))
        x_arrays.append(x)
        z_arrays.append(z)
        residuals.append(y - x @ beta)
    return x_arrays, z_arrays, np.column_stack(residuals)


def scores(z_arrays, residuals):
    """The moment conditions z_ij * e_ij of the equations side by side"""
    return np.column_stack([z * residuals[:, [i]] for i, z in enumerate(z_arrays)])


def estimated_bandwidth(moments, kernel="bartlett", center=False):
    """The bandwidth that is estimated from the moment conditions, by definition"""
    if center:
        moments = moments - moments.mean(0)
    return kernel_optimal_bandwidth((moments / moments.std(0)[None, :]).sum(1), kernel)


def first_step_bandwidth(eqns, kernel="bartlett", center=False):
    """The bandwidth for the residuals of the first step estimates"""
    _, z_arrays, residuals = two_stage_least_squares(eqns)
    return estimated_bandwidth(scores(z_arrays, residuals), kernel, center)


def fit_spec(eqns, name, options):
    weights, fit_kwargs = SPECS[name]
    mod = IVSystemGMM(eqns, weight_type="kernel", **weights, **options)
    return mod.fit(**fit_kwargs)


# The two fits are the same calculation, which only differs in the rounding of the
# linear algebra of the platform. Changing the data in the last bit changes the
# results by up to 4e-10 (the covariance, with the Parzen kernel)
SAME_RTOL = 1e-8


def assert_same_results(res, other):
    """The results are the same, up to rounding"""
    assert_allclose(res.params, other.params, rtol=SAME_RTOL)
    assert_allclose(res.std_errors, other.std_errors, rtol=SAME_RTOL)
    assert_allclose(res.cov, other.cov, rtol=SAME_RTOL)
    assert_allclose(res.w, other.w, rtol=SAME_RTOL)
    assert_allclose(res.j_stat.stat, other.j_stat.stat, rtol=SAME_RTOL)
    assert dict(res.weight_config) == dict(other.weight_config)
    assert dict(res.cov_config) == dict(other.cov_config)


@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("name", SPECS)
def test_old_bandwidth_reproduces_the_results_of_7_0(eqns, name, restore):
    # bandwidth=nobs - 2 and optimal_bw=False give the numbers of 7.0 and do not warn
    with no_default_bandwidth_warning():
        res = fit_spec(eqns, name, RESTORE[restore])
    params, std_errors, j_stat, weight_bandwidth, cov_bandwidth = REFERENCE[name]
    assert_allclose(res.params, np.concatenate(params), rtol=1e-6)
    assert_allclose(res.std_errors, np.concatenate(std_errors), rtol=1e-6)
    assert_allclose(res.j_stat.stat, j_stat, rtol=1e-6)
    assert res.weight_config["bandwidth"] == weight_bandwidth == NOBS - 2
    if cov_bandwidth is not None:
        # which estimates the bandwidth from the data, as it always did
        assert res.cov_config["bandwidth"] == cov_bandwidth


@pytest.mark.parametrize("name", SPECS)
def test_both_ways_to_restore_the_old_bandwidth_are_identical(eqns, name):
    with no_default_bandwidth_warning():
        by_bandwidth = fit_spec(eqns, name, RESTORE["bandwidth"])
        by_flag = fit_spec(eqns, name, RESTORE["optimal_bw=False"])
    assert_same_results(by_bandwidth, by_flag)


@pytest.mark.parametrize("name", SPECS)
def test_the_default_is_optimal_bw(eqns, name):
    res, caught = default_bandwidth_warnings(lambda: fit_spec(eqns, name, {}))
    assert len(caught) == 1
    with no_default_bandwidth_warning():
        optimal = fit_spec(eqns, name, {"optimal_bw": True})
    assert_same_results(res, optimal)
    # It is a bandwidth that was estimated, and not the old default of 7.0
    assert 0 < res.weight_config["bandwidth"] < NOBS // 10
    # The numbers of 7.0 were not close to these ones
    reference_errors = np.concatenate(REFERENCE[name][1])
    assert not np.allclose(res.std_errors, reference_errors, rtol=1e-2)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen"])
@pytest.mark.parametrize("center", [False, True])
def test_the_default_bandwidth_is_estimated_from_the_first_step(eqns, kernel, center):
    # The weight matrix of two-step GMM is estimated once, using the residuals of
    # the first step, which is 2SLS for each equation
    mod = IVSystemGMM(eqns, weight_type="kernel", kernel=kernel, center=center)
    res, caught = default_bandwidth_warnings(mod.fit)
    assert len(caught) == 1
    expected = first_step_bandwidth(eqns, kernel, center)
    assert res.weight_config["bandwidth"] == expected
    assert 0 < expected < NOBS // 10
    with no_default_bandwidth_warning():
        fixed = IVSystemGMM(
            eqns, weight_type="kernel", kernel=kernel, center=center, bandwidth=expected
        ).fit()
    assert_same_results(res, fixed)


def test_kernel_covariance_always_estimated_its_bandwidth(eqns):
    # The covariance of the parameters estimates its bandwidth from the residuals of
    # the final estimates, unless it is given. This is not the default that changed
    with no_default_bandwidth_warning():
        res = IVSystemGMM(eqns, weight_type="kernel", optimal_bw=False).fit(
            cov_type="kernel"
        )
    x_arrays, z_arrays, _ = two_stage_least_squares(eqns)
    params = res.params.to_numpy()
    residuals, start = [], 0
    for x, eqn in zip(x_arrays, eqns.values(), strict=True):
        residuals.append(eqn["dependent"] - x @ params[start : start + x.shape[1]])
        start += x.shape[1]
    expected = estimated_bandwidth(scores(z_arrays, np.column_stack(residuals)))
    assert res.cov_config["bandwidth"] == expected
    assert 0 < expected < NOBS // 10
    with no_default_bandwidth_warning():
        given = IVSystemGMM(eqns, weight_type="kernel", optimal_bw=False).fit(
            cov_type="kernel", bandwidth=3
        )
    assert given.cov_config["bandwidth"] == 3


# weight_type is "kernel" with these options of the weight matrix, and the keyword
# arguments of fit. The last element is the number of warnings
SCENARIOS = {
    "default": ({}, {}, 1),
    "bandwidth": ({"bandwidth": 5}, {}, 0),
    "optimal_bw": ({"optimal_bw": True}, {}, 0),
    "not_optimal_bw": ({"optimal_bw": False}, {}, 0),
    "bandwidth_none": ({"bandwidth": None, "optimal_bw": None}, {}, 1),
    "default_and_kernel_covariance": ({}, {"cov_type": "kernel"}, 1),
    "default_and_kernel_covariance_bandwidth": (
        {},
        {"cov_type": "kernel", "bandwidth": 5},
        1,
    ),
    "explicit_and_kernel_covariance": (
        {"optimal_bw": False},
        {"cov_type": "kernel"},
        0,
    ),
    "default_and_robust_covariance": ({}, {"cov_type": "robust"}, 1),
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_warning_only_for_the_default(eqns, scenario):
    weights, fit_kwargs, count = SCENARIOS[scenario]
    mod = IVSystemGMM(eqns, weight_type="kernel", **weights)

    def call_fit():
        return mod.fit(**fit_kwargs)

    _, caught = default_bandwidth_warnings(call_fit)
    assert len(caught) == count
    if count == 0:
        return
    warning = caught[0]
    assert issubclass(warning.category, UserWarning)
    message = str(warning.message)
    # The covariance does not use the default, so it is not mentioned
    assert 'weight_type="kernel"' in message
    assert 'cov_type="kernel"' not in message
    # The warning refers to the line that called fit, and not to the package or
    # to the helper that called this function
    assert os.path.abspath(warning.filename) == os.path.abspath(__file__)
    assert warning.lineno == call_fit.__code__.co_firstlineno + 1


@pytest.mark.parametrize("weight_type", ["robust", "unadjusted"])
def test_other_weights_do_not_warn(eqns, weight_type):
    mod = IVSystemGMM(eqns, weight_type=weight_type)
    _, caught = default_bandwidth_warnings(lambda: mod.fit(cov_type="kernel"))
    assert len(caught) == 0


def test_warning_is_not_issued_when_the_model_is_created(eqns):
    with no_default_bandwidth_warning():
        IVSystemGMM(eqns, weight_type="kernel")
        IVSystemGMM(eqns, weight_type="kernel", bandwidth=5)


def test_warning_message(eqns):
    mod = IVSystemGMM(eqns, weight_type="kernel")
    _, caught = default_bandwidth_warnings(mod.fit)
    message = str(caught[0].message)
    # What changed, and when
    assert "7.1" in message
    assert "nobs - 2" in message
    assert "estimated from the data" in message
    # Both arguments restore the old bandwidth, in the model
    assert "optimal_bw=False" in message
    assert "bandwidth=nobs - 2" in message
    assert 'IVSystemGMM(equations, weight_type="kernel", optimal_bw=False)' in message
    # and any of the arguments silences the warning
    assert "silences this warning" in message
    assert "bandwidth or optimal_bw" in message
    # which will not be issued from 8.0
    assert message.endswith(" This warning will be removed in linearmodels 8.0.")


def test_warning_message_has_the_form_of_the_message_of_ivgmm(eqns):
    # It is the message of IVGMM, with the example for this model
    eqn = eqns["eq1"]
    ivgmm = IVGMM(
        eqn["dependent"],
        eqn["exog"],
        eqn["endog"],
        eqn["instruments"],
        weight_type="kernel",
    )
    _, ivgmm_caught = default_bandwidth_warnings(ivgmm.fit)
    _, caught = default_bandwidth_warnings(IVSystemGMM(eqns, weight_type="kernel").fit)
    old = 'IVGMM(dependent, exog, endog, instruments, weight_type="kernel", '
    new = 'IVSystemGMM(equations, weight_type="kernel", '
    assert str(ivgmm_caught[0].message).replace(old, new) == str(caught[0].message)


def count_weight_matrices(mod):
    """Count the calls of the weight matrix of the model"""
    calls = []
    weight_matrix = mod._weight_est.weight_matrix

    def counter(*args, **kwargs):
        calls.append(1)
        return weight_matrix(*args, **kwargs)

    mod._weight_est.weight_matrix = counter
    return calls


def test_one_warning_for_a_fit_with_many_iterations(eqns):
    # Iterated GMM estimates the weight matrix at every iteration
    mod = IVSystemGMM(eqns, weight_type="kernel")
    calls = count_weight_matrices(mod)
    _, caught = default_bandwidth_warnings(lambda: mod.fit(iter_limit=10, tol=1e-12))
    assert len(calls) > 3
    assert len(caught) == 1


def test_one_warning_for_each_fit(eqns):
    mod = IVSystemGMM(eqns, weight_type="kernel")
    _, first = default_bandwidth_warnings(mod.fit)
    _, second = default_bandwidth_warnings(mod.fit)
    assert len(first) == 1
    assert len(second) == 1


def test_results_do_not_warn_again(eqns):
    res, caught = default_bandwidth_warnings(
        IVSystemGMM(eqns, weight_type="kernel").fit
    )
    assert len(caught) == 1
    with no_default_bandwidth_warning():
        assert np.isfinite(res.j_stat.stat)
        assert isinstance(str(res.summary), str)
        assert res.weight_config["bandwidth"] > 0


def test_from_formula(eqns):
    data, formulas = {}, {}
    for i, (label, eqn) in enumerate(eqns.items()):
        data[f"y{i}"] = eqn["dependent"]
        data[f"x{i}"] = eqn["exog"][:, 1]
        data[f"d{i}"] = eqn["endog"][:, 0]
        for j in range(3):
            data[f"z{i}{j}"] = eqn["instruments"][:, j]
        formulas[label] = f"y{i} ~ 1 + x{i} + [d{i} ~ z{i}0 + z{i}1 + z{i}2]"
    frame = pd.DataFrame(data)
    mod = IVSystemGMM.from_formula(formulas, frame, weight_type="kernel")
    res, caught = default_bandwidth_warnings(mod.fit)
    assert len(caught) == 1
    assert res.weight_config["bandwidth"] == first_step_bandwidth(eqns)
    mod = IVSystemGMM.from_formula(
        formulas, frame, weight_type="kernel", optimal_bw=False
    )
    with no_default_bandwidth_warning():
        old = mod.fit()
    assert old.weight_config["bandwidth"] == NOBS - 2
    assert_allclose(old.params, np.concatenate(REFERENCE["weights_bartlett"][0]))


def weight_arrays(eqns):
    x_arrays, z_arrays, residuals = two_stage_least_squares(eqns)
    return x_arrays, z_arrays, residuals


def test_kernel_weight_matrix_default(eqns):
    x, z, eps = weight_arrays(eqns)
    weight = KernelWeightMatrix()

    def compute():
        return weight.weight_matrix(x, z, eps)

    w, caught = default_bandwidth_warnings(compute)
    assert len(caught) == 1
    assert "7.1" in str(caught[0].message)
    assert os.path.abspath(caught[0].filename) == os.path.abspath(__file__)
    assert caught[0].lineno == compute.__code__.co_firstlineno + 1
    # Once for each instance, and not for each call
    again, caught = default_bandwidth_warnings(compute)
    assert len(caught) == 0
    assert_close_matrix(again, w)

    with no_default_bandwidth_warning():
        optimal = KernelWeightMatrix(optimal_bw=True)
        w_optimal = optimal.weight_matrix(x, z, eps)
    assert_close_matrix(w_optimal, w)
    assert weight.bandwidth == optimal.bandwidth
    assert weight.bandwidth == estimated_bandwidth(scores(z, eps))
    assert 0 < weight.bandwidth < NOBS // 10


def test_kernel_weight_matrix_old_bandwidth(eqns):
    x, z, eps = weight_arrays(eqns)
    for options in RESTORE.values():
        with no_default_bandwidth_warning():
            weight = KernelWeightMatrix(**options)
            weight.weight_matrix(x, z, eps)
        assert weight.bandwidth == NOBS - 2
        assert weight.config["bandwidth"] == NOBS - 2


@pytest.mark.parametrize("options", [{"bandwidth": 4}, {"optimal_bw": True}])
def test_kernel_weight_matrix_explicit(eqns, options):
    x, z, eps = weight_arrays(eqns)
    with no_default_bandwidth_warning():
        KernelWeightMatrix(**options).weight_matrix(x, z, eps)


def test_kernel_weight_matrix_bandwidth_wins_over_optimal_bw(eqns):
    x, z, eps = weight_arrays(eqns)
    with no_default_bandwidth_warning():
        weight = KernelWeightMatrix(bandwidth=7, optimal_bw=True)
        weight.weight_matrix(x, z, eps)
    assert weight.bandwidth == 7
