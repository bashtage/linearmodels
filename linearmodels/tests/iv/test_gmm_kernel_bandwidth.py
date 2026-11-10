"""
Tests of the default bandwidth of the kernel estimators of IVGMM and IVGMMCUE

The kernel weight matrix and the kernel covariance used a bandwidth of nobs - 2
when neither ``bandwidth`` nor ``optimal_bw`` was given. That is not a consistent
estimator: the intervals of IVGMM covered about 65% of the time for a nominal
95%. The bandwidth is now estimated from the data, as in IV2SLS, IVLIML, IVJIVE
and the first-stage rank tests, and a DefaultBandwidthWarning says how to
restore the previous bandwidth.

The reference values are from linearmodels 7.0, which is the code of the last
commit before this change, with the data of simulated-data.dta. The bandwidth of
7.0 is nobs - 2 = 598.
"""

from contextlib import contextmanager
import os
import warnings

import numpy as np
from numpy.linalg import inv
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML
from linearmodels.iv.covariance import kernel_optimal_bandwidth
from linearmodels.iv.gmm import IVGMMCovariance, KernelWeightMatrix
from linearmodels.shared.exceptions import DefaultBandwidthWarning
from linearmodels.shared.utility import AttrDict
from linearmodels.tests._utility import assert_close_matrix, warns_if_default_bandwidth

CWD = os.path.split(os.path.abspath(__file__))[0]
SIMULATED = pd.read_stata(os.path.join(CWD, "results", "simulated-data.dta"))
SIMULATED["const"] = 1.0
NOBS = SIMULATED.shape[0]

# The two ways to restore the bandwidth of 7.0
RESTORE = {
    "bandwidth": {"bandwidth": NOBS - 2},
    "optimal_bw=False": {"optimal_bw": False},
}

# name: (model, options of the kernel weight matrix or None for robust weights,
#        options of the kernel covariance or None for a robust covariance,
#        keyword arguments of fit)
SPECS = {
    "weights_bartlett": (IVGMM, {}, None, {}),
    "weights_centered": (IVGMM, {"center": True}, None, {}),
    "weights_three_steps": (IVGMM, {}, None, {"iter_limit": 3}),
    "covariance_bartlett": (IVGMM, None, {}, {}),
    "covariance_qs": (IVGMM, None, {"kernel": "qs"}, {}),
    "weights_and_covariance_debiased": (
        IVGMM,
        {"debiased": True},
        {},
        {"debiased": True},
    ),
    "cue_weights_bartlett": (IVGMMCUE, {}, None, {}),
    "cue_weights_parzen": (IVGMMCUE, {"kernel": "parzen"}, None, {}),
    "cue_covariance": (IVGMMCUE, None, {}, {}),
}

# Estimates, standard errors, J and centered J of 7.0. In 7.0 the J statistic is
# not close to the centered J since the weight matrix with a bandwidth of 598 is
# not consistent
REFERENCE = {
    "weights_bartlett": (
        [1.071624969969398, 0.32804972271941907, 2.907889533547241],
        [0.08935682722946905, 0.1313216522538519, 0.25963224113738476],
        1.3866332586143622,
        424.9393597602415,
    ),
    "weights_centered": (
        [0.8750563962630555, 0.02712480298287136, 2.953464228506674],
        [0.1100625567626914, 0.1471217250241003, 0.2699473369941144],
        133.56483437971684,
        135.56312323520288,
    ),
    "weights_three_steps": (
        [1.0606063455934849, 0.31259190135584447, 2.9197404004826053],
        [0.0901281398138085, 0.13169769014586097, 0.2602117999670635],
        1.3858538292540599,
        402.73408454899976,
    ),
    "covariance_bartlett": (
        [1.1041932229586693, 0.3419657325304502, 2.8976285761224556],
        [0.031247605869870923, 0.05308501551539399, 0.11836601828622118],
        24.657296231656982,
        25.649534638397093,
    ),
    "covariance_qs": (
        [1.1041932229586693, 0.3419657325304502, 2.8976285761224556],
        [0.019333422765751947, 0.021327172477498234, 0.07893673202146821],
        24.657296231656982,
        25.649534638397093,
    ),
    "weights_and_covariance_debiased": (
        [1.0716249699693978, 0.32804972271941946, 2.9078895335472392],
        [0.025765517429928036, 0.05125315440821739, 0.08675782279587814],
        1.379700092321304,
        422.814662961439,
    ),
    "cue_weights_bartlett": (
        [0.902327458150663, -0.28400600849819135, 3.279592616832745],
        [0.11062146604779992, 0.19065909191616504, 0.2641550580777918],
        113.09959680477508,
        113.09959680477508,
    ),
    "cue_weights_parzen": (
        [0.8583554621969073, -0.30172932143889497, 3.255145131955993],
        [0.11459037455260472, 0.19714851610503573, 0.30732988127109834],
        66.60508259799519,
        66.60508259799741,
    ),
    "cue_covariance": (
        [1.0995626096360471, 0.18298066893555495, 3.262531091017902],
        [0.032463335676012416, 0.15295791107254517, 0.3132539855667002],
        23.49944558981567,
        23.49944558981567,
    ),
}
assert set(REFERENCE) == set(SPECS)


@pytest.fixture(scope="module")
def args():
    data = SIMULATED
    return (
        data.y_kernel,
        data[["const", "x3"]],
        data[["x1"]],
        data[["z1", "z2", "x4"]],
    )


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


def fit_spec(args, name, options):
    """Fit a spec, with the same options for each kernel estimator that it has"""
    model, weights, covariance, fit_kwargs = SPECS[name]
    if weights is None:
        mod = model(*args)
    else:
        mod = model(*args, weight_type="kernel", **weights, **options)
    if covariance is None:
        return mod.fit(**fit_kwargs)
    return mod.fit(cov_type="kernel", **covariance, **options, **fit_kwargs)


def check_against_reference(res, name):
    params, std_errors, j_stat, centered_j_stat = REFERENCE[name]
    cue = name.startswith("cue")
    assert_allclose(res.params, params, rtol=1e-5 if cue else 1e-7)
    assert_allclose(res.std_errors, std_errors, rtol=1e-4 if cue else 1e-7)
    assert_allclose(res.j_stat.stat, j_stat, rtol=1e-6 if cue else 1e-7)
    assert_allclose(res.centered_j_stat.stat, centered_j_stat, rtol=1e-6)


def same_rtol(name):
    """
    The relative tolerance of fits that are the same calculation, up to rounding

    The fits of IVGMM are closed form. Changing the data in the last bit changes them
    by 1e-12 or less, and the elements of the weight matrix by 6e-11. IVGMMCUE
    minimizes its objective with BFGS, and the estimates and the covariance of its
    fits differ by up to 6e-6 and 3e-6.
    """
    return 1e-4 if name.startswith("cue") else 1e-8


def assert_same_results(res, other, rtol=1e-8):
    """The results are the same, up to rounding"""
    assert_allclose(res.params, other.params, rtol=rtol)
    assert_allclose(res.std_errors, other.std_errors, rtol=rtol)
    assert_close_matrix(res.cov, other.cov, rtol)
    assert_allclose(res.j_stat.stat, other.j_stat.stat, rtol=1e-8)
    assert_allclose(res.centered_j_stat.stat, other.centered_j_stat.stat, rtol=1e-8)
    assert_close_matrix(res.weight_matrix, other.weight_matrix, rtol)
    assert res.weight_config == other.weight_config
    assert res.cov_config == other.cov_config


@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("name", SPECS)
def test_old_bandwidth_reproduces_the_results_of_7_0(args, name, restore):
    # bandwidth=nobs - 2 and optimal_bw=False give the numbers of 7.0, for the
    # weight matrix and for the covariance, and do not warn
    with no_default_bandwidth_warning():
        res = fit_spec(args, name, RESTORE[restore])
    check_against_reference(res, name)
    _, weights, covariance, _ = SPECS[name]
    if weights is not None:
        assert res.weight_config["bandwidth"] == NOBS - 2
    if covariance is not None:
        assert res.cov_config["bandwidth"] == NOBS - 2


@pytest.mark.parametrize("name", SPECS)
def test_both_ways_to_restore_the_old_bandwidth_are_identical(args, name):
    with no_default_bandwidth_warning():
        by_bandwidth = fit_spec(args, name, RESTORE["bandwidth"])
        by_flag = fit_spec(args, name, RESTORE["optimal_bw=False"])
    assert_same_results(by_bandwidth, by_flag, same_rtol(name))


@pytest.mark.parametrize("name", SPECS)
def test_the_default_is_optimal_bw(args, name):
    res, caught = default_bandwidth_warnings(lambda: fit_spec(args, name, {}))
    assert len(caught) == 1
    with no_default_bandwidth_warning():
        optimal = fit_spec(args, name, {"optimal_bw": True})
    assert_same_results(res, optimal, same_rtol(name))
    # It is a bandwidth that was estimated, and not the old default of 7.0
    _, weights, covariance, _ = SPECS[name]
    if weights is not None:
        assert 0 < res.weight_config["bandwidth"] < NOBS // 10
    if covariance is not None:
        assert 0 < res.cov_config["bandwidth"] < NOBS // 10
    # The numbers of 7.0 were not close to these ones
    assert not np.allclose(res.std_errors, REFERENCE[name][1], rtol=1e-2)


def data_driven_bandwidth(eps, kernel="bartlett"):
    """The bandwidth for residuals eps, computed from the definition"""
    data = SIMULATED
    z = data[["const", "x3", "z1", "z2", "x4"]].to_numpy()
    ze = z * np.asarray(eps)[:, None]
    scores = (ze / ze.std(0)[None, :]).sum(1)
    return kernel_optimal_bandwidth(scores, kernel)


def test_the_default_weight_bandwidth_is_estimated_from_the_first_step(args):
    # In two-step GMM the weight matrix is estimated once, using the residuals of
    # the first step, which uses the weight matrix (Z'Z / n)^{-1}
    data = SIMULATED
    x = data[["const", "x3", "x1"]].to_numpy()
    z = data[["const", "x3", "z1", "z2", "x4"]].to_numpy()
    y = data.y_kernel.to_numpy()[:, None]
    first = IVGMM.estimate_parameters(x, y, z, inv(z.T @ z / NOBS))
    bandwidth = data_driven_bandwidth((y - x @ first)[:, 0])
    assert 0 < bandwidth < NOBS // 10

    res, caught = default_bandwidth_warnings(
        lambda: IVGMM(*args, weight_type="kernel").fit()
    )
    assert len(caught) == 1
    assert res.weight_config["bandwidth"] == bandwidth
    with no_default_bandwidth_warning():
        fixed = IVGMM(*args, weight_type="kernel", bandwidth=bandwidth).fit()
    assert_same_results(res, fixed)


def hac_covariance(res, bandwidth):
    """GMM covariance with a Bartlett kernel, from the definition"""
    data = SIMULATED
    x = data[["const", "x3", "x1"]].to_numpy()
    z = data[["const", "x3", "z1", "z2", "x4"]].to_numpy()
    eps = data.y_kernel.to_numpy() - x @ res.params.to_numpy()
    ze = z * eps[:, None]
    s = ze.T @ ze / NOBS
    for lag in range(1, bandwidth + 1):
        gamma = ze[lag:].T @ ze[:-lag] / NOBS
        s += (1 - lag / (bandwidth + 1)) * (gamma + gamma.T)
    xpz = x.T @ z / NOBS
    w = res.weight_matrix.to_numpy()
    bread = inv(xpz @ w @ xpz.T)
    cov = bread @ (xpz @ w @ s @ w @ xpz.T) @ bread / NOBS
    return (cov + cov.T) / 2, eps


def test_the_default_covariance_bandwidth_is_estimated_from_the_residuals(args):
    res, caught = default_bandwidth_warnings(
        lambda: IVGMM(*args).fit(cov_type="kernel")
    )
    assert len(caught) == 1
    # The bandwidth is estimated from the final residuals
    bandwidth = res.cov_config["bandwidth"]
    cov, eps = hac_covariance(res, bandwidth)
    assert bandwidth == data_driven_bandwidth(eps)
    assert 0 < bandwidth < NOBS // 10
    assert_allclose(res.cov, cov, rtol=1e-10)
    # and the covariance with the old bandwidth is different
    old, _ = hac_covariance(res, NOBS - 2)
    assert not np.allclose(np.sqrt(np.diag(old)), res.std_errors, rtol=0.1)
    with no_default_bandwidth_warning():
        old_res = IVGMM(*args).fit(cov_type="kernel", bandwidth=NOBS - 2)
    assert_allclose(old_res.cov, old, rtol=1e-10)


# weight_type is "kernel" if the weight configuration is not None, and the
# covariance is kernel if the configuration of fit is not None. The fourth
# element is where the warning refers to
SCENARIOS = {
    "weights": ({}, {}, "weights"),
    "weights_bandwidth": ({"bandwidth": 5}, {}, None),
    "weights_optimal_bw": ({"optimal_bw": True}, {}, None),
    "weights_not_optimal_bw": ({"optimal_bw": False}, {}, None),
    "weights_bandwidth_none": ({"bandwidth": None, "optimal_bw": None}, {}, "weights"),
    "covariance": (None, {"cov_type": "kernel"}, "covariance"),
    "covariance_bandwidth": (None, {"cov_type": "kernel", "bandwidth": 5}, None),
    "covariance_optimal_bw": (None, {"cov_type": "kernel", "optimal_bw": True}, None),
    "covariance_not_optimal_bw": (
        None,
        {"cov_type": "kernel", "optimal_bw": False},
        None,
    ),
    "weights_and_covariance": ({}, {"cov_type": "kernel"}, "both"),
    "weights_and_covariance_bandwidth": (
        {"bandwidth": 5},
        {"cov_type": "kernel", "bandwidth": 5},
        None,
    ),
    "weights_and_covariance_optimal_bw": (
        {"optimal_bw": True},
        {"cov_type": "kernel", "optimal_bw": True},
        None,
    ),
    "explicit_weights_default_covariance": (
        {"bandwidth": 5},
        {"cov_type": "kernel"},
        "covariance",
    ),
    "default_weights_explicit_covariance": (
        {},
        {"cov_type": "kernel", "bandwidth": 5},
        "weights",
    ),
    "old_weights_explicit_covariance": (
        {"optimal_bw": False},
        {"cov_type": "kernel", "bandwidth": 5},
        None,
    ),
    "no_kernel": (None, {}, None),
    "other_covariances": (None, {"cov_type": "unadjusted"}, None),
}


@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_warning_only_for_the_default(args, model, scenario):
    weights, fit_kwargs, where = SCENARIOS[scenario]
    if weights is None:
        mod = model(*args)
    else:
        mod = model(*args, weight_type="kernel", **weights)

    def call_fit():
        return mod.fit(**fit_kwargs)

    _, caught = default_bandwidth_warnings(call_fit)
    assert len(caught) == (0 if where is None else 1)
    if where is None:
        return
    warning = caught[0]
    assert issubclass(warning.category, UserWarning)
    message = str(warning.message)
    assert ('weight_type="kernel"' in message) == (where in ("weights", "both"))
    assert ('cov_type="kernel"' in message) == (where in ("covariance", "both"))
    # The warning refers to the line that called fit, and not to the package or
    # to the helper that called this function
    assert os.path.abspath(warning.filename) == os.path.abspath(__file__)
    assert warning.lineno == call_fit.__code__.co_firstlineno + 1
    # and the example is for the model
    assert (f"{model.__name__}(dependent, exog" in message) == (
        where in ("weights", "both")
    )


@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
def test_warning_is_not_issued_when_the_model_is_created(args, model):
    with no_default_bandwidth_warning():
        model(*args, weight_type="kernel")
        model(*args, weight_type="kernel", bandwidth=5)


def test_warning_class():
    assert issubclass(DefaultBandwidthWarning, UserWarning)
    # A DeprecationWarning is not shown by default
    assert not issubclass(
        DefaultBandwidthWarning, (DeprecationWarning, PendingDeprecationWarning)
    )
    assert not issubclass(DefaultBandwidthWarning, FutureWarning)
    assert DefaultBandwidthWarning.__doc__
    # The class is not deprecated, but the warning will be removed in 8.0
    doc = " ".join(DefaultBandwidthWarning.__doc__.split())
    assert "This warning will be removed in linearmodels 8.0" in doc
    assert "the estimated bandwidth has been the default for a release" in doc
    assert "deprecated" not in doc


def test_warning_message(args):
    mod = IVGMM(*args, weight_type="kernel")
    _, caught = default_bandwidth_warnings(lambda: mod.fit(cov_type="kernel"))
    message = str(caught[0].message)
    # What changed, and when
    assert "7.1" in message
    assert "nobs - 2" in message
    assert "estimated from the data" in message
    # Both arguments restore the old bandwidth, for the weights and the covariance
    assert "optimal_bw=False" in message
    assert "bandwidth=nobs - 2" in message
    assert 'weight_type="kernel", optimal_bw=False' in message
    assert 'fit(cov_type="kernel", optimal_bw=False)' in message
    # and any of the arguments silences the warning
    assert "silences this warning" in message
    assert "bandwidth or optimal_bw" in message
    # which will not be issued from 8.0, in one sentence at the end
    assert message.endswith(" This warning will be removed in linearmodels 8.0.")


def count_weight_matrices(mod):
    """Count the calls of the weight matrix of the model"""
    calls = []
    weight_matrix = mod._weight.weight_matrix

    def counter(*args, **kwargs):
        calls.append(1)
        return weight_matrix(*args, **kwargs)

    mod._weight.weight_matrix = counter
    return calls


def test_one_warning_for_a_fit_with_many_iterations(args):
    # Iterated GMM estimates the weight matrix at every iteration, and the CUE
    # objective is evaluated many times
    mod = IVGMM(*args, weight_type="kernel")
    calls = count_weight_matrices(mod)
    _, caught = default_bandwidth_warnings(lambda: mod.fit(iter_limit=10, tol=1e-12))
    assert len(calls) > 3
    assert len(caught) == 1

    mod = IVGMMCUE(*args, weight_type="kernel")
    calls = count_weight_matrices(mod)
    _, caught = default_bandwidth_warnings(lambda: mod.fit(cov_type="kernel"))
    assert len(calls) > 10
    assert len(caught) == 1


@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
def test_one_warning_for_each_fit(args, model):
    mod = model(*args, weight_type="kernel")
    _, first = default_bandwidth_warnings(mod.fit)
    _, second = default_bandwidth_warnings(mod.fit)
    assert len(first) == 1
    assert len(second) == 1


@pytest.mark.parametrize("model", [IVGMM, IVGMMCUE])
def test_results_do_not_warn_again(args, model):
    # The statistics that are computed from the results estimate other models
    res, caught = default_bandwidth_warnings(
        lambda: model(*args, weight_type="kernel").fit(cov_type="kernel")
    )
    assert len(caught) == 1
    with no_default_bandwidth_warning():
        assert res.c_stat("x1").stat >= 0
        assert np.isfinite(res.centered_j_stat.stat)
        assert np.isfinite(res.j_stat.stat)
        assert isinstance(str(res.summary), str)
        assert isinstance(str(res.first_stage), str)


def test_c_stat_uses_the_bandwidth_that_the_model_estimates(args):
    res, _ = default_bandwidth_warnings(
        lambda: IVGMM(*args, weight_type="kernel").fit()
    )
    with no_default_bandwidth_warning():
        optimal = IVGMM(*args, weight_type="kernel", optimal_bw=True).fit()
        old = IVGMM(*args, weight_type="kernel", bandwidth=NOBS - 2).fit()
    assert_allclose(res.c_stat("x1").stat, optimal.c_stat("x1").stat, rtol=1e-8)
    assert res.c_stat("x1").stat != old.c_stat("x1").stat


@pytest.mark.parametrize(
    ("options", "start_options"),
    [
        ({}, {"optimal_bw": True}),
        ({"optimal_bw": True}, {"optimal_bw": True}),
        ({"optimal_bw": False}, {"optimal_bw": False}),
        ({"bandwidth": 5}, {"bandwidth": 5}),
    ],
    ids=["default", "optimal_bw", "not_optimal_bw", "bandwidth"],
)
def test_cue_starting_values_have_the_bandwidth_of_the_objective(
    args, options, start_options
):
    # The starting values are the estimates of IVGMM with the bandwidth of the
    # objective function: the one that is estimated from the first step residuals
    # by default and with optimal_bw=True, as IVGMM does, the one that was given,
    # and nobs - 2 with optimal_bw=False
    start = IVGMM(*args, weight_type="kernel", center=True, **start_options).fit()
    # The default warns
    with warns_if_default_bandwidth(**options):
        res = IVGMMCUE(*args, weight_type="kernel", **options).fit()
    with warns_if_default_bandwidth(**options):
        res_start = IVGMMCUE(*args, weight_type="kernel", **options).fit(
            starting=start.params
        )
    assert_allclose(res.params, res_start.params, rtol=same_rtol("cue"))
    assert res.iterations == res_start.iterations
    assert_allclose(res.j_stat.stat, res_start.j_stat.stat, rtol=1e-8)
    if start_options.get("optimal_bw"):
        # and the weight matrix of the objective has the same bandwidth too
        assert res.weight_config["bandwidth"] == start.weight_config["bandwidth"]


def test_from_formula(args):
    formula = "y_kernel ~ 1 + x3 + [x1 ~ z1 + z2 + x4]"
    for model in (IVGMM, IVGMMCUE):
        mod = model.from_formula(formula, SIMULATED, weight_type="kernel")
        res, caught = default_bandwidth_warnings(mod.fit)
        assert len(caught) == 1
        mod = model.from_formula(
            formula, SIMULATED, weight_type="kernel", optimal_bw=False
        )
        with no_default_bandwidth_warning():
            old = mod.fit()
        assert old.weight_config["bandwidth"] == NOBS - 2
        assert res.weight_config["bandwidth"] < NOBS // 10


def test_other_estimators_are_silent(args):
    # They always estimate the bandwidth from the data
    with no_default_bandwidth_warning():
        for model in (IV2SLS, IVLIML, IVJIVE):
            res = model(*args).fit(cov_type="kernel")
            assert 0 < res.cov_config["bandwidth"] < NOBS // 10


@pytest.fixture(scope="module")
def arrays():
    """Arrays for the classes that the models use, and residuals at fixed params"""
    data = SIMULATED
    x = data[["const", "x3", "x1"]].to_numpy()
    z = data[["const", "x3", "z1", "z2", "x4"]].to_numpy()
    y = data.y_kernel.to_numpy()[:, None]
    params = np.array([[1.0], [0.3], [2.9]])
    return AttrDict(x=x, y=y, z=z, params=params, eps=y - x @ params)


def test_kernel_weight_matrix_default(arrays):
    x, z, eps = arrays.x, arrays.z, arrays.eps
    weight = KernelWeightMatrix()
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        KernelWeightMatrix().weight_matrix(x, z, eps)

    def compute():
        return weight.weight_matrix(x, z, eps)

    w, caught = default_bandwidth_warnings(compute)
    assert len(caught) == 1
    assert os.path.abspath(caught[0].filename) == os.path.abspath(__file__)
    assert caught[0].lineno == compute.__code__.co_firstlineno + 1
    # Once for each instance, and not for each call
    again, caught = default_bandwidth_warnings(compute)
    assert len(caught) == 0
    assert_close_matrix(again, w, 1e-10)

    with no_default_bandwidth_warning():
        optimal = KernelWeightMatrix(optimal_bw=True)
        w_optimal = optimal.weight_matrix(x, z, eps)
    assert_close_matrix(w_optimal, w, 1e-10)
    assert weight.bandwidth == optimal.bandwidth
    assert 0 < weight.bandwidth < NOBS // 10


@pytest.mark.parametrize("options", RESTORE.values())
def test_kernel_weight_matrix_old_bandwidth(arrays, options):
    x, z, eps = arrays.x, arrays.z, arrays.eps
    with no_default_bandwidth_warning():
        weight = KernelWeightMatrix(**options)
        w = weight.weight_matrix(x, z, eps)
    assert weight.bandwidth == NOBS - 2
    assert weight.config["bandwidth"] == NOBS - 2
    # Computed from the definition, with the weights of the Bartlett kernel
    ze = z * eps
    s = ze.T @ ze
    for lag in range(1, NOBS - 1):
        gamma = ze[lag:].T @ ze[:-lag]
        s += (1 - lag / (NOBS - 1)) * (gamma + gamma.T)
    assert_allclose(w, s / NOBS, rtol=1e-12)


@pytest.mark.parametrize("options", [{"bandwidth": 4}, {"optimal_bw": True}])
def test_kernel_weight_matrix_explicit(arrays, options):
    with no_default_bandwidth_warning():
        KernelWeightMatrix(**options).weight_matrix(arrays.x, arrays.z, arrays.eps)


def test_kernel_weight_matrix_bandwidth_wins_over_optimal_bw(arrays):
    with no_default_bandwidth_warning():
        weight = KernelWeightMatrix(bandwidth=7, optimal_bw=True)
        weight.weight_matrix(arrays.x, arrays.z, arrays.eps)
    assert weight.bandwidth == 7


def test_gmm_covariance_default(arrays):
    x, y, z, params = arrays.x, arrays.y, arrays.z, arrays.params
    w = inv(z.T @ z / NOBS)
    cov = IVGMMCovariance(x, y, z, params, w, "kernel")

    def compute():
        return cov.cov

    default, caught = default_bandwidth_warnings(compute)
    assert len(caught) == 1
    assert 'cov_type="kernel"' in str(caught[0].message)
    assert os.path.abspath(caught[0].filename) == os.path.abspath(__file__)
    assert caught[0].lineno == compute.__code__.co_firstlineno + 1
    with no_default_bandwidth_warning():
        optimal = IVGMMCovariance(x, y, z, params, w, "kernel", optimal_bw=True)
        optimal_cov = optimal.cov
    assert_close_matrix(optimal_cov, default, 1e-10)
    assert cov.config["bandwidth"] == optimal.config["bandwidth"]
    assert 0 < cov.config["bandwidth"] < NOBS // 10


@pytest.mark.parametrize("options", RESTORE.values())
def test_gmm_covariance_old_bandwidth(arrays, options):
    x, y, z, params = arrays.x, arrays.y, arrays.z, arrays.params
    w = inv(z.T @ z / NOBS)
    with no_default_bandwidth_warning():
        cov = IVGMMCovariance(x, y, z, params, w, "kernel", **options)
        c = cov.cov
    assert cov.config["bandwidth"] == NOBS - 2
    score = KernelWeightMatrix(bandwidth=NOBS - 2).weight_matrix(x, z, arrays.eps)
    xpz = x.T @ z / NOBS
    bread = inv(xpz @ w @ xpz.T)
    expected = bread @ (xpz @ w @ score @ w @ xpz.T) @ bread / NOBS
    assert_allclose(c, (expected + expected.T) / 2, rtol=1e-12)


@pytest.mark.parametrize("cov_type", ["robust", "unadjusted", "heteroskedastic"])
def test_gmm_covariance_other_types_are_silent(arrays, cov_type):
    x, y, z, params = arrays.x, arrays.y, arrays.z, arrays.params
    w = inv(z.T @ z / NOBS)
    with no_default_bandwidth_warning():
        cov = IVGMMCovariance(x, y, z, params, w, cov_type).cov
    assert cov.shape == (3, 3)
