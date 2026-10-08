"""
Tests of IVGMMResults.weight_config and IVGMMResults.weight_type

weight_config used to be the name of the weight type, e.g., "robust", instead of
the configuration of the weight estimator.
"""

import os

from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from linearmodels.iv import IVGMM, IVGMMCUE

CWD = os.path.split(os.path.abspath(__file__))[0]
SIMULATED = pd.read_stata(os.path.join(CWD, "results", "simulated-data.dta"))
SIMULATED["const"] = 1.0
NOBS = SIMULATED.shape[0]
CLUSTERS = SIMULATED.cluster_id.to_numpy()


@pytest.fixture(scope="module")
def args():
    data = SIMULATED
    return (
        data.y_robust,
        data[["const", "x3"]],
        data[["x1"]],
        data[["z1", "z2", "x4"]],
    )


# weight_type, options and the configuration that is expected, which does not
# list center and debiased since they are set separately
CONFIGS = [
    ("robust", {}, {}),
    ("unadjusted", {}, {}),
    ("kernel", {"bandwidth": 4}, {"bandwidth": 4, "kernel": "bartlett"}),
    (
        "kernel",
        {"bandwidth": 3, "kernel": "parzen"},
        {"bandwidth": 3, "kernel": "parzen"},
    ),
    # The default bandwidth is nobs - 2
    ("kernel", {}, {"bandwidth": NOBS - 2, "kernel": "bartlett"}),
    ("clustered", {"clusters": CLUSTERS}, {"clusters": CLUSTERS}),
]
MODELS = [IVGMM, IVGMMCUE]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(("center", "debiased"), [(False, False), (True, True)])
@pytest.mark.parametrize(("weight_type", "options", "expected"), CONFIGS)
def test_weight_config(args, model, weight_type, options, expected, center, debiased):
    mod = model(
        *args, weight_type=weight_type, center=center, debiased=debiased, **options
    )
    res = mod.fit() if model is IVGMM else mod.fit(display=False)
    config = res.weight_config
    assert res.weight_type == weight_type
    assert isinstance(config, dict)
    assert set(config) == {"center", "debiased", *expected}
    assert config["center"] is center
    assert config["debiased"] is debiased
    for key, value in expected.items():
        if key == "clusters":
            assert_array_equal(config[key], value)
        else:
            assert config[key] == value


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(("weight_type", "options", "expected"), CONFIGS)
def test_weight_config_round_trip(args, model, weight_type, options, expected):
    # The configuration is enough to create a model with the same weight matrix
    mod = model(*args, weight_type=weight_type, center=True, **options)
    res = mod.fit() if model is IVGMM else mod.fit(display=False)
    again = model(*args, weight_type=res.weight_type, **res.weight_config)
    res2 = again.fit() if model is IVGMM else again.fit(display=False)
    assert_allclose(res2.params, res.params, rtol=1e-10)
    assert_allclose(res2.weight_matrix, res.weight_matrix, rtol=1e-10)
    assert_allclose(res2.j_stat.stat, res.j_stat.stat, rtol=1e-10)


@pytest.mark.parametrize("model", MODELS)
def test_weight_config_reports_the_bandwidth_that_was_used(args, model):
    # When the bandwidth is selected from the data, the configuration holds the
    # value used for the final weight matrix, and not None
    mod = model(*args, weight_type="kernel", optimal_bw=True, center=True)
    res = mod.fit() if model is IVGMM else mod.fit(display=False)
    bandwidth = res.weight_config["bandwidth"]
    assert isinstance(bandwidth, (int, float))
    assert 0 < bandwidth < NOBS - 2
    assert bandwidth == mod._weight.bandwidth


def test_weight_config_bandwidth_reproduces_two_step_estimates(args):
    # In two-step GMM the bandwidth is selected once, so using the reported
    # value reproduces the estimates. This is not true of CUE, where the
    # bandwidth depends on the parameters and is selected at every evaluation
    mod = IVGMM(*args, weight_type="kernel", optimal_bw=True, center=True)
    res = mod.fit()
    bandwidth = res.weight_config["bandwidth"]
    fixed = IVGMM(*args, weight_type="kernel", bandwidth=bandwidth, center=True)
    res_fixed = fixed.fit()
    assert_allclose(res_fixed.params, res.params, rtol=1e-10)
    assert_allclose(res_fixed.weight_matrix, res.weight_matrix, rtol=1e-10)


def test_weight_config_is_not_shared_between_fits(args):
    # Each result holds the configuration at the time that it was estimated
    mod = IVGMM(*args, weight_type="kernel", optimal_bw=True)
    res = mod.fit()
    first = dict(res.weight_config)
    res2 = mod.fit(iter_limit=3)
    assert res.weight_config == first
    assert res2.weight_config is not res.weight_config
