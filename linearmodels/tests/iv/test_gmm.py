import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from linearmodels.iv.covariance import (
    kernel_weight_bartlett,
    kernel_weight_parzen,
    kernel_weight_quadratic_spectral,
)
from linearmodels.iv.gmm import (
    HeteroskedasticWeightMatrix,
    HomoskedasticWeightMatrix,
    IVGMMCovariance,
    KernelWeightMatrix,
    OneWayClusteredWeightMatrix,
)
from linearmodels.shared.utility import AttrDict
from linearmodels.tests.iv._utility import generate_data


@pytest.fixture(params=[None, 12], scope="module")
def bandwidth(request):
    return request.param


@pytest.fixture(params=["bartlett", "qs", "parzen"], scope="module")
def kernel(request):
    kernel_name = request.param
    if kernel_name == "bartlett":
        weight_func = kernel_weight_bartlett
        alt_names = ["newey-west"]
    elif kernel_name == "parzen":
        weight_func = kernel_weight_parzen
        alt_names = ["gallant"]
    else:
        weight_func = kernel_weight_quadratic_spectral
        alt_names = ["quadratic-spectral", "andrews"]
    return AttrDict(kernel=kernel_name, alt_names=alt_names, weight=weight_func)


@pytest.fixture(scope="module")
def data():
    return generate_data()


def test_homoskedastic_center(data):
    wm = HomoskedasticWeightMatrix(True)
    weight = wm.weight_matrix(data.x, data.z, data.e)
    z, e, nobs = data.z, data.e, data.nobs
    s2 = (e - e.mean()).T @ (e - e.mean()) / nobs
    assert_allclose(weight, s2 * z.T @ z / nobs)


def test_homoskedastic_debiased(data):
    wm = HomoskedasticWeightMatrix(debiased=True)
    weight = wm.weight_matrix(data.x, data.z, data.e)
    z, e, nobs, nvar = data.z, data.e, data.nobs, data.nvar
    s2 = (e - e.mean()).T @ (e - e.mean()) / nobs
    scale = nobs / (nobs - nvar)
    assert_allclose(weight, scale * s2 * z.T @ z / nobs)


def test_homoskedastic_defaults(data):
    wm = HomoskedasticWeightMatrix()
    z, e, nobs = data.z, data.e, data.nobs
    weight = wm.weight_matrix(data.x, z, e)
    s2 = (e - e.mean()).T @ (e - e.mean()) / nobs
    assert_allclose(weight, s2 * z.T @ z / nobs)


def test_homoskedastic_config(data):
    wm = HomoskedasticWeightMatrix()
    z, e, nobs = data.z, data.e, data.nobs
    weight = wm.weight_matrix(data.x, z, e)
    s2 = (e - e.mean()).T @ (e - e.mean()) / nobs
    assert_allclose(weight, s2 * z.T @ z / nobs)
    assert wm.config["center"] is False
    assert wm.config["debiased"] is False


def test_heteroskedastic_center(data):
    wm = HeteroskedasticWeightMatrix(True)
    z, e, nobs = data.z, data.e, data.nobs
    weight = wm.weight_matrix(data.x, z, e)
    ze = z * e
    ze -= ze.mean(0)
    assert_allclose(weight, ze.T @ ze / nobs)


def test_heteroskedastic_debiased(data):
    wm = HeteroskedasticWeightMatrix(debiased=True)
    z, e, nobs, nvar = data.z, data.e, data.nobs, data.nvar
    weight = wm.weight_matrix(data.x, z, e)
    ze = z * e
    scale = nobs / (nobs - nvar)
    assert_allclose(weight, scale * ze.T @ ze / nobs)


def test_heteroskedastic_config(data):
    wm = HeteroskedasticWeightMatrix()
    z, e, nobs = data.z, data.e, data.nobs
    weight = wm.weight_matrix(data.x, z, e)
    ze = z * e

    assert_allclose(weight, ze.T @ ze / nobs)
    assert wm.config["center"] is False
    assert wm.config["debiased"] is False


def test_kernel_weight_center(data, kernel, bandwidth):
    wm = KernelWeightMatrix(kernel.kernel, bandwidth, center=True, optimal_bw=True)
    weight = wm.weight_matrix(data.x, data.z, data.e)
    z, e, nobs = data.z, data.e, data.nobs
    bw = bandwidth or wm.bandwidth
    w = kernel.weight(bw, nobs - 1)
    ze = z * e
    ze = ze - ze.mean(0)
    s = ze.T @ ze
    for i in range(1, len(w)):
        op = ze[i:].T @ ze[:-i]
        s += w[i] * (op + op.T)
    assert_allclose(weight, s / nobs)
    assert wm.config["bandwidth"] == bw
    assert wm.config["kernel"] == kernel.kernel
    for name in kernel.alt_names:
        wm = KernelWeightMatrix(name, bandwidth, center=True, optimal_bw=True)
        weight2 = wm.weight_matrix(data.x, data.z, data.e)
        assert_allclose(weight, weight2, rtol=1e-12)


def test_kernel_weight_debiased(kernel, data, bandwidth):
    wm = KernelWeightMatrix(debiased=True, kernel=kernel.kernel, bandwidth=bandwidth)
    weight = wm.weight_matrix(data.x, data.z, data.e)
    z, e, nobs, nvar = data.z, data.e, data.nobs, data.nvar
    bw = bandwidth or wm.bandwidth
    w = kernel.weight(bw, nobs - 1)
    ze = z * e
    s = ze.T @ ze
    for i in range(1, len(w)):
        op = ze[i:].T @ ze[:-i]
        s += w[i] * (op + op.T)
    assert_allclose(weight, s / (nobs - nvar))
    assert wm.config["bandwidth"] == bw
    assert wm.config["kernel"] == kernel.kernel


def test_kernel_weight_config(data, kernel, bandwidth):
    wm = KernelWeightMatrix(kernel=kernel.kernel, bandwidth=bandwidth)
    weight = wm.weight_matrix(data.x, data.z, data.e)
    z, e, nobs = data.z, data.e, data.nobs
    bw = bandwidth or wm.bandwidth
    w = kernel.weight(bw, nobs - 1)
    ze = z * e
    s = ze.T @ ze
    for i in range(1, len(w)):
        op = ze[i:].T @ ze[:-i]
        s += w[i] * (op + op.T)
    assert_allclose(weight, s / nobs)
    assert wm.config["center"] is False
    assert wm.config["debiased"] is False
    assert wm.config["bandwidth"] == bw
    assert wm.config["kernel"] == kernel.kernel

    for name in kernel.alt_names:
        wm = KernelWeightMatrix(kernel=name, bandwidth=bandwidth)
        weight2 = wm.weight_matrix(data.x, data.z, data.e)
        assert_allclose(weight, weight2, rtol=1e-12)


def test_cluster_weight__center(data):
    wm = OneWayClusteredWeightMatrix(data.clusters, True)
    weight = wm.weight_matrix(data.x, data.z, data.e)
    ze = data.z * data.e
    ze -= ze.mean(0)
    uc = np.unique(data.clusters)
    s = np.zeros((ze.shape[1], ze.shape[1]))
    for val in uc:
        obs = ze[data.clusters == val]
        total = obs.sum(0)[:, None]
        s += total @ total.T
    assert_allclose(weight, s / data.nobs)


def test_cluster_weight__debiased(data):
    wm = OneWayClusteredWeightMatrix(data.clusters, debiased=True)
    weight = wm.weight_matrix(data.x, data.z, data.e)
    ze = data.z * data.e
    uc = np.unique(data.clusters)
    s = np.zeros((ze.shape[1], ze.shape[1]))
    for val in uc:
        obs = ze[data.clusters == val]
        total = obs.sum(0)[:, None]
        s += total @ total.T
    nobs, nvar = data.nobs, data.nvar
    groups = len(uc)
    scale = (nobs - 1) / (nobs - nvar) * groups / (groups - 1)
    assert_allclose(weight, scale * s / data.nobs)


def test_cluster_weight__config(data):
    wm = OneWayClusteredWeightMatrix(data.clusters)
    assert wm.config["center"] is False
    assert wm.config["debiased"] is False
    assert_equal(wm.config["clusters"], data.clusters)


def test_cluster_weight__errors(data):
    wm = OneWayClusteredWeightMatrix(data.clusters[:10])
    with pytest.raises(ValueError):
        wm.weight_matrix(data.x, data.z, data.e)


def test_gmm_covariance_homoskedastic(data):
    c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, "unadjusted")
    s = HomoskedasticWeightMatrix().weight_matrix(data.x, data.z, data.e)
    x, z = data.x, data.z
    xzwswzx = x.T @ z @ s @ z.T @ x / data.nobs
    cov = data.xzizx_inv @ xzwswzx @ data.xzizx_inv
    cov = (cov + cov.T) / 2
    assert_allclose(c.cov, cov)
    assert c.config["debiased"] is False


def test_gmm_covariance_heteroskedastic(data):
    c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, "robust")
    s = HeteroskedasticWeightMatrix().weight_matrix(data.x, data.z, data.e)
    x, z = data.x, data.z
    xzwswzx = x.T @ z @ s @ z.T @ x / data.nobs
    cov = data.xzizx_inv @ xzwswzx @ data.xzizx_inv
    cov = (cov + cov.T) / 2
    assert_allclose(c.cov, cov)
    assert c.config["debiased"] is False


def test_gmm_covariance_clustered(data):
    c = IVGMMCovariance(
        data.x,
        data.y,
        data.z,
        data.params,
        data.i,
        "clustered",
        clusters=data.clusters,
    )
    s = OneWayClusteredWeightMatrix(clusters=data.clusters).weight_matrix(
        data.x, data.z, data.e
    )
    x, z = data.x, data.z
    xzwswzx = x.T @ z @ s @ z.T @ x / data.nobs
    cov = data.xzizx_inv @ xzwswzx @ data.xzizx_inv
    cov = (cov + cov.T) / 2
    assert_allclose(c.cov, cov)
    assert c.config["debiased"] is False
    assert_equal(c.config["clusters"], data.clusters)
    c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, "clustered")
    assert "Clustered" in str(c)


def test_gmm_covariance_kernel(data, kernel, bandwidth):
    c = IVGMMCovariance(
        data.x,
        data.y,
        data.z,
        data.params,
        data.i,
        "kernel",
        kernel=kernel.kernel,
        bandwidth=bandwidth,
    )
    s = KernelWeightMatrix(kernel=kernel.kernel, bandwidth=bandwidth).weight_matrix(
        data.x, data.z, data.e
    )
    x, z, nobs = data.x, data.z, data.nobs
    xzwswzx = x.T @ z @ s @ z.T @ x / data.nobs
    cov = data.xzizx_inv @ xzwswzx @ data.xzizx_inv
    cov = (cov + cov.T) / 2
    assert_allclose(c.cov, cov)
    assert c.config["kernel"] == kernel.kernel
    assert c.config["debiased"] is False
    assert c.config["bandwidth"] == bandwidth or nobs - 2
    c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, "kernel")
    assert "Kernel" in str(c)


def test_gmm_covariance_unknown(data):
    with pytest.raises(ValueError):
        assert isinstance(
            IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, "unknown").cov,
            np.ndarray,
        )
