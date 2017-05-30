import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from linearmodels.iv.covariance import (kernel_weight_bartlett,
                                        kernel_weight_parzen,
                                        kernel_weight_quadratic_spectral)
from linearmodels.iv.gmm import (HeteroskedasticWeightMatrix,
                                 HomoskedasticWeightMatrix, IVGMMCovariance,
                                 KernelWeightMatrix,
                                 OneWayClusteredWeightMatrix)
from linearmodels.tests.iv._utility import generate_data
from linearmodels.utility import AttrDict


@pytest.fixture(params=[None, 12], scope='module')
def bandwidth(request):
    return request.param


@pytest.fixture(params=['bartlett', 'qs', 'parzen'], scope='module')
def kernel(request):
    kernel_name = request.param
    if kernel_name == 'bartlett':
        weight_func = kernel_weight_bartlett
        alt_names = ['newey-west']
    elif kernel_name == 'parzen':
        weight_func = kernel_weight_parzen
        alt_names = ['gallant']
    else:
        weight_func = kernel_weight_quadratic_spectral
        alt_names = ['quadratic-spectral', 'andrews']
    return AttrDict(kernel=kernel_name, alt_names=alt_names,
                    weight=weight_func)


@pytest.fixture(scope='module')
def data():
    return generate_data()


class TestHomoskedasticWeight(object):
    def test_center(self, data):
        wm = HomoskedasticWeightMatrix(True)
        weight = wm.weight_matrix(data.x, data.z, data.e)
        z, e, nobs = data.z, data.e, data.nobs
        s2 = (e - e.mean()).T @ (e - e.mean()) / nobs
        assert_allclose(weight, s2 * z.T @ z / nobs)

    def test_debiased(self, data):
        wm = HomoskedasticWeightMatrix(debiased=True)
        weight = wm.weight_matrix(data.x, data.z, data.e)
        z, e, nobs, nvar = data.z, data.e, data.nobs, data.nvar
        s2 = (e - e.mean()).T @ (e - e.mean()) / nobs
        scale = nobs / (nobs - nvar)
        assert_allclose(weight, scale * s2 * z.T @ z / nobs)

    def test_defaults(self, data):
        wm = HomoskedasticWeightMatrix()
        z, e, nobs = data.z, data.e, data.nobs
        weight = wm.weight_matrix(data.x, z, e)
        s2 = (e - e.mean()).T @ (e - e.mean()) / nobs
        assert_allclose(weight, s2 * z.T @ z / nobs)

    def test_config(self, data):
        wm = HomoskedasticWeightMatrix()
        z, e, nobs = data.z, data.e, data.nobs
        weight = wm.weight_matrix(data.x, z, e)
        s2 = (e - e.mean()).T @ (e - e.mean()) / nobs
        assert_allclose(weight, s2 * z.T @ z / nobs)
        assert wm.config['center'] is False
        assert wm.config['debiased'] is False


class TestHeteroskedasticWeight(object):
    def test_center(self, data):
        wm = HeteroskedasticWeightMatrix(True)
        z, e, nobs = data.z, data.e, data.nobs
        weight = wm.weight_matrix(data.x, z, e)
        ze = z * e
        ze -= ze.mean(0)
        assert_allclose(weight, ze.T @ ze / nobs)

    def test_debiased(self, data):
        wm = HeteroskedasticWeightMatrix(debiased=True)
        z, e, nobs, nvar = data.z, data.e, data.nobs, data.nvar
        weight = wm.weight_matrix(data.x, z, e)
        ze = z * e
        scale = nobs / (nobs - nvar)
        assert_allclose(weight, scale * ze.T @ ze / nobs)

    def test_config(self, data):
        wm = HeteroskedasticWeightMatrix()
        z, e, nobs = data.z, data.e, data.nobs
        weight = wm.weight_matrix(data.x, z, e)
        ze = z * e

        assert_allclose(weight, ze.T @ ze / nobs)
        assert wm.config['center'] is False
        assert wm.config['debiased'] is False


class TestKernelWeight(object):
    def test_center(self, data, kernel, bandwidth):
        wm = KernelWeightMatrix(kernel.kernel, bandwidth, True)
        weight = wm.weight_matrix(data.x, data.z, data.e)
        z, e, nobs = data.z, data.e, data.nobs
        bw = bandwidth or nobs - 2
        w = kernel.weight(bw, nobs - 1)
        ze = z * e
        ze = ze - ze.mean(0)
        s = ze.T @ ze
        for i in range(1, len(w)):
            op = ze[i:].T @ ze[:-i]
            s += w[i] * (op + op.T)
        assert_allclose(weight, s / nobs)
        assert wm.config['bandwidth'] == bw
        assert wm.config['kernel'] == kernel.kernel
        for name in kernel.alt_names:
            wm = KernelWeightMatrix(name, bandwidth, True)
            weight2 = wm.weight_matrix(data.x, data.z, data.e)
            assert_equal(weight, weight2)

    def test_debiased(self, kernel, data, bandwidth):
        wm = KernelWeightMatrix(debiased=True, kernel=kernel.kernel, bandwidth=bandwidth)
        weight = wm.weight_matrix(data.x, data.z, data.e)
        z, e, nobs, nvar = data.z, data.e, data.nobs, data.nvar
        bw = bandwidth or nobs - 2
        w = kernel.weight(bw, nobs - 1)
        ze = z * e
        s = ze.T @ ze
        for i in range(1, len(w)):
            op = ze[i:].T @ ze[:-i]
            s += w[i] * (op + op.T)
        assert_allclose(weight, s / (nobs - nvar))
        assert wm.config['bandwidth'] == bw
        assert wm.config['kernel'] == kernel.kernel

    def test_config(self, data, kernel, bandwidth):
        wm = KernelWeightMatrix(kernel=kernel.kernel, bandwidth=bandwidth)
        weight = wm.weight_matrix(data.x, data.z, data.e)
        z, e, nobs = data.z, data.e, data.nobs
        bw = bandwidth or nobs - 2
        w = kernel.weight(bw, nobs - 1)
        ze = z * e
        s = ze.T @ ze
        for i in range(1, len(w)):
            op = ze[i:].T @ ze[:-i]
            s += w[i] * (op + op.T)
        assert_allclose(weight, s / nobs)
        assert wm.config['center'] is False
        assert wm.config['debiased'] is False
        assert wm.config['bandwidth'] == bw
        assert wm.config['kernel'] == kernel.kernel

        for name in kernel.alt_names:
            wm = KernelWeightMatrix(kernel=name, bandwidth=bandwidth)
            weight2 = wm.weight_matrix(data.x, data.z, data.e)
            assert_equal(weight, weight2)


class TestClusterWeight(object):
    def test_center(self, data):
        wm = OneWayClusteredWeightMatrix(data.clusters, True)
        weight = wm.weight_matrix(data.x, data.z, data.e)
        ze = data.z * data.e
        ze -= ze.mean(0)
        uc = np.unique(data.clusters)
        s = np.zeros((ze.shape[1], ze.shape[1]))
        for val in uc:
            obs = ze[data.clusters == val]
            sum = obs.sum(0)[:, None]
            s += sum @ sum.T
        assert_allclose(weight, s / data.nobs)

    def test_debiased(self, data):
        wm = OneWayClusteredWeightMatrix(data.clusters, debiased=True)
        weight = wm.weight_matrix(data.x, data.z, data.e)
        ze = data.z * data.e
        uc = np.unique(data.clusters)
        s = np.zeros((ze.shape[1], ze.shape[1]))
        for val in uc:
            obs = ze[data.clusters == val]
            sum = obs.sum(0)[:, None]
            s += sum @ sum.T
        nobs, nvar = data.nobs, data.nvar
        groups = len(uc)
        scale = (nobs - 1) / (nobs - nvar) * groups / (groups - 1)
        assert_allclose(weight, scale * s / data.nobs)

    def test_config(self, data):
        wm = OneWayClusteredWeightMatrix(data.clusters)
        assert wm.config['center'] is False
        assert wm.config['debiased'] is False
        assert_equal(wm.config['clusters'], data.clusters)

    def test_errors(self, data):
        wm = OneWayClusteredWeightMatrix(data.clusters[:10])
        with pytest.raises(ValueError):
            wm.weight_matrix(data.x, data.z, data.e)


class TestGMMCovariance(object):
    def test_homoskedastic(self, data):
        c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, 'unadjusted')
        s = HomoskedasticWeightMatrix().weight_matrix(data.x, data.z, data.e)
        x, z = data.x, data.z
        xzwswzx = x.T @ z @ s @ z.T @ x / data.nobs
        cov = data.xzizx_inv @ xzwswzx @ data.xzizx_inv
        cov = (cov + cov.T) / 2
        assert_allclose(c.cov, cov)
        assert c.config['debiased'] is False

    def test_heteroskedastic(self, data):
        c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, 'robust')
        s = HeteroskedasticWeightMatrix().weight_matrix(data.x, data.z, data.e)
        x, z = data.x, data.z
        xzwswzx = x.T @ z @ s @ z.T @ x / data.nobs
        cov = data.xzizx_inv @ xzwswzx @ data.xzizx_inv
        cov = (cov + cov.T) / 2
        assert_allclose(c.cov, cov)
        assert c.config['debiased'] is False

    def test_clustered(self, data):
        c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, 'clustered',
                            clusters=data.clusters)
        s = OneWayClusteredWeightMatrix(clusters=data.clusters).weight_matrix(data.x, data.z,
                                                                              data.e)
        x, z = data.x, data.z
        xzwswzx = x.T @ z @ s @ z.T @ x / data.nobs
        cov = data.xzizx_inv @ xzwswzx @ data.xzizx_inv
        cov = (cov + cov.T) / 2
        assert_allclose(c.cov, cov)
        assert c.config['debiased'] is False
        assert_equal(c.config['clusters'], data.clusters)
        c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, 'clustered')
        assert 'Clustered' in str(c)

    def test_kernel(self, data, kernel, bandwidth):
        c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, 'kernel',
                            kernel=kernel.kernel, bandwidth=bandwidth)
        s = KernelWeightMatrix(kernel=kernel.kernel, bandwidth=bandwidth).weight_matrix(data.x,
                                                                                        data.z,
                                                                                        data.e)
        x, z, nobs = data.x, data.z, data.nobs
        xzwswzx = x.T @ z @ s @ z.T @ x / data.nobs
        cov = data.xzizx_inv @ xzwswzx @ data.xzizx_inv
        cov = (cov + cov.T) / 2
        assert_allclose(c.cov, cov)
        assert c.config['kernel'] == kernel.kernel
        assert c.config['debiased'] is False
        assert c.config['bandwidth'] == bandwidth or nobs - 2
        c = IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, 'kernel')
        assert 'Kernel' in str(c)

    def test_unknown(self, data):
        with pytest.raises(ValueError):
            IVGMMCovariance(data.x, data.y, data.z, data.params, data.i, 'unknown').cov
