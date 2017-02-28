import numpy as np
import pytest
from numpy import sin, cos, pi
from numpy.linalg import inv
from numpy.testing import assert_allclose, assert_equal

from panel.iv.covariance import HomoskedasticCovariance, kernel_weight_bartlett, kernel_weight_parzen, \
    kernel_weight_quadratic_spectral


@pytest.fixture(scope='module')
def data():
    n, k, p = 1000, 5, 3
    np.random.seed(12345)
    rho = 0.5
    r = np.zeros((k + p + 1, k + p + 1))
    r.fill(rho)
    r[-1, 2:] = 0
    r[2:, -1] = 0
    r[-1, -1] = 0.5
    r += np.eye(9) * 0.5
    v = np.random.multivariate_normal(np.zeros(r.shape[0]), r, n)
    x = v[:, :k]
    z = v[:, 2:k + p]
    e = v[:, [-1]]
    params = np.arange(1, k + 1) / k
    params = params[:, None]
    y = x @ params + e
    xhat = z @ np.linalg.pinv(z) @ x
    nobs = x.shape[0]
    return nobs, e, x, y, z, xhat, params


def test_kernel_bartlett():
    w = kernel_weight_bartlett(0)
    assert_equal(w,1 - np.arange(1))
    w = kernel_weight_bartlett(10)
    assert_equal(w, 1 - np.arange(11)/11)


def test_kernel_parzen():
    def w(k):
        w = np.empty(k+1)
        for i in range(k+1):
            z = i / (k+1)
            if z > 0.5:
                w[i] = 2 * (1 - z ** 3)
            else:
                w[i] = 1 - 6 * z ** 2 + 6 * z ** 3
        return w

    assert_equal(w(0), kernel_weight_parzen(0))
    assert_equal(w(1), kernel_weight_parzen(1))
    assert_equal(w(10), kernel_weight_parzen(10))

def test_kernel_qs():
    def w(k, n):
        w = np.empty(n+1)
        w[0] = 1
        for i in range(1, n+1):
            q = 6*pi*(i/k)/5
            w[i] = 3/q**2 * (sin(q)/q-cos(q))
        return w

    assert_equal(w(10, 0), kernel_weight_quadratic_spectral(10,0))
    assert_equal(w(2, 0), kernel_weight_quadratic_spectral(2,0))
    assert_equal(w(10,200), kernel_weight_quadratic_spectral(10,200))
    assert_equal(w(2.5, 200), kernel_weight_quadratic_spectral(2.5, 200))



class TestHomoskedasticCovariance(object):
    def test_asymptotic(self, data):
        nobs, eps, x, y, z, xhat, params = data
        s2 = eps.T @ eps / nobs
        c = HomoskedasticCovariance(x, y, z, params)
        assert c._kappa == 1
        assert c.debiased == False
        assert c.config == {'debiased': False, 'name': 'HomoskedasticCovariance', }
        assert_allclose(c.cov, s2 * inv(xhat.T @ xhat / nobs) / nobs)
        assert c.s2 == s2
        assert_allclose(c.s, s2 * (xhat.T @ xhat / nobs))

    def test_debiased(self, data):
        nobs, eps, x, y, z, xhat, params = data
        c = HomoskedasticCovariance(x, y, z, params, True)
        assert c._kappa == 1
        assert c.debiased == True
        assert c.config == {'debiased': True, 'name': 'HomoskedasticCovariance', }
        # TODO

    def test_kappa(self, data):
        nobs, eps, x, y, z, xhat, params = data
        c = HomoskedasticCovariance(x, y, z, params, True, .99)
        assert c._kappa == 0.99
        assert c.debiased == True
        assert c.config == {'debiased': True, 'name': 'HomoskedasticCovariance', }
        # TODO

    def test_kappa_debiased(self, data):
        nobs, eps, x, y, z, xhat, params = data
        c = HomoskedasticCovariance(x, y, z, params, False, .99)
        assert c._kappa == 0.99
        assert c.debiased == False
        assert c.config == {'debiased': False, 'name': 'HomoskedasticCovariance', }
        # TODO
