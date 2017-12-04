import numpy as np
import pytest
from numpy.testing import assert_allclose

from linearmodels.iv.covariance import kernel_weight_parzen, kernel_weight_bartlett
from linearmodels.system.covariance import GMMHeteroskedasticCovariance, \
    GMMHomoskedasticCovariance, HeteroskedasticCovariance, HomoskedasticCovariance, \
    KernelCovariance, GMMKernelCovariance
from linearmodels.system.gmm import HomoskedasticWeightMatrix, HeteroskedasticWeightMatrix, \
    KernelWeightMatrix
from linearmodels.system.model import IV3SLS
from linearmodels.tests.system._utility import generate_3sls_data_v2

covs = [HeteroskedasticCovariance, HomoskedasticCovariance]
names = ['Heteroskedastic', 'Homoskedastic']


@pytest.fixture(params=list(zip(covs, names)))
def cov(request):
    eqns = generate_3sls_data_v2(k=3)
    est = request.param[0]
    name = request.param[1]
    sigma = full_sigma = np.eye(3)
    x = [eqns[key].exog for key in eqns]
    n = x[0].shape[0]
    eps = np.random.standard_normal((n, 3))
    return est(x, eps, sigma, full_sigma, gls=True, debiased=True), name


gmm_covs = [GMMHeteroskedasticCovariance, GMMHomoskedasticCovariance]


@pytest.fixture(params=list(zip(gmm_covs, names)))
def gmm_cov(request):
    eqns = generate_3sls_data_v2(k=3)
    est = request.param[0]
    name = request.param[1]
    sigma = np.eye(3)
    x = [eqns[key].exog for key in eqns]
    z = [np.concatenate([eqns[key].exog, eqns[key].instruments], 1) for key in eqns]
    kz = sum(map(lambda a: a.shape[1], z))
    w = np.eye(kz)
    n = x[0].shape[0]
    eps = np.random.standard_normal((n, 3))
    return est(x, z, eps, w, sigma=sigma), name


@pytest.fixture(scope='module')
def cov_data():
    data = generate_3sls_data_v2(k=2)
    mod = IV3SLS(data)
    res = mod.fit(cov_type='unadjusted')
    x = mod._x
    z = mod._z
    eps = res.resids.values
    sigma = res.sigma
    return x, z, eps, sigma


@pytest.fixture(params=[True, False])
def debias(request):
    return request.param


def _xpxi(x):
    """Compute x'x^{-1} from block diagonal x"""
    kx = sum(map(lambda a: a.shape[1], x))
    k = len(x)
    nobs = x[0].shape[0]
    xpx = np.zeros((kx, kx))
    loc = 0
    for i in range(k):
        for j in range(k):
            if i == j:
                kx = x[i].shape[1]
                xpx[loc:loc + kx, loc:loc + kx] = x[i].T @ x[i] / nobs
                loc += kx
    return np.linalg.inv(xpx)


def _xpz(x, z):
    k = len(x)
    nobs = x[0].shape[0]
    xpz = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(x[i].T @ z[i] / nobs)
            else:
                k1, k2 = x[i].shape[1], z[j].shape[1]
                row.append(np.zeros((k1, k2)))
        xpz.append(np.concatenate(row, 1))
    xpz = np.concatenate(xpz, 0)
    return xpz


def _xpz_wi_zpxi(x, z, w):
    xpz = _xpz(x, z)
    out = xpz @ np.linalg.inv(w) @ xpz.T
    out = (out + out.T) / 2
    return np.linalg.inv(out)


def test_str_repr(cov):
    est, name = cov
    assert name in str(est)
    assert name in est.__repr__()
    assert str(hex(id(est))) in est.__repr__()
    assert 'Debiased: True' in str(est)


def test_gmm_str_repr(gmm_cov):
    est, name = gmm_cov
    assert name in str(est)
    assert name in est.__repr__()
    assert str(hex(id(est))) in est.__repr__()
    assert 'GMM' in str(est)


def test_homoskedastic_direct(cov_data, debias):
    x, z, eps, sigma = cov_data
    cov = HomoskedasticCovariance(x, eps, sigma, sigma, debiased=debias)
    k = len(x)
    nobs = x[0].shape[0]
    big_x = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(x[i])
            else:
                row.append(np.zeros((nobs, x[j].shape[1])))
        big_x.append(np.concatenate(row, 1))
    big_x = np.concatenate(big_x, 0)
    xeex = big_x.T @ np.kron(sigma, np.eye(nobs)) @ big_x / nobs
    xpxi = _xpxi(x)
    direct = xpxi @ xeex @ xpxi / nobs
    direct = (direct + direct.T) / 2
    assert_allclose(np.diag(direct), np.diag(cov.cov))
    s = np.sqrt(np.diag(direct))[:, None]
    r_direct = direct / (s @ s.T)
    s = np.sqrt(np.diag(cov.cov))[:, None]
    c_direct = direct / (s @ s.T)
    assert_allclose(r_direct, c_direct, atol=1e-5)


def test_heteroskedastic_direct(cov_data, debias):
    x, z, eps, sigma = cov_data
    cov = HeteroskedasticCovariance(x, eps, sigma, sigma, debiased=debias)
    k = len(x)
    xe = [x[i] * eps[:, i:i + 1] for i in range(k)]
    xe = np.concatenate(xe, 1)
    nobs = xe.shape[0]
    xeex = np.zeros((xe.shape[1], xe.shape[1]))
    xeex += xe.T @ xe / nobs
    xpxi = _xpxi(x)
    direct = xpxi @ xeex @ xpxi / nobs
    direct = (direct + direct.T) / 2
    if debias:
        df = [x[i].shape[1] * np.ones(x[i].shape[1]) for i in range(k)]
        df = np.concatenate(df)[:, None]
        adj = nobs / (nobs - np.sqrt(df) @ np.sqrt(df).T)
        direct *= adj
    direct = (direct + direct.T) / 2
    assert_allclose(np.diag(direct), np.diag(cov.cov))
    s = np.sqrt(np.diag(direct))[:, None]
    r_direct = direct / (s @ s.T)
    s = np.sqrt(np.diag(cov.cov))[:, None]
    c_direct = direct / (s @ s.T)
    assert_allclose(r_direct, c_direct, atol=1e-5)


def test_kernel_direct(cov_data, debias):
    x, z, eps, sigma = cov_data
    k = len(x)
    bandwidth = 12
    cov = KernelCovariance(x, eps, sigma, sigma, gls=False, debiased=debias,
                           kernel='parzen', bandwidth=bandwidth)
    assert cov.bandwidth == 12
    xe = [x[i] * eps[:, i:i + 1] for i in range(k)]
    xe = np.concatenate(xe, 1)
    w = kernel_weight_parzen(12)
    nobs = xe.shape[0]
    xeex = np.zeros((xe.shape[1], xe.shape[1]))
    xeex += xe.T @ xe / nobs
    for i in range(1, bandwidth + 1):
        op = xe[:-i].T @ xe[i:] / nobs
        xeex += w[i] * (op + op.T)
    xpxi = _xpxi(x)
    direct = xpxi @ xeex @ xpxi / nobs
    direct = (direct + direct.T) / 2
    if debias:
        df = [x[i].shape[1] * np.ones(x[i].shape[1]) for i in range(k)]
        df = np.concatenate(df)[:, None]
        adj = nobs / (nobs - np.sqrt(df) @ np.sqrt(df).T)
        direct *= adj
    direct = (direct + direct.T) / 2
    assert_allclose(np.diag(direct), np.diag(cov.cov))
    s = np.sqrt(np.diag(direct))[:, None]
    r_direct = direct / (s @ s.T)
    s = np.sqrt(np.diag(cov.cov))[:, None]
    c_direct = direct / (s @ s.T)
    assert_allclose(r_direct, c_direct, atol=1e-5)


def test_gmm_homoskedastic_direct(cov_data, debias):
    x, z, eps, sigma = cov_data
    k = len(x)
    nobs = x[0].shape[0]
    wm = HomoskedasticWeightMatrix()
    w = wm.weight_matrix(x, z, eps, sigma=sigma)
    cov_est = GMMHomoskedasticCovariance(x, z, eps, w, sigma=sigma, debiased=debias)
    xpz_wi_zpxi = _xpz_wi_zpxi(x, z, w)
    xpz = _xpz(x, z)
    wi = np.linalg.inv(w)
    xpz_wi = xpz @ wi

    big_z = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(z[i])
            else:
                row.append(np.zeros((nobs, z[j].shape[1])))
        big_z.append(np.concatenate(row, 1))
    big_z = np.concatenate(big_z, 0)
    zeez = big_z.T @ np.kron(sigma, np.eye(nobs)) @ big_z / nobs
    assert_allclose(zeez, cov_est._omega())

    direct = xpz_wi_zpxi @ (xpz_wi @ zeez @ xpz_wi.T) @ xpz_wi_zpxi / nobs
    direct = (direct + direct.T) / 2
    if debias:
        df = [vx.shape[1] * np.ones(vx.shape[1]) for vx in x]
        df = np.concatenate(df)[:, None]
        df = np.sqrt(df)
        adj = nobs / (nobs - df @ df.T)
        direct *= adj
    direct = (direct + direct.T) / 2
    assert_allclose(direct, cov_est.cov)


def test_gmm_heterosedastic_direct(cov_data, debias):
    x, z, eps, sigma = cov_data
    k = len(x)
    nobs = x[0].shape[0]
    wm = HeteroskedasticWeightMatrix()
    w = wm.weight_matrix(x, z, eps, sigma=sigma)
    cov_est = GMMHeteroskedasticCovariance(x, z, eps, w, sigma=sigma, debiased=debias)
    xpz_wi_zpxi = _xpz_wi_zpxi(x, z, w)
    xpz = _xpz(x, z)
    wi = np.linalg.inv(w)
    xpz_wi = xpz @ wi
    ze = [z[i] * eps[:, i:i + 1] for i in range(k)]
    ze = np.concatenate(ze, 1)
    zeez = ze.T @ ze / nobs
    assert_allclose(zeez, cov_est._omega())

    direct = xpz_wi_zpxi @ (xpz_wi @ zeez @ xpz_wi.T) @ xpz_wi_zpxi / nobs
    direct = (direct + direct.T) / 2
    if debias:
        df = [vx.shape[1] * np.ones(vx.shape[1]) for vx in x]
        df = np.concatenate(df)[:, None]
        df = np.sqrt(df)
        adj = nobs / (nobs - df @ df.T)
        direct *= adj
    direct = (direct + direct.T) / 2
    assert_allclose(direct, cov_est.cov)


def test_gmm_kernel_direct(cov_data):
    x, z, eps, sigma = cov_data
    bandwidth = 12
    k = len(x)
    nobs = x[0].shape[0]
    wm = KernelWeightMatrix(kernel='bartlett', bandwidth=bandwidth)
    w = wm.weight_matrix(x, z, eps, sigma=sigma)
    cov_est = GMMKernelCovariance(x, z, eps, w, sigma=sigma, debiased=debias, kernel='bartlett',
                                  bandwidth=bandwidth)

    xpz_wi_zpxi = _xpz_wi_zpxi(x, z, w)
    xpz = _xpz(x, z)
    wi = np.linalg.inv(w)
    xpz_wi = xpz @ wi
    ze = [z[i] * eps[:, i:i + 1] for i in range(k)]
    ze = np.concatenate(ze, 1)
    zeez = ze.T @ ze / nobs
    w = kernel_weight_bartlett(bandwidth)
    for i in range(1, bandwidth + 1):
        op = ze[:-i].T @ ze[i:] / nobs
        zeez += w[i] * (op + op.T)
    assert_allclose(zeez, cov_est._omega())

    direct = xpz_wi_zpxi @ (xpz_wi @ zeez @ xpz_wi.T) @ xpz_wi_zpxi / nobs
    direct = (direct + direct.T) / 2
    if debias:
        df = [vx.shape[1] * np.ones(vx.shape[1]) for vx in x]
        df = np.concatenate(df)[:, None]
        df = np.sqrt(df)
        adj = nobs / (nobs - df @ df.T)
        direct *= adj
    direct = (direct + direct.T) / 2
    assert_allclose(direct, cov_est.cov)
