import numpy as np
import pytest
from linearmodels.system.covariance import HeteroskedasticCovariance, HomoskedasticCovariance, \
    GMMHeteroskedasticCovariance, GMMHomoskedasticCovariance
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
    sigma = full_sigma = np.eye(3)
    x = [eqns[key].exog for key in eqns]
    z = [np.concatenate([eqns[key].exog, eqns[key].instruments], 1) for key in eqns]
    kz = sum(map(lambda a: a.shape[1], z))
    w = np.eye(kz)
    n = x[0].shape[0]
    eps = np.random.standard_normal((n, 3))
    return est(x, z, eps, w, sigma=sigma), name


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
