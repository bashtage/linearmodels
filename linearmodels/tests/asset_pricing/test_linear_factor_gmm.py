import numpy as np
import pytest
from numpy.testing import assert_allclose

from linearmodels.asset_pricing.model import LinearFactorModelGMM
from linearmodels.tests.asset_pricing._utility import generate_data, get_all


@pytest.fixture(params=['numpy', 'pandas'])
def data(request):
    return generate_data(nportfolio=10, output=request.param)


def test_linear_model_gmm_moments_jacobian(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type='robust', disp=0, debiased=False)
    params = np.r_[res.betas.values.ravel(),
                   res.risk_premia.values.ravel(),
                   mod.factors.ndarray.mean(0)]
    mod_mom = mod._moments(params[:, None], True)

    mom = []
    p = mod.portfolios.ndarray
    f = mod.factors.ndarray
    n = f.shape[0]
    fc = np.c_[np.ones((n, 1)), f]
    mu = f.mean(0)[None, :]
    lam = res.risk_premia.values[None, :]
    x = f - mu + lam
    b = res.betas.values
    for i in range(p.shape[1]):
        eps = p[:, i:(i + 1)] - x @ b[[i]].T
        for j in range(fc.shape[1]):
            mom.append(eps * fc[:, [j]])
    mom.append(f - mu)
    mom = np.hstack(tuple(mom))

    mod_jac = mod._jacobian(params, True)
    jac = np.zeros((mom.shape[1], params.shape[0]))
    nport, nf = p.shape[1], f.shape[1]
    # 1,1
    jac[:(nport * (nf + 1)), :nport * nf] = np.kron(np.eye(nport), fc.T @ x / n)
    # 1, 2
    col = []
    for i in range(nport):
        col.append(fc.T @ np.ones((n, 1)) @ b[[i]] / n)
    col = np.vstack(tuple(col))
    jac[:(nport * (nf + 1)), nport * nf:nport * nf + nf] = col
    # 1, 3
    col = []
    for i in range(nport):
        col.append(-fc.T @ np.ones((n, 1)) @ b[[i]] / n)
    col = np.vstack(tuple(col))
    jac[:(nport * (nf + 1)), -nf:] = col
    # 2,2
    jac[-nf:, -nf:] = np.eye(nf)

    assert_allclose(mom, mod_mom)
    assert_allclose(jac, mod_jac)

    me = mom - mom.mean(0)[None, :]
    s = me.T @ me / n
    s = (s + s.T) / 2
    cov = np.linalg.inv(jac.T @ np.linalg.inv(s) @ jac) / n
    cov = (cov + cov.T) / 2
    assert_allclose(np.diag(cov), np.diag(res.cov), rtol=5e-3)
    get_all(res)


def test_linear_model_gmm_smoke_iterate(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type='robust', disp=5, steps=20)
    get_all(res)


def test_linear_model_gmm_smoke_risk_free(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type='robust', disp=10)
    get_all(res)
    str(res._cov_est)
    res._cov_est.__repr__()
    str(res._cov_est.config)


def test_linear_model_gmm_kernel_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type='kernel', disp=10)
    get_all(res)
    str(res._cov_est)
    res._cov_est.__repr__()
    str(res._cov_est.config)


def test_linear_model_gmm_kernel_bandwidth_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type='kernel', bandwidth=10, disp=10)
    get_all(res)


def test_linear_model_gmm_cue_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type='robust', disp=10, use_cue=True)
    get_all(res)
