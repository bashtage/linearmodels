import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.asset_pricing.model import LinearFactorModelGMM
from linearmodels.tests.asset_pricing._utility import generate_data, get_all


@pytest.fixture(params=["numpy", "pandas"])
def data(request):
    return generate_data(nportfolio=10, output=request.param)


def test_linear_model_gmm_moments_jacobian(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type="robust", disp=0, debiased=False)
    params = np.r_[
        res.betas.values.ravel(),
        res.risk_premia.values.ravel(),
        mod.factors.ndarray.mean(0),
    ]
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
        eps = p[:, i : (i + 1)] - x @ b[[i]].T
        mom.extend([(eps * fc[:, [j]]) for j in range(fc.shape[1])])
    mom.append(f - mu)
    mom_arr = np.hstack(tuple(mom))

    mod_jac = mod._jacobian(params, True)
    jac = np.zeros((mom_arr.shape[1], params.shape[0]))
    nport, nf = p.shape[1], f.shape[1]
    # 1,1
    jac[: (nport * (nf + 1)), : nport * nf] = np.kron(np.eye(nport), fc.T @ x / n)
    # 1, 2
    col = [(fc.T @ np.ones((n, 1)) @ b[[i]] / n) for i in range(nport)]
    col = np.vstack(tuple(col))
    jac[: (nport * (nf + 1)), nport * nf : nport * nf + nf] = col
    # 1, 3
    col = []
    for i in range(nport):
        col.append(-fc.T @ np.ones((n, 1)) @ b[[i]] / n)
    col = np.vstack(tuple(col))
    jac[: (nport * (nf + 1)), -nf:] = col
    # 2,2
    jac[-nf:, -nf:] = np.eye(nf)

    assert_allclose(mom_arr, mod_mom)
    assert_allclose(jac, mod_jac)

    me = mom_arr - mom_arr.mean(0)[None, :]
    s = me.T @ me / n
    s = (s + s.T) / 2
    cov = np.linalg.inv(jac.T @ np.linalg.inv(s) @ jac) / n
    cov = (cov + cov.T) / 2
    assert_allclose(np.diag(cov), np.diag(res.cov), rtol=5e-3)
    get_all(res)


@pytest.mark.smoke
def test_linear_model_gmm_smoke_iterate(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type="robust", disp=5, steps=20)
    get_all(res)


@pytest.mark.smoke
def test_linear_model_gmm_smoke_risk_free(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type="robust", disp=10)
    get_all(res)
    # Smoke tests
    assert isinstance(str(res._cov_est), str)
    assert isinstance(res._cov_est.__repr__(), str)
    assert isinstance(str(res._cov_est.config), str)


@pytest.mark.smoke
def test_linear_model_gmm_kernel_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type="kernel", disp=10)
    get_all(res)
    # Smoke tests
    assert isinstance(str(res._cov_est), str)
    assert isinstance(res._cov_est.__repr__(), str)
    assert isinstance(str(res._cov_est.config), str)


@pytest.mark.smoke
def test_linear_model_gmm_kernel_bandwidth_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type="kernel", bandwidth=10, disp=10)
    get_all(res)


@pytest.mark.smoke
def test_linear_model_gmm_cue_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type="robust", disp=10, use_cue=True)
    get_all(res)


@pytest.mark.parametrize("nfactor", [1, 3])
@pytest.mark.parametrize("risk_free", [False, True])
def test_linear_model_gmm_std_errors_layout(nfactor, risk_free):
    # Each row is a portfolio, with the standard error of the alpha followed by the
    # standard errors of the loadings, which are the standard errors in cov
    data = generate_data(nfactor=nfactor, nportfolio=8, nobs=500, output="pandas")
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=risk_free)
    res = mod.fit(cov_type="robust", disp=0)
    cov = res.cov
    alpha_se = np.sqrt(np.diag(res._alpha_vcv))
    expected = pd.DataFrame(0.0, index=res.params.index, columns=res.params.columns)
    for i, portfolio in enumerate(expected.index):
        expected.loc[portfolio, "alpha"] = alpha_se[i]
        for factor in data.factors.columns:
            name = f"beta-{portfolio}-{factor}"
            expected.loc[portfolio, factor] = np.sqrt(cov.loc[name, name])
    assert_allclose(res.std_errors, expected)
    assert_allclose(res.tstats, res.params / expected)
