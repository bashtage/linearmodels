import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.optimize import minimize

import linearmodels.asset_pricing.model as model_module
from linearmodels.asset_pricing.model import LinearFactorModel, LinearFactorModelGMM
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


def cue_moments(theta, p, f):
    """Moment conditions of the model with excess returns, written out directly"""
    nobs, nport = p.shape
    nfactor = f.shape[1]
    betas = theta[: nport * nfactor].reshape(nport, nfactor)
    lam = theta[nport * nfactor : nport * nfactor + nfactor]
    mu = theta[nport * nfactor + nfactor :]
    fe = f - mu
    eps = p - (betas @ lam)[None, :] - fe @ betas.T
    fc = np.column_stack([np.ones(nobs), f])
    return np.column_stack([np.hstack([fc * eps[:, [i]] for i in range(nport)]), fe])


def cue_objective(theta, p, f):
    g = cue_moments(theta, p, f)
    gc = g - g.mean(0)
    w = np.linalg.inv(gc.T @ gc / g.shape[0])
    return g.shape[0] * float(g.mean(0) @ w @ g.mean(0))


def test_linear_model_gmm_cue_j_statistic():
    # The J statistic is the minimized value of the continuously updated objective,
    # which uses the weight matrix at the estimates and not at the starting values
    rng = np.random.default_rng(3)
    nobs, nport, nfactor = 150, 8, 2
    f = 0.05 * (
        0.2 * rng.standard_normal((nobs, 1)) + rng.standard_normal((nobs, nfactor))
    )
    f += np.arange(1, nfactor + 1)[None, :] / 1200
    betas = rng.chisquare(2, (nfactor, nport)) / 2
    p = f @ betas + 0.03 * rng.standard_normal((nobs, nport))
    p += np.arange(nport)[None, :] / nport / 200
    res = LinearFactorModelGMM(p, f).fit(use_cue=True, disp=0)
    start = LinearFactorModel(p, f).fit()
    theta0 = np.r_[
        np.asarray(start.betas).ravel(), np.asarray(start.risk_premia), f.mean(0)
    ]
    best = minimize(
        cue_objective,
        theta0,
        args=(p, f),
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 5000},
    )
    assert_allclose(res.j_statistic.stat, best.fun, rtol=1e-4)


@pytest.mark.parametrize("kwargs", [{"steps": 3}, {"use_cue": True}])
def test_linear_model_gmm_opt_options_every_step(monkeypatch, kwargs):
    # The options of the optimizer apply to every minimization of the estimator
    data = generate_data(nfactor=1, nportfolio=4, nobs=300, output="numpy")
    calls = []
    minimize_original = minimize

    def recording_minimize(*args, **kw):
        calls.append(kw)
        return minimize_original(*args, **kw)

    monkeypatch.setattr(model_module, "minimize", recording_minimize)
    opt_options = {"method": "L-BFGS-B", "tol": 1e-8, "options": {"maxcor": 7}}
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    mod.fit(disp=0, opt_options=opt_options, **kwargs)
    assert len(calls) >= 2
    for call in calls:
        assert call["method"] == "L-BFGS-B"
        assert call["tol"] == 1e-8
        assert call["options"] == {"maxiter": 1000, "maxcor": 7}


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
