import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose, assert_equal
from scipy import stats

from linearmodels.asset_pricing.model import LinearFactorModel, LinearFactorModelGMM, \
    TradedFactorModel
from linearmodels.iv.data import IVData
from linearmodels.iv.model import _OLS
from linearmodels.tests.asset_pricing._utility import generate_data


@pytest.fixture(params=['numpy', 'pandas'])
def data(request):
    return generate_data(nportfolio=10, output=request.param)


def get_all(res):
    attrs = dir(res)
    for attr_name in attrs:
        if attr_name.startswith('_'):
            continue
        attr = getattr(res, attr_name)
        if callable(attr):
            attr()


def test_linear_model_parameters(data):
    # TODO: Test Jacobian
    mod = LinearFactorModel(data.portfolios, data.factors)
    res = mod.fit()
    f = mod.factors.ndarray
    p = mod.portfolios.ndarray
    n = f.shape[0]
    moments = np.zeros((n, p.shape[1] * (f.shape[1] + 1) + f.shape[1] + p.shape[1]))
    fc = np.c_[np.ones((n, 1)), f]
    betas = np.linalg.lstsq(fc, p)[0]
    eps = p - fc @ betas
    loc = 0
    for i in range(eps.shape[1]):
        for j in range(fc.shape[1]):
            moments[:, loc] = eps[:, i] * fc[:, j]
            loc += 1
    b = betas[1:, :].T
    lam = np.linalg.lstsq(b, p.mean(0)[:, None])[0]
    pricing_errors = p - (b @ lam).T
    for i in range(lam.shape[0]):
        lam_error = (p - (b @ lam).T) @ b[:, [i]]
        moments[:, loc] = lam_error.squeeze()
        loc += 1
    alphas = pricing_errors.mean(0)[:, None]
    moments[:, loc:] = pricing_errors - alphas.T
    mod_moments = mod._moments(eps, b, lam, alphas, pricing_errors)

    assert_allclose(res.betas, b)
    assert_allclose(res.risk_premia, lam.squeeze())
    assert_allclose(res.alphas, alphas.squeeze())
    assert_allclose(moments, mod_moments)


def test_linear_model_parameters_risk_free(data):
    # TODO: Test Jacobian
    mod = LinearFactorModel(data.portfolios, data.factors, risk_free=True)
    res = mod.fit()
    f = mod.factors.ndarray
    p = mod.portfolios.ndarray
    n = f.shape[0]
    moments = np.zeros((n, p.shape[1] * (f.shape[1] + 1) + f.shape[1] + 1 + p.shape[1]))
    fc = np.c_[np.ones((n, 1)), f]
    betas = np.linalg.lstsq(fc, p)[0]
    eps = p - fc @ betas
    loc = 0
    for i in range(eps.shape[1]):
        for j in range(fc.shape[1]):
            moments[:, loc] = eps[:, i] * fc[:, j]
            loc += 1

    bc = np.c_[np.ones((p.shape[1], 1)), betas[1:, :].T]
    lam = np.linalg.lstsq(bc, p.mean(0)[:, None])[0]
    pricing_errors = p - (bc @ lam).T
    for i in range(lam.shape[0]):
        lam_error = (p - (bc @ lam).T) @ bc[:, [i]]
        moments[:, loc] = lam_error.squeeze()
        loc += 1
    alphas = p.mean(0)[:, None] - bc @ lam
    moments[:, loc:] = pricing_errors - alphas.T
    mod_moments = mod._moments(eps, bc, lam, alphas, pricing_errors)

    assert_allclose(res.betas, bc[:, 1:])
    assert_allclose(res.risk_premia, lam.squeeze())
    assert_allclose(res.alphas, alphas.squeeze())
    assert_allclose(moments, mod_moments)


def test_linear_model_parameters_risk_free_gls(data):
    # TODO: Test Jacobian
    mod = LinearFactorModel(data.portfolios, data.factors, risk_free=True)
    p = mod.portfolios.ndarray
    sigma = np.cov(p.T)
    val, vec = np.linalg.eigh(sigma)
    sigma_m12 = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T
    sigma_inv = np.linalg.inv(sigma)

    mod = LinearFactorModel(data.portfolios, data.factors, risk_free=True, sigma=sigma)
    res = mod.fit()
    f = mod.factors.ndarray
    p = mod.portfolios.ndarray
    n = f.shape[0]
    moments = np.zeros((n, p.shape[1] * (f.shape[1] + 1) + f.shape[1] + 1 + p.shape[1]))
    fc = np.c_[np.ones((n, 1)), f]
    betas = np.linalg.lstsq(fc, p)[0]
    eps = p - fc @ betas
    loc = 0
    for i in range(eps.shape[1]):
        for j in range(fc.shape[1]):
            moments[:, loc] = eps[:, i] * fc[:, j]
            loc += 1
    bc = np.c_[np.ones((p.shape[1], 1)), betas[1:, :].T]
    lam = np.linalg.lstsq(sigma_m12 @ bc, sigma_m12 @ p.mean(0)[:, None])[0]
    pricing_errors = p - (bc @ lam).T

    for i in range(lam.shape[0]):
        lam_error = pricing_errors @ sigma_inv @ bc[:, [i]]
        moments[:, loc] = lam_error.squeeze()
        loc += 1
    alphas = p.mean(0)[:, None] - bc @ lam
    moments[:, loc:] = pricing_errors - alphas.T
    mod_moments = mod._moments(eps, bc, lam, alphas, pricing_errors)

    assert_allclose(res.betas, bc[:, 1:])
    assert_allclose(res.risk_premia, lam.squeeze())
    assert_allclose(res.alphas, alphas.squeeze())
    assert_allclose(moments, mod_moments)


def test_linear_model_cross_section_smoke(data):
    mod = LinearFactorModel(data.portfolios, data.factors)
    res = mod.fit(cov_type='robust')
    get_all(res)


def test_linear_model_cross_section_kernel_smoke(data):
    mod = LinearFactorModel(data.portfolios, data.factors)
    res = mod.fit(cov_type='kernel')
    get_all(res)


def test_linear_model_cross_section_risk_free_smoke(data):
    mod = LinearFactorModel(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type='robust')
    get_all(res)


def test_linear_model_gmm_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type='robust', disp=5)
    get_all(res)


def test_linear_model_gmm_smoke_iterate(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type='robust', disp=5, steps=20)
    get_all(res)


def test_linear_model_gmm_smoke_risk_free(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type='robust', disp=10)
    get_all(res)


def test_linear_model_gmm_kernel_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type='kernel', disp=10)
    get_all(res)


def test_linear_model_gmm_kernel_bandwidth_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type='kernel', bandwidth=10, disp=10)
    get_all(res)


def test_linear_model_gmm_cue_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type='robust', disp=10, use_cue=True)
    get_all(res)


def test_linear_model_time_series(data):
    mod = TradedFactorModel(data.portfolios, data.factors)
    mod.fit()
    res = mod.fit()
    get_all(res)
    all_params = []
    all_tstats = []
    nobs, nport = data.portfolios.shape
    nf = data.factors.shape[1]
    e = np.zeros((nobs, (nport * (nf + 1))))
    x = np.zeros((nobs, (nport * (nf + 1))))
    factors = sm.add_constant(data.factors)
    loc = 0
    for i in range(data.portfolios.shape[1]):
        if isinstance(data.portfolios, pd.DataFrame):
            p = data.portfolios.iloc[:, i:(i + 1)]
        else:
            p = data.portfolios[:, i:(i + 1)]
        ols_res = _OLS(p, factors).fit(cov_type='robust', debiased=True)
        all_params.extend(list(ols_res.params))
        all_tstats.extend(list(ols_res.tstats))
        x[:, loc:(loc + nf + 1)] = factors
        e[:, loc:(loc + nf + 1)] = ols_res.resids.values[:, None]
        loc += nf + 1
        cov = res.cov.values[(nf + 1) * i:(nf + 1) * (i + 1), (nf + 1) * i:(nf + 1) * (i + 1)]
        ols_cov = ols_res.cov.values

        assert_allclose(cov, ols_cov)
    assert_allclose(res.params.values.ravel(), np.array(all_params))
    assert_allclose(res.tstats.values.ravel(), np.array(all_tstats))
    assert_allclose(res.risk_premia, np.asarray(factors).mean(0)[1:])

    xpxi_direct = np.eye((nf + 1) * nport + nf)
    f = np.asarray(factors)
    fpfi = np.linalg.inv(f.T @ f / nobs)
    nfp1 = nf + 1
    for i in range(nport):
        st, en = i * nfp1, (i + 1) * nfp1
        xpxi_direct[st:en, st:en] = fpfi
    f = np.asarray(factors)[:, 1:]
    xe = np.c_[x * e, f - f.mean(0)[None, :]]
    xeex_direct = xe.T @ xe / nobs
    cov = xpxi_direct @ xeex_direct @ xpxi_direct / (nobs - nfp1)
    assert_allclose(cov, res.cov.values)

    alphas = np.array(all_params)[0::nfp1][:, None]
    alpha_cov = cov[0:(nfp1 * nport):nfp1, 0:(nfp1 * nport):nfp1]
    stat_direct = float(alphas.T @ np.linalg.inv(alpha_cov) @ alphas)
    assert_allclose(res.j_statistic.stat, stat_direct)
    assert_allclose(1.0 - stats.chi2.cdf(stat_direct, nport), res.j_statistic.pval)


def test_linear_model_time_series_kernel_smoke(data):
    mod = TradedFactorModel(data.portfolios, data.factors)
    mod.fit(cov_type='kernel')


def test_linear_model_time_series_error(data):
    mod = TradedFactorModel(data.portfolios, data.factors)
    with pytest.raises(ValueError):
        mod.fit(cov_type='unknown')


def test_errors(data):
    p = data.portfolios.copy()
    f = data.factors.copy()
    if isinstance(p, pd.DataFrame):
        p2 = p.copy()
        p3 = p.copy().iloc[:-1]
        p4 = p.copy()
        p5 = p.copy().iloc[:f.shape[1] - 1, :1]
        p4 = p4.iloc[:, :(f.shape[1] - 1)]
        p2['dupe'] = p.iloc[:, 0]
        p['const'] = 1.0

        f5 = f.copy()
        f5 = f5.iloc[:p5.shape[0]]
        f2 = f.copy()
        f2['dupe'] = f.iloc[:, 0]
        f['const'] = 1.0
    else:
        p2 = np.c_[p, p[:, [0]]]
        p3 = p.copy()[:-1]
        p4 = p.copy()
        p5 = p.copy()[:f.shape[1] - 1, :1]
        p4 = p4[:, :(f.shape[1] - 1)]
        p = np.c_[np.ones((p.shape[0], 1)), p]

        f5 = f.copy()
        f5 = f5[:p5.shape[0]]
        f2 = np.c_[f, f[:, [0]]]
        f = np.c_[np.ones((f.shape[0], 1)), f]

    with pytest.raises(ValueError):
        TradedFactorModel(p, data.factors)
    with pytest.raises(ValueError):
        TradedFactorModel(p2, data.factors)
    with pytest.raises(ValueError):
        TradedFactorModel(p3, data.factors)
    with pytest.raises(ValueError):
        TradedFactorModel(data.portfolios, f)
    with pytest.raises(ValueError):
        TradedFactorModel(data.portfolios, f2)
    with pytest.raises(ValueError):
        TradedFactorModel(p5, f5)
    with pytest.raises(ValueError):
        LinearFactorModel(p4, data.factors)


def test_drop_missing(data):
    p = data.portfolios
    if isinstance(p, pd.DataFrame):
        p.iloc[::33] = np.nan
    else:
        p[::33] = np.nan

    res = TradedFactorModel(p, data.factors).fit()

    p = IVData(p)
    f = IVData(data.factors)
    isnull = p.isnull | f.isnull
    p.drop(isnull)
    f.drop(isnull)

    res2 = TradedFactorModel(p, f).fit()
    assert_equal(res.params.values, res2.params.values)


def test_unknown_kernel(data):
    mod = LinearFactorModel(data.portfolios, data.factors)
    with pytest.raises(ValueError):
        mod.fit(cov_type='unknown')
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    with pytest.raises(ValueError):
        mod.fit(cov_type='unknown')


def test_all_missing():
    p = np.nan * np.ones((1000, 10))
    f = np.nan * np.ones((1000, 3))
    with pytest.raises(ValueError):
        TradedFactorModel(p, f)
