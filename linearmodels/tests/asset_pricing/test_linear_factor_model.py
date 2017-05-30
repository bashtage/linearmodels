import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from linearmodels.asset_pricing.model import LinearFactorModel
from linearmodels.iv.covariance import (_cov_kernel, kernel_optimal_bandwidth,
                                        kernel_weight_bartlett)
from linearmodels.tests.asset_pricing._utility import generate_data, get_all


@pytest.fixture(params=['numpy', 'pandas'])
def data(request):
    return generate_data(nportfolio=10, output=request.param)


def test_linear_model_parameters(data):
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

    m = moments.shape[1]
    jac = np.eye((m))
    block1 = p.shape[1] * (f.shape[1] + 1)
    # 1,1

    jac[:block1, :block1] = np.kron(np.eye(p.shape[1]), fc.T @ fc / n)
    # 2, 1
    loc = 0
    nport, nf = p.shape[1], f.shape[1]
    block2 = block1 + nf
    for i in range(nport):
        block = np.zeros((nf, nf + 1))
        for j in range(nf):  # rows
            for k in range(1, nf + 1):  # cols
                block[j, k] = b[i][j] * lam[k - 1]
                if j + 1 == k:
                    block[j, k] -= alphas[i]
        jac[block1:block2, loc:loc + nf + 1] = block
        loc += nf + 1
    # 2, 2
    jac[block1:block2, block1:block2] = b.T @ b
    # 3,1
    block = np.zeros((nport, nport * (nf + 1)))
    row = col = 0
    for i in range(nport):
        for j in range(nf + 1):
            if j != 0:
                block[row, col] = lam[j - 1]
            col += 1
        row += 1
    jac[-nport:, :(nport * (nf + 1))] = block
    # 3, 2
    jac[-nport:, (nport * (nf + 1)):(nport * (nf + 1)) + nf] = b
    # 3, 3: already done since eye
    mod_jac = mod._jacobian(b, lam, alphas)
    assert_allclose(mod_jac[:block1], jac[:block1])
    assert_allclose(mod_jac[block1:block2, :block1], jac[block1:block2, :block1])
    assert_allclose(mod_jac[block1:block2, block1:block2], jac[block1:block2, block1:block2])
    assert_allclose(mod_jac[block1:block2, block2:], jac[block1:block2, block2:])
    assert_allclose(mod_jac[block2:], jac[block2:])

    s = moments.T @ moments / (n - (nf + 1))
    ginv = np.linalg.inv(jac)
    cov = ginv @ s @ ginv.T / n
    order = np.zeros((nport, nf + 1), dtype=np.int64)
    order[:, 0] = np.arange(block2, block2 + nport)
    for i in range(nf):
        order[:, i + 1] = (nf + 1) * np.arange(nport) + (i + 1)
    order = np.r_[order.ravel(), block1:block2]
    cov = cov[order][:, order]
    cov = (cov + cov.T) / 2
    assert_allclose(cov, res.cov)

    acov = cov[:block1:(nf + 1), :block1:(nf + 1)]
    jstat = float(alphas.T @ np.linalg.pinv(acov) @ alphas)
    assert_allclose(res.j_statistic.stat, jstat)
    assert_allclose(res.j_statistic.pval, 1 - stats.chi2(nport - nf).cdf(jstat))

    get_all(res)

    res = LinearFactorModel(data.portfolios, data.factors).fit(cov_type='kernel', debiased=False)
    std_mom = moments / moments.std(0)[None, :]
    mom = std_mom.sum(1)
    bw = kernel_optimal_bandwidth(mom)
    w = kernel_weight_bartlett(bw, n - 1)
    s = _cov_kernel(moments, w)
    cov = ginv @ s @ ginv.T / n
    cov = cov[order][:, order]
    cov = (cov + cov.T) / 2
    assert_allclose(cov, res.cov)


def test_linear_model_parameters_risk_free(data):
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

    m = moments.shape[1]
    jac = np.eye((m))
    block1 = p.shape[1] * (f.shape[1] + 1)
    # 1,1

    jac[:block1, :block1] = np.kron(np.eye(p.shape[1]), fc.T @ fc / n)
    # 2, 1
    loc = 0
    nport, nf = p.shape[1], f.shape[1]
    block2 = block1 + nf + 1
    for i in range(nport):
        block = np.zeros((nf + 1, nf + 1))
        for j in range(nf + 1):  # rows
            for k in range(1, nf + 1):  # cols
                block[j, k] = bc[i][j] * lam[k]
                if j == k:
                    block[j, k] -= alphas[i]
        jac[block1:block2, loc:loc + nf + 1] = block
        loc += nf + 1
    # 2, 2
    jac[block1:block2, block1:block2] = bc.T @ bc
    # 3,1
    block = np.zeros((nport, nport * (nf + 1)))
    row = col = 0
    for i in range(nport):
        for j in range(nf + 1):
            if j != 0:
                block[row, col] = lam[j]
            col += 1
        row += 1
    jac[-nport:, :(nport * (nf + 1))] = block
    # 3, 2
    jac[-nport:, (nport * (nf + 1)):(nport * (nf + 1)) + nf + 1] = bc
    # 3, 3: already done since eye
    mod_jac = mod._jacobian(bc, lam, alphas)
    assert_allclose(mod_jac[:block1], jac[:block1])
    assert_allclose(mod_jac[block1:block2, :block1], jac[block1:block2, :block1])
    assert_allclose(mod_jac[block1:block2, block1:block2], jac[block1:block2, block1:block2])
    assert_allclose(mod_jac[block1:block2, block2:], jac[block1:block2, block2:])
    assert_allclose(mod_jac[block2:], jac[block2:])

    s = moments.T @ moments / (n - (nf + 1))
    ginv = np.linalg.inv(jac)
    cov = ginv @ s @ ginv.T / n
    order = np.zeros((nport, nf + 1), dtype=np.int64)
    order[:, 0] = np.arange(block2, block2 + nport)
    for i in range(nf):
        order[:, i + 1] = (nf + 1) * np.arange(nport) + (i + 1)
    order = np.r_[order.ravel(), block1:block2]
    cov = cov[order][:, order]
    cov = (cov + cov.T) / 2
    assert_allclose(cov, res.cov)

    acov = cov[:block1:(nf + 1), :block1:(nf + 1)]
    jstat = float(alphas.T @ np.linalg.pinv(acov) @ alphas)
    assert_allclose(res.cov.values[:block1:(nf + 1), :block1:(nf + 1)], acov)
    assert_allclose(res.j_statistic.stat, jstat, rtol=1e-1)
    assert_allclose(res.j_statistic.pval, 1 - stats.chi2(nport - nf - 1).cdf(jstat), rtol=1e-2)

    get_all(res)


def test_linear_model_parameters_risk_free_gls(data):
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

    m = moments.shape[1]
    jac = np.eye((m))
    block1 = p.shape[1] * (f.shape[1] + 1)
    # 1,1

    jac[:block1, :block1] = np.kron(np.eye(p.shape[1]), fc.T @ fc / n)
    # 2, 1
    loc = 0
    nport, nf = p.shape[1], f.shape[1]
    block2 = block1 + nf + 1
    bct = sigma_inv @ bc
    at = sigma_inv @ alphas
    for i in range(nport):
        block = np.zeros((nf + 1, nf + 1))
        for j in range(nf + 1):  # rows
            for k in range(1, nf + 1):  # cols
                block[j, k] = bct[i][j] * lam[k]
                if j == k:
                    block[j, k] -= at[i]
        jac[block1:block2, loc:loc + nf + 1] = block
        loc += nf + 1
    # 2, 2
    jac[block1:block2, block1:block2] = bc.T @ sigma_inv @ bc
    # 3,1
    block = np.zeros((nport, nport * (nf + 1)))
    row = col = 0
    for i in range(nport):
        for j in range(nf + 1):
            if j != 0:
                block[row, col] = lam[j]
            col += 1
        row += 1
    jac[-nport:, :(nport * (nf + 1))] = block
    # 3, 2
    jac[-nport:, (nport * (nf + 1)):(nport * (nf + 1)) + nf + 1] = bc
    # 3, 3: already done since eye
    mod_jac = mod._jacobian(bc, lam, alphas)
    assert_allclose(mod_jac[:block1], jac[:block1])
    assert_allclose(mod_jac[block1:block2, :block1], jac[block1:block2, :block1])
    assert_allclose(mod_jac[block1:block2, block1:block2], jac[block1:block2, block1:block2])
    assert_allclose(mod_jac[block1:block2, block2:], jac[block1:block2, block2:])
    assert_allclose(mod_jac[block2:], jac[block2:])

    s = moments.T @ moments / (n - (nf + 1))
    ginv = np.linalg.inv(jac)
    cov = ginv @ s @ ginv.T / n
    order = np.zeros((nport, nf + 1), dtype=np.int64)
    order[:, 0] = np.arange(block2, block2 + nport)
    for i in range(nf):
        order[:, i + 1] = (nf + 1) * np.arange(nport) + (i + 1)
    order = np.r_[order.ravel(), block1:block2]
    cov = cov[order][:, order]
    cov = (cov + cov.T) / 2
    assert_allclose(cov, res.cov)

    acov = cov[:block1:(nf + 1), :block1:(nf + 1)]
    jstat = float(alphas.T @ np.linalg.pinv(acov) @ alphas)
    assert_allclose(res.cov.values[:block1:(nf + 1), :block1:(nf + 1)], acov)
    assert_allclose(res.j_statistic.stat, jstat, rtol=1e-1)
    assert_allclose(res.j_statistic.pval, 1 - stats.chi2(nport - nf - 1).cdf(jstat), rtol=1e-2)

    get_all(res)
