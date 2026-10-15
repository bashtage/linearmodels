"""
Tests of the J statistic of LinearFactorModel

The second stage estimates make the pricing errors orthogonal to the columns of
sigma^-1 [1 betas], so that they have nportfolio - nfactor - nrf free dimensions and
their covariance has that rank. The statistic is the Wald statistic of the pricing
errors in that space.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from linearmodels.asset_pricing.model import LinearFactorModel, _j_statistic
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.tests.asset_pricing._utility import simulate_data, textbook_j_stat


def second_stage_restrictions(portfolios, factors, risk_free, sigma):
    """sigma^-1 [1 betas] from the first stage regressions"""
    nobs, nportfolio = portfolios.shape
    fc = np.c_[np.ones((nobs, 1)), factors]
    betas = np.linalg.lstsq(fc, portfolios, rcond=None)[0][1:].T
    if risk_free:
        betas = np.c_[np.ones((nportfolio, 1)), betas]
    sigma_inv = np.eye(nportfolio) if sigma is None else np.linalg.inv(sigma)
    return sigma_inv @ betas


def alpha_block(res, nfactor):
    """The covariance of the alphas, from the public covariance of the parameters"""
    nportfolio = len(res.alphas)
    cov = np.asarray(res.cov)
    return cov[:: nfactor + 1, :: nfactor + 1][:nportfolio, :nportfolio]


@pytest.mark.parametrize("risk_free", [False, True])
@pytest.mark.parametrize("use_sigma", [False, True])
@pytest.mark.parametrize("cov_type", ["robust", "kernel"])
@pytest.mark.parametrize("debiased", [False, True])
def test_j_statistic_textbook(risk_free, use_sigma, cov_type, debiased):
    nportfolio, nfactor = 7, 2
    portfolios, factors = simulate_data(300, nportfolio, nfactor, seed=11)
    sigma = np.cov(portfolios.T) if use_sigma else None
    config = {"bandwidth": 4} if cov_type == "kernel" else {}
    mod = LinearFactorModel(portfolios, factors, risk_free=risk_free, sigma=sigma)
    res = mod.fit(cov_type=cov_type, debiased=debiased, **config)

    df = nportfolio - nfactor - int(risk_free)
    alphas = np.asarray(res.alphas)[:, None]
    restrictions = second_stage_restrictions(portfolios, factors, risk_free, sigma)
    assert restrictions.shape[1] == nfactor + int(risk_free)
    assert_allclose(restrictions.T @ alphas, 0.0, atol=1e-10)
    vcv = alpha_block(res, nfactor)
    expected = textbook_j_stat(alphas, vcv, restrictions)
    assert isinstance(res.j_statistic, WaldTestStatistic)
    assert_allclose(res.j_statistic.stat, expected, rtol=1e-9)
    assert res.j_statistic.df == df
    assert_allclose(res.j_statistic.pval, stats.chi2(df).sf(expected), rtol=1e-8)


@pytest.mark.parametrize("risk_free", [False, True])
@pytest.mark.parametrize("seed", [3, 11])
def test_j_statistic_does_not_depend_on_row_order(risk_free, seed):
    # The statistic changed in the third digit when the rows were only reordered
    # since the null direction of the covariance was inverted
    portfolios, factors = simulate_data(300, 7, 2, seed=seed)
    mod = LinearFactorModel(portfolios, factors, risk_free=risk_free)
    base = mod.fit().j_statistic.stat
    for i in range(5):
        perm = np.random.default_rng(i).permutation(portfolios.shape[0])
        mod = LinearFactorModel(portfolios[perm], factors[perm], risk_free=risk_free)
        assert_allclose(mod.fit().j_statistic.stat, base, rtol=1e-8)


@pytest.mark.parametrize("risk_free", [False, True])
@pytest.mark.parametrize("use_sigma", [False, True])
@pytest.mark.parametrize("cov_type", ["robust", "kernel"])
def test_j_statistic_does_not_depend_on_portfolio_order(risk_free, use_sigma, cov_type):
    # Reordering the portfolios (and sigma in the same way) reorders the pricing
    # errors, their covariance and the restrictions in the same way, and leaves the
    # statistic of the space where the pricing errors lie unchanged
    portfolios, factors = simulate_data(300, 7, 2, seed=3)
    sigma = np.cov(portfolios.T) if use_sigma else None
    config = {"bandwidth": 4} if cov_type == "kernel" else {}
    mod = LinearFactorModel(portfolios, factors, risk_free=risk_free, sigma=sigma)
    base = mod.fit(cov_type=cov_type, **config).j_statistic.stat
    for i in range(4):
        order = np.random.default_rng(i).permutation(portfolios.shape[1])
        assert not np.array_equal(order, np.arange(portfolios.shape[1]))
        reordered = None if sigma is None else sigma[np.ix_(order, order)]
        mod = LinearFactorModel(
            portfolios[:, order], factors, risk_free=risk_free, sigma=reordered
        )
        stat = mod.fit(cov_type=cov_type, **config).j_statistic.stat
        assert_allclose(stat, base, rtol=1e-8)


def test_j_statistic_helper_does_not_depend_on_the_basis():
    # Only the space that is orthogonal to the restrictions matters, and not the
    # basis Q of that space that is used to compute the statistic
    rng = np.random.default_rng(3)
    nportfolio, nrestriction = 8, 3
    restrictions = rng.standard_normal((nportfolio, nrestriction))
    q = np.linalg.svd(restrictions)[0][:, nrestriction:]
    g = rng.standard_normal((nportfolio, nportfolio))
    vcv = g @ g.T + np.eye(nportfolio)
    alphas = q @ rng.standard_normal((nportfolio - nrestriction, 1))
    expected = _j_statistic(alphas, vcv, restrictions).stat
    other_basis = restrictions @ rng.standard_normal((nrestriction, nrestriction))
    assert_allclose(_j_statistic(alphas, vcv, other_basis).stat, expected, rtol=1e-10)
    # The same statistic for any other orthonormal basis of the space
    rotation = np.linalg.qr(rng.standard_normal((5, 5)))[0]
    qr = q @ rotation
    qa = qr.T @ alphas
    rotated = float(np.squeeze(qa.T @ np.linalg.solve(qr.T @ vcv @ qr, qa)))
    assert_allclose(rotated, expected, rtol=1e-10)


@pytest.mark.parametrize("risk_free", [False, True])
def test_j_statistic_just_identified(risk_free):
    # With as many portfolios as risk premia the pricing errors are 0
    nrp = 2 + int(risk_free)
    portfolios, factors = simulate_data(300, nrp, 2, seed=5)
    res = LinearFactorModel(portfolios, factors, risk_free=risk_free).fit()
    assert_allclose(res.j_statistic.stat, 0.0, atol=1e-12)
    assert res.j_statistic.df == 0


def test_j_statistic_gls_is_not_a_truncated_inverse():
    # With GLS weights, the covariance of the pricing errors has no gap between the
    # eigenvalues of the directions that are restricted and those that are not, so
    # that inverting the largest eigenvalues does not give the statistic of the
    # pricing errors, which lie in the unrestricted space
    portfolios, factors = simulate_data(250, 10, 2, seed=0, alpha=False)
    sigma = np.cov(portfolios.T)
    res = LinearFactorModel(portfolios, factors, sigma=sigma).fit(debiased=False)
    alphas = np.asarray(res.alphas)[:, None]
    vcv = alpha_block(res, 2)
    values, vectors = np.linalg.eigh(vcv)
    top = vectors[:, -8:]
    truncated = float(np.sum((top.T @ alphas) ** 2 / values[-8:, None]))
    restrictions = second_stage_restrictions(portfolios, factors, False, sigma)
    expected = textbook_j_stat(alphas, vcv, restrictions)
    assert_allclose(res.j_statistic.stat, expected, rtol=1e-9)
    assert truncated < 0.7 * expected


def test_j_statistic_helper():
    rng = np.random.default_rng(0)
    nportfolio, nrestriction = 7, 2
    restrictions = rng.standard_normal((nportfolio, nrestriction))
    q = np.linalg.svd(restrictions)[0][:, nrestriction:]
    # A covariance that is exactly singular in the directions of the restrictions
    m = rng.standard_normal((5, 5))
    vcv = q @ (m @ m.T + np.eye(5)) @ q.T
    alphas = q @ rng.standard_normal((5, 1))
    stat = _j_statistic(alphas, vcv, restrictions)
    assert isinstance(stat, WaldTestStatistic)
    assert stat.df == 5
    # The generalized inverse is then the Moore-Penrose pseudo-inverse
    pinv_stat = float(np.squeeze(alphas.T @ np.linalg.pinv(vcv) @ alphas))
    assert_allclose(stat.stat, pinv_stat)
    assert_allclose(stat.stat, textbook_j_stat(alphas, vcv, restrictions))
    # A covariance that is not singular in those directions: the statistic of the
    # pricing errors in the space where they lie is not the one that inverts all
    # the directions
    g = rng.standard_normal((nportfolio, nportfolio))
    full = vcv + 0.1 * g @ g.T
    stat = _j_statistic(alphas, full, restrictions)
    assert_allclose(stat.stat, textbook_j_stat(alphas, full, restrictions))
    assert stat.stat < float(np.squeeze(alphas.T @ np.linalg.inv(full) @ alphas))


def test_j_statistic_helper_no_degrees_of_freedom():
    rng = np.random.default_rng(1)
    restrictions = rng.standard_normal((3, 3))
    stat = _j_statistic(np.zeros((3, 1)), np.eye(3), restrictions)
    assert_allclose(stat.stat, 0.0, atol=1e-12)
    assert stat.df == 0


def test_j_statistic_helper_singular_covariance():
    rng = np.random.default_rng(2)
    nportfolio, nrestriction = 6, 2
    restrictions = rng.standard_normal((nportfolio, nrestriction))
    q = np.linalg.svd(restrictions)[0][:, nrestriction:]
    # The covariance has rank 3 in the space of the 4 unrestricted directions
    vcv = q[:, :3] @ np.diag([3.0, 2.0, 1.0]) @ q[:, :3].T
    alphas = q @ rng.standard_normal((4, 1))
    stat = _j_statistic(alphas, vcv, restrictions)
    assert isinstance(stat, InvalidTestStatistic)
    assert np.isnan(stat.stat)
    assert np.isnan(stat.pval)
    assert "rank smaller than the 4 degrees of freedom" in str(stat)


def test_j_statistic_invalid_summary(monkeypatch):
    # An invalid statistic can be summarized
    portfolios, factors = simulate_data(300, 7, 2, seed=5)
    mod = LinearFactorModel(portfolios, factors)

    def invalid(alphas, alpha_vcv, restrictions):
        return InvalidTestStatistic("rank", name="J-statistic")

    monkeypatch.setattr("linearmodels.asset_pricing.model._j_statistic", invalid)
    res = mod.fit()
    assert isinstance(res.j_statistic, InvalidTestStatistic)
    assert "J-statistic" in str(res.summary)
