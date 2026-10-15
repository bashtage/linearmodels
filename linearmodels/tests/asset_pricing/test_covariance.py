import numpy as np
from numpy.testing import assert_allclose
import pytest

from linearmodels.asset_pricing.covariance import (
    HeteroskedasticCovariance,
    HeteroskedasticWeight,
    KernelCovariance,
    KernelWeight,
)
from linearmodels.shared.utility import AttrDict


@pytest.fixture
def data():
    moments = np.random.randn(500, 10)
    jacobian = np.random.rand(10, 8)
    jacobian_inv = np.eye(10)
    return AttrDict(moments=moments, jacobian=jacobian, inv_jacobian=jacobian_inv)


def test_kernel_errors(data):
    with pytest.raises(ValueError, match=r"Unknown kernel"):
        KernelWeight(data.moments, kernel="unknown")
    with pytest.raises(ValueError, match=r"bandwidth must be non-negative"):
        KernelWeight(data.moments, bandwidth=-0.5)
    with pytest.raises(ValueError, match=r"Unknown kernel"):
        KernelCovariance(data.moments, jacobian=data.jacobian, kernel="unknown")
    with pytest.raises(ValueError, match=r"bandwidth must be non-negative"):
        KernelCovariance(data.moments, jacobian=data.jacobian, bandwidth=-4)


def test_no_jacobian(data):
    with pytest.raises(ValueError, match=r"One and only one of jacobian"):
        KernelCovariance(data.moments)
    with pytest.raises(ValueError, match=r"One and only one of jacobian "):
        KernelCovariance(
            data.moments, jacobian=data.jacobian, inv_jacobian=data.inv_jacobian
        )


def test_alt_jacobians(data):
    hc = HeteroskedasticCovariance(data.moments, jacobian=data.inv_jacobian)
    assert_allclose(hc.inv_jacobian, data.inv_jacobian)
    hc = HeteroskedasticCovariance(data.moments, inv_jacobian=data.inv_jacobian)
    assert_allclose(hc.jacobian, np.eye(10))


@pytest.mark.parametrize("debiased", [False, True])
@pytest.mark.parametrize("kernel", [False, True])
def test_moment_cov(debiased, kernel):
    rng = np.random.default_rng(12345)
    nobs, nmoment, nparam, df = 400, 10, 7, 3
    moments = rng.standard_normal((nobs, nmoment)) @ rng.standard_normal(
        (nmoment, nmoment)
    )
    jacobian = rng.standard_normal((nmoment, nparam))
    if kernel:
        cov_est = KernelCovariance(
            moments, jacobian=jacobian, bandwidth=4, debiased=debiased, df=df
        )
    else:
        cov_est = HeteroskedasticCovariance(
            moments, jacobian=jacobian, debiased=debiased, df=df
        )
    s = cov_est.s
    dof = nobs - int(debiased) * df
    s_inv = np.linalg.inv(s)
    a_inv = np.linalg.inv(jacobian.T @ s_inv @ jacobian)
    # The covariance of the parameters and of the average moments
    assert_allclose(cov_est.cov, a_inv / dof)
    expected = (s - jacobian @ a_inv @ jacobian.T) / dof
    assert_allclose(cov_est._moment_cov, expected, atol=1e-14)
    # The same covariance in the form M S M', with M the residual projection
    proj = np.eye(nmoment) - jacobian @ a_inv @ jacobian.T @ s_inv
    assert_allclose(cov_est._moment_cov, proj @ s @ proj.T / dof, atol=1e-14)
    assert_allclose(cov_est._moment_cov, cov_est._moment_cov.T, atol=1e-18)
    # Rank is the number of moments less the number of parameters
    eigenvalues = np.linalg.eigvalsh(cov_est._moment_cov)
    assert np.sum(eigenvalues > 1e-10 * eigenvalues.max()) == nmoment - nparam
    assert eigenvalues.min() > -1e-10 * eigenvalues.max()
    # The average moments are uncorrelated with the estimates, so that the covariance
    # of the average moments is smaller than that of the moments
    assert np.all(np.diag(expected) < np.diag(s) / dof)


@pytest.mark.parametrize("cov_class", [HeteroskedasticCovariance, KernelCovariance])
def test_moment_cov_is_private(cov_class):
    # The covariance of the average moment conditions is used for the standard
    # errors of the alphas of LinearFactorModelGMM, and is not part of the API
    assert not hasattr(cov_class, "moment_cov")
    assert isinstance(cov_class._moment_cov, property)


def test_moment_cov_exactly_identified():
    rng = np.random.default_rng(0)
    moments = rng.standard_normal((100, 5))
    jacobian = rng.standard_normal((5, 5))
    for kwargs in ({"jacobian": jacobian}, {"inv_jacobian": np.linalg.inv(jacobian)}):
        cov_est = HeteroskedasticCovariance(moments, **kwargs)
        assert cov_est._moment_cov.shape == (5, 5)
        assert_allclose(cov_est._moment_cov, 0.0, atol=1e-12)


def test_center(data):
    kw = KernelWeight(data.moments, center=True)
    kw2 = KernelWeight(data.moments, center=False)

    assert kw.bandwidth == kw2.bandwidth
    assert np.any(kw.w(data.moments) != kw2.w(data.moments))

    hw = HeteroskedasticWeight(data.moments, center=True)
    hw2 = HeteroskedasticWeight(data.moments, center=False)

    assert np.any(hw.w(data.moments) != hw2.w(data.moments))
