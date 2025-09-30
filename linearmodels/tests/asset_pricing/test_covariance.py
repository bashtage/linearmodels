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


def test_center(data):
    kw = KernelWeight(data.moments, center=True)
    kw2 = KernelWeight(data.moments, center=False)

    assert kw.bandwidth == kw2.bandwidth
    assert np.any(kw.w(data.moments) != kw2.w(data.moments))

    hw = HeteroskedasticWeight(data.moments, center=True)
    hw2 = HeteroskedasticWeight(data.moments, center=False)

    assert np.any(hw.w(data.moments) != hw2.w(data.moments))
