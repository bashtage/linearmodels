import numpy as np
import pytest
from numpy.testing import assert_allclose

from linearmodels.system._utility import blocked_column_product, blocked_diag_product, \
    blocked_inner_prod, inv_matrix_sqrt


@pytest.fixture(params=(3, np.arange(1, 6)), ids=['common-size', 'different-size'])
def data(request):
    k = 5
    t = 200
    p = request.param
    if np.isscalar(p):
        p = p * np.ones(k, dtype=np.int64)

    x = [np.random.randn(t, p[i]) for i in range(k)]
    y = [np.random.randn(t, 1) for _ in range(k)]
    sigma = 0.5 * np.eye(k) + 0.5 * np.ones((5, 5))
    return y, x, sigma


def test_inner_product(data):
    y, x, sigma = data
    efficient = blocked_inner_prod(x, sigma)
    nobs = x[0].shape[0]
    omega = np.kron(sigma, np.eye(nobs))
    k = len(x)
    bigx = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(x[i])
            else:
                row.append(np.zeros((nobs, x[j].shape[1])))
        bigx.append(np.hstack(row))
    bigx = np.vstack(bigx)
    expected = bigx.T @ omega @ bigx
    assert_allclose(efficient, expected)


def test_inner_product_short_circuit(data):
    y, x, sigma = data
    sigma = np.eye(len(x))
    efficient = blocked_inner_prod(x, sigma)
    nobs = x[0].shape[0]
    omega = np.kron(sigma, np.eye(nobs))
    k = len(x)
    bigx = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(x[i])
            else:
                row.append(np.zeros((nobs, x[j].shape[1])))
        bigx.append(np.hstack(row))
    bigx = np.vstack(bigx)
    expected = bigx.T @ omega @ bigx
    assert_allclose(efficient, expected)


def test_column_product(data):
    y, x, sigma = data
    efficient = blocked_column_product(y, sigma)
    nobs = y[0].shape[0]
    omega = np.kron(sigma, np.eye(nobs))
    bigy = np.vstack(y)
    expected = omega @ bigy
    assert_allclose(efficient, expected)


def test_diag_product(data):
    y, x, sigma = data
    efficient = blocked_diag_product(x, sigma)
    nobs = x[0].shape[0]
    omega = np.kron(sigma, np.eye(nobs))
    k = len(x)
    bigx = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(x[i])
            else:
                row.append(np.zeros((nobs, x[j].shape[1])))
        bigx.append(np.hstack(row))
    bigx = np.vstack(bigx)
    expected = omega @ bigx
    assert_allclose(efficient, expected)


def test_inv_matrix_sqrt(data):
    y, x, sigma = data
    k = sigma.shape[0]
    sigma_m12 = inv_matrix_sqrt(sigma)
    assert_allclose(sigma_m12 - sigma_m12.T, np.zeros((k, k)))
    assert_allclose(np.linalg.inv(sigma_m12 @ sigma_m12), sigma)
    assert_allclose(sigma_m12 @ sigma @ sigma_m12, np.eye(k), atol=1e-14)
