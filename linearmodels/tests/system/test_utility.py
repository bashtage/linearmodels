from linearmodels.compat.numpy import lstsq

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.system._utility import LinearConstraint, blocked_column_product, \
    blocked_cross_prod, blocked_diag_product, blocked_full_inner_product, blocked_inner_prod, \
    inv_matrix_sqrt


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


def test_linear_constraint():
    r = np.zeros((2, 5))
    r[0, 0] = r[1, 1] = 1
    lc = LinearConstraint(r, require_pandas=False)
    assert np.all(lc.t[:2] == 0)
    assert np.all(np.sum(lc.t, 1)[2:] == 1)
    assert np.all(lc.a == 0)

    x = np.random.randn(200, 5)
    y = np.random.randn(200, 1)
    xt = x @ lc.t
    bc = lstsq(xt, y)[0]
    ec = y - xt @ bc
    b = lstsq(x[:, 2:], y)[0]
    e = y - x[:, 2:] @ b
    assert_allclose(ec.T @ ec, e.T @ e)

    lc = LinearConstraint(r, require_pandas=False)
    assert np.all(lc.a == 0)


def test_linear_constraint_errors():
    r = np.zeros((2, 5))
    r[0, 0] = r[1, 1] = 1
    r_df = pd.DataFrame(r)
    q = np.zeros(2)
    with pytest.raises(TypeError):
        LinearConstraint(r)
    with pytest.raises(TypeError):
        LinearConstraint(r_df, q)
    with pytest.raises(TypeError):
        LinearConstraint([[0, 0, 1]])
    with pytest.raises(ValueError):
        LinearConstraint(r[0], require_pandas=False)
    with pytest.raises(TypeError):
        LinearConstraint(r_df, q)
    with pytest.raises(TypeError):
        LinearConstraint(r_df, [0, 0])


def test_linear_constraint_repr():
    r = np.eye(10)
    lc = LinearConstraint(r, require_pandas=False)
    assert hex(id(lc)) in lc.__repr__()
    assert '10 constraints' in lc.__repr__()
    assert isinstance(lc.q, pd.Series)
    assert np.all(lc.q == 0)
    assert lc.q.shape == (10,)
    assert isinstance(lc.r, pd.DataFrame)
    assert np.all(lc.r == np.eye(10))


def test_blocked_full_inner_product():
    k = 3
    t = 100
    kt = k * t
    x = np.random.randn(kt, kt)
    s = np.eye(k) + 1.0
    i = np.eye(t)
    blocked = blocked_full_inner_product(x, s)
    direct = x.T @ np.kron(s, i) @ x
    assert_allclose(blocked, direct)


def test_blocked_outer_product():
    nobs = 200
    k = 3
    x = [np.random.standard_normal((nobs, k)) for _ in range(k)]
    z = [np.random.standard_normal((nobs, k + 2)) for _ in range(k)]
    scale = np.linspace(1, 2, k)
    s = np.ones(k) * (scale[:, None] @ scale[None, :]) + np.diag(np.arange(1, k + 1))
    actual = blocked_cross_prod(x, z, s)
    _x = []
    _z = []
    for i in range(k):
        xrow = []
        zrow = []
        for j in range(k):
            if i == j:
                xrow.append(x[i])
                zrow.append(z[i])
            else:
                xrow.append(np.zeros((nobs, k)))
                zrow.append(np.zeros((nobs, k + 2)))
        _x.append(np.concatenate(xrow, 1))
        _z.append(np.concatenate(zrow, 1))
    _x = np.concatenate(_x, 0)
    _z = np.concatenate(_z, 0)
    desired = _x.T @ np.kron(s, np.eye(nobs)) @ _z
    assert_allclose(actual, desired)

# TODO: One complex constrain test of equivalence
