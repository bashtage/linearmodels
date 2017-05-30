import numpy as np

from linearmodels.utility import AttrDict


def generate_data(n=500, k=10, p=3, const=True, rho=0.8, common_exog=False,
                  included_weights=False, output_dict=True, seed=1234):
    np.random.seed(seed)
    p = np.array(p)
    if p.ndim == 0:
        p = [int(p)] * k
    assert len(p) == k

    eps = np.random.standard_normal((n, k))
    eps *= np.sqrt(1 - rho ** 2)
    eps += rho * np.random.standard_normal((n, 1))

    data = AttrDict()

    x = np.random.standard_normal((n, p[0]))
    if const:
        x = np.c_[np.ones((n, 1)), x]

    for i in range(k):
        beta = np.random.chisquare(1, (const + p[i], 1))
        if not common_exog:
            x = np.random.standard_normal((n, p[i]))
            if const:
                x = np.c_[np.ones((n, 1)), x]
        y = x @ beta + eps[:, [i]]
        if included_weights:
            w = np.random.chisquare(5, (n, 1)) / 5
        if output_dict:
            data['equ.{0}'.format(i)] = {'dependent': y, 'exog': x}
            if included_weights:
                data['equ.{0}'.format(i)]['weights'] = w
        else:
            data['equ.{0}'.format(i)] = (y, x)
            if included_weights:
                data['equ.{0}'.format(i)] = tuple(list(data['equ.{0}'.format(i)]) + [w])

    return data


def simple_sur(y, x):
    out = AttrDict()
    k = len(y)
    b = []
    eps = []
    for i in range(k):
        b.append(np.linalg.lstsq(x[i], y[i])[0])
        eps.append(y[i] - x[i] @ b[-1])
    b = np.vstack(b)
    out['beta0'] = b
    out['eps0'] = eps
    eps = np.hstack(eps)
    nobs = eps.shape[0]
    sigma = eps.T @ eps / nobs
    omega = np.kron(sigma, np.eye(nobs))
    omegainv = np.linalg.inv(omega)
    by = np.vstack([y[i] for i in range(k)])
    bx = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(x[i])
            else:
                row.append(np.zeros((nobs, x[j].shape[1])))
        row = np.hstack(row)
        bx.append(row)
    bx = np.vstack(bx)
    xpx = (bx.T @ omegainv @ bx)
    xpy = (bx.T @ omegainv @ by)
    beta1 = np.linalg.solve(xpx, xpy)
    out['beta1'] = beta1
    out['xps'] = xpx
    out['xpy'] = xpy

    return out
