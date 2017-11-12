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


def atleast_k_elem(x, k):
    x = np.array(x)
    if x.ndim == 0:
        x = x * np.ones(k, dtype=np.int)
    return x


def generate_3sls_data(n=500, k=10, p=3, en=2, instr=3, const=True, rho=0.8, kappa=0.5,
                       beta=0.5, common_exog=False, included_weights=False, output_dict=True,
                       seed=1234):
    np.random.seed(seed)
    p = atleast_k_elem(p, k)
    en = atleast_k_elem(en, k)
    instr = atleast_k_elem(instr, k)

    eps = np.random.standard_normal((n, k))
    eps *= np.sqrt(1 - rho ** 2)
    eps += rho * np.random.standard_normal((n, 1))

    data = AttrDict()

    # y, ex, en, z
    # if common_exog, then ex, en, z are constant

    count = 0
    common_shocks = []
    for _p, _en, _instr in zip(p, en, instr):
        total = _p + _en + _instr
        corr = np.eye(_p + _en + _instr + 1)
        corr[_p:_p + _en, _p:_p + _en] = kappa * np.eye(_en)
        corr[_p:_p + _en, -1] = np.sqrt(1 - kappa ** 2) * np.ones(_en)
        corr[_p + _en:_p + _en + _instr, _p:_p + _en] = beta * np.ones((_instr,_en))
        val = np.sqrt(1 - beta ** 2) / _instr * np.eye(_instr)
        corr[_p + _en:_p + _en + _instr, _p + _en:_p + _en + _instr] = val
        if common_exog:
            shocks = np.random.standard_normal((n, total))
            common_shocks = common_shocks if common_shocks is not None else shocks
        else:
            shocks = np.random.standard_normal((n, total))
        shocks = np.concatenate([shocks, eps[:, count:count + 1]],1)
        variables = shocks @ corr.T
        x = variables[:, :_p + _en]
        exog = variables[:, :_p]

        endog = variables[:, _p:_p + _en]
        instr = variables[:, _p + _en:total]
        e = variables[:, total:total + 1]
        if const:
            x = np.c_[np.ones((n, 1)), x]
            exog = np.c_[np.ones((n, 1)), exog]
        assert np.linalg.matrix_rank(x) == x.shape[1]
        params = np.random.chisquare(1, (const + _p + _en, 1))
        dep = x @ params + e
        if included_weights:
            w = np.random.chisquare(5, (n, 1)) / 5
        if _en == 0:
            endog = None
        if _instr == 0:
            _instr = None
        if output_dict:
            data['equ.{0}'.format(count)] = {'dependent': dep, 'exog': exog,
                                             'endog': endog, 'instruments': instr}
            if included_weights:
                data['equ.{0}'.format(count)]['weights'] = w
        else:
            if included_weights:
                data['equ.{0}'.format(count)] = (dep, exog, endog, instr, w)
            else:
                data['equ.{0}'.format(count)] = (dep, exog, endog, instr)
        count += 1

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
    out['xpx'] = xpx
    out['xpy'] = xpy

    return out


def simple_3sls(y, x, z):
    out = AttrDict()
    k = len(y)
    b = []
    eps = []
    xhat = []
    for i in range(k):
        xhat.append(z[i] @ np.linalg.lstsq(z[i], x[i])[0])
        b.append(np.linalg.lstsq(xhat[i], y[i])[0])
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
                row.append(xhat[i])
            else:
                row.append(np.zeros((nobs, xhat[j].shape[1])))
        row = np.hstack(row)
        bx.append(row)
    bx = np.vstack(bx)
    xpx = (bx.T @ omegainv @ bx)
    xpy = (bx.T @ omegainv @ by)
    beta1 = np.linalg.solve(xpx, xpy)
    out['beta1'] = beta1
    out['xpx'] = xpx
    out['xpy'] = xpy
    idx = 0
    eps = []
    for i in range(k):
        k = x[i].shape[1]
        b = beta1[idx:idx+k]
        eps.append(y[i] - x[i] @ b)
        idx += k
    out['eps'] = eps
    return out
