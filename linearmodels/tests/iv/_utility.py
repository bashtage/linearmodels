import numpy as np

from linearmodels.utility import AttrDict


def generate_data(nkp=(1000, 5, 3)):
    n, k, p = nkp
    np.random.seed(12345)
    clusters = np.random.randint(0, 10, n)
    rho = 0.5
    r = np.zeros((k + p + 1, k + p + 1))
    r.fill(rho)
    r[-1, 2:] = 0
    r[2:, -1] = 0
    r[-1, -1] = 0.5
    r += np.eye(9) * 0.5
    v = np.random.multivariate_normal(np.zeros(r.shape[0]), r, n)

    x = v[:, :k]
    z = v[:, 2:k + p]
    e = v[:, [-1]]
    endog = x[:, :2]
    exog = x[:, 2:]
    instr = z[:, k - 2:]
    params = np.arange(1, k + 1) / k
    params = params[:, None]
    y = x @ params + e
    dep = y
    xhat = z @ np.linalg.pinv(z) @ x
    nobs, nvar = x.shape
    s2 = e.T @ e / nobs
    s2_debiased = e.T @ e / (nobs - nvar)
    v = xhat.T @ xhat / nobs
    vinv = np.linalg.inv(v)
    kappa = 0.99
    vk = (x.T @ x * (1 - kappa) + kappa * xhat.T @ xhat) / nobs
    xzizx = x.T @ z @ z.T @ x / nobs
    xzizx_inv = np.linalg.inv(xzizx)

    return AttrDict(nobs=nobs, e=e, x=x, y=y, z=z, xhat=xhat,
                    params=params, s2=s2, s2_debiased=s2_debiased,
                    clusters=clusters, nvar=nvar, v=v, vinv=vinv, vk=vk,
                    i=np.eye(k + p - 2), kappa=kappa,
                    xzizx=xzizx, xzizx_inv=xzizx_inv,
                    dep=dep, exog=exog, endog=endog, instr=instr)
