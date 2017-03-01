import numpy as np

from panel.iv import IVGMM

n, k, p = 1000, 5, 3
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
params = np.arange(1, k + 1) / k
params = params[:, None]
y = x @ params + e

mod = IVGMM(y, x[:, 2:], x[:, :2], z[:, 3:])
res = mod.fit()
print(res.cov)

mod = IVGMM(np.tile(y, (2, 1)),
            np.tile(x[:, 2:], (2, 1)),
            np.tile(x[:, :2], (2, 1)),
            np.tile(z[:, 3:], (2, 1))g)
res = mod.fit()
print(res.cov)
