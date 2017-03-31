"""
Simulation of test data to compare with Stata

Truth has 2 endog regressors and 2 instruments available.

Simulation designed to allow following models to be estimated:

2 endog, 2 IV
1 endog, 2 IV
1 endog, 1 IV

Will also test other configurations - small, covariance-estimators, constant

"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import multivariate_normal, seed

seed(12345)

# Layout: e - endog reg, x - exog reg, z - instr
# e1 e2 x1 x2 x3 z1 z2 e
k, p, n = 5, 2, 600
r = np.empty((k + p + 1, k + p + 1))
r[:, :] = 0.5
r[p:k + p, -1] = r[-1, p:k + 1 + p] = 0
r[-1, -1] = 0.5
r += 0.5 * np.eye(k + p + 1)

w = multivariate_normal(np.zeros(k + p + 1), r, n)
x = w[:, :k]
z = w[:, k:k + p]
e = w[:, -1]
x = sm.add_constant(x)
beta = np.arange(k + 1) / k
beta[0] = 1
e_homo = e
y_unadjusted = x @ beta[:, None] + e[:, None]

e_orig = e.copy()
scale = (x ** 2).sum(1)
scale = n * scale / scale.sum()
e *= scale
e /= e.std()
e_hetero = e
y_robust = x @ beta[:, None] + e[:, None]

e = e_orig.copy()
cluster_size = 5
r = 0.5 * np.ones((cluster_size, cluster_size))
r += 0.5 * np.eye(cluster_size)
rsqrt = np.linalg.cholesky(r)
for i in range(0, len(r), 5):
    e[i:i + 5] = (rsqrt @ e[i:i + 5][:, None]).squeeze()
e_cluster = e
clusters = np.tile(np.arange(n // 5)[None, :], (5, 1)).T.ravel()

y_clustered = x @ beta[:, None] + e[:, None]

e = e_orig.copy()
u = e.copy()
for i in range(2, n):
    e[i] = 0.8 * u[i - 1] + 0.4 * u[i - 2] + u[i]

y_kernel = x @ beta[:, None] + e[:, None]
e_autoc = e

weights = np.random.chisquare(10, size=y_kernel.shape) / 10
weights = weights / weights.mean()

time = np.arange(n)
data = np.c_[time, y_unadjusted, y_robust, y_clustered, y_kernel, x, z, e_homo, e_hetero,
             e_cluster, e_autoc, clusters, weights]
data = pd.DataFrame(data, columns=['time', 'y_unadjusted', 'y_robust', 'y_clustered',
                                   'y_kernel', '_cons', 'x1', 'x2', 'x3',
                                   'x4', 'x5', 'z1', 'z2', 'e_homo', 'e_hetero', 'e_cluster',
                                   'e_autoc', 'cluster_id', 'weights'])
data.to_stata('simulated-data.dta')
