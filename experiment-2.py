from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from linearmodels.datasets import wage_panel
import statsmodels.api as sm
data = wage_panel.load()
data = data.set_index(['nr','year'])
dependent = data.lwage
exog = sm.add_constant(data[['expersq','married','union']])
mod = PanelOLS(dependent, exog, entity_effect=True, time_effect=True)
res = mod.fit(cov_type='unadjusted')
res2 = mod.fit(cov_type='robust')
exog = sm.add_constant(data[['exper', 'expersq','married','union']])
mod = PanelOLS(dependent, exog, entity_effect=True)
res3 = mod.fit(cov_type='clustered',cluster_entity=True)
mod = RandomEffects(dependent, exog)
res4 = mod.fit(cov_type='robust')
from linearmodels.panel.results import compare

exog = sm.add_constant(data[['exper', 'expersq','married','union']].copy())
import pandas as pd
exog['year'] = pd.Categorical(data.reset_index()['year'])
mod = PooledOLS(dependent, exog)
res5 = mod.fit(cov_type='robust')
print(compare([res,res2, res3, res4, res5]))

print(data.columns)


from linearmodels.panel.data import PanelData
import numpy as np
import pandas as pd
from statsmodels.datasets import grunfeld

from linearmodels.panel.model import RandomEffects
from linearmodels.tests.panel._utility import generate_data

data = pd.read_stata(r'C:\git\linearmodels\linearmodels\tests\panel\results\simulated-panel.dta')
data= data.set_index(['firm','time'])
mod = RandomEffects(data.y, data[['intercept', 'x1','x2','x3','x4','x5']])
res = mod.fit()
print(res.variance_decomposition)
res = mod.fit(small_sample=True)
print(res.variance_decomposition)


mod = RandomEffects(data.y_light, data[['intercept', 'x1_light','x2_light','x3_light','x4_light','x5_light']])
res = mod.fit()

import numpy as np
from statsmodels.datasets import grunfeld

data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
from linearmodels import PanelOLS

etdata = data.set_index(['firm', 'year'])
mod2 = PanelOLS(etdata.invest, etdata[['value', 'capital']], entity_effect=True)
res2 = mod2.fit(debiased=True)

import numpy as np

x = np.random.randn(10, 2)
p = np.zeros((10, 10))
p[:5, :5] = 1 / 5
p[5:, 5:] = 1 / 5
z = np.zeros((10, 2))
z[:5, 0] = 1
z[5:, 1] = 1
a = x.T @ p @ x
b = (x.T @ z) @ (x.T @ z).T

import numpy as np
from statsmodels.datasets import grunfeld

data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
from linearmodels import PanelOLS

etdata = data.set_index(['firm', 'year'])
res = PanelOLS(etdata.invest, etdata[['value', 'capital']], entity_effect=True).fit(debiased=True)
x = np.random.randn(10, 2)
p = np.zeros((10, 10))
p[:5, :5] = 1 / 5
p[5:, 5:] = 1 / 5
p
x
x.T @ p
x.T @ p @ x
x
x.T @ p @ x
x.mean(0)
x.sum(0)
x.sum(0)[:, None].T
x.sum(0)[:, None]
x.sum(0)[:, None] @ x.sum(0)[:, None].T
x.sum(0)[:, None] @ x.sum(0)[:, None].T / 5
p
x.T @ p
(x.T @ p).T
x.T @ ((x.T @ p).T)
x[:5].mean(0)
pd.DataFrame(x)
import pandas as pd

pd.DataFrame(x, index=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
z = pd.DataFrame(x, index=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
z.groupby(level=0).transform('mean')
z.groupby(level=0).transform('mean').values.T @ x
z = np.zeros((10, 2))
z[:5, 0] = 1
z[5:, 1] = 1
z
x.T @ z
x.sum()
x.sum(0)
x[:5].sum(0)
zz = pd.DataFrame(x, index=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
zz.groupby(level=0).sum()
zz.groupby(level=0).sum().T
o = zz.groupby(level=0).sum().values
o
o.T @ o
(x.T @ z) @ (x.T @ z).T
a = x.T @ p @ x
b = (x.T @ z) @ (x.T @ z).T
a
b
np.linalg.inv(a) @ b
np.trace(np.linalg.inv(a) @ b)
30
30

data = generate_data(0, 'pandas', ntk=(101, 3, 5), other_effects=1, const=False)

y = PanelData(data.y)
x = PanelData(data.x)
w = PanelData(data.w)

x.dataframe.iloc[:, 0] = 1
mod = PanelOLS(data.y, data.x, weights=data.w)
mod.fit()
mod = PanelOLS(y, x, weights=data.w, entity_effect=True)
mod.fit()
mod = PanelOLS(data.y, data.x, weights=data.w, time_effect=True)
mod.fit()
mod = PanelOLS(data.y, data.x, weights=data.w, time_effect=True, entity_effect=True)
mod.fit()

missing = y.isnull | x.isnull | w.isnull
y.drop(missing)
x.drop(missing)
w.drop(missing)

x.dataframe.iloc[:, 0] = 1
ydw = y.demean(weights=w)
xdw = x.demean(weights=w)
d = x.dummies('entity', drop_first=False)
root_w = np.sqrt(w.values2d)
wd = root_w * d
wdx_direct = root_w * x.values2d - wd @ np.linalg.lstsq(wd, root_w * x.values2d)[0]
print(np.abs(wdx_direct[0] - xdw.values2d[0]) > 1e-14)

mux = (w.values2d * x.values2d).sum(0) / w.values2d.sum()
muy = (w.values2d * y.values2d).sum(0) / w.values2d.sum()
xx = xdw.values2d + root_w * mux
yy = ydw.values2d + root_w * muy.squeeze()
print(np.linalg.lstsq(xx, yy)[0])

yyy = root_w * y.values2d
xxx = root_w * x.values2d
ddd = root_w * x.dummies(drop_first=True)
zzz = root_w * np.ones_like(y.values2d)
ddd = ddd - zzz @ np.linalg.lstsq(zzz, ddd)[0]
xxx = xxx - ddd @ np.linalg.lstsq(ddd, xxx)[0]
yyy = yyy - ddd @ np.linalg.lstsq(ddd, yyy)[0]
print(np.linalg.lstsq(xxx, yyy)[0])

xdw = x.demean('time', weights=w)
d = x.dummies('time', drop_first=False)
root_w = np.sqrt(w.values2d)
wd = root_w * d
wdx_direct = root_w * x.values2d - wd @ np.linalg.lstsq(wd, root_w * x.values2d)[0]
print(np.abs(wdx_direct[0] - xdw.values2d[0]) > 1e-14)

# 1. Weighted demean equity, time and both
# 2. Weighted MAC for 2 dimensions

xdw = x.demean('both', weights=w)
d1 = x.dummies('entity', drop_first=False)
d2 = x.dummies('time', drop_first=True)
d = np.c_[d1, d2]
root_w = np.sqrt(w.values2d)
wd = root_w * d
wdx_direct = root_w * x.values2d - wd @ np.linalg.lstsq(wd, root_w * x.values2d)[0]
print(np.abs(wdx_direct[0] - xdw.values2d[0]) > 1e-14)

xm = root_w * x.values2d.copy()
d1 = x.dummies('entity', drop_first=False)
d2 = x.dummies('time', drop_first=False)
root_w = np.sqrt(w.values2d)
wd1 = root_w * d1
wd2 = root_w * d2
for i in range(50):
    xm -= wd1 @ np.linalg.lstsq(wd1, xm)[0]
xm -= wd2 @ np.linalg.lstsq(wd2, xm)[0]

print(np.max(np.abs(xdw.values2d - xm)))
