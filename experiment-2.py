import numpy as np
from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
from linearmodels import PanelOLS
etdata = data.set_index(['firm','year'])
PanelOLS(etdata.invest,etdata[['value','capital']],entity_effect=True).fit(debiased=True)

import numpy as np

from linearmodels import PanelOLS
from linearmodels.panel.data import PanelData
from linearmodels.tests.panel._utility import generate_data

data = generate_data(0, 'pandas', ntk=(101, 3, 5), other_effects=1, const=False)

y = PanelData(data.y)
x = PanelData(data.x)
w = PanelData(data.w)

x.dataframe.iloc[:,0] = 1
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
