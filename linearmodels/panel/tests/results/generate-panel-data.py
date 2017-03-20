import numpy as np
import pandas as pd

from linearmodels.panel.data import PanelDataHandler

np.random.seed(12345)
n, t, k = 1000, 3, 6
x = np.random.randn(k + 1, t, n)
x[0, :, :] = 1
beta = np.arange(1, k + 2) / (k + 1)
eps = np.random.randn(t, n)
beta.shape = (k + 1, 1, 1)
y = (beta * x).sum(0) + eps
y += np.random.randn(1, n)

items = ['x' + str(i) for i in range(1, k + 1)]
items = ['intercept'] + items
major = pd.date_range('12-31-1999', periods=t, freq='A-DEC')
minor = ['firm.' + str(i) for i in range(1, n + 1)]

x = pd.Panel(x, items=items, major_axis=major, minor_axis=minor)
x.major_axis.name = 'time'
x.minor_axis.name = 'firm'

y = pd.DataFrame(y, index=major, columns=minor)
y = pd.Panel({'y': y})
y.major_axis.name = 'time'
y.minor_axis.name = 'firm'

x = PanelDataHandler(x)
y = PanelDataHandler(y)
z = pd.concat([x.dataframe, y.dataframe], 1)
z = z.reset_index()
z['firm_id'] = z.firm.astype('category')
z['firm_id'] = z.firm_id.cat.codes

vars = ['y', 'x1', 'x2', 'x3', 'x4', 'x5']
primes = [11, 13, 17, 19, 23, 29]
for p, v in zip(primes, vars):
    temp = z[v].copy()
    temp.iloc[::p] = np.nan
    z[v + '_missing'] = temp

z.to_stata('simulated-panel.dta')
