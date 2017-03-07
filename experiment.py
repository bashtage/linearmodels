from linearmodels.iv.data import DataHandler
import xarray as xr
import numpy as np

x = np.zeros((1000,10))
for i in range(10):
    x[i::10,i] = 1
from linearmodels.utility import has_constant
has_constant(x)
x[:,0] = 1
has_constant(x)
x[:,0] = 0
x[::17,0] = 1
has_constant(x)

o = xr.DataArray(np.random.randn(10))
t = xr.DataArray(np.random.randn(10,2))
DataHandler(o)

import statsmodels.api as sm
from linearmodels.iv import *
from linearmodels.iv.model import _OLS
import pandas as pd
data = pd.read_stata(r'C:\git\linearmodels\linearmodels\iv\tests\results\simulated-data.dta')
exog = sm.add_constant(data[['x3','x4','x5']])
res = _OLS(data.y_robust,exog).fit()
print(res.summary)

res = IV2SLS(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit()
res.durbin()
print('\n'*4)
print(res.summary)

res = IV2SLS(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit('robust')
print('\n'*4)
print(res.summary)

res = IV2SLS(data.y_robust,exog,data[['x1']],data[['z1','z2']]).fit(debiased=True)
print('\n'*4)
print(res.summary)


res = IVGMM(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit()
print('\n'*4)
print(res.summary)
