from linearmodels.iv.data import DataHandler
from linearmodels.iv.results import compare
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

res1 = IV2SLS(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit()
res2 = IV2SLS(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit('robust')
res3 = IV2SLS(data.y_robust,exog,data[['x1']],data[['z1','z2']]).fit(debiased=True)
res4 = IVGMM(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit()

comp = compare([res1,res2,res3,res4])
comp.tstats
comp.params
print(comp.pvalues)
print(comp.rsquared_adj)
print(comp.rsquared)
print(comp.f_statistic)
print(comp.cov_estimator)
print(comp.summary)