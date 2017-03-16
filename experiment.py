import numpy as np
from statsmodels.api import add_constant
from linearmodels.datasets import mroz
from linearmodels.datasets import card
from linearmodels.iv import IV2SLS, compare

print(card.DESCR)
data = card.load()
data = add_constant(data)
data.dtypes
dep = ['wage']
endog = ['educ']
exog = ['const', 'exper', 'expersq', 'black', 'smsa', 'south', 'smsa66', 'reg662',
        'reg663', 'reg664', 'reg665', 'reg666', 'reg667', 'reg668', 'reg669']
instr = ['nearc4']
data = data[dep + exog + endog + instr].dropna()

res = IV2SLS(data.educ, data[instr + exog], None, None).fit()
print(res.summary)

res_ols = IV2SLS(np.log(data.wage), data[exog+endog],None,None).fit()
res_2sls = IV2SLS(np.log(data.wage), data[exog], data[endog], data[instr]).fit()
print(compare({'OLS':res_ols,'2SLS':res_2sls}))
print(res_2sls.first_stage)

raise ValueError
import numpy as np
from statsmodels.api import add_constant
from linearmodels.datasets import mroz

data = mroz.load()
print(mroz.DESCR)
data = data.dropna()
data = add_constant(data, has_constant='add')

from linearmodels.iv import IV2SLS
res = IV2SLS(np.log(data.wage), data[['const','educ']], None, None).fit('unadjusted')
print(res.summary)
res_first = IV2SLS(data.educ, data[['const','fatheduc']], None, None).fit('unadjusted')
data['educ_hat'] = data.educ - res_first.resids
print(res_first.summary)
res_second = IV2SLS(np.log(data.wage), data[['const']], data.educ, data.fatheduc).fit('unadjusted')
print(res_second.summary)


formula = 'y ~ x1 + x2 + log(x3) + [ x4 + x5 ~ z1 + z4 + exp(z6)] + x9'
formula = 'y ~ x1 + x2 + log(x3) + [ x4 + x5 ~ z1 + z4 + exp(z6)]'



raise ValueError

from linearmodels.iv.data import DataHandler
from linearmodels.iv.results import compare
import xarray as xr
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import *
from linearmodels.iv.model import _OLS
import pandas as pd

data = pd.read_stata(r'C:\git\linearmodels\linearmodels\iv\tests\results\simulated-data.dta')

# res1 = IV2SLS.from_formula('y_robust ~ 1 + x3 + x4 (x1 x2 ~ z1 + z2) + x5', data)
mod= IV2SLS.from_formula('y_robust ~ 1 + x3 + x4 + (x1 x2 ~ z1 + z2) + x5', data)
res = mod.fit()
print(res)
fs = res.first_stage
print(fs)

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

exog = sm.add_constant(data[['x3','x4','x5']])
res = _OLS(data.y_robust,exog).fit()
print(res.summary)

res1 = IV2SLS(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit()
res2 = IV2SLS(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit('robust')
res3 = IV2SLS(data.y_robust,exog,data[['x1']],data[['z1','z2']]).fit(debiased=True)
res4 = IVGMM(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit()

comp = compare([res1,res2,res3,res4])
print(comp)