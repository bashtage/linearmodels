import statsmodels.api as sm
from linearmodels.iv import *
import pandas as pd
data = pd.read_stata(r'C:\git\linearmodels\linearmodels\iv\tests\results\simulated-data.dta')
exog = sm.add_constant(data[['x3','x4','x5']])
res = IV2SLS(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit()
res.durbin()
print(res.durbin())
print(res.durbin(['x1']))
print(res.durbin(['x2']))

print(res.wu_hausman())
print(res.wu_hausman(['x1']))
print(res.wu_hausman(['x2']))

res = IV2SLS(data.y_robust,exog,data[['x1','x2']],data[['z1','z2']]).fit('robust')
print(res.wooldridge_score)
print(res.wooldridge_regression)


res = IV2SLS(data.y_robust,exog,data[['x1']],data[['z1','z2']]).fit()
res.durbin()
print(res.sargan)
print(res.basmann)

