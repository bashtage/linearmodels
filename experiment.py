import numpy as np
import pandas as pd
data = pd.read_stata(r'C:\git\linearmodels\linearmodels\iv\tests\results\simulated-data.dta')
from linearmodels.iv import IV2SLS, IVLIML, IVGMM
data['const'] = 1
#ivregress gmm y_robust x3 x4 x5 (x1 = z1 z2)  [aweight=weights] , wmatrix(robust) center
res = IVGMM(data.y_robust, data[['const', 'x3','x4','x5']],data.x1, data[['z1','z2']], weights=data.weights).fit(cov_type='robust')
print(res)

