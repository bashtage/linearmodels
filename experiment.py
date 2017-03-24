from linearmodels.panel.data import PanelData
from linearmodels.tests.panel.test_data import panel

import pandas as pd
data =pd.read_stata(r'C:\git\linearmodels\linearmodels\tests\panel\results\simulated-panel.dta')
data = data.set_index(['firm','time'])
y = data[['y']]
x = data[['intercept','x1','x2','x3','x4','x5']]
from linearmodels.panel.model import PanelOLS

print(PanelOLS(y,x).fit())