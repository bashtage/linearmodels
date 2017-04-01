import numpy as np

from linearmodels.panel.data import PanelData
from linearmodels.tests.panel._utility import generate_data

data = generate_data(0.00, 'pandas', ntk=(37, 5, 5))

# Is weighted dummy demeaning the same as simple weighted demaneing and adding back in weighted average
# Might need ot readd weighted mean?
y_pd = PanelData(data.y)
y = y_pd.dataframe
w = PanelData(data.w).dataframe
x = np.ones_like(y)
d = y_pd.dummies('entity', drop_first=True).values

wd = np.sqrt(w.values) * d
wx = np.sqrt(w.values) * x
wy = np.sqrt(w.values) * y.values

_wy = wy - wd @ np.linalg.lstsq(wd, wy)[0]
_wx = wx - wd @ np.linalg.lstsq(wd, wx)[0]
print(float((_wx.T @ _wy) / (_wx.T @ _wx)))
_wz = np.c_[wx,wd]

print(np.linalg.lstsq(_wz, wy)[0][0])
_wd = wd - wx @ np.linalg.lstsq(wx, wd)[0]