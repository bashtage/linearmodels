import numpy as np
from numpy.linalg import lstsq
from linearmodels.panel.data import PanelData
from linearmodels.tests.panel._utility import generate_data

data = generate_data(0.00, 'pandas', ntk=(101, 5, 5), other_effects=1, const=False)

# Is weighted demeaning identical to dummy regression?
y = PanelData(data.y)
x = PanelData(data.x)
w = PanelData(data.w)

missing = y.isnull | x.isnull | w.isnull

y.drop(missing)
x.drop(missing)
w.drop(missing)

ydm = y.demean('entity', weights=w)
xdm = x.demean('entity', weights=w)
beta = np.linalg.lstsq(xdm.values2d, ydm.values2d)[0]

root_w = np.sqrt(w.values2d)
d = y.dummies().values
wy = root_w * y.values2d
wx = root_w * x.values2d
wd = root_w * d

wy = wy - wd @ np.linalg.lstsq(wd, wy)[0]
wx = wx - wd @ np.linalg.lstsq(wd, wx)[0]
beta2 = np.linalg.lstsq(wx, wy)[0]
print(beta)
print(beta2)


wy = root_w * y.values2d
wx = root_w * x.values2d
muy = np.l




wd = wd - root_w @ np.linalg.lstsq(root_w, wd)[0]
wy = wy - wd @ np.linalg.lstsq(wd, wy)[0]
wx = wx - wd @ np.linalg.lstsq(wd, wx)[0]
beta3 = np.linalg.lstsq(wx, wy)[0]
print(beta3)
wy = wy - root_w @ np.linalg.lstsq(root_w, wy)[0]
wx = wx - root_w @ np.linalg.lstsq(root_w, wx)[0]
beta4 = np.linalg.lstsq(wx, wy)[0]
print(beta4)

