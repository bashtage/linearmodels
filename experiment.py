import numpy as np
from numpy.linalg import lstsq
from linearmodels.panel.data import PanelData
from linearmodels.tests.panel._utility import generate_data
from linearmodels import PanelOLS
from linearmodels import PooledOLS

data = generate_data(0.00, 'pandas', ntk=(101, 5, 5), other_effects=1, const=False)
res = PooledOLS(data.y, data.x).fit()
res2 = PanelOLS(data.y, data.x).fit()
res3 = PanelOLS(data.y, data.x, entity_effect=True).fit()
res4 = PanelOLS(data.y, data.x, entity_effect=True, time_effect=True).fit()