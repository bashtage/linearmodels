from linearmodels.panel.model import BetweenOLS, PanelOLS, FirstDifferenceOLS, PooledOLS
from linearmodels.tests.panel._utility import generate_data

data = generate_data(0.00, 'pandas')

import numpy as np
y = np.arange(12.0)[:,None]
import pandas as pd
entities = pd.Categorical(pd.Series(['a']*6+['b']*6))
dummies = pd.get_dummies(entities)
w = np.random.chisquare(5, (12,1)) / 5
w = w/w.mean()
root_w = np.sqrt(w)
wd = root_w  * dummies.values
wy = root_w  * y
b = np.linalg.pinv(wd) @ wy