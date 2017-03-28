import numpy as np
import pandas as pd
from numpy.linalg import pinv

from itertools import product

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.panel.model import PooledOLS, PanelOLS, BetweenOLS, FirstDifferenceOLS
from linearmodels.tests.panel._utility import generate_data

data = generate_data(0.0, 'pandas')
formula = 'y ~ x0 + x1 + x2 + x3 + x4'
model = PanelOLS


joined = data.x
joined['y'] = data.y
mod = model.from_formula(formula, joined)
res = mod.fit()
parts = formula.split('~')
vars = parts[1].replace(' 1 ', ' const ').split('+')
vars = list(map(lambda s: s.strip(), vars))
x = data.x
res2 = PanelOLS(data.y, x[vars], entity_effect=True).fit()
