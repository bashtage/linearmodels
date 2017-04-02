import numpy as np

from linearmodels.panel.data import PanelData
from linearmodels.tests.panel._utility import generate_data

data = generate_data(0.00, 'pandas', ntk=(101, 5, 5), other_effects=1)

from linearmodels import PanelOLS
mod = PanelOLS(data.y, data.x, other_effects=data.c)
mod.fit()

mod = PanelOLS(data.y, data.x, entity_effect=True, other_effects=data.c).fit()
mod = PanelOLS(data.y, data.x, entity_effect=True, other_effects=data.c).fit()

data = generate_data(0.00, 'pandas', ntk=(101, 5, 5), other_effects=2)
mod = PanelOLS(data.y, data.x, other_effects=data.c).fit()
mod = PanelOLS(data.y, data.x, entity_effect=True, other_effects=data.c)