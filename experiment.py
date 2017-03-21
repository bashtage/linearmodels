import numpy as np
import pandas as pd
from linearmodels.panel.data import PanelData
from linearmodels.panel.tests.test_data import panel

data = PanelData(panel())
print(data)
print(data.demean('time'))