from linearmodels.panel.data import PanelData
from linearmodels.tests.panel.test_data import panel

data = PanelData(panel())
print(data)
print(data.demean('time'))