import numpy as np
import pandas as pd

from linearmodels.panel.data import PanelData
from linearmodels.shared.utility import panel_to_frame

np.random.seed(12345)
n, t, k = 1000, 3, 6
x = np.random.randn(k + 1, t, n)
x[0, :, :] = 1
beta = np.arange(1, k + 2) / (k + 1)
eps = np.random.randn(t, n)
beta.shape = (k + 1, 1, 1)
y = (beta * x).sum(0) + eps
y += np.random.randn(1, n)
w = np.random.chisquare(10, size=(1, n)) / 10.0
w = np.ones((t, 1)) @ w
w = w / float(w.mean())

items = ["x" + str(i) for i in range(1, k + 1)]
items = ["intercept"] + items
major = pd.date_range("12-31-1999", periods=t, freq="A-DEC")
minor = ["firm." + str(i) for i in range(1, n + 1)]

x = panel_to_frame(x, items, major, minor, swap=True)
y = panel_to_frame(y[None, :], ["y"], major, minor, swap=True)
w = panel_to_frame(w[None, :], ["w"], major, minor, swap=True)

x_panel_data = PanelData(x)
y_panel_data = PanelData(y)
w_panel_data = PanelData(w)

z = pd.concat(
    [x_panel_data.dataframe, y_panel_data.dataframe, w_panel_data.dataframe],
    1,
    sort=False,
)
final_index = pd.MultiIndex.from_product([minor, major])
final_index.levels[0].name = "firm"
z = z.reindex(final_index)
z.index.levels[0].name = "firm"
z.index.levels[1].name = "time"

z = z.reset_index()
z["firm_id"] = z.firm.astype("category")
z["firm_id"] = z.firm_id.cat.codes

variables = ["y", "x1", "x2", "x3", "x4", "x5"]
missing = 0.05
for v in variables:
    locs = np.random.choice(n * t, int(n * t * missing))
    temp = z[v].copy()
    temp.loc[locs] = np.nan
    z[v + "_light"] = temp

variables = ["y", "x1", "x2", "x3", "x4", "x5"]
missing = 0.20
for v in variables:
    locs = np.random.choice(n * t, int(n * t * missing))
    temp = z[v].copy()
    temp.loc[locs] = np.nan
    z[v + "_heavy"] = temp

z.to_stata("simulated-panel.dta")
