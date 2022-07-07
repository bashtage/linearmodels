from typing import cast

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
major = list(pd.date_range("12-31-1999", periods=t, freq="A-DEC"))
minor = [f"firm.{i:0>4}" for i in range(1, n + 1)]

x_df = panel_to_frame(x, items, major, minor, swap=True)
y_df = panel_to_frame(y[None, :], ["y"], major, minor, swap=True)
w_df = panel_to_frame(w[None, :], ["w"], major, minor, swap=True)

x_panel_data = PanelData(x_df)
y_panel_data = PanelData(y_df)
w_panel_data = PanelData(w_df)

z = pd.concat(
    [x_panel_data.dataframe, y_panel_data.dataframe, w_panel_data.dataframe],
    axis=1,
    sort=False,
)
final_index = pd.MultiIndex.from_product([minor, major])
final_index.set_names("firm", level=0)
z = z.reindex(final_index)
idx = cast(pd.MultiIndex, z.index)
idx = idx.set_names(["firm", "time"], level=[0, 1])
z.index = idx

z = z.reset_index()
z["firm_id"] = z.index.get_level_values(0).astype("category")
z["firm_id"] = z.firm_id.cat.codes

variables = ["y", "x1", "x2", "x3", "x4", "x5"]
missing = 0.05
for v in variables:
    locs = np.random.choice(n * t, int(n * t * missing))
    temp = z[v].copy()
    temp.iloc[locs] = np.nan
    z[v + "_light"] = temp

variables = ["y", "x1", "x2", "x3", "x4", "x5"]
missing = 0.20
for v in variables:
    locs = np.random.choice(n * t, int(n * t * missing))
    temp = z[v].copy()
    temp.iloc[locs] = np.nan
    z[v + "_heavy"] = temp

z.to_stata("simulated-panel.dta")
