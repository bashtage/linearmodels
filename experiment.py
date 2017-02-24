from panel.iv.covariance import kernel_optimal_bandwidth
import numpy as np
import pandas as pd
import xarray as xr

x = np.random.standard_normal(1000)
m = kernel_optimal_bandwidth(x)
m_qs = kernel_optimal_bandwidth(x, 'qs')
m_p = kernel_optimal_bandwidth(x, 'parzen')

np.random.seed(12345)
panel = pd.Panel(np.random.randn(125, 200, 10))
panel.iloc[:, :, 0] = np.round(panel.iloc[:, :, 0])
panel.iloc[:, :, 1] = np.round(panel.iloc[:, :, 1])
x = panel
cols = [0, 1]

_x = x.swapaxes(0, 2).to_frame()
orig_cols = _x.columns
demean_cols = []
for df_col in _x:
    if df_col not in cols and pd.core.common.is_numeric_dtype(_x[df_col].dtype):
        demean_cols.append(df_col)

no_change_cols = [col for col in _x if col not in demean_cols]

# Function start
_x = x.swapaxes(0, 2).to_frame()
_x = _x[demean_cols + cols]
# Get original index
index = _x.index
# Reset to RangeIndex to work around GH #13
_x.index = pd.RangeIndex(0, _x.shape[0])
no_change = _x[no_change_cols]
groups = _x.groupby(cols)
means = groups.transform('mean')
out = _x[demean_cols] - means
out = pd.concat([out, no_change], 1)
out = out[orig_cols]

# Setup
np.random.seed(12345)
panel = pd.Panel(np.random.randn(125, 200, 10))
panel.loc[:, :, 0] = np.round(panel.loc[:, :, 0]).astype(np.int64)
panel.loc[:, :, 1] = np.round(panel.loc[:, :, 1]).astype(np.int64)
x = panel
cols = [0, 1]

# Function start
_x = x.swapaxes(0, 2).to_frame()

demean_cols = []
for df_col in _x:
    if df_col not in cols and pd.core.common.is_numeric_dtype(_x[df_col].dtype):
        demean_cols.append(df_col)
no_change_cols = [col for col in _x if col not in demean_cols]

index = _x.index
_x.index = pd.RangeIndex(0, _x.shape[0])

temp = _x[demean_cols + cols]
groups = temp.groupby(cols)
means = groups.mean()
means = means.reindex_like(temp.set_index([0, 1]))
out = pd.DataFrame(temp[demean_cols].values - means.values, columns=demean_cols, index=temp.index)
out = pd.concat([temp, _x[no_change_cols]], 1)
out.index = index

print(calls)

# import numpy as np
# import pandas as pd
#
# data = pd.Panel(np.random.randn(1000, 3, 10))
# temp = data[:, :, 0]
# temp.iloc[0] -= 100
# temp.iloc[1] += 10
# temp.iloc[2] += 20
# time_id = temp.copy()
# for i in range(temp.shape[0]):
#     time_id.iloc[i, :] = i
# entity_id = temp.copy()
# for i in range(temp.shape[1]):
#     entity_id.iloc[:, i] = i
#
# time_dummies = np.zeros((3000, 3))
# time_dummies[:1000, 0] = 1
# time_dummies[1000:2000, 1] = 1
# time_dummies[2000:, 2] = 1
#
# entity_dummies = np.zeros((3000, 1000))
# for i in range(1000):
#     entity_dummies[i, i] = 1
#     entity_dummies[i + 1000, i] = 1
#     entity_dummies[i + 2000, i] = 1
#
# dummies = np.column_stack([time_dummies, entity_dummies])
#
# import statsmodels.api as sm
#
# mod = sm.OLS(temp.values.ravel(), dummies[:, :-1])
# res = mod.fit()

cols = ['var0', 'var1']
data = xr.DataArray(np.random.randn(300, 200, 5), dims=['entity', 'time', 'var'])
data.coords[data.dims[-1]] = xr.Coordinate('var', ['var0', 'var1', 'var2', 'var3', 'var4'])
data[:, :, 0] = np.round(data[:, :, 0])
data[:, :, 1] = np.round(data[:, :, 1])
x = data
coords = x.coords[x.dims[2]]
col_index = [i for i, c in enumerate(coords) if c.data in cols]

for i, c in enumerate(coords):
    print(i, c)
