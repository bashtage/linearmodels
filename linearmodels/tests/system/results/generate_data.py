"""
Important cases

1. Base
2. Small sample adjustement
3. Constraints across equations
"""

import numpy as np
import pandas as pd

from linearmodels.tests.system._utility import generate_data

basic_data = generate_data(n=200, k=3, p=[2, 3, 4], const=True, seed=0)
common_data = generate_data(n=200, k=3, p=3, common_exog=True, seed=1)
missing_data = generate_data(n=200, k=3, p=[2, 3, 4], const=True, seed=2)

np.random.seed(1234)
for key in missing_data:
    dep = missing_data[key]['dependent']
    locs = np.where(np.random.random_sample(dep.shape[0]) < 0.02)[0]
    if np.any(locs):
        dep.flat[locs] = np.nan
    exog = missing_data[key]['exog']
    locs = np.where(np.random.random_sample(np.prod(exog.shape)) < 0.02)[0]
    if np.any(locs):
        exog.flat[locs] = np.nan

out = []
for i, dataset in enumerate((basic_data, common_data, missing_data)):
    base = 'mod_{0}'.format(i)
    for j, key in enumerate(dataset):
        dep = dataset[key]['dependent']
        dep = pd.DataFrame(dep, columns=[base + '_y_{0}'.format(j)])
        dataset[key]['dependent'] = dep
        exog = dataset[key]['exog'][:, 1:]
        exog_cols = [base + '_x_{0}{1}'.format(j, k) for k in range(exog.shape[1])]
        exog = pd.DataFrame(exog, columns=exog_cols)
        exog = exog.copy()
        exog['cons'] = 1.0
        dataset[key]['exog'] = exog
        if i != 1 or j == 0:
            out.extend([dep, exog])
        else:
            out.extend([dep])


if __name__ == '__main__':
    df = pd.concat(out, 1)
    df.to_stata('simulated-sur.dta')
