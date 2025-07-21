# Linear Models

| Metric                     |                                                                                                                                                                                                                                                          |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Latest Release**         | [![PyPI version](https://badge.fury.io/py/linearmodels.svg)](https://badge.fury.io/py/linearmodels)                                                                                                                                                      |
| **Continuous Integration** | [![Build Status](https://dev.azure.com/kevinksheppard/kevinksheppard/_apis/build/status/bashtage.linearmodels?branchName=main)](https://dev.azure.com/kevinksheppard/kevinksheppard/_build/latest?definitionId=2&branchName=main)                        |
| **Coverage**               | [![codecov](https://codecov.io/gh/bashtage/linearmodels/branch/main/graph/badge.svg)](https://codecov.io/gh/bashtage/linearmodels)                                                                                                                       |
| **Code Quality**           | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/745a24a69cb2466b95df6a53c83892de)](https://www.codacy.com/manual/bashtage/linearmodels?utm_source=github.com&utm_medium=referral&utm_content=bashtage/linearmodels&utm_campaign=Badge_Grade) |
|                            | [![codebeat badge](https://codebeat.co/badges/aaae2fb4-72b5-4a66-97cd-77b93488f243)](https://codebeat.co/projects/github-com-bashtage-linearmodels-main)                                                                                                 |
| **Citation**               | [![DOI](https://zenodo.org/badge/82291672.svg)](https://zenodo.org/badge/latestdoi/82291672)                                                                                                                                                             |

Linear (regression) models for Python. Extends
[statsmodels](http://www.statsmodels.org) with Panel regression,
instrumental variable estimators, system estimators and models for
estimating asset prices:

-   **Panel models**:
    -   Fixed effects (maximum two-way)
    -   First difference regression
    -   Between estimator for panel data
    -   Pooled regression for panel data
    -   Fama-MacBeth estimation of panel models

-   **High-dimensional Regresssion**:
    -   Absorbing Least Squares

-   **Instrumental Variable estimators**
    -   Two-stage Least Squares
    -   Limited Information Maximum Likelihood
    -   k-class Estimators
    -   Generalized Method of Moments, also with continuously updating

-   **Factor Asset Pricing Models**:
    -   2- and 3-step estimation
    -   Time-series estimation
    -   GMM estimation

-   **System Regression**:
    -   Seemingly Unrelated Regression (SUR/SURE)
    -   Three-Stage Least Squares (3SLS)
    -   Generalized Method of Moments (GMM) System Estimation

Designed to work equally well with NumPy, Pandas or xarray data.

## Panel models

Like [statsmodels](http://www.statsmodels.org) to include, supports
formulas for specifying models. For example, the classic Grunfeld regression can be
specified

```python
import numpy as np
from statsmodels.datasets import grunfeld

data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
# MultiIndex, entity - time
data = data.set_index(['firm', 'year'])
from linearmodels import PanelOLS

mod = PanelOLS(data.invest, data[['value', 'capital']], entity_effects=True)
res = mod.fit(cov_type='clustered', cluster_entity=True)
```

Models can also be specified using the formula interface.

```python
from linearmodels import PanelOLS

mod = PanelOLS.from_formula('invest ~ value + capital + EntityEffects', data)
res = mod.fit(cov_type='clustered', cluster_entity=True)
```

The formula interface for `PanelOLS` supports the special values
`EntityEffects` and `TimeEffects` which add entity (fixed) and time
effects, respectively.

Formula support comes from the [formulaic](https://github.com/matthewwardrop/formulaic/)
package which is a replacement for [patsy](https://patsy.readthedocs.io/en/latest/).

## Instrumental Variable Models

IV regression models can be similarly specified.

```python
import numpy as np
from linearmodels.iv import IV2SLS
from linearmodels.datasets import mroz

data = mroz.load()
mod = IV2SLS.from_formula('np.log(wage) ~ 1 + exper + exper ** 2 + [educ ~ motheduc + fatheduc]', data)
```

The expressions in the `[ ]` indicate endogenous regressors (before `~`)
and the instruments.

## Installing

The latest release can be installed using pip

```bash
pip install linearmodels
```

The main branch can be installed by cloning the repo and running setup

```bash
git clone https://github.com/bashtage/linearmodels
cd linearmodels
pip install .
```

## Documentation

[Stable Documentation](https://bashtage.github.io/linearmodels/) is
built on every tagged version using
[doctr](https://github.com/drdoctr/doctr).
[Development Documentation](https://bashtage.github.io/linearmodels/devel)
is automatically built on every successful build of main.

## Plan and status

Should eventually add some useful linear model estimators such as panel
regression. Currently only the single variable IV estimators are polished.

-   Linear Instrumental variable estimation - **complete**
-   Linear Panel model estimation - **complete**
-   Fama-MacBeth regression - **complete**
-   Linear Factor Asset Pricing - **complete**
-   System regression - **complete**
-   Linear IV Panel model estimation - _not started_
-   Dynamic Panel model estimation - _not started_

## Requirements

### Running

-   Python 3.9+
-   NumPy (1.22+)
-   SciPy (1.8+)
-   pandas (1.4+)
-   statsmodels (0.12+)
-   formulaic (1.0.0+)
-   xarray (0.16+, optional)
-   Cython (3.0.10+, optional)


### Testing

-   py.test

### Documentation

-   sphinx
-   sphinx-immaterial
-   nbsphinx
-   nbconvert
-   nbformat
-   ipython
-   jupyter
