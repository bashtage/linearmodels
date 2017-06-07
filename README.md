# Linear Models

[![Build Status](https://travis-ci.org/bashtage/linearmodels.svg?branch=master)](https://travis-ci.org/bashtage/linearmodels)
[![codecov](https://codecov.io/gh/bashtage/linearmodels/branch/master/graph/badge.svg)](https://codecov.io/gh/bashtage/linearmodels)



Linear (regression) models for Python.  Extends [statsmodels](http://www.statsmodels.org) to 
include Panel regression and instrumental variable estimators:
 
- **Panel models**:
    - Fixed effects (maximum two-way)
    - First difference regression
    - Between estimator for panel data
    - Pooled regression for panel data
    - Fama-MacBeth estimation of panel models
  
- **Instrumental Variable estimators**
    - Two-stage Least Squares
    - Limited Information Maximum Likelihood
    - k-class Estimators
    - Generalized Method of Moments, also with continuously updating

- **Factor Asset Pricing Models**:
    - 2- and 3-step estimation 
    - Time-series estimation
    - GMM estimation

- **System Regression**:
    - Seemingly Unrelated Regression (SUR/SURE)
       
Designed to work equally well with NumPy, Pandas or xarray data.

### Panel models

Like [statsmodels](http://www.statsmodels.org) to include, supports 
[patsy](https://patsy.readthedocs.io/en/latest/) formulas for specifying models. For example,
 the classic Grunfeld regression can be specified

```python
import numpy as np
from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
# MultiIndex, entity - time
data = data.set_index(['firm','year'])
from linearmodels import PanelOLS
mod = PanelOLS(data.invest, data[['value','capital']], entity_effect=True)
res = mod.fit(cov_type='clustered', cluster_entity=True)
```

Models can also be specified using the formula interface.
 
```python
from linearmodels import PanelOLS
mod = PanelOLS.from_formula('invest ~ value + capital + EntityEffect', data)
res = mod.fit(cov_type='clustered', cluster_entity=True)
```

The formula interface for ``PanelOLS`` supports the special values ``EntityEffects`` 
and ``TimeEffects`` which add entity (fixed) and time effects, respectively. 


### Instrumental Variable Models

IV regression models can be similarly specified.

```python
import numpy as np
from linearmodels.iv import IV2SLS
from linearmodels.datasets import mroz
data = mroz.load()
mod = IV2SLS.from_formula('np.log(wage) ~ 1 + exper + exper ** 2 + [educ ~ motheduc + fatheduc]', data)
```

The expressions in the `[ ]` indicate endogenous regressors (before `~`) and the instruments.  


## Installing

The latest release can be installed using pip

```bash
pip install linearmodels
```

The master branch can be installed by cloning the repo and running setup

```bash
git clone https://github.com/bashtage/linearmodels
cd linearmodels
python setup.py install
```

## Documentation

[Stable Documentation](https://bashtage.github.io/linearmodels/doc) is built on every tagged 
version using [doctr](https://github.com/drdoctr/doctr). 
[Development Documentation](https://bashtage.github.io/linearmodels/devel) is automatically 
built on every successful build of master. 

## Plan and status

Should eventually add some useful linear model estimators such as panel regression. Currently
only the single variable IV estimators are polished.

* Linear Instrumental variable estimation - **complete**
* Linear Panel model estimation - **complete**
* Fama-MacBeth regression - **complete**
* Linear Factor Asset Pricing - **complete**
* System regression - **partially complete** (3SLS not started)
* Linear IV Panel model estimation - *not started*


## Requirements

### Running

With the exception of Python 3.5+, which is a hard requirement, the others are the version 
that are being used in the test environment.  It is possible that older versions work.

* **Python 3.5+**: extensive use of `@` operator
* NumPy (1.11+)
* SciPy (0.17+)
* pandas (0.18+)
* xarray (0.9+)
* statsmodels (0.8+)

### Testing

* py.test

### Documentation

* sphinx
* guzzle_sphinx_theme
* nbsphinx
* nbconvert
* nbformat
* ipython
* jupyter