# One

[![Build Status](https://travis-ci.org/bashtage/linearmodels.svg?branch=master)](https://travis-ci.org/bashtage/linearmodels) 
[![codecov](https://codecov.io/gh/bashtage/linearmodels/branch/master/graph/badge.svg)](https://codecov.io/gh/bashtage/linearmodels)


# Linear Models

Linear (regression) models for Python.  Extends [statsmodels](http://www.statsmodels.org) to 
include instrumental variable estimators:
 
  * Two-stage Least Squares
  * Limited Information Maximum Likelihood
  * k-class Estimators
  * Generalized Method of Moments, also with continuously updating
  
Designed to work equally well with NumPy, Pandas or xarray data.

Like [statsmodels](http://www.statsmodels.org) to include, supports 
[patsy](https://patsy.readthedocs.io/en/latest/) formulas for specifying models. For example, 

```python
import numpy as np
from linearmodels.iv import IV2SLS
from linearmodels.datasets import mroz
data = mroz.load()
mod = IV2SLS.from_formula('np.log(wage) ~ 1 + exper + exper ** 2 + [educ ~ motheduc + fatheduc]', data)
```

The expressions in the `[ ]` indicate endogenous regressors (before `~`) and the instruments.  

## Documentation

[Documentation](https://bashtage.github.io/linearmodels/doc) is automatically built using 
[doctr](https://github.com/drdoctr/doctr) on every successful build of master. The documentation 
is still rough but should improve quickly. 

## Plan and status

Should eventually add some useful linear model estimators such as panel regression. Currently
only the single variable IV estimators are polished.

* Linear Instrumental variable estimation - *complete*
* Linear Panel model estimation - *incomplete*
* Linear IV Panel model estimation - *not started*
* System regression - *not started*

## Requirements

### Running

With the exception of Python 3.5+, which is a hard requirement, the others are the version 
that are being used in the test environment.  It is possible that older versions work.

* **Python 3.5+**: extensive use of `@` operator
* NumPy (1.11+)
* SciPy (0.17+)
* Pandas (0.19+)
* xarray (0.9+)
* Statsmodels (0.8+)

### Testing

* py.test

### Documentation

* sphinx
* sphinx_rtd_theme
* nbsphinx
* nbconvert
* nbformat
* ipython
* jupyter