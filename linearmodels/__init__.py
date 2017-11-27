"""
Linear (regression) models for Python. Extends statsmodels with Panel regression,
instrumental variable estimators and system estimators:

-  Panel models:

   -  Fixed effects (maximum two-way)
   -  First difference regression
   -  Between estimator for panel data
   -  Pooled regression for panel data
   -  Fama-MacBeth estimation of panel models

-  Instrumental Variable estimators

   -  Two-stage Least Squares
   -  Limited Information Maximum Likelihood
   -  k-class Estimators
   -  Generalized Method of Moments, also with continuously updating

-  Factor Asset Pricing Models:

   -  2- and 3-step estimation
   -  Time-series estimation
   -  GMM estimation

-  System Regression:

   -  Seemingly Unrelated Regression (SUR/SURE)
   -  Three-Stage Least Squares (3SLS)
   -  System Estimation using Generalized Method of Moments (GMM)

Designed to work equally well with NumPy, Pandas or xarray data.
"""
import os

from linearmodels.asset_pricing.model import (LinearFactorModel,
                                              LinearFactorModelGMM,
                                              TradedFactorModel)
from linearmodels.iv.model import IV2SLS, IVGMM, IVGMMCUE, IVLIML, _OLS
from linearmodels.panel.model import (BetweenOLS, FirstDifferenceOLS, PanelOLS,
                                      PooledOLS, RandomEffects, FamaMacBeth)
from linearmodels.system import SUR, IV3SLS, IVSystemGMM
from ._version import get_versions

OLS = _OLS
WARN_ON_MISSING = os.environ.get('LINEARMODELS_WARN_ON_MISSING', True)
WARN_ON_MISSING = False if WARN_ON_MISSING in ('0', 'False') else True
DROP_MISSING = os.environ.get('LINEARMODELS_DROP_MISSING', True)
DROP_MISSING = False if DROP_MISSING in ('0', 'False') else True

__all__ = ['PooledOLS', 'PanelOLS', 'FirstDifferenceOLS', 'BetweenOLS',
           'RandomEffects',
           'FamaMacBeth',
           'IVLIML', 'IVGMM', 'IVGMMCUE', 'IV2SLS', 'OLS',
           'SUR', 'IV3SLS', 'IVSystemGMM',
           'LinearFactorModel', 'LinearFactorModelGMM', 'TradedFactorModel',
           'WARN_ON_MISSING', 'DROP_MISSING']

__version__ = get_versions()['version']
del get_versions
