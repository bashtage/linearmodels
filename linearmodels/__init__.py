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
from typing import List, Optional, Union

from ._version import get_versions
from .asset_pricing.model import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from .iv.absorbing import AbsorbingLS
from .iv.model import _OLS, IV2SLS, IVGMM, IVGMMCUE, IVLIML
from .panel.model import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from .system import IV3SLS, SUR, IVSystemGMM

OLS = _OLS
WARN_ON_MISSING = os.environ.get("LINEARMODELS_WARN_ON_MISSING", True)
WARN_ON_MISSING = False if WARN_ON_MISSING in ("0", "False") else True
DROP_MISSING = os.environ.get("LINEARMODELS_DROP_MISSING", True)
DROP_MISSING = False if DROP_MISSING in ("0", "False") else True

__all__ = [
    "AbsorbingLS",
    "PooledOLS",
    "PanelOLS",
    "FirstDifferenceOLS",
    "BetweenOLS",
    "RandomEffects",
    "FamaMacBeth",
    "IVLIML",
    "IVGMM",
    "IVGMMCUE",
    "IV2SLS",
    "OLS",
    "SUR",
    "IV3SLS",
    "IVSystemGMM",
    "LinearFactorModel",
    "LinearFactorModelGMM",
    "TradedFactorModel",
    "WARN_ON_MISSING",
    "DROP_MISSING",
]


def test(
    extra_args: Optional[Union[str, List[str]]] = None,
    exit: bool = True,
    append: bool = True,
    location: str = "",
) -> int:
    import sys

    try:
        import pytest
    except ImportError:
        raise ImportError("Need pytest to run tests")

    cmd = ["--tb=auto"]
    if extra_args:
        if not isinstance(extra_args, list):
            pytest_args = [extra_args]
        else:
            pytest_args = extra_args
        if append:
            cmd += pytest_args[:]
        else:
            cmd = pytest_args
    print(location)
    pkg = os.path.dirname(__file__)
    print(pkg)
    if location:
        pkg = os.path.abspath(os.path.join(pkg, location))
        print(pkg)
    if not os.path.exists(pkg):
        raise RuntimeError(f"{pkg} was not found. Unable to run tests")
    cmd = [pkg] + cmd
    print("running: pytest {}".format(" ".join(cmd)))
    status = pytest.main(cmd)
    if exit:
        sys.exit(status)
    return status


__version__ = get_versions()["version"]
del get_versions
