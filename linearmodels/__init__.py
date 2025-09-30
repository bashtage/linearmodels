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

from __future__ import annotations

import os
import sys

from ._version import version as __version__, version_tuple
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
WARN_ON_MISSING = os.environ.get("LINEARMODELS_WARN_ON_MISSING", "1")
WARN_ON_MISSING = False if WARN_ON_MISSING in ("", "0", "false", "False") else True
DROP_MISSING = os.environ.get("LINEARMODELS_DROP_MISSING", "1")
DROP_MISSING = False if DROP_MISSING in ("", "0", "false", "False") else True

__all__ = [
    "DROP_MISSING",
    "IV2SLS",
    "IV3SLS",
    "IVGMM",
    "IVGMMCUE",
    "IVLIML",
    "OLS",
    "SUR",
    "WARN_ON_MISSING",
    "AbsorbingLS",
    "BetweenOLS",
    "FamaMacBeth",
    "FirstDifferenceOLS",
    "IVSystemGMM",
    "LinearFactorModel",
    "LinearFactorModelGMM",
    "PanelOLS",
    "PooledOLS",
    "RandomEffects",
    "TradedFactorModel",
    "__version__",
    "version_tuple",
]


def test(
    extra_args: str | list[str] | None = None,
    exit: bool = True,
    append: bool = True,
    location: str = "",
) -> int:

    try:
        import pytest  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Need pytest to run tests") from exc

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
    cmd = [pkg, *cmd]
    print("running: pytest {}".format(" ".join(cmd)))
    status = pytest.main(cmd)
    if exit:  # pragma: no cover
        sys.exit(status)
    return status
