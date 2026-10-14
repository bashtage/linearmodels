"""
Linear (regression) models for Python.

Extends statsmodels with Panel regression, instrumental variable estimators and
system estimators:

-  Panel models:

   -  Fixed effects (maximum two-way)
   -  First difference regression
   -  Between estimator for panel data
   -  Pooled regression for panel data
   -  Fama-MacBeth estimation of panel models

-  High-dimensional Regression:

   -  Absorbing Least Squares

-  Instrumental Variable estimators

   -  Two-stage Least Squares
   -  Limited Information Maximum Likelihood
   -  k-class Estimators
   -  Generalized Method of Moments, also with continuously updating
   -  Jackknife IV (JIVE)

-  Factor Asset Pricing Models:

   -  2- and 3-step estimation
   -  Time-series estimation
   -  GMM estimation

-  System Regression:

   -  Seemingly Unrelated Regression (SUR/SURE)
   -  Three-Stage Least Squares (3SLS)
   -  System Estimation using Generalized Method of Moments (GMM)

Designed to work equally well with NumPy, Pandas or xarray data.

Notes
-----
A model drops the observations that have a missing value in any of its inputs when
it is created, and issues a ``MissingValueWarning`` if it drops any. The warning
can be turned off in three ways:

-  Set ``linearmodels.WARN_ON_MISSING = False``. It applies to every model that is
   created afterwards.
-  Set the environment variable ``LINEARMODELS_WARN_ON_MISSING`` before importing
   linearmodels. The values ``""``, ``"0"``, ``"false"`` and ``"False"`` turn the
   warning off. Any other value leaves it on.
-  Use a warnings filter, for example
   ``warnings.simplefilter("ignore", MissingValueWarning)``, where
   ``MissingValueWarning`` is imported from ``linearmodels.shared.exceptions``. The
   filter applies where it is set, for example inside ``warnings.catch_warnings()``.

``linearmodels.DROP_MISSING`` and the environment variable
``LINEARMODELS_DROP_MISSING`` have never had any effect, because observations with
missing values are always dropped. They are deprecated and will be removed in
linearmodels 8.0. Reading ``DROP_MISSING``, and importing linearmodels when the
environment variable is set, issue a ``FutureWarning``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING
import warnings

from ._version import version as __version__, version_tuple
from .asset_pricing.model import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from .iv.absorbing import AbsorbingLS
from .iv.model import _OLS, IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML
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
_WARN_ON_MISSING = os.environ.get("LINEARMODELS_WARN_ON_MISSING", "1")
WARN_ON_MISSING = False if _WARN_ON_MISSING in ("", "0", "false", "False") else True

# DROP_MISSING and LINEARMODELS_DROP_MISSING have never had an effect. They are
# deprecated and will be removed in 8.0, together with __getattr__ and the tests of
# the deprecation in tests/test_deprecations.py.
_DROP_MISSING = os.environ.get("LINEARMODELS_DROP_MISSING")
if _DROP_MISSING is not None:
    warnings.warn(
        "The environment variable LINEARMODELS_DROP_MISSING has no effect and will "
        "be removed in linearmodels 8.0. Observations with missing values are always "
        "dropped. Use LINEARMODELS_WARN_ON_MISSING to turn off the "
        "MissingValueWarning.",
        FutureWarning,
        stacklevel=2,
    )

__all__ = [
    "IV2SLS",
    "IV3SLS",
    "IVGMM",
    "IVGMMCUE",
    "IVJIVE",
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

if TYPE_CHECKING:
    # Type checkers must still see DROP_MISSING, and must not accept any other name
    # because of the __getattr__ below
    DROP_MISSING: bool
else:

    def __getattr__(name: str) -> bool:
        # Python calls this function only for names that the module does not define.
        # DROP_MISSING is deliberately not defined, so that reading it warns, and it
        # is not in __all__, so that import * does not read it. Assigning to
        # linearmodels.DROP_MISSING defines it, and then it does not warn.
        if name == "DROP_MISSING":
            warnings.warn(
                "linearmodels.DROP_MISSING has no effect and will be removed in "
                "linearmodels 8.0. Observations with missing values are always "
                "dropped. Use linearmodels.WARN_ON_MISSING to turn off the "
                "MissingValueWarning.",
                FutureWarning,
                stacklevel=2,
            )
            # What 7.0 gave: True, unless the environment variable turned it off
            return _DROP_MISSING not in ("", "0", "false", "False")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def test(
    extra_args: str | list[str] | None = None,
    exit: bool = True,
    append: bool = True,
    location: str = "",
) -> int:
    """
    Run the tests of linearmodels using pytest.

    Parameters
    ----------
    extra_args : str or list of str, optional
        Command line arguments for pytest. A single string is used as one
        argument.
    exit : bool, optional
        If True, the default, exit the interpreter with the status of pytest
        when the tests finish. If False, return the status instead.
    append : bool, optional
        If True, the default, add ``extra_args`` to ``--tb=auto``. If False,
        use only ``extra_args``.
    location : str, optional
        Path, relative to the directory of the package, of the tests to run.
        The default is the empty string, which runs all the tests.

    Returns
    -------
    int
        The exit status of pytest. The function returns only if ``exit`` is
        False.

    Raises
    ------
    ImportError
        If pytest is not installed.
    RuntimeError
        If the directory of the tests does not exist.
    """
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
