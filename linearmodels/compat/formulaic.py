import sys
import warnings

import formulaic
from packaging.version import parse

FORMULAIC_GTE_0_6 = parse(formulaic.__version__) >= parse("0.6.0")

FUTURE_ORDERING = {"enabled": False}


def future_ordering() -> bool:
    if FUTURE_ORDERING["enabled"]:
        return True
    mods = sys.modules
    FUTURE_ORDERING["enabled"] = future_imported = (
        "linearmodels.__future__.ordering" in mods
    )
    if future_imported and not FORMULAIC_GTE_0_6:
        FUTURE_ORDERING["enabled"] = False
        warnings.warn(
            "Importing ordering from linearmodels.__future__ has no effect if "
            "formulaic is not at least version 0.6.0.",
            RuntimeWarning,
        )
    return FUTURE_ORDERING["enabled"]
