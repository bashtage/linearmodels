"""
Opt in to formulas that do not sort variables.

This was the future behavior in linearmodels 4.31. It has been the default, and
importing this module has had no effect, since linearmodels 5.0. The module is
kept so that existing code keeps working and will be removed in linearmodels 8.0.
"""

import warnings

warnings.warn(
    "linearmodels.__future__.ordering has no effect and will be removed in "
    "linearmodels 8.0",
    FutureWarning,
    stacklevel=2,
)
