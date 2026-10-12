import warnings

import numpy as np

from linearmodels.typing import BoolArray


class MissingValueWarning(Warning):
    """
    Warning that observations with missing values were dropped.

    Notes
    -----
    A model drops the observations that have a missing value in any of its inputs
    when it is created, and issues this warning if it drops any. The warning is on
    by default. Set ``linearmodels.WARN_ON_MISSING = False`` or the environment
    variable ``LINEARMODELS_WARN_ON_MISSING=0`` to turn it off for every model. To
    change it for part of a program, use a warnings filter for this class, with the
    action ``"ignore"`` or ``"error"``.
    """


missing_value_warning_msg = """
Inputs contain missing values. Dropping rows with missing observations."""


class MemoryWarning(Warning):
    pass


class InferenceUnavailableWarning(Warning):
    pass


class SingletonWarning(Warning):
    pass


class IndexWarning(Warning):
    pass


def missing_warning(missing: BoolArray, stacklevel: int = 4) -> None:
    """Utility function to perform missing value check and warning"""
    if not np.any(missing):
        return
    import linearmodels  # noqa: PLC0415

    if linearmodels.WARN_ON_MISSING:

        warnings.warn(
            missing_value_warning_msg, MissingValueWarning, stacklevel=stacklevel
        )


__all__ = [
    "IndexWarning",
    "InferenceUnavailableWarning",
    "MemoryWarning",
    "MissingValueWarning",
    "SingletonWarning",
    "missing_value_warning_msg",
    "missing_warning",
]
