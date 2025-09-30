import warnings

import numpy as np

from linearmodels.typing import BoolArray


class MissingValueWarning(Warning):
    pass


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
