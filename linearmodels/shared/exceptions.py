import numpy as np

from linearmodels.typing import NDArray


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


def missing_warning(missing: NDArray) -> None:
    """Utility function to perform missing value check and warning"""
    if not np.any(missing):
        return
    import linearmodels

    if linearmodels.WARN_ON_MISSING:
        import warnings

        warnings.warn(missing_value_warning_msg, MissingValueWarning)
