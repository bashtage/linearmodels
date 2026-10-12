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


class DefaultBandwidthWarning(UserWarning):
    """
    Warning that a kernel estimator uses the bandwidth that it selects by default.

    This warning will be removed in linearmodels 8.0, when the estimated
    bandwidth has been the default for a release.

    Notes
    -----
    The default bandwidth of the kernel estimators of the covariance of the
    moment conditions used by :class:`~linearmodels.iv.model.IVGMM`,
    :class:`~linearmodels.iv.model.IVGMMCUE` and
    :class:`~linearmodels.system.model.IVSystemGMM` changed in linearmodels
    7.1. When neither ``bandwidth`` nor ``optimal_bw`` is given, the bandwidth
    is now estimated from the data. Earlier versions used ``nobs - 2``, which
    is not a consistent estimator of the covariance. The estimates, standard
    errors and J statistic can differ from those of earlier versions.

    The warning is a ``UserWarning``, so it is shown by default. It is issued
    once by each call to ``fit`` that uses the new default for the weight
    matrix (``weight_type="kernel"``) or, in ``IVGMM`` and ``IVGMMCUE``, for
    the parameter covariance (``cov_type="kernel"``), and once by a
    :class:`~linearmodels.iv.gmm.KernelWeightMatrix`,
    :class:`~linearmodels.system.gmm.KernelWeightMatrix` or
    :class:`~linearmodels.iv.gmm.IVGMMCovariance` that is used directly.
    ``IVSystemGMM`` estimated the bandwidth of its kernel covariance from the
    data before, and so only its weight matrix uses a new default.

    Pass ``optimal_bw=False`` or ``bandwidth=nobs - 2`` to restore the
    previous bandwidth. Passing ``bandwidth`` or ``optimal_bw`` silences the
    warning, and ``optimal_bw=True`` keeps the new default.
    """


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
    "DefaultBandwidthWarning",
    "IndexWarning",
    "InferenceUnavailableWarning",
    "MemoryWarning",
    "MissingValueWarning",
    "SingletonWarning",
    "missing_value_warning_msg",
    "missing_warning",
]
