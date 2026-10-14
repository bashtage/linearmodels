"""
Warnings and errors that linearmodels raises.
"""

from collections.abc import Iterable, Mapping, Sequence
import difflib
from typing import Any, TypeVar
import warnings

import numpy as np

from linearmodels.typing import BoolArray

_T = TypeVar("_T")


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
    """
    Warning that an estimator uses the algorithm that saves memory.

    ``PanelOLS`` issues it when a model has entity and time effects, ``low_memory``
    is not given and the dummy variables of the smaller of the two effects would
    need more than 1 GiB.
    """


class InferenceUnavailableWarning(Warning):
    """
    Warning that inference on the parameters is not available.

    ``FamaMacBeth`` issues it if the number of cross-sectional regressions that
    can be estimated is smaller than the number of parameters.
    """


class SingletonWarning(Warning):
    """
    Warning that observations of singleton groups were dropped.

    ``PanelOLS`` issues it when ``singletons=False`` and the model drops
    observations that are the only observation of a group of an effect.
    """


class IndexWarning(Warning):
    """
    Warning that the indices of two inputs do not match.

    ``predict`` issues it if the index of ``exog`` is not the index of ``endog``.
    The predictions use the index of ``exog``.
    """


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
    """
    Warn if any observation has a missing value.

    Parameters
    ----------
    missing : ndarray
        Boolean array that is True for the observations that have a missing value.
    stacklevel : int, optional
        Stack level of the warning. The default is 4.
    """
    if not np.any(missing):
        return
    import linearmodels  # noqa: PLC0415

    if linearmodels.WARN_ON_MISSING:

        warnings.warn(
            missing_value_warning_msg, MissingValueWarning, stacklevel=stacklevel
        )


class UnknownOptionError(ValueError, KeyError):
    """
    Error that a name is not one of the supported names.

    It is raised if the name of a covariance estimator (``cov_type``), of a
    weight matrix estimator (``weight_type``) or of a kernel (``kernel``) is not
    supported. The message gives the parameter, the value that was passed and
    the supported names.

    Notes
    -----
    Releases before 7.1 raised a ``KeyError`` for most of these names and a
    ``ValueError`` for the others. The error is both, so that code that catches
    either of them keeps working. ``str(error)`` is the message. It is not quoted,
    as it is for a ``KeyError``.
    """

    def __str__(self) -> str:
        # KeyError is next in the method resolution order, and its __str__ is the
        # repr of the message
        return Exception.__str__(self)


def unknown_option_error(
    parameter: str,
    value: object,
    supported: Iterable[str],
    *,
    problem: str | None = None,
) -> UnknownOptionError:
    """
    Create the error for a name that is not one of the supported names.

    Parameters
    ----------
    parameter : str
        Name of the parameter, for example ``"cov_type"``.
    value : object
        The value of the parameter, which is not supported.
    supported : iterable of str
        The supported names. The message lists them in sorted order. A name that
        is close to ``value`` is suggested.
    problem : str, optional
        The first sentence of the message. The default says that the value of the
        parameter is unknown.

    Returns
    -------
    UnknownOptionError
        The error, which the caller raises.

    Notes
    -----
    The message is the first sentence, then a suggestion if one of the supported
    names is close to ``value``, and then the list. For example, ``cov_type``
    ``"cluster"`` with the names ``"clustered"`` and ``"robust"`` gives

    .. code-block:: text

        Unknown cov_type: 'cluster'. Did you mean 'clustered'? The supported values
        of cov_type are: 'clustered', 'robust'.
    """
    names = sorted(supported)
    sentences = [problem or f"Unknown {parameter}: {value!r}."]
    if isinstance(value, str):
        close = difflib.get_close_matches(value, names, n=1)
        if close:
            sentences.append(f"Did you mean {close[0]!r}?")
    listed = ", ".join(repr(name) for name in names)
    sentences.append(f"The supported values of {parameter} are: {listed}.")
    return UnknownOptionError(" ".join(sentences))


def lookup_option(table: Mapping[str, _T], value: str, parameter: str) -> _T:
    """
    Return an entry of a table of supported names.

    Parameters
    ----------
    table : dict of str to object
        The supported names and what they select.
    value : str
        The name to look up.
    parameter : str
        Name of the parameter that has the name, for example ``"cov_type"``.

    Returns
    -------
    object
        The entry of ``table`` for ``value``.

    Raises
    ------
    UnknownOptionError
        If ``value`` is not in ``table``. It is a ``KeyError``, as it is for a
        dict, and a ``ValueError``.

    Notes
    -----
    A str is also matched to the names of ``table`` without regard to case.
    """
    try:
        return table[value]
    except (KeyError, TypeError):
        pass
    # The names are not case sensitive
    if isinstance(value, str):
        for name, entry in table.items():
            if name.lower() == value.lower():
                return entry
    raise unknown_option_error(parameter, value, table) from None


def lower_option(value: Any) -> Any:
    """
    Return the name of an option in lower case.

    Parameters
    ----------
    value : object
        The value of an option that is a name, e.g., ``cov_type`` or ``kernel``.

    Returns
    -------
    object
        The lower case of ``value`` if it is a str, and ``value`` otherwise, which
        the caller reports as an unsupported name or type.
    """
    return value.lower() if isinstance(value, str) else value


def check_cov_config(
    cov_config: Mapping[str, object],
    cov_type: str,
    allowed: Mapping[str, Sequence[str]],
    kernels: Mapping[str, object] | None = None,
) -> None:
    """
    Check that the keyword arguments of a covariance estimator are supported.

    Parameters
    ----------
    cov_config : dict
        The keyword arguments of ``fit`` that configure the covariance estimator.
    cov_type : str
        Name of the covariance estimator.
    allowed : dict of str to sequence of str
        The keys that each covariance estimator accepts, by lower-case name.
    kernels : dict, optional
        The supported kernels, which are checked if ``cov_config`` has a
        ``kernel``.

    Raises
    ------
    ValueError
        If a key of ``cov_config`` is not accepted by the estimator. The message
        names the keys that are not accepted and the keys that are. An unknown
        ``cov_type`` is not checked here.
    UnknownOptionError
        If the ``kernel`` is not one of ``kernels``.
    """
    keys = allowed.get(str(cov_type).lower())
    if keys is None:
        return
    unknown = [repr(str(key)) for key in cov_config if key not in keys]
    if unknown:
        if len(unknown) > 1:
            problem = f"The keyword arguments {', '.join(unknown)} are"
        else:
            problem = f"The keyword argument {unknown[0]} is"
        accepted = ", ".join(sorted(keys)) if keys else "none"
        raise ValueError(
            f"{problem} not accepted with cov_type={cov_type!r}. "
            f"The accepted keyword arguments are: {accepted}."
        )
    if kernels is not None and "kernel" in cov_config:
        kernel = cov_config["kernel"]
        # The estimator reports a kernel that is not a str
        if isinstance(kernel, str) and kernel.lower() not in kernels:
            raise unknown_option_error("kernel", kernel.lower(), kernels)


__all__ = [
    "DefaultBandwidthWarning",
    "IndexWarning",
    "InferenceUnavailableWarning",
    "MemoryWarning",
    "MissingValueWarning",
    "SingletonWarning",
    "UnknownOptionError",
    "check_cov_config",
    "lookup_option",
    "lower_option",
    "missing_value_warning_msg",
    "missing_warning",
    "unknown_option_error",
]
