from collections.abc import Iterable, Mapping
import difflib
from typing import TypeVar
import warnings

import numpy as np

from linearmodels.typing import BoolArray

_T = TypeVar("_T")


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
        The first sentence of the message. The default is "Unknown cov_type:
        'cluster'." for the parameter ``"cov_type"`` and the value ``"cluster"``.

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
    """
    try:
        return table[value]
    except KeyError:
        raise unknown_option_error(parameter, value, table) from None


__all__ = [
    "IndexWarning",
    "InferenceUnavailableWarning",
    "MemoryWarning",
    "MissingValueWarning",
    "SingletonWarning",
    "UnknownOptionError",
    "lookup_option",
    "missing_value_warning_msg",
    "missing_warning",
    "unknown_option_error",
]
