"""
Conversion of the array_like inputs of the estimators.

A parameter that is documented as array_like accepts a NumPy array, a pandas
Series or DataFrame, an xarray DataArray and anything else that NumPy converts
to an array, such as a list or a tuple. The estimators use ``ndim`` and
``shape`` of the inputs, so everything that is not already an array is
converted with ``np.asarray`` when it is received.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload
import warnings

import numpy as np
from pandas import DataFrame, Series
from pandas.api.types import is_complex_dtype

if TYPE_CHECKING:
    from xarray import DataArray

    from linearmodels.iv.data import IVData
    from linearmodels.panel.data import PanelData

try:
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:  # NumPy < 1.25
    # The stubs of NumPy 2 do not have the name in the main namespace
    VisibleDeprecationWarning = np.VisibleDeprecationWarning  # type: ignore[misc,attr-defined]

__all__ = [
    "as_array_like",
    "as_float64",
    "as_ndarray",
    "is_data_array",
    "reject_complex",
]

# The containers that as_array_like returns unchanged
_Container = TypeVar(
    "_Container", np.ndarray, Series, DataFrame, "DataArray", "IVData", "PanelData"
)

COMPLEX_MSG = (
    "{name} has a complex dtype. Complex values are not supported since the "
    "imaginary part would be discarded. Use real values, for example the real "
    "part of the array."
)
SCALAR_MSG = (
    "{name} must be array_like, such as a list, a tuple, a NumPy array, a pandas "
    "Series or DataFrame or an xarray DataArray. Got {kind}."
)
STR_MSG = (
    " To use a column of a DataFrame, pass the values of the column, for "
    "example data['{value}']."
)


def is_data_array(value: object) -> bool:
    """
    Whether value is an xarray DataArray, if xarray is installed.

    Parameters
    ----------
    value : object
        The object to check.

    Returns
    -------
    bool
        True if value is a DataArray.
    """
    try:
        from xarray import DataArray  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(value, DataArray)


def reject_complex(value: object, name: str) -> None:
    """
    Raise a TypeError if the data in value have a complex dtype.

    NumPy discards the imaginary part when it converts complex values to
    float64, and only emits a ComplexWarning. This is never what is wanted.

    Parameters
    ----------
    value : object
        The input. Only NumPy arrays, Series, DataFrames and DataArrays are
        checked, since the other containers are converted to float64 on entry.
    name : str
        The name of the input to use in the message.

    Raises
    ------
    TypeError
        If the dtype of value, or of any of its columns, is complex.
    """
    if isinstance(value, DataFrame):
        complex_data = any(is_complex_dtype(dtype) for dtype in value.dtypes)
    elif isinstance(value, (np.ndarray, Series)) or is_data_array(value):
        complex_data = is_complex_dtype(value.dtype)  # type: ignore[attr-defined]
    else:
        return
    if complex_data:
        raise TypeError(COMPLEX_MSG.format(name=name))


@overload
def as_array_like(  # numpydoc ignore=GL08
    value: list[Any] | tuple[Any, ...], name: str
) -> np.ndarray: ...


@overload
def as_array_like(  # numpydoc ignore=GL08
    value: _Container, name: str
) -> _Container: ...


def as_array_like(value: Any, name: str) -> Any:
    """
    Convert an array_like input to an array unless it is a known container.

    Parameters
    ----------
    value : array_like
        The input. NumPy arrays, pandas objects, xarray DataArrays, IVData and
        PanelData are returned unchanged. Everything else, such as a list or a
        tuple, is converted using ``np.asarray``.
    name : str
        The name of the input to use in the messages of the errors.

    Returns
    -------
    object
        The input if it is a known container, or an ndarray.

    Raises
    ------
    TypeError
        If value is not array_like, e.g., it is a scalar, or if it contains
        complex values.
    ValueError
        If a nested sequence is ragged.
    """
    from linearmodels.iv.data import IVData  # noqa: PLC0415
    from linearmodels.panel.data import PanelData  # noqa: PLC0415

    if isinstance(
        value, (np.ndarray, Series, DataFrame, IVData, PanelData)
    ) or is_data_array(value):
        reject_complex(value, name)
        return value
    try:
        with warnings.catch_warnings():
            # NumPy < 1.24 warns and makes an object array of the rows of a ragged
            # sequence instead of raising
            warnings.simplefilter("ignore", VisibleDeprecationWarning)
            out = np.asarray(value)
        if out.dtype == object and any(
            isinstance(item, (list, tuple, np.ndarray)) for item in out.flat
        ):
            raise ValueError("the nested sequences have different lengths")
    except ValueError as exc:
        raise ValueError(
            f"{name} could not be converted to an array. Nested sequences must "
            f"have the same length. The error was: {exc}"
        ) from exc
    if out.ndim == 0:
        msg = SCALAR_MSG.format(name=name, kind=type(value).__name__)
        if isinstance(value, str):
            msg += STR_MSG.format(value=value)
        raise TypeError(msg)
    reject_complex(out, name)
    return out


def as_ndarray(value: Any, name: str) -> np.ndarray:
    """
    Convert an array_like input that is not used with labels to an ndarray.

    Parameters
    ----------
    value : array_like
        The input. The values of a pandas object or of an xarray DataArray are
        used, and so are the values of IVData and PanelData.
    name : str
        The name of the input to use in the messages of the errors.

    Returns
    -------
    ndarray
        The values of the input.

    Raises
    ------
    TypeError
        If value is not array_like, or if it contains complex values.
    """
    from linearmodels.iv.data import IVData  # noqa: PLC0415
    from linearmodels.panel.data import PanelData  # noqa: PLC0415

    value = as_array_like(value, name)
    if isinstance(value, IVData):
        return value.ndarray
    if isinstance(value, PanelData):
        return value.values2d
    return np.asarray(value)


def as_float64(value: np.ndarray, name: str) -> np.ndarray:
    """
    Convert an array to float64 with an error that names the input.

    Parameters
    ----------
    value : ndarray
        The array to convert. Complex values should have been rejected before.
    name : str
        The name of the input to use in the message.

    Returns
    -------
    ndarray
        The array as float64. It is a copy of the input.

    Raises
    ------
    TypeError
        If the values cannot be converted to float64, e.g., because they are
        text.
    """
    try:
        return value.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must contain numbers, but it cannot be converted to float64: "
            f"{exc}"
        ) from exc
