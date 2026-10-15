"""
Type aliases for arrays and array-like inputs.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

base_data_types = [np.ndarray, pd.DataFrame, pd.Series]

# A data container is a NumPy array, a pandas Series or DataFrame, or an xarray
# DataArray. An array_like input is also allowed to be a list or a tuple, which is
# converted using np.asarray when it is received. xarray is optional, so the runtime
# alias is built without it if it is not installed.
if TYPE_CHECKING:
    import xarray as xr

    DataContainer = Union[np.ndarray, xr.DataArray, pd.DataFrame, pd.Series]
else:
    try:
        import xarray as xr

        DataContainer = Union[np.ndarray, xr.DataArray, pd.DataFrame, pd.Series]

    except ImportError:

        # Always needed to allow optional xarray
        DataContainer = Union[np.ndarray, pd.DataFrame, pd.Series]

ArrayLike = Union[DataContainer, list[Any], tuple[Any, ...]]

NDArray = Union[np.ndarray]

Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]  # pragma: no cover
Int64Array = np.ndarray[tuple[int, ...], np.dtype[np.int64]]  # pragma: no cover
Int32Array = np.ndarray[tuple[int, ...], np.dtype[np.int32]]  # pragma: no cover
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int_]]  # pragma: no cover
AnyIntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]  # pragma: no cover
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]  # pragma: no cover
AnyArray = np.ndarray[tuple[int, ...], Any]  # pragma: no cover
Uint32Array = np.ndarray[tuple[int, ...], np.dtype[np.uint32]]  # pragma: no cover
FloatArray1D = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
__all__ = [
    "AnyArray",
    "AnyIntArray",
    "ArrayLike",
    "BoolArray",
    "DataContainer",
    "Float64Array",
    "FloatArray1D",
    "FloatArray2D",
    "Int32Array",
    "Int64Array",
    "IntArray",
    "Uint32Array",
]
