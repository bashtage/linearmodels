from typing import Any, Union

import numpy as np
import pandas as pd

base_data_types = [np.ndarray, pd.DataFrame, pd.Series]
try:
    import xarray as xr

    ArrayLike = Union[np.ndarray, xr.DataArray, pd.DataFrame, pd.Series]

except ImportError:
    # Always needed to allow optional xarray
    ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


NDArray = Union[np.ndarray]

Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]  # pragma: no cover
Int64Array = np.ndarray[tuple[int, ...], np.dtype[np.int64]]  # pragma: no cover
Int32Array = np.ndarray[tuple[int, ...], np.dtype[np.int32]]  # pragma: no cover
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int_]]  # pragma: no cover
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]  # pragma: no cover
AnyArray = np.ndarray[tuple[int, ...], Any]  # pragma: no cover
Uint32Array = np.ndarray[tuple[int, ...], np.dtype[np.uint32]]  # pragma: no cover
FloatArray1D = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
__all__ = [
    "AnyArray",
    "ArrayLike",
    "BoolArray",
    "Float64Array",
    "FloatArray1D",
    "FloatArray2D",
    "Int32Array",
    "Int64Array",
    "IntArray",
    "Uint32Array",
]
