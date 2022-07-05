from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

base_data_types = [np.ndarray, pd.DataFrame, pd.Series]
try:
    import xarray as xr

    ArrayLike = Union[np.ndarray, xr.DataArray, pd.DataFrame, pd.Series]

except ImportError:
    ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]  # type: ignore


NP_GTE_121 = np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("1.21.0")

NDArray = Union[np.ndarray]

if NP_GTE_121 and TYPE_CHECKING:
    Float64Array = np.ndarray[Any, np.dtype[np.float64]]
    Int64Array = np.ndarray[Any, np.dtype[np.int64]]
    Int32Array = np.ndarray[Any, np.dtype[np.int32]]
    IntArray = np.ndarray[Any, np.dtype[np.int_]]
    BoolArray = np.ndarray[Any, np.dtype[np.bool_]]
    AnyArray = np.ndarray[Any, Any]
    Uint32Array = np.ndarray[Any, np.dtype[np.uint32]]
else:
    Uint32Array = (
        IntArray
    ) = Float64Array = Int64Array = Int32Array = BoolArray = AnyArray = NDArray

__all__ = [
    "Float64Array",
    "Int32Array",
    "Int64Array",
    "IntArray",
    "BoolArray",
    "AnyArray",
    "Uint32Array",
    "ArrayLike",
]
