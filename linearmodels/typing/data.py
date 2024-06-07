from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

base_data_types = [np.ndarray, pd.DataFrame, pd.Series]
try:
    import xarray as xr

    ArrayLike = Union[np.ndarray, xr.DataArray, pd.DataFrame, pd.Series]

except ImportError:
    # Always needed to allow optional xarray
    ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]  # type: ignore


NDArray = Union[np.ndarray]

if TYPE_CHECKING:
    Float64Array = np.ndarray[Any, np.dtype[np.float64]]  # pragma: no cover
    Int64Array = np.ndarray[Any, np.dtype[np.int64]]  # pragma: no cover
    Int32Array = np.ndarray[Any, np.dtype[np.int32]]  # pragma: no cover
    IntArray = np.ndarray[Any, np.dtype[np.int_]]  # pragma: no cover
    BoolArray = np.ndarray[Any, np.dtype[np.bool_]]  # pragma: no cover
    AnyArray = np.ndarray[Any, Any]  # pragma: no cover
    Uint32Array = np.ndarray[Any, np.dtype[np.uint32]]  # pragma: no cover
else:
    Uint32Array = IntArray = Float64Array = Int64Array = Int32Array = BoolArray = (
        AnyArray
    ) = NDArray

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
