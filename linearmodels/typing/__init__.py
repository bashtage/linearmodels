from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from pandas import DataFrame, Series

from .data import ArrayLike as ArrayLike

__all__ = [
    "ArrayLike",
    "Numeric",
    "OptionalNumeric",
    "AnyPandas",
    "Label",
    "ArraySequence",
    "IntArray",
    "Int32Array",
    "Int64Array",
    "AnyArray",
    "Float64Array",
    "BoolArray",
    "NumericArray",
]

ArraySequence = Sequence[np.ndarray]

Numeric = Union[int, float]
OptionalNumeric = Optional[Union[int, float]]

AnyPandas = Union[Series, DataFrame]
Label = Optional[Hashable]

if TYPE_CHECKING:
    Float64Array = np.ndarray[Any, np.dtype[np.float64]]  # pragma: no cover
    Int64Array = np.ndarray[Any, np.dtype[np.int64]]  # pragma: no cover
    Int32Array = np.ndarray[Any, np.dtype[np.int32]]  # pragma: no cover
    IntArray = np.ndarray[Any, np.dtype[np.int_]]  # pragma: no cover
    BoolArray = np.ndarray[Any, np.dtype[np.bool_]]  # pragma: no cover
    AnyArray = np.ndarray[Any, Any]  # pragma: no cover
    NumericArray = Union[  # pragma: no cover
        np.ndarray[Any, np.dtype[np.signedinteger[Any]]],  # pragma: no cover
        np.ndarray[Any, np.dtype[np.unsignedinteger[Any]]],  # pragma: no cover
        np.ndarray[Any, np.dtype[np.floating[Any]]],  # pragma: no cover
    ]  # pragma: no cover
else:
    IntArray = Float64Array = Int64Array = Int32Array = BoolArray = AnyArray = (
        NumericArray
    ) = np.ndarray
