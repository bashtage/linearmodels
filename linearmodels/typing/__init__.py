from collections.abc import Hashable, Sequence
from typing import Any, Optional, Union

import numpy as np
from pandas import DataFrame, Series

from .data import (
    AnyArray as AnyArray,
    ArrayLike as ArrayLike,
    BoolArray as BoolArray,
    Float64Array as Float64Array,
    FloatArray2D as FloatArray2D,
    Int32Array as Int32Array,
    Int64Array as Int64Array,
    IntArray as IntArray,
)

__all__ = [
    "AnyArray",
    "AnyPandas",
    "ArrayLike",
    "ArraySequence",
    "BoolArray",
    "Float64Array",
    "FloatArray2D",
    "Int32Array",
    "Int64Array",
    "IntArray",
    "Label",
    "Numeric",
    "NumericArray",
    "OptionalNumeric",
]

ArraySequence = Sequence[np.ndarray]

Numeric = Union[int, float]
OptionalNumeric = Optional[Union[int, float]]

AnyPandas = Union[Series, DataFrame]
Label = Optional[Hashable]

NumericArray = Union[  # pragma: no cover
    np.ndarray[tuple[int, ...], np.dtype[np.signedinteger[Any]]],  # pragma: no cover
    np.ndarray[tuple[int, ...], np.dtype[np.unsignedinteger[Any]]],  # pragma: no cover
    np.ndarray[tuple[int, ...], np.dtype[np.floating[Any]]],  # pragma: no cover
]  # pragma: no cover
