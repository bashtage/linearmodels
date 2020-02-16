from typing import Hashable, Optional, Sequence, Union

from numpy import ndarray
from pandas import DataFrame, Series

from .data import ArrayLike, OptionalArrayLike

__all__ = [
    "ArrayLike",
    "OptionalArrayLike",
    "Numeric",
    "OptionalNumeric",
    "AnyPandas",
    "Label",
    "NDArray",
    "ArraySequence",
]

# Workaround for https://github.com/python/mypy/issues/7866
NDArray = Union[ndarray]
ArraySequence = Sequence[ndarray]

Numeric = Union[int, float]
OptionalNumeric = Optional[Union[int, float]]

AnyPandas = Union[Series, DataFrame]
Label = Optional[Hashable]
