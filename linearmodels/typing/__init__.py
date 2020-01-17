from typing import Hashable, Optional, Union

from pandas import DataFrame, Series

from linearmodels.typing.data import ArrayLike, OptionalArrayLike

__all__ = [
    "ArrayLike",
    "OptionalArrayLike",
    "Numeric",
    "OptionalNumeric",
    "AnyPandas",
    "Label",
]

Numeric = Union[int, float]
OptionalNumeric = Optional[Union[int, float]]

AnyPandas = Union[Series, DataFrame]
Label = Optional[Hashable]
