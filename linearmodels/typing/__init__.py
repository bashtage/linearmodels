import sys
from typing import TYPE_CHECKING, Hashable, Optional, Sequence, Union

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

if sys.version_info >= (3, 8):
    from typing import Literal
elif TYPE_CHECKING:
    from typing_extensions import Literal
else:

    class _Literal:
        def __getitem__(self, item):
            pass

    Literal = _Literal()
