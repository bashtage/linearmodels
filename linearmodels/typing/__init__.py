from typing import Optional, Union

from pandas import DataFrame, Series

Numeric = Union[int, float]
OptionalNumeric = Optional[Union[int, float]]

AnyPandas = Union[Series, DataFrame]
