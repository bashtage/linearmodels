from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from pandas import CategoricalIndex, MultiIndex, Series
from pandas.core.arrays.categorical import CategoricalAccessor

from linearmodels.typing import AnyPandas, NDArray

__all__ = ["is_string_like", "concat", "get_codes", "to_numpy"]


def concat(
    objs: Iterable[AnyPandas], axis: int = 0, sort: Optional[bool] = None
) -> AnyPandas:
    """
    Shim around pandas concat that passes sort if allowed

    See pandas.compat
    """
    if sort is None:
        sort = False
    assert sort is not None

    return pd.concat(objs, axis=axis, sort=sort)


# From pandas 0.20.1
def is_string_like(obj: object) -> bool:
    """
    Check if the object is a string.

    Parameters
    ----------
    obj : The object to check.

    Returns
    -------
    bool
        Whether `obj` is a string or not.
    """
    return isinstance(obj, str)


def get_codes(
    index: Union[MultiIndex, CategoricalIndex, CategoricalAccessor]
) -> Series:
    """
    Tries .codes before falling back to .labels
    """
    try:
        return index.codes
    except AttributeError:
        return index.labels


def to_numpy(df: AnyPandas) -> NDArray:
    try:
        return df.to_numpy()
    except AttributeError:
        return np.asarray(df)
