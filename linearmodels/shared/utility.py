from __future__ import annotations

from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    Sequence,
    ValuesView,
)
from typing import Any, Callable, Protocol, TypeVar, cast

import numpy as np
import pandas
from pandas import DataFrame, Index, MultiIndex, Series

from linearmodels.typing import AnyArray, Label

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_VT_co = TypeVar("_VT_co", covariant=True)


class SupportsKeysAndGetItem(Protocol[_KT, _VT_co]):
    def keys(self) -> Iterable[_KT]: ...  # noqa: E704

    def __getitem__(self, __k: _KT) -> _VT_co: ...  # noqa: E704


def _new_attr_dict_(*args: Iterable[tuple[Any, Any]]) -> AttrDict:
    attr_dict = AttrDict()
    for k, v in args:
        attr_dict[k] = v
    return attr_dict


class AttrDict(MutableMapping):
    """
    Ordered dictionary-like object that exposes keys as attributes
    """

    def update(
        self,
        *args: SupportsKeysAndGetItem[Any, Any] | Iterable[tuple[Any, Any]],
        **kwargs: Any,
    ) -> None:
        """
        Update AD from dictionary or iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: AD[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: AD[k] = v
        In either case, this is followed by: for k in F:  AD[k] = F[k]
        """
        self.__private_dict__.update(*args, **kwargs)

    def clear(self) -> None:
        """Remove all items from AD."""
        self.__private_dict__.clear()

    def copy(self) -> AttrDict:
        """Create a shallow copy of AD"""
        ad = AttrDict()
        for key in self.__private_dict__.keys():
            ad[key] = self.__private_dict__[key]
        return ad

    def keys(self) -> KeysView[Any]:
        """Return an ordered list-like object providing a view on AD's keys"""
        return self.__private_dict__.keys()

    def items(self) -> ItemsView[Any, Any]:
        """Return an ordered list-like object providing a view on AD's items"""
        return self.__private_dict__.items()

    def values(self) -> ValuesView[Any]:
        """Return an ordered list-like object object providing a view on AD's values"""
        return self.__private_dict__.values()

    def pop(self, key: Label, default: Any = None) -> Any:
        """
        Remove specified key and return the corresponding value.
        If key is not found, default is returned if given, otherwise KeyError is raised
        """

        return self.__private_dict__.pop(key, default)

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[Iterable[tuple[Any, Any]]], AttrDict], tuple[tuple[Any, Any], ...]
    ]:
        return _new_attr_dict_, tuple((k, v) for k, v in self.items())

    def __len__(self) -> int:
        return self.__private_dict__.__len__()

    def __repr__(self) -> str:
        out = self.__private_dict__.__str__()
        return "AttrDict" + out

    def __str__(self) -> str:
        return self.__repr__()

    def __init__(
        self, *args: Mapping[Any, Any] | Sequence[tuple[Any, Any]], **kwargs: Any
    ) -> None:
        self.__dict__["__private_dict__"] = dict(*args, **kwargs)

    def __contains__(self, item: Label) -> bool:
        return self.__private_dict__.__contains__(item)

    def __getitem__(self, item: Label) -> Any:
        return self.__private_dict__[item]

    def __setitem__(self, key: Label, value: Any) -> None:
        if key == "__private_dict__":
            raise KeyError("__private_dict__ is reserved and cannot be set.")
        self.__private_dict__[key] = value

    def __delitem__(self, key: Label) -> None:
        del self.__private_dict__[key]

    def __getattr__(self, key: Label) -> Any:
        if key not in self.__private_dict__:
            raise AttributeError
        return self.__private_dict__[key]

    def __setattr__(self, key: Label, value: Any) -> None:
        if key == "__private_dict__":
            raise AttributeError("__private_dict__ is invalid")
        self.__private_dict__[key] = value

    def __delattr__(self, key: Label) -> None:
        del self.__private_dict__[key]

    def __dir__(self) -> Iterable[str]:
        out = [str(key) for key in self.__private_dict__.keys()]
        out += list(super().__dir__())
        filtered = [key for key in out if key.isidentifier()]
        return sorted(set(filtered))

    def __iter__(self) -> Iterator[Label]:
        return self.__private_dict__.__iter__()


def ensure_unique_column(
    col_name: str, df: pandas.DataFrame, addition: str = "_"
) -> str:
    while col_name in df:
        col_name = addition + col_name + addition
    return col_name


def panel_to_frame(
    x: AnyArray | None,
    items: Sequence[Label] | Index,
    major_axis: Sequence[Label] | Index,
    minor_axis: Sequence[Label] | Index,
    swap: bool = False,
) -> DataFrame:
    """
    Construct a multiindex DataFrame using Panel-like arguments

    Parameters
    ----------
    x : ndarray
        3-d array with size nite, nmajor, nminor
    items : list-like
        List like object with item labels
    major_axis : list-like
        List like object with major_axis labels
    minor_axis : list-like
        List like object with minor_axis labels
    swap : bool
        Swap is major and minor axes

    Notes
    -----
    This function is equivalent to

    Panel(x, items, major_axis, minor_axis).to_frame()

    if `swap` is True, it is equivalent to

    Panel(x, items, major_axis, minor_axis).swapaxes(1,2).to_frame()
    """
    nmajor = np.arange(len(major_axis))
    nminor = np.arange(len(minor_axis))
    final_levels = [major_axis, minor_axis]
    mi = MultiIndex.from_product([nmajor, nminor])
    if x is not None:
        shape = x.shape
        x = x.reshape((shape[0], shape[1] * shape[2])).T
    df = DataFrame(x, columns=Index(items), index=mi)
    if swap:
        df.index = mi.swaplevel()
        df.sort_index(inplace=True)
        final_levels = [minor_axis, major_axis]
    mi_index = cast(MultiIndex, df.index)
    df.index = mi_index.set_levels(levels=final_levels, level=[0, 1])
    df.index.names = ["major", "minor"]
    return df


class DataFrameWrapper:
    """
    Wrapper around a pandas DataFrame deferring frame's construction.

    Parameters
    ----------
    values : ndarray
        The data to use in the DataFrame's constructor
    columns : list[str]
        The column names of the frame
    index : {Index, list[str]}
        The index to use.
    """

    def __init__(
        self,
        values: AnyArray,
        *,
        columns: list[str] | None = None,
        index: Index | list[str] | None = None,
    ) -> None:
        self._values = values
        self._columns = columns
        if isinstance(index, list):
            index = Index(index)
        self._index: Index | None = index

    def __call__(self) -> DataFrame:
        return DataFrame(self._values, columns=self._columns, index=self._index)


class SeriesWrapper:
    """
    Wrapper around a pandas Series deferring series' construction.

    Parameters
    ----------
    values : ndarray
        The data to use in the Series' constructor
    name : str
        The name of the series
    index : {Index, list[str]}
        The index to use.
    """

    def __init__(
        self,
        values: AnyArray,
        *,
        name: str | None = None,
        index: Index | list[str] | None = None,
    ) -> None:
        self._values = values
        self._name = name
        if isinstance(index, list):
            index = Index(index)
        self._index: Index | None = index

    def __call__(self) -> Series:
        return Series(self._values, name=self._name, index=self._index)
