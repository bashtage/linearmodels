from collections.abc import MutableMapping
from typing import (
    AbstractSet,
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    ValuesView,
)

import numpy as np
from pandas import DataFrame, MultiIndex, Series

from linearmodels.typing import ArrayLike, Label, NDArray


class AttrDict(MutableMapping):
    """
    Ordered dictionary-like object that exposes keys as attributes
    """

    def update(
        self, *args: Union[Mapping[Any, Any], Iterable[Tuple[Any, Any]]], **kwargs: Any
    ) -> None:
        """
        Update AD from dictionary or iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: AD[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: AD[k] = v
        In either case, this is followed by: for k in F:  AD[k] = F[k]
        """
        self.__private_dict__.update(*args, **kwargs)

    def clear(self) -> None:
        """Remove all items from AD. """
        self.__private_dict__.clear()

    def copy(self) -> "AttrDict":
        """Create a shallow copy of AD """
        ad = AttrDict()
        for key in self.__private_dict__.keys():
            ad[key] = self.__private_dict__[key]
        return ad

    def keys(self) -> AbstractSet[Any]:
        """Return an ordered list-like object providing a view on AD's keys """
        return self.__private_dict__.keys()

    def items(self) -> AbstractSet[Tuple[Any, Any]]:
        """Return an ordered list-like object providing a view on AD's items """
        return self.__private_dict__.items()

    def values(self) -> ValuesView[Any]:
        """Return an ordered list-like object object providing a view on AD's values """
        return self.__private_dict__.values()

    def pop(self, key: Label, default: Any = None) -> Any:
        """
        Remove specified key and return the corresponding value.
        If key is not found, default is returned if given, otherwise KeyError is raised
        """

        return self.__private_dict__.pop(key, default)

    def __len__(self) -> int:
        return self.__private_dict__.__len__()

    def __repr__(self) -> str:
        out = self.__private_dict__.__str__()
        return "Attr" + out[7:]

    def __str__(self) -> str:
        return self.__repr__()

    def __init__(
        self, *args: Union[Mapping[Any, Any], Sequence[Tuple[Any, Any]]], **kwargs: Any
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
        out = list(map(str, self.__private_dict__.keys()))
        out += list(super(AttrDict, self).__dir__())
        filtered = [key for key in out if key.isidentifier()]
        return sorted(set(filtered))

    def __iter__(self) -> Iterator[Label]:
        return self.__private_dict__.__iter__()


def ensure_unique_column(col_name: str, df: DataFrame, addition: str = "_") -> str:
    while col_name in df:
        col_name = addition + col_name + addition
    return col_name


def panel_to_frame(
    x: NDArray,
    items: Sequence[Label],
    major_axis: Sequence[Label],
    minor_axis: Sequence[Label],
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
    df = DataFrame(x, columns=items, index=mi)
    if swap:
        df.index = mi.swaplevel()
        df.sort_index(inplace=True)
        final_levels = [minor_axis, major_axis]
    df.index.set_levels(final_levels, [0, 1], inplace=True)
    df.index.names = ["major", "minor"]
    return df


def get_string(d: Mapping[str, Any], key: str) -> Optional[str]:
    """
    Helper function that gets a string or None

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    {str, None}
        The string or None if the key is not in the dictionary. If in the
        dictionary, a type check is performed and TypeError is raised if
        not found.
    """
    out: Optional[str] = None
    if key in d:
        out = d[key]
        if out is not None:
            if isinstance(out, str):
                return out
            else:

                raise TypeError(f"{key} found in the dictionary but it is not a str.")
    return out


def get_float(d: Mapping[str, Any], key: str) -> Optional[float]:
    """
    Helper function that gets a float or None

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    {float, None}
        The string or None if the key is not in the dictionary. If in the
        dictionary, a type check is performed and TypeError is raised if
        not found.
    """
    out: Optional[float] = None
    if key in d:
        out = d[key]
        if out is not None:
            if isinstance(out, (int, float, np.floating)):
                return float(out)
            else:
                raise TypeError(f"{key} found in the dictionary but it is not a float.")
    return out


def get_bool(d: Mapping[str, Any], key: str) -> bool:
    """
    Helper function that gets a bool, defaulting to False.

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    bool
        The boolean if the key is in the dictionary. If not found, returns
        False.
    """
    out: Optional[bool] = False
    if key in d:
        out = d[key]
        if not (out is None or isinstance(out, bool)):
            raise TypeError(f"{key} found in the dictionary but it is not a bool.")
    return bool(out)


def get_array_like(d: Mapping[str, Any], key: str) -> Optional[ArrayLike]:
    """
    Helper function that gets a bool or None

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    {bool, None}
        The string or None if the key is not in the dictionary. If in the
        dictionary, a type check is performed and TypeError is raised if
        not found.
    """

    out: Optional[bool] = None
    if key in d:
        out = d[key]
        if out is not None:
            array_like: Union[Any] = (np.ndarray, DataFrame, Series)
            try:
                import xarray as xr

                array_like += (xr.DataArray,)
            except ImportError:
                pass

            if isinstance(out, array_like):
                return out
            else:
                raise TypeError(
                    f"{key} found in the dictionary but it is not array-like."
                )
    return out
