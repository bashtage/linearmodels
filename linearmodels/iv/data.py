"""
A data abstraction that allow multiple input data formats
"""
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from linearmodels.typing import AnyPandas, ArrayLike, NDArray

dim_err = "{0} has too many dims.  Maximum is 2, actual is {1}"
type_err = "Only ndarrays, DataArrays and Series and DataFrames are supported"


def convert_columns(s: pd.Series, drop_first: bool) -> AnyPandas:
    if is_categorical_dtype(s):
        out = pd.get_dummies(s, drop_first=drop_first)
        out.columns = [str(s.name) + "." + str(c) for c in out]
        return out
    return s


def expand_categoricals(x: AnyPandas, drop_first: bool) -> AnyPandas:
    if x.shape[1] == 0:
        return x
    return pd.concat(
        [convert_columns(x[c], drop_first) for c in x.columns], axis=1, sort=False
    )


class IVData(object):
    """
    Type abstraction for use in univariate models.

    Parameters
    ----------
    x : {ndarray, Series, DataFrame, DataArray}, optional
        Data to wrap and standardize.  If None, then nobs must be provided to
        produce an IVData instance with shape (nobs, 0).
    var_name : str, optional
        Variable name to use when naming variables in NumPy arrays or
        xarray DataArrays
    nobs : int, optional
        Number of observation, used when `x` is None. If `x` is array_like,
        then nobs is used to check the number of observations in `x`.
    convert_dummies : bool, optional
        Flat indicating whether pandas categoricals or string input data
        should be converted to dummy variables
    drop_first : bool, optional
        Flag indicating to drop first dummy category
    """

    def __init__(
        self,
        x: Optional["IVDataLike"],
        var_name: str = "x",
        nobs: Optional[int] = None,
        convert_dummies: bool = True,
        drop_first: bool = True,
    ):

        if isinstance(x, IVData):
            self.__dict__.update(copy.deepcopy(x.__dict__))
            return
        if x is None and nobs is not None:
            x = np.empty((nobs, 0))
        elif x is None:
            raise ValueError("nobs required when x is None")

        self.original = x
        assert x is not None
        xndim = x.ndim
        if xndim > 2:
            raise ValueError(dim_err.format(var_name, xndim))

        if isinstance(x, np.ndarray):
            x = x.astype(dtype=np.float64)
            if xndim == 1:
                x = x.reshape((x.shape[0], -1))

            self._ndarray = x.astype(np.float64)
            index = list(range(x.shape[0]))
            if x.shape[1] == 1:
                cols = [var_name]
            else:
                cols = [var_name + ".{0}".format(i) for i in range(x.shape[1])]
            self._pandas = pd.DataFrame(x, index=index, columns=cols)
            self._row_labels = index
            self._col_labels = cols

        elif isinstance(x, (pd.Series, pd.DataFrame)):
            if isinstance(x, pd.Series):
                name = var_name if not x.name else x.name
                x = pd.DataFrame({name: x})
            copied = False
            columns = list(x.columns)
            if len(set(columns)) != len(columns):
                raise ValueError(
                    "DataFrame contains duplicate column names. "
                    "All column names must be distinct"
                )
            all_numeric = True
            for col in x:
                c = x[col]
                if (
                    is_string_dtype(c.dtype)
                    and c.map(lambda v: isinstance(v, str)).all()
                ):
                    c = c.astype("category")
                    if not copied:
                        x = x.copy()
                        copied = True
                    x[col] = c
                dt = c.dtype
                all_numeric = all_numeric and is_numeric_dtype(dt)
                if not (is_numeric_dtype(dt) or is_categorical_dtype(dt)):
                    raise ValueError(
                        "Only numeric, string  or categorical " "data permitted"
                    )

            if convert_dummies:
                x = expand_categoricals(x, drop_first)

            self._pandas = x
            self._ndarray = np.asarray(self._pandas)
            if all_numeric or convert_dummies:
                self._ndarray = self._ndarray.astype(np.float64)
            self._row_labels = list(x.axes[0])
            self._col_labels = list(x.axes[1])

        else:
            try:
                import xarray as xr
            except ImportError:
                raise TypeError(type_err)
            if isinstance(x, xr.DataArray):
                if x.ndim == 1:
                    x = xr.concat([x], dim=var_name)
                    assert isinstance(x, xr.DataArray)
                    x = x.transpose()

                index = list(x.coords[x.dims[0]].values)
                xr_col_values = x.coords[x.dims[1]].values
                xr_cols = list(xr_col_values)
                if is_numeric_dtype(xr_col_values.dtype):
                    xr_cols = [var_name + ".{0}".format(i) for i in range(x.shape[1])]
                self._ndarray = x.values.astype(np.float64)
                self._pandas = pd.DataFrame(self._ndarray, columns=xr_cols, index=index)
                self._row_labels = index
                self._col_labels = xr_cols
            else:
                raise TypeError(type_err)

        if nobs is not None:
            if self._ndarray.shape[0] != nobs:
                msg = "Array required to have {nobs} obs, has " "{act}".format(
                    nobs=nobs, act=self._ndarray.shape[0]
                )
                raise ValueError(msg)

    @property
    def pandas(self) -> pd.DataFrame:
        """DataFrame view of data"""
        return self._pandas

    @property
    def ndarray(self) -> NDArray:
        """ndarray view of data, always 2d"""
        return self._ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        """Tuple containing shape"""
        return self._ndarray.shape[0], self._ndarray.shape[1]

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return self._ndarray.ndim

    @property
    def cols(self) -> List[Any]:
        """Column labels"""
        return self._col_labels

    @property
    def rows(self) -> List[Any]:
        """Row labels (index)"""
        return self._row_labels

    @property
    def labels(self) -> Dict[int, Any]:
        """Dictionary containing row and column labels keyed by axis"""
        return {0: self._row_labels, 1: self._col_labels}

    @property
    def isnull(self) -> pd.Series:
        return self._pandas.isnull().any(axis=1)

    def drop(self, locs: ArrayLike) -> None:
        locs = np.asarray(locs)
        self._pandas = self.pandas.loc[~locs]
        self._ndarray = self._ndarray[~locs]
        self._row_labels = list(pd.Series(self._row_labels).loc[~locs])


IVDataLike = Union[IVData, ArrayLike]
