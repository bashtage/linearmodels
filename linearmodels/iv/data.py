"""
A data abstraction that allows multiple input data formats.
"""

from __future__ import annotations

import copy
from typing import Any, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from linearmodels.shared.inputs import as_array_like, as_float64
import linearmodels.typing.data

dim_err = "{0} has too many dims.  Maximum is 2, actual is {1}"


def convert_columns(s: pd.Series, drop_first: bool) -> linearmodels.typing.AnyPandas:
    """
    Convert a categorical Series to dummy variables.

    Parameters
    ----------
    s : Series
        The data to convert.
    drop_first : bool
        Whether to drop the first category of a categorical Series.

    Returns
    -------
    Series or DataFrame
        A DataFrame of dummy variables, with columns named
        ``"<name>.<category>"``, if ``s`` is categorical. Otherwise ``s``.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        out = pd.get_dummies(s, drop_first=drop_first)
        # TODO: Remove once pandas typing fixed
        out.columns = pd.Index([str(s.name) + "." + str(c) for c in out])
        return out
    return s


def expand_categoricals(x: pd.DataFrame, drop_first: bool) -> pd.DataFrame:
    """
    Convert the categorical columns of a DataFrame to dummy variables.

    Parameters
    ----------
    x : DataFrame
        The data to convert.
    drop_first : bool
        Whether to drop the first category of each categorical column.

    Returns
    -------
    DataFrame
        The data with every categorical column replaced by dummy variables.
    """
    if x.shape[1] == 0:
        return x
    return pd.concat(
        [convert_columns(x[c], drop_first) for c in x.columns], axis=1, sort=False
    )


class IVData:
    """
    Type abstraction for use in univariate models.

    Parameters
    ----------
    x : array_like or None
        Data to wrap and standardize. Anything that is not a pandas Series or
        DataFrame, an xarray DataArray or an IVData, such as a list or a tuple,
        is converted using ``np.asarray``. A ``TypeError`` is raised if the data
        are complex or cannot be converted to float64. If None, then nobs must
        be provided to produce an IVData instance with shape (nobs, 0).
    var_name : str, optional
        Variable name to use when naming variables in NumPy arrays or
        xarray DataArrays. The default is ``"x"``.
    nobs : int, optional
        Number of observations, used when `x` is None. If `x` is array_like,
        then nobs is used to check the number of observations in `x`.
    convert_dummies : bool, optional
        Flag indicating whether pandas categoricals or string input data
        should be converted to dummy variables. The default is True.
    drop_first : bool, optional
        Flag indicating to drop the first dummy category. The default is True.

    Raises
    ------
    ValueError
        If `x` is None and `nobs` is not provided, if `x` has more than two
        dimensions, if the number of observations in `x` is not `nobs`, if a
        DataFrame has duplicate column names or has a column that is not
        numeric, string or categorical.
    TypeError
        If `x` is complex or cannot be converted to float64.
    """

    def __init__(
        self,
        x: IVDataLike | None,
        var_name: str = "x",
        nobs: int | None = None,
        convert_dummies: bool = True,
        drop_first: bool = True,
    ):
        if isinstance(x, IVData):
            self.__dict__.update(copy.deepcopy(x.__dict__))
            self._check_nobs(nobs, var_name)
            return
        if x is None and nobs is not None:
            x = np.empty((nobs, 0))
        elif x is None:
            raise ValueError("nobs required when x is None")

        # The input as it was passed, which is what labels are taken from
        self.original = x
        assert x is not None
        # Lists and tuples are converted to arrays, and complex values are rejected
        x = as_array_like(x, var_name)
        xndim = x.ndim
        if xndim > 2:
            raise ValueError(dim_err.format(var_name, xndim))

        if isinstance(x, np.ndarray):
            x = as_float64(x, var_name)
            if xndim == 1:
                x = x.reshape((x.shape[0], -1))

            self._ndarray = x.astype(np.float64)
            index = list(range(x.shape[0]))
            if x.shape[1] == 1:
                cols = [var_name]
            else:
                cols = [var_name + f".{i}" for i in range(x.shape[1])]
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
                if not (is_numeric_dtype(dt) or isinstance(dt, pd.CategoricalDtype)):
                    raise ValueError(
                        f"{var_name} has a column ({col!r}) of type {dt}. "
                        "Only numeric, string  or categorical data permitted"
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
            # as_array_like only returns an xarray DataArray if xarray is installed
            import xarray as xr  # noqa: PLC0415

            assert isinstance(x, xr.DataArray)
            if x.ndim == 1:
                x = xr.concat([x], dim=var_name)
                assert isinstance(x, xr.DataArray)
                x = x.transpose()
            assert isinstance(x, xr.DataArray)
            index = list(x.coords[x.dims[0]].values)
            xr_col_values = x.coords[x.dims[1]].values
            xr_cols = list(xr_col_values)
            if is_numeric_dtype(xr_col_values.dtype):
                xr_cols = [var_name + f".{i}" for i in range(x.shape[1])]
            self._ndarray = as_float64(x.values, var_name)
            self._pandas = pd.DataFrame(self._ndarray, columns=xr_cols, index=index)
            self._row_labels = index
            self._col_labels = xr_cols

        self._check_nobs(nobs, var_name)

    def _check_nobs(self, nobs: int | None, var_name: str) -> None:
        if nobs is not None and self._ndarray.shape[0] != nobs:
            raise ValueError(
                f"{var_name} has the wrong number of observations. Array required "
                f"to have {nobs} obs, has {self._ndarray.shape[0]}"
            )

    @property
    def pandas(self) -> pd.DataFrame:
        """Return a DataFrame view of the data."""
        return self._pandas

    @property
    def ndarray(self) -> linearmodels.typing.data.Float64Array:
        """Return an ndarray view of the data, which is always 2d."""
        return self._ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the data as a tuple."""
        return self._ndarray.shape[0], self._ndarray.shape[1]

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self._ndarray.ndim

    @property
    def cols(self) -> list[Any]:
        """Return the column labels."""
        return self._col_labels

    @property
    def rows(self) -> list[Any]:
        """Return the row labels (index)."""
        return self._row_labels

    @property
    def labels(self) -> dict[int, Any]:
        """Return a dictionary containing row and column labels keyed by axis."""
        return {0: self._row_labels, 1: self._col_labels}

    @property
    def isnull(self) -> pd.Series:
        """Return a Series that is True for the rows that contain a missing value."""
        return self._pandas.isnull().any(axis=1)

    def drop(self, locs: linearmodels.typing.data.ArrayLike) -> None:
        """
        Drop rows in place.

        Parameters
        ----------
        locs : array_like
            Boolean array that is True for the rows to drop. It must have one
            element for each row.
        """
        locs = np.asarray(locs)
        self._pandas = self.pandas.loc[~locs]
        self._ndarray = self._ndarray[~locs]
        self._row_labels = list(pd.Series(self._row_labels).loc[~locs])


IVDataLike = Union[IVData, linearmodels.typing.data.ArrayLike]
