"""
A data abstraction that allow multiple input data formats
"""
import copy

import numpy as np
import pandas as pd

from linearmodels.compat.pandas import (is_categorical, is_categorical_dtype,
                                        is_numeric_dtype, is_string_dtype,
                                        is_string_like)

dim_err = '{0} has too many dims.  Maximum is 2, actual is {1}'
type_err = 'Only ndarrays, DataArrays and Series and DataFrames are permitted'


def convert_columns(s, drop_first):
    if is_categorical(s):
        out = pd.get_dummies(s, drop_first=drop_first)
        out.columns = [str(s.name) + '.' + str(c) for c in out]
        return out
    return s


def expand_categoricals(x, drop_first):
    if x.shape[1] == 0:
        return x
    return pd.concat([convert_columns(x[c], drop_first) for c in x.columns], axis=1)


class IVData(object):
    """Simple class to abstract different input data formats

    Parameters
    ----------
    x : {ndarray, Series, DataFrame, DataArray}
    var_name : str, optional
        Variable name to use when naming variables in NumPy arrays or
        xarray DataArrays
    convert_dummies : bool, optional
        Flat indicating whether pandas categoricals or string input data
        should be converted to dummy variables
    drop_first : bool, optional
        Flag indicating to drop first dummy category
    """

    def __init__(self, x, var_name='x', nobs=None, convert_dummies=True, drop_first=True):

        if isinstance(x, IVData):
            self.__dict__.update(copy.deepcopy(x.__dict__))
            return
        if x is None and nobs is not None:
            x = np.empty((nobs, 0))
        elif x is None:
            raise ValueError('nobs required when x is None')

        self.original = x
        xndim = x.ndim
        if xndim > 2:
            raise ValueError(dim_err.format(var_name, xndim))

        if isinstance(x, np.ndarray):
            x = x.astype(np.float64)
            if xndim == 1:
                x.shape = (x.shape[0], -1)

            self._ndarray = x.astype(np.float64)
            index = list(range(x.shape[0]))
            if x.shape[1] == 1:
                cols = [var_name]
            else:
                cols = [var_name + '.{0}'.format(i) for i in range(x.shape[1])]
            self._pandas = pd.DataFrame(x, index=index, columns=cols)
            self._labels = {0: index, 1: cols}

        elif isinstance(x, (pd.Series, pd.DataFrame)):
            if isinstance(x, pd.Series):
                name = var_name if not x.name else x.name
                x = pd.DataFrame({name: x})
            copied = False
            for col in x:
                c = x[col]
                if is_string_dtype(c.dtype) and \
                        c.map(lambda v: is_string_like(v)).all():

                    c = c.astype('category')
                    if not copied:
                        x = x.copy()
                        copied = True
                    x[col] = c
                dt = c.dtype
                if not (is_numeric_dtype(dt) or is_categorical_dtype(dt)):
                    raise ValueError('Only numeric, string  or categorical '
                                     'data permitted')

            if convert_dummies:
                x = expand_categoricals(x, drop_first)

            self._pandas = x
            self._ndarray = self._pandas.values.astype(np.float64)
            self._labels = {i: list(label) for i, label in zip(range(x.ndim), x.axes)}

        else:
            import xarray as xr
            if isinstance(x, xr.DataArray):
                if x.ndim == 1:
                    x = xr.concat([x], dim=var_name).transpose()

                index = list(x.coords[x.dims[0]].values)
                cols = x.coords[x.dims[1]].values
                if is_numeric_dtype(cols.dtype):
                    cols = [var_name + '.{0}'.format(i) for i in range(x.shape[1])]
                cols = list(cols)
                self._ndarray = x.values.astype(np.float64)
                self._pandas = pd.DataFrame(self._ndarray, columns=cols,
                                            index=index)
                self._labels = {0: index, 1: cols}
            else:
                raise TypeError(type_err)

        if nobs is not None:
            if self._ndarray.shape[0] != nobs:
                msg = 'Array required to have {nobs} obs, has ' \
                      '{act}'.format(nobs=nobs, act=self._ndarray.shape[0])
                raise ValueError(msg)

    @property
    def pandas(self):
        """DataFrame view of data"""
        return self._pandas

    @property
    def ndarray(self):
        """ndarray view of data, always 2d"""
        return self._ndarray

    @property
    def shape(self):
        """Tuple containing shape"""
        return self._ndarray.shape

    @property
    def ndim(self):
        """Number of dimensions"""
        return self._ndarray.ndim

    @property
    def cols(self):
        """Column labels"""
        return self._labels[1]

    @property
    def rows(self):
        """Row labels (index)"""
        return self._labels[0]

    @property
    def labels(self):
        """Dictionary containing row and column labels keyed by axis"""
        return self._labels

    @property
    def isnull(self):
        return np.any(self._pandas.isnull(), axis=1)

    def drop(self, locs):
        self._pandas = self.pandas.loc[~locs]
        self._ndarray = self._ndarray[~locs]
        self._labels[0] = list(pd.Series(self._labels[0]).loc[~locs])
