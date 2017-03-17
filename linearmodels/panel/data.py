import numpy as np
import pandas as pd
import xarray as xr


class PanelData(object):
    """
    Uniform linearmodels data access

    Parameters
    ----------
    data : {np.ndarray, pd.DataFrame, pd.Panel, xr.DataArray}
        Input data
    """
    n, t, k = 0, 0, 0
    _2d = None
    time_index = None
    columns = None
    entity_index = None

    def __init__(self, data):
        self.data = data
        if isinstance(data, PanelData):
            data = self.data = data.data

        self._validate_input()
        n, t, k = self.n, self.t, self.k = data.shape

        if isinstance(data, np.ndarray):
            self._2d = data.reshape((n * t, k))
            self.time_index = list(np.arange(t))
            self.columns = self.time_index = list(np.arange(k))
            self.entity_index = self.time_index = list(np.arange(n))
        elif isinstance(data, pd.Panel):
            self._2d = data.swapaxes(0, 2).to_frame(False)
            self.time_index = list(data.major_axis)
            self.columns = list(self._2d.columns)
            self.entity_index = list(data.items)
        elif isinstance(data, xr.DataArray):
            d0, d1 = data.dims[:2]
            self._2d = data.stack(z=(d0, d1)).T
            self.entity_index = data.coords[data.dims[0]]
            self.time_index = data.coords[data.dims[1]]
            self.columns = data.coords[data.dims[2]]

    @property
    def id(self):
        return id(self.data)

    @property
    def is_pandas(self):
        return isinstance(self.data, pd.Panel)

    @property
    def is_numpy(self):
        return isinstance(self.data, np.ndarray)

    @property
    def is_xarray(self):
        return isinstance(self.data, xr.DataArray)

    @property
    def as3d(self):
        return self.data

    @property
    def as2d(self):
        return self._2d

    @property
    def asnumpy3d(self):
        if self.is_numpy:
            return self.data
        else:
            return self.data.values

    @property
    def asnumpy2d(self):
        if self.is_numpy:
            return self.as2d
        else:
            return self.as2d.values

    def _validate_input(self):
        """Ensures the input is one of the supported types"""
        if self.is_numpy:
            if self.data.ndim > 3:
                raise ValueError('NumPy array must have 3 or fewer dimensions')
            elif self.data.ndim < 3:
                if self.data.ndim == 1:
                    self.data = self.data[:, None]
                else:
                    self.data = self.data[:, None, None]
        elif self.is_xarray:

            if len(self.data.dims) > 3:
                raise ValueError('xarray DataArray must have 3 or fewer dimensions')
            if len(self.data.dims) == 1:
                self.data = xr.concat([self.data], dim='fake_dim1')
                self.data = self.data.transpose()
            if len(self.data.dims) == 2:
                self.data = xr.concat([self.data], dim='fake_dim')
                self.data = self.data.transpose((*self.data.dims[1:], self.data.dims[0]))
        elif self.is_pandas:
            if self.data.ndim == 1:
                self.data = pd.DataFrame({0: self.data})
            if self.data.ndim == 2:
                self.data = pd.Panel({0: self.data}).swapaxes(0, 1).swapaxes(1, 2)
            if self.data.ndim > 3:
                raise ValueError('data must have 3 or fewer dimensions')
        else:
            raise ValueError('Unknown type of data -- must have 3 dimensions '
                             'and subclasses are not supported')

    def column_index(self, index):
        """Return numerical columns index"""
        if not isinstance(index, list):
            index = [index]
        return [i for i, c in enumerate(self.columns) if c in index]


class PanelDataHandler(object):
    # Panel -> entity, time, vars
    # df 2x -> (entity,time), vars
    # a2s -> df.values
    # a3d -> panel.values
    def __init__(self, x, var_name='x'):
        if isinstance(x, np.ndarray):
            if x.ndim > 3:
                raise ValueError('1, 2 or 3-d array required for numpy input')
            if x.ndim == 1:
                x = x[:, None]
            if x.ndim == 2:
                x = x[:, :, None]

            n, t, k = x.shape
            x = x.astype(np.float64)
            minor = [var_name + '.{0}'.format(i) for i in range(k)]
            panel = pd.Panel(x, items=['entity.{0}'.format(i) for i in range(n)],
                             major_axis=list(range(t)),
                             minor_axis=minor)
            panel = panel.swapaxes(0, 1).swapaxes(0, 2)
            self._dataframe = panel.to_frame(filter_observations=False)
        elif isinstance(x, (pd.DataFrame, pd.Panel, xr.DataArray)):
            if isinstance(x, xr.DataArray):
                if x.ndim not in (2, 3):
                    raise ValueError('Only 2-d or 3-d DataArrays are supported')
                x = x.to_pandas()
            if isinstance(x, pd.DataFrame):
                if isinstance(x.index, pd.MultiIndex):
                    if len(x.index.levels) != 2:
                        raise ValueError('DataFrame input must have a '
                                         'MultiIndex with 2 levels')
                    x = x.to_panel()
                else:
                    x = pd.Panel({var_name + '.0': x})
                    x = x.swapaxes(0, 1).swapaxes(1, 2)
            panel = x.swapaxes(0, 1).swapaxes(0, 2)
            self._dataframe = panel.to_frame(filter_observations=False)
        else:
            raise TypeError('Only ndarrays, DataFrames, Panels or DataArrays '
                            'supported.')
        self._n, self._t, self._k = self.panel.shape

    @property
    def panel(self):
        return self._dataframe.to_panel().swapaxes(0, 2).swapaxes(0, 1)

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def a2d(self):
        return self._dataframe.values

    @property
    def a3d(self):
        return self.panel.values

    def drop(self, locs):
        locs = locs.ravel()
        self._dataframe = self._dataframe.loc[~locs]
        self._n, self._t, self._k = self.shape

    @property
    def shape(self):
        return self.panel.shape

    @property
    def ndim(self):
        return 3

    @property
    def isnull(self):
        return np.any(self._dataframe.isnull(), axis=1)

    @property
    def nobs(self):
        return self._t

    @property
    def nvar(self):
        return self._k

    @property
    def nitems(self):
        return self._n

    @property
    def items(self):
        return list(self.panel.items)

    @property
    def index(self):
        return list(self.panel.major_axis)

    @property
    def vars(self):
        return list(self.panel.minor_axis)
