import numpy as np
import pandas as pd
import xarray as xr


class PanelData(object):
    """
    Uniform panel data access
    
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
