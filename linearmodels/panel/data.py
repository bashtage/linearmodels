import numpy as np
import pandas as pd
import xarray as xr


class PanelData(object):
    # Panel -> entity, time, vars
    # df 2x -> (entity,time), vars
    # a2s -> df.values
    # a3d -> panel.values

    # 3d -> variables, time, entities
    # 2d -> time, entities (single variable)
    # 2d, multiindex -> (entities, time), variables
    def __init__(self, x, var_name='x'):
        if isinstance(x, PanelData):
            x = x._original
        self._original = x

        if isinstance(x, xr.DataArray):
            if x.ndim not in (2, 3):
                raise ValueError('Only 2-d or 3-d DataArrays are supported')
            x = x.to_pandas()

        if isinstance(x, (pd.Panel, pd.DataFrame)):

            if isinstance(x, pd.DataFrame):
                if isinstance(x.index, pd.MultiIndex):
                    if len(x.index.levels) != 2:
                        raise ValueError('DataFrame input must have a '
                                         'MultiIndex with 2 levels')
                    self._frame = x
                else:
                    self._frame = pd.DataFrame({var_name + '.0': x.T.stack()})
            else:
                self._frame = x.swapaxes(1, 2).to_frame(filter_observations=False)
        elif isinstance(x, np.ndarray):
            if not 2 <= x.ndim <= 3:
                raise ValueError('2 or 3-d array required for numpy input')
            if x.ndim == 2:
                x = x[None, :, :]

            k, t, n = x.shape
            variables = [var_name + '.{0}'.format(i) for i in range(k)]
            entities = ['entity.{0}'.format(i) for i in range(n)]
            time = list(range(t))
            x = x.astype(np.float64)
            panel = pd.Panel(x, items=variables, major_axis=time,
                             minor_axis=entities)
            self._frame = panel.swapaxes(1, 2).to_frame(filter_observations=False)
        else:
            raise TypeError('Only ndarrays, DataFrames, Panels or DataArrays '
                            'supported.')
        self._k, self._t, self._n = self.panel.shape

    @property
    def panel(self):
        return self._frame.to_panel().swapaxes(1, 2)

    @property
    def dataframe(self):
        return self._frame

    @property
    def a2d(self):
        return self._frame.values

    @property
    def a3d(self):
        return self.panel.values

    def drop(self, locs):
        locs = locs.ravel()
        self._frame = self._frame.loc[~locs]
        self._n, self._t, self._k = self.shape

    @property
    def shape(self):
        return self.panel.shape

    @property
    def isnull(self):
        return np.any(self._frame.isnull(), axis=1)

    @property
    def nobs(self):
        return self._t

    @property
    def nvar(self):
        return self._k

    @property
    def nentity(self):
        return self._n

    @property
    def vars(self):
        return list(self.panel.items)

    @property
    def time(self):
        return list(self.panel.major_axis)

    @property
    def entities(self):
        return list(self.panel.minor_axis)
