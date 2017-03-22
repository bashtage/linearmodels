import numpy as np
import pandas as pd
import xarray as xr


class PanelData(object):
    """
    Abstraction to handle alternative formats for panel data
    
    Parameters
    ----------
    x : {ndarray, Series, DataFrame, DataArray}
       Input data, either 2 or 3 dimensaional
    var_name : str, optional
        Name to use when generating labels for the variables in the data
    convert_categoricals : bool, optional
        Flag indicating whether categorical or string variables should be 
        converted to dummies
    
    Notes
    -----
    Data can be either 2- or 3-dimensional. The three key dimensions are 
    
      * nvar - number of variables
      * nobs - number of time periods
      * nentity - number of entities
    
    All 3-d inputs should be in the form (nvar, nobs, nentity). With one
    exception, 2-d inputs are treated as (nobs, nentity) so that the input
    can be treated as being (1, nobs, nentity). 
    
    If the 2-d input is a pandas DataFrame and has a MultiIndex then it is 
    treated differently.  Index level 0 is assumed ot be entity.  Index level
    1 is time.  The columns are the variables.  This is the most precise format
    to use since pandas Panels do not preserve all variable type information
    across transformations between Panel and MultiIndex DataFrame.
    
    Raises
    ------
    TypeError
        If the input type is not supported
    ValueError
        If the input has the wrong number of dimensions or a MultiIndex 
        DataFrame does not have 2 levels
    """

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
                    self._frame = pd.DataFrame({var_name + '.0': x.T.stack(dropna=False)})
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
        self._frame.index.levels[0].name = 'entity'
        self._frame.index.levels[1].name = 'time'

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
        self._frame = self._frame.loc[~locs.ravel()]
        self._frame = self._minimize_multiindex(self._frame)
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

    @property
    def entity_ids(self):
        """
        Get array containing entity group membership information 
        
        Returns
        -------
        id : array 
            2d array containing entity ids corresponding dataframe view
        """
        ids = self._frame.reset_index()['entity']
        ids = pd.Categorical(ids, ordered=True)
        return ids.codes[:, None]

    @property
    def time_ids(self):
        """
        Get array containing time membership information 

        Returns
        -------
        id : array 
            2d array containing time ids corresponding dataframe view
        """
        ids = self._frame.reset_index()['time']
        ids = pd.Categorical(ids, ordered=True)
        return ids.codes[:, None]

    def demean(self, group='entity'):
        """
        Demeans data by either entity or time group
        
        Parameters
        ----------
        group : {'entity', 'time'}
            Group to use in demeaning
        
        Returns
        -------
        demeaned : PanelData
            Demeaned data according to type
        """
        v = self.panel.values
        axis = 2 if group == 'time' else 1
        mu = np.nanmean(v, axis=axis)
        mu = np.expand_dims(mu, axis=axis)
        out = pd.Panel(v - mu, items=self.vars,
                       major_axis=self.time, minor_axis=self.entities)
        out = out.swapaxes(1, 2).to_frame(filter_observations=False)
        if out.shape != self._frame.shape:
            out = out.loc[self._frame.index]
        return PanelData(out)

    def __str__(self):
        return self.__class__.__name__ + '\n' + str(self._frame)

    def __repr__(self):
        return self.__str__() + '\n' + self.__class__.__name__ + ' object, id: ' + hex(id(self))

    def mean(self, group='entity'):
        """
        Compute data mean by either entity or time group

        Parameters
        ----------
        group : {'entity', 'time'}
            Group to use in demeaning

        Returns
        -------
        mean : DataFrame
            Data mean according to type. Either (entity by var) or (time by var)
        """
        v = self.panel.values
        axis = 1 if group == 'time' else 2
        mu = np.nanmean(v, axis=axis)
        index = self.entities if group == 'time' else self.time
        return pd.DataFrame(mu.T, index=index, columns=self.vars)

    def first_difference(self):
        """
        Compute first differences of variables
        
        Returns
        -------
        diffs : PanelData
            Differenced values
        """
        diffs = self._frame.diff(1).iloc[1:]
        elabels = self._frame.index.labels[0]
        same = elabels[1:] == elabels[:-1]
        diffs = diffs.loc[same]
        diffs = self._minimize_multiindex(diffs)
        return PanelData(diffs)

    @staticmethod
    def _minimize_multiindex(df):
        index_cols = list(df.index.names)
        df = df.reset_index()
        df = df.set_index(index_cols)
        return df
