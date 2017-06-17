from itertools import product

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Panel, Series

from linearmodels.compat.pandas import (is_categorical,
                                        is_datetime64_any_dtype,
                                        is_numeric_dtype, is_string_dtype,
                                        is_string_like)
from linearmodels.utility import ensure_unique_column

__all__ = ['PanelData']


class _Panel(object):
    """
    Convert a MI DataFrame to a 3-d structure where columns are items

    Parameters
    ----------
    df : DataFrame
        Multiindex DataFrame containing floats

    Notes
    -----
    Contains the logic needed to transform a MI DataFrame with 2 levels
    into a minimal pandas Panel-like object
    """

    def __init__(self, df):
        self._items = df.columns
        index = df.index
        self._major_axis = pd.Series(index.levels[1][index.labels[1]]).unique()
        self._minor_axis = pd.Series(index.levels[0][index.labels[0]]).unique()
        full_index = list(product(self._minor_axis, self._major_axis))
        self._full_index = pd.MultiIndex.from_tuples(full_index)
        new_df = df.copy().loc[self._full_index]
        self._frame = new_df
        i, j, k = len(self._items), len(self._major_axis), len(self.minor_axis)
        self._shape = (i, j, k)
        self._values = np.swapaxes(np.reshape(new_df.values.copy().T, (i, k, j)), 1, 2)

    @classmethod
    def from_array(cls, values, items, major_axis, minor_axis):
        index = list(product(minor_axis, major_axis))
        index = pd.MultiIndex.from_tuples(index)
        i, j, k = len(items), len(major_axis), len(minor_axis)
        values = np.swapaxes(values.copy(), 0, 2).ravel()
        values = np.reshape(values, ((j * k), i))

        df = pd.DataFrame(values, index=index, columns=items)
        return cls(df)

    @property
    def shape(self):
        return self._shape

    @property
    def items(self):
        return self._items

    @property
    def major_axis(self):
        return self._major_axis

    @property
    def minor_axis(self):
        return self._minor_axis

    @property
    def values(self):
        return self._values

    def to_frame(self):
        return self._frame


def convert_columns(s, drop_first):
    if is_string_dtype(s.dtype) and s.map(lambda v: is_string_like(v)).all():
        s = s.astype('category')

    if is_categorical(s):
        out = pd.get_dummies(s, drop_first=drop_first)
        out.columns = [str(s.name) + '.' + str(c) for c in out]
        return out
    return s


def expand_categoricals(x, drop_first):
    return pd.concat([convert_columns(x[c], drop_first) for c in x.columns], axis=1)


class PanelData(object):
    """
    Abstraction to handle alternative formats for panel data

    Parameters
    ----------
    x : {ndarray, Series, DataFrame, Panel, DataArray}
       Input data
    var_name : str, optional
        Variable name to use when naming variables in NumPy arrays or
        xarray DataArrays
    convert_dummies : bool, optional
        Flat indicating whether pandas categoricals or string input data
        should be converted to dummy variables
    drop_first : bool, optional
        Flag indicating to drop first dummy category when converting

    Notes
    -----
    Data can be either 2- or 3-dimensional. The three key dimensions are

    * nvar - number of variables
    * nobs - number of time periods
    * nentity - number of entities

    All 3-d inputs should be in the form (nvar, nobs, nentity). With one
    exception, 2-d inputs are treated as (nobs, nentity) so that the input
    can be treated as-if being (1, nobs, nentity).

    If the 2-d input is a pandas DataFrame with a 2-level MultiIndex then the
    input is treated differently.  Index level 0 is assumed ot be entity.
    Index level 1 is time.  The columns are the variables.  This is the most
    precise format to use since pandas Panels do not preserve all variable
    type information across transformations between Panel and MultiIndex
    DataFrame. MultiIndex Series are also accepted and treated as single
    column MultiIndex DataFrames.

    Raises
    ------
    TypeError
        If the input type is not supported
    ValueError
        If the input has the wrong number of dimensions or a MultiIndex
        DataFrame does not have 2 levels
    """

    def __init__(self, x, var_name='x', convert_dummies=True, drop_first=True):
        self._var_name = var_name
        self._convert_dummies = convert_dummies
        self._drop_first = drop_first
        if isinstance(x, PanelData):
            x = x.dataframe
        self._original = x

        if not isinstance(x, (Series, DataFrame, Panel, ndarray)):
            from xarray import DataArray
            if isinstance(x, DataArray):
                if x.ndim not in (2, 3):
                    raise ValueError('Only 2-d or 3-d DataArrays are supported')
                x = x.to_pandas()

        if isinstance(x, Series) and isinstance(x.index, pd.MultiIndex):
            x = DataFrame(x)
        elif isinstance(x, Series):
            raise ValueError('Series can only be used with a 2-level MultiIndex')

        if isinstance(x, (Panel, DataFrame)):
            if isinstance(x, DataFrame):
                if isinstance(x.index, pd.MultiIndex):
                    if len(x.index.levels) != 2:
                        raise ValueError('DataFrame input must have a '
                                         'MultiIndex with 2 levels')
                    self._frame = x.copy()
                else:
                    self._frame = DataFrame({var_name: x.T.stack(dropna=False)})
            else:
                self._frame = x.swapaxes(1, 2).to_frame(filter_observations=False)
        elif isinstance(x, ndarray):
            if x.ndim not in (2, 3):
                raise ValueError('2 or 3-d array required for numpy input')
            if x.ndim == 2:
                x = x[None, :, :]

            k, t, n = x.shape
            var_str = var_name + '.{0:0>' + str(int(np.log10(k) + .01)) + '}'
            variables = [var_name] if k == 1 else [var_str.format(i) for i in range(k)]
            entity_str = 'entity.{0:0>' + str(int(np.log10(n) + .01)) + '}'
            entities = [entity_str.format(i) for i in range(n)]
            time = list(range(t))
            x = x.astype(np.float64)
            panel = _Panel.from_array(x, items=variables, major_axis=time,
                                      minor_axis=entities)
            self._fake_panel = panel
            self._frame = panel.to_frame()
        else:
            raise TypeError('Only ndarrays, DataFrames, Panels or DataArrays '
                            'supported.')
        if convert_dummies:
            self._frame = expand_categoricals(self._frame, drop_first)
            self._frame = self._frame.astype(np.float64)

        time_index = Series(self._frame.index.levels[1])
        if not (is_numeric_dtype(time_index.dtype) or
                is_datetime64_any_dtype(time_index.dtype)):
            raise ValueError('The index on the time dimension must be either '
                             'numeric or date-like')
        self._k, self._t, self._n = self.panel.shape
        self._frame.index.levels[0].name = 'entity'
        self._frame.index.levels[1].name = 'time'

    @property
    def panel(self):
        """pandas Panel view of data"""
        return _Panel(self._frame)

    @property
    def dataframe(self):
        """pandas DataFrame view of data"""
        return self._frame

    @property
    def values2d(self):
        """NumPy ndarray view of dataframe"""
        return self._frame.values

    @property
    def values3d(self):
        """NumPy ndarray view of panel"""
        return self.panel.values

    def drop(self, locs):
        """
        Parameters
        ----------
        locs : ndarray
            Booleam array indicating observations to drop with reference to
            the dataframe view of the data
        """
        self._frame = self._frame.loc[~locs.ravel()]
        self._frame = self._minimize_multiindex(self._frame)
        self._k, self._t, self._n = self.shape

    @property
    def shape(self):
        """Shape of panel view of data"""
        return self.panel.shape

    @property
    def ndim(self):
        """Number of dimensions of panel view of data"""
        return 3

    @property
    def isnull(self):
        """Locations with missing observations"""
        return np.any(self._frame.isnull(), axis=1)

    @property
    def nobs(self):
        """Number of time observations"""
        return self._t

    @property
    def nvar(self):
        """Number of variables"""
        return self._k

    @property
    def nentity(self):
        """Number of entities"""
        return self._n

    @property
    def vars(self):
        """List of variable names"""
        return list(self._frame.columns)

    @property
    def time(self):
        """List of time index names"""
        index = self._frame.index
        return list(index.levels[1][index.labels[1]].unique())

    @property
    def entities(self):
        """List of entity index names"""
        index = self._frame.index
        return list(index.levels[0][index.labels[0]].unique())

    @property
    def entity_ids(self):
        """
        Get array containing entity group membership information

        Returns
        -------
        id : ndarray
            2d array containing entity ids corresponding dataframe view
        """
        return np.asarray(self._frame.index.labels[0])[:, None]

    @property
    def time_ids(self):
        """
        Get array containing time membership information

        Returns
        -------
        id : ndarray
            2d array containing time ids corresponding dataframe view
        """
        return np.asarray(self._frame.index.labels[1])[:, None]

    def _demean_both(self, weights):
        """
        Entity and time demean

        Parameters
        ----------
        weights : PanelData, optional
             Weights to use in demeaning
        """
        if self.nentity > self.nobs:
            group = 'entity'
            dummy = 'time'
        else:
            group = 'time'
            dummy = 'entity'
        e = self.demean(group, weights=weights)
        d = self.dummies(dummy, drop_first=True)
        d.index = e.index
        d = PanelData(d).demean(group, weights=weights)
        d = d.values2d
        e = e.values2d
        resid = e - d @ np.linalg.lstsq(d, e)[0]
        resid = DataFrame(resid, index=self._frame.index, columns=self._frame.columns)

        return PanelData(resid)

    def general_demean(self, groups, weights=None):
        """
        Multi-way demeaning using only groupby

        Parameters
        ----------
        groups : PanelData
            Arrays with the same size containing group identifiers
        weights : PanelData, optional
            Weights to use in the weighted demeaning

        Returns
        -------
        demeaned : PanelData
            Weighted, demeaned data according to groups

        Notes
        -----
        Iterates until convergence
        """
        if not isinstance(groups, PanelData):
            groups = PanelData(groups)
        if weights is None:
            weights = PanelData(pd.DataFrame(np.ones((self._frame.shape[0], 1)),
                                             index=self.index,
                                             columns=['weights']))
        weights = weights.values2d
        groups = groups.values2d.astype(np.int64)

        weight_sum = {}

        def weighted_group_mean(df, weights, root_w, level):
            num = (root_w * df).groupby(level=level).transform('sum')
            if level in weight_sum:
                denom = weight_sum[level]
            else:
                denom = weights.groupby(level=level).transform('sum')
                weight_sum[level] = denom
            return num.values / denom.values

        def demean_pass(frame, weights, root_w):
            levels = groups.shape[1]
            for level in range(levels):
                mu = weighted_group_mean(frame, weights, root_w, level)
                if level == 0:
                    frame = frame - root_w * mu
                else:
                    frame -= root_w * mu

            return frame

        # Swap out the index for better performance
        init_index = pd.DataFrame(groups)
        init_index.set_index(list(init_index.columns), inplace=True)

        root_w = np.sqrt(weights)
        weights = pd.DataFrame(weights, index=init_index.index)
        wframe = root_w * self._frame
        wframe.index = init_index.index

        previous = wframe
        current = demean_pass(previous, weights, root_w)
        if groups.shape[1] == 1:
            current.index = self._frame.index
            return PanelData(current)

        exclude = np.ptp(self._frame.values, 0) == 0
        max_rmse = np.sqrt(self._frame.values.var(0).max())
        scale = self._frame.std().values
        exclude = exclude | (scale < 1e-14 * max_rmse)
        replacement = np.maximum(scale, 1)
        scale[exclude] = replacement[exclude]
        scale = scale[None, :]

        while np.max(np.abs(current.values - previous.values) / scale) > 1e-8:
            previous = current
            current = demean_pass(previous, weights, root_w)
        current.index = self._frame.index

        return PanelData(current)

    def demean(self, group='entity', weights=None):
        """
        Demeans data by either entity or time group

        Parameters
        ----------
        group : {'entity', 'time'}
            Group to use in demeaning
        weights : PanelData, optional
            Weights to implement weighted averaging

        Returns
        -------
        demeaned : PanelData
            Demeaned data according to type

        Notes
        -----
        If weights are provided, the values returned will be scaled by
        sqrt(weights) so that they can be used in WLS estimation.
        """
        if group not in ('entity', 'time', 'both'):
            raise ValueError
        if group == 'both':
            return self._demean_both(weights)

        level = 0 if group == 'entity' else 1
        if weights is None:
            group_mu = self._frame.groupby(level=level).transform('mean')
            return PanelData(self._frame - group_mu)
        else:
            w = weights.values2d
            frame = self._frame.copy()
            frame = w * frame
            weighted_sum = frame.groupby(level=level).transform('sum')
            frame.iloc[:, :] = w
            sum_weights = frame.groupby(level=level).transform('sum')
            group_mu = weighted_sum / sum_weights
            return PanelData(np.sqrt(w) * (self._frame - group_mu))

    def __str__(self):
        return self.__class__.__name__ + '\n' + str(self._frame)

    def __repr__(self):
        return self.__str__() + '\n' + self.__class__.__name__ + ' object, id: ' + hex(id(self))

    def _repr_html_(self):
        return self.__class__.__name__ + '<br/>' + self._frame._repr_html_()

    def count(self, group='entity'):
        """
        Count number of observations by entity or time

        Parameters
        ----------
        group : {'entity', 'time'}
            Group to use in demeaning

        Returns
        -------
        count : DataFrame
            Counts according to type. Either (entity by var) or (time by var)
        """
        v = self.panel.values
        axis = 1 if group == 'entity' else 2
        count = np.sum(np.isfinite(v), axis=axis)

        index = self.panel.minor_axis if group == 'entity' else self.panel.major_axis
        out = DataFrame(count.T, index=index, columns=self.vars)
        reindex = self.entities if group == 'entity' else self.time
        out = out.loc[reindex].astype(np.int64)
        out.index.name = group
        return out

    @property
    def index(self):
        """Return the index of the multi-index dataframe view"""
        return self._frame.index

    def copy(self):
        """Return a deep copy"""
        return PanelData(self._frame.copy(), var_name=self._var_name,
                         convert_dummies=self._convert_dummies, drop_first=self._drop_first)

    def mean(self, group='entity', weights=None):
        """
        Compute data mean by either entity or time group

        Parameters
        ----------
        group : {'entity', 'time'}
            Group to use in demeaning
        weights : PanelData, optional
            Weights to implement weighted averaging

        Returns
        -------
        mean : DataFrame
            Data mean according to type. Either (entity by var) or (time by var)
        """
        level = 0 if group == 'entity' else 1
        if weights is None:
            mu = self._frame.groupby(level=level).mean()
        else:
            w = weights.values2d
            frame = self._frame.copy()
            frame = w * frame
            weighted_sum = frame.groupby(level=level).sum()
            frame.iloc[:, :] = w
            sum_weights = frame.groupby(level=level).sum()
            mu = weighted_sum / sum_weights

        reindex = self.entities if group == 'entity' else self.time
        out = mu.loc[reindex]

        return out

    def first_difference(self):
        """
        Compute first differences of variables

        Returns
        -------
        diffs : PanelData
            Differenced values
        """
        diffs = self.panel.values
        diffs = diffs[:, 1:] - diffs[:, :-1]
        diffs = Panel(diffs, items=self.panel.items,
                      major_axis=self.panel.major_axis[1:],
                      minor_axis=self.panel.minor_axis)
        diffs = diffs.swapaxes(1, 2).to_frame(filter_observations=False)
        diffs = diffs.reindex(self._frame.index).dropna(how='any')
        return PanelData(diffs)

    @staticmethod
    def _minimize_multiindex(df):
        index_cols = list(df.index.names)
        orig_names = index_cols[:]
        for i, col in enumerate(index_cols):
            col = ensure_unique_column(col, df)
            index_cols[i] = col
        df.index.names = index_cols
        df = df.reset_index()
        df = df.set_index(index_cols)
        df.index.names = orig_names
        return df

    def dummies(self, group='entity', drop_first=False):
        """
        Generate entity or time dummies

        Parameters
        ----------
        group : {'entity', 'time'}, optional
            Type of dummies to generate
        drop_first : bool, optional
            Flag indicating that the dummy column corresponding to the first
            entity or time period should be dropped

        Returns
        -------
        dummies : DataFrame
            Dummy variables
        """
        if group not in ('entity', 'time'):
            raise ValueError
        axis = 0 if group == 'entity' else 1
        labels = self._frame.index.labels
        levels = self._frame.index.levels
        cat = pd.Categorical(levels[axis][labels[axis]])
        dummies = pd.get_dummies(cat, drop_first=drop_first)
        cols = self.entities if group == 'entity' else self.time
        return dummies[[c for c in cols if c in dummies]].astype(np.float64)
