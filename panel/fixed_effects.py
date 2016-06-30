# TODO: Dummies
# TODO: Dummy iterator
# TODO: FE Estimation
# TODO: Ordering?
"""
Tools for manipulating FixedEffect in Panel models
"""
from collections import OrderedDict
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from pandas.core.common import is_numeric_dtype
from scipy.sparse import csc_matrix


class SaturatedEffectWarning(Warning):
    pass


def _get_values(original):
    if isinstance(original, (xr.DataArray, pd.Panel)):
        values = np.asarray(original.values, dtype=np.float64)
    else:
        values = original
    return values


class DummyVariableIterator(object):
    """
    Iterator object that produces dummy variables
    """

    def __init__(self, n, t, groups, drop=False, max_size=10, sparse=False):
        self.n = n
        self.t = t
        self.groups = groups
        self.max_size = max_size
        self.sparse = sparse
        self.drop = drop
        self._array_cols = max(1, int(self.max_size * 2.0 ** 20 / 8.0 / (n * t)))
        groups = groups.astype(np.int64)
        self._rows = int(n * t)
        self._index = 0
        ugroups = np.unique(groups)
        if np.min(ugroups) != 0 or np.any(np.diff(ugroups) != 1):
            raise ValueError('groups must contain elements in {0,1,...,max}')
        if len(groups) != self._rows:
            raise ValueError('groups must have n * t elements')
        self._group_index = np.argsort(groups)
        ordered = self._ordered_groups = groups[self._group_index]
        locs = np.argwhere(np.diff(ordered) != 0)
        self._ends = np.concatenate([[[0]], locs + 1, [[len(ordered)]]])
        self._ends = self._ends.ravel()

    def __iter__(self):
        self._iter_count = 0
        self._remaining_cols = self.groups.max() + 1
        return self

    def __next__(self):
        if self._remaining_cols <= 0:
            raise StopIteration

        cols = min(self._remaining_cols, self._array_cols)
        self._remaining_cols -= cols
        ends = self._ends
        rows = self._group_index[ends[self._index]:ends[self._index + cols]]
        group_ids = self._ordered_groups[ends[self._index]:ends[self._index + cols]]
        columns = group_ids - self._index
        self._index += cols
        if not self.sparse:
            out = np.zeros((self._rows, cols))
            out[rows, columns] = 1
        else:
            values = np.ones_like(columns)
            locs = (rows, columns)
            shape = (self._rows, cols)
            out = csc_matrix((values, locs), shape=shape, dtype=np.float64)

        if self.drop and np.any(group_ids == 0):
            out = out[:, 1:]
            # Ensure never return empty column
            if out.shape[1] == 0:
                return self.__next__()

        return out


class FixedEffectSet(object):
    def __init__(self, *effects):
        self._fixed_effects = OrderedDict()
        for effect in effects:
            self._fixed_effects[str(effect)] = effect

    def __str__(self):
        effects_str = ', '.join(list(map(str, self.fixed_effects)))
        return 'FixedEffectSet: ' + effects_str

    def _repr_html_(self):
        effects_str = ', '.join(list(map(str, self.fixed_effects)))
        return '<b>FixedEffectSet</b>: ' + effects_str

    def __repr__(self):
        return self.__str__()

    @property
    def fixed_effects(self):
        return list(self._fixed_effects.values())

    def __add__(self, other):
        if isinstance(other, FixedEffectSet):
            return FixedEffectSet(*(self.fixed_effects + other.fixed_effects))
        elif issubclass(other.__class__, FixedEffect):
            return FixedEffectSet(*(self.fixed_effects + [other]))
        else:
            raise ValueError('Unknown input type')

    def __sub__(self, other):
        if isinstance(other, FixedEffectSet):
            for fe in other.fixed_effects:
                key = str(fe)
                if key not in self._fixed_effects:
                    raise ValueError(key + ' is not in fixed effect set')
                del self._fixed_effects[key]
            return FixedEffectSet(*self.fixed_effects)
        elif issubclass(other.__class__, FixedEffect):
            key = str(other)
            if key not in self._fixed_effects:
                raise ValueError(key + ' is not in fixed effect set')
            del self._fixed_effects[key]
            return FixedEffectSet(*self.fixed_effects)
        else:
            raise ValueError('Unknown input type')

    def orthogonalize(self, x):
        for key in self._fixed_effects:
            x = self._fixed_effects[key].orthogonalize(x)
        return x


class FixedEffect(object):
    _transform = None
    _groups = None
    _effect_name = ''

    def _effect_labels(self, first, last):
        lbls = ['UnnamedEffect.{0:d}'.format(i) for i in range(first, last)]
        return lbls

    def _select_columns(self, x, exclude):
        skip = self._skip_columns(x)
        exclude = [] if exclude is None else exclude
        skip_or_exlude = list(set(skip + list(exclude)))

        if isinstance(x, (np.ndarray, xr.DataArray)):
            original = np.arange(x.shape[2])
        else:
            original = x.minor_axis  # pandas

        self._selected = [c for c in original if c not in skip_or_exlude]

    def _split(self, x):
        return x[:, :, self._selected]

    def _restore(self, values, orig):
        restored = orig.copy()
        if isinstance(orig, pd.Panel):
            restored.loc[:, :, self._selected] = values
            return restored
        restored[:, :, self._selected] = values

        return restored

    @staticmethod
    def _skip_columns(x):
        skip = []
        """Determine columns to skip"""
        if isinstance(x, (np.ndarray, xr.DataArray)):
            return skip

        for col in x.minor_axis:
            temp = x[:, :, col]
            if np.all(list(map(is_numeric_dtype, temp.dtypes))):
                continue
            try:
                temp.apply(pd.to_numeric)
            except ValueError:
                skip.append(col)

        return skip

    @staticmethod
    def _validate_input(x):
        """Ensures the input is one of the supported types"""
        if isinstance(x, np.ndarray):
            if x.ndim != 3:
                raise ValueError('NumPy array must have 3 dimensions')
        elif isinstance(x, xr.DataArray):
            if x.values.ndim != 3:
                raise ValueError('xarray DataArray must have 3 dimensions')
        elif isinstance(x, pd.Panel):
            pass
        else:
            raise ValueError('Unknown data type -- subclasses not supported')

    def __add__(self, other):
        if not issubclass(other.__class__, FixedEffect) or \
                isinstance(other, FixedEffectSet):
            raise ValueError('Can only add other fixed effects')
        return FixedEffectSet(self, other)

    def _repr_html_(self):
        return '<b>' + self.__str__() + '</b>'

    def __str__(self):
        raise NotImplementedError('Subclasses must implement') \
            # pragma: no cover

    def __repr__(self):
        return self.__str__()  # pragma: no cover

    def orthogonalize(self, x, exclude=None):
        raise NotImplementedError('Subclasses must implement') \
            # pragma: no cover

    def groups(self, x):
        if self._groups is None:
            self._groups = self._construct_groups(x)
        return self._groups

    def _construct_groups(self, x):
        raise NotImplementedError('Subclasses must implement') \
            # pragma: no cover

    def estimate(self, endog, exog=None, drop=False):
        dummies = self.dummies(endog, drop=drop, iterator=True)
        n,t,_ = endog.shape

        endog = np.asarray(endog).reshape((n*t,1))

        if exog is not None:
            k = exog.shape[2]
            exog = np.asarray(exog).reshape((n*t,k))

        effects = []
        if exog is not None:
            xpxi = np.linalg.inv(exog.T.dot(exog))
        for d in dummies:
            if exog is None:
                effects.append(d.T.dot(endog).ravel() / d.sum(0))
            else:
                beta = xpxi.dot(exog.T.dot(d))
                r = d - exog.dot(beta)
                lstsq_out = np.linalg.lstsq(r, endog)
                effects.append(lstsq_out[0].ravel())
        effects = np.concatenate(effects)

        index = self._effect_labels(int(drop), len(effects) + int(drop))
        return pd.Series(effects, index=index)

    def dummies(self, x, drop=False, iterator=False, max_size=10):
        n, t, k = x.shape
        groups = self.groups(x)
        max_size = max_size if iterator else 2 ** 62
        dummy_iterator = DummyVariableIterator(n, t, groups, max_size=max_size, drop=drop)
        if iterator:
            return dummy_iterator
        dummy_iterator.__iter__()
        return dummy_iterator.__next__()


class EntityEffect(FixedEffect):
    def _effect_labels(self, first, last):
        lbls = ['EntityEffect.{0:d}'.format(i) for i in range(first, last)]
        return lbls

    def __str__(self):
        return 'Entity effect'

    def _construct_groups(self, x):
        n, t, k = x.shape
        return np.tile(np.arange(n)[:, None], (1, t)).ravel()

    def orthogonalize(self, x, exclude=None):
        # Validate input
        self._validate_input(x)
        # Take subset of data to transform
        self._select_columns(x, exclude)
        _x = self._split(x)
        # Transform subset
        _x = _get_values(_x)
        _x = _x.swapaxes(0, 1)
        # Check for saturation
        counts = np.sum(np.isnan(_x), 0)
        non_zero = _x.shape[0] - counts
        if np.any(non_zero == 1):
            warn('Entity effects have saturated the data for one or more '
                 'series.  All residuals will be 0.', SaturatedEffectWarning)

        _x = (_x - np.nanmean(_x, 0))
        _x = _x.swapaxes(0, 1)
        # Restore to original form
        out = self._restore(_x, x)

        return out



class TimeEffect(FixedEffect):
    def _effect_labels(self, first, last):
        lbls = ['TimeEffect.{0:d}'.format(i) for i in range(first, last)]
        return lbls

    def __str__(self):
        return 'Time effect'

    def _construct_groups(self, x):
        n, t, k = x.shape
        return np.tile(np.arange(t)[:, None], (n, 1)).ravel()

    def orthogonalize(self, x, exclude=None):
        # Validate input
        self._validate_input(x)
        # Take subset of data to transform
        self._select_columns(x, exclude)
        _x = self._split(x)
        # Transform subset
        _x = _get_values(_x)

        # Check for saturation
        counts = np.sum(np.isnan(_x), 0)
        non_zero = _x.shape[0] - counts
        if np.any(non_zero == 1):
            warn('Time effects have saturated the data for one or more '
                 'series.  All residuals will be 0.', SaturatedEffectWarning)

        _x = _x - np.nanmean(_x, 0)
        # Restore to original form
        out = self._restore(_x, x)

        return out


class GroupEffect(FixedEffect):
    def __init__(self, columns, time=False, entity=False, data=None):
        if not isinstance(columns, (list, tuple)):
            raise ValueError('columns must be a list or tuple')
        self.data = data
        self.columns = columns
        self.time = time
        self.entity = entity
        if time and entity:
            raise ValueError('Effects with both time and entity components '
                             'fully saturate the model')

    def _effect_labels(self, first, last):
        name = 'GroupEffect(' + ', '.join(map(str, self.columns)) + ')'
        lbls = ['{0}.{1:d}'.format(name, i) for i in range(first, last)]
        return lbls

    def _construct_groups(self, x):
        self._validate_input(x)
        if isinstance(x, np.ndarray):
            return _numpy_groups(x, self.columns, self.time, self.entity)
        elif isinstance(x, pd.Panel):
            return _pandas_groups(x, self.columns, self.time, self.entity)
        else:
            raise NotImplementedError

    def orthogonalize(self, x, exclude=None):
        columns = self.columns
        if len(columns) == 0:
            if self.time:
                return TimeEffect().orthogonalize(x)
            elif self.entity:
                return EntityEffect().orthogonalize(x)

        self._validate_input(x)
        if isinstance(x, np.ndarray):
            return _numpy_groupwise_demean(x, self.columns,
                                           self.time, self.entity)
        elif isinstance(x, pd.Panel):
            return _pandas_groupwise_demean(x, self.columns,
                                            self.time, self.entity)
        else:
            return _xarray_groupwise_demean(x, self.columns,
                                            self.time, self.entity)

    def __str__(self):
        return 'Group Effect (columns: ' + \
               ', '.join(map(str, self.columns)) + ')'


def _numpy_groups(x, cols, time=False, entity=False):
    """generic groupby demean for numpy (numeric) arrays"""

    n, t, k = x.shape
    # Use a copy of the sort columns
    sort_columns = x.reshape((n * t, k))[:, cols].copy()
    # Add time/entity columns if needed
    if time:
        tc = np.tile(np.arange(t), (n,)).reshape((n * t, 1))
        sort_columns = np.column_stack((sort_columns, tc))
    if entity:
        ec = np.tile(np.arange(n), (t, 1)).T.reshape((n * t, 1))
        sort_columns = np.column_stack((sort_columns, ec))
    # Lexicgraphic sort
    ind = np.lexsort(sort_columns.T)
    # Compute block start and end points
    sort_columns = sort_columns[ind]
    loc = np.argwhere((sort_columns[:-1] != sort_columns[1:]).any(1))
    st = np.concatenate([[[0]], loc + 1]).ravel()
    en = np.concatenate([loc + 1, [[n * t]]]).ravel()
    groups = np.empty(n * t, dtype=np.int64)
    for i in range(len(st)):
        groups[st[i]:en[i]] = i
    out = np.empty(n * t, dtype=np.int64)
    out[ind] = groups

    return out


def _pandas_groups(x, cols, time=False, entity=False):
    _x = x.swapaxes(0, 2).to_frame()
    if time or entity:
        n, t, k = x.shape
        extra_col = '___entity_or_time__'
        while extra_col in _x:
            extra_col = '_' + extra_col + '_'
        cols = cols[:] + [extra_col]

        if time:
            col_vals = np.tile(np.arange(t), (n,))
        else:
            col_vals = np.tile(np.arange(t)[:, None], (1, n)).ravel()
        _x[extra_col] = pd.Series(col_vals, index=_x.index)

    # Reset to RangeIndex to work around GH #13
    _x.index = pd.RangeIndex(0, _x.shape[0])
    _x = _x[cols]
    # Get original index
    groups = _x.groupby(cols)
    labels = groups.grouper.labels
    out = labels[0].copy()
    if len(cols) > 1:
        for i in range(1, len(labels)):
            out += (labels[i] * (out.max() + 1))
        uniques = np.unique(out)
        if len(uniques) != (np.max(uniques) + 1):
            out = pd.DataFrame(out).groupby([0]).grouper.labels[0]

    return out


def _numpy_groupwise_demean(x, cols, time=False, entity=False):
    """generic groupby demean for numpy (numeric) arrays"""

    n, t, k = x.shape
    _x = x.copy()
    # Flatten to 2d array
    _x = _x.reshape((n * t, k))
    # Use a copy of the sort columns
    orig_values = _x[:, cols].copy()
    sort_columns = _x[:, cols].copy()
    # Add time/entity columns if needed
    if time:
        tc = np.tile(np.arange(t), (n,)).reshape((n * t, 1))
        sort_columns = np.column_stack((sort_columns, tc))
    if entity:
        ec = np.tile(np.arange(n), (t, 1)).T.reshape((n * t, 1))
        sort_columns = np.column_stack((sort_columns, ec))
    # Lexicgraphic sort
    ind = np.lexsort(sort_columns.T)
    # Compute block start and end points
    sort_columns = sort_columns[ind]
    loc = np.argwhere((sort_columns[:-1] != sort_columns[1:]).any(1))
    st = np.concatenate([[[0]], loc + 1]).ravel()
    en = np.concatenate([loc + 1, [[n * t]]]).ravel()
    # Reorder x
    _x = _x[ind]
    # Demean within a block
    for i in range(len(st)):
        _x[st[i]:en[i]] -= np.nanmean(_x[st[i]:en[i]], 0)
    # Restore original order
    _x = _x[np.argsort(ind)]
    # Restore values to group by columns
    _x[:, cols] = orig_values
    # Restore original shape
    _x = _x.reshape((n, t, k))

    return _x


def _xarray_groupwise_demean(x, cols, time=False, entity=False):
    """generic groupby demean for xarray (mixed) arrays"""
    coords = x.coords[x.dims[2]]
    col_index = [i for i, c in enumerate(coords) if c.data in cols]
    values = _numpy_groupwise_demean(x.values, col_index,
                                     time=False, entity=False)
    out = x.copy()
    out[:, :, :] = values
    return out


def _pandas_groupwise_demean(x, cols, time=False, entity=False):
    _x = x.swapaxes(0, 2).to_frame()
    orig_cols = _x.columns
    if time or entity:
        _x = _x.copy()
        n, t, k = x.shape
        extra_col = '___entity_or_time__'
        while extra_col in _x:
            extra_col = '_' + extra_col + '_'
        cols = cols[:] + [extra_col]

        if time:
            col_vals = np.tile(np.arange(t), (n,))
        else:
            col_vals = np.tile(np.arange(t)[:, None], (1, n)).ravel()
        _x[extra_col] = pd.Series(col_vals, index=_x.index)

    demean_cols = []
    for df_col in _x:
        if df_col not in cols:
            if pd.core.common.is_numeric_dtype(_x[df_col].dtype):
                demean_cols.append(df_col)
                continue
            try:
                pd.to_numeric(_x.loc[:, df_col])
                demean_cols.append(df_col)
            except ValueError:
                pass

    no_change_cols = [col for col in _x if col not in demean_cols]

    # Function start
    orig_index = _x.index
    # Reset to RangeIndex to work around GH #13
    _x.index = pd.RangeIndex(0, _x.shape[0])
    no_change = _x[no_change_cols]
    _x = _x[demean_cols + cols]
    for col in demean_cols:
        _x[col] = _x[col].astype(np.float64)
    # Get original index
    groups = _x.groupby(cols)
    means = groups.transform('mean')
    out = _x[demean_cols] - means
    out.sort_index(inplace=True)
    out = pd.concat([out, no_change], 1)
    out.sort_index(inplace=True)
    out.index = orig_index
    out = out[orig_cols]

    out = out.to_panel().swapaxes(0, 2)
    return out
