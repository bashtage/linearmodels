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

from .data import PanelData
from .dummy_iterator import DummyVariableIterator


class RequiredSubclassingError(Exception):
    def __init__(self):
        # Call the base class constructor with the parameters it needs
        super(RequiredSubclassingError, self).__init__('Subclasses must implement')


class SaturatedEffectWarning(Warning):
    pass


not_implemented_msg = 'Subclasses must implement'


def _get_values(original):
    if isinstance(original, (xr.DataArray, pd.Panel)):
        values = np.asarray(original.values, dtype=np.float64)
    else:
        values = original
    return values


class FixedEffectSet(object):
    data = None
    _fixed_effects = None
    id = None

    def __init__(self, *effects):
        self._fixed_effects = OrderedDict()
        self.data = effects[0].data

        for effect in effects:
            self._check_id(effect)
            self._fixed_effects[str(effect)] = effect

    def __len__(self):
        return len(self._fixed_effects)

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

    def _check_id(self, other):
        if other.data.id != self.data.id:
            raise ValueError('Can only add fixed effect to other '
                             'fixed effects based on the same data.')

    def __add__(self, other):
        if isinstance(other, FixedEffectSet):
            self._check_id(other)
            return FixedEffectSet(*(self.fixed_effects + other.fixed_effects))
        elif issubclass(other.__class__, FixedEffect):
            self._check_id(other)
            return FixedEffectSet(*(self.fixed_effects + [other]))
        else:
            raise ValueError('Unknown input type')

    def __sub__(self, other):
        if isinstance(other, FixedEffectSet):
            self._check_id(other)
            for fe in other.fixed_effects:
                key = str(fe)
                if key not in self._fixed_effects:
                    raise ValueError(key + ' is not in fixed effect set')
                del self._fixed_effects[key]
            return FixedEffectSet(*self.fixed_effects)
        elif issubclass(other.__class__, FixedEffect):
            self._check_id(other)
            key = str(other)
            if key not in self._fixed_effects:
                raise ValueError(key + ' is not in fixed effect set')
            del self._fixed_effects[key]
            return FixedEffectSet(*self.fixed_effects)
        else:
            raise ValueError('Cannot subtract unknown types')

    def orthogonalize(self):
        for i, key in enumerate(self._fixed_effects):
            if i == 0:
                x = self._fixed_effects[key].orthogonalize()
            else:
                effect = self._fixed_effects[key].__class__
                x = effect(x).orthogonalize()
        return x


class FixedEffect(object):
    _transform = None
    _groups = None
    _effect_name = ''
    _data = None

    def __init__(self, data):
        self.data = PanelData(data)

    def _effect_labels(self, first, last):
        raise RequiredSubclassingError  # pragma: no cover

    def _select_columns(self, exclude):
        skip = self._skip_columns()
        exclude = [] if exclude is None else exclude
        skip_or_exlude = list(set(skip + list(exclude)))

        original = self.data.columns
        self._selected = [c for c in original if c not in skip_or_exlude]

    def _split(self):
        return self.data.as3d[:, :, self._selected]

    def _restore(self, values):
        restored = self.data.data.copy()
        if self.data.is_pandas:
            restored.loc[:, :, self._selected] = values
            return restored
        restored[:, :, self._selected] = values

        return restored

    def _skip_columns(self):
        skip = []
        """Determine columns to skip"""
        if self.data.is_numpy or self.data.is_xarray:
            return skip

        for col in self.data.columns:
            temp = self.data.data[:, :, col]
            if np.all(list(map(is_numeric_dtype, temp.dtypes))):
                continue
            try:
                temp.apply(pd.to_numeric)
            except ValueError:
                skip.append(col)

        return skip

    def __add__(self, other):
        # TODO Fix this
        if not issubclass(other.__class__, FixedEffect) or \
                isinstance(other, FixedEffectSet):
            raise ValueError('Can only add other fixed effects')
        return FixedEffectSet(self, other)

    def _repr_html_(self):
        return '<b>' + self.__str__() + '</b>'

    def __str__(self):
        raise RequiredSubclassingError  # pragma: no cover

    def __repr__(self):
        return self.__str__()  # pragma: no cover

    def orthogonalize(self, exclude=None):
        raise RequiredSubclassingError  # pragma: no cover

    def groups(self):
        if self._groups is None:
            self._groups = self._construct_groups()
        return self._groups

    def _construct_groups(self):
        raise RequiredSubclassingError  # pragma: no cover

    def estimate(self, endog, exog=None, drop=False):

        dummies = self.dummies(drop=drop, iterator=True)

        endog_col = self.data.column_index(endog)
        endog = self.data.asnumpy2d[:, endog_col]

        if exog is not None:
            exog_col = self.data.column_index(exog)
            exog = self.data.asnumpy2d[:, exog_col]

        effects = []
        xpxi = np.linalg.inv(exog.T.dot(exog)) if exog is not None else None
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

    def dummies(self, drop=False, iterator=False, max_size=10):
        n, t = self.data.n, self.data.t
        groups = self.groups()
        max_size = max_size if iterator else 2 ** 62
        dummy_iterator = DummyVariableIterator(n, t, groups, max_size=max_size, drop=drop)
        if iterator:
            return dummy_iterator
        dummy_iterator.__iter__()
        return dummy_iterator.__next__()


class EntityEffect(FixedEffect):
    def __init__(self, data):
        super(EntityEffect, self).__init__(data)

    def _effect_labels(self, first, last):
        index = self.data.entity_index
        lbls = ['EntityEffect.{0}'.format(index) for i in range(first, last)]
        return lbls

    def __str__(self):
        return 'Entity effect'

    def _construct_groups(self):
        n, t = self.data.n, self.data.t
        return np.tile(np.arange(n)[:, None], (1, t)).ravel()

    def orthogonalize(self, exclude=None):
        # Take subset of data to transform
        self._select_columns(exclude)
        _data = self._split()
        # Transform subset
        _data = _get_values(_data)
        _data = _data.swapaxes(0, 1)
        # Check for saturation
        counts = np.sum(np.isnan(_data), 0)
        non_zero = _data.shape[0] - counts
        if np.any(non_zero == 1):
            warn('Entity effects have saturated the data for one or more '
                 'series.  All residuals will be 0.', SaturatedEffectWarning)

        _data = (_data - np.nanmean(_data, 0))
        _data = _data.swapaxes(0, 1)
        # Restore to original form
        out = self._restore(_data)

        return out


class TimeEffect(FixedEffect):
    def _effect_labels(self, first, last):
        lbls = ['TimeEffect.{0:d}'.format(i) for i in range(first, last)]
        return lbls

    def __str__(self):
        return 'Time effect'

    def _construct_groups(self):
        n, t = self.data.n, self.data.t
        return np.tile(np.arange(t)[:, None], (n, 1)).ravel()

    def orthogonalize(self, exclude=None):
        # Take subset of data to transform
        self._select_columns(exclude)
        _data = self._split()
        # Transform subset
        _data = _get_values(_data)

        # Check for saturation
        counts = np.sum(np.isnan(_data), 0)
        non_zero = _data.shape[0] - counts
        if np.any(non_zero == 1):
            warn('Time effects have saturated the data for one or more '
                 'series.  All residuals will be 0.', SaturatedEffectWarning)

        _data = _data - np.nanmean(_data, 0)
        # Restore to original form
        return self._restore(_data)


class GroupEffect(FixedEffect):
    def __init__(self, data, columns, time=False, entity=False):
        super(GroupEffect, self).__init__(data)
        if not isinstance(columns, (list, tuple)):
            raise ValueError('columns must be a list or tuple')
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

    def _construct_groups(self):
        if self.data.is_numpy:
            return _numpy_groups(self.data.data, self.columns, self.time, self.entity)
        elif self.data.is_pandas:
            return _pandas_groups(self.data.data, self.columns, self.time, self.entity)
        else:
            return _numpy_groups(self.data.asnumpy3d, self.columns, self.time, self.entity)

    def orthogonalize(self, exclude=None):
        columns = self.columns
        if len(columns) == 0:
            if self.time:
                return TimeEffect(self.data).orthogonalize()
            elif self.entity:
                return EntityEffect(self.data).orthogonalize()

        data = self.data.data
        if self.data.is_numpy:
            return _numpy_groupwise_demean(data, self.columns,
                                           self.time, self.entity)
        elif self.data.is_pandas:
            return _pandas_groupwise_demean(data, self.columns,
                                            self.time, self.entity)
        else:
            return _xarray_groupwise_demean(data, self.columns,
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
