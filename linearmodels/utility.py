import functools
from collections import OrderedDict
from collections.abc import MutableMapping

import numpy as np
from numpy import NaN, ceil, diag, isnan, log10, sqrt
from numpy.linalg import eigh, matrix_rank
from pandas import DataFrame, Series, concat
from scipy.stats import chi2, f
from statsmodels.iolib.summary import SimpleTable, fmt_params


class MissingValueWarning(Warning):
    pass


missing_value_warning_msg = """
Inputs contain missing values. Dropping rows with missing observations."""

OrderedDict


class AttrDict(MutableMapping):
    """
    Ordered dictionary-like object that exposes keys as attributes
    """

    def update(self, *args, **kwargs):
        self.__ordered_dict__.update(*args, **kwargs)

    def clear(self):
        self.__ordered_dict__.clear()

    def copy(self):
        ad = AttrDict()
        for key in self.__ordered_dict__.keys():
            ad[key] = self.__ordered_dict__[key]
        return ad

    def keys(self):
        return self.__ordered_dict__.keys()

    def items(self):
        return self.__ordered_dict__.items()

    def values(self):
        return self.__ordered_dict__.values()

    def pop(self, key, default=None):
        return self.__ordered_dict__.pop(key, default)

    def __len__(self):
        return self.__ordered_dict__.__len__()

    def __repr__(self):
        out = self.__ordered_dict__.__str__()
        return 'Attr' + out[7:]

    def __str__(self):
        return self.__repr__()

    def __init__(self, *args, **kwargs):
        self.__dict__['__ordered_dict__'] = OrderedDict(*args, **kwargs)

    def __contains__(self, item):
        return self.__ordered_dict__.__contains__(item)

    def __getitem__(self, item):
        return self.__ordered_dict__[item]

    def __setitem__(self, key, value):
        if key == '__ordered_dict__':
            raise KeyError(key + ' is reserved and cannot be set.')
        self.__ordered_dict__[key] = value

    def __delitem__(self, key):
        del self.__ordered_dict__[key]

    def __getattr__(self, item):
        if item not in self.__ordered_dict__:
            raise AttributeError
        return self.__ordered_dict__[item]

    def __setattr__(self, key, value):
        if key == '__ordered_dict__':
            raise AttributeError(key + ' is invalid')
        self.__ordered_dict__[key] = value

    def __delattr__(self, name):
        del self.__ordered_dict__[name]

    def __dir__(self):
        out = super(AttrDict, self).__dir__() + list(self.__ordered_dict__.keys())
        out = filter(lambda s: isinstance(s, str) and s.isidentifier(), out)
        return sorted(set(out))

    def __iter__(self):
        return self.__ordered_dict__.__iter__()


def has_constant(x):
    """
    Parameters
    ----------
    x: ndarray
        Array to be checked for a constant (n,k)

    Returns
    -------
    const : bool
        Flag indicating whether x contains a constant or has column span with
        a constant
    loc : int
        Column location of constant
    """
    if np.any(np.all(x == 1, axis=0)):
        loc = np.argwhere(np.all(x == 1, axis=0))
        return True, int(loc)

    if np.any((np.ptp(x, axis=0) == 0) & ~np.all(x == 0, axis=0)):
        loc = np.any((np.ptp(x, axis=0) == 0) & ~np.all(x == 0, axis=0))
        loc = np.argwhere(loc)
        return True, int(loc)

    n = x.shape[0]
    aug_rank = matrix_rank(np.c_[np.ones((n, 1)), x])
    rank = matrix_rank(x)
    has_const = bool(aug_rank == rank)
    loc = None
    if has_const:
        out = np.linalg.lstsq(x, np.ones((n, 1)))
        beta = out[0].ravel()
        loc = np.argmax(np.abs(beta) * x.var(0))
    return has_const, loc


def inv_sqrth(x):
    """
    Matrix inverse square root

    Parameters
    ----------
    x : ndarray
        Real, symmetric matrix

    Returns
    -------
    invsqrt : ndarray
        Input to the power -1/2
    """
    vals, vecs = eigh(x)
    return vecs @ diag(1 / sqrt(vals)) @ vecs.T


class WaldTestStatistic(object):
    """
    Test statistic holder for Wald-type iv

    Parameters
    ----------
    stat : float
        The test statistic
    null : str
        A statement of the test's null hypothesis
    df : int
        Degree of freedom.
    df_denom : int, optional
        Numerator degree of freedome.  If provided, uses an
        F(df, df_denom) distribution.
    name : str, optional
        Name of test

    See Also
    --------
    InvalidTestStatistic
    """

    def __init__(self, stat, null, df, df_denom=None, name=None):
        self._stat = stat
        self._null = null
        self.df = df
        self.df_denom = df_denom
        self._name = name
        if df_denom is None:
            self.dist = chi2(df)
            self.dist_name = 'chi2({0})'.format(df)
        else:
            self.dist = f(df, df_denom)
            self.dist_name = 'F({0},{1})'.format(df, df_denom)

    @property
    def stat(self):
        """Test statistic"""
        return self._stat

    @property
    def pval(self):
        """P-value of test statistic"""
        return 1 - self.dist.cdf(self.stat)

    @property
    def critical_values(self):
        """Critical values test for common test sizes"""
        return OrderedDict(zip(['10%', '5%', '1%'],
                               self.dist.ppf([.9, .95, .99])))

    @property
    def null(self):
        """Null hypothesis"""
        return self._null

    def __str__(self):
        name = '' if not self._name else self._name + '\n'
        msg = '{name}H0: {null}\nStatistic: {stat:0.4f}\n' \
              'P-value: {pval:0.4f}\nDistributed: {dist}'
        return msg.format(name=name, null=self.null, stat=self.stat,
                          pval=self.pval, dist=self.dist_name)

    def __repr__(self):
        return self.__str__() + '\n' + \
               self.__class__.__name__ + \
               ', id: {0}'.format(hex(id(self)))


class InvalidTestWarning(UserWarning):
    pass


class InvalidTestStatistic(WaldTestStatistic):
    """
    Class returned if a requested test is not valid for a model

    Parameters
    ----------
    reason : str
        Explanation why test is invalid
    name : str, optional
        Name of test

    See Also
    --------
    WaldTestStatistic
    """

    def __init__(self, reason, *, name=None):
        self._reason = reason
        super(InvalidTestStatistic, self).__init__(NaN, NaN, df=1, df_denom=1, name=name)
        self.dist_name = 'None'

    @property
    def pval(self):
        """Always returns NaN"""
        return NaN

    @property
    def critical_values(self):
        """Always returns None"""
        return None

    def __str__(self):
        msg = "Invalid test statistic\n{reason}\n{name}"
        name = '' if self._name is None else self._name
        return msg.format(name=name, reason=self._reason)


class InapplicableTestStatistic(WaldTestStatistic):
    """
    Class returned if a requested test is not applicable for a specification

    Parameters
    ----------
    reason : str
        Explanation why test is invalid
    name : str, optional
        Name of test

    See Also
    --------
    WaldTestStatistic
    """

    def __init__(self, *, reason=None, name=None):
        self._reason = reason
        if reason is None:
            self._reason = 'Test is not applicable to model specification'

        super(InapplicableTestStatistic, self).__init__(NaN, NaN, df=1, df_denom=1, name=name)
        self.dist_name = 'None'

    @property
    def pval(self):
        """Always returns NaN"""
        return NaN

    @property
    def critical_values(self):
        """Always returns None"""
        return None

    def __str__(self):
        msg = "Irrelevant test statistic\n{reason}\n{name}"
        name = '' if self._name is None else self._name
        return msg.format(name=name, reason=self._reason)


# cahced_property taken from bottle.py
# Copyright (c) 2016, Marcel Hellkamp.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
def update_wrapper(wrapper, wrapped, *a, **ka):
    try:
        functools.update_wrapper(wrapper, wrapped, *a, **ka)
    except AttributeError:  # pragma: no cover
        pass


class CachedProperty(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. """

    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:  # pragma: no cover
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


cached_property = CachedProperty


def _str(v):
    """Preferred basic formatter"""
    if isnan(v):
        return '        '
    av = abs(v)
    digits = 0
    if av != 0:
        digits = ceil(log10(av))
    if digits > 4 or digits <= -4:
        return '{0:8.4g}'.format(v)

    if digits > 0:
        d = int(5 - digits)
    else:
        d = int(4)

    format_str = '{0:' + '0.{0}f'.format(d) + '}'
    return format_str.format(v)


def pval_format(v):
    """Preferred formatting for x in [0,1]"""
    return '{0:4.4f}'.format(v)


class _SummaryStr(object):
    def __str__(self):
        return self.summary.as_text()

    def __repr__(self):
        return self.__str__() + '\n' + \
               self.__class__.__name__ + \
               ', id: {0}'.format(hex(id(self)))

    def _repr_html_(self):
        return self.summary.as_html() + '<br/>id: {0}'.format(hex(id(self)))


def ensure_unique_column(col_name, df, addition='_'):
    while col_name in df:
        col_name = addition + col_name + addition
    return col_name


class _ModelComparison(_SummaryStr):
    """
    Base class for model comparisons
    """
    _supported = tuple([])

    def __init__(self, results):
        if not isinstance(results, (dict, OrderedDict)):
            _results = OrderedDict()
            for i, res in enumerate(results):
                _results['Model ' + str(i)] = results[i]
            results = _results
        elif not isinstance(results, OrderedDict):
            _results = OrderedDict()
            for key in sorted(results.keys()):
                _results[key] = results[key]
            results = _results
        self._results = results

        for key in self._results:
            if not isinstance(self._results[key], self._supported):
                raise TypeError('Results from unknown model')

    def _get_series_property(self, name):
        out = ([(k, getattr(v, name)) for k, v in self._results.items()])
        cols = [v[0] for v in out]
        values = concat([v[1] for v in out], 1)
        values.columns = cols
        return values

    def _get_property(self, name):
        out = OrderedDict()
        items = []
        for k, v in self._results.items():
            items.append(k)
            out[k] = getattr(v, name)
        return Series(out, name=name).loc[items]

    @property
    def nobs(self):
        """Parameters for all models"""
        return self._get_property('nobs')

    @property
    def params(self):
        """Parameters for all models"""
        return self._get_series_property('params')

    @property
    def tstats(self):
        """Parameter t-stats for all models"""
        return self._get_series_property('tstats')

    @property
    def pvalues(self):
        """Parameter p-vals for all models"""
        return self._get_series_property('pvalues')

    @property
    def rsquared(self):
        """Coefficients of determination (R**2)"""
        return self._get_property('rsquared')

    @property
    def f_statistic(self):
        """F-statistics and P-values"""
        out = self._get_property('f_statistic')
        out_df = DataFrame(np.empty((len(out), 2)), columns=['F stat', 'P-value'], index=out.index)
        for loc in out.index:
            out_df.loc[loc] = out[loc].stat, out[loc].pval
        return out_df


def missing_warning(missing):
    """Utility function to perform missing value check and warning"""
    if not np.any(missing):
        return
    import linearmodels
    if linearmodels.WARN_ON_MISSING:
        import warnings
        warnings.warn(missing_value_warning_msg, MissingValueWarning)


def param_table(results, title, pad_bottom=False):
    """Formatted standard parameter table"""
    param_data = np.c_[results.params.values[:, None],
                       results.std_errors.values[:, None],
                       results.tstats.values[:, None],
                       results.pvalues.values[:, None],
                       results.conf_int()]
    data = []
    for row in param_data:
        txt_row = []
        for i, v in enumerate(row):
            f = _str
            if i == 3:
                f = pval_format
            txt_row.append(f(v))
        data.append(txt_row)
    header = ['Parameter', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']
    table_stubs = list(results.params.index)
    if pad_bottom:
        # Append blank row for spacing
        data.append([''] * 6)
        table_stubs += ['']

    return SimpleTable(data, stubs=table_stubs, txt_fmt=fmt_params,
                       headers=header, title=title)
