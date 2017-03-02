from collections import OrderedDict

import numpy as np
from numpy import diag, sqrt
from numpy.linalg import matrix_rank, eigh
from scipy.stats import chi2, f

class AttrDict(dict):
    """
    Dictionary-like object that exposes keys as attributes 
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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
    """
    if np.any(np.all(x == 1, axis=0)):
        return True

    if np.any((np.ptp(x, axis=0) == 0) & ~np.all(x == 0, axis=0)):
        return True

    n = x.shape[0]
    aug_rank = matrix_rank(np.c_[np.ones((n, 1)), x])
    rank = matrix_rank(x)
    return aug_rank == rank


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
    def __init__(self, stat, null, df, df_denom=None):
        self._stat = stat
        self._null = null
        self.df = df
        self.df_denom = df_denom
        if df_denom is None:
            self.dist = chi2(df)
            self.dist_name = 'chi2({0})'.format(df)
        else:
            self.dist = f(df, df_denom)
            self.dist_name = 'f({0},{1})'.format(df, df_denom)

    @property
    def stat(self):
        return self._stat

    @property
    def pval(self):
        return 1 - self.dist.cdf(self.stat)

    @property
    def critical_values(self):
        return OrderedDict(zip(['10%', '5%', '1%'],
                               self.dist.ppf([.9, .95, .99])))

    @property
    def null(self):
        return self._null

    def __str__(self):
        msg = 'WaldTestStatistic(H0: {null}, stat={stat}, pval={pval:0.3f}, dist={dist})'
        return msg.format(null=self.null, stat=self.stat,
                          pval=self.pval, dist=self.dist_name)

    def __repr__(self):
        return self.__str__() + '\nid={id}'.format(id=hex(id(self)))
