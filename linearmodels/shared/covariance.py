"""
Core covariance estimators that the models share.
"""

from numpy import (
    any as npany,
    arange,
    argsort,
    cumsum,
    lexsort,
    r_,
    unique,
    where,
    zeros,
)
from pandas import isnull

import linearmodels.typing.data


def group_debias_coefficient(clusters: linearmodels.typing.data.IntArray) -> float:
    r"""
    Compute the group debiasing scale.

    Parameters
    ----------
    clusters : ndarray
        One-dimensional array containing cluster group membership.

    Returns
    -------
    float
        The scale to debias.

    Notes
    -----
    The debiasing coefficient is defined

    .. math::

       \frac{g}{g-1}\frac{n-1}{n}

    where g is the number of groups and n is the sample size.
    """
    n = clusters.shape[0]
    ngroups = unique(clusters).shape[0]
    return (ngroups / (ngroups - 1)) * ((n - 1) / n)


def cluster_union(
    clusters: linearmodels.typing.data.IntArray,
) -> linearmodels.typing.data.IntArray:
    """
    Compute a set of clusters that is nested within 2 clusters.

    Parameters
    ----------
    clusters : ndarray
        An array with shape (nobs, 2) of integer values of cluster group
        membership.

    Returns
    -------
    ndarray
        An array with shape (nobs,) of integer cluster group memberships. Two
        observations are in the same group if they are in the same group of both
        columns of ``clusters``.
    """
    sort_keys = lexsort(clusters.T)
    locs = arange(clusters.shape[0])
    lex_sorted = clusters[sort_keys]
    sorted_locs = locs[sort_keys]
    diff = npany(lex_sorted[1:] != lex_sorted[:-1], 1)
    union = cumsum(r_[0, diff])
    resort_locs = argsort(sorted_locs)
    return union[resort_locs]


def cov_cluster(
    z: linearmodels.typing.data.Float64Array,
    clusters: linearmodels.typing.data.AnyArray,
) -> linearmodels.typing.data.Float64Array:
    """
    Core cluster covariance estimator.

    Parameters
    ----------
    z : ndarray
        Mean zero data with shape (n, k).
    clusters : ndarray
        Array with shape (n,) that contains the cluster that each observation
        belongs to.

    Returns
    -------
    ndarray
        Cluster asymptotic covariance with shape (k, k).

    Raises
    ------
    ValueError
        If clusters contains missing values, since the observations they belong to
        cannot be assigned to a cluster, or has fewer than two groups.
    """
    if isnull(clusters).any():
        raise ValueError("clusters cannot contain missing values.")
    num_clusters = len(unique(clusters))
    if num_clusters < 2:
        raise ValueError(
            "A clustered covariance requires at least two clusters, but clusters "
            "has one group."
        )

    sort_args = argsort(clusters)
    clusters = clusters[sort_args]
    locs = where(r_[True, clusters[:-1] != clusters[1:], True])[0]
    z = z[sort_args]
    n, k = z.shape
    s = zeros((k, k))

    for i in range(num_clusters):
        st, en = locs[i], locs[i + 1]
        z_bar = z[st:en].sum(axis=0)[:, None]
        s += z_bar @ z_bar.T

    s /= n
    return s


def cov_kernel(
    z: linearmodels.typing.data.Float64Array, w: linearmodels.typing.data.Float64Array
) -> linearmodels.typing.data.Float64Array:
    """
    Core kernel covariance estimator.

    Parameters
    ----------
    z : ndarray
        Mean zero data with shape (n, k).
    w : ndarray
        Kernel weights with shape (m,) for lags 0, 1, ..., m - 1. The weight of
        lag 0 is not used, since the variance is always included with a weight
        of 1.

    Returns
    -------
    ndarray
        Kernel asymptotic covariance with shape (k, k).

    Raises
    ------
    ValueError
        If the number of weights is larger than the number of observations.
    """
    k = len(w)
    n = z.shape[0]
    if k > n:
        raise ValueError(
            "Length of w ({}) is larger than the number "
            "of elements in z ({})".format(k, n)
        )
    s = z.T @ z
    for i in range(1, len(w)):
        op = z[i:].T @ z[:-i]
        s += w[i] * (op + op.T)

    s /= n
    return s
