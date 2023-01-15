from __future__ import annotations

from itertools import combinations

from numpy import (
    any as npany,
    arange,
    argsort,
    cumsum,
    diag,
    lexsort,
    maximum,
    r_,
    unique,
    where,
    zeros,
)
from numpy.linalg import eig

from linearmodels.shared.exceptions import VCOVWarning
from linearmodels.typing import AnyArray, Float64Array, IntArray


def group_debias_coefficient(clusters: IntArray) -> float:
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

       `\frac{g}{g-1}\frac{n-1}{n}`

    where g is the number of groups and n is the sample size.
    """
    n = clusters.shape[0]
    ngroups = unique(clusters).shape[0]
    return (ngroups / (ngroups - 1)) * ((n - 1) / n)


def cluster_union(clusters: IntArray) -> IntArray:
    """
    Compute a set of clusters that is nested within 2 clusters

    Parameters
    ----------
    clusters : ndarray
        A nobs by 2 array of integer values of cluster group membership.

    Returns
    -------
    ndarray
        A nobs array of integer cluster group memberships
    """
    sort_keys = lexsort(clusters.T)
    locs = arange(clusters.shape[0])
    lex_sorted = clusters[sort_keys]
    sorted_locs = locs[sort_keys]
    diff = npany(lex_sorted[1:] != lex_sorted[:-1], 1)
    union = cumsum(r_[0, diff])
    resort_locs = argsort(sorted_locs)
    return union[resort_locs]


def cov_cluster(z: Float64Array, clusters: AnyArray) -> Float64Array:
    """
    Core cluster covariance estimator

    Parameters
    ----------
    z : ndarray
        n by k mean zero data array
    clusters : ndarray
        n by 1 array

    Returns
    -------
    ndarray
       k by k cluster asymptotic covariance
    """
    num_clusters = len(unique(clusters))

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


def cov_kernel(z: Float64Array, w: Float64Array) -> Float64Array:
    """
    Core kernel covariance estimator

    Parameters
    ----------
    z : ndarray
        n by k mean zero data array
    w : ndarray
        m by 1

    Returns
    -------
    ndarray
       k by k kernel asymptotic covariance
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


def multi_way_cluster_iter(
    clusters: Float64Array, xe: Float64Array, group_debias: bool = True
):
    clus_adj = 1

    xeex = {}

    if clusters.ndim == 1:
        clusters = clusters[:, None]

    n_cols = arange(clusters.shape[1])

    for i in range(1, clusters.shape[1] + 1):

        for subset_cols in combinations(n_cols, i):

            sign = 1 if len(subset_cols) % 2 else -1

            if len(subset_cols) == 1:

                if group_debias:
                    clus_adj = group_debias_coefficient(
                        clusters=clusters[:, subset_cols].flatten(),
                    )

                xeex[subset_cols] = (
                    cov_cluster(xe, clusters[:, subset_cols].flatten())
                    * clus_adj
                    * sign
                )

            else:
                clusu = cluster_union(clusters[:, subset_cols])

                if group_debias:
                    clus_adj = group_debias_coefficient(
                        clusters=clusu,
                    )

                xeex[subset_cols] = cov_cluster(xe, clusu) * clus_adj * sign

    return sum(xeex.values())


def cgm_vcov_fix(vcov: Float64Array, toll: float = 1e-10):
    from warnings import warn

    if npany(diag(vcov) < 0):
        evalues, evectors = eig(vcov)
        vcov = (evectors @ diag(maximum(toll, evalues))) @ evectors.T

        warn(
            "Non-positive semi-definite VCOV matrix; adjusted "
            "a la Cameron, Gelbach & Miller 2011",
            VCOVWarning,
        )

    return vcov
