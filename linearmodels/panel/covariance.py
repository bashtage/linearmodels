import numpy as np


def homoskedastic_covariance(x, epsilon, *, debiased=False):
    r"""
    Homoskedastic covariance estimation

    Parameters
    ----------
    x : ndarray
        (entity x time) by variables stacked array of regressors
    epsilon : ndarray
        (entity x time) by 1 stacked array of errors
    debiased : bool
        Flag indicating whether to dias adjust

    Returns
    -------
    cov : array
        Estimated parameter covariance

    Notes
    -----
    The estimator of the covariance is

    .. math:: s^2\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T x_{it}x_{it}^{\prime}

    and

    .. math::

        s^2 = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T \hat{\epsilon}_{it}^2

    where NT is replace by NT-k if ``debiased`` is ``True``.

    """
    nt, k = x.shape
    xpx = x.T @ x / nt
    xpxi = np.linalg.inv(xpx)
    scale = nt - k if debiased else nt
    s2 = epsilon.T @ epsilon / scale

    return s2 * (xpxi + xpxi.T) / 2


def heteroskedastic_covariance(x, epsilon, *, debiased=False):
    r"""
    Covariance estimation using White estimator

    Parameters
    ----------
    x : ndarray
        (entity x time) by variables stacked array of regressors
    epsilon : ndarray
        (entity x time) by 1 stacked array of errors
    debiased : bool
        Flag indicating whether to dias adjust

    Returns
    -------
    cov : array
        Estimated parameter covariance

    Notes
    -----
    The estimator of the covariance is

    .. math::

        \hat{\Sigma}_{xx}^{-1}\widehat{Cov}(x_{it}\epsilon_{it})\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T x_{it}x_{it}^{\prime}

    and

    .. math::

        \widehat{Cov}(x_it\epsilon_{it}) = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T
                                           \hat{\epsilon}_{it}^2 x_{it}x_{it}^{\prime}

    where NT is replace by NT-k if ``debiased`` is ``True``.
    """
    nt, k = x.shape
    xpx = x.T @ x / nt
    xpxi = np.linalg.inv(xpx)

    xe = x * epsilon
    scale = nt - k if debiased else nt
    xeex = xe.T @ xe / scale

    cov = xpxi @ xeex @ xpxi

    return (cov + cov.T) / 2


def oneway_clustered_covariance(x, epsilon, cluster, *, debiased=False):
    r"""
    One-way clustered (Rogers) covariance estimation

    Parameters
    ----------
    x : ndarray
        (entity x time) by variables stacked array of regressors
    epsilon : ndarray
        (entity x time) by 1 stacked array of errors
    cluster : ndarray
        (entity x time) by 1 stacked array of cluster group
    debiased : bool
        Flag indicating whether to dias adjust

    Returns
    -------
    cov : array
        Estimated parameter covariance

    Notes
    -----
    The estimator of the covariance is

    .. math::

        \hat{\Sigma}_{xx}^{-1}\widehat{Cov}(x_{it}\epsilon_{it})\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T x_{it}x_{it}^{\prime}

    and

    .. math::

        \widehat{Cov}(x_it\epsilon_{it}) = (NT)^{-1}\sum_{j=1}^G xe_j^{\prime}xe_j

    where ...

    .. todo::

        * Complete docstring
        * Small sample adjustments

    """
    nt, k = x.shape
    xpx = x.T @ x / nt
    xpxi = np.linalg.inv(xpx)

    ind = np.argsort(cluster.flat)
    x = x[ind]
    epsilon = epsilon[ind]
    cluster = cluster[ind]
    locs = np.where(np.r_[True, cluster.flat[:-1] != cluster.flat[1:], True].flat)[0]
    xeex = np.zeros((k, k))
    for i in range(locs.shape[0] - 1):
        st, en = locs[i], locs[i + 1]
        xe = x[st:en] * epsilon[st:en]
        xeex += xe.T @ xe / (en - st)

    cov = xpxi @ xeex @ xpxi

    return (cov + cov.T) / 2
