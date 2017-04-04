import numpy as np
from numpy.linalg import inv

from linearmodels.iv.covariance import _cov_cluster
from linearmodels.utility import cached_property


class HomoskedasticCovariance(object):
    r"""
    Homoskedastic covariance estimation

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    df_resid : int
        Residual degree of freedom to use when normalizing covariance

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

    def __init__(self, y, x, params, df_resid):
        self._y = y
        self._x = x
        self._params = params
        self._df_resid = df_resid

    @property
    def eps(self):
        return self._y - self._x @ self._params

    @property
    def s2(self):
        eps = self.eps
        return float(eps.T @ eps) / self._df_resid

    @cached_property
    def cov(self):
        x = self._x
        return self.s2 * inv(x.T @ x)

    def deferred_cov(self):
        return self.cov


class HeteroskedasticCovariance(HomoskedasticCovariance):
    r"""
    Covariance estimation using White estimator

    Parameters
    ----------
    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    df_resid : int
        Residual degree of freedom to use when normalizing covariance

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

    def __init__(self, y, x, params, df_resid):
        super(HeteroskedasticCovariance, self).__init__(y, x, params, df_resid)

    @cached_property
    def cov(self):
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)
        eps = self.eps
        xe = x * eps
        xeex = xe.T @ xe / self._df_resid
        out = (xpxi @ xeex @ xpxi) / nobs
        out = (out + out.T) / 2
        return out


class OneWayClusteredCovariance(HomoskedasticCovariance):
    r"""
    One-way clustered (Rogers) covariance estimation

    Parameters
    ----------
    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    df_resid : int
        Residual degree of freedom to use when normalizing covariance
    cluster : ndarray, optional
        (entity x time) by 1 stacked array of cluster group

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

    def __init__(self, y, x, params, df_resid, clusters=None):
        super(OneWayClusteredCovariance, self).__init__(y, x, params, df_resid)
        if clusters is None:
            clusters = np.arange(self._x.shape[0])
        self._clusters = clusters.squeeze()

    @cached_property
    def cov(self):
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)

        eps = self.eps
        xe = x * eps
        xeex = _cov_cluster(xe, self._clusters) * (nobs / self._df_resid)
        out = (xpxi @ xeex @ xpxi) / nobs
        out = (out + out.T) / 2
        return out


class ClusteredCovariance(HomoskedasticCovariance):
    r"""
    One-way (Rogers) or two-way clustered covariance estimation

    Parameters
    ----------
    Parameters
    ----------
    y : ndarray
        nobs by 1 stacked array of dependent
    x : ndarray
        nobs by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    df_resid : int
        Residual degree of freedom to use when normalizing covariance
    cluster : ndarray, optional
        nobs by 1 or nobs by 2 array of cluster group ids
    group_adj : bool, optional
        Flag indicating whether to apply small-number of groups adjustment

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

        * Complete -- math is wrong
        * Small sample adjustments
    """

    def __init__(self, y, x, params, df_resid, clusters=None, group_adj=False):
        super(ClusteredCovariance, self).__init__(y, x, params, df_resid)
        if clusters is None:
            clusters = np.arange(self._x.shape[0])
        self._clusters = clusters.squeeze()
        self._group_adj = group_adj
        dim1 = 1 if self._clusters.ndim == 1 else self._clusters.shape[1]
        if self._clusters.ndim > 2 or dim1 > 2:
            raise ValueError('Onle 1 or 2-way clustering supported.')


    def _calc_group_adj(self, clusters):
        ngroups = len(np.unique(clusters))
        return ngroups / (ngroups - 1)

    @cached_property
    def cov(self):
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)

        eps = self.eps
        xe = x * eps
        if self._clusters.ndim == 1:
            xeex = _cov_cluster(xe, self._clusters)
            if self._group_adj:
                xeex *= self._calc_group_adj(self._clusters)

        else:
            clusters0 = self._clusters[:, 0]
            clusters1 = self._clusters[:, 1]
            xeex0 = _cov_cluster(xe, clusters0)
            xeex1 = _cov_cluster(xe, clusters1)

            sort_keys = np.lexsort(self._clusters.T)
            locs = np.arange(self._clusters.shape[0])
            lex_sorted = self._clusters[sort_keys]
            sorted_locs = locs[sort_keys]
            diff = np.any(lex_sorted[1:] != lex_sorted[:-1], 1)
            clusters01 = np.cumsum(np.r_[0, diff])
            resort_locs = np.argsort(sorted_locs)
            clusters01 = clusters01[resort_locs]
            xeex01 = _cov_cluster(xe, clusters01)

            if self._group_adj:
                xeex0 *= self._calc_group_adj(clusters0)
                xeex1 *= self._calc_group_adj(clusters1)
                xeex01 *= self._calc_group_adj(clusters01)

            xeex = xeex0 + xeex1 - xeex01

        xeex *= (nobs / self._df_resid)

        out = (xpxi @ xeex @ xpxi) / nobs
        out = (out + out.T) / 2
        return out
