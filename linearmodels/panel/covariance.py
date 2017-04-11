import numpy as np
from numpy.linalg import inv

from linearmodels.iv.covariance import _cov_cluster, CLUSTER_ERR
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
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estiamtors are always
        adjusted for extra_df irrespective of the setting of debiased

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

    def __init__(self, y, x, params, *, debiased=False, extra_df=0):
        self._y = y
        self._x = x
        self._params = params
        self._debiased = debiased
        self._extra_df = extra_df
        self._nobs, self._nvar = x.shape
        self._nobs_eff = self._nobs - extra_df
        if debiased:
            self._nobs_eff -= self._nvar
        self._scale = self._nobs / self._nobs_eff
        self._name = 'Unadjusted'

    @property
    def name(self):
        return self._name

    @property
    def eps(self):
        return self._y - self._x @ self._params

    @property
    def s2(self):
        eps = self.eps
        return self._scale * float(eps.T @ eps) / self._nobs

    @cached_property
    def cov(self):
        x = self._x
        out = self.s2 * inv(x.T @ x)
        return (out + out.T) / 2

    def deferred_cov(self):
        return self.cov


class HeteroskedasticCovariance(HomoskedasticCovariance):
    r"""
    Covariance estimation using White estimator

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estiamtors are always
        adjusted for extra_df irrespective of the setting of debiased

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

    def __init__(self, y, x, params, *, debiased=False, extra_df=0):
        super(HeteroskedasticCovariance, self).__init__(y, x, params, debiased=debiased,
                                                        extra_df=extra_df)
        self._name = 'Robust'

    @cached_property
    def cov(self):
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)
        eps = self.eps
        xe = x * eps
        xeex = self._scale * xe.T @ xe / nobs

        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2


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
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estiamtors are always
        adjusted for extra_df irrespective of the setting of debiased
    cluster : ndarray, optional
        nobs by 1 or nobs by 2 array of cluster group ids
    group_debias : bool, optional
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

    def __init__(self, y, x, params, *, debiased=False, extra_df=0, clusters=None,
                 group_debias=False):
        super(ClusteredCovariance, self).__init__(y, x, params,
                                                  debiased=debiased,
                                                  extra_df=extra_df)
        if clusters is None:
            clusters = np.arange(self._x.shape[0])
        clusters = np.asarray(clusters).squeeze()
        self._group_debias = group_debias
        dim1 = 1 if clusters.ndim == 1 else clusters.shape[1]
        if clusters.ndim > 2 or dim1 > 2:
            raise ValueError('Only 1 or 2-way clustering supported.')
        nobs = y.shape[0]
        if clusters.shape[0] != nobs:
            raise ValueError(CLUSTER_ERR.format(nobs, clusters.shape[0]))
        self._clusters = clusters
        self._name = 'Clustered'

    def _calc_group_debias(self, clusters):
        n = clusters.shape[0]
        ngroups = np.unique(clusters).shape[0]
        return (ngroups / (ngroups - 1)) * ((n-1) / n)

    @cached_property
    def cov(self):
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)

        eps = self.eps
        xe = x * eps
        if self._clusters.ndim == 1:
            xeex = _cov_cluster(xe, self._clusters)
            if self._group_debias:
                xeex *= self._calc_group_debias(self._clusters)

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

            if self._group_debias:
                xeex0 *= self._calc_group_debias(clusters0)
                xeex1 *= self._calc_group_debias(clusters1)
                xeex01 *= self._calc_group_debias(clusters01)

            xeex = xeex0 + xeex1 - xeex01

        xeex *= self._scale
        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2
