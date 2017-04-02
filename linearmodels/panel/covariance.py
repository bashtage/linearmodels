import numpy as np
from numpy.linalg import inv

from linearmodels.iv.covariance import _cov_cluster


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

    @property
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

    @property
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

    @property
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
