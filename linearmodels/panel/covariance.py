import numpy as np
from numpy.linalg import inv
from pandas import DataFrame

from linearmodels.iv.covariance import (CLUSTER_ERR, KERNEL_LOOKUP,
                                        _cov_cluster, _cov_kernel,
                                        kernel_optimal_bandwidth)
from linearmodels.utility import cached_property

__all__ = ['HomoskedasticCovariance', 'HeteroskedasticCovariance',
           'ClusteredCovariance', 'DriscollKraay', 'CovarianceManager']


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
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased

    Notes
    -----
    The estimator of the covariance is

    .. math:: s^2\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = X'X

    and

    .. math::

        s^2 = (n-df)^{-1} \hat{\epsilon}'\hat{\epsilon}

    where df is ``extra_df`` and n-df is replace by n-df-k if ``debiased`` is
    ``True``.
    """

    def __init__(self, y, x, params, entity_ids, time_ids, *, debiased=False, extra_df=0):
        self._y = y
        self._x = x
        self._params = params
        self._entity_ids = entity_ids
        self._time_ids = time_ids
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
        """Covariance estimator name"""
        return self._name

    @property
    def eps(self):
        """Model residuals"""
        return self._y - self._x @ self._params

    @property
    def s2(self):
        """Error variance"""
        eps = self.eps
        return self._scale * float(eps.T @ eps) / self._nobs

    @cached_property
    def cov(self):
        """Estimated covariance"""
        x = self._x
        out = self.s2 * inv(x.T @ x)
        return (out + out.T) / 2

    def deferred_cov(self):
        """Covariance calculation deferred until executed"""
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
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased

    Notes
    -----
    The estimator of the covariance is

    .. math::

        n^{-1}\hat{\Sigma}_{xx}^{-1}\hat{S}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = n^{-1}X'X

    and

    .. math::

        \hat{S} = (n-df)^{-1} \sum_{i=1}^n \hat{\epsilon}_i^2 x_i'x_i

    where df is ``extra_df`` and n-df is replace by n-df-k if ``debiased`` is
    ``True``.
    """

    def __init__(self, y, x, params, entity_ids, time_ids, *, debiased=False, extra_df=0):
        super(HeteroskedasticCovariance, self).__init__(y, x, params, entity_ids, time_ids,
                                                        debiased=debiased, extra_df=extra_df)
        self._name = 'Robust'

    @cached_property
    def cov(self):
        """Estimated covariance"""
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
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased
    clusters : ndarray, optional
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

        \hat{\Sigma}_{xx}^{-1}\hat{S}_{\mathcal{G}}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = X'X

    and :math:`\hat{S}_{\mathcal{G}}` is a one- or two-way cluster covariance
    of the scores.  Two-way clustering is implemented by summing up the two
    one-way cluster covariances and then subtracting the one-way clustering
    covariance computed using the group formed from the intersection of the
    two groups.

    Two small sample adjustment are available.  ``debias=True`` will account
    for regressors in the main model. ``group_debias=True`` will provide a
    small sample adjustment for the number of clusters of the form

    .. math ::

      (g / (g- 1)) ((n - 1) / n)

    where g is the number of distinct groups and n is the number of
    observations.
    """

    def __init__(self, y, x, params, entity_ids, time_ids, *, debiased=False, extra_df=0,
                 clusters=None,
                 group_debias=False):
        super(ClusteredCovariance, self).__init__(y, x, params, entity_ids, time_ids,
                                                  debiased=debiased, extra_df=extra_df)
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
        return (ngroups / (ngroups - 1)) * ((n - 1) / n)

    @cached_property
    def cov(self):
        """Estimated covariance"""
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


class DriscollKraay(HomoskedasticCovariance):
    r"""
    Driscoll-Kraay heteroskedasticity-autocorrelation robust covariance estimation

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased
    kernel : str, options
        Name of one of the supported kernels
    bandwidth : int, optional
        Non-negative integer to use as bandwidth.  If not provided a rule-of-
        thumb value is used.

    Notes
    -----
    Supported kernels:

    * 'bartlett', 'newey-west' - Bartlett's kernel
    * 'quadratic-spectral', 'qs', 'andrews' - Quadratic-Spectral Kernel
    * 'parzen', 'gallant' - Parzen kernel

    Bandwidth is set to the common value for the Bartlett kernel if not
    provided.

    The estimator of the covariance is

    .. math::

        n^{-1}\hat{\Sigma}_{xx}^{-1}\hat{S}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = n^{-1}X'X

    and

    .. math::
      \xi_t & = \sum_{i=1}^{n_t} \epsilon_i x_{i} \\
      \hat{S}_0 & = \sum_{i=1}^{t} \xi'_t \xi_t \\
      \hat{S}_j & = \sum_{i=1}^{t-j} \xi'_t \xi_{t+j} + \xi'_{t+j} \xi_t  \\
      \hat{S}   & = (n-df)^{-1} \sum_{j=0}^{bw} K(j, bw) \hat{S}_j

    where df is ``extra_df`` and n-df is replace by n-df-k if ``debiased`` is
    ``True``. :math:`K(i, bw)` is the kernel weighting function.
    """

    # TODO: Test

    def __init__(self, y, x, params, entity_ids, time_ids, *, debiased=False, extra_df=0,
                 kernel='newey-west', bandwidth=None):
        super(DriscollKraay, self).__init__(y, x, params, entity_ids, time_ids,
                                            debiased=debiased, extra_df=extra_df)
        self._name = 'Driscoll-Kraay'
        self._kernel = kernel
        self._bandwidth = bandwidth

    @cached_property
    def cov(self):
        """Estimated covariance"""
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)
        eps = self.eps

        xe = x * eps
        xe = DataFrame(xe, index=self._time_ids.squeeze())
        xe = xe.groupby(level=0).sum()
        xe.sort_index(inplace=True)
        xe_nobs = xe.shape[0]
        bw = self._bandwidth
        if self._bandwidth is None:
            bw = int(np.floor(4 * (xe_nobs / 100) ** (2 / 9)))
        w = KERNEL_LOOKUP[self._kernel](bw, xe_nobs - 1)
        xeex = _cov_kernel(xe.values, w) * (xe_nobs / nobs)
        xeex *= self._scale

        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2


class ACCovariance(HomoskedasticCovariance):
    r"""
    Autocorrelation robust covariance estimation

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased
    kernel : str, options
        Name of one of the supported kernels
    bandwidth : int, optional
        Non-negative integer to use as bandwidth.  If not provided a rule-of-
        thumb value is used.

    Notes
    -----
    Estimator is robust to autocorrelation but no cross-sectional correlation.

    Supported kernels:

    * 'bartlett', 'newey-west' - Bartlett's kernel
    * 'quadratic-spectral', 'qs', 'andrews' - Quadratic-Spectral Kernel
    * 'parzen', 'gallant' - Parzen kernel

    Bandwidth is set to the common value for the Bartlett kernel if not
    provided.

    The estimator of the covariance is

    .. math::

        n^{-1}\hat{\Sigma}_{xx}^{-1}\hat{S}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = n^{-1}X'X

    and

    .. math::

      \xi_t & = \epsilon_{it} x_{it} \\
      \hat{S} & = n / (N(n-df))  \sum_{i=1}^N S_i \\
      \hat{S}_i & = \sum_{j=0}^{bw} K(j, bw) \hat{S}_{ij} \\
      \hat{S}_{i0} & = \sum_{t=1}^{T} \xi'_{it} \xi_{it} \\
      \hat{S}_{ij} & = \sum_{t=1}^{T-j} \xi'_{it} \xi_{it+j} + \xi'_{it+j} \xi_{it}


    where df is ``extra_df`` and n-df is replace by n-df-k if ``debiased`` is
    ``True``. :math:`K(i, bw)` is the kernel weighting function.
    """

    # TODO: Docstring

    def __init__(self, y, x, params, entity_ids, time_ids, *, debiased=False, extra_df=0,
                 kernel='newey-west', bandwidth=None):
        super(ACCovariance, self).__init__(y, x, params, entity_ids, time_ids,
                                           debiased=debiased, extra_df=extra_df)
        self._name = 'Autocorrelation Rob. Cov.'
        self._kernel = kernel
        self._bandwidth = bandwidth

    def _single_cov(self, xe, bw):
        nobs = xe.shape[0]
        w = KERNEL_LOOKUP[self._kernel](bw, nobs - 1)
        return _cov_kernel(xe, w)

    @cached_property
    def cov(self):
        """Estimated covariance"""
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)
        eps = self.eps

        time_ids = np.unique(self._time_ids.squeeze())
        nobs = len(time_ids)
        bw = self._bandwidth
        if self._bandwidth is None:
            bw = int(np.floor(4 * (nobs / 100) ** (2 / 9)))

        xe = x * eps
        index = [self._entity_ids.squeeze(), self._time_ids.squeeze()]
        xe = DataFrame(xe, index=index)
        xe = xe.sort_index(level=[0, 1])

        entities = xe.index.levels[0]
        nentity = len(entities)
        xeex = np.zeros((xe.shape[1], xe.shape[1]))
        for entity in entities:
            _xe = xe.loc[entity].values
            _bw = max(len(_xe) - 1, bw)
            xeex += self._single_cov(_xe, _bw)
        xeex /= nentity
        xeex *= self._scale

        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2


class CovarianceManager(object):
    COVARIANCE_ESTIMATORS = {'unadjusted': HomoskedasticCovariance,
                             'conventional': HomoskedasticCovariance,
                             'homoskedastic': HomoskedasticCovariance,
                             'robust': HeteroskedasticCovariance,
                             'heteroskedastic': HeteroskedasticCovariance,
                             'clustered': ClusteredCovariance,
                             'driscoll-kraay': DriscollKraay,
                             'dk': DriscollKraay,
                             'kernel': DriscollKraay,
                             'ac': ACCovariance,
                             'autocorrelated': ACCovariance}

    def __init__(self, estimator, *cov_estimators):
        self._estimator = estimator
        self._supported = cov_estimators

    def __getitem__(self, item):
        if item not in self.COVARIANCE_ESTIMATORS:
            raise KeyError('Unknown covariance estimator type.')
        cov_est = self.COVARIANCE_ESTIMATORS[item]
        if cov_est not in self._supported:
            raise ValueError('Requested covariance estimator is not supported '
                             'for the {0}.'.format(self._estimator))
        return cov_est


class FamaMacBethCovariance(HomoskedasticCovariance):
    """
    Heteroskedasticty robust covariance estimator for Fama-MacBeth

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        (variables by 1) array of estimated model parameters
    all_params : ndarray
        (nobs by variables) array of all estimated model parameters
    debiased : bool, optional
        Flag indicating whether to debias the estimator

    Notes
    -----
    Covariance is the sample covariance of the set of all estimated
    parameters.
    """

    def __init__(self, y, x, params, all_params, *, debiased=False):
        super(FamaMacBethCovariance, self).__init__(y, x, params, None, None, debiased=debiased)
        self._all_params = all_params
        self._name = 'Fama-MacBeth Std Cov'

    @cached_property
    def cov(self):
        """Estimated covariance"""
        e = self._all_params - self._params.T
        e = e[np.all(np.isfinite(e), 1)]
        nobs = e.shape[0]
        cov = (e.T @ e / nobs)
        return cov / (nobs - int(bool(self._debiased)))


class FamaMacBethKernelCovariance(FamaMacBethCovariance):
    """
    HAC estimator for Fama-MacBeth estimator

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        (variables by 1) array of estimated model parameters
    all_params : ndarray
        (nobs by variables) array of all estimated model parameters
    debiased : bool, optional
        Flag indicating whether to debias the estimator
    kernel : str, options
        Name of one of the supported kernels
    bandwidth : int, optional
        Non-negative integer to use as bandwidth.  If not provided a rule-of-
        thumb value is used.

    Notes
    -----
    Covariance is a Kernel covariance of all estimated parameters.
    """

    def __init__(self, y, x, params, all_params, *, debiased=False, kernel='newey-west',
                 bandwidth=None):
        super(FamaMacBethKernelCovariance, self).__init__(y, x, params, all_params,
                                                          debiased=debiased)
        self._name = 'Fama-MacBeth Kernel Cov'
        self._bandwidth = bandwidth
        self._kernel = kernel

    @cached_property
    def bandwidth(self):
        """Estimator bandwidth"""
        if self._bandwidth is None:
            e = self._all_params - self._params.T
            e = e[np.all(np.isfinite(e), 1)]
            stde = np.sum(e / e.std(0)[None, :], 1)
            self._bandwidth = kernel_optimal_bandwidth(stde, self._kernel)
        return self._bandwidth

    @cached_property
    def cov(self):
        """Estimated covariance"""
        e = self._all_params - self._params.T
        e = e[np.all(np.isfinite(e), 1)]
        nobs = e.shape[0]

        bw = self.bandwidth
        w = KERNEL_LOOKUP[self._kernel](bw, nobs - 1)
        cov = _cov_kernel(e, w)
        return cov / (nobs - int(bool(self._debiased)))
