"""
Covariance estimation for 2SLS and LIML IV estimators
"""
from __future__ import absolute_import, division, print_function

from numpy import (arange, argsort, asarray, ceil, cos, empty, int64, ones, pi,
                   r_, sin, sum, unique, where, zeros)
from numpy.linalg import inv, pinv

CLUSTER_ERR = """
clusters has the wrong nobs. Expected {0}, got {1}.  Any missing observation
in the regression variables have have been dropped.  When using a clustered
covariance estimator, drop missing data before estimating the model. The model
property `notnull` contains the locations of the observations that have no
missing values."""


def _cov_cluster(z, clusters):
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
    c : ndarray
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


def _cov_kernel(z, w):
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
    c : ndarray
       k by k kernel asymptotic covariance
    """
    k = len(w)
    n = z.shape[0]
    if k > n:
        raise ValueError('Length of w ({0}) is larger than the number '
                         'of elements in z ({1})'.format(k, n))
    s = z.T @ z
    for i in range(1, len(w)):
        op = z[i:].T @ z[:-i]
        s += w[i] * (op + op.T)

    s /= n
    return s


def kernel_weight_bartlett(bw, *args):
    """
    Kernel weights from a Bartlett kernel

    Parameters
    ----------
    bw: int
       Maximum lag to used in kernel

    Returns
    -------
    weights : ndarray
        Weight array  ordered by lag position (maxlag + 1)

    Notes
    -----
    .. math::

       w_i = 1 - i / (m + 1), \, i < m
    """
    return 1 - arange(bw + 1) / (bw + 1)


def kernel_weight_quadratic_spectral(bw, n):
    r"""
    Kernel weights from a quadratic-spectral kernel

    Parameters
    ----------
    bw: {int, float}
       Maximum lag to used in kernel
    n : int
        Positive number of weight to return

    Returns
    -------
    weights : ndarray
        Weight array  ordered by lag position (maxlag + 1)

    Notes
    -----
    Unlike the Barrlett or Parzen kernels, the QS kernel is not truncated at
    a specific lag, and so weights are computed for all available lags in
    the sample.

    .. math::

       z_i & = 6 \pi (i / m) / 5                                \\
       w_0 &  = 1                                                \\
       w_i &  = 3(\sin(z_i)/z_i - cos(z_i))/z_i^ 2, \, i \geq 1

    where m is the bandwidth.
    """

    z = arange(n + 1) / bw
    w = 6 * pi * z / 5
    w[0] = 1
    w[1:] = 3 / w[1:] ** 2 * (sin(w[1:]) / w[1:] - cos(w[1:]))

    return w


def kernel_weight_parzen(bw, *args):
    r"""
    Kernel weights from a Parzen kernel

    Parameters
    ----------
    bw : int
       Maximum lag to used in kernel

    Returns
    -------
    weights : ndarray
        Weight array  ordered by lag position (maxlag + 1)

    Notes
    -----
    .. math::

       z_i & = i / (m+1)                    \\
       w_i &  = 1-6z_i^2+6z_i^3, z \leq 0.5 \\
       w_i &  = 2(1-z_i)^3, z > 0.5
    """
    z = arange(bw + 1) / (bw + 1)
    w = 1 - 6 * z ** 2 + 6 * z ** 3
    w[z > 0.5] = 2 * (1 - z[z > 0.5] ** 3)
    return w


def kernel_optimal_bandwidth(x, kernel='bartlett'):
    """
    Parameters
    x : ndarray
        Array of data to use when computing optimal bandwidth
    kernel : str, optional
        Name of kernel to use.  Supported kernels include:

          * 'bartlett', 'newey-west' : Bartlett's kernel
          * 'parzen', 'gallane' : Parzen's kernel
          * 'qs', 'quadratic-spectral', 'andrews : Quadratic spectral kernel

    Returns
    -------
    m : int
        Optimal bandwidth. Set to nobs - 1 if computed bandwidth is larger.

    Notes
    -----

    .. todo::

      * Explain mathematics involved
      * References

    See Also
    --------
    linearmodels.iv.covariance.kernel_weight_bartlett,
    linearmodels.iv.covariance.kernel_weight_parzen,
    linearmodels.iv.covariance.kernel_weight_quadratic_spectral

    """
    t = x.shape[0]
    x = x.squeeze()
    if kernel in ('bartlett', 'newey-west'):
        q, c = 1, 1.1447
        m_star = int(ceil(4 * (t / 100) ** (2 / 9)))
    elif kernel in ('qs', 'andrews', 'quadratic-spectral'):
        q, c = 2, 1.3221
        m_star = int(ceil(4 * (t / 100) ** (4 / 25)))
    elif kernel in ('gallant', 'parzen'):
        q, c = 2, 2.6614
        m_star = int(ceil(4 * (t / 100) ** (2 / 25)))
    else:
        raise ValueError('Unknown kernel: {0}'.format(kernel))
    sigma = empty(m_star + 1)
    sigma[0] = x.T @ x / t
    for i in range(1, m_star + 1):
        sigma[i] = x[i:].T @ x[:-i] / t
    s0 = sigma[0] + 2 * sigma[1:].sum()
    sq = 2 * sum(sigma[1:] * arange(1, m_star + 1) ** q)
    rate = 1 / (2 * q + 1)
    gamma = c * ((sq / s0) ** 2) ** rate
    m = gamma * t ** rate
    return min(int(ceil(m)), t - 1)


KERNEL_LOOKUP = {'bartlett': kernel_weight_bartlett,
                 'newey-west': kernel_weight_bartlett,
                 'quadratic-spectral': kernel_weight_quadratic_spectral,
                 'qs': kernel_weight_quadratic_spectral,
                 'andrews': kernel_weight_quadratic_spectral,
                 'gallant': kernel_weight_parzen,
                 'parzen': kernel_weight_parzen}


class HomoskedasticCovariance(object):
    r"""
    Covariance estimation for homoskedastic data

    Parameters
    ----------
    x : ndarray
        Model regressors (nobs by nvar)
    y : ndarray
        Series modeled (nobs by 1)
    z : ndarray
        Instruments used for endogenous regressors (nobs by ninstr)
    params : ndarray
        Estimated model parameters (nvar by 1)
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment
    kappa : float, optional
        Value of kappa in k-class estimator

    Notes
    -----
    Covariance is estimated using

    .. math::

        n^{-1} s^2 V^{-1}

    where

    .. math::

      s^2 = n^{-1} \sum_{i=1}^n \hat{\epsilon}_i^2

    If ``debiased`` is true, then :math:`s^2` is scaled by n / (n-k).

    .. math::

      V = n^{-1} X'Z(Z'Z)^{-1}Z'X

    where :math:`X` is the matrix of variables included in the model and
    :math:`Z` is the matrix of instruments, including exogenous regressors.
    """

    def __init__(self, x, y, z, params, debiased=False, kappa=1):
        if not (x.shape[0] == y.shape[0] == z.shape[0]):
            raise ValueError('x, y and z must have the same number of rows')
        if not x.shape[1] == len(params):
            raise ValueError('x and params must have compatible dimensions')

        self.x = x
        self.y = y
        self.z = z
        self.params = params
        self._debiased = debiased
        self.eps = y - x @ params
        self._kappa = kappa
        self._pinvz = pinv(z)
        nobs, nvar = x.shape
        self._scale = nobs / (nobs - nvar) if self._debiased else 1
        self._name = 'Unadjusted Covariance (Homoskedastic)'

    def __str__(self):
        out = self._name
        out += '\nDebiased: {0}'.format(self._debiased)
        if self._kappa != 1:
            out += '\nKappa: {0:0.3f}'.format(self._kappa)
        return out

    def __repr__(self):
        return self.__str__() + '\n' + \
               self.__class__.__name__ + \
               ', id: {0}'.format(hex(id(self)))

    @property
    def s(self):
        """Score covariance estimate"""
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        s2 = eps.T @ eps / nobs
        pinvz = self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        if self._kappa != 1:
            kappa = self._kappa
            xpx = x.T @ x / nobs
            v = (1 - kappa) * xpx + kappa * v

        return self._scale * s2 * v

    @property
    def cov(self):
        """Covariance of estimated parameters"""

        x, z = self.x, self.z
        nobs, nvar = x.shape

        pinvz = self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        if self._kappa != 1:
            kappa = self._kappa
            xpx = x.T @ x / nobs
            v = (1 - kappa) * xpx + kappa * v

        vinv = inv(v)
        c = vinv @ self.s @ vinv / nobs
        return (c + c.T) / 2

    @property
    def s2(self):
        """
        Estimated variance of residuals. Small-sample adjusted if debiased.
        """
        nobs, nvar = self.x.shape
        eps = self.eps

        return self._scale * eps.T @ eps / nobs

    @property
    def debiased(self):
        """Flag indicating if covariance is debiased"""
        return self._debiased

    @property
    def config(self):
        return {'debiased': self.debiased,
                'kappa': self._kappa}


class HeteroskedasticCovariance(HomoskedasticCovariance):
    """
    Covariance estimation for heteroskedastic data

    Parameters
    ----------
    x : ndarray
        Model regressors (nobs by nvar)
    y : ndarray
        Series ,modeled (nobs by 1)
    z : ndarray
        Instruments used for endogenous regressors (nobs by ninstr)
    params : ndarray
        Estimated model parameters (nvar by 1)
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment
    kappa : float, optional
        Value of kappa in k-class estimator

    Notes
    -----
    Covariance is estimated using

    .. math::

        n^{-1} V^{-1} \hat{S} V^{-1}

    where

    .. math::

      \hat{S} = n^{-1} \sum_{i=1}^n \hat{\epsilon}_i^2 \hat{x}_i^{\prime} \hat{x}_i

    where :math:`\hat{\gamma}=(Z'Z)^{-1}(Z'X)` and
    :math:`\hat{x}_i = z_i\hat{\gamma}`. If ``debiased`` is true, then
    :math:`S` is scaled by n / (n-k).

    .. math::

      V = n^{-1} X'Z(Z'Z)^{-1}Z'X

    where :math:`X` is the matrix of variables included in the model and
    :math:`Z` is the matrix of instruments, including exogenous regressors.
    """

    def __init__(self, x, y, z, params, debiased=False, kappa=1):
        super(HeteroskedasticCovariance, self).__init__(x, y, z, params, debiased, kappa)
        self._name = 'Robust Covariance (Heteroskedastic)'

    @property
    def s(self):
        """Heteroskedasticity-robust score covariance estimate"""
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        pinvz = self._pinvz
        xhat_e = z @ (pinvz @ x) * eps
        s = xhat_e.T @ xhat_e / nobs

        return self._scale * s


class KernelCovariance(HomoskedasticCovariance):
    r"""
    Kernel weighted (HAC) covariance estimation

    Parameters
    ----------
    x : ndarray
        Model regressors (nobs by nvar)
    y : ndarray
        Series ,modeled (nobs by 1)
    z : ndarray
        Instruments used for endogenous regressors (nobs by ninstr)
    params : ndarray
        Estimated model parameters (nvar by 1)
    kernel : str
        Kernel name. Supported kernels are:

        * 'bartlett', 'newey-west' - Triangular kernel
        * 'qs', 'quadratic-spectral', 'andrews' - Quadratic spectral kernel
        * 'parzen', 'gallant' - Parzen's kernel;

    bandwidth : {int, None}
        Non-negative bandwidth to use with kernel. If None, automatic
        bandwidth selection is used.
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment
    kappa : float, optional
        Value of kappa in k-class estimator

    Notes
    -----
    Covariance is estimated using

    .. math::

        n^{-1} V^{-1} \hat{S} V^{-1}

    where

    .. math::

      \hat{S}_0 & = n^{-1} \sum_{i=1}^{n} \hat{\epsilon}^2_i \hat{x}_i^{\prime}
           \hat{x}_{i} \\
      \hat{S}_j & = n^{-1} \sum_{i=1}^{n-j}
          \hat{\epsilon}_i\hat{\epsilon}_{i+j} (\hat{x}_i^{\prime}
          \hat{x}_{i+j} + \hat{x}_{i+j}^{\prime} \hat{x}_{i}) \\
      \hat{S}   & = \sum_{i=0}^{bw} K(i, bw) \hat{S}_i

    where :math:`\hat{\gamma}=(Z'Z)^{-1}(Z'X)`,
    :math:`\hat{x}_i = z_i\hat{\gamma}` and :math:`K(i,bw)` is a weight that
    depends on the kernel. If ``debiased`` is true, then :math:`S` is scaled
    by n / (n-k).

    .. math::

      V = n^{-1} X'Z(Z'Z)^{-1}Z'X

    where :math:`X` is the matrix of variables included in the model and
    :math:`Z` is the matrix of instruments, including exogenous regressors.

    See Also
    --------
    linearmodels.iv.covariance.kernel_weight_bartlett,
    linearmodels.iv.covariance.kernel_weight_parzen,
    linearmodels.iv.covariance.kernel_weight_quadratic_spectral

    """

    def __init__(self, x, y, z, params, kernel='bartlett',
                 bandwidth=None, debiased=False, kappa=1):
        super(KernelCovariance, self).__init__(x, y, z, params, debiased, kappa)
        self._kernels = KERNEL_LOOKUP
        self._kernel = kernel
        self._bandwidth = bandwidth
        self._auto_bandwidth = False
        self._name = 'Kernel Covariance (HAC)'

        if kernel not in KERNEL_LOOKUP:
            raise ValueError('Unknown kernel: {0}'.format(kernel))

    def __str__(self):
        out = super(KernelCovariance, self).__str__()
        out += '\nKernel: {0}'.format(self._kernel)
        out += '\nAutomatic Bandwidth: {0}'.format(self._auto_bandwidth)
        if self._bandwidth:
            out += '\nBandwidth: {0}'.format(self._bandwidth)
        return out

    @property
    def s(self):
        """HAC score covariance estimate"""
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape

        pinvz = self._pinvz
        xhat = z @ (pinvz @ x)
        xhat_e = xhat * eps

        kernel = self.config['kernel']
        bw = self.config['bandwidth']
        if bw is None:
            self._auto_bandwidth = True
            from linearmodels.utility import has_constant
            const, loc = has_constant(xhat)
            sel = ones((xhat.shape[1], 1))
            if const:
                sel[loc] = 0
            scores = xhat_e @ sel
            bw = kernel_optimal_bandwidth(scores, kernel)

        self._bandwidth = bw = int(bw)
        w = self._kernels[kernel](bw, nobs - 1)

        s = _cov_kernel(xhat_e, w)

        return self._scale * s

    @property
    def config(self):
        return {'debiased': self.debiased,
                'bandwidth': self._bandwidth,
                'kernel': self._kernel,
                'kappa': self._kappa}


class ClusteredCovariance(HomoskedasticCovariance):
    r"""
    Covariance estimation for clustered data

    Parameters
    ----------
    x : ndarray
        Model regressors (nobs by nvar)
    y : ndarray
        Series ,modeled (nobs by 1)
    z : ndarray
        Instruments used for endogenous regressors (nobs by ninstr)
    params : ndarray
        Estimated model parameters (nvar by 1)
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment
    clusters : ndarray, optional
        Cluster group assignment.  If not provided, uses clusters of 1.
        Either nobs by ncluster where ncluster is 1 or 2.
    kappa : float, optional
        Value of kappa in k-class estimator


    Notes
    -----
    Covariance is estimated using

    .. math::

        n^{-1} V^{-1} \hat{S} V^{-1}

    where

    .. math::

      \hat{S} & = n^{-1} (G/(G-1)) \sum_{g=1}^G \xi_{g}^\prime \xi_{g} \\
      \xi_{g} & = \sum_{i\in\mathcal{G}_g} \hat{\epsilon}_i \hat{x}_i \\

    where :math:`\hat{\gamma}=(Z'Z)^{-1}(Z'X)` and
    :math:`\hat{x}_i = z_i\hat{\gamma}`.  :math:`\mathcal{G}_g` contains the
    indices of elements in cluster g. If ``debiased`` is true, then
    :math:`S` is scaled by g(n - 1) / ((g-1)(n-k)) where g is the number
    of groups..

    .. math::

      V = n^{-1} X'Z(Z'Z)^{-1}Z'X

    where :math:`X` is the matrix of variables included in the model and
    :math:`Z` is the matrix of instruments, including exogenous regressors.
    """

    def __init__(self, x, y, z, params, clusters=None, debiased=False, kappa=1):
        super(ClusteredCovariance, self).__init__(x, y, z, params, debiased, kappa)

        nobs = x.shape[0]
        clusters = arange(nobs) if clusters is None else clusters
        clusters = asarray(clusters).squeeze()
        self._clusters = clusters
        if clusters.ndim == 1:
            self._num_clusters = [len(unique(clusters))]
            self._num_clusters_str = str(self._num_clusters[0])
        else:
            self._num_clusters = [len(unique(clusters[:, 0])), len(unique(clusters[:, 1]))]
            self._num_clusters_str = ', '.join(map(str, self._num_clusters))
        if clusters is not None and clusters.shape[0] != nobs:
            raise ValueError(CLUSTER_ERR.format(nobs, clusters.shape[0]))
        self._name = 'Clustered Covariance (One-Way)'

    def __str__(self):
        out = super(ClusteredCovariance, self).__str__()
        out += '\nNum Clusters: {0}'.format(self._num_clusters_str)
        return out

    @property
    def s(self):
        """Clustered estimator of score covariance"""

        def rescale(s, nc, nobs):
            scale = self._scale * (nc / (nc - 1)) * ((nobs - 1) / nobs)
            return s * scale if self.debiased else s

        x, z, eps = self.x, self.z, self.eps
        pinvz = self._pinvz
        xhat_e = z @ (pinvz @ x) * eps

        nobs, nvar = x.shape
        clusters = self._clusters
        if self._clusters.ndim == 1:
            s = _cov_cluster(xhat_e, clusters)
            s = rescale(s, self._num_clusters[0], nobs)
        else:
            s0 = _cov_cluster(xhat_e, clusters[:, 0].squeeze())
            s0 = rescale(s0, self._num_clusters[0], nobs)

            s1 = _cov_cluster(xhat_e, clusters[:, 1].squeeze())
            s1 = rescale(s1, self._num_clusters[1], nobs)

            c0 = clusters[:, 0] - clusters[:, 0].min() + 1
            c1 = clusters[:, 1] - clusters[:, 1].min() + 1
            c01 = (c0 * (c1.max() + 1) + c1).astype(int64)
            s01 = _cov_cluster(xhat_e, c01.squeeze())
            nc = len(unique(c01))
            s01 = rescale(s01, nc, nobs)

            s = s0 + s1 - s01

        return s

    @property
    def config(self):
        return {'debiased': self.debiased,
                'clusters': self._clusters,
                'kappa': self._kappa}
