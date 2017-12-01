"""
Covariance and weight estimation for GMM IV estimators
"""
from numpy import array, empty, repeat, sqrt

from linearmodels.iv.covariance import kernel_optimal_bandwidth
from linearmodels.asset_pricing.covariance import _HACMixin
from linearmodels.system._utility import blocked_inner_prod
from linearmodels.utility import AttrDict


class HomoskedasticWeightMatrix(object):
    r"""
    Homoskedastic (unadjusted) weight estimation

    Parameters
    ----------
    center : bool, optional
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix.
    debiased : bool, optional
        Flag indicating whether to use small-sample adjustments

    Notes
    -----
    The weight matrix estimator is

    .. math::

      Z'(\Sigma \otimes I_N)Z

    where :math:`Z` is a block diagonal matrix containing both the exogenous
    regressors and instruments and :math:`\Sigma` is the covariance of the
    model residuals.

    ``center`` has no effect on this estimator since it is always centered.
    """

    def __init__(self, center=False, debiased=False):
        self._center = center
        self._debiased = debiased
        self._bandwidth = 0
        self._name = 'Homoskedastic (Unadjusted) Weighting'

    def __str__(self):
        out = self._name
        extra = []
        for key in self._str_extra:
            extra.append(': '.join([key, str(self._str_extra[key])]))
        if extra:
            out += ' (' + ', '.join(extra) + ')'
        return out

    def __repr__(self):
        return self.__str__() + ', id: {0}'.format(hex(id(self)))

    @property
    def _str_extra(self):
        return AttrDict(Debiased=self._debiased, Center=self._center)

    def sigma(self, eps, x):
        nobs = eps.shape[0]
        eps = eps - eps.mean(0)
        sigma = eps.T @ eps / nobs
        scale = 1.0
        if self._debiased:
            k = array(list(map(lambda a: a.shape[1], x)))[:, None]
            k = sqrt(k)
            scale = nobs / (nobs - k @ k.T)
        sigma *= scale

        return sigma

    def weight_matrix(self, x, z, eps, sigma):
        """
        Parameters
        ----------
        x : ndarray
            List of containing model regressors for each equation in the system
        z : ndarray
            List of containing instruments for each equation in the system
        eps : ndarray
            Model errors (nobs by neqn)
        sigma : ndarray
            Fixed covariance of model errors

        Returns
        -------
        weight : ndarray
            Covariance of GMM moment conditions.
        """
        nobs = z[0].shape[0]
        w = blocked_inner_prod(z, sigma) / nobs
        return w

    @property
    def config(self):
        """
        Weight estimator configuration

        Returns
        -------
        config : dict
            Dictionary containing weight estimator configuration information
        """
        return {'center': self._center,
                'debiased': self._debiased}


class HeteroskedasticWeightMatrix(HomoskedasticWeightMatrix):
    r"""
    Heteroskedasticity robust weight estimation

    Parameters
    ----------
    center : bool, optional
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix.
    debiased : bool, optional
        Flag indicating whether to use small-sample adjustments

    Notes
    -----
    The weight matrix estimator is

    .. math::

      W   & = n^{-1}\sum_{i=1}^{n}g'_ig_i \\
      g_i & = (z_{1i}\epsilon_{1i},z_{2i}\epsilon_{2i},\ldots,z_{ki}\epsilon_{ki})

    where :math:`g_i` is the vector of scores across all equations for
    observation i.  :math:`z_{ji}` is the vector of instruments for equation
    j and :\math:`\epsilon_{ji}` is the error for equation j for observation
    i.  This form allows for heteroskedasticity and arbitrary cross-sectional
    dependence between the moment conditions.
    """

    def __init__(self, center=False, debiased=False):
        super(HeteroskedasticWeightMatrix, self).__init__(center, debiased)
        self._name = 'Heteroskedastic (Robust) Weighting'

    def weight_matrix(self, x, z, eps, *, sigma=None):
        """
        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar)
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr)
        eps : ndarray
            Model errors (nobs by 1)

        Returns
        -------
        weight : ndarray
            Covariance of GMM moment conditions.
        """
        nobs = x[0].shape[0]
        k = len(x)
        k_total = sum(map(lambda a: a.shape[1], z))
        ze = empty((nobs, k_total))
        loc = 0
        for i in range(k):
            e = eps[:, [i]]
            zk = z[i].shape[1]
            ze[:, loc:loc + zk] = z[i] * e
            loc += zk
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu
        w = ze.T @ ze / nobs
        scale = self._debias_scale(nobs, x, z)
        w *= scale

        return w

    def _debias_scale(self, nobs, x, z):
        if not self._debiased:
            return 1
        nvar = array(list(map(lambda a: a.shape[1], x)))
        ninstr = array(list(map(lambda a: a.shape[1], z)))
        nvar = repeat(nvar, ninstr)
        nvar = sqrt(nvar)[:, None]
        scale = nobs / (nobs - nvar @ nvar.T)
        return scale


class KernelWeightMatrix(HeteroskedasticWeightMatrix, _HACMixin):
    def __init__(self, center=False, debiased=False, kernel='bartlett', bandwidth=None):
        super(HeteroskedasticWeightMatrix, self).__init__(center, debiased)
        self._name = 'Kernel (HAC) Weighting'
        self._check_kernel(kernel)
        self._check_bandwidth(bandwidth)

    def weight_matrix(self, x, z, eps, *, sigma=None):
        """
        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar)
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr)
        eps : ndarray
            Model errors (nobs by 1)

        Returns
        -------
        weight : ndarray
            Covariance of GMM moment conditions.
        """
        nobs = x[0].shape[0]
        k = len(x)
        k_total = sum(map(lambda a: a.shape[1], z))
        ze = empty((nobs, k_total))
        loc = 0
        for i in range(k):
            e = eps[:, [i]]
            zk = z[i].shape[1]
            ze[:, loc:loc + zk] = z[i] * e
            loc += zk
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu
        self._optimal_bandwidth(ze)
        w = self._kernel_cov(ze)
        scale = self._debias_scale(nobs, x, z)
        w *= scale

        return w

    def _optimal_bandwidth(self, moments):
        """Compute optimal bandwidth used in estimation"""
        m = moments / moments.std(0)[None, :]
        m = m.sum(1)
        self._bandwidth = kernel_optimal_bandwidth(m, kernel=self.kernel)
        return self._bandwidth

    @property
    def bandwidth(self):
        return self._bandwidth
