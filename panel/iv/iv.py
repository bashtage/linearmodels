from __future__ import print_function, absolute_import, division

import scipy.stats as stats
from numpy import sqrt, diag, abs, eye, array, all, any, zeros
from numpy.linalg import pinv, inv, matrix_rank

from panel.utility import has_constant
from panel.iv.covariance import HomoskedasticCovariance, HeteroskedasticCovariance, \
    KernelCovariance, OneWayClusteredCovariance, IVGMMCovariance, \
    kernel_weight_bartlett, kernel_weight_parzen, kernel_weight_quadratic_spectral

COVARIANCE_ESTIMATORS = {'homoskedastic': HomoskedasticCovariance,
                         'unadjusted': HomoskedasticCovariance,
                         'homo': HomoskedasticCovariance,
                         'robust': HeteroskedasticCovariance,
                         'heteroskedastic': HeteroskedasticCovariance,
                         'hccm': HeteroskedasticCovariance,
                         'newey-west': KernelCovariance,
                         'bartlett': KernelCovariance,
                         'one-way': OneWayClusteredCovariance}


class IV2SLS(object):
    """
    Parameters
    ----------
    endog : array-like
        Endogenous variables
    exog : array-like
        Exogenous variables
    instruments : array-like
        Instrumental variables

    Notes
    -----
    .. todo::
    
        * VCV: cluster clustvar, bootstrap, jackknife, or hac kernel
        * small sample adjustments
    """

    def __init__(self, endog, exog, instruments):
        self.endog = endog
        self.exog = exog
        self.instruments = instruments

        self._has_constant = False
        self._regressor_is_exog = array([True] * exog.shape[1])
        self._validate_inputs()

    def _validate_inputs(self):
        x, z = self.exog, self.instruments
        self._has_constant = has_constant(x)

        if matrix_rank(x) < x.shape[1]:
            raise ValueError('exogenous data does not have full column rank')
        if matrix_rank(z) < z.shape[1]:
            raise ValueError('exogenous data does not have full column rank')

        for col in range(x.shape[1]):
            xc = x[:, col]
            pinvz = pinv(z)
            if all(xc == 1) or (xc.ptp(axis=0) == 0 and xc[0] != 0):
                self._regressor_is_exog[col] = True
                continue

            if any(all(xc[:, None] == z, axis=0)):
                self._regressor_is_exog[col] = True
                continue

            params = pinvz @ xc
            e = xc - z @ params

            self._regressor_is_exog[col] = ((e.T @ e) / (xc.T @ xc)) < 1e-8

    @staticmethod
    def estimate_parameters(x, y, z):
        """
        Parameters
        ----------
        x : ndarray
            Regressor matrix (nobs by nvar)
        y : ndarray
            Regressand matrix (nobs by 1)
        z : ndarray
            Instrument matrix (nobs by ninstr)

        Returns
        -------
        params : ndarray
            Estimated parameters (nvar by 1)

        Notes
        -----
        Exposed as a static method to facilitate estimation with other data,
        e.g., bootstrapped samples.  Performs no error checking.
        """
        pinvz = pinv(z)
        return inv((x.T @ z) @ (pinvz @ x)) @ ((x.T @ z) @ (pinvz @ y))

    def fit(self, cov_type='robust', **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str
            Name of covariance estimator to use
        **cov_config
            Additional parameters to pass to covariance estimator

        Returns
        -------
        results : IVResults
            Results container

        Notes
        -----
        Additional covariance parameters depend on specific covariance used.
        The see default property for a specific covariance estimator for a 
        list of supported options.  Defaults are used if no covariance 
        configuration is provided.
        """
        y, x, z = self.endog, self.exog, self.instruments
        params = self.estimate_parameters(x, y, z)

        cov_estimator = COVARIANCE_ESTIMATORS[cov_type]
        eps = self.resids(params)
        cov = cov_estimator(x, z, eps, **cov_config).cov

        mu = self.endog.mean() if self.has_constant else 0
        residual_ss = (eps.T @ eps)
        model_ss = ((y - mu).T @ (y - mu))
        r2 = 1 - residual_ss / model_ss

        return IVResults(params, cov, r2, cov_type, residual_ss, model_ss, self)

    def resids(self, params):
        return self.endog - self.exog @ params

    @property
    def has_constant(self):
        return self._has_constant


class HomoskedasticWeightMatrix(object):
    """
    Parameters
    ----------
    **weight_config
        Keywords to pass to weight matrix
    """

    def __init__(self, **weight_config):
        for key in weight_config:
            if key not in self.defaults:
                raise ValueError('Unknown weighting matrix configuration '
                                 'parameter {0}'.format(key))
        self._weight_config = weight_config
        self._bandwidth = 0

    def weight_matrix(self, x, z, eps):
        nobs = x.shape[0]
        s2 = eps.T @ eps / nobs
        return s2 * z.T @ z / nobs

    @property
    def defaults(self):
        return {}

    @property
    def bandwidth(self):
        return self._bandwidth


class HeteroskedasticWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, **weight_config):
        super(HeteroskedasticWeightMatrix, self).__init__(**weight_config)

    def weight_matrix(self, x, z, eps):
        nobs = x.shape[0]
        ze = z * eps
        return ze.T @ ze / nobs


class KernelWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, **weight_config):
        super(KernelWeightMatrix, self).__init__(**weight_config)
        self._bandwidth = 0
        self._kernels = {'bartlett': kernel_weight_bartlett,
                         'newey-west': kernel_weight_bartlett,
                         'parzen': kernel_weight_parzen,
                         'gallant': kernel_weight_parzen,
                         'andrews': kernel_weight_quadratic_spectral,
                         'quadratic-spectral': kernel_weight_quadratic_spectral,
                         'qs': kernel_weight_quadratic_spectral}

    def weight_matrix(self, x, z, eps):
        ze = z * eps
        nobs, ninstr = z.shape

        # TODO: Fix this to allow optimal bw selection by default
        wc = self._weight_config
        bw = wc.get('bw', nobs - 2)
        kernel = wc.get('kernel', self.defaults['kernel'])
        w = self._kernels[kernel](bw)
        s = ze.T @ ze
        for i in range(1, bw + 1):
            s += w[i] * ze[i:].T @ ze[:-i]
        return s / nobs

    def _optimal_bandwidth(self, x, z, eps):
        # TODO: Implement this
        pass

    @property
    def defaults(self):
        return {'kernel': 'bartlett',
                'bw': None}


WEIGHT_MATRICES = {'unadjusted': HomoskedasticWeightMatrix,
                   'homoskedastic': HomoskedasticWeightMatrix,
                   'robust': HeteroskedasticWeightMatrix,
                   'heteroskedastic': HeteroskedasticWeightMatrix,
                   'kernel': KernelWeightMatrix}


class IVGMM(IV2SLS):
    """
    Parameters
    ----------
    endog : array-like
    exog : array-like
    instruments : array-like
    weight_type : str
        Name of weight function to use. 
    **weight_config
        Additional keyword arguments to pass to the weight function.  

    Notes
    -----
    Available weight functions are:
      * 'unadjusted', 'homoskedastic' - Assumes moment conditions are homoskedastic 
      * 'robust' - Allows for heterosedasticity by not autocorrelation
      * 'kernel' - Allows for heteroskedasticity and autocorrelation
      * 'cluster' - Allows for one-way cluster dependence 
      
    .. todo:
         * VCV: unadjusted, robust, cluster clustvar, bootstrap, jackknife, or hac kernel
         * small sample adjustments
         * Colinearity check
         * Options for weighting matrix calculation

    """

    def __init__(self, endog, exog, instruments, weight_type='robust', **weight_config):
        super(IVGMM, self).__init__(endog, exog, instruments)
        weight_matrix_estimator = WEIGHT_MATRICES[weight_type]
        config = weight_matrix_estimator().defaults
        config.update(weight_config)
        self._weight = weight_matrix_estimator(**weight_config)
        self._weight_type = weight_type
        self._weight_config = config

    @staticmethod
    def estimate_parameters(x, y, z, w):
        """
        Parameters
        ----------
        x : ndarray
            Regressor matrix (nobs by nvar)
        y : ndarray
            Regressand matrix (nobs by 1)
        z : ndarray
            Instrument matrix (nobs by ninstr)
        w : ndarray
            GMM weight matrix (ninstr by ninstr)

        Returns
        -------
        params : ndarray
            Estimated parameters (nvar by 1)

        Notes
        -----
        Exposed as a static method to facilitate estimation with other data, 
        e.g., bootstrapped samples.  Performs no error checking.
        """
        omega = z @ w @ z.T
        return inv(x.T @ omega @ x) @ (x.T @ omega @ y)

    def fit(self, iter_limit=2, tol=1e-4, cov_type='robust', **cov_config):
        y, x, z = self.endog, self.exog, self.instruments
        nobs, ninstr = y.shape[0], z.shape[1]
        weight_matrix = self._weight.weight_matrix
        _params = params = self.estimate_parameters(x, y, z, eye(ninstr))
        i, norm = 1, 10 * tol
        while i < iter_limit and norm > tol:
            eps = y - x @ params
            ze = z * eps
            w = inv(weight_matrix(x, z, eps))
            params = self.estimate_parameters(x, y, z, w)
            delta = params - _params
            xpz = x.T @ z / nobs
            if i == 1:
                v = (xpz @ w @ xpz.T) / nobs
                vinv = inv(v)
            _params = params
            norm = delta.T @ vinv @ delta
            i += 1

        cov = IVGMMCovariance(x, z, eps, w, **cov_config).cov

        mu = self.endog.mean() if self.has_constant else 0
        residual_ss = (eps.T @ eps)
        model_ss = ((y - mu).T @ (y - mu))
        r2 = 1 - residual_ss / model_ss

        return IVGMMResults(params, cov, r2, cov_type, residual_ss, model_ss,
                            w, self._weight_type, self._weight_config, i, self)


class IVResults(object):
    """
    Results from IV estimation

    Notes
    -----
    .. todo::
        
        * F
        * chi2 -- what is this?
        * kappa - for LIML
        * J_stat - for GMM
        * Hypothesis testing
        * First stage diagnostics

    """

    def __init__(self, params, cov, r2, cov_type, rss, tss, model):
        self._params = params
        self._cov = cov
        self._model = model
        self._r2 = r2
        self._cov_type = cov_type
        self._rss = rss
        self._tss = tss
        self._cache = {}

    @property
    def cov(self):
        """Estimated covariance of parameters"""
        return self._cov

    @property
    def params(self):
        """Estimated parameters"""
        return self._params

    @property
    def resids(self):
        """Estimated residuals"""
        return self._model.resids(self._params)

    @property
    def nobs(self):
        """Number of observations"""
        return self._model.endog.shape[0]

    @property
    def df_resid(self):
        """Residual degree of freedom"""
        return self.nobs - self._model.exog.shape[1]

    @property
    def df_model(self):
        """Model degree of freedom"""
        return self._model.exog.shape[1]

    @property
    def rsquared(self):
        """Coefficient of determination (R**2)"""
        return self._r2

    @property
    def rsquared_adj(self):
        """Sample-size adjusted coefficient of determination (R**2)"""
        n, k = self.nobs, self.df_model
        return 1 - ((n - 1) / (n - k)) * (1 - self._r2)

    @property
    def cov_type(self):
        """Covariance estimator used"""
        return self._cov_type

    @property
    def std_errors(self):
        """Estimated parameter standard errors"""
        return sqrt(diag(self.cov))[:, None]

    @property
    def tstats(self):
        """Parameter t-statistics"""
        return self.params / self.std_errors

    @property
    def pvalues(self):
        """P-values of parameter t-statistics"""
        if 'pvalues' not in self._cache:
            self._cache['pvalues'] = 2 - 2 * stats.norm.cdf(abs(self.tstats))
        return self._cache['pvalues']

    @property
    def total_ss(self):
        return self._tss

    @property
    def resid_ss(self):
        return self._rss


class IVGMMResults(IVResults):
    def __init__(self, params, cov, r2, cov_type, rss, tss, weight_mat,
                 weight_type, weight_config, iterations, model):
        super(IVGMMResults, self).__init__(params, cov, r2, cov_type, rss, tss, model)
        self._weight_mat = weight_mat
        self._weight_type = weight_type
        self._weight_config = weight_config
        self._iterations = iterations

    @property
    def weight_matrix(self):
        """Weight matrix used in the final-step GMM estimation"""
        return self._weight_mat

    @property
    def iterations(self):
        """Iterations used in GMM estimation"""
        return self._iterations

    @property
    def weight_type(self):
        """Weighting matrix method used in estimation"""
        return self._weight_type

    @property
    def weight_config(self):
        """Weighting matrix parameters used in estimation"""
        return self._weight_config
