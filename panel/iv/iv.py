from __future__ import print_function, absolute_import, division

import scipy.stats as stats
from numpy import sqrt, diag, abs, eye
from numpy.linalg import pinv, inv

from panel.utility import has_constant
from panel.iv.covariance import HomoskedasticCovariance, HeteroskedasticCovariance, \
    KernelCovariance, OneWayClusteredCovariance

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
        * Colinearity check
    """

    def __init__(self, endog, exog, instruments):
        self.endog = endog
        self.exog = exog
        self.instruments = instruments

        self._validate_inputs()
        self._has_constant = False

    def _validate_inputs(self):
        self._has_constant = has_constant(self.exog)
        pass

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

    def cov(self, params):
        estimator = COVARIANCE_ESTIMATORS[self.cov_type]
        eps = self.resids(params)
        x, z = self.exog, self.instruments
        return estimator(x, z, eps, **self.cov_config).cov

    def resids(self, params):
        return self.endog - self.exog @ params

    @property
    def has_constant(self):
        return self._has_constant

    @property
    def cov_type(self):
        return self._cov_type

    @property
    def cov_config(self):
        return self._cov_config

    def change_cov_estimator(self, cov_type, **cov_config):
        self._cov_type = cov_type
        self._cov_config = cov_config


class IVGMM(object):
    """
    Parameters
    ----------
    endog : array-like
    exog : array-like
    instruments : array-like

    Notes
    -----
    .. todo:
         * VCV: unadjusted, robust, cluster clustvar, bootstrap, jackknife, or hac kernel
         * small sample adjustments
         * Colinearity check
         * Options for weighting matrix calculation
         * 1-step, 2-step and iterative

    """

    def __init__(self, endog, exog, instruments):
        self.endog = endog
        self.exog = exog
        self.instruments = instruments

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

    def fit(self, iter=2, tol=1e-4):
        y, x, z = self.endog, self.exog, self.instruments
        nobs, ninstr = y.shape[0], z.shape[1]
        _params = params = self.estimate_parameters(x, y, z, eye(ninstr))
        i, norm = 0, 10 * tol
        while i < (iter - 1) and norm > tol:
            e = y - x @ params
            ze = z * e
            s = ze.T @ ze / nobs
            w = inv(s)
            params = self.estimate_parameters(x, y, z, w)
            delta = params - _params
            xpz = x.T @ z / nobs
            if i == 0:
                v = (xpz @ w @ xpz.T) / nobs
                vinv = inv(v)
            _params = params
            norm = delta.T @ vinv @ delta
            i += 1

        return params


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
