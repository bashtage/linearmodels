from __future__ import print_function, absolute_import, division

import scipy.stats as stats
from numpy import sqrt, diag, abs, ceil, where, argsort, r_, unique, zeros, \
    arange
from numpy.linalg import pinv, inv

from panel.utility import has_constant


class IVCovariance(object):
    def __init__(self, x, z, eps, **config):
        self.x = x
        self.z = z
        self.eps = eps
        self.config = self._check_config(**config)
        self._pinvz = None

    def _check_config(self, **config):
        if len(config) == 0:
            return config

        valid_keys = list(self.defaults.keys())
        invalid = []
        for key in config:
            if key not in valid_keys:
                invalid.append(key)
        if invalid:
            keys = ', '.join(config.keys())
            raise ValueError('Unexpected keywords in config: {0}'.format(keys))

        return config

    @property
    def defaults(self):
        return {}

    @property
    def cov(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape

        scale = nobs / (nobs - nvar) if self.config.get('debiased', False) else 1
        self._pinvz = pinvz = pinv(z) if self._pinvz is None else self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        vinv = inv(v)

        return scale * vinv @ self.s @ vinv / nobs


class HomoskedasticCovariance(IVCovariance):
    def __init__(self, x, z, eps, **config):
        super(HomoskedasticCovariance, self).__init__(x, z, eps, **config)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        s2 = eps.T @ eps / nobs
        v = (x.T @ z) @ (pinv(z) @ x) / nobs

        return s2 * v

    @property
    def defaults(self):
        return {'debiased': False}


class NeweyWestCovariance(HomoskedasticCovariance):
    def __init__(self, x, z, eps, **config):
        super(NeweyWestCovariance, self).__init__(x, z, eps, **config)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        # TODO: Bandwidth selection method
        bw = self.config.get('bw', ceil(20 * (nobs / 100) ** (2 / 9)))
        bw = int(bw)

        self._pinvz = pinvz = pinv(z) if self._pinvz is None else self._pinvz
        xhat_e = z @ (pinvz @ x) * eps
        s = xhat_e.T @ xhat_e
        for i in range(bw):
            w = (1 - (i + 1) / (bw + 1))
            s += w * xhat_e[i + 1:].T @ xhat_e[:-(i + 1)]
        s /= nobs

        return s

    @property
    def defaults(self):
        """
        Default values
        
        Returns
        -------
        defaults : dict
            Dictionary containing valid options and their default value
        
        Notes
        -----
        When ``bw`` is None, automatic bandwidth selection is used.
        """
        return {'bw': None,
                'debiased': False}


class HeteroskedasticCovariance(HomoskedasticCovariance):
    def __init__(self, x, z, eps, **config):
        super(HeteroskedasticCovariance, self).__init__(x, z, eps, **config)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        self._pinvz = pinvz = pinv(z) if self._pinvz is None else self._pinvz
        xhat_e = z @ (pinvz @ x) * eps
        s = xhat_e.T @ xhat_e / nobs
        return s


class OneWayClusteredCovariance(HomoskedasticCovariance):
    def __init__(self, x, z, eps, **config):
        super(OneWayClusteredCovariance, self).__init__(x, z, eps, **config)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        self._pinvz = pinvz = pinv(z) if self._pinvz is None else self._pinvz
        xhat_e = z @ (pinvz @ x) * eps

        nobs, nvar = x.shape
        clusters = self.config.get('clusters', arange(nobs))
        num_clusters = len(unique(clusters))

        clusters = clusters.squeeze()
        if num_clusters > 1:
            sort_args = argsort(clusters)
        else:
            sort_args = list(range(nobs))

        clusters = clusters[sort_args]
        locs = where(r_[True, clusters[:-1] != clusters[1:], True])[0]
        xhat_e = xhat_e[sort_args]

        s = zeros((nvar, nvar))
        for i in range(num_clusters):
            st, en = locs[i], locs[i + 1]
            xhat_e_bar = xhat_e[st:en].sum(axis=0)[:, None]
            s += xhat_e_bar @ xhat_e_bar.T

        s *= num_clusters / (num_clusters - 1) / nobs

        return s

    @property
    def defaults(self):
        return {'debiased': False,
                'clusters': None}


COVARIANCE_ESTIMATORS = {'homoskedastic': HomoskedasticCovariance,
                         'unadjusted': HomoskedasticCovariance,
                         'homo': HomoskedasticCovariance,
                         'robust': HeteroskedasticCovariance,
                         'heteroskedastic': HeteroskedasticCovariance,
                         'hccm': HeteroskedasticCovariance,
                         'newey-west': NeweyWestCovariance,
                         'bartlett': NeweyWestCovariance,
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

    def fit(self):
        y, x, z = self.endog, self.exog, self.instruments
        nobs = y.shape[0]
        params = IV2SLS(y, x, z).fit()
        e = y - x @ params
        ze = z @ e
        s = ze.T @ ze / nobs
        w = inv(s)
        omega = z @ w @ z.t
        return inv(x.T @ omega @ x) @ (x.T @ omega @ y)


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
