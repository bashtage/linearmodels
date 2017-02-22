from __future__ import print_function, absolute_import, division

import scipy.stats as stats
from numpy import sqrt, diag, abs, eye, array, all, any, isscalar, c_
from numpy.linalg import pinv, inv, matrix_rank, eigvalsh

from panel.iv.covariance import HomoskedasticCovariance, IVGMMCovariance, \
    HeteroskedasticCovariance, KernelCovariance, OneWayClusteredCovariance
from panel.iv.weighting import HomoskedasticWeightMatrix, KernelWeightMatrix, \
    HeteroskedasticWeightMatrix, OneWayClusteredWeightMatrix
from panel.utility import has_constant, inv_sqrth, WaldTestStatistic

COVARIANCE_ESTIMATORS = {'homoskedastic': HomoskedasticCovariance,
                         'unadjusted': HomoskedasticCovariance,
                         'homo': HomoskedasticCovariance,
                         'robust': HeteroskedasticCovariance,
                         'heteroskedastic': HeteroskedasticCovariance,
                         'hccm': HeteroskedasticCovariance,
                         'kernel': KernelCovariance,
                         'one-way': OneWayClusteredCovariance}

WEIGHT_MATRICES = {'unadjusted': HomoskedasticWeightMatrix,
                   'homoskedastic': HomoskedasticWeightMatrix,
                   'robust': HeteroskedasticWeightMatrix,
                   'heteroskedastic': HeteroskedasticWeightMatrix,
                   'kernel': KernelWeightMatrix,
                   'clustered': OneWayClusteredWeightMatrix}


class IV2SLS(object):
    """
    Estimation of IV models using two-stage least squares
    
    Parameters
    ----------
    endog : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous variables (nobs by nvar)
    instruments : array-like
        Instrumental variables (nobs by ninstr)

    Notes
    -----

    .. todo::

        * VCV: bootstrap
        * testing
    """

    def __init__(self, endog, exog, instrumented, instruments):
        self.endog = endog
        self.exog = exog
        self.instrumented = instrumented
        self.instruments = instruments
        self._x = c_[exog, instrumented]  # model regressors
        self._z = c_[exog, instruments]  # first-stage regressors


        self._has_constant = False
        self._regressor_is_exog = array([True] * exog.shape[1] +
                                        [False] * instrumented.shape[1])
        self._validate_inputs()

    def _validate_inputs(self):
        x, z = self._x, self._z
        self._has_constant = has_constant(x)

        if matrix_rank(x) < x.shape[1]:
            raise ValueError('regressors not have full column rank')
        if matrix_rank(z) < z.shape[1]:
            raise ValueError('instruments do not have full column rank')

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
        y, x, z = self.endog, self._x, self._z
        params = self.estimate_parameters(x, y, z)

        cov_estimator = COVARIANCE_ESTIMATORS[cov_type]
        cov_estimator = cov_estimator(x, y, z, params, **cov_config)
        cov = cov_estimator.cov
        cov_config = cov_estimator.config
        s2, debiased = cov_estimator.s2, cov_estimator.debiased

        eps = self.resids(params)
        mu = self.endog.mean() if self.has_constant else 0
        residual_ss = (eps.T @ eps)
        model_ss = ((y - mu).T @ (y - mu))
        r2 = 1 - residual_ss / model_ss
        fstat = self._f_statistic(params, cov, cov_config)

        return IVResults(params, cov, r2, cov_type, residual_ss, model_ss,
                         s2, debiased, fstat, self)

    def resids(self, params):
        return self.endog - self._x @ params

    @property
    def has_constant(self):
        return self._has_constant

    def _f_statistic(self, params, cov, cov_config):
        debiased = cov_config['debiased']
        non_const = ~(self._x.ptp(0) == 0)
        test_params = params[non_const]
        test_cov = cov[non_const][:, non_const]
        test_stat = test_params.T @ inv(test_cov) @ test_params
        nobs, nvar = self._x.shape
        null = 'All parameters ex. constant not zero'
        df = test_params.shape[0]
        if debiased:
            wald = WaldTestStatistic(test_stat / df, null, df, nobs - nvar)
        else:
            wald = WaldTestStatistic(test_stat, null, df)

        return wald


class IVLIML(IV2SLS):
    """
    Limited information ML estimation of IV models
    
    Parameters
    ----------
    endog : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous variables (nobs by nvar)
    instruments : array-like
        Instrumental variables (nobs by ninstr)

    Notes
    -----

    .. todo::

        * VCV: bootstrap
        * testing
    """

    def __init__(self, endog, exog, instrumented, instruments, kappa=None):
        super(IVLIML, self).__init__(endog, exog, instrumented, instruments)
        self._kappa = kappa
        if kappa is not None and not isscalar(kappa):
            raise ValueError('kappa must be None or a scalar')

    @staticmethod
    def estimate_parameters(x, y, z, kappa):
        """
        Parameters
        ----------
        x : ndarray
            Regressor matrix (nobs by nvar)
        y : ndarray
            Regressand matrix (nobs by 1)
        z : ndarray
            Instrument matrix (nobs by ninstr)
        kappa : scalar
            Parameter value for k-class esimtator

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
        p1 = (x.T @ x) * (1 - kappa) + kappa * ((x.T @ z) @ (pinvz @ x))
        p2 = (x.T @ y) * (1 - kappa) + kappa * ((x.T @ z) @ (pinvz @ y))
        return inv(p1) @ p2

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
        y, x, z = self.endog, self._x, self._z
        kappa = self._kappa
        if kappa is None:
            is_exog = self._regressor_is_exog
            e = c_[y, x[:, ~is_exog]]
            x1 = x[:, is_exog]

            ez = e - z @ (pinv(z) @ e)
            ex1 = e - x1 @ (pinv(x1) @ e)

            vpmzv_sqinv = inv_sqrth(ez.T @ ez)
            q = vpmzv_sqinv @ (ex1.T @ ex1) @ vpmzv_sqinv
            kappa = min(eigvalsh(q))

        params = self.estimate_parameters(x, y, z, kappa)

        cov_estimator = COVARIANCE_ESTIMATORS[cov_type]
        cov_estimator = cov_estimator(x, y, z, params, **cov_config)
        cov = cov_estimator.cov
        s2, debiased = cov_estimator.s2, cov_estimator.debiased
        cov_config = cov_estimator.config

        eps = self.resids(params)
        mu = self.endog.mean() if self.has_constant else 0
        residual_ss = (eps.T @ eps)
        model_ss = ((y - mu).T @ (y - mu))
        r2 = 1 - residual_ss / model_ss
        fstat = self._f_statistic(params, cov, cov_config)

        return IVResults(params, cov, r2, cov_type, residual_ss, model_ss,
                         s2, debiased, fstat, self, kappa=kappa)


class IVGMM(IV2SLS):
    """
    Estimation of IV models using the generalized method of moments (GMM)
    
    Parameters
    ----------
    endog : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous variables (nobs by nvar)
    instruments : array-like
        Instrumental variables (nobs by ninstr)
    weight_type : str
        Name of weight function to use.
    **weight_config
        Additional keyword arguments to pass to the weight function.

    Notes
    -----
    Available weight functions are:
      * 'unadjusted', 'homoskedastic' - Assumes moment conditions are
        homoskedastic
      * 'robust' - Allows for heterosedasticity by not autocorrelation
      * 'kernel' - Allows for heteroskedasticity and autocorrelation
      * 'cluster' - Allows for one-way cluster dependence

    .. todo:
         * VCV: unadjusted, robust, cluster clustvar, bootstrap, jackknife,
           or hac kernel
         * small sample adjustments
         * Colinearity check
         * Options for weighting matrix calculation
    """

    def __init__(self, endog, exog, instrumented, instruments, weight_type='robust',
                 **weight_config):
        super(IVGMM, self).__init__(endog, exog, instrumented, instruments)
        weight_matrix_estimator = WEIGHT_MATRICES[weight_type]
        self._weight = weight_matrix_estimator(**weight_config)
        self._weight_type = weight_type
        self._weight_config = self._weight.config

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
        xpz = x.T @ z
        zpy = z.T @ y
        return inv(xpz @ w @ xpz.T) @ (xpz @ w @ zpy)

    def fit(self, iter_limit=2, tol=1e-4, cov_type='robust', **cov_config):
        y, x, z = self.endog, self._x, self._z
        nobs, ninstr = y.shape[0], z.shape[1]
        weight_matrix = self._weight.weight_matrix
        _params = params = self.estimate_parameters(x, y, z, eye(ninstr))
        eps = y - x @ params
        i, norm = 1, 10 * tol
        while i < iter_limit and norm > tol:
            w = inv(weight_matrix(x, z, eps))
            params = self.estimate_parameters(x, y, z, w)
            eps = y - x @ params
            delta = params - _params
            xpz = x.T @ z / nobs
            if i == 1:
                v = (xpz @ w @ xpz.T) / nobs
                vinv = inv(v)
            _params = params
            norm = delta.T @ vinv @ delta
            i += 1

        cov_estimator = IVGMMCovariance(x, y, z, params, w, **cov_config)
        cov = cov_estimator.cov
        s2, debiased = cov_estimator.s2, cov_estimator.debiased
        cov_config = cov_estimator.config

        mu = self.endog.mean() if self.has_constant else 0
        residual_ss = (eps.T @ eps)
        model_ss = ((y - mu).T @ (y - mu))
        r2 = 1 - residual_ss / model_ss
        fstat = self._f_statistic(params, cov, cov_config)

        return IVGMMResults(params, cov, r2, cov_type, residual_ss, model_ss,
                            s2, debiased, w, self._weight_type,
                            self._weight_config, i, fstat, self)


class IVResults(object):
    """
    Results from IV estimation

    Notes
    -----
    .. todo::

        * Information about covariance estimator 
        * J_stat - for GMM
        * Hypothesis testing
        * First stage diagnostics

    """

    def __init__(self, params, cov, r2, cov_type, rss, tss, s2, debiased,
                 fstat, model, kappa=1):
        self._params = params
        self._cov = cov
        self._model = model
        self._r2 = r2
        self._cov_type = cov_type
        self._rss = rss
        self._tss = tss
        self._s2 = s2
        self._debiased = debiased
        self._kappa = kappa
        self._f_statistic = fstat
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
        return self._model.exog.shape[1] - self._model.has_constant

    @property
    def rsquared(self):
        """Coefficient of determination (R**2)"""
        return self._r2

    @property
    def rsquared_adj(self):
        """Sample-size adjusted coefficient of determination (R**2)"""
        n, k, c = self.nobs, self.df_model, self._model.has_constant
        return 1 - ((n - c) / (n - k)) * (1 - self._r2)

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

    @property
    def kappa(self):
        return self._kappa

    @property
    def s2(self):
        """Residual variance estimator"""
        return self._s2

    @property
    def debiased(self):
        """Flag indicating whether covariance uses a small-sample adjustment"""
        return self._debiased

    @property
    def f_statistic(self):
        return self._f_statistic


class IVGMMResults(IVResults):
    def __init__(self, params, cov, r2, cov_type, rss, tss, s2, debiased,
                 weight_mat, weight_type, weight_config, iterations, fstat,
                 model):
        super(IVGMMResults, self).__init__(params, cov, r2, cov_type, rss,
                                           tss, s2, debiased, fstat, model)
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

class IVContinuousUpdatingGMM(IVGMM):
    """
    Estimation of IV models using the generalized method of moments (GMM)

    Parameters
    ----------
    endog : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous variables (nobs by nvar)
    instruments : array-like
        Instrumental variables (nobs by ninstr)
    weight_type : str
        Name of weight function to use.
    **weight_config
        Additional keyword arguments to pass to the weight function.

    Notes
    -----
    Available weight functions are:
      * 'unadjusted', 'homoskedastic' - Assumes moment conditions are
        homoskedastic
      * 'robust' - Allows for heterosedasticity by not autocorrelation
      * 'kernel' - Allows for heteroskedasticity and autocorrelation
      * 'cluster' - Allows for one-way cluster dependence

    .. todo:
         * VCV: unadjusted, robust, cluster clustvar, bootstrap, jackknife,
           or hac kernel
         * small sample adjustments
         * Colinearity check
         * Options for weighting matrix calculation
    """

    def __init__(self, endog, exog, instruments, weight_type='robust',
                 **weight_config):
        super(IVContinuousUpdatingGMM, self).__init__(endog, exog, instruments,
                                                      weight_type,
                                                      **weight_config)

    def j(self, params):
        y, x, z = self.endog, self.exog, self.instruments
        nobs, ninstr = y.shape[0], z.shape[1]
        weight_matrix = self._weight.weight_matrix
        eps = y - x @ params
        w = inv(weight_matrix(x, z, eps))
        g_bar = (z * eps).mean(0)
        return nobs * g_bar.T @ w @ g_bar.T

    def estimate_parameters(self, x, y, z):
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
        from scipy.optimize import minimize
        IV2SLS()
        res = minimize(self.j, sv, options={'disp': True})
        return

    def fit(self, iter_limit=2, tol=1e-4, cov_type='robust', **cov_config):
        y, x, z = self.endog, self.exog, self.instruments
        nobs, ninstr = y.shape[0], z.shape[1]
        weight_matrix = self._weight.weight_matrix
        _params = params = self.estimate_parameters(x, y, z, eye(ninstr))
        eps = y - x @ params
        i, norm = 1, 10 * tol
        while i < iter_limit and norm > tol:
            w = inv(weight_matrix(x, z, eps))
            params = self.estimate_parameters(x, y, z, w)
            eps = y - x @ params
            delta = params - _params
            xpz = x.T @ z / nobs
            if i == 1:
                v = (xpz @ w @ xpz.T) / nobs
                vinv = inv(v)
            _params = params
            norm = delta.T @ vinv @ delta
            i += 1

        cov_estimator = IVGMMCovariance(x, y, z, params, w, **cov_config)
        cov = cov_estimator.cov
        s2, debiased = cov_estimator.s2, cov_estimator.debiased
        cov_config = cov_estimator.config

        mu = self.endog.mean() if self.has_constant else 0
        residual_ss = (eps.T @ eps)
        model_ss = ((y - mu).T @ (y - mu))
        r2 = 1 - residual_ss / model_ss
        fstat = self._f_statistic(params, cov, cov_config)

        return IVGMMResults(params, cov, r2, cov_type, residual_ss, model_ss,
                            s2, debiased, w, self._weight_type,
                            self._weight_config, i, fstat, self)
