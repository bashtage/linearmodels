"""
Instrumental variable estimators
"""
from __future__ import print_function, absolute_import, division

import scipy.stats as stats
from numpy import sqrt, diag, abs, array, isscalar, c_
from numpy.linalg import pinv, inv, matrix_rank, eigvalsh
from pandas import Series, DataFrame
from panel.iv.covariance import (HomoskedasticCovariance,
                                 HeteroskedasticCovariance, KernelCovariance,
                                 OneWayClusteredCovariance)
from panel.iv.data import DataHandler
from panel.utility import has_constant, inv_sqrth, WaldTestStatistic
from scipy.optimize import minimize

from panel.iv.gmm import (HomoskedasticWeightMatrix, KernelWeightMatrix,
                          HeteroskedasticWeightMatrix, OneWayClusteredWeightMatrix,
                          IVGMMCovariance)

COVARIANCE_ESTIMATORS = {'homoskedastic': HomoskedasticCovariance,
                         'unadjusted': HomoskedasticCovariance,
                         'HomoskedasticCovariance': HomoskedasticCovariance,
                         'homo': HomoskedasticCovariance,
                         'robust': HeteroskedasticCovariance,
                         'heteroskedastic': HeteroskedasticCovariance,
                         'HeteroskedasticCovariance': HeteroskedasticCovariance,
                         'hccm': HeteroskedasticCovariance,
                         'kernel': KernelCovariance,
                         'KernelCovariance': KernelCovariance,
                         'one-way': OneWayClusteredCovariance,
                         'clustered': OneWayClusteredCovariance,
                         'OneWayClusteredCovariance': OneWayClusteredCovariance}

WEIGHT_MATRICES = {'unadjusted': HomoskedasticWeightMatrix,
                   'homoskedastic': HomoskedasticWeightMatrix,
                   'robust': HeteroskedasticWeightMatrix,
                   'heteroskedastic': HeteroskedasticWeightMatrix,
                   'kernel': KernelWeightMatrix,
                   'clustered': OneWayClusteredWeightMatrix,
                   'one-way': OneWayClusteredWeightMatrix, }


def _proj(y, x):
    """
    Projection of y on x from y
    
    Parameters
    ----------
    x : ndarray
        Array to project onto (nobs by nvar)
    y : ndarray
        Array to project (nobs by nseries)
        
    Returns
    -------
    yhat : ndarray
        Projected values of y (nobs by nseries)
    """
    return x @ pinv(x) @ y


def _annihilate(y, x):
    """
    Remove projection of y on x from y
    
    Parameters
    ----------
    x : ndarray
        Array to project onto (nobs by nvar)
    y : ndarray
        Array to project (nobs by nseries)

    Returns
    -------
    eps : ndarray
        Residuals values of y minus y projected on x (nobs by nseries)
    """
    return y - _proj(y, x)


class IVLIML(object):
    r"""
    Limited information ML estimation of IV models

    Parameters
    ----------
    dependent : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous regressors  (nobs by nexog)
    endog : array-like
        Endogenous regressors (nobs by nendog)
    instruments : array-like
        Instrumental variables (nobs by ninstr)
    fuller : float, optional
        Fuller's alpha to modify LIML estimator. Default returns unmodified
        LIML estimator.
    kappa : float, optional
        Parameter value for k-class estimation.  If not provided, computed to
        produce LIML parameter estimate.

    Notes
    -----
    ``kappa`` and ``fuller`` should not be used simultaneously since Fuller's
    alpha applies an adjustment to ``kappa``, and so the same result can be
    computed using only ``kappa``. Fuller's alpha is used to adjust the 
    LIML estimate of :math:`\kappa`, which is computed whenever ``kappa``
    is not provided.
    
    The LIML estimator is defined as 
    
    .. math::
    
      \hat{\beta}_{\kappa} & =(X(I-\kappa M_{z})X)^{-1}X(I-\kappa M_{z})Y\\
      M_{z} & =I-P_{z}\\
      P_{z} & =Z(Z'Z)^{-1}Z'
    
    where :math:`Z` contains both the exogenous regressors and the instruments.
    :math:`\kappa` is estimated as part of the LIML estimator.
    
    When using Fuller's :math:`\alpha`, the value used is modified to 
    
    .. math::
    
      \kappa-\alpha/(n-n_{instr})  

    .. todo::

        * VCV: bootstrap
        * testing
    """

    def __init__(self, dependent, exog, endog, instruments, fuller=0, kappa=None):
        self.dependent = DataHandler(dependent, var_name='dependent')
        nobs = self.dependent.shape[0]
        self.exog = DataHandler(exog, var_name='exog', nobs=nobs)
        self.endog = DataHandler(endog, var_name='endog', nobs=nobs)
        self.instruments = DataHandler(instruments, var_name='instruments', nobs=nobs)

        # dependent variable
        self._y = self.dependent.ndarray
        # model regressors
        self._x = c_[self.exog.ndarray, self.endog.ndarray]
        # first-stage regressors
        self._z = c_[self.exog.ndarray, self.instruments.ndarray]

        self._has_constant = False
        self._regressor_is_exog = array([True] * self.exog.shape[1] +
                                        [False] * self.endog.shape[1])
        self._columns = self.exog.cols + self.endog.cols
        self._instr_columns = self.exog.cols + self.instruments.cols
        self._index = self.endog.rows
        self._validate_inputs()
        self._method = 'liml'
        self._result_container = IVResults

        self._kappa = kappa
        self._fuller = fuller
        if kappa is not None and not isscalar(kappa):
            raise ValueError('kappa must be None or a scalar')
        if not isscalar(fuller):
            raise ValueError('fuller must be None or a scalar')
        if kappa is not None and fuller != 0:
            import warnings
            warnings.warn('kappa and fuller should not normally be used '
                          'simulaneously.  Identical results can be computed '
                          'using kappa only', UserWarning)
        self._method = 'liml'
        additional = []
        if fuller != 0:
            additional.append('fuller(alpha={0})'.format(fuller))
        if kappa is not None:
            additional.append('kappa={0}'.format(fuller))
        if additional:
            self._method += '(' + ', '.join(additional) + ')'

    def _validate_inputs(self):
        x, z = self._x, self._z
        if x.shape[1] == 0:
            raise ValueError('Model must contain at least one regressor.')
        if self.instruments.shape[1] < self.endog.shape[1]:
            raise ValueError('The number of instruments ({0}) must be at least '
                             'as large as the number of endogenous regressors'
                             ' ({1}).'.format(self.instruments.shape[1],
                                              self.endog.shape[1]))
        if matrix_rank(x) < x.shape[1]:
            raise ValueError('regressors [exog endog] not have full '
                             'column rank')
        if matrix_rank(z) < z.shape[1]:
            raise ValueError('instruments [exog insruments]  do not have full '
                             'column rank')
        self._has_constant = has_constant(x)

    @staticmethod
    def estimate_parameters(x, y, z, kappa):
        """
        Parameter estimation without error checking

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
        cov_type : str, optional
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
        The see the docstring of specific covariance estimator for a list of
        supported options. Defaults are used if no covariance configuration
        is provided.
        """
        y, x, z = self._y, self._x, self._z

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

        if self._fuller != 0:
            nobs, ninstr = z.shape
            kappa -= self._fuller / (nobs - ninstr)

        params = self.estimate_parameters(x, y, z, kappa)

        cov_estimator = COVARIANCE_ESTIMATORS[cov_type]
        cov_config['kappa'] = kappa
        cov_estimator = cov_estimator(x, y, z, params, **cov_config)

        results = {'kappa': kappa}
        pe = self._post_estimation(params, cov_estimator, cov_type)
        results.update(pe)

        return self._result_container(results, self)

    def resids(self, params):
        """
        Compute model residuals

        Parameters
        ----------
        params : ndarray
            Model parameters (nvar by 1)

        Returns
        -------
        resids : ndarray
            Model residuals
        """
        return self._y - self._x @ params

    @property
    def has_constant(self):
        """Flag indicating the model includes a constant or equivalent"""
        return self._has_constant

    def _f_statistic(self, params, cov, debiased):
        non_const = ~(self._x.ptp(0) == 0)
        test_params = params[non_const]
        test_cov = cov[non_const][:, non_const]
        test_stat = test_params.T @ inv(test_cov) @ test_params
        test_stat = float(test_stat)
        nobs, nvar = self._x.shape
        null = 'All parameters ex. constant not zero'
        df = test_params.shape[0]
        if debiased:
            wald = WaldTestStatistic(test_stat / df, null, df, nobs - nvar)
        else:
            wald = WaldTestStatistic(test_stat, null, df)

        return wald

    def _post_estimation(self, params, cov_estimator, cov_type):
        vars = self._columns
        index = self._index
        eps = self.resids(params)
        cov = cov_estimator.cov
        debiased = cov_estimator.debiased

        residual_ss = (eps.T @ eps)
        y = self._y
        mu = self._y.mean() if self.has_constant else 0
        total_ss = ((y - mu).T @ (y - mu))
        r2 = 1 - residual_ss / total_ss

        fstat = self._f_statistic(params, cov, debiased)
        out = {'params': Series(params.squeeze(), vars, name='parameter'),
               'eps': Series(eps.squeeze(), index=index, name='residual'),
               'cov': DataFrame(cov, columns=vars, index=vars),
               's2': float(cov_estimator.s2),
               'debiased': debiased,
               'residual_ss': float(residual_ss),
               'total_ss': float(total_ss),
               'r2': float(r2),
               'fstat': fstat,
               'vars': vars,
               'instruments': self._instr_columns,
               'cov_config': cov_estimator.config,
               'cov_type': cov_type,
               'method': self._method}

        return out


class IV2SLS(IVLIML):
    r"""
    Estimation of IV models using two-stage least squares

    Parameters
    ----------
    dependent : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous regressors  (nobs by nexog)
    endog : array-like
        Endogenous regressors (nobs by nendog)
    instruments : array-like
        Instrumental variables (nobs by ninstr)

    Notes
    -----
    The 2SLS estimator is defined
    
    .. math::
    
      \hat{\beta}_{2SLS} & =(X'Z(Z'Z)^{-1}Z'X)^{-1}X'Z(Z'Z)^{-1}Z'Y\\
                         & =(\hat{X}'\hat{X})^{-1}\hat{X}Y\\
                 \hat{X} & =Z(Z'Z)^{-1}Z'X
    
    The 2SLS estimator is a special case of a k-class estimator with
    :math:`\kappa=1`,
    
    .. todo::

        * VCV: bootstrap
        * Mathematical notation
    
    """

    def __init__(self, dependent, exog, endog, instruments):
        super(IV2SLS, self).__init__(dependent, exog, endog, instruments,
                                     fuller=0, kappa=1)
        self._method = '2sls'


class IVGMM(IVLIML):
    r"""
    Estimation of IV models using the generalized method of moments (GMM)

    Parameters
    ----------
    dependent : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous regressors  (nobs by nexog)
    endog : array-like
        Endogenous regressors (nobs by nendog)
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
      * 'robust', 'heteroskedastic' - Allows for heterosedasticity by not 
        autocorrelation
      * 'kernel' - Allows for heteroskedasticity and autocorrelation
      * 'cluster' - Allows for one-way cluster dependence
    
    The estimator is defined as 
    
    .. math::
    
      \hat{\beta}_{gmm}=(X'ZW^{-1}Z'X)^{-1}X'ZW^{-1}Z'Y
    
    where :math:`W` is a positive definite weight matrix and :math:`Z` 
    contains both the exogenous regressors and the instruments.

    .. todo:

         * VCV: bootstrap
         * Post-estimation results
    """

    def __init__(self, dependent, exog, endog, instruments, weight_type='robust',
                 **weight_config):
        super(IVGMM, self).__init__(dependent, exog, endog, instruments)
        weight_matrix_estimator = WEIGHT_MATRICES[weight_type]
        self._weight = weight_matrix_estimator(**weight_config)
        self._weight_type = weight_type
        self._weight_config = self._weight.config
        self._method = 'gmm'
        self._result_container = IVGMMResults

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
        """
        Estimate model parameters

        Parameters
        ----------
        iter_limit : int, optional
            Maximum number of iterations.  Default is 2, which produces
            two-step efficient GMM estimates.  Larger values can be used
            to iterate between parameter estimation and optimal weight
            matrix estimation until convergence.
        tol : float, optional
            Convergence criteria.  Measured as covariance normalized change in
            parameters across iterations where the covariance estimator is
            based on the first step parameter estimates.
        cov_type : str, optional
                Name of covariance estimator to use
        **cov_config
            Additional parameters to pass to covariance estimator

        Returns
        -------
        results : IVGMMResults
            Results container

        Notes
        -----
        Additional covariance parameters depend on specific covariance used.
        The see the docstring of specific covariance estimator for a list of
        supported options. Defaults are used if no covariance configuration
        is provided.
        
        Available covariance functions are:
    
          * 'unadjusted', 'homoskedastic' - Assumes moment conditions are 
            homoskedastic
          * 'robust', 'heteroskedastic' - Allows for heterosedasticity by not 
            autocorrelation
          * 'kernel' - Allows for heteroskedasticity and autocorrelation
          * 'cluster' - Allows for one-way cluster dependence
        """

        y, x, z = self._y, self._x, self._z
        nobs = y.shape[0]
        weight_matrix = self._weight.weight_matrix
        w = inv(z.T @ z / nobs)
        _params = params = self.estimate_parameters(x, y, z, w)
        eps = y - x @ params

        iters, norm = 1, 10 * tol
        while iters < iter_limit and norm > tol:
            w = inv(weight_matrix(x, z, eps))
            params = self.estimate_parameters(x, y, z, w)
            eps = y - x @ params
            delta = params - _params
            xpz = x.T @ z / nobs
            if iters == 1:
                v = (xpz @ w @ xpz.T) / nobs
                vinv = inv(v)
            _params = params
            norm = delta.T @ vinv @ delta
            iters += 1

        cov_estimator = IVGMMCovariance(x, y, z, params, w,
                                        cov_type, **cov_config)

        results = self._post_estimation(params, cov_estimator, cov_type)
        gmm_pe = self._gmm_post_estimation(params, w, iters)

        results.update(gmm_pe)

        return self._result_container(results, self)

    def _gmm_post_estimation(self, params, weight_mat, iters):
        """GMM-specific post-estimation results"""
        instr = self._instr_columns
        gmm_specific = {'weight_mat': DataFrame(weight_mat, columns=instr, index=instr),
                        'weight_type': self._weight_type,
                        'weight_config': self._weight_type,
                        'iterations': iters,
                        'j_stat': self._j_statistic(params, weight_mat)}

        return gmm_specific

    def _j_statistic(self, params, weight_mat):
        """J-stat and test"""
        y, x, z = self._y, self._x, self._z
        nobs, nvar, ninstr = y.shape[0], x.shape[1], z.shape[1]
        eps = y - x @ params
        g_bar = (z * eps).mean(0)
        stat = float(nobs * g_bar.T @ weight_mat @ g_bar.T)
        null = 'Expected moment conditions are equal to 0'
        return WaldTestStatistic(stat, null, ninstr - nvar)


class IVGMMCUE(IVGMM):
    r"""
    Estimation of IV models using continuously updating GMM

    Parameters
    ----------
    dependent : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous regressors  (nobs by nexog)
    endog : array-like
        Endogenous regressors (nobs by nendog)
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
      * 'robust', 'heteroskedastic' - Allows for heterosedasticity by not 
        autocorrelation
      * 'kernel' - Allows for heteroskedasticity and autocorrelation
      * 'cluster' - Allows for one-way cluster dependence
    
    In most circumstances, the ``center`` weight option should be ``True`` to
    avoid starting value dependence.
    
    .. math::
     
      \hat{\beta}_{cue} & =\min_{\beta}\bar{g}(\beta)'W(\beta)^{-1}g(\beta)\\
      g(\beta) & =n^{-1}\sum_{i=1}^{n}z_{i}(y_{i}-x_{i}\beta)
    
    where :math:`W(\beta)` is a weight matrix that depends on :math:`\beta`
    through :math:`\epsilon_i = y_i - x_i\beta`.
    
    .. todo ::
    
      * Mathematical notation
    """

    def __init__(self, dependent, exog, endog, instruments, weight_type='robust',
                 **weight_config):
        super(IVGMMCUE, self).__init__(dependent, exog, endog, instruments, weight_type,
                                       **weight_config)
        if 'center' not in weight_config:
            weight_config['center'] = True
        self._method = 'gmm-cue'

    def j(self, params, x, y, z):
        r"""
        Optimization target 
        
        Parameters
        ----------
        params : ndarray
            Parameter vector (nvar,)
        x : ndarray
            Regressor matrix (nobs by nvar)
        y : ndarray
            Regressand matrix (nobs by 1)
        z : ndarray
            Instrument matrix (nobs by ninstr)
        
        Returns
        -------
        j : float
            GMM objective function, also known as the J-statistic
        
        Notes
        -----
        
        The GMM objective function is defined as
        
        .. math::
        
          J(\beta) = \bar{g}(\beta)'W(\beta)^{-1}\bar{g}(\beta)
        
        where :math:`\bar{g}(\beta)` is the average of the moment 
        conditions, :math:`z_i \hat{\epsilon}_i`, where 
        :math:`\hat{\epsilon}_i = y_i - x_i\beta`.  The weighting matrix
        is some estimator of the long-run variance of the moment conditions.
        
        Unlike tradition GMM, the weighting matrix is simulteneously computed 
        with the moment conditions, and so has explicit dependence on 
        :math:`\beta`.
        """
        nobs = y.shape[0]
        weight_matrix = self._weight.weight_matrix
        eps = y - x @ params[:, None]
        w = inv(weight_matrix(x, z, eps))
        g_bar = (z * eps).mean(0)
        return nobs * g_bar.T @ w @ g_bar.T

    def estimate_parameters(self, x, y, z):
        r"""
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
        Exposed to facilitate estimation with other data, e.g., bootstrapped
        samples.  Performs no error checking.
        
        See Also
        --------
        scipy.optimize.minimize
        """
        res = IV2SLS(self.dependent, self.exog, self.endog, self.instruments).fit()
        sv = res.params
        args = (x, y, z)
        res = minimize(self.j, sv, args=args, options={'disp': False})

        return res.x[:, None], res.nit

    def fit(self, cov_type='robust', **cov_config):
        r"""
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator to use
        **cov_config
            Additional parameters to pass to covariance estimator

        Returns
        -------
        results : IVGMMResults
            Results container

        Notes
        -----
        Additional covariance parameters depend on specific covariance used.
        The see the docstring of specific covariance estimator for a list of
        supported options. Defaults are used if no covariance configuration
        is provided.
        
        Starting values are computed by IV2SLS.

        .. todo::

          * Expose method to pass optimization options
          * Allow starting values to be passed
        """

        y, x, z = self._y, self._x, self._z
        weight_matrix = self._weight.weight_matrix
        params, iters = self.estimate_parameters(x, y, z)
        eps = y - x @ params
        w = inv(weight_matrix(x, z, eps))

        cov_estimator = IVGMMCovariance(x, y, z, params, w, **cov_config)
        results = self._post_estimation(params, cov_estimator, cov_type)
        gmm_pe = self._gmm_post_estimation(params, w, iters)
        results.update(gmm_pe)

        return self._result_container(results, self)


class OLSResults(object):
    def __init__(self, results, model):
        self._resid = results['eps']
        self._params = results['params']
        self._cov = results['cov']
        self._model = model
        self._r2 = results['r2']
        self._cov_type = results['cov_type']
        self._rss = results['residual_ss']
        self._tss = results['total_ss']
        self._s2 = results['s2']
        self._debiased = results['debiased']
        self._f_statistic = results['fstat']
        self._vars = results['vars']
        self._cov_config = results['cov_config']
        self._method = results['method']
        self._kappa = results.get('kappa', None)
        self._cache = {}

    @property
    def cov_config(self):
        """Parameter values from covariance estimator"""
        return self._cov_config

    @property
    def cov_estimator(self):
        """Type of covariance estimator used to compute covariance"""
        return self._cov_type

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
        return self._resid

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
        return self._model._x.shape[1]

    @property
    def has_constant(self):
        """Flag indicating the model includes a constant or equivalent"""
        return self._model.has_constant

    @property
    def kappa(self):
        """k-class estimator value"""
        return self._kappa

    @property
    def rsquared(self):
        """Coefficient of determination (R**2)"""
        return self._r2

    @property
    def rsquared_adj(self):
        """Sample-size adjusted coefficient of determination (R**2)"""
        n, k, c = self.nobs, self.df_model, int(self.has_constant)
        return 1 - ((n - c) / (n - k)) * (1 - self._r2)

    @property
    def cov_type(self):
        """Covariance estimator used"""
        return self._cov_type

    @property
    def std_errors(self):
        """Estimated parameter standard errors"""
        std_errors = sqrt(diag(self.cov))
        return Series(std_errors, index=self._vars, name='stderr')

    @property
    def tstats(self):
        """Parameter t-statistics"""
        return self.params / self.std_errors

    @property
    def pvalues(self):
        """
        Parameter p-vals. Uses t(df_resid) if debiased is True, other normal.
        """
        if 'pvalues' not in self._cache:
            if self.debiased:
                pvals = 2 - 2 * stats.t.cdf(abs(self.tstats), self.df_resid)
            else:
                pvals = 2 - 2 * stats.norm.cdf(abs(self.tstats))
            self._cache['pvalues'] = Series(pvals, index=self._vars, name='pvalue')

        return self._cache['pvalues']

    @property
    def total_ss(self):
        """Total sum of squares"""
        return self._tss

    @property
    def model_ss(self):
        """Residual sum of squares"""
        return self._tss - self._rss

    @property
    def resid_ss(self):
        """Residual sum of squares"""
        return self._rss

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
        """Joint test of significance for non-constant regressors"""
        return self._f_statistic

    @property
    def method(self):
        """Method used to estimate model parameters"""
        return self._method

    def conf_int(self, level=0.95):
        """
        Confidence interval construction

        Parameters
        ----------
        level : float
            Confidence level for interval

        Returns
        -------
        ci : ndarray
            Confidence interval of the form [lower, upper] for each parameters
        """
        q = stats.norm.ppf([(1 - level) / 2, 1 - (1 - level) / 2])
        q = q[None, :]
        ci = self.params[:, None] + self.std_errors[:, None] * q
        return DataFrame(ci, index=self._vars, columns=['lower', 'upper'])


class _IVResults(OLSResults):
    """
    Results from IV estimation

    Notes
    -----
    .. todo::

        * Hypothesis testing
        * First stage diagnostics
        * Model diagnostics
    """

    def __init__(self, results, model):
        super(_IVResults, self).__init__(results, model)
        self._kappa = results.get('kappa', 1)

    @property
    def first_stage(self):
        """
        First stage regression results
        
        Returns
        -------
        first : FirstStageResults
            Object containing results for diagnosing instrument relevance issues.
        """
        # TODO: Need to have a custom object here!
        # TODO: cache shoudl be like sm

        return FirstStageResults(self._model.dependent, self._model.exog,
                                 self._model.endog, self._model.instruments,
                                 self._cov_type, self._cov_config)


class IVResults(_IVResults):
    """
    Results from IV estimation

    Notes
    -----
    .. todo::

        * Hypothesis testing
        * First stage diagnostics
        * Model diagnostics
    """

    def __init__(self, results, model):
        super(IVResults, self).__init__(results, model)
        self._kappa = results.get('kappa', 1)

    @property
    def sargan(self):
        """
        Sargan test of overidentifying restrictions
        """
        if 'sargan' in self._cache:
            return self._cache['sargan']
        z = self._model.instruments.ndarray
        nobs, ninstr = z.shape
        nendog = self._model.endog.shape[1]
        if ninstr - nendog == 0:
            import warnings
            warnings.warn('Sargan test requires more instruments than '
                          'endogenous variables',
                          UserWarning)
            return WaldTestStatistic(0, 'Test is not feasible.', 1)

        eps = self.resids.values[:, None]
        u = eps - z @ (pinv(z) @ eps)
        stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
        null = 'The model is not overidentified.'
        self._cache['sargan'] = WaldTestStatistic(stat, null, ninstr - nendog)
        return self._cache['sargan']

    @property
    def basmann(self):
        """
        Basmann's test of overidentifying restrictions
        """
        mod = self._model
        nobs, ninstr = mod.instruments.shape
        nendog = mod.endog.shape[1]
        nvar = mod.exog.shape[1] + nendog
        if ninstr - nendog == 0:
            import warnings
            warnings.warn('Basmann test requires more instruments than '
                          'endogenous variables',
                          UserWarning)
            return WaldTestStatistic(0, 'Test is not feasible.', 1)

        sargan_test = self.sargan
        s = sargan_test.stat
        stat = s * (nobs - ninstr) / (nobs - nvar)
        return WaldTestStatistic(stat, sargan_test.null, sargan_test.df)

    @property
    def durbin(self):
        exog_endog = c_[self._model.exog.ndarray, self._model.endog.ndarray]
        nendog = self._model.endog.shape[1]
        e_ols = _annihilate(self._model.dependent.ndarray, exog_endog)
        nobs = e_ols.shape[0]
        e_2sls = self.resids
        e_ols_pz = _proj(e_ols, self._model.instruments.ndarray)
        e_2sls_pz = _proj(e_2sls, self._model.instruments.ndarray)
        stat = e_ols_pz.T @ e_ols_pz - e_2sls_pz.T @ e_2sls_pz
        stat /= (e_ols.T @ e_ols) / nobs
        null = 'Endogenous variables are endogenous'
        return WaldTestStatistic(stat.squeeze(), null, nendog)

    def wu_hausman(self):
        # TODO
        pass

    def wooldridge(self):
        # TODO
        pass


class IVGMMResults(IVResults):
    """
    Results from GMM estimation of IV models

    Notes
    -----
    .. todo::

        * Hypothesis testing
        * First stage diagnostics
        * Model diagnostics
    """

    def __init__(self, results, model):
        super(IVGMMResults, self).__init__(results, model)
        self._weight_mat = results['weight_mat']
        self._weight_type = results['weight_type']
        self._weight_config = results['weight_config']
        self._iterations = results['iterations']
        self._j_stat = results['j_stat']

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

    @property
    def j_stat(self):
        """J-test of overidentifying restrictions"""
        return self._j_stat

    def c_stat(self):
        """C-test of endogeneity"""
        # TODO
        pass


class FirstStageResults(object):
    """
    .. todo ::
    
      * Docstrings
      * Summary
    """

    def __init__(self, dep, exog, endog, instr, cov_type, cov_config):
        self.dep = dep
        self.exog = exog
        self.endog = endog
        self.instr = instr
        reg = c_[self.exog.ndarray, self.endog.ndarray]
        self._reg = DataFrame(reg, columns=self.exog.cols + self.endog.cols)
        self._cov_type = cov_type
        self._cov_config = cov_config
        self._fitted = {}

    @property
    def rsquared(self):
        """
        Partial R2 - endog on instr, controlling for exog
        F-stat exog only, same reg
        F-pval exog only, same
        Shea's partial R2 -- 2SLS rsquare and homosk cov, OLS rsquare and homosk cov
        :return: 
        """
        endog, exog, instr = self.endog, self.exog, self.instr
        z = instr.ndarray
        x = exog.ndarray
        px = x @ pinv(x)
        ez = z - px @ z
        out = {}
        for col in endog.pandas:
            inner = {}
            inner['rsquared'] = self._fitted[col].rsquared
            y = endog.pandas[[col]].values
            ey = y - px @ y
            mod = IV2SLS(ey, ez, None, None)
            res = mod.fit(self._cov_type, **self._cov_config)
            inner['partial.rsquared'] = res.rsquared
            params = res.params.values
            params = params[:, None]
            stat = params.T @ inv(res.cov) @ params
            stat = stat.squeeze()
            w = WaldTestStatistic(stat, null='', df=params.shape[0])
            inner['f.stat'] = w.stat
            inner['f.pval'] = w.pval
            out[col] = Series(inner)
        out = DataFrame(out).T

        dep = self.dep
        r2sls = IV2SLS(dep, exog, endog, instr).fit('unadjusted')
        rols = IV2SLS(dep, self._reg, None, None).fit('unadjusted')
        shea = (rols.std_errors / r2sls.std_errors) ** 2
        shea *= (1 - r2sls.rsquared) / (1 - rols.rsquared)
        out['shea'] = shea[out.index]

        return out

    @property
    def individual(self):
        exog_instr = c_[self.exog.ndarray, self.instr.ndarray]
        if not self._fitted:
            for col in self.endog.pandas:
                mod = IV2SLS(self.endog.pandas[col], exog_instr, None, None)
                self._fitted[col] = mod.fit(self._cov_type, **self._cov_config)

        return self._fitted
