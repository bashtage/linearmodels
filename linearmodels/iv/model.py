"""
Instrumental variable estimators
"""
from __future__ import absolute_import, division, print_function

from numpy import (any, array, asarray, average, c_, isscalar, logical_not,
                   ones, sqrt, nanmean)
from numpy.linalg import eigvalsh, inv, matrix_rank, pinv
from pandas import DataFrame, Series
from scipy.optimize import minimize

from linearmodels.iv._utility import parse_formula
from linearmodels.iv.covariance import (ClusteredCovariance,
                                        HeteroskedasticCovariance,
                                        HomoskedasticCovariance,
                                        KernelCovariance)
from linearmodels.iv.data import IVData
from linearmodels.iv.gmm import (HeteroskedasticWeightMatrix,
                                 HomoskedasticWeightMatrix, IVGMMCovariance,
                                 KernelWeightMatrix,
                                 OneWayClusteredWeightMatrix)
from linearmodels.iv.results import IVGMMResults, IVResults, OLSResults
from linearmodels.utility import (WaldTestStatistic, has_constant, inv_sqrth,
                                  missing_warning)

__all__ = ['COVARIANCE_ESTIMATORS', 'WEIGHT_MATRICES', 'IVGMM', 'IVLIML', 'IV2SLS',
           'IVGMMCUE', '_OLS']

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
                         'one-way': ClusteredCovariance,
                         'clustered': ClusteredCovariance,
                         'OneWayClusteredCovariance': ClusteredCovariance}

WEIGHT_MATRICES = {'unadjusted': HomoskedasticWeightMatrix,
                   'homoskedastic': HomoskedasticWeightMatrix,
                   'robust': HeteroskedasticWeightMatrix,
                   'heteroskedastic': HeteroskedasticWeightMatrix,
                   'kernel': KernelWeightMatrix,
                   'clustered': OneWayClusteredWeightMatrix,
                   'one-way': OneWayClusteredWeightMatrix
                   }


class IVLIML(object):
    r"""
    Limited information ML and k-class estimation of IV models

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
    weights : array-like, optional
        Observation weights used in estimation
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

    See Also
    --------
    IV2SLS, IVGMM, IVGMMCUE
    """

    def __init__(self, dependent, exog, endog, instruments, *, weights=None,
                 fuller=0, kappa=None):

        self.dependent = IVData(dependent, var_name='dependent')
        nobs = self.dependent.shape[0]
        self.exog = IVData(exog, var_name='exog', nobs=nobs)
        self.endog = IVData(endog, var_name='endog', nobs=nobs)
        self.instruments = IVData(instruments, var_name='instruments', nobs=nobs)
        if weights is None:
            weights = ones(self.dependent.shape)
        weights = IVData(weights).ndarray
        if any(weights <= 0):
            raise ValueError('weights must be strictly positive.')
        weights = weights / nanmean(weights)
        self.weights = IVData(weights, var_name='weights', nobs=nobs)

        self._drop_locs = self._drop_missing()
        # dependent variable
        w = sqrt(self.weights.ndarray)
        self._y = self.dependent.ndarray
        self._wy = self._y * w
        # model regressors
        self._x = c_[self.exog.ndarray, self.endog.ndarray]
        self._wx = self._x * w
        # first-stage regressors
        self._z = c_[self.exog.ndarray, self.instruments.ndarray]
        self._wz = self._z * w

        self._has_constant = False
        self._regressor_is_exog = array([True] * self.exog.shape[1] +
                                        [False] * self.endog.shape[1])
        self._columns = self.exog.cols + self.endog.cols
        self._instr_columns = self.exog.cols + self.instruments.cols
        self._index = self.dependent.rows

        self._validate_inputs()
        if not hasattr(self, '_method'):
            self._method = 'IV-LIML'
            additional = []
            if fuller != 0:
                additional.append('fuller(alpha={0})'.format(fuller))
            if kappa is not None:
                additional.append('kappa={0}'.format(fuller))
            if additional:
                self._method += '(' + ', '.join(additional) + ')'
        if not hasattr(self, '_result_container'):
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
                          'simultaneously.  Identical results can be computed '
                          'using kappa only', UserWarning)
        if endog is None or instruments is None:
            self._result_container = OLSResults
            self._method = 'OLS'
        self._formula = None

    @staticmethod
    def from_formula(formula, data, *, weights=None, fuller=0, kappa=None):
        """
        Parameters
        ----------
        formula : str
            Patsy formula modified for the IV syntax described in the notes
            section
        data : DataFrame
            DataFrame containing the variables used in the formula
        weights : array-like, optional
            Observation weights used in estimation
        fuller : float, optional
            Fuller's alpha to modify LIML estimator. Default returns unmodified
            LIML estimator.
        kappa : float, optional
            Parameter value for k-class estimation.  If not provided, computed to
            produce LIML parameter estimate.

        Returns
        -------
        model : IVLIML
            Model instance

        Notes
        -----
        The IV formula modifies the standard Patsy formula to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure is `dependent ~ exog [endog ~ instruments]` and it must
        be the case that the formula expressions constructed from blocks
        `dependent ~ exog endog` and `dependent ~ exog instruments` are both
        valid Patsy formulas.

        A constant must be explicitly included using '1 +' if required.

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IVLIML
        >>> data = wage.load()
        >>> formula = 'np.log(wage) ~ 1 + exper + exper ** 2 + brthord + [educ ~ sibs]'
        >>> mod = IVLIML.from_formula(formula, data)
        """
        dep, exog, endog, instr = parse_formula(formula, data)
        mod = IVLIML(dep, exog, endog, instr, weights=weights,
                     fuller=fuller, kappa=kappa)
        mod.formula = formula
        return mod

    @property
    def formula(self):
        """Formula used to create the model"""
        return self._formula

    @formula.setter
    def formula(self, value):
        """Formula used to create the model"""
        self._formula = value

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
            raise ValueError('regressors [exog endog] do not have full '
                             'column rank')
        if matrix_rank(z) < z.shape[1]:
            raise ValueError('instruments [exog instruments]  do not have '
                             'full column rank')
        self._has_constant, self._const_loc = has_constant(x)

    def _drop_missing(self):
        data = (self.dependent, self.exog, self.endog, self.instruments, self.weights)
        missing = any(c_[[dh.isnull for dh in data]], 0)
        if any(missing):
            if all(missing):
                raise ValueError('All observations contain missing data. '
                                 'Model cannot be estimated.')
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self.endog.drop(missing)
            self.instruments.drop(missing)
            self.weights.drop(missing)

        missing_warning(missing)
        return missing

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
            Parameter value for k-class estimator

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

    def _estimate_kappa(self):
        y, x, z = self._wy, self._wx, self._wz
        is_exog = self._regressor_is_exog
        e = c_[y, x[:, ~is_exog]]
        x1 = x[:, is_exog]
        if x1.shape[1] == 0:
            # No exogenous regressors
            return 1

        ez = e - z @ (pinv(z) @ e)
        ex1 = e - x1 @ (pinv(x1) @ e)

        vpmzv_sqinv = inv_sqrth(ez.T @ ez)
        q = vpmzv_sqinv @ (ex1.T @ ex1) @ vpmzv_sqinv
        return min(eigvalsh(q))

    def fit(self, *, cov_type='robust', **cov_config):
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
        wy, wx, wz = self._wy, self._wx, self._wz
        liml_kappa = self._estimate_kappa()
        kappa = self._kappa
        if kappa is None:
            kappa = liml_kappa

        if self._fuller != 0:
            nobs, ninstr = wz.shape
            kappa -= self._fuller / (nobs - ninstr)

        params = self.estimate_parameters(wx, wy, wz, kappa)

        cov_estimator = COVARIANCE_ESTIMATORS[cov_type]
        cov_config['kappa'] = kappa
        cov_config_copy = {k: v for k, v in cov_config.items()}
        if 'center' in cov_config_copy:
            del cov_config_copy['center']
        cov_estimator = cov_estimator(wx, wy, wz, params, **cov_config_copy)

        results = {'kappa': kappa,
                   'liml_kappa': liml_kappa}
        pe = self._post_estimation(params, cov_estimator, cov_type)
        results.update(pe)

        return self._result_container(results, self)

    def wresids(self, params):
        """
        Compute weighted model residuals

        Parameters
        ----------
        params : ndarray
            Model parameters (nvar by 1)

        Returns
        -------
        wresids : ndarray
            Weighted model residuals

        Notes
        -----
        Uses weighted versions of data instead of raw data.  Identical to
        resids if all weights are unity.
        """
        return self._wy - self._wx @ params

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

    @property
    def isnull(self):
        """Locations of observations with missing values"""
        return self._drop_locs

    @property
    def notnull(self):
        """Locations of observations included in estimation"""
        return logical_not(self._drop_locs)

    def _f_statistic(self, params, cov, debiased):
        non_const = ~(self._x.ptp(0) == 0)
        test_params = params[non_const]
        test_cov = cov[non_const][:, non_const]
        test_stat = test_params.T @ inv(test_cov) @ test_params
        test_stat = float(test_stat)
        nobs, nvar = self._x.shape
        null = 'All parameters ex. constant are zero'
        name = 'Model F-statistic'
        df = test_params.shape[0]
        if debiased:
            wald = WaldTestStatistic(test_stat / df, null, df, nobs - nvar,
                                     name=name)
        else:
            wald = WaldTestStatistic(test_stat, null, df, name=name)

        return wald

    def _post_estimation(self, params, cov_estimator, cov_type):
        vars = self._columns
        index = self._index
        eps = self.resids(params)
        weps = self.wresids(params)
        cov = cov_estimator.cov
        debiased = cov_estimator.debiased

        residual_ss = (weps.T @ weps)

        w = self.weights.ndarray
        e = self._wy
        if self.has_constant:
            e = e - sqrt(self.weights.ndarray) * average(self._y, weights=w)

        total_ss = float(e.T @ e)
        r2 = 1 - residual_ss / total_ss

        fstat = self._f_statistic(params, cov, debiased)
        out = {'params': Series(params.squeeze(), vars, name='parameter'),
               'eps': Series(eps.squeeze(), index=index, name='residual'),
               'weps': Series(weps.squeeze(), index=index, name='weighted residual'),
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
               'method': self._method,
               'cov_estimator': cov_estimator}

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
    weights : array-like, optional
        Observation weights used in estimation

    Notes
    -----
    The 2SLS estimator is defined

    .. math::

      \hat{\beta}_{2SLS} & =(X'Z(Z'Z)^{-1}Z'X)^{-1}X'Z(Z'Z)^{-1}Z'Y\\
                         & =(\hat{X}'\hat{X})^{-1}\hat{X}'Y\\
                 \hat{X} & =Z(Z'Z)^{-1}Z'X

    The 2SLS estimator is a special case of a k-class estimator with
    :math:`\kappa=1`,

    .. todo::

      * VCV: bootstrap

    See Also
    --------
    IVLIML, IVGMM, IVGMMCUE
    """

    def __init__(self, dependent, exog, endog, instruments, *, weights=None):
        self._method = 'IV-2SLS'
        super(IV2SLS, self).__init__(dependent, exog, endog, instruments,
                                     weights=weights, fuller=0, kappa=1)

    @staticmethod
    def from_formula(formula, data, *, weights=None):
        """
        Parameters
        ----------
        formula : str
            Patsy formula modified for the IV syntax described in the notes
            section
        data : DataFrame
            DataFrame containing the variables used in the formula
        weights : array-like, optional
            Observation weights used in estimation

        Returns
        -------
        model : IV2SLS
            Model instance

        Notes
        -----
        The IV formula modifies the standard Patsy formula to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure is `dependent ~ exog [endog ~ instruments]` and it must
        be the case that the formula expressions constructed from blocks
        `dependent ~ exog endog` and `dependent ~ exog instruments` are both
        valid Patsy formulas.

        A constant must be explicitly included using '1 +' if required.

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IV2SLS
        >>> data = wage.load()
        >>> formula = 'np.log(wage) ~ 1 + exper + exper ** 2 + brthord + [educ ~ sibs]'
        >>> mod = IV2SLS.from_formula(formula, data)
        """
        dep, exog, endog, instr = parse_formula(formula, data)
        mod = IV2SLS(dep, exog, endog, instr, weights=weights)
        mod.formula = formula
        return mod


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
    weights : array-like, optional
        Observation weights used in estimation
    weight_type : str
        Name of moment condition weight function to use in the GMM estimation
    **weight_config
        Additional keyword arguments to pass to the moment condition weight
        function

    Notes
    -----
    Available GMM weight functions are:

    * 'unadjusted', 'homoskedastic' - Assumes moment conditions are
      homoskedastic
    * 'robust', 'heteroskedastic' - Allows for heteroskedasticity by not
      autocorrelation
    * 'kernel' - Allows for heteroskedasticity and autocorrelation
    * 'cluster' - Allows for one-way cluster dependence

    The estimator is defined as

    .. math::

      \hat{\beta}_{gmm}=(X'ZW^{-1}Z'X)^{-1}X'ZW^{-1}Z'Y

    where :math:`W` is a positive definite weight matrix and :math:`Z`
    contains both the exogenous regressors and the instruments.

    .. todo::

      * VCV: bootstrap

    See Also
    --------
    IV2SLS, IVLIML, IVGMMCUE
    """

    def __init__(self, dependent, exog, endog, instruments, *, weights=None,
                 weight_type='robust', **weight_config):
        self._method = 'IV-GMM'
        self._result_container = IVGMMResults
        super(IVGMM, self).__init__(dependent, exog, endog, instruments, weights=weights)
        weight_matrix_estimator = WEIGHT_MATRICES[weight_type]
        self._weight = weight_matrix_estimator(**weight_config)
        self._weight_type = weight_type
        self._weight_config = self._weight.config

    @staticmethod
    def from_formula(formula, data, *, weights=None, weight_type='robust', **weight_config):
        """
        Parameters
        ----------
        formula : str
            Patsy formula modified for the IV syntax described in the notes
            section
        data : DataFrame
            DataFrame containing the variables used in the formula
        weights : array-like, optional
            Observation weights used in estimation
        weight_type : str
            Name of moment condition weight function to use in the GMM estimation
        **weight_config
            Additional keyword arguments to pass to the moment condition weight
            function

        Notes
        -----
        The IV formula modifies the standard Patsy formula to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure is `dependent ~ exog [endog ~ instruments]` and it must
        be the case that the formula expressions constructed from blocks
        `dependent ~ exog endog` and `dependent ~ exog instruments` are both
        valid Patsy formulas.

        A constant must be explicitly included using '1 +' if required.

        Returns
        -------
        model : IVGMM
            Model instance

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IVGMM
        >>> data = wage.load()
        >>> formula = 'np.log(wage) ~ 1 + exper + exper ** 2 + brthord + [educ ~ sibs]'
        >>> mod = IVGMM.from_formula(formula, data)
        """
        dep, exog, endog, instr = parse_formula(formula, data)
        mod = IVGMM(dep, exog, endog, instr, weights=weights, weight_type=weight_type,
                    **weight_config)
        mod.formula = formula
        return mod

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

    def fit(self, *, iter_limit=2, tol=1e-4, initial_weight=None,
            cov_type='robust', **cov_config):
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
        initial_weight : ndarray, optional
            Initial weighting matrix to use in the first step.  If not
            specified, uses the average outer-product of the set containing
            the exogenous variables and instruments.
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
        * 'robust', 'heteroskedastic' - Allows for heteroskedasticity by not
          autocorrelation
        * 'kernel' - Allows for heteroskedasticity and autocorrelation
        * 'cluster' - Allows for one-way cluster dependence
        """
        wy, wx, wz = self._wy, self._wx, self._wz
        nobs = wy.shape[0]
        weight_matrix = self._weight.weight_matrix
        wmat = inv(wz.T @ wz / nobs) if initial_weight is None else initial_weight
        sv = IV2SLS(self.dependent, self.exog, self.endog, self.instruments,
                    weights=self.weights)
        _params = params = sv.fit().params.values[:, None]
        # _params = params = self.estimate_parameters(wx, wy, wz, wmat)

        iters, norm = 1, 10 * tol
        while iters < iter_limit and norm > tol:
            eps = wy - wx @ params
            wmat = inv(weight_matrix(wx, wz, eps))
            params = self.estimate_parameters(wx, wy, wz, wmat)
            delta = params - _params
            if iters == 1:
                xpz = wx.T @ wz / nobs
                v = (xpz @ wmat @ xpz.T) / nobs
                vinv = inv(v)
            _params = params
            norm = delta.T @ vinv @ delta
            iters += 1

        cov_estimator = IVGMMCovariance(wx, wy, wz, params, wmat,
                                        cov_type, **cov_config)

        results = self._post_estimation(params, cov_estimator, cov_type)
        gmm_pe = self._gmm_post_estimation(params, wmat, iters)

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
        """J stat and test"""
        y, x, z = self._wy, self._wx, self._wz
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
    weights : array-like, optional
        Observation weights used in estimation
    weight_type : str
        Name of moment condition weight function to use in the GMM estimation
    **weight_config
        Additional keyword arguments to pass to the moment condition weight
        function

    Notes
    -----
    Available weight functions are:

    * 'unadjusted', 'homoskedastic' - Assumes moment conditions are
      homoskedastic
    * 'robust', 'heteroskedastic' - Allows for heteroskedasticity by not
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

    See Also
    --------
    IV2SLS, IVLIML, IVGMM
    """

    def __init__(self, dependent, exog, endog, instruments, *, weights=None,
                 weight_type='robust', **weight_config):
        self._method = 'IV-GMM-CUE'
        super(IVGMMCUE, self).__init__(dependent, exog, endog, instruments, weights=weights,
                                       weight_type=weight_type, **weight_config)
        if 'center' not in weight_config:
            weight_config['center'] = True

    @staticmethod
    def from_formula(formula, data, *, weights=None, weight_type='robust', **weight_config):
        """
        Parameters
        ----------
        formula : str
            Patsy formula modified for the IV syntax described in the notes
            section
        data : DataFrame
            DataFrame containing the variables used in the formula
        weights : array-like, optional
            Observation weights used in estimation
        weight_type : str
            Name of moment condition weight function to use in the GMM estimation
        **weight_config
            Additional keyword arguments to pass to the moment condition weight
            function

        Returns
        -------
        model : IVGMMCUE
            Model instance

        Notes
        -----
        The IV formula modifies the standard Patsy formula to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure is `dependent ~ exog [endog ~ instruments]` and it must
        be the case that the formula expressions constructed from blocks
        `dependent ~ exog endog` and `dependent ~ exog instruments` are both
        valid Patsy formulas.

        A constant must be explicitly included using '1 +' if required.

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IVGMMCUE
        >>> data = wage.load()
        >>> formula = 'np.log(wage) ~ 1 + exper + exper ** 2 + brthord + [educ ~ sibs]'
        >>> mod = IVGMMCUE.from_formula(formula, data)
        """
        dep, exog, endog, instr = parse_formula(formula, data)
        mod = IVGMMCUE(dep, exog, endog, instr, weights=weights, weight_type=weight_type,
                       **weight_config)
        mod.formula = formula
        return mod

    def j(self, params, x, y, z):
        r"""
        Optimization target

        Parameters
        ----------
        params : ndarray
            Parameter vector (nvar)
        x : ndarray
            Regressor matrix (nobs by nvar)
        y : ndarray
            Regressand matrix (nobs by 1)
        z : ndarray
            Instrument matrix (nobs by ninstr)

        Returns
        -------
        j : float
            GMM objective function, also known as the J statistic

        Notes
        -----

        The GMM objective function is defined as

        .. math::

          J(\beta) = \bar{g}(\beta)'W(\beta)^{-1}\bar{g}(\beta)

        where :math:`\bar{g}(\beta)` is the average of the moment
        conditions, :math:`z_i \hat{\epsilon}_i`, where
        :math:`\hat{\epsilon}_i = y_i - x_i\beta`.  The weighting matrix
        is some estimator of the long-run variance of the moment conditions.

        Unlike tradition GMM, the weighting matrix is simultaneously computed
        with the moment conditions, and so has explicit dependence on
        :math:`\beta`.
        """
        nobs = y.shape[0]
        weight_matrix = self._weight.weight_matrix
        eps = y - x @ params[:, None]
        w = inv(weight_matrix(x, z, eps))
        g_bar = (z * eps).mean(0)
        return nobs * g_bar.T @ w @ g_bar.T

    def estimate_parameters(self, starting, x, y, z, display=False):
        r"""
        Parameters
        ----------
        starting : ndarray
            Starting values for the optimization
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
        args = (x, y, z)
        res = minimize(self.j, starting, args=args, options={'disp': display})

        return res.x[:, None], res.nit

    def fit(self, *, starting=None, display=False, cov_type='robust', **cov_config):
        r"""
        Estimate model parameters

        Parameters
        ----------
        starting : ndarray, optional
            Starting values to use in optimization.  If not provided, 2SLS
            estimates are used.
        display : bool, optional
            Flag indicating whether to display optimization output
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

        Starting values are computed by IVGMM.

        .. todo::

          * Expose method to pass optimization options
        """

        wy, wx, wz = self._wy, self._wx, self._wz
        weight_matrix = self._weight.weight_matrix
        if starting is None:
            exog = None if self.exog.shape[1] == 0 else self.exog
            endog = None if self.endog.shape[1] == 0 else self.endog
            instr = None if self.instruments.shape[1] == 0 else \
                self.instruments

            res = IVGMM(self.dependent, exog, endog, instr,
                        weights=self.weights, weight_type=self._weight_type,
                        **self._weight_config).fit()
            starting = res.params.values
        else:
            starting = asarray(starting)
            if len(starting) != self.exog.shape[1] + self.endog.shape[1]:
                raise ValueError('starting does not have the correct number '
                                 'of values')
        params, iters = self.estimate_parameters(starting, wx, wy, wz, display)
        eps = wy - wx @ params
        wmat = inv(weight_matrix(wx, wz, eps))

        cov_estimator = IVGMMCovariance(wx, wy, wz, params, wmat, cov_type,
                                        **cov_config)
        results = self._post_estimation(params, cov_estimator, cov_type)
        gmm_pe = self._gmm_post_estimation(params, wmat, iters)
        results.update(gmm_pe)

        return self._result_container(results, self)


class _OLS(IVLIML):
    """
    Computes OLS estimated when required

    Parameters
    ----------
    dependent : array-like
        Endogenous variables (nobs by 1)
    exog : array-like
        Exogenous regressors  (nobs by nexog)
    weights : array-like, optional
        Observation weights used in estimation

    Notes
    -----
    Uses IV2SLS internally by setting endog and instruments to None.
    Uses IVLIML with kappa=0 to estimate OLS models.

    See Also
    --------
    statsmodels.regression.linear_model.OLS,
    statsmodels.regression.linear_model.GLS
    """

    def __init__(self, dependent, exog, *, weights=None):
        super(_OLS, self).__init__(dependent, exog, None, None, weights=weights, kappa=0.0)
        self._result_container = OLSResults
