import numpy as np
from numpy.linalg import pinv, inv

from .data import PanelData
from .fixed_effects import EntityEffect, TimeEffect


class IVCovariance(object):
    def __init__(self, x, z, eps, **config):
        self.x = x
        self.z = z
        self.eps = eps
        self.config = config
        self._valid_config = []

    @property
    def cov(self):
        raise NotImplementedError('subclasses must implement')

    def check_config(self, **config):
        if len(config) == 0:
            return

        invalid = []
        for key in config:
            if key not in self._valid_config:
                invalid.append(key)
        if invalid:
            keys = ', '.join(config.keys())
            raise ValueError('Unexpected keywords in config: {0}'.format(keys))

        return


class HomoskedasticCovariance(IVCovariance):
    def __init__(self, x, z, eps, **config):
        super(HomoskedasticCovariance, self).__init__(x, z, eps, **config)
        self._valid_config = ['debiased']

    @property
    def cov(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        scale = 1 if self.config.get('debiased', False) else nobs / (nobs - nvar)
        s2 = eps.T @ eps / nobs
        pz = z @ pinv(z)
        v = x.T @ pz @ x
        # mz = eye(k) - pz
        return scale * s2 * inv(v)


class HeteroskedasticCovariance(HomoskedasticCovariance):
    def __init__(self, x, z, eps, **config):
        super(HeteroskedasticCovariance, self).__init__(x, z, eps, **config)

    @property
    def cov(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        scale = 1 if self.config.get('debiased', False) else nobs / (nobs - nvar)
        pz = z @ pinv(z)
        v = x.T @ pz @ x
        vinv = inv(v)
        xhat_e = pz @ x * eps
        s = xhat_e.T @ xhat_e / nobs

        return scale * vinv @ s @ vinv


COVARIANCE_ESTIMATORS = {'homoskedastic': HomoskedasticCovariance,
                         'robust': HeteroskedasticCovariance}


class PooledOLS(object):
    r"""
    Estimation of linear model with pooled parameters
    
    Parameters
    ----------
    endog: array-like
        Endogenous or left-hand-side variable (entities by time)
    exog: array-like
        Exogenous or right-hand-side variables (entities by time by variable). Should not contain 
        an intercept or have a constant column in the column span.
    intercept : bool, optional
        Flag whether to include an intercept in the model
    
    Notes
    -----
    The model is given by 
    
    .. math::
    
        y_{it}=\alpha+\beta^{\prime}x_{it}+\epsilon_{it}
    
    where :math:`\alpha` is omitted if ``intercept`` is ``False``.
    """

    def __init__(self, endog, exog, *, intercept=True):
        self.endog = PanelData(endog)
        self.exog = PanelData(exog)
        self.intercept = intercept

    def fit(self):
        y = self.endog.asnumpy2d
        x = self.exog.asnumpy2d
        if self.intercept:
            x = np.c_[np.ones((x.shape[0], 1)), x]
        return pinv(x) @ y


class PanelOLS(PooledOLS):
    r"""
    Parameters
    ----------
    endog: array-like
        Endogenous or left-hand-side variable (entities by time)
    exog: array-like
        Exogenous or right-hand-side variables (entities by time by variable). Should not contain 
        an intercept or have a constant column in the column span.
    intercept : bool, optional
        Flag whether to include an intercept in the model
    entity_effect : bool, optional
        Flag whether to include an intercept in the model
    time_effect : bool, optional
        Flag whether to include an intercept in the model

    Notes
    -----
    The model is given by 
    
    .. math::
    
        y_{it}=\alpha_i + \gamma_t +\beta^{\prime}x_{it}+\epsilon_{it}
    
    where :math:`\alpha_i` is omitted if ``entity_effect`` is ``False`` and
    :math:`\gamma_i` is omitted if ``time_effect`` is ``False``. If both ``entity_effect``  and
    ``time_effect`` are ``False``, the model reduces to :class:`PooledOLS`.
    """

    def __init__(self, endog, exog, *, intercept=True, entity_effect=False, time_effect=False):
        super(PanelOLS, self).__init__(endog, exog, intercept=intercept)
        if intercept and (entity_effect or time_effect):
            import warnings
            warnings.warn('Intercept must be False when using entity or time effects.')
            self.intercept = False
        self.entity_effect = entity_effect
        self.time_effect = time_effect

    def fit(self):
        y = self.endog.asnumpy2d
        x = self.exog.asnumpy2d
        if self.intercept:
            x = np.c_[np.ones((x.shape[0], 1)), x]
        if self.entity_effect:
            y = EntityEffect(y).orthogonalize()
            x = EntityEffect(x).orthogonalize()
        if self.time_effect:
            y = TimeEffect(y).orthogonalize()
            x = TimeEffect(x).orthogonalize()

        return pinv(x) @ y


class BetweenOLS(PooledOLS):
    r"""
    Parameters
    ----------
    endog: array-like
        Endogenous or left-hand-side variable (entities by time)
    exog: array-like
        Exogenous or right-hand-side variables (entities by time by variable). Should not contain 
        an intercept or have a constant column in the column span.
    intercept : bool, optional
        Flag whether to include an intercept in the model

    Notes
    -----
    The model is given by 
    
    .. math::
    
        \bar{y}_{i}=\alpha + \beta^{\prime}\bar{x}_{i}+\bar{\epsilon}_{i}
    
    where :math:`\alpha` is omitted if ``intercept`` is ``False`` and 
    :math:`\bar{z}` is the time-average. 
    """

    def __init__(self, endog, exog, *, intercept=True):
        super(BetweenOLS, self).__init__(endog, exog, intercept=intercept)

    def fit(self):
        y = self.endog.asnumpy3d.mean(axis=1)
        x = self.exog.asnumpy3d.mean(axis=1)
        if self.intercept:
            x = np.c_[np.ones((x.shape[0], 1)), x]

        return pinv(x) @ y


class FirstDifferenceOLS(PooledOLS):
    r"""
    Parameters
    ----------
    endog: array-like
        Endogenous or left-hand-side variable (entities by time)
    exog: array-like
        Exogenous or right-hand-side variables (entities by time by variable). Should not contain 
        an intercept or have a constant column in the column span.

    Notes
    -----
    The model is given by 

    .. math::

        \Delta y_{it}=\beta^{\prime}\Delta x_{it}+\Delta\epsilon_{it}
    """

    def __init__(self, endog, exog, *, intercept=True):
        super(FirstDifferenceOLS, self).__init__(endog, exog, intercept=intercept)

    def fit(self):
        y = np.diff(self.endog.asnumpy3d, axis=1)
        x = np.diff(self.exog.asnumpy3d, axis=1)
        n, t, k = self.exog.n, self.exog.t, self.exog.k
        y = y.reshape((n * (t - 1), 1))
        x = x.reshape((n * (t - 1), k))
        return pinv(x) @ y


class IV2SLS(object):
    """
    Parameters
    ----------
    endog : array-like
    exog : array-like
    instruments : array-like
    
    .. todo::
         * VCV: unadjusted, robust, cluster clustvar, bootstrap, jackknife, or hac kernel
         * small sample adjustments
         * Colinearity check
         

    """

    def __init__(self, endog, exog, instruments):
        self.endog = endog
        self.exog = exog
        self.instruments = instruments

        self._validate_inputs()

    def _validate_inputs(self):
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
        pz = z @ pinv(z)
        return (x.T @ pz @ x) @ (x.T @ pz @ y)

    def fit(self):
        y, x, z = self.endog, self.exog, self.instruments
        self.params = self.estimate_parameters(x, y, z)
        return self.params

    def cov(self, params):
        estimator = COVARIANCE_ESTIMATORS[self.cov_type]
        eps = self.resids(params)
        x, z = self.exog, self.instruments
        return estimator(x, z, eps, **self.cov_config).cov

    def resids(self, params):
        return self.endog - self.exog @ params

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
    
    nobs
    mss
    df
    residss
    df_r
    r2
    rbar2
    F
    chi2 -- what is this?
    kappa - for LIML
    J_stat - for GMM
    
    """
    pass