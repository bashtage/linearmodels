"""
Linear factor models for applications in asset pricing
"""
import numpy as np
from numpy.linalg import pinv
from patsy.highlevel import dmatrix
from patsy.missing import NAAction
from scipy.optimize import minimize

from linearmodels.asset_pricing.covariance import (HeteroskedasticCovariance,
                                                   HeteroskedasticWeight,
                                                   KernelCovariance,
                                                   KernelWeight)
from linearmodels.asset_pricing.results import (GMMFactorModelResults,
                                                LinearFactorModelResults)
from linearmodels.iv.data import IVData
from linearmodels.utility import (AttrDict, WaldTestStatistic, has_constant,
                                  matrix_rank, missing_warning)


def callback_factory(obj, args, disp=1):
    d = {'iter': 0}
    disp = int(disp)

    def callback(params):
        fval = obj(params, *args)
        if disp > 0 and (d['iter'] % disp == 0):
            print('Iteration: {0}, Objective: {1}'.format(d['iter'], fval))
        d['iter'] += 1

    return callback


class TradedFactorModel(object):
    r"""Linear factor models estimator applicable to traded factors

    Parameters
    ----------
    portfolios : array-like
        Test portfolio returns (nobs by nportfolio)
    factors : array-like
        Priced factor returns (nobs by nfactor)

    Notes
    -----
    Implements both time-series estimators of risk premia, factor loadings
    and zero-alpha tests.

    The model estimated is

    .. math::

        r_{it}^e = \alpha_i + f_t \beta_i + \epsilon_{it}

    where :math:`r_{it}^e` is the excess return on test portfolio i and
    :math:`f_t` are the traded factor returns.  The model is directly
    tested using the estimated values :math:`\hat{\alpha}_i`. Risk premia,
    :math:`\lambda_i` are estimated using the sample averages of the factors,
    which must be excess returns on traded portfolios.
    """

    def __init__(self, portfolios, factors):
        self.portfolios = IVData(portfolios, var_name='portfolio')
        self.factors = IVData(factors, var_name='factor')
        self._name = self.__class__.__name__
        self._formula = None
        self._validate_data()

    def _drop_missing(self):
        data = (self.portfolios, self.factors)
        missing = np.any(np.c_[[dh.isnull for dh in data]], 0)
        if any(missing):
            if all(missing):
                raise ValueError('All observations contain missing data. '
                                 'Model cannot be estimated.')
            self.portfolios.drop(missing)
            self.factors.drop(missing)
        missing_warning(missing)

        return missing

    def _validate_data(self):
        p = self.portfolios.ndarray
        f = self.factors.ndarray
        if p.shape[0] != f.shape[0]:
            raise ValueError('The number of observations in portfolios and '
                             'factors is not the same.')
        self._drop_missing()

        p = self.portfolios.ndarray
        f = self.factors.ndarray
        if has_constant(p)[0]:
            raise ValueError('portfolios must not contains a constant or equivalent.')
        if has_constant(f)[0]:
            raise ValueError('factors must not contain a constant or equivalent.')
        if matrix_rank(f) < f.shape[1]:
            raise ValueError('Model cannot be estimated. factors do not have full column rank.')
        if matrix_rank(p) < p.shape[1]:
            raise ValueError('Model cannot be estimated. portfolios do not have full column rank.')

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        self._formula = value

    @classmethod
    def from_formula(cls, formula, data, *, portfolios=None):
        """
        Parameters
        ----------
        formula : str
            Patsy formula modified for the syntax described in the notes
        data : DataFrame
            DataFrame containing the variables used in the formula
        portfolios : array-like, optional
            Portfolios to be used in the model

        Returns
        -------
        model : TradedFactorModel
            Model instance

        Notes
        -----
        The formula can be used in one of two ways.  The first specified only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specified the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

        Examples
        --------
        >>> from linearmodels.datasets import french
        >>> from linearmodels.asset_pricing import TradedFactorModel
        >>> data = french.load()
        >>> formula = 'S1M1 + S1M5 + S3M3 + S5M1 S5M5 ~ MktRF + SMB + HML'
        >>> mod = TradedFactorModel.from_formula(formula, data)

        Using only factors

        >>> portfolios = data[['S1M1', 'S1M5', 'S3M1', 'S3M5', 'S5M1', 'S5M5']]
        >>> formula = 'MktRF + SMB + HML'
        >>> mod = TradedFactorModel.from_formula(formula, data, portfolios=portfolios)
        """
        na_action = NAAction(on_NA='raise', NA_types=[])
        orig_formula = formula
        if portfolios is not None:
            factors = dmatrix(formula + ' + 0', data, return_type='dataframe', NA_action=na_action)
        else:
            formula = formula.split('~')
            portfolios = dmatrix(formula[0].strip() + ' + 0', data,
                                 return_type='dataframe', NA_action=na_action)
            factors = dmatrix(formula[1].strip() + ' + 0', data,
                              return_type='dataframe', NA_action=na_action)
        mod = cls(portfolios, factors)
        mod.formula = orig_formula
        return mod

    def fit(self, cov_type='robust', debiased=True, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator
        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment
        **cov_config : dict
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        results : LinearFactorModelResults
            Results class with parameter estimates, covariance and test statistics

        Notes
        -----
        Supported covariance estimators are:

        * 'robust' - Heteroskedasticity-robust covariance estimator
        * 'kernel' - Heteroskedasticity and Autocorrelation consistent (HAC)
          covariance estimator

        The kernel covariance estimator takes the optional arguments
        ``kernel``, one of 'bartlett', 'parzen' or 'qs' (quadratic spectral)
        and ``bandwidth`` (a positive integer).
        """
        p = self.portfolios.ndarray
        f = self.factors.ndarray
        nportfolio = p.shape[1]
        nobs, nfactor = f.shape
        fc = np.c_[np.ones((nobs, 1)), f]
        rp = f.mean(0)[:, None]
        fe = f - f.mean(0)
        b = pinv(fc) @ p
        eps = p - fc @ b
        alphas = b[:1].T

        nloading = (nfactor + 1) * nportfolio
        xpxi = np.eye(nloading + nfactor)
        xpxi[:nloading, :nloading] = np.kron(np.eye(nportfolio), pinv(fc.T @ fc / nobs))
        f_rep = np.tile(fc, (1, nportfolio))
        eps_rep = np.tile(eps, (nfactor + 1, 1))  # 1 2 3 ... 25 1 2 3 ...
        eps_rep = eps_rep.ravel(order='F')
        eps_rep = np.reshape(eps_rep, (nobs, (nfactor + 1) * nportfolio), order='F')
        xe = f_rep * eps_rep
        xe = np.c_[xe, fe]
        if cov_type in ('robust', 'heteroskedastic'):
            cov_est = HeteroskedasticCovariance(xe, inv_jacobian=xpxi, center=False,
                                                debiased=debiased, df=fc.shape[1])
            rp_cov_est = HeteroskedasticCovariance(fe, jacobian=np.eye(f.shape[1]), center=False,
                                                   debiased=debiased, df=1)
        elif cov_type == 'kernel':
            cov_est = KernelCovariance(xe, inv_jacobian=xpxi, center=False, debiased=debiased,
                                       df=fc.shape[1], **cov_config)
            bw = cov_est.bandwidth
            _cov_config = {k: v for k, v in cov_config.items()}
            _cov_config['bandwidth'] = bw
            rp_cov_est = KernelCovariance(fe, jacobian=np.eye(f.shape[1]), center=False,
                                          debiased=debiased, df=1, **_cov_config)
        else:
            raise ValueError('Unknown cov_type: {0}'.format(cov_type))
        full_vcv = cov_est.cov
        rp_cov = rp_cov_est.cov
        vcv = full_vcv[:nloading, :nloading]

        # Rearrange VCV
        order = np.reshape(np.arange((nfactor + 1) * nportfolio), (nportfolio, nfactor + 1))
        order = order.T.ravel()
        vcv = vcv[order][:, order]

        # Return values
        alpha_vcv = vcv[:nportfolio, :nportfolio]
        stat = float(alphas.T @ pinv(alpha_vcv) @ alphas)
        jstat = WaldTestStatistic(stat, 'All alphas are 0', nportfolio, name='J-statistic')
        params = b.T
        betas = b[1:].T
        residual_ss = (eps ** 2).sum()
        e = p - p.mean(0)[None, :]
        total_ss = (e ** 2).sum()
        r2 = 1 - residual_ss / total_ss
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.append('alpha-{0}'.format(portfolio))
            for factor in self.factors.cols:
                param_names.append('beta-{0}-{1}'.format(portfolio, factor))
        for factor in self.factors.cols:
            param_names.append('lambda-{0}'.format(factor))

        res = AttrDict(params=params, cov=full_vcv, betas=betas, rp=rp, rp_cov=rp_cov,
                       alphas=alphas, alpha_vcv=alpha_vcv, jstat=jstat,
                       rsquared=r2, total_ss=total_ss, residual_ss=residual_ss,
                       param_names=param_names, portfolio_names=self.portfolios.cols,
                       factor_names=self.factors.cols, name=self._name,
                       cov_type=cov_type, model=self, nobs=nobs, rp_names=self.factors.cols,
                       cov_est=cov_est)

        return LinearFactorModelResults(res)


class LinearFactorModel(TradedFactorModel):
    r"""Linear factor model estimator

    Parameters
    ----------
    portfolios : array-like
        Test portfolio returns (nobs by nportfolio)
    factors : array-like
        Priced factor returns (nobs by nfactor)
    risk_free : bool, optional
        Flag indicating whether the risk-free rate should be estimated
        from returns along other risk premia.  If False, the returns are
        assumed to be excess returns using the correct risk-free rate.
    sigma : array-like, optional
        Positive definite residual covariance (nportfolio by nportfolio)

    Notes
    -----
    Suitable for traded or non-traded factors.

    Implements a 2-step estimator of risk premia, factor loadings and model
    tests.

    The first stage model estimated is

    .. math::

        r_{it} = c_i + f_t \beta_i + \epsilon_{it}

    where :math:`r_{it}` is the return on test portfolio i and
    :math:`f_t` are the traded factor returns.  The parameters :math:`c_i`
    are required to allow non-traded to be tested, but are not economically
    interesting.  These are not reported.

    The second stage model uses the estimated factor loadings from the first
    and is

    .. math::

        \bar{r}_i = \lambda_0 + \hat{\beta}_i^\prime \lambda + \eta_i

    where :math:`\bar{r}_i` is the average excess return to portfolio i and
    :math:`\lambda_0` is only included if estimating the risk-free rate. GLS
    is used in the second stage if ``sigma`` is provided.

    The model is tested using the estimated values
    :math:`\hat{\alpha}_i=\hat{\eta}_i`.
    """

    def __init__(self, portfolios, factors, *, risk_free=False, sigma=None):
        self._risk_free = bool(risk_free)
        super(LinearFactorModel, self).__init__(portfolios, factors)
        if sigma is None:
            self._sigma_m12 = self._sigma_inv = self._sigma = np.eye(self.portfolios.shape[1])
        else:
            self._sigma = np.asarray(sigma)
            vals, vecs = np.linalg.eigh(sigma)
            self._sigma_m12 = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
            self._sigma_inv = np.linalg.inv(self._sigma)

    def _validate_data(self):
        super(LinearFactorModel, self)._validate_data()
        f = self.factors.ndarray
        p = self.portfolios.ndarray
        nrp = (f.shape[1] + int(self._risk_free))
        if p.shape[1] < nrp:
            raise ValueError('The number of test portfolio must be at least as '
                             'large as the number of risk premia, including the '
                             'risk free rate if estimated.')

    @classmethod
    def from_formula(cls, formula, data, *, portfolios=None, risk_free=False,
                     sigma=None):
        """
        Parameters
        ----------
        formula : str
            Patsy formula modified for the syntax described in the notes
        data : DataFrame
            DataFrame containing the variables used in the formula
        portfolios : array-like, optional
            Portfolios to be used in the model. If provided, must use formula
            syntax containing only factors.
        risk_free : bool, optional
            Flag indicating whether the risk-free rate should be estimated
            from returns along other risk premia.  If False, the returns are
            assumed to be excess returns using the correct risk-free rate.
        sigma : array-like, optional
            Positive definite residual covariance (nportfolio by nportfolio)

        Returns
        -------
        model : LinearFactorModel
            Model instance

        Notes
        -----
        The formula can be used in one of two ways.  The first specified only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specified the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

        Examples
        --------
        >>> from linearmodels.datasets import french
        >>> from linearmodels.asset_pricing import LinearFactorModel
        >>> data = french.load()
        >>> formula = 'S1M1 + S1M5 + S3M3 + S5M1 S5M5 ~ MktRF + SMB + HML'
        >>> mod = LinearFactorModel.from_formula(formula, data)

        Using only factors

        >>> portfolios = data[['S1M1', 'S1M5', 'S3M1', 'S3M5', 'S5M1', 'S5M5']]
        >>> formula = 'MktRF + SMB + HML'
        >>> mod = LinearFactorModel.from_formula(formula, data, portfolios=portfolios)
        """
        na_action = NAAction(on_NA='raise', NA_types=[])
        orig_formula = formula
        if portfolios is not None:
            factors = dmatrix(formula + ' + 0', data, return_type='dataframe', NA_action=na_action)
        else:
            formula = formula.split('~')
            portfolios = dmatrix(formula[0].strip() + ' + 0', data,
                                 return_type='dataframe', NA_action=na_action)
            factors = dmatrix(formula[1].strip() + ' + 0', data,
                              return_type='dataframe', NA_action=na_action)
        if sigma is not None:
            mod = cls(portfolios, factors, risk_free=risk_free, sigma=sigma)
        else:
            mod = cls(portfolios, factors, risk_free=risk_free)
        mod.formula = orig_formula
        return mod

    def fit(self, cov_type='robust', debiased=True, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator
        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        results : LinearFactorModelResults
            Results class with parameter estimates, covariance and test statistics

        Notes
        -----
        The kernel covariance estimator takes the optional arguments
        ``kernel``, one of 'bartlett', 'parzen' or 'qs' (quadratic spectral)
        and ``bandwidth`` (a positive integer).
        """
        nobs, nf, nport, nrf, s1, s2, s3 = self._boundaries()
        excess_returns = not self._risk_free
        f = self.factors.ndarray
        p = self.portfolios.ndarray
        nport = p.shape[1]

        # Step 1, n regressions to get B
        fc = np.c_[np.ones((nobs, 1)), f]
        b = np.linalg.lstsq(fc, p)[0]  # nf+1 by np
        eps = p - fc @ b
        if excess_returns:
            betas = b[1:].T
        else:
            betas = b.T.copy()
            betas[:, 0] = 1.0

        sigma_m12 = self._sigma_m12
        lam = np.linalg.lstsq(sigma_m12 @ betas, sigma_m12 @ p.mean(0)[:, None])[0]
        expected = betas @ lam
        pricing_errors = p - expected.T
        # Moments
        alphas = pricing_errors.mean(0)[:, None]
        moments = self._moments(eps, betas, lam, alphas, pricing_errors)
        # Jacobian
        jacobian = self._jacobian(betas, lam, alphas)

        if cov_type not in ('robust', 'heteroskedastic', 'kernel'):
            raise ValueError('Unknown weight: {0}'.format(cov_type))
        if cov_type in ('robust', 'heteroskedastic'):
            cov_est = HeteroskedasticCovariance
        else:  # 'kernel':
            cov_est = KernelCovariance
        cov_est = cov_est(moments, jacobian=jacobian, center=False,
                          debiased=debiased, df=fc.shape[1], **cov_config)

        # VCV
        full_vcv = cov_est.cov
        alpha_vcv = full_vcv[s2:, s2:]
        stat = float(alphas.T @ np.linalg.pinv(alpha_vcv) @ alphas)
        jstat = WaldTestStatistic(stat, 'All alphas are 0', nport - nf - nrf,
                                  name='J-statistic')

        total_ss = ((p - p.mean(0)[None, :]) ** 2).sum()
        residual_ss = (eps ** 2).sum()
        r2 = 1 - residual_ss / total_ss
        rp = lam
        rp_cov = full_vcv[s1:s2, s1:s2]
        betas = betas if excess_returns else betas[:, 1:]
        params = np.c_[alphas, betas]
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.append('alpha-{0}'.format(portfolio))
            for factor in self.factors.cols:
                param_names.append('beta-{0}-{1}'.format(portfolio, factor))
        if not excess_returns:
            param_names.append('lambda-risk_free')
        for factor in self.factors.cols:
            param_names.append('lambda-{0}'.format(factor))

        # Pivot vcv to remove unnecessary and have correct order
        order = np.reshape(np.arange(s1), (nport, nf + 1))
        order[:, 0] = np.arange(s2, s3)
        order = order.ravel()
        order = np.r_[order, s1:s2]
        full_vcv = full_vcv[order][:, order]
        factor_names = list(self.factors.cols)
        rp_names = factor_names[:]
        if not excess_returns:
            rp_names.insert(0, 'risk_free')
        res = AttrDict(params=params, cov=full_vcv, betas=betas, rp=rp, rp_cov=rp_cov,
                       alphas=alphas, alpha_vcv=alpha_vcv, jstat=jstat,
                       rsquared=r2, total_ss=total_ss, residual_ss=residual_ss,
                       param_names=param_names, portfolio_names=self.portfolios.cols,
                       factor_names=factor_names, name=self._name,
                       cov_type=cov_type, model=self, nobs=nobs, rp_names=rp_names,
                       cov_est=cov_est)

        return LinearFactorModelResults(res)

    def _boundaries(self):
        nobs, nf = self.factors.ndarray.shape
        nport = self.portfolios.ndarray.shape[1]
        nrf = int(bool(self._risk_free))

        s1 = (nf + 1) * nport
        s2 = s1 + (nf + nrf)
        s3 = s2 + nport

        return nobs, nf, nport, nrf, s1, s2, s3

    def _jacobian(self, betas, lam, alphas):
        nobs, nf, nport, nrf, s1, s2, s3 = self._boundaries()
        f = self.factors.ndarray
        fc = np.c_[np.ones((nobs, 1)), f]
        excess_returns = not self._risk_free
        bc = betas
        sigma_inv = self._sigma_inv

        jac = np.eye((nport * (nf + 1)) + (nf + nrf) + nport)
        fpf = fc.T @ fc / nobs
        jac[:s1, :s1] = np.kron(np.eye(nport), fpf)

        b_tilde = sigma_inv @ bc
        alpha_tilde = sigma_inv @ alphas
        _lam = lam if excess_returns else lam[1:]
        for i in range(nport):
            block = np.zeros((nf + nrf, nf + 1))
            block[:, 1:] = b_tilde[[i]].T @ _lam.T
            block[nrf:, 1:] -= alpha_tilde[i] * np.eye(nf)
            jac[s1:s2, (i * (nf + 1)):((i + 1) * (nf + 1))] = block
        jac[s1:s2, s1:s2] = bc.T @ sigma_inv @ bc
        zero_lam = np.r_[[[0]], _lam]
        jac[s2:s3, :s1] = np.kron(np.eye(nport), zero_lam.T)
        jac[s2:s3, s1:s2] = bc

        return jac

    def _moments(self, eps, betas, lam, alphas, pricing_errors):
        sigma_inv = self._sigma_inv

        f = self.factors.ndarray
        nobs, nf, nport, nrf, s1, s2, s3 = self._boundaries()
        fc = np.c_[np.ones((nobs, 1)), f]
        # Moments
        f_rep = np.tile(fc, (1, nport))
        eps_rep = np.tile(eps, (nf + 1, 1))
        eps_rep = np.reshape(eps_rep.T, (nport * (nf + 1), nobs)).T

        g1 = f_rep * eps_rep
        g2 = pricing_errors @ sigma_inv @ betas
        g3 = pricing_errors - alphas.T

        return np.c_[g1, g2, g3]


class LinearFactorModelGMM(LinearFactorModel):
    r"""GMM estimator of Linear factor models

    Parameters
    ----------
    portfolios : array-like
        Test portfolio returns (nobs by nportfolio)
    factors : array-like
        Priced factors values (nobs by nfactor)
    risk_free : bool, optional
        Flag indicating whether the risk-free rate should be estimated
        from returns along other risk premia.  If False, the returns are
        assumed to be excess returns using the correct risk-free rate.

    Notes
    -----
    Suitable for traded or non-traded factors.

    Implements a GMM estimator of risk premia, factor loadings and model
    tests.

    The moments are

    .. math::

        \left[\begin{array}{c}
           \epsilon_{t}\otimes f_{c,t}\\
           f_{t}-\mu
        \end{array}\right]

    and

    .. math::

      \epsilon_{t}=r_{t}-\left[1_{N}\;\beta\right]\lambda-\beta\left(f_{t}-\mu\right)

    where :math:`r_{it}` is the return on test portfolio i and
    :math:`f_t` are the factor returns.

    The model is tested using the optimized objective function using the
    usual GMM J statistic.
    """

    def __init__(self, factors, portfolios, *, risk_free=False):
        super(LinearFactorModelGMM, self).__init__(factors, portfolios, risk_free=risk_free)

    @classmethod
    def from_formula(cls, formula, data, *, portfolios=None, risk_free=False):
        """
        Parameters
        ----------
        formula : str
            Patsy formula modified for the syntax described in the notes
        data : DataFrame
            DataFrame containing the variables used in the formula
        portfolios : array-like, optional
            Portfolios to be used in the model. If provided, must use formula
            syntax containing only factors.
        risk_free : bool, optional
            Flag indicating whether the risk-free rate should be estimated
            from returns along other risk premia.  If False, the returns are
            assumed to be excess returns using the correct risk-free rate.

        Returns
        -------
        model : LinearFactorModelGMM
            Model instance

        Notes
        -----
        The formula can be used in one of two ways.  The first specified only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specified the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

        Examples
        --------
        >>> from linearmodels.datasets import french
        >>> from linearmodels.asset_pricing import LinearFactorModel
        >>> data = french.load()
        >>> formula = 'S1M1 + S1M5 + S3M3 + S5M1 S5M5 ~ MktRF + SMB + HML'
        >>> mod = LinearFactorModel.from_formula(formula, data)

        Using only factors

        >>> portfolios = data[['S1M1', 'S1M5', 'S3M1', 'S3M5', 'S5M1', 'S5M5']]
        >>> formula = 'MktRF + SMB + HML'
        >>> mod = LinearFactorModel.from_formula(formula, data, portfolios=portfolios)
        """
        return super(LinearFactorModelGMM, cls).from_formula(formula, data,
                                                             portfolios=portfolios,
                                                             risk_free=risk_free)

    def fit(self, center=True, use_cue=False, steps=2, disp=10, max_iter=1000,
            cov_type='robust', debiased=True, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        center : bool, optional
            Flag indicating to center the moment conditions before computing
            the weighting matrix.
        use_cue : bool, optional
            Flag indicating to use continuously updating estimator
        steps : int, optional
            Number of steps to use when estimating parameters.  2 corresponds
            to the standard efficient gmm estimator. Higher values will
            iterate until convergence or up to the number of steps given
        disp : int, optional
            Number of iterations between printed update. 0 or negative values
            suppresses output
        max_iter : int, positive, optional
            Maximum number of iterations when minimizing objective
        cov_type : str, optional
            Name of covariance estimator
        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        results : GMMFactorModelResults
            Results class with parameter estimates, covariance and test statistics

        Notes
        -----
        The kernel covariance estimator takes the optional arguments
        ``kernel``, one of 'bartlett', 'parzen' or 'qs' (quadratic spectral)
        and ``bandwidth`` (a positive integer).
        """

        nobs, n = self.portfolios.shape
        k = self.factors.shape[1]
        excess_returns = not self._risk_free
        nrf = int(not bool(excess_returns))
        # 1. Starting Values - use 2 pass
        mod = LinearFactorModel(self.portfolios, self.factors, risk_free=self._risk_free)
        res = mod.fit()
        betas = res.betas.values.ravel()
        lam = res.risk_premia.values
        mu = self.factors.ndarray.mean(0)
        sv = np.r_[betas, lam, mu][:, None]
        g = self._moments(sv, excess_returns)
        g -= g.mean(0)[None, :] if center else 0
        if cov_type not in ('robust', 'heteroskedastic', 'kernel'):
            raise ValueError('Unknown weight: {0}'.format(cov_type))
        if cov_type in ('robust', 'heteroskedastic'):
            weight_est = HeteroskedasticWeight
            cov_est = HeteroskedasticCovariance
        else:  # 'kernel':
            weight_est = KernelWeight
            cov_est = KernelCovariance
        weight_est = weight_est(g, center=center, **cov_config)
        w = weight_est.w(g)

        args = (excess_returns, w)

        # 2. Step 1 using w = inv(s) from SV
        callback = callback_factory(self._j, args, disp=disp)
        res = minimize(self._j, sv, args=args, callback=callback,
                       options={'disp': bool(disp), 'maxiter': max_iter})
        params = res.x
        last_obj = res.fun
        iters = 1
        # 3. Step 2 using step 1 estimates
        if not use_cue:
            while iters < steps:
                iters += 1
                g = self._moments(params, excess_returns)
                w = weight_est.w(g)
                args = (excess_returns, w)

                # 2. Step 1 using w = inv(s) from SV
                callback = callback_factory(self._j, args, disp=disp)
                res = minimize(self._j, params, args=args, callback=callback,
                               options={'disp': bool(disp), 'maxiter': max_iter})
                params = res.x
                obj = res.fun
                if np.abs(obj - last_obj) < 1e-6:
                    break
                last_obj = obj

        else:
            args = (excess_returns, weight_est)
            obj = self._j_cue
            callback = callback_factory(obj, args, disp=disp)
            res = minimize(obj, params, args=args, callback=callback,
                           options={'disp': bool(disp), 'maxiter': max_iter})
            params = res.x

        # 4. Compute final S and G for inference
        g = self._moments(params, excess_returns)
        s = g.T @ g / nobs
        jac = self._jacobian(params, excess_returns)

        cov_est = cov_est(g, jacobian=jac, center=center, debiased=debiased,
                          df=self.factors.shape[1], **cov_config)

        full_vcv = cov_est.cov
        sel = slice((n * k), (n * k + k + nrf))
        rp = params[sel]
        rp_cov = full_vcv[sel, sel]
        sel = slice(0, (n * (k + 1)), (k + 1))
        alphas = g.mean(0)[sel, None]
        alpha_vcv = s[sel, sel] / nobs
        stat = self._j(params, excess_returns, w)
        jstat = WaldTestStatistic(stat, 'All alphas are 0', n - k - nrf, name='J-statistic')

        # R2 calculation
        betas = np.reshape(params[:(n * k)], (n, k))
        resids = self.portfolios.ndarray - self.factors.ndarray @ betas.T
        resids -= resids.mean(0)[None, :]
        residual_ss = (resids ** 2).sum()
        total = self.portfolios.ndarray
        total = total - total.mean(0)[None, :]
        total_ss = (total ** 2).sum()
        r2 = 1.0 - residual_ss / total_ss
        param_names = []
        for portfolio in self.portfolios.cols:
            for factor in self.factors.cols:
                param_names.append('beta-{0}-{1}'.format(portfolio, factor))
        if not excess_returns:
            param_names.append('lambda-risk_free')
        param_names.extend(['lambda-{0}'.format(f) for f in self.factors.cols])
        param_names.extend(['mu-{0}'.format(f) for f in self.factors.cols])
        rp_names = list(self.factors.cols)[:]
        if not excess_returns:
            rp_names.insert(0, 'risk_free')
        params = np.c_[alphas, betas]
        # 5. Return values
        res = AttrDict(params=params, cov=full_vcv, betas=betas, rp=rp, rp_cov=rp_cov,
                       alphas=alphas, alpha_vcv=alpha_vcv, jstat=jstat,
                       rsquared=r2, total_ss=total_ss, residual_ss=residual_ss,
                       param_names=param_names, portfolio_names=self.portfolios.cols,
                       factor_names=self.factors.cols, name=self._name,
                       cov_type=cov_type, model=self, nobs=nobs, rp_names=rp_names,
                       iter=iters, cov_est=cov_est)

        return GMMFactorModelResults(res)

    def _moments(self, parameters, excess_returns):
        """Calculate nobs by nmoments moment condifions"""
        nrf = int(not excess_returns)
        p = self.portfolios.ndarray
        nobs, n = p.shape
        f = self.factors.ndarray
        k = f.shape[1]
        s1, s2 = n * k, n * k + k + nrf
        betas = parameters[:s1]
        lam = parameters[s1:s2]
        mu = parameters[s2:]
        betas = np.reshape(betas, (n, k))
        expected = np.c_[np.ones((n, nrf)), betas] @ lam
        fe = f - mu.T
        eps = p - expected.T - fe @ betas.T
        f = np.c_[np.ones((nobs, 1)), f]
        f = np.tile(f, (1, n))
        eps = np.reshape(np.tile(eps, (k + 1, 1)).T, (n * (k + 1), nobs)).T
        g = np.c_[eps * f, fe]
        return g

    def _j(self, parameters, excess_returns, w):
        """Objective function"""
        g = self._moments(parameters, excess_returns)
        nobs = self.portfolios.shape[0]
        gbar = g.mean(0)[:, None]
        return nobs * float(gbar.T @ w @ gbar)

    def _j_cue(self, parameters, excess_returns, weight_est):
        """CUE Objective function"""
        g = self._moments(parameters, excess_returns)
        gbar = g.mean(0)[:, None]
        nobs = self.portfolios.shape[0]
        w = weight_est.w(g)
        return nobs * float(gbar.T @ w @ gbar)

    def _jacobian(self, params, excess_returns):
        """Jacobian matrix for inference"""
        nobs, k = self.factors.shape
        n = self.portfolios.shape[1]
        nrf = int(bool(not excess_returns))
        jac = np.zeros((n * k + n + k, params.shape[0]))
        s1, s2 = (n * k), (n * k) + k + nrf
        betas = params[:s1]
        betas = np.reshape(betas, (n, k))
        lam = params[s1:s2]
        mu = params[-k:]
        lam_tilde = lam if excess_returns else lam[1:]
        f = self.factors.ndarray
        fe = f - mu.T + lam_tilde.T
        f_aug = np.c_[np.ones((nobs, 1)), f]
        fef = f_aug.T @ fe / nobs
        r1 = n * (k + 1)
        jac[:r1, :s1] = np.kron(np.eye(n), fef)

        jac12 = np.zeros((r1, (k + nrf)))
        jac13 = np.zeros((r1, k))
        iota = np.ones((nobs, 1))
        for i in range(n):
            if excess_returns:
                b = betas[[i]]
            else:
                b = np.c_[[1], betas[[i]]]
            jac12[(i * (k + 1)):(i + 1) * (k + 1)] = f_aug.T @ (iota @ b) / nobs

            b = betas[[i]]
            jac13[(i * (k + 1)):(i + 1) * (k + 1)] = -f_aug.T @ (iota @ b) / nobs
        jac[:r1, s1:s2] = jac12
        jac[:r1, s2:] = jac13
        jac[-k:, -k:] = np.eye(k)

        return jac
