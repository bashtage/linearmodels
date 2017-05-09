import numpy as np
from numpy.linalg import pinv
from patsy.highlevel import dmatrix
from patsy.missing import NAAction
from scipy.optimize import minimize

from linearmodels.asset_pricing.results import GMMFactorModelResults, LinearFactorModelResults
from linearmodels.iv.covariance import KERNEL_LOOKUP, _cov_kernel
from linearmodels.iv.data import IVData
from linearmodels.utility import AttrDict, WaldTestStatistic, has_constant, matrix_rank, \
    missing_warning


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
            missing_warning(missing)
            self.portfolios.drop(missing)
            self.factors.drop(missing)

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
        >>> mod = TradedFactorModel.from_formula(formula, data)
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
        # TODO: Homoskedastic covariance

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
            xeex = xe.T @ xe / nobs
            rp_cov = fe.T @ fe / nobs
        elif cov_type == 'kernel':
            kernel = cov_config.get('kernel', 'bartlett')
            bw = cov_config.get('bandwidth', None)
            bw = int(np.ceil(4 * (nobs / 100) ** (2 / 9))) if bw is None else bw
            w = KERNEL_LOOKUP[kernel](bw, nobs - 1)
            xeex = _cov_kernel(xe, w)
            rp_cov = _cov_kernel(fe, w)
        else:
            raise ValueError('Unknown cov_type: {0}'.format(cov_type))
        debiased = int(bool(debiased))
        df = fc.shape[1]
        full_vcv = xpxi @ xeex @ xpxi / (nobs - debiased * df)
        vcv = full_vcv[:nloading, :nloading]
        rp_cov = rp_cov / (nobs - debiased)

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
                       cov_type=cov_type, model=self, nobs=nobs, rp_names=self.factors.cols)

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
    GLS estimation using ``sigma`` has not been implemented

    Suitable for traded or non-traded factors.

    Implements a 2-step estimator of risk premia, factor loadings and model
    tests.

    The first stage model estimated is

    .. math::

        r_{it}^e = a_i + f_t \beta_i + \epsilon_{it}

    where :math:`r_{it}^e` is the excess return on test portfolio i and
    :math:`f_t` are the traded factor returns.  The parameters :math:`a_i`
    are required to allow non-traded to be tested, but are not economically
    interesting.  These are not reported.

    The second stage model uses the estimated factor loadings from the first
    and is

    .. math::

        \bar{r}_i^e = \hat{\beta}_i^\prime \lambda + \eta_i

    where :math:`\bar{r}_i^e` is the average excess return to portfolio i.

    The model is tested using the estimated values
    :math:`\hat{\alpha}_i=\hat{\eta}_i`.
    """

    def __init__(self, portfolios, factors, *, risk_free=False, sigma=None):
        self._sigma = sigma
        self._risk_free = bool(risk_free)
        super(LinearFactorModel, self).__init__(portfolios, factors)

    def _validate_data(self):
        super(LinearFactorModel, self)._validate_data()
        f = self.factors.ndarray
        p = self.portfolios.ndarray
        nrp = (f.shape[1] + int(self._risk_free))
        if p.shape[0] < nrp:
            raise ValueError('Must have more observations then factors')
        if p.shape[1] < nrp:
            raise ValueError('The number of test portfolio must be at least as '
                             'large as the number of risk premia, including the '
                             'risk free rate if estimated.')

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
        >>> from linearmodels.asset_pricing import LinearFactorModel
        >>> data = french.load()
        >>> formula = 'S1M1 + S1M5 + S3M3 + S5M1 S5M5 ~ MktRF + SMB + HML'
        >>> mod = LinearFactorModel.from_formula(formula, data)
        
        Using only factors
        
        >>> portfolios = data[['S1M1', 'S1M5', 'S3M1', 'S3M5', 'S5M1', 'S5M5']]
        >>> formula = 'MktRF + SMB + HML'
        >>> mod = LinearFactorModel.from_formula(formula, data)
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

        mod = cls(portfolios, factors, risk_free=risk_free)
        mod.formula = orig_formula
        return mod

    def fit(self, cov_type='robust', **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator
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
        # TODO: Kernel estimator
        # TODO: Refactor commonalities in estimation
        excess_returns = not self._risk_free
        nrf = int(not excess_returns)
        f = self.factors.ndarray
        nobs, nfactor = f.shape
        p = self.portfolios.ndarray
        nportfolio = p.shape[1]

        # Step 1, n regressions to get B
        fc = np.c_[np.ones((nobs, 1)), f]
        b = np.linalg.lstsq(fc, p)[0]  # nf+1 by np
        eps = p - fc @ b
        if excess_returns:
            betas = b[1:].T
        else:
            betas = b.T.copy()
            betas[:, 0] = 1.0

        # Step 2, t regressions to get lambda(T)
        # if self._sigma is not None:
        #    vals, vecs = np.linalg.eigh(self._sigma)
        #    sigma_m12 = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        # else:
        #    sigma_m12 = np.eye(nportfolio)

        lam = np.linalg.lstsq(betas, p.mean(0)[:, None])[0]

        # Moments
        f_rep = np.tile(fc, (1, nportfolio))
        eps_rep = np.tile(eps, (nfactor + 1, 1))
        eps_rep = np.reshape(eps_rep.T, (nportfolio * (nfactor + 1), nobs)).T
        expected = betas @ lam
        pricing_errors = p - expected.T
        alphas = pricing_errors.mean(0)[:, None]

        moments = np.c_[f_rep * eps_rep, pricing_errors @ betas, pricing_errors - alphas.T]
        S = moments.T @ moments / nobs
        # Jacobian
        G = np.eye(S.shape[0])
        s2 = (nfactor + 1) * nportfolio
        s3 = s2 + (nfactor + nrf)
        fpf = fc.T @ fc / nobs
        G[:s2, :s2] = np.kron(np.eye(nportfolio), fpf)
        G[s2:s3, s2:s3] = betas.T @ betas
        for i in range(nportfolio):
            _lam = lam if excess_returns else lam[1:]
            block = betas[[i]].T @ _lam.T
            if excess_returns:
                block -= alphas[i] * np.eye(nfactor)
            else:
                block -= np.r_[np.zeros((1, nfactor)), alphas[i] * np.eye(nfactor)]

            block = np.c_[np.zeros((nfactor + nrf, 1)), block]
            G[s2:s3, (i * (nfactor + 1)):((i + 1) * (nfactor + 1))] = block
        zero_lam = np.r_[[[0]], lam] if excess_returns else np.r_[[[0]], lam[1:]]
        G[s3:, :s2] = np.kron(np.eye(nportfolio), zero_lam.T)
        Ginv = np.linalg.inv(G)
        # VCV
        full_vcv = Ginv @ S @ Ginv.T / nobs
        alpha_vcv = full_vcv[s3:, s3:]
        stat = float(alphas.T @ np.linalg.inv(alpha_vcv) @ alphas)
        jstat = WaldTestStatistic(stat, 'All alphas are 0', nportfolio - nfactor,
                                  name='J-statistic')

        total_ss = ((p - p.mean(0)[None, :]) ** 2).sum()
        residual_ss = (eps ** 2).sum()
        r2 = 1 - residual_ss / total_ss
        rp = lam
        rp_cov = full_vcv[s2:s3, s2:s3]
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
        order = np.reshape(np.arange(s2), (nportfolio, nfactor + 1))
        order[:, 0] = np.arange(s3, s3 + nportfolio)
        order = order.ravel()
        order = np.r_[order, s2:s3]
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
                       cov_type=cov_type, model=self, nobs=nobs, rp_names=rp_names)

        return LinearFactorModelResults(res)


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
        \epsilon_{t}\otimes\left[1\:f_{t}^{\prime}\right]^{\prime}\\
        f_{t}-\mu
        \end{array}\right]

    and

    .. math::

      \epsilon_{t}=r_{t}-\left[1_{N}\;\beta\right]\lambda-\beta\left(f_{t}-\mu\right)

    where :math:`r_{it}^e` is the excess return on test portfolio i and
    :math:`f_t` are the traded factor returns.

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
        >>> from linearmodels.asset_pricing import LinearFactorModel
        >>> data = french.load()
        >>> formula = 'S1M1 + S1M5 + S3M3 + S5M1 S5M5 ~ MktRF + SMB + HML'
        >>> mod = LinearFactorModel.from_formula(formula, data)

        Using only factors
        
        >>> portfolios = data[['S1M1', 'S1M5', 'S3M1', 'S3M5', 'S5M1', 'S5M5']]
        >>> formula = 'MktRF + SMB + HML'
        >>> mod = LinearFactorModel.from_formula(formula, data)
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
            suppress iterative output
        max_iter : int, positive, optional
            Maximum number of iterations when minimizing objective
        cov_type : str, optional
            Name of covariance estimator
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
        # TODO: allow different weights type
        w = np.linalg.inv(g.T @ g / nobs)
        args = (excess_returns, w)

        # 2. Step 1 using w = inv(s) from SV
        callback = callback_factory(self._j, args, disp=disp)
        res = minimize(self._j, sv, args=args, callback=callback,
                       options={'disp': bool(disp), 'maxiter': max_iter})
        params = res.x
        last_obj = res.fun
        # 3. Step 2 using step 1 estimates
        if not use_cue:
            # TODO: Add convergence criteria
            for i in range(steps - 1):
                g = self._moments(params, excess_returns)
                g -= g.mean(0)[None, :] if center else 0
                w = np.linalg.inv(g.T @ g / nobs)
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
            args = (excess_returns, center)
            obj = self._j_cue
            callback = callback_factory(obj, args, disp=disp)
            res = minimize(obj, params, args=args, callback=callback,
                           options={'disp': bool(disp), 'maxiter': max_iter})
            params = res.x

        # 4. Compute final S and G for inference
        g = self._moments(params, excess_returns)
        s = g.T @ g / nobs
        jac = self._jacobian(params, excess_returns)

        full_vcv = np.linalg.inv(jac.T @ np.linalg.inv(s) @ jac) / nobs
        rp = params[(n * k):(n * k + k + nrf)]
        rp_cov = full_vcv[(n * k):(n * k + k + nrf), (n * k):(n * k + k + nrf)]
        alphas = g.mean(0)[0:(n * (k + 1)):(k + 1), None]
        alpha_vcv = s[0:(n * (k + 1)):(k + 1), 0:(n * (k + 1)):(k + 1)] / nobs  # TODO: Fix this
        stat = self._j(params, excess_returns, w)
        jstat = WaldTestStatistic(stat, 'All alphas are 0', n, name='J-statistic')

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
        rp_names = param_names[(n * k):(n * k + k + nrf)]
        params = np.c_[alphas, betas]
        # 5. Return values
        res = AttrDict(params=params, cov=full_vcv, betas=betas, rp=rp, rp_cov=rp_cov,
                       alphas=alphas, alpha_vcv=alpha_vcv, jstat=jstat,
                       rsquared=r2, total_ss=total_ss, residual_ss=residual_ss,
                       param_names=param_names, portfolio_names=self.portfolios.cols,
                       factor_names=self.factors.cols, name=self._name,
                       cov_type=cov_type, model=self, nobs=nobs, rp_names=rp_names)

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

    def _j_cue(self, parameters, excess_returns, center):
        """CUE Objective function"""
        g = self._moments(parameters, excess_returns)
        gbar = g.mean(0)[:, None]
        nobs = self.portfolios.shape[0]
        if center:
            g -= gbar.T
        w = np.linalg.inv(g.T @ g / nobs)
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
