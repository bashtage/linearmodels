import numpy as np
from numpy.linalg import pinv

from linearmodels.asset_pricing.results import LinearFactorModelResults
from linearmodels.iv.covariance import KERNEL_LOOKUP, _cov_kernel
from linearmodels.iv.data import IVData
from linearmodels.utility import AttrDict, WaldTestStatistic


class TradedFactorModel(object):
    r"""Linear factor models estimator applicable to traded factors
    
    Parameters
    ----------
    portfolios : array-like
        Test portfolio returns (nobs by nportfolio)
    factors : array-like
        Priced factor returns (nobs by nfactor 
   
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

    def fit(self, cov_type='robust', debiased=True, **cov_config):
        """
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
        Supported covariance estiamtors are:
        
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
                       cov_type=cov_type, model=self, nobs=nobs)

        return LinearFactorModelResults(res)


class LinearFactorModel(TradedFactorModel):
    r"""Linear factor model estimator 

    Parameters
    ----------
    portfolios : array-like
        Test portfolio returns (nobs by nportfolio)
    factors : array-like
        Priced factorreturns (nobs by nfactor 
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

    def __init__(self, portfolios, factors, *, sigma=None):
        super(LinearFactorModel, self).__init__(portfolios, factors)
        self._sigma = sigma

    def fit(self, cov_type='robust', **cov_config):
        """
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
        f = self.factors.ndarray
        nobs, nfactor = f.shape
        p = self.portfolios.ndarray
        nportfolio = p.shape[1]

        # Step 1, n regressions to get B
        fc = np.c_[np.ones((nobs, 1)), f]
        b = np.linalg.lstsq(fc, p)[0]  # nf+1 by np
        eps = p - fc @ b
        betas = b[1:].T
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
        s1, s2 = 0, (nfactor + 1) * nportfolio,
        s3 = s2 + nfactor
        fpf = fc.T @ fc / nobs
        G[:s2, :s2] = np.kron(np.eye(nportfolio), fpf)
        G[s2:s3, s2:s3] = betas.T @ betas
        zero_lam = np.r_[[[0]], lam]
        for i in range(nportfolio):
            block = betas[[i]].T @ lam.T - alphas[i] * np.eye(nfactor)
            block = np.c_[np.zeros((nfactor, 1)), block]
            G[s2:s3, (i * (nfactor + 1)):((i + 1) * (nfactor + 1))] = block
        G[s3:, :s2] = np.kron(np.eye(nportfolio), zero_lam.T)
        Ginv = np.linalg.inv(G)
        # VCV
        full_vcv = Ginv @ S @ Ginv.T / nobs
        alpha_vcv = full_vcv[s3:, s3:]
        stat = float(alphas.T @ np.linalg.inv(alpha_vcv) @ alphas)
        jstat = WaldTestStatistic(stat, 'All alphas are 0', nportfolio - nfactor,
                                  name='J-statistic')

        total_ss = ((p - p.mean(0)[None,:]) ** 2).sum()
        residual_ss = (eps ** 2).sum()
        r2 = 1 - residual_ss / total_ss
        rp = lam
        rp_cov = full_vcv[s2:s3, s2:s3]
        params = np.c_[alphas, betas]
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.append('alpha-{0}'.format(portfolio))
            for factor in self.factors.cols:
                param_names.append('beta-{0}-{1}'.format(portfolio, factor))
        for factor in self.factors.cols:
            param_names.append('lambda-{0}'.format(factor))
        # Pivot vcv to remove unnecessary and have correct order
        order = np.reshape(np.arange(s2), (nportfolio, nfactor + 1))
        order[:, 0] = np.arange(s3, s3 + nportfolio)
        order = order.ravel()
        order = np.r_[order, s2:s3]
        full_vcv = full_vcv[order][:, order]
        res = AttrDict(params=params, cov=full_vcv, betas=betas, rp=rp, rp_cov=rp_cov,
                       alphas=alphas, alpha_vcv=alpha_vcv, jstat=jstat,
                       rsquared=r2, total_ss=total_ss, residual_ss=residual_ss,
                       param_names=param_names, portfolio_names=self.portfolios.cols,
                       factor_names=self.factors.cols, name=self._name,
                       cov_type=cov_type, model=self, nobs=nobs)

        return LinearFactorModelResults(res)


class LinearFactorModelGMM(TradedFactorModel):
    """GMM estimation of asset pricing linear factor model"""

    def __init__(self, factors, portfolios, *, constant=True):
        super(LinearFactorModelGMM, self).__init__(factors, portfolios, constant=constant)

    def fit(self):
        raise NotImplementedError()
