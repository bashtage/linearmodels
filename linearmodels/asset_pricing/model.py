import numpy as np
from linearmodels.asset_pricing.results import TradedFactorModelResults
from linearmodels.iv.covariance import KERNEL_LOOKUP, _cov_kernel
from linearmodels.iv.data import IVData
from linearmodels.utility import AttrDict, WaldTestStatistic
from numpy.linalg import pinv


class TradedFactorModel(object):
    r"""Asset pricing linear factor models estimator for traded factors
    
    Parameters
    ----------
    portfolios : array-like
        nobs by nportfolio array of test portfolios
    factors : array-like
        nobs by nfactor array of priced factors
   
    Notes
    -----
    Implements both time-series estimators of risk premia, factor loadings 
    and zero-alpha tests.
    
    The model estimated is 
    
    .. math::
    
        r_{p,it}^e = \alpha_i + f_t \beta_i + \epsilon_{it}
    
    where :math:`r_{p,it}^e` is the excess return to test portfolio i.  The
    model is directly tested using the estimated values
    :\math:`\hat{\alpha}_i`. Risk premia, :math:`\lambda_i` are estimated 
    using the sample averages of the factors, which must be excess returns on 
    traded portfolios.
    """

    def __init__(self, portfolios, factors):
        self.portfolios = IVData(portfolios, var_name='portfolio')
        self.factors = IVData(factors, var_name='factor')
        self._name = self.__class__.__name__

    def fit(self, cov_type='robust', debiased=True, **cov_config):
        """
        Parameters
        ----------
        cov_type : {'robust', 'kernel'}
            Name of covariance estimator
        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using 
            a degree of freedom adjustment
        **cov_config : dict
            Optional keyword arguments to pass to covariance estimator
        
        Returns
        -------
        results : TradedFactorModelResults
            Results class with parameter estimates, covariance and test statistics
        
        Notes
        -----
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

        # Classic stat

        # omega = fe.T @ fe / nobs
        # sigma = eps.T @ eps / nobs
        # stat = alphas.T @ pinv(sigma) @ alphas
        # stat *= (nobs - nfactor - nportfolio) / nportfolio
        # stat /= (1 + rp.T @ pinv(omega) @ rp)

        # VCV calculation
        # Need to rearrange ex-post
        nloading = (nfactor + 1) * nportfolio
        xpxi = np.eye(nloading + nfactor)
        xpxi[:nloading, :nloading] = np.kron(pinv(fc.T @ fc / nobs), np.eye(nportfolio))
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
        param_order = np.reshape(np.arange((nfactor + 1) * nportfolio),
                                 (nfactor + 1, nportfolio)).T.ravel()
        params = b.T
        vcv = vcv[param_order][:, param_order]
        betas = b[1:].T
        residual_ss = (eps ** 2).sum()
        e = p - p.mean(0)[None, :]
        total_ss = (e ** 2).sum()
        r2 = 1 - residual_ss / total_ss
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.append('{0}-alpha'.format(portfolio))
            for factor in self.factors.cols:
                param_names.append('{0}-beta.{1}'.format(portfolio, factor))
        for factor in self.factors.cols:
            param_names.append('lambda.{0}'.format(factor))

        res = AttrDict(params=params, cov=full_vcv, betas=betas, rp=rp, rp_cov=rp_cov,
                       alphas=alphas, alpha_vcv=alpha_vcv, jstat=jstat,
                       rsquared=r2, total_ss=total_ss, residual_ss=residual_ss,
                       param_names=param_names,
                       portfolio_names=self.portfolios.cols, factor_names=self.factors.cols,
                       name=self._name, cov_type=cov_type, model=self, nobs=nobs)

        return TradedFactorModelResults(res)


class LinearFactorModel(TradedFactorModel):
    """Asset pricing linear factor model estimator 

    Parameters
    ----------
    portfolios : array-like
        nobs by nportfolio array of test portfolios
    factors : array-like
        nobs by nfactor array of priced factors
    sigma : array-like, optiona
        positive definite residual covariance (nportfolio by nportfolio) 

    Notes
    -----
    Implements a 2-step estimator of risk premia, factor loadings and model 
    tests
    """

    def __init__(self, portfolios, factors, *, sigma=None):
        super(LinearFactorModel, self).__init__(portfolios, factors)
        self._sigma = sigma

    def fit(self, cov_type='robust', debiased=True, **cov_config):
        f = self.factors.ndarray
        nobs, nfactor = f.shape
        p = self.portfolios.ndarray
        nportfolio = p.shape[1]

        # Step 1, n regressions to get B
        fc = np.c_[np.ones((nobs, 1)), f]
        b = np.linalg.lstsq(fc, p)[0]  # nf+1 by np
        eps = p - fc @ b
        bc = b[1:]
        # Step 2, t regressions to get lambda(T)
        lam = np.linalg.lstsq(bc.T, p.mean(0)[:, None])[0].T

        f_rep = np.tile(fc, (1, nportfolio))
        eps_rep = np.tile(eps, (nfactor + 1, 1))
        eps_rep = np.reshape(eps_rep.T, (nportfolio * (nfactor + 1), nobs)).T
        expected = lam @ bc
        pricing_errors = p - expected
        alpha = pricing_errors.mean(0)
        scores = np.c_[f_rep * eps_rep, pricing_errors @ bc.T]
        xeex = scores.T @ scores / nobs

        return lam, alpha


class LinearFactorModelGMM(TradedFactorModel):
    """GMM estimation of asset pricing linear factor model"""

    def __init__(self, factors, portfolios, *, constant=True):
        super(LinearFactorModelGMM, self).__init__(factors, portfolios, constant=constant)

    def fit(self):
        raise NotImplementedError()
