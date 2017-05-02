import numpy as np
from numpy.linalg import pinv

from linearmodels.asset_pricing.results import TimeSeriesFactorModelResults
from linearmodels.iv.covariance import KERNEL_LOOKUP, _cov_kernel
from linearmodels.iv.data import IVData
from linearmodels.utility import AttrDict, WaldTestStatistic


class LinearFactorModel(object):
    """Asset pricing linear factor models estimator
    
    Parameters
    ----------
    portfolios : array-like
        nobs by nportfolio array of test portfolios
    factors : array-like
        nobs by nfactor array of priced factors
    constant : bool, optional
        Flag indicating whether to include a constant when estimating
        portfolio loadings
    sigma : array-like, optional
        nportfolio by nportfolio positive definite covariance matrix of
        portfolio residuals for use in estimation.  If not provided, an
        identify matrix is used
   
    Notes
    -----
    Implements both time-series and cross-section estimators of risk premia,
    factor loadings and model tests
    """
    
    def __init__(self, portfolios, factors, *, constant=True, sigma=None):
        self.portfolios = IVData(portfolios)
        self.factors = IVData(factors)
        self.constant = constant
        nportfolio = self.portfolios.shape[1]
        self.sigma = sigma if sigma is not None else np.eye(nportfolio)
        self._name = self.__class__.__name__
    
    def _fit_ts(self, cov_type='robust', debiased=True, **cov_config):
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
        xpxi = np.kron(pinv(fc.T @ fc / nobs), np.eye(nportfolio))
        f_rep = np.tile(fc, (1, nportfolio))
        eps_rep = np.tile(eps, (nfactor + 1, 1))  # 1 2 3 ... 25 1 2 3 ...
        eps_rep = eps_rep.ravel(order='F')
        eps_rep = np.reshape(eps_rep, (nobs, (nfactor + 1) * nportfolio), order='F')
        xe = f_rep * eps_rep
        if cov_type in ('robust', 'heteroskedastic'):
            xeex = xe.T @ xe / nobs
            rp_cov = fe.T @ fe / nobs
        elif cov_type == 'kernel':
            kernel = cov_config.get('kernel', 'bartlett')
            bw = cov_config.get('bandwidth', None)
            bw = int(np.ceil(4 * (nobs / 100) ** (2 / 9))) if bw is None else bw
            w = KERNEL_LOOKUP[self._kernel](bw, nobs - 1)
            xeex = _cov_kernel(xe, w)
            rp_cov = _cov_kernel(fe, w)
        else:
            raise ValueError('Unknown cov_type: {0}'.format(cov_type))
        debiased = int(bool(debiased))
        df = fc.shape[1]
        vcv = xpxi @ xeex @ xpxi / (nobs - debiased * df)
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
            param_names.append('alpha.{0}'.format(portfolio))
            for factor in self.factors.cols:
                param_names.append('{0}-beta.{1}'.format(portfolio, factor))
        
        res = AttrDict(params=params, cov=vcv, betas=betas, rp=rp, rp_cov=rp_cov,
                       alphas=alphas, alpha_vcv=alpha_vcv, jstat=jstat,
                       rsquared=r2, total_ss=total_ss, residual_ss=residual_ss,
                       param_names=param_names,
                       portfolio_names=self.portfolios.cols, factor_names=self.factors.cols,
                       name=self._name, cov_type=cov_type, model=self, nobs=nobs)
        
        return TimeSeriesFactorModelResults(res)
    
    def _fit_cs(self):
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
    
    def fit(self, method='cs'):
        """
        Parameters
        ----------
        method : {'cs', 'ts', 'time-series', 'cross-section'}
            Method to use when estimating risk premia and constructing
            model tests
        
        Returns
        -------
        results : LinearFactorResult
            Results class with parameter estimates, covariance and test statistics
        """
        if method.lower() in ('ts', 'time-series'):
            return self._fit_ts()
        elif method.lower() in ('cs', 'cross-section'):
            return self._fit_cs()
        else:
            raise ValueError('Unknown method: {0}'.format(method))


class LinearFactorModelGMM(LinearFactorModel):
    """GMM estimation of asset pricing linear factor model"""
    
    def __init__(self, factors, portfolios, *, constant=True):
        super(LinearFactorModelGMM, self).__init__(factors, portfolios, constant=constant)
    
    def fit(self):
        raise NotImplementedError()
