import numpy as np
from numpy.linalg import pinv

from linearmodels.iv.data import IVData


class LinearFactorModel(object):
    """Asset pricing linear factor models estimator
    
    Parameters
    ----------
    factors : array-like
        nobs by nfactor array of priced factors
    portfolio : array-like
        nobs by nportfolio array of test portfolios
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
    
    def __init__(self, factors, portfolios, *, constant=True, sigma=None):
        self.factors = IVData(factors)
        self.portfolios = IVData(portfolios)
        self.constant = constant
        nportfolio = self.portfolios.shape[1]
        self.sigma = sigma if sigma is not None else np.eye(nportfolio)
    
    def _fit_ts(self):
        f = self.factors.ndarray
        p = self.portfolios.ndarray
        nportfolio = p.shape[1]
        nobs, nfactor = f.shape
        fc = np.c_[np.ones((nobs, 1)), f]
        rp = f.mean(0)[:, None]
        b = pinv(fc) @ p
        eps = p - fc @ b
        alpha = b[:1].T
        beta = b[1:].T
        # Classic stat
        fe = f - f.mean(0)
        omega = fe.T @ fe / nobs
        sigma = eps.T @ eps / nobs
        stat = alpha.T @ pinv(sigma) @ alpha
        stat *= (nobs - nfactor - nportfolio) / nportfolio
        stat /= (1 + rp.T @ pinv(omega) @ rp)
        
        # VCV calculation
        # Need to rearrange ex-post
        xpxi = np.kron(pinv(fc.T @ fc / nobs), np.eye(nportfolio))
        f_rep = np.tile(fc, (1, nportfolio))
        eps_rep = np.tile(eps, (nfactor + 1, 1))  # 1 2 3 ... 25 1 2 3 ...
        eps_rep = eps_rep.ravel(order='F')
        eps_rep = np.reshape(eps_rep, (nobs, (nfactor + 1) * nportfolio), order='F')
        xe = f_rep * eps_rep
        xeex = xe.T @ xe / nobs
        vcv = xpxi @ xeex @ xpxi / nobs
        
        # Rearrange VCV
        order = np.reshape(np.arange((nfactor + 1) * nportfolio), (nportfolio, nfactor + 1))
        order = order.T.ravel()
        vcv = vcv[order][:, order]
        
        # Return values
        alpha_vcv = vcv[:nportfolio, :nportfolio]
        stat = float(alpha.T @ pinv(alpha_vcv) @ alpha)
        return alpha, stat
    
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
