import numpy as np


def homoskedastic_covariance(x, epsilon, debiased=False):
    r"""
    Homoskedastic covariance estimation
    
    Parameters
    ----------
    x : array
        (entity x time) by variables stacked array of regressors
    epsilon : array
        (entity x time) by 1 stacked array of errors
    debiased : bool
        Flag indicating whether to dias adjust
    
    Returns
    -------
    cov : array
        Estimated parameter covariance
        
    Notes
    -----
    The estimator of the covariance is 
    
    .. math:: s^2\hat{\Sigma}_{xx}^{-1}
    
    where
    
    .. math ::
    
        \hat{\Sigma}_{xx} = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T x_{it}x_{it}^{\prime}

    and
    
    .. math ::
    
        s^2 = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T \hat{\epsilon}_{it}^2
    
    where NT is replace by NT-k if ``debiased`` is ``True``.
    
    """
    nt, k = x.shape
    xpx = x.T @ x / nt
    xpxi = np.linalg.inv(xpx)
    scale = nt - k if debiased else nt
    s2 = epsilon.T @ epsilon / scale

    return s2 * (xpxi + xpxi.T) / 2


def heteroskedastic_covariance(x, epsilon, debiased=False):
    r"""
    Covariance estimation using White estimator

    Parameters
    ----------
    x : array
        (entity x time) by variables stacked array of regressors
    epsilon : array
        (entity x time) by 1 stacked array of errors
    debiased : bool
        Flag indicating whether to dias adjust

    Returns
    -------
    cov : array
        Estimated parameter covariance

    Notes
    -----
    The estimator of the covariance is 
    
    .. math::
    
        \hat{\Sigma}_{xx}^{-1}\widehat{Cov}(x_{it}\epsilon_{it})\hat{\Sigma}_{xx}^{-1}
    
    where
    
    .. math ::
    
        \hat{\Sigma}_{xx} = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T x_{it}x_{it}^{\prime}
    
    and 
    
    .. math ::
    
        \widehat{Cov}(x_it\epsilon_{it}) = (NT)^{-1}\sum_{i=1}^N\sum_{t=1}^T 
                                           \hat{\epsilon}_{it}^2 x_{it}x_{it}^{\prime}
    
    where NT is replace by NT-k if ``debiased`` is ``True``.
    """
    nt, k = x.shape
    xpx = x.T @ x / nt
    xpxi = np.linalg.inv(xpx)

    xe = x * epsilon
    scale = nt - k if debiased else nt
    xeex = xe @ xe.T / scale

    cov = xpxi @ xeex @ xpxi

    return (cov + cov.T) / 2


def one_way_clustered_covariance(x, epsilon, debiased=False):
    raise NotImplementedError('Has not been implemented')
