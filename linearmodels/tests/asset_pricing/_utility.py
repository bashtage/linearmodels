import numpy as np
import pandas as pd

from linearmodels.utility import AttrDict


def generate_data(nfactor=3, nportfolio=25, nobs=1000, premia=None, output='pandas',
                  alpha=False):
    np.random.seed(12345)
    if premia is None:
        premia = np.arange(1, nfactor + 1) / (10 * nfactor)
    rho = 0.2
    r = np.eye(nfactor) + rho - np.diag(np.ones(nfactor) * rho)
    factors = rho * np.random.randn(nobs, 1) + np.sqrt(1 - rho ** 2) * np.random.randn(nobs,
                                                                                       nfactor)
    factors = np.sqrt(0.20 ** 2 / 12) * factors
    factors += premia[None, :] / 12
    idio = np.sqrt(0.10 ** 2 / 12) * np.random.randn(nobs, nportfolio)
    betas = np.random.chisquare(2, (nfactor, nportfolio)) / 2.0
    portfolios = factors @ betas + idio
    if alpha:
        portfolios += np.arange(nportfolio)[None,:] / nportfolio / 100
    index = pd.date_range('1930-1-1', periods=nobs, freq='D')
    if output == 'pandas':
        cols = ['factor.{0}'.format(i) for i in range(1, nfactor + 1)]
        factors = pd.DataFrame(factors,
                               columns=cols,
                               index=index)
        cols = ['test_port.{0}'.format(i) for i in range(1, nportfolio + 1)]
        portfolios = pd.DataFrame(portfolios,
                                  columns=cols,
                                  index=index)
    
    return AttrDict(factors=factors, portfolios=portfolios)
