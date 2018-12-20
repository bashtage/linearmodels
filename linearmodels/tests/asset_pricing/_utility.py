import numpy as np
import pandas as pd

from linearmodels.utility import AttrDict


def generate_data(nfactor=3, nportfolio=25, nobs=1000, premia=None, output='pandas',
                  alpha=False):
    np.random.seed(12345)
    if premia is None:
        premia = np.arange(1, nfactor + 1) / (10 * nfactor)
    rho = 0.2
    e = np.random.randn(nobs, nfactor)
    factors = rho * np.random.randn(nobs, 1) + np.sqrt(1 - rho ** 2) * e
    factors = np.sqrt(0.20 ** 2 / 12) * factors
    factors += premia[None, :] / 12
    idio = np.sqrt(0.10 ** 2 / 12) * np.random.randn(nobs, nportfolio)
    betas = np.random.chisquare(2, (nfactor, nportfolio)) / 2.0
    portfolios = factors @ betas + idio
    if alpha:
        portfolios += np.arange(nportfolio)[None, :] / nportfolio / 100
    index = pd.date_range('1930-1-1', periods=nobs, freq='D')
    if output == 'pandas':
        cols = ['factor_{0}'.format(i) for i in range(1, nfactor + 1)]
        factors = pd.DataFrame(factors,
                               columns=cols,
                               index=index)
        cols = ['port_{0}'.format(i) for i in range(1, nportfolio + 1)]
        portfolios = pd.DataFrame(portfolios,
                                  columns=cols,
                                  index=index)

    return AttrDict(factors=factors, portfolios=portfolios)


def get_all(res):
    attrs = dir(res)
    for attr_name in attrs:
        if attr_name.startswith('_'):
            continue
        attr = getattr(res, attr_name)
        if callable(attr):
            attr()
