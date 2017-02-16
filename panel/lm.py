import numpy as np
from numpy.linalg import pinv
import pandas as pd
from .data import PanelData


class PooledOLS(object):
    """
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
    Simple implementation of a PooledOLS estimator
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
