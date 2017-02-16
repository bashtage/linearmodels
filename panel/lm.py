import numpy as np
from numpy.linalg import pinv

from .data import PanelData


class PooledOLS(object):
    """
    Parameters
    ----------
    endog: {np.ndarray, pd.DataFrame, pd.Panel, xr.DataArray, PanelData}
        Endogenous or left-hand-side variable (entities by time)
    exog: {np.ndarray, pd.DataFrame, pd.Panel, xr.DataArray, PanelData}
        Exogenous or right-hand-side variables (entities by time by variable)
    intercept : bool
    
    Notes
    -----
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
