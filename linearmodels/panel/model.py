import numpy as np
from numpy.linalg import pinv

from .data import PanelDataHandler
from .fixed_effects import EntityEffect, TimeEffect


class PooledOLS(object):
    r"""
    Estimation of linear model with pooled parameters

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
    The model is given by

    .. math::

        y_{it}=\alpha+\beta^{\prime}x_{it}+\epsilon_{it}

    where :math:`\alpha` is omitted if ``intercept`` is ``False``.
    """

    def __init__(self, endog, exog, *, intercept=True):
        self.endog = PanelDataHandler(endog)
        self.exog = PanelDataHandler(exog)
        self.intercept = intercept

    def fit(self):
        y = self.endog.a2d
        x = self.exog.a2d
        if self.intercept:
            x = np.c_[np.ones((x.shape[0], 1)), x]
        return pinv(x) @ y


class PanelOLS(PooledOLS):
    r"""
    Parameters
    ----------
    endog: array-like
        Endogenous or left-hand-side variable (entities by time)
    exog: array-like
        Exogenous or right-hand-side variables (entities by time by variable). Should not contain
        an intercept or have a constant column in the column span.
    intercept : bool, optional
        Flag whether to include an intercept in the model
    entity_effect : bool, optional
        Flag whether to include an intercept in the model
    time_effect : bool, optional
        Flag whether to include an intercept in the model

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\alpha_i + \gamma_t +\beta^{\prime}x_{it}+\epsilon_{it}

    where :math:`\alpha_i` is omitted if ``entity_effect`` is ``False`` and
    :math:`\gamma_i` is omitted if ``time_effect`` is ``False``. If both ``entity_effect``  and
    ``time_effect`` are ``False``, the model reduces to :class:`PooledOLS`.
    """

    def __init__(self, endog, exog, *, intercept=True, entity_effect=False, time_effect=False):
        super(PanelOLS, self).__init__(endog, exog, intercept=intercept)
        if intercept and (entity_effect or time_effect):
            import warnings
            warnings.warn('Intercept must be False when using entity or time effects.')
            self.intercept = False
        self.entity_effect = entity_effect
        self.time_effect = time_effect

    def fit(self):
        y = self.endog.a2d
        x = self.exog.a2d
        if self.intercept:
            x = np.c_[np.ones((x.shape[0], 1)), x]
        if self.entity_effect:
            y = EntityEffect(y).orthogonalize()
            x = EntityEffect(x).orthogonalize()
        if self.time_effect:
            y = TimeEffect(y).orthogonalize()
            x = TimeEffect(x).orthogonalize()

        return pinv(x) @ y


class BetweenOLS(PooledOLS):
    r"""
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
    The model is given by

    .. math::

        \bar{y}_{i}=\alpha + \beta^{\prime}\bar{x}_{i}+\bar{\epsilon}_{i}

    where :math:`\alpha` is omitted if ``intercept`` is ``False`` and
    :math:`\bar{z}` is the time-average.
    """

    def __init__(self, endog, exog, *, intercept=True):
        super(BetweenOLS, self).__init__(endog, exog, intercept=intercept)

    def fit(self):
        y = self.endog.a3d.mean(axis=1)
        x = self.exog.a3d.mean(axis=1)
        if self.intercept:
            x = np.c_[np.ones((x.shape[0], 1)), x]

        return pinv(x) @ y


class FirstDifferenceOLS(PooledOLS):
    r"""
    Parameters
    ----------
    endog: array-like
        Endogenous or left-hand-side variable (entities by time)
    exog: array-like
        Exogenous or right-hand-side variables (entities by time by variable). Should not contain
        an intercept or have a constant column in the column span.

    Notes
    -----
    The model is given by

    .. math::

        \Delta y_{it}=\beta^{\prime}\Delta x_{it}+\Delta\epsilon_{it}
    """

    def __init__(self, endog, exog, *, intercept=True):
        super(FirstDifferenceOLS, self).__init__(endog, exog, intercept=intercept)

    def fit(self):
        y = np.diff(self.endog.a3d, axis=1)
        x = np.diff(self.exog.a3d, axis=1)
        n, t, k = self.exog.nitems, self.exog.nobs, self.exog.nvar
        y = y.reshape((n * (t - 1), 1))
        x = x.reshape((n * (t - 1), k))
        return pinv(x) @ y
