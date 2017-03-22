import numpy as np
from numpy.linalg import matrix_rank, pinv
from patsy.highlevel import dmatrices
from patsy.missing import NAAction

from linearmodels.utility import has_constant
from linearmodels.panel.data import PanelData


class PooledOLS(object):
    r"""
    Estimation of linear model with pooled parameters

    Parameters
    ----------
    dependent: array-like
        Dependent (left-hand-side) variable (time by entity)
    exog: array-like
        Exogenous or right-hand-side variables (variable by time by entity). 

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\beta^{\prime}x_{it}+\epsilon_{it}
    """

    def __init__(self, dependent, exog):
        self.dependent = PanelData(dependent)
        self.exog = PanelData(exog)
        self._constant = None
        self._formula = None
        self._validate_data()

    def _validate_data(self):
        y = self._y = self.dependent.a2d
        x = self._x = self.exog.a2d
        if y.shape[0] != x.shape[0]:
            raise ValueError('dependent and exog must have the same number of '
                             'observations.')
        all_missing = np.any(np.isnan(y), axis=1) & np.all(np.isnan(x), axis=1)
        missing = np.any(np.isnan(y), axis=1) | np.any(np.isnan(x), axis=1)
        if np.any(missing):
            if np.any(all_missing ^ missing):
                import warnings
                warnings.warn('Missing values detected. Dropping rows with one '
                              'or more missing observation.', UserWarning)
            self.dependent.drop(missing)
            self.exog.drop(missing)
            x = self.exog.a2d

        self._constant, self._constant_index = has_constant(x)
        if matrix_rank(x) < x.shape[1]:
            raise ValueError('exog does not have full column rank.')

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        self._formula = value

    @staticmethod
    def from_formula(formula, data):
        na_action = NAAction(on_NA='raise', NA_types=[])
        data = PanelData(data)
        parts = formula.split('~')
        parts[1] = ' 0 + ' + parts[1]
        cln_formula = '~'.join(parts)
        dependent, exog = dmatrices(cln_formula, data.dataframe,
                                    return_type='dataframe', NA_action=na_action)
        mod = PooledOLS(dependent, exog)
        mod.formula = formula
        return mod

    def fit(self):
        y = self.dependent.a2d
        x = self.exog.a2d
        return pinv(x) @ y


class PanelOLS(PooledOLS):
    r"""
    Parameters
    ----------
    dependent: array-like
        Dependent (left-hand-side) variable (time by entity)
    exog: array-like
        Exogenous or right-hand-side variables (variable by time by entity). 
    entity_effect : bool, optional
        Flag whether to include entity (fixed) effects in the model
    time_effect : bool, optional
        Flag whether to include time effects in the model

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\alpha_i + \gamma_t +\beta^{\prime}x_{it}+\epsilon_{it}

    where :math:`\alpha_i` is omitted if ``entity_effect`` is ``False`` and
    :math:`\gamma_i` is omitted if ``time_effect`` is ``False``. If both ``entity_effect``  and
    ``time_effect`` are ``False``, the model reduces to :class:`PooledOLS`.
    """

    def __init__(self, dependent, exog, *, entity_effect=False, time_effect=False):
        super(PanelOLS, self).__init__(dependent, exog)
        self.entity_effect = entity_effect
        self.time_effect = time_effect

    def fit(self):
        y = self.dependent
        x = self.exog
        if self.entity_effect:
            y = y.demean('entity')
            x = x.demean('entity')
        if self.time_effect:
            y = y.demean('time')
            x = x.demean('time')
        y = y.a2d
        x = x.a2d

        return pinv(x) @ y


class BetweenOLS(PooledOLS):
    r"""
    Parameters
    ----------
    dependent: array-like
        Dependent (left-hand-side) variable (time by entity)
    exog: array-like
        Exogenous or right-hand-side variables (variable by time by entity). 

    Notes
    -----
    The model is given by

    .. math::

        \bar{y}_{i}=  \beta^{\prime}\bar{x}_{i}+\bar{\epsilon}_{i}

    where :math:`\bar{z}` is the time-average.
    """

    def __init__(self, dependent, exog):
        super(BetweenOLS, self).__init__(dependent, exog)

    def fit(self):
        y = self.dependent.mean('time').values
        x = self.exog.mean('time').values

        return pinv(x) @ y


class FirstDifferenceOLS(PooledOLS):
    r"""
    Parameters
    ----------
    dependent: array-like
        Dependent (left-hand-side) variable (time by entity)
    exog: array-like
        Exogenous or right-hand-side variables (variable by time by entity). 

    Notes
    -----
    The model is given by

    .. math::

        \Delta y_{it}=\beta^{\prime}\Delta x_{it}+\Delta\epsilon_{it}
    """

    def __init__(self, dependent, exog):
        super(FirstDifferenceOLS, self).__init__(dependent, exog)

    def fit(self):
        y = self.dependent.first_difference().a2d
        x = self.exog.first_difference().a2d

        return pinv(x) @ y
