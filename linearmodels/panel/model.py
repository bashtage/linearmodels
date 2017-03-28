import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank, lstsq
from patsy.highlevel import dmatrices, ModelDesc
from patsy.missing import NAAction

from linearmodels.panel.covariance import HomoskedasticCovariance
from linearmodels.panel.data import PanelData
from linearmodels.panel.results import PanelResults
from linearmodels.utility import has_constant, AttrDict


class AbsorbingEffectError(Exception):
    pass


absorbing_error_msg = """
The model cannot be estimated. The included effects have fully absorbed 
one or more of the variables. This occurs when one or more of the dependent 
variable is perfectly explained using the effects included in the model.
"""


class AmbiguityError(Exception):
    pass


# TODO: Heteroskedastic covariance
# TODO: One-way cluster covariance
# TODO: Bootstrap covariance
# TODO: Regression F-stat
# TODO: Pooled F-stat
# TODO: number of entities/time
# TODO: Warning about 2 way cluster
# TODO: Group stats, min, max, avg
# TODO: WLS for Between


class PanelOLS(object):
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

    def __init__(self, dependent, exog, *, weights=None, entity_effect=False, time_effect=False):
        self.dependent = PanelData(dependent, 'dependent')
        self.exog = PanelData(exog, 'exog')
        self._constant = None
        self._formula = None
        self.weights = self._adapt_weights(weights)
        self._validate_data()
        # Normalize weights
        avg_w = np.nanmean(self.weights.dataframe.values)
        self.weights = PanelData(self.weights.dataframe / avg_w)
        self._entity_effect = entity_effect
        self._time_effect = time_effect
        self._name = self.__class__.__name__

    def _adapt_weights(self, weights):
        frame = self.dependent.panel.iloc[0].copy()
        nobs, nentity = self.exog.nobs, self.exog.nentity
        if weights is None:
            frame.iloc[:, :] = 1
            frame = frame.T.stack(dropna=False)
            frame.name = 'weight'
            return PanelData(pd.DataFrame(frame))

        if np.asarray(weights).squeeze().ndim == 1:
            if weights.shape[0] == nobs and nobs == nentity:
                raise AmbiguityError('Unable to distinguish nobs form nentity since they are '
                                     'equal. You must use an 2-d array to avoid ambiguity.')
            if weights.shape[0] == nobs:
                weights = np.asarray(weights).squeeze()[:, None]
                weights = weights @ np.ones((1, nentity))
                frame.iloc[:, :] = weights
            elif weights.shape[0] == nentity:
                weights = np.asarray(weights).squeeze()[None, :]
                weights = np.ones((nobs, 1)) @ weights
                frame.iloc[:, :] = weights
            elif weights.shape[0] == nentity * nobs:
                frame = self.dependent.dataframe.copy()
                frame.iloc[:, :] = weights[:, None]
            else:
                raise ValueError('Weights do not have a supported shape.')
            return PanelData(frame)

        return PanelData(weights)

    def _validate_data(self):
        y = self._y = self.dependent.values2d
        x = self._x = self.exog.values2d
        w = self._w = self.weights.values2d
        if y.shape[0] != x.shape[0]:
            raise ValueError('dependent and exog must have the same number of '
                             'observations.')
        if y.shape[0] != w.shape[0]:
            raise ValueError('weights must have the same number of observations as dependent.')

        all_missing = np.any(np.isnan(y), axis=1) & np.all(np.isnan(x), axis=1)
        missing = (np.any(np.isnan(y), axis=1) |
                   np.any(np.isnan(x), axis=1) |
                   np.any(np.isnan(w), axis=1))
        if np.any(missing):
            if np.any(all_missing ^ missing):
                import warnings
                warnings.warn('Missing values detected. Dropping rows with one '
                              'or more missing observation.', UserWarning)
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self.weights.drop(missing)
            x = self.exog.values2d

        self._constant, self._constant_index = has_constant(x)
        if matrix_rank(x) < x.shape[1]:
            raise ValueError('exog does not have full column rank.')

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        self._formula = value

    @property
    def entity_effect(self):
        return self._entity_effect

    @property
    def time_effect(self):
        return self._time_effect

    @classmethod
    def from_formula(cls, formula, data):
        na_action = NAAction(on_NA='raise', NA_types=[])
        data = PanelData(data)
        cln_formula = formula + ' + 0 '
        mod_descr = ModelDesc.from_formula(cln_formula)
        rm_list = []
        effects = {'EntityEffect': False, 'FixedEffect': False, 'TimeEffect': False}
        for term in mod_descr.rhs_termlist:
            if term.name() in effects:
                effects[term.name()] = True
                rm_list.append(term)
        for term in rm_list:
            mod_descr.rhs_termlist.remove(term)

        if effects['EntityEffect'] and effects['FixedEffect']:
            raise ValueError('Cannot use both FixedEffect and EntityEffect')
        entity_effect = effects['EntityEffect'] or effects['FixedEffect']
        time_effect = effects['TimeEffect']

        dependent, exog = dmatrices(mod_descr, data.dataframe,
                                    return_type='dataframe', NA_action=na_action)
        mod = cls(dependent, exog, entity_effect=entity_effect,
                  time_effect=time_effect)
        mod.formula = formula
        return mod

    def _rsquared(self, params):
        # TODO: Fix to be correct for time/entity effects or both
        ym = self.dependent.demean('entity').values2d
        xm = self.exog.demean('entity').values2d
        yhatm = xm @ params
        r2w = np.corrcoef(yhatm.squeeze(), ym.squeeze())[0,1]

        yb = self.dependent.mean('entity').values
        xb = self.exog.mean('entity').values
        yhatb = xb @ params
        r2b = np.corrcoef(yhatb.squeeze(), yb.squeeze())[0,1]
        return r2w, r2b

    def _info(self):
        def stats(ids):
            bc = np.bincount(ids)
            index = ['mean', 'median', 'max', 'min']
            out = [bc.mean(), np.median(bc), bc.max(), bc.min()]
            return pd.Series(out, index=index)

        entity_info = stats(self.dependent.entity_ids.squeeze())
        time_info = stats(self.dependent.time_ids.squeeze())
        return entity_info, time_info


    def fit(self, debiased=False):
        # TODO: Check got absorbing effects, aside form the constant column
        has_effect = self.entity_effect or self.time_effect
        y = self.dependent
        x = self.exog
        y_gm = y.values2d.mean(0)
        x_gm = x.values2d.mean(0)
        neffects = 0

        if self.entity_effect and self.time_effect:
            y = y.demean('both')
            x = x.demean('both')
            neffects = y.nentity - self._constant
        elif self.entity_effect:
            y = y.demean('entity')
            x = x.demean('entity')
            neffects = y.nobs - self._constant
        elif self.time_effect:
            y = y.demean('time')
            x = x.demean('time')
            neffects = y.nobs + y.nentity - self._constant
        y = y.values2d
        x = x.values2d

        if self._constant:
            y = y + y_gm
            x = x + x_gm

        if has_effect:
            if matrix_rank(x) < x.shape[1]:
                raise AbsorbingEffectError(absorbing_error_msg)

        params = lstsq(x, y)[0]
        df_resid = y.shape[0] - x.shape[1] - neffects
        cov = HomoskedasticCovariance(y, x, params, df_resid)
        eps = y - x @ params
        resid_ss = float(eps.T @ eps)
        if self._constant or self._entity_effect or self._time_effect:
            mu = y.mean()
        else:
            mu = 0
        total_ss = float((y - mu).T @ (y - mu))
        r2w, r2b = self._rsquared(params)
        entity_info, time_info = self._info()
        res = AttrDict(params=params, deferred_cov=cov.deferred_cov,
                       debiased=debiased, df_resid=df_resid,
                       df_model=x.shape[1] + neffects, nobs=y.shape[0],
                       name=self._name, var_names=self.exog.vars,
                       residual_ss=resid_ss, total_ss=total_ss,
                       r2w=r2w, r2b=r2b, r2=r2w, s2=cov.s2,
                       entity_info=entity_info, time_info=time_info)  # TODO: Fix R2 definitions for multiple effects

        return res


class PooledOLS(PanelOLS):
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

    def __init__(self, dependent, exog, *, weights=None):
        super(PooledOLS, self).__init__(dependent, exog, weights=weights)

    @classmethod
    def from_formula(cls, formula, data):
        na_action = NAAction(on_NA='raise', NA_types=[])
        data = PanelData(data)
        parts = formula.split('~')
        parts[1] = ' 0 + ' + parts[1]
        cln_formula = '~'.join(parts)
        dependent, exog = dmatrices(cln_formula, data.dataframe,
                                    return_type='dataframe', NA_action=na_action)
        mod = cls(dependent, exog)
        mod.formula = formula
        return mod

    def _rsquared(self, params):
        ym = self.dependent.demean('entity').values2d
        xm = self.exog.demean('entity').values2d
        yhatm = xm @ params
        r2w = np.corrcoef(yhatm.squeeze(), ym.squeeze())[0,1]

        yb = self.dependent.mean('entity').values
        xb = self.exog.mean('entity').values
        yhatb = xb @ params
        r2b = np.corrcoef(yhatb.squeeze(), yb.squeeze())[0,1]
        return r2w, r2b

    def fit(self, debiased=False):
        y = self.dependent.values2d
        x = self.exog.values2d
        params = lstsq(x, y)[0]
        df_resid = y.shape[0] - x.shape[1]
        cov = HomoskedasticCovariance(y, x, params, df_resid)
        from linearmodels.utility import AttrDict

        eps = y - x @ params
        resid_ss = float(eps.T @ eps)
        if self._constant:
            mu = y.mean()
        else:
            mu = 0
        total_ss = float((y - mu).T @ (y - mu))
        r2 = 1 - resid_ss / total_ss
        r2w, r2b = self._rsquared(params)
        entity_info, time_info = self._info()
        res = AttrDict(params=params, deferred_cov=cov.deferred_cov,
                       debiased=debiased, df_resid=df_resid,
                       df_model=x.shape[1], nobs=y.shape[0],
                       name=self._name, var_names=self.exog.vars,
                       residual_ss=resid_ss, total_ss=total_ss,
                       r2=r2, r2w=r2w, r2b=r2b, s2=cov.s2,
                       entity_info=entity_info, time_info=time_info)
        return PanelResults(res)


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

    def __init__(self, dependent, exog, *, weights=None):
        super(BetweenOLS, self).__init__(dependent, exog, weights=weights)

    def fit(self, debiased=False):
        # TODO: Add WLS for unbalanced
        y = self.dependent.mean('entity').values
        x = self.exog.mean('entity').values

        params = lstsq(x, y)[0]
        df_resid = y.shape[0] - x.shape[1]
        cov = HomoskedasticCovariance(y, x, params, df_resid)

        eps = y - x @ params
        resid_ss = float(eps.T @ eps)
        if self._constant:
            mu = y.mean()
        else:
            mu = 0
        total_ss = float((y - mu).T @ (y - mu))
        r2 = 1 - resid_ss / total_ss
        r2w, r2b = self._rsquared(params)
        entity_info, time_info = self._info()
        res = AttrDict(params=params, deferred_cov=cov.deferred_cov,
                       debiased=debiased, df_resid=df_resid,
                       df_model=x.shape[1], nobs=self.dependent.values2d.shape[0],
                       name=self._name, var_names=self.exog.vars,
                       residual_ss=resid_ss, total_ss=total_ss,
                       r2=r2, r2w=r2w, r2b=r2b, s2=cov.s2,
                       entity_info=entity_info, time_info=time_info)
        return PanelResults(res)

    @classmethod
    def from_formula(cls, formula, data):
        return super(BetweenOLS, cls).from_formula(formula, data)


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

    def __init__(self, dependent, exog, *, weights=None):
        super(FirstDifferenceOLS, self).__init__(dependent, exog, weights=weights)
        if self._constant:
            raise ValueError('Constants are not allowed in first difference regressions.')

    def fit(self, debiased=False):
        y = self.dependent.first_difference().values2d
        x = self.exog.first_difference().values2d

        params = lstsq(x, y)[0]
        df_resid = y.shape[0] - x.shape[1]
        cov = HomoskedasticCovariance(y, x, params, df_resid)
        eps = y - x @ params
        resid_ss = float(eps.T @ eps)
        total_ss = float(y.T @ y)

        r2 = 1 - resid_ss / total_ss
        r2w, r2b = self._rsquared(params)
        entity_info, time_info = self._info()
        res = AttrDict(params=params, deferred_cov=cov.deferred_cov,
                       debiased=debiased, df_resid=df_resid,
                       df_model=x.shape[1], nobs=y.shape[0],
                       name=self._name, var_names=self.exog.vars,
                       total_ss=total_ss, residual_ss=resid_ss,
                       r2=r2, r2w=r2w, r2b=r2b, s2=cov.s2,
                       entity_info=entity_info, time_info=time_info)

        return PanelResults(res)

    @classmethod
    def from_formula(cls, formula, data):
        return super(FirstDifferenceOLS, cls).from_formula(formula, data)
