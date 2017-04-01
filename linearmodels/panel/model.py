import numpy as np
import pandas as pd
from numpy.linalg import lstsq, matrix_rank
from patsy.highlevel import ModelDesc, dmatrices
from patsy.missing import NAAction

from linearmodels.panel.covariance import HomoskedasticCovariance
from linearmodels.panel.data import PanelData
from linearmodels.panel.results import PanelResults
from linearmodels.utility import AttrDict, has_constant, MissingValueWarning, \
    missing_value_warning_msg


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
# TODO: Warning about 2 way cluster


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
        self._entity_effect = entity_effect
        self._time_effect = time_effect
        self._name = self.__class__.__name__

    def _adapt_weights(self, weights):
        if weights is None:
            frame = self.dependent.dataframe.copy()
            frame.iloc[:, :] = 1
            frame.columns = ['weight']
            return PanelData(frame)

        frame = self.dependent.panel.iloc[0].copy()
        nobs, nentity = self.exog.nobs, self.exog.nentity

        if weights.ndim == 3 or weights.shape == (nobs, nentity):
            return PanelData(weights)

        weights = np.squeeze(weights)
        if weights.shape[0] == nobs and nobs == nentity:
            raise AmbiguityError('Unable to distinguish nobs form nentity since they are '
                                 'equal. You must use an 2-d array to avoid ambiguity.')
        if weights.shape[0] == nobs:
            weights = np.asarray(weights)[:, None]
            weights = weights @ np.ones((1, nentity))
            frame.iloc[:, :] = weights
        elif weights.shape[0] == nentity:
            weights = np.asarray(weights)[None, :]
            weights = np.ones((nobs, 1)) @ weights
            frame.iloc[:, :] = weights
        elif weights.shape[0] == nentity * nobs:
            frame = self.dependent.dataframe.copy()
            frame.iloc[:, :] = weights[:, None]
        else:
            raise ValueError('Weights do not have a supported shape.')
        return PanelData(frame)

    def _validate_data(self):
        y = self._y = self.dependent.values2d
        x = self._x = self.exog.values2d
        w = self._w = self.weights.values2d
        if y.shape[0] != x.shape[0]:
            raise ValueError('dependent and exog must have the same number of '
                             'observations.')
        if y.shape[0] != w.shape[0]:
            raise ValueError('weights must have the same number of '
                             'observations as dependent.')

        all_missing = np.any(np.isnan(y), axis=1) & np.all(np.isnan(x), axis=1)
        missing = (np.any(np.isnan(y), axis=1) |
                   np.any(np.isnan(x), axis=1) |
                   np.any(np.isnan(w), axis=1))
        if np.any(missing):
            if np.any(all_missing ^ missing):
                import warnings
                warnings.warn(missing_value_warning_msg, MissingValueWarning)
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self.weights.drop(missing)
            x = self.exog.values2d

        w = self.weights.values2d
        w = w / w.mean()
        self.weights.dataframe.iloc[:, :] = w

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
    def from_formula(cls, formula, data, *, weights=None):
        na_action = NAAction(on_NA='raise', NA_types=[])
        data = PanelData(data)
        parts = formula.split('~')
        parts[1] = ' 0 + ' + parts[1]
        cln_formula = '~'.join(parts)

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
                  time_effect=time_effect, weights=weights)
        mod.formula = formula
        return mod

    def _rsquared(self, params):
        # TODO: Fix to be correct for time/entity effects or both
        # TODO: Handle no constant case
        y = self.dependent.values2d
        x = self.exog.values2d
        yhat = x @ params
        yhat = yhat.squeeze()
        if yhat.std() / np.abs(yhat).mean() <= np.finfo(np.float64).eps:
            r2o = 0
        else:
            r2o = np.corrcoef(yhat.squeeze(), y.squeeze())[0, 1] ** 2

        ym = self.dependent.demean('entity').values2d
        xm = self.exog.demean('entity').values2d
        yhatm = xm @ params
        if np.all(yhatm == 0) or yhatm.ptp() == 0 or \
                (yhatm.std() / np.abs(yhatm).mean() <= np.finfo(np.float64).eps):
            r2w = 0
        else:
            r2w = np.corrcoef(yhatm.squeeze(), ym.squeeze())[0, 1] ** 2

        yb = self.dependent.mean('entity').values
        xb = self.exog.mean('entity').values
        yhatb = xb @ params
        if np.all(yhatb == 0) or yhatb.ptp() == 0 or \
                (yhatb.std() / np.abs(yhatb).mean() <= np.finfo(np.float64).eps):
            r2b = 0
        else:
            r2b = np.corrcoef(yhatb.squeeze(), yb.squeeze())[0, 1] ** 2

        return r2o, r2w, r2b

    def _info(self):
        def stats(ids, name):
            bc = np.bincount(ids)
            index = ['mean', 'median', 'max', 'min', 'total']
            out = [bc.mean(), np.median(bc), bc.max(), bc.min(), bc.shape[0]]
            return pd.Series(out, index=index, name=name)

        entity_info = stats(self.dependent.entity_ids.squeeze(),
                            'Observations per entity')
        time_info = stats(self.dependent.time_ids.squeeze(),
                          'Observations per time period')
        return entity_info, time_info

    def _postestimation(self, params, cov, debiased):
        r2o, r2w, r2b = self._rsquared(params)
        entity_info, time_info = self._info()
        res = AttrDict(params=params, deferred_cov=cov.deferred_cov,
                       debiased=debiased, name=self._name, var_names=self.exog.vars,
                       r2w=r2w, r2b=r2b, r2=r2w, r2o=r2o, s2=cov.s2,
                       entity_info=entity_info, time_info=time_info,
                       model=self,
                       cov_type='Unadjusted')
        return res

    def _fit_lvsd(self, debiased=False):
        constant = self._constant
        has_effect = self.entity_effect or self.time_effect
        w = self.weights
        y = self.dependent
        x = self.exog
        # 1. Construct dummies
        # 2. Append to x
        # 3. Estimate parameters using x and y
        if self.entity_effect:
            d = ed = y.dummies('entity')
        if self.time_effect:
            d = td = y.dummies('time', drop_first=self.entity_effect)
        if self.entity_effect and self.time_effect:
            d = np.c_[ed, td]
        root_w = np.sqrt(w.values2d)
        wx = root_w * x.values2d
        if has_effect:
            wd = root_w * d
            if constant:
                wd -= root_w @ lstsq(root_w, wd)[0]
            wx = np.c_[wx, wd]
        wy = root_w * y.values2d
        params = lstsq(wx, wy)[0]
        return params

    @property
    def has_constant(self):
        return self._constant

    def _slow_path(self):
        """Frisch-Waigh-Lovell Implemtation, mostly for weighted"""
        has_effect = self.entity_effect or self.time_effect
        w = self.weights.values2d
        root_w = np.sqrt(w)

        y = root_w * self.dependent.values2d
        x = root_w * self.exog.values2d
        if not has_effect:
            ybar = root_w @ lstsq(root_w, y)[0]
            return y, x, ybar

        d = []
        if self.entity_effect:
            d.append(self.dependent.dummies('entity', drop_first=self._constant).values)
        if self.time_effect:
            drop = self._constant or self.entity_effect
            d.append(self.dependent.dummies('time', drop_first=drop).values)
        d = np.column_stack(d)
        if self.has_constant:
            d -= d.mean(0)
        d = root_w * d
        x = x - d @ np.linalg.lstsq(d, x)[0]
        y = y - d @ np.linalg.lstsq(d, y)[0]
        ybar = root_w @ lstsq(root_w, y)[0]
        return y, x, ybar

    def _fast_path(self):
        has_effect = self.entity_effect or self.time_effect
        y = self.dependent.values2d
        x = self.exog.values2d
        ybar = y.mean(0)

        if not has_effect:
            return y, x, ybar

        y_gm = ybar
        x_gm = x.mean(0)

        y = self.dependent
        x = self.exog

        if self.entity_effect and self.time_effect:
            y = y.demean('both')
            x = x.demean('both')
            # TODO: Remove
            self._y_both = y
            self._x_both = x
        elif self.entity_effect:
            y = y.demean('entity')
            x = x.demean('entity')
        elif self.time_effect:
            y = y.demean('time')
            x = x.demean('time')

        y = y.values2d
        x = x.values2d

        if self.has_constant:
            y = y + y_gm
            x = x + x_gm
        else:
            ybar = 0

        return y, x, ybar

    def fit(self, debiased=False):
        unweighted = np.all(self.weights.values2d == 1.0)
        if unweighted:
            y, x, ybar = self._fast_path()
        else:
            y, x, ybar = self._slow_path()

        neffects = 0
        if self.entity_effect:
            neffects = self.dependent.nentity - self.has_constant
            if self.time_effect:
                neffects += self.dependent.nobs - 1
        elif self.time_effect:
            neffects = self.dependent.nobs - self.has_constant

        if self.entity_effect or self.time_effect:
            if matrix_rank(x) < x.shape[1]:
                raise AbsorbingEffectError(absorbing_error_msg)

        params = np.linalg.lstsq(x, y)[0]

        df_model = x.shape[1] + neffects
        df_resid = y.shape[0] - df_model
        cov_denom = df_resid if debiased else y.shape[0]
        cov = HomoskedasticCovariance(y, x, params, cov_denom)
        weps = y - x @ params
        eps = weps
        if not unweighted:
            # TODO: eps is not right when there are effects and weighting
            # TODO: Need the estimated effect matrices to construct epsilon
            # TODO: Should be (y - effects) - (x - effects) @ params
            eps = self.dependent.values2d - self.exog.values2d @ params
        resid_ss = float(weps.T @ weps)

        if self.has_constant or self.entity_effect or self.time_effect:
            mu = ybar
        else:
            mu = 0
        total_ss = float((y - mu).T @ (y - mu))
        res = self._postestimation(params, cov, debiased)
        r2 = 1 - resid_ss / total_ss
        res.update(dict(df_resid=df_resid, df_model=df_model, nobs=y.shape[0],
                        residual_ss=resid_ss, total_ss=total_ss, wresid=weps, resid=eps,
                        r2=r2))
        return PanelResults(res)


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

    def _prepare_between(self):
        y = self.dependent.mean('entity', weights=self.weights).values
        x = self.exog.mean('entity', weights=self.weights).values
        # Weight transformation
        wcount, wmean = self.weights.count('entity'), self.weights.mean('entity')
        wsum = wcount * wmean
        w = wsum.values
        w = w / w.mean()

        return y, x, w

    def _prepare_within(self):
        y = self.dependent.demean('entity', weights=self.weights).values2d
        x = self.exog.demean('entity', weights=self.weights).values2d
        w = self.weights.values2d

        return y, x, w

    @classmethod
    def from_formula(cls, formula, data, *, weights=None):
        na_action = NAAction(on_NA='raise', NA_types=[])
        data = PanelData(data)
        parts = formula.split('~')
        parts[1] = ' 0 + ' + parts[1]
        cln_formula = '~'.join(parts)
        dependent, exog = dmatrices(cln_formula, data.dataframe,
                                    return_type='dataframe', NA_action=na_action)
        mod = cls(dependent, exog, weights=weights)
        mod.formula = formula
        return mod

    def _rsquared(self, params):
        y = self.dependent.values2d
        x = self.exog.values2d
        yhat = x @ params
        r2o = np.corrcoef(yhat.squeeze(), y.squeeze())[0, 1] ** 2

        ym = self.dependent.demean('entity').values2d
        xm = self.exog.demean('entity').values2d
        yhatm = xm @ params
        r2w = np.corrcoef(yhatm.squeeze(), ym.squeeze())[0, 1] ** 2

        yb = self.dependent.mean('entity').values
        xb = self.exog.mean('entity').values
        yhatb = xb @ params
        r2b = np.corrcoef(yhatb.squeeze(), yb.squeeze())[0, 1] ** 2

        return r2o, r2w, r2b

    def fit(self, debiased=False):
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
        root_w = np.sqrt(w)
        wx = root_w * x
        wy = root_w * y

        params = lstsq(wx, wy)[0]

        nobs = y.shape[0]
        df_model = x.shape[1]
        df_resid = nobs - df_model
        cov_denom = nobs if not debiased else df_resid
        cov = HomoskedasticCovariance(wy, wx, params, cov_denom)
        weps = wy - wx @ params
        eps = y - x @ params
        residual_ss = float(weps.T @ weps)
        e = y
        if self._constant:
            e = e - (w * y).sum() / w.sum()

        total_ss = float(w.T @ (e ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(params, cov, debiased)
        res.update(dict(df_resid=df_resid, df_model=df_model, nobs=y.shape[0],
                        residual_ss=residual_ss, total_ss=total_ss, r2=r2, wresid=weps,
                        resid=eps))
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

    def fit(self, debiased=False, reweight=False):
        """
        Estimate model parameters

        Parameters
        ----------
        debiased : bool
            Flag indicating to use a debiased parameter covariance estimator
        reweight : bool
            Flag indicating to reweight observations if the input data is
            unbalanced

        Returns
        -------
        results : PanelResults
            Estimation results
        """
        y, x, w = self._prepare_between()
        self._avgy = y
        self._avgx = x
        self._avgw = w
        if np.all(self.weights.values2d == 1.0) and not reweight:
            w = root_w = np.ones_like(w)
        else:
            root_w = np.sqrt(w)

        wx = root_w * x
        wy = root_w * y
        params = lstsq(wx, wy)[0]

        df_resid = y.shape[0] - x.shape[1]
        df_model = x.shape[1],
        nobs = y.shape[0]
        cov_denom = y.shape[0] if not debiased else df_resid
        cov = HomoskedasticCovariance(wy, wx, params, cov_denom)
        weps = wy - wx @ params
        eps = y - x @ params
        residual_ss = float(weps.T @ weps)
        e = y
        if self._constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(w.T @ (e ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(params, cov, debiased)
        res.update(dict(df_resid=df_resid, df_model=df_model, nobs=nobs,
                        residual_ss=residual_ss, total_ss=total_ss, r2=r2, wresid=weps, resid=eps))

        return PanelResults(res)

    @classmethod
    def from_formula(cls, formula, data, *, weights=None):
        return super(BetweenOLS, cls).from_formula(formula, data, weights=weights)


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
        if self.dependent.nobs < 2:
            raise ValueError('Panel must have at least 2 time periods')

    def fit(self, debiased=False):
        y = self.dependent.first_difference().values2d
        x = self.exog.first_difference().values2d

        w = 1.0 / self.weights.panel.values
        w = w[:, :-1] + w[:, 1:]
        w = 1.0 / w
        w = pd.Panel(w, items=self.weights.panel.items,
                     major_axis=self.weights.panel.major_axis[1:],
                     minor_axis=self.weights.panel.minor_axis)
        w = w.swapaxes(1, 2).to_frame(filter_observations=False)
        w = w.reindex(self.weights.dataframe.index).dropna(how='any')
        w = w.values

        w /= w.mean()
        root_w = np.sqrt(w)

        wx = root_w * x
        wy = root_w * y
        params = lstsq(wx, wy)[0]
        df_resid = y.shape[0] - x.shape[1]
        cov_denom = df_resid if debiased else y.shape[0]
        cov = HomoskedasticCovariance(wy, wx, params, cov_denom)

        weps = wy - wx @ params
        eps = y - x @ params
        residual_ss = float(weps.T @ weps)
        total_ss = float(w.T @ (y ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(params, cov, debiased)
        res.update(dict(df_resid=df_resid, df_model=x.shape[1], nobs=y.shape[0],
                        residual_ss=residual_ss, total_ss=total_ss, r2=r2,
                        resid=eps, wresid=weps))

        return PanelResults(res)

    @classmethod
    def from_formula(cls, formula, data, *, weights=None):
        return super(FirstDifferenceOLS, cls).from_formula(formula, data, weights=weights)
