import numpy as np
import pandas as pd
from numpy.linalg import lstsq, matrix_rank
from patsy.highlevel import ModelDesc, dmatrices
from patsy.missing import NAAction

from linearmodels.panel.covariance import HeteroskedasticCovariance, HomoskedasticCovariance, \
    ClusteredCovariance
from linearmodels.panel.data import PanelData
from linearmodels.panel.results import PanelResults
from linearmodels.utility import AttrDict, InvalidTestStatistic, MissingValueWarning, \
    WaldTestStatistic, has_constant, \
    missing_value_warning_msg


class CovarianceManager(object):
    COVARIANCE_ESTIMATORS = {'unadjusted': HomoskedasticCovariance,
                             'homoskedastic': HomoskedasticCovariance,
                             'robust': HeteroskedasticCovariance,
                             'heteroskedastic': HeteroskedasticCovariance,
                             'clustered': ClusteredCovariance}

    def __init__(self, estimator, *cov_estimators):
        self._estimator = estimator
        self._supported = cov_estimators

    def __getitem__(self, item):
        if item not in self.COVARIANCE_ESTIMATORS:
            raise KeyError('Unknown covariance estimator type.')
        cov_est = self.COVARIANCE_ESTIMATORS[item]
        if cov_est not in self._supported:
            raise ValueError('Requested covariance estimator is not supported '
                             'for the {0}.'.format(self._estimator))
        return cov_est


class AbsorbingEffectError(Exception):
    pass


absorbing_error_msg = """
The model cannot be estimated. The included effects have fully absorbed
one or more of the variables. This occurs when one or more of the dependent
variable is perfectly explained using the effects included in the model.
"""


class AmbiguityError(Exception):
    pass


# TODO: Bootstrap covariance
# TODO: Add likelihood and possibly AIC/BIC
# TODO: Correlation between FE and XB
# TODO: Component variance estimators
# TODO: Documentation
# TODO: Example notebooks
# TODO: Formal test of other outputs
# TODO: Test Pooled F-stat
# TODO: Add fast path for no-constant, entity or time effect
# TODO: Test alternative acceptable cluster formats

class PanelOLS(object):
    r"""
    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
        Exogenous or right-hand-side variables (variable by time by entity).
    entity_effect : bool, optional
        Flag whether to include entity (fixed) effects in the model
    time_effect : bool, optional
        Flag whether to include time effects in the model
    other_effects : array-like, optional
        Category codes to use for any effects that are not entity or time
        effects. Each variable is treated as an effect.

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\alpha_i + \gamma_t +\beta^{\prime}x_{it}+\epsilon_{it}

    where :math:`\alpha_i` is omitted if ``entity_effect`` is ``False`` and
    :math:`\gamma_i` is omitted if ``time_effect`` is ``False``. If both ``entity_effect``  and
    ``time_effect`` are ``False``, the model reduces to :class:`PooledOLS`. If ``other_effects``
    is provided, then additional terms are present to reflect these effects.

    Model supports at most 2 effects.  These can be entity-time, entity-other, time-other or
    2 other.
    """

    def __init__(self, dependent, exog, *, weights=None, entity_effect=False, time_effect=False,
                 other_effects=None):
        self.dependent = PanelData(dependent, 'Dep')
        self._original_shape = self.dependent.shape
        self.exog = PanelData(exog, 'Exog')
        self._entity_effect = entity_effect
        self._time_effect = time_effect
        self._constant = None
        self._formula = None
        self.weights = self._adapt_weights(weights)
        self._other_effect_cats = None
        self.other_effects = self._validate_effects(other_effects)
        self._not_null = np.ones(self.dependent.values2d.shape[0], dtype=np.bool)
        self._validate_data()
        self._cov_estimators = CovarianceManager(self.__class__.__name__, HomoskedasticCovariance,
                                                 HeteroskedasticCovariance, ClusteredCovariance)

        self._name = self.__class__.__name__

    def _validate_effects(self, effects):
        if effects is None:
            return False
        effects = PanelData(effects, var_name='OtherEffect',
                            convert_dummies=False)
        num_effects = effects.nvar
        if num_effects + self.entity_effect + self.time_effect > 2:
            raise ValueError('At most two effects supported.')
        cats = {}
        effects_frame = effects.dataframe
        for col in effects_frame:
            cat = pd.Categorical(effects_frame[col])
            cats[col] = cat.codes.astype(np.int64)
        cats = pd.DataFrame(cats, index=effects_frame.index)
        cats = cats[effects_frame.columns]
        self._other_effect_cats = PanelData(cats)
        return True

    def reformat_clusters(self, clusters):
        """Reformat cluster variables

        Parameters
        ----------
        clusters : array-like
            Values to use for variance clustering

        Returns
        -------
        reformatted : PanelData
            Original data with matching axis and observation dropped where
            missing in the model data.
        """
        clusters = PanelData(clusters, var_name='cov.cluster', convert_dummies=False)
        if clusters.shape[1:] != self._original_shape[1:]:
            raise ValueError('clusters must have the same number of entities '
                             'and time periods as the model data.')
        clusters.drop(~self.not_null)
        return clusters

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
        if self.other_effects:
            oe = self._other_effect_cats.dataframe
            if oe.shape[0] != y.shape[0]:
                raise ValueError('other_effects must have the same number of '
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
            if self.other_effects:
                self._other_effect_cats.drop(missing)
            x = self.exog.values2d
            self._not_null = ~missing

        w = self.weights.dataframe
        w = w / w.mean()
        self.weights = PanelData(w)

        if matrix_rank(x) < x.shape[1]:
            raise ValueError('exog does not have full column rank.')
        self._constant, self._constant_index = has_constant(x)

    @property
    def formula(self):
        """Formula used to construct the model"""
        return self._formula

    @formula.setter
    def formula(self, value):
        self._formula = value

    @property
    def entity_effect(self):
        """Flag indicating whether entity effects are included"""
        return self._entity_effect

    @property
    def time_effect(self):
        """Flag indicating whether time effects are included"""
        return self._time_effect

    @property
    def not_null(self):
        """Locations of non-missing observations"""
        return self._not_null

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

    def _f_statistic(self, weps, y, x, root_w, df_resid):
        weps_const = y
        num_df = x.shape[1]
        name = 'Model F-statistic (homoskedastic)'
        if self.has_constant:
            if num_df == 1:
                return InvalidTestStatistic('Model contains only a constant',
                                            name=name)

            num_df -= 1
            weps_const = y - float((root_w.T @ y) / (root_w.T @ root_w))

        resid_ss = weps.T @ weps
        num = float(weps_const.T @ weps_const - resid_ss)
        denom = resid_ss
        denom_df = df_resid
        stat = (num / num_df) / (denom / denom_df)
        return WaldTestStatistic(stat, null='All parameters ex. constant not zero',
                                 df=num_df, df_denom=denom_df, name=name)

    def _f_statistic_robust(self, params, cov_est, debiased, df_resid):
        sel = np.ones(params.shape[0], dtype=np.bool)
        name = 'Model F-statistic (robust)'

        def invalid_f():
            return InvalidTestStatistic('Model contains only a constant',
                                        name=name)

        if self.has_constant:
            if len(sel) == 1:
                return invalid_f
            sel[self._constant_index] = False

        def deferred_f():
            test_params = params[sel]
            test_cov = cov_est.cov[sel][:, sel]
            test_stat = test_params.T @ np.linalg.inv(test_cov) @ test_params
            test_stat = float(test_stat)
            df = sel.sum()
            null = 'All parameters ex. constant not zero'

            if debiased:
                wald = WaldTestStatistic(test_stat / df, null, df, df_resid,
                                         name=name)
            else:
                wald = WaldTestStatistic(test_stat, null, df, name=name)
            return wald

        return deferred_f

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
        other_info = None
        if self.other_effects:
            other_info = []
            oe = self._other_effect_cats.dataframe
            for c in oe:
                name = 'Observations per group (' + str(c) + ')'
                other_info.append(stats(oe[c].values.astype(np.int32), name))
            other_info = pd.DataFrame(other_info)

        return entity_info, time_info, other_info

    def _prepare_between(self):
        y = self.dependent.mean('entity', weights=self.weights).values
        x = self.exog.mean('entity', weights=self.weights).values
        # Weight transformation
        wcount, wmean = self.weights.count('entity'), self.weights.mean('entity')
        wsum = wcount * wmean
        w = wsum.values
        w = w / w.mean()

        return y, x, w

    def _rsquared(self, params, reweight=False):
        if self.has_constant and self.exog.nvar == 1:
            # Constant only fast track
            return 0.0, 0.0, 0.0

        #############################################
        # R2 - Between
        #############################################
        y, x, w = self._prepare_between()
        if np.all(self.weights.values2d == 1.0) and not reweight:
            w = root_w = np.ones_like(w)
        else:
            root_w = np.sqrt(w)
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(weps.T @ weps)
        e = y
        if self.has_constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(w.T @ (e ** 2))
        r2b = 1 - residual_ss / total_ss

        #############################################
        # R2 - Overall
        #############################################
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
        root_w = np.sqrt(w)
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(weps.T @ weps)
        mu = (w * y).sum() / w.sum() if self.has_constant else 0
        we = wy - root_w * mu
        total_ss = float(we.T @ we)
        r2o = 1 - residual_ss / total_ss

        #############################################
        # R2 - Within
        #############################################
        wy = self.dependent.demean('entity', weights=self.weights).values2d
        wx = self.exog.demean('entity', weights=self.weights).values2d
        weps = wy - wx @ params
        residual_ss = float(weps.T @ weps)
        total_ss = float(wy.T @ wy)
        if self.dependent.nobs == 1:
            r2w = 0
        else:
            r2w = 1 - residual_ss / total_ss

        return r2o, r2w, r2b

    def _postestimation(self, params, cov, debiased, df_resid, weps, y, x, root_w):
        deferred_f = self._f_statistic_robust(params, cov, debiased, df_resid)
        f_stat = self._f_statistic(weps, y, x, root_w, df_resid)
        r2o, r2w, r2b = self._rsquared(params)
        entity_info, time_info, other_info = self._info()
        f_pooled = InvalidTestStatistic('Model has no effects', name='Pooled F-stat')
        res = AttrDict(params=params, deferred_cov=cov.deferred_cov,
                       deferred_f=deferred_f, f_stat=f_stat,
                       debiased=debiased, name=self._name, var_names=self.exog.vars,
                       r2w=r2w, r2b=r2b, r2=r2w, r2o=r2o, s2=cov.s2,
                       entity_info=entity_info, time_info=time_info,
                       other_info=other_info, model=self,
                       cov_type=cov.name, index=self.dependent.dataframe.index,
                       f_pooled=f_pooled)
        return res

    @property
    def has_constant(self):
        return self._constant

    def _slow_path(self):
        """Frisch-Waigh-Lovell Implemtation, mostly for weighted"""
        has_effect = self.entity_effect or self.time_effect or self.other_effects
        w = self.weights.values2d
        root_w = np.sqrt(w)

        y = root_w * self.dependent.values2d
        x = root_w * self.exog.values2d
        if not has_effect:
            ybar = root_w @ lstsq(root_w, y)[0]
            return y, x, ybar, 0, 0

        drop_first = self._constant
        d = []
        if self.entity_effect:
            d.append(self.dependent.dummies('entity', drop_first=drop_first).values)
            drop_first = True
        if self.time_effect:
            d.append(self.dependent.dummies('time', drop_first=drop_first).values)
            drop_first = True
        if self.other_effects:
            oe = self._other_effect_cats.dataframe
            for c in oe:
                dummies = pd.get_dummies(oe[c], drop_first=drop_first).astype(np.float64)
                d.append(dummies.values)
                drop_first = True

        d = np.column_stack(d)
        wd = root_w * d
        if self.has_constant:
            wd -= root_w * (w.T @ d / w.sum())
            z = np.ones_like(root_w)
            d -= z * (z.T @ d / z.sum())

        x_mean = np.linalg.lstsq(wd, x)[0]
        y_mean = np.linalg.lstsq(wd, y)[0]

        # Save fitted unweighted effects to use in eps calculation
        x_effects = d @ x_mean
        y_effects = d @ y_mean

        # Purge fitted, weighted values
        x = x - wd @ x_mean
        y = y - wd @ y_mean

        ybar = root_w @ lstsq(root_w, y)[0]
        return y, x, ybar, y_effects, x_effects

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
        elif self.entity_effect:
            y = y.demean('entity')
            x = x.demean('entity')
        else:  # self.time_effect
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

    def _choose_cov(self, cov_type, **cov_config):

        cov_est = self._cov_estimators[cov_type]
        if cov_type != 'clustered':
            return cov_est, cov_config

        cov_config_upd = {k: v for k, v in cov_config.items()}

        clusters = cov_config.get('clusters', None)
        if clusters is not None:
            clusters = self.reformat_clusters(clusters).copy()
            for col in clusters.dataframe:
                cat = pd.Categorical(clusters.dataframe[col])
                clusters.dataframe[col] = cat.codes.astype(np.int64)
            clusters = clusters.dataframe

        cluster_entity = cov_config_upd.pop('cluster_entity', False)
        if cluster_entity:
            group_ids = self.dependent.entity_ids.squeeze()
            name = 'cov.cluster.entity'
            group_ids = pd.Series(group_ids,
                                  index=self.dependent.dataframe.index,
                                  name=name)
            if clusters is not None:
                clusters[name] = group_ids
            else:
                clusters = pd.DataFrame(group_ids)

        cluster_time = cov_config_upd.pop('cluster_time', False)
        if cluster_time:
            group_ids = self.dependent.time_ids.squeeze()
            name = 'cov.cluster.time'
            group_ids = pd.Series(group_ids,
                                  index=self.dependent.dataframe.index,
                                  name=name)
            if clusters is not None:
                clusters[name] = group_ids
            else:
                clusters = pd.DataFrame(group_ids)

        cov_config_upd['clusters'] = clusters.values if clusters is not None else clusters

        return cov_est, cov_config_upd

    def fit(self, *, cov_type='unadjusted', debiased=False, **cov_config):

        unweighted = np.all(self.weights.values2d == 1.0)
        y_effects = x_effects = 0
        if unweighted and not self.other_effects:
            y, x, ybar = self._fast_path()
        else:
            y, x, ybar, y_effects, x_effects = self._slow_path()

        neffects = 0
        drop_first = self.has_constant
        if self.entity_effect:
            neffects += self.dependent.nentity - drop_first
            drop_first = True
        if self.time_effect:
            neffects += self.dependent.nobs - drop_first
            drop_first = True
        if self.other_effects:
            oe = self._other_effect_cats.dataframe
            for c in oe:
                neffects += oe[c].nunique() - drop_first
                drop_first = True

        if self.entity_effect or self.time_effect or self.other_effects:
            if matrix_rank(x) < x.shape[1]:
                raise AbsorbingEffectError(absorbing_error_msg)

        params = np.linalg.lstsq(x, y)[0]

        df_model = x.shape[1] + neffects
        df_resid = y.shape[0] - df_model
        cov_est, cov_config = self._choose_cov(cov_type, **cov_config)
        cov = cov_est(y, x, params, debiased=debiased, extra_df=neffects, **cov_config)
        weps = y - x @ params
        eps = weps
        if not unweighted:
            _y = self.dependent.values2d
            _x = self.exog.values2d
            eps = (_y - y_effects) - (_x - x_effects) @ params
            if self.has_constant:
                # TODO: Understand source for this correction
                w = self.weights.values2d
                eps -= (w * eps).sum() / w.sum()
        resid_ss = float(weps.T @ weps)

        if self.has_constant:
            mu = ybar
        else:
            mu = 0
        total_ss = float((y - mu).T @ (y - mu))
        r2 = 1 - resid_ss / total_ss

        root_w = np.sqrt(self.weights.values2d)
        res = self._postestimation(params, cov, debiased, df_resid, weps, y, x, root_w)
        ######################################
        # Pooled f-stat
        ######################################
        if self.entity_effect or self.time_effect or self.other_effects:
            wy, wx = root_w * self.dependent.values2d, root_w * self.exog.values2d
            weps_pooled = wy - wx @ np.linalg.lstsq(wx, wy)[0]
            resid_ss_pooled = float(weps_pooled.T @ weps_pooled)
            df_num, df_denom = (df_model - wx.shape[1]), df_resid
            num = (resid_ss_pooled - resid_ss) / df_num
            denom = resid_ss / df_denom
            stat = num / denom
            f_pooled = WaldTestStatistic(stat, 'Effects are zero',
                                         df_num, df_denom=df_denom,
                                         name='Pooled F-statistic')
            res.update(f_pooled=f_pooled)

        res.update(dict(df_resid=df_resid, df_model=df_model, nobs=y.shape[0],
                        residual_ss=resid_ss, total_ss=total_ss, wresids=weps, resids=eps,
                        r2=r2))
        return PanelResults(res)


class PooledOLS(PanelOLS):
    r"""
    Estimation of linear model with pooled parameters

    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
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

    def fit(self, *, cov_type='unadjusted', debiased=False, **cov_config):
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
        cov_est, cov_config = self._choose_cov(cov_type, **cov_config)
        cov = cov_est(wy, wx, params, debiased=debiased, **cov_config)
        weps = wy - wx @ params
        eps = y - x @ params
        residual_ss = float(weps.T @ weps)
        e = y
        if self._constant:
            e = e - (w * y).sum() / w.sum()

        total_ss = float(w.T @ (e ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(params, cov, debiased, df_resid, weps, wy, wx, root_w)
        res.update(dict(df_resid=df_resid, df_model=df_model, nobs=y.shape[0],
                        residual_ss=residual_ss, total_ss=total_ss, r2=r2, wresids=weps,
                        resids=eps, index=self.dependent.dataframe.index))
        return PanelResults(res)


class BetweenOLS(PooledOLS):
    r"""
    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
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

    def _choose_cov(self, cov_type, **cov_config):

        cov_est = self._cov_estimators[cov_type]
        if cov_type != 'clustered':
            return cov_est, cov_config
        cov_config_upd = {k: v for k, v in cov_config.items()}

        clusters = cov_config.get('clusters', None)
        if clusters is not None:
            clusters = self.reformat_clusters(clusters).copy()
            cluster_max = np.nanmax(clusters.values3d, axis=1)
            delta = cluster_max - np.nanmin(clusters.values3d, axis=1)
            if np.any(delta != 0):
                raise ValueError('clusters must not vary within an entity')

            index = clusters.panel.minor_axis
            reindex = clusters.entities
            clusters = pd.DataFrame(cluster_max.T, index=index, columns=clusters.vars)
            clusters = clusters.loc[reindex].astype(np.int64)
            cov_config_upd['clusters'] = clusters

        return cov_est, cov_config_upd

    def fit(self, *, reweight=False, cov_type='unadjusted', debiased=False, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        reweight : bool
            Flag indicating to reweight observations if the input data is
            unbalanced
        cov_type : str
            Either 'unadjusted' for homoskedastic covariance or 'robust' for
            heteroskedasticity robust covariance estimation.
        debiased : bool
            Flag indicating to use a debiased parameter covariance estimator

        Returns
        -------
        results : PanelResults
            Estimation results
        """
        y, x, w = self._prepare_between()
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
        cov_est, cov_config = self._choose_cov(cov_type, **cov_config)
        cov = cov_est(wy, wx, params, debiased=debiased, **cov_config)
        weps = wy - wx @ params
        eps = y - x @ params
        residual_ss = float(weps.T @ weps)
        e = y
        if self._constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(w.T @ (e ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(params, cov, debiased, df_resid, weps, wy, wx, root_w)
        res.update(dict(df_resid=df_resid, df_model=df_model, nobs=nobs,
                        residual_ss=residual_ss, total_ss=total_ss, r2=r2, wresids=weps,
                        resids=eps, index=self.dependent.entities))

        return PanelResults(res)

    @classmethod
    def from_formula(cls, formula, data, *, weights=None):
        return super(BetweenOLS, cls).from_formula(formula, data, weights=weights)


class FirstDifferenceOLS(PooledOLS):
    r"""
    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
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

    def _choose_cov(self, cov_type, **cov_config):
        cov_est = self._cov_estimators[cov_type]
        if cov_type != 'clustered':
            return cov_est, cov_config

        cov_config_upd = {k: v for k, v in cov_config.items()}
        clusters = cov_config.get('clusters', None)
        if clusters is not None:
            clusters = self.reformat_clusters(clusters).copy()
            fd = clusters.first_difference()
            fd = fd.values2d
            if np.any(fd.flat[np.isfinite(fd.flat)] != 0):
                raise ValueError('clusters must be identical for values used '
                                 'to compute the first difference')
            clusters = clusters.dataframe.copy()

        cluster_entity = cov_config_upd.pop('cluster_entity', False)
        if cluster_entity:
            group_ids = self.dependent.entity_ids.squeeze()
            name = 'cov.cluster.entity'
            group_ids = pd.Series(group_ids,
                                  index=self.dependent.dataframe.index,
                                  name=name)
            if clusters is not None:
                clusters[name] = group_ids
            else:
                clusters = pd.DataFrame(group_ids)
        clusters = PanelData(clusters)
        values = clusters.values3d[:, 1:]
        clusters = pd.Panel(values, items=clusters.panel.items,
                            major_axis=clusters.panel.major_axis[1:],
                            minor_axis=clusters.panel.minor_axis)
        clusters = PanelData(clusters).dataframe
        clusters = clusters.loc[self.dependent.first_difference().dataframe.index]
        clusters = clusters.astype(np.int64)

        cov_config_upd['clusters'] = clusters.values if clusters is not None else clusters

        return cov_est, cov_config_upd

    def fit(self, *, cov_type='unadjusted', debiased=False, **cov_config):
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
        index = w.index
        w = w.values

        w /= w.mean()
        root_w = np.sqrt(w)

        wx = root_w * x
        wy = root_w * y
        params = lstsq(wx, wy)[0]
        df_resid = y.shape[0] - x.shape[1]
        cov_est, cov_config = self._choose_cov(cov_type, **cov_config)
        cov = cov_est(wy, wx, params, debiased=debiased, **cov_config)

        weps = wy - wx @ params
        eps = y - x @ params
        residual_ss = float(weps.T @ weps)
        total_ss = float(w.T @ (y ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(params, cov, debiased, df_resid, weps, wy, wx, root_w)
        res.update(dict(df_resid=df_resid, df_model=x.shape[1], nobs=y.shape[0],
                        residual_ss=residual_ss, total_ss=total_ss, r2=r2,
                        resids=eps, wresids=weps, index=index))

        return PanelResults(res)

    @classmethod
    def from_formula(cls, formula, data, *, weights=None):
        return super(FirstDifferenceOLS, cls).from_formula(formula, data, weights=weights)
