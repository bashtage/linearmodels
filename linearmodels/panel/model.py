import numpy as np
import pandas as pd
from numpy.linalg import lstsq, matrix_rank
from patsy.highlevel import ModelDesc, dmatrices
from patsy.missing import NAAction

from linearmodels.panel.covariance import (ACCovariance, ClusteredCovariance,
                                           CovarianceManager, DriscollKraay,
                                           FamaMacBethCovariance,
                                           FamaMacBethKernelCovariance,
                                           HeteroskedasticCovariance,
                                           HomoskedasticCovariance)
from linearmodels.panel.data import PanelData
from linearmodels.panel.results import (PanelEffectsResults, PanelResults,
                                        RandomEffectsResults)
from linearmodels.utility import (AttrDict, InapplicableTestStatistic,
                                  InvalidTestStatistic, WaldTestStatistic,
                                  ensure_unique_column, has_constant,
                                  missing_warning)


class AbsorbingEffectError(Exception):
    pass


absorbing_error_msg = """
The model cannot be estimated. The included effects have fully absorbed
one or more of the variables. This occurs when one or more of the dependent
variable is perfectly explained using the effects included in the model.
"""


class AmbiguityError(Exception):
    pass


__all__ = ['PanelOLS', 'PooledOLS', 'RandomEffects', 'FirstDifferenceOLS',
           'BetweenOLS', 'AbsorbingEffectError', 'AmbiguityError',
           'FamaMacBeth']


# Likely
# TODO: Formal test of other outputs
# Future
# TODO: Bootstrap covariance
# TODO: Possibly add AIC/BIC
# TODO: ML Estimation of RE model
# TODO: Defer estimation of 3 R2 values -- slow


class PooledOLS(object):
    r"""
    Pooled coefficient estimator for panel data

    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array-like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\beta^{\prime}x_{it}+\epsilon_{it}
    """

    def __init__(self, dependent, exog, *, weights=None):
        self.dependent = PanelData(dependent, 'Dep')
        self.exog = PanelData(exog, 'Exog')
        self._original_shape = self.dependent.shape
        self._constant = None
        self._formula = None
        self._name = self.__class__.__name__
        self.weights = self._adapt_weights(weights)
        self._not_null = np.ones(self.dependent.values2d.shape[0], dtype=np.bool)
        self._cov_estimators = CovarianceManager(self.__class__.__name__, HomoskedasticCovariance,
                                                 HeteroskedasticCovariance, ClusteredCovariance,
                                                 DriscollKraay, ACCovariance)

        self._validate_data()

    def __str__(self):
        out = '{name} \nNum exog: {num_exog}, Constant: {has_constant}'
        return out.format(name=self.__class__.__name__,
                          num_exog=self.exog.dataframe.shape[1],
                          has_constant=self.has_constant)

    def __repr__(self):
        return self.__str__() + '\nid: ' + str(hex(id(self)))

    def reformat_clusters(self, clusters):
        """
        Reformat cluster variables

        Parameters
        ----------
        clusters : array-like
            Values to use for variance clustering

        Returns
        -------
        reformatted : PanelData
            Original data with matching axis and observation dropped where
            missing in the model data.

        Notes
        -----
        This is exposed for testing and is not normally needed for estimation
        """
        clusters = PanelData(clusters, var_name='cov.cluster', convert_dummies=False)
        if clusters.shape[1:] != self._original_shape[1:]:
            raise ValueError('clusters must have the same number of entities '
                             'and time periods as the model data.')
        clusters.drop(~self.not_null)
        return clusters

    def _info(self):
        """Information about panel structure"""

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

        return entity_info, time_info, other_info

    def _adapt_weights(self, weights):
        """Check and transform weights depending on size"""
        if weights is None:
            frame = self.dependent.dataframe.copy()
            frame.iloc[:, :] = 1
            frame.columns = ['weight']
            return PanelData(frame)

        frame = pd.DataFrame(columns=self.dependent.entities,
                             index=self.dependent.time)
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
        """Check input shape and remove missing"""
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

        missing_warning(all_missing ^ missing)
        if np.any(missing):
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self.weights.drop(missing)

            x = self.exog.values2d
            self._not_null = ~missing

        w = self.weights.dataframe
        if np.any(w.values <= 0):
            raise ValueError('weights must be strictly positive.')
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
    def has_constant(self):
        """Flag indicating the model a constant or implicit constant"""
        return self._constant

    def _f_statistic(self, weps, y, x, root_w, df_resid):
        """Compute model F-statistic"""
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
        stat = float((num / num_df) / (denom / denom_df))
        return WaldTestStatistic(stat, null='All parameters ex. constant not zero',
                                 df=num_df, df_denom=denom_df, name=name)

    def _f_statistic_robust(self, params, cov_est, debiased, df_resid):
        """Compute Wald test that all parameters are 0, ex. constant"""
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

    def _prepare_between(self):
        """Prepare values for between estimation of R2"""
        y = self.dependent.mean('entity', weights=self.weights).values
        x = self.exog.mean('entity', weights=self.weights).values
        # Weight transformation
        wcount, wmean = self.weights.count('entity'), self.weights.mean('entity')
        wsum = wcount * wmean
        w = wsum.values
        w = w / w.mean()

        return y, x, w

    def _rsquared(self, params, reweight=False):
        """Compute alternative measures of R2"""
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
        if self.dependent.nobs == 1 or (self.exog.nvar == 1 and self.has_constant):
            r2w = 0
        else:
            r2w = 1 - residual_ss / total_ss

        return r2o, r2w, r2b

    def _postestimation(self, params, cov, debiased, df_resid, weps, y, x, root_w):
        """Common post-estimation values"""
        deferred_f = self._f_statistic_robust(params, cov, debiased, df_resid)
        f_stat = self._f_statistic(weps, y, x, root_w, df_resid)
        r2o, r2w, r2b = self._rsquared(params)
        f_pooled = InapplicableTestStatistic(reason='Model has no effects',
                                             name='Pooled F-stat')
        entity_info, time_info, other_info = self._info()
        nobs = weps.shape[0]
        sigma2 = float(weps.T @ weps / nobs)
        loglik = -0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        res = AttrDict(params=params, deferred_cov=cov.deferred_cov,
                       deferred_f=deferred_f, f_stat=f_stat,
                       debiased=debiased, name=self._name, var_names=self.exog.vars,
                       r2w=r2w, r2b=r2b, r2=r2w, r2o=r2o, s2=cov.s2,
                       model=self, cov_type=cov.name, index=self.dependent.index,
                       entity_info=entity_info, time_info=time_info, other_info=other_info,
                       f_pooled=f_pooled, loglik=loglik)
        return res

    @property
    def not_null(self):
        """Locations of non-missing observations"""
        return self._not_null

    @classmethod
    def from_formula(cls, formula, data, *, weights=None):
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array-like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array-like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        model : PooledOLS
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import PooledOLS
        >>> mod = PooledOLS.from_formula('y ~ 1 + x1', data)
        >>> res = mod.fit()
        """
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
                                  index=self.dependent.index,
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
                                  index=self.dependent.index,
                                  name=name)
            if clusters is not None:
                clusters[name] = group_ids
            else:
                clusters = pd.DataFrame(group_ids)

        cov_config_upd['clusters'] = clusters.values if clusters is not None else clusters

        return cov_est, cov_config_upd

    def fit(self, *, cov_type='unadjusted', debiased=True, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        results :  PanelResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import PooledOLS
        >>> mod = PooledOLS(y, x)
        >>> res = mod.fit(cov_type='clustered', cluster_entity=True)

        Notes
        -----
        Four covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters
          * ``cluster_time`` - Boolean indicating to use time clusters

        * 'kernel' - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """
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
        cov = cov_est(wy, wx, params, self.dependent.entity_ids, self.dependent.time_ids,
                      debiased=debiased, **cov_config)
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
                        resids=eps, index=self.dependent.index))

        return PanelResults(res)


class PanelOLS(PooledOLS):
    r"""
    One- and two-way fixed effects estimator for panel data

    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity).
    exog : array-like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array-like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.
    entity_effects : bool, optional
        Flag whether to include entity (fixed) effects in the model
    time_effects : bool, optional
        Flag whether to include time effects in the model
    other_effects : array-like, optional
        Category codes to use for any effects that are not entity or time
        effects. Each variable is treated as an effect.

    Notes
    -----
    Many models can be estimated. The most common included entity effects and
    can be described

    .. math::

        y_{it} = \alpha_i + \beta^{\prime}x_{it} + \epsilon_{it}

    where :math:`\alpha_i` is included if ``entity_effects=True``.

    Time effect are also supported, which leads to a model of the form

    .. math::

        y_{it}= \gamma_t + \beta^{\prime}x_{it} + \epsilon_{it}

    where :math:`\gamma_i` is included if ``time_effects=True``.

    Both effects can be simultaneously used,

    .. math::

        y_{it}=\alpha_i + \gamma_t + \beta^{\prime}x_{it} + \epsilon_{it}

    Additionally , arbitrary effects can be specified using categorical variables.

    If both ``entity_effect``  and``time_effects`` are ``False``, and no other
    effects are included, the model reduces to :class:`PooledOLS`.

    Model supports at most 2 effects.  These can be entity-time, entity-other, time-other or
    2 other.
    """

    def __init__(self, dependent, exog, *, weights=None, entity_effects=False, time_effects=False,
                 other_effects=None):
        super(PanelOLS, self).__init__(dependent, exog, weights=weights)

        self._entity_effects = entity_effects
        self._time_effects = time_effects
        self._other_effect_cats = None
        self._other_effects = self._validate_effects(other_effects)

    def __str__(self):
        out = super(PanelOLS, self).__str__()
        additional = '\nEntity Effects: {ee}, Time Effects: {te}, Num Other Effects: {oe}'
        oe = 0
        if self.other_effects:
            oe = self._other_effect_cats.nvar
        additional = additional.format(ee=self.entity_effects, te=self.time_effects, oe=oe)
        out += additional
        return out

    def _validate_effects(self, effects):
        """Check model effects"""
        if effects is None:
            return False
        effects = PanelData(effects, var_name='OtherEffect',
                            convert_dummies=False)

        if effects.shape[1:] != self._original_shape[1:]:
            raise ValueError('other_effects must have the same number of '
                             'entites and time periods as dependent.')

        num_effects = effects.nvar
        if num_effects + self.entity_effects + self.time_effects > 2:
            raise ValueError('At most two effects supported.')
        cats = {}
        effects_frame = effects.dataframe
        for col in effects_frame:
            cat = pd.Categorical(effects_frame[col])
            cats[col] = cat.codes.astype(np.int64)
        cats = pd.DataFrame(cats, index=effects_frame.index)
        cats = cats[effects_frame.columns]
        other_effects = PanelData(cats)
        other_effects.drop(~self.not_null)
        self._other_effect_cats = other_effects
        cats = other_effects.values2d
        nested = False
        if cats.shape[1] == 2:
            nested = self._is_effect_nested(cats[:, [0]], cats[:, [1]])
            nested |= self._is_effect_nested(cats[:, [1]], cats[:, [0]])
            nesting_effect = 'other effects'
        elif self.entity_effects:
            nested = self._is_effect_nested(cats[:, [0]], self.dependent.entity_ids)
            nested |= self._is_effect_nested(self.dependent.entity_ids, cats[:, [0]])
            nesting_effect = 'entity effects'
        elif self.time_effects:
            nested = self._is_effect_nested(cats[:, [0]], self.dependent.time_ids)
            nested |= self._is_effect_nested(self.dependent.time_ids, cats[:, [0]])
            nesting_effect = 'time effects'
        if nested:
            raise ValueError('Included other effects nest or are nested '
                             'by {effect}'.format(effect=nesting_effect))

        return True

    @property
    def entity_effects(self):
        """Flag indicating whether entity effects are included"""
        return self._entity_effects

    @property
    def time_effects(self):
        """Flag indicating whether time effects are included"""
        return self._time_effects

    @property
    def other_effects(self):
        """Flag indicating whether other (generic) effects are included"""
        return self._other_effects

    @classmethod
    def from_formula(cls, formula, data, *, weights=None, other_effects=None):
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules
            with two special variable names, EntityEffects and TimeEffects
            which can be used to specify that the model should contain an
            entity effect or a time effect, respectively. See Examples.
        data : array-like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array-like
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual time
            the weight should be homoskedastic.
        other_effects : array-like, optional
            Category codes to use for any effects that are not entity or time
            effects. Each variable is treated as an effect.


        Returns
        -------
        model : PanelOLS
            Model specified using the formula

        Examples
        --------
        >>> from linearmodels import PanelOLS
        >>> mod = PanelOLS.from_formula('y ~ 1 + x1 + EntityEffects', data)
        >>> res = mod.fit(cov_type='clustered', cluster_entity=True)
        """
        na_action = NAAction(on_NA='raise', NA_types=[])
        data = PanelData(data)
        parts = formula.split('~')
        parts[1] = ' 0 + ' + parts[1]
        cln_formula = '~'.join(parts)

        mod_descr = ModelDesc.from_formula(cln_formula)
        rm_list = []
        effects = {'EntityEffects': False, 'FixedEffects': False, 'TimeEffects': False}
        for term in mod_descr.rhs_termlist:
            if term.name() in effects:
                effects[term.name()] = True
                rm_list.append(term)
        for term in rm_list:
            mod_descr.rhs_termlist.remove(term)

        if effects['EntityEffects'] and effects['FixedEffects']:
            raise ValueError('Cannot use both FixedEffects and EntityEffects')
        entity_effect = effects['EntityEffects'] or effects['FixedEffects']
        time_effect = effects['TimeEffects']

        dependent, exog = dmatrices(mod_descr, data.dataframe,
                                    return_type='dataframe', NA_action=na_action)
        mod = cls(dependent, exog, entity_effects=entity_effect,
                  time_effects=time_effect, weights=weights, other_effects=other_effects)
        mod.formula = formula
        return mod

    def _slow_path(self):
        """Frisch-Waigh-Lovell implementation, works for all scenarios"""
        has_effect = self.entity_effects or self.time_effects or self.other_effects
        w = self.weights.values2d
        root_w = np.sqrt(w)

        y = root_w * self.dependent.values2d
        x = root_w * self.exog.values2d
        if not has_effect:
            ybar = root_w @ lstsq(root_w, y)[0]
            return y, x, ybar, 0, 0

        drop_first = self._constant
        d = []
        if self.entity_effects:
            d.append(self.dependent.dummies('entity', drop_first=drop_first).values)
            drop_first = True
        if self.time_effects:
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
        """Dummy-variable free estimation without weights"""
        has_effect = self.entity_effects or self.time_effects or self.other_effects
        y = self.dependent.values2d
        x = self.exog.values2d
        ybar = y.mean(0)

        if not has_effect:
            return y, x, ybar

        y_gm = ybar
        x_gm = x.mean(0)

        y = self.dependent
        x = self.exog

        if self.other_effects:
            groups = self._other_effect_cats
            if self.entity_effects or self.time_effects:
                groups = groups.copy()
                if self.entity_effects:
                    effect = self.dependent.entity_ids
                else:
                    effect = self.dependent.time_ids
                col = ensure_unique_column('additional.effect', groups.dataframe)
                groups.dataframe[col] = effect
            y = y.general_demean(groups)
            x = x.general_demean(groups)
        elif self.entity_effects and self.time_effects:
            y = y.demean('both')
            x = x.demean('both')
        elif self.entity_effects:
            y = y.demean('entity')
            x = x.demean('entity')
        else:  # self.time_effects
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

    def _weighted_fast_path(self):
        """Dummy-variable free estimation with weights"""
        has_effect = self.entity_effects or self.time_effects or self.other_effects
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
        root_w = np.sqrt(w)
        wybar = root_w * (w.T @ y / w.sum())

        if not has_effect:
            wy = root_w * self.dependent.values2d
            wx = root_w * self.exog.values2d
            return wy, wx, wybar, 0, 0

        wy_gm = wybar
        wx_gm = root_w * (w.T @ x / w.sum())

        y = self.dependent
        x = self.exog

        if self.other_effects:
            groups = self._other_effect_cats
            if self.entity_effects or self.time_effects:
                groups = groups.copy()
                if self.entity_effects:
                    effect = self.dependent.entity_ids
                else:
                    effect = self.dependent.time_ids
                col = ensure_unique_column('additional.effect', groups.dataframe)
                groups.dataframe[col] = effect
            wy = y.general_demean(groups, weights=self.weights)
            wx = x.general_demean(groups, weights=self.weights)
        elif self.entity_effects and self.time_effects:
            wy = y.demean('both', weights=self.weights)
            wx = x.demean('both', weights=self.weights)
        elif self.entity_effects:
            wy = y.demean('entity', weights=self.weights)
            wx = x.demean('entity', weights=self.weights)
        else:  # self.time_effects
            wy = y.demean('time', weights=self.weights)
            wx = x.demean('time', weights=self.weights)

        wy = wy.values2d
        wx = wx.values2d

        if self.has_constant:
            wy += wy_gm
            wx += wx_gm
        else:
            wybar = 0

        wy_effects = y.values2d - wy / root_w
        wx_effects = x.values2d - wx / root_w

        return wy, wx, wybar, wy_effects, wx_effects

    def _info(self):
        """Information about model effects and panel structure"""

        def stats(ids, name):
            bc = np.bincount(ids)
            index = ['mean', 'median', 'max', 'min', 'total']
            out = [bc.mean(), np.median(bc), bc.max(), bc.min(), bc.shape[0]]
            return pd.Series(out, index=index, name=name)

        entity_info, time_info, other_info = super(PanelOLS, self)._info()

        if self.other_effects:
            other_info = []
            oe = self._other_effect_cats.dataframe
            for c in oe:
                name = 'Observations per group (' + str(c) + ')'
                other_info.append(stats(oe[c].values.astype(np.int32), name))
            other_info = pd.DataFrame(other_info)

        return entity_info, time_info, other_info

    @staticmethod
    def _is_effect_nested(effects, clusters):
        """Determine whether an effect is nested by the covariance clusters"""
        is_nested = np.zeros(effects.shape[1], dtype=np.bool)
        for i, e in enumerate(effects.T):
            e = (e - e.min()).astype(np.int64)
            e_count = len(np.unique(e))
            for c in clusters.T:
                c = (c - c.min()).astype(np.int64)
                cmax = c.max()
                ec = e * (cmax + 1) + c
                is_nested[i] = len(np.unique(ec)) == e_count
        return np.all(is_nested)

    def _determine_df_adjustment(self, cov_type, **cov_config):
        has_effect = self.entity_effects or self.time_effects or self.other_effects
        if cov_type != 'clustered' or not has_effect:
            return True
        num_effects = self.entity_effects + self.time_effects
        if self.other_effects:
            num_effects += self._other_effect_cats.shape[1]

        clusters = cov_config.get('clusters', None)
        if clusters is None:  # No clusters
            return True

        effects = [self._other_effect_cats] if self.other_effects else []
        if self.entity_effects:
            effects.append(self.dependent.entity_ids)
        if self.time_effects:
            effects.append(self.dependent.time_ids)
        effects = np.column_stack(effects)
        if num_effects == 1:
            return not self._is_effect_nested(effects, clusters)
        return True  # Default case for 2-way -- not completely clear

    def fit(self, *, use_lsdv=False, cov_type='unadjusted', debiased=True, auto_df=True,
            count_effects=True, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        use_lsdv : bool, optional
            Flag indicating to use the Least Squares Dummy Variable estimator
            to eliminate effects.  The default value uses only means and does
            note require constructing dummy variables for each effect.
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        auto_df : bool, optional
            Flag indicating that the treatment of estimated effects in degree
            of freedom adjustment is automatically handled. This is useful
            since clustered standard errors that are clustered using the same
            variable as an effect do not require degree of freedom correction
            while other estimators such as the unadjusted covariance do.
        count_effects : bool, optional
            Flag indicating that the covariance estimator should be adjusted
            to account for the estimation of effects in the model. Only used
            if ``auto_df=False``.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        results :  PanelEffectsResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import PanelOLS
        >>> mod = PanelOLS(y, x, entity_effects=True)
        >>> res = mod.fit(cov_type='clustered', cluster_entity=True)

        Notes
        -----
        Three covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters
          * ``cluster_time`` - Boolean indicating to use time clusters

        * 'kernel' - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """

        weighted = np.any(self.weights.values2d != 1.0)
        y_effects = x_effects = 0
        if use_lsdv:
            y, x, ybar, y_effects, x_effects = self._slow_path()
        elif not weighted:
            y, x, ybar = self._fast_path()
        else:
            y, x, ybar, y_effects, x_effects = self._weighted_fast_path()

        neffects = 0
        drop_first = self.has_constant
        if self.entity_effects:
            neffects += self.dependent.nentity - drop_first
            drop_first = True
        if self.time_effects:
            neffects += self.dependent.nobs - drop_first
            drop_first = True
        if self.other_effects:
            oe = self._other_effect_cats.dataframe
            for c in oe:
                neffects += oe[c].nunique() - drop_first
                drop_first = True

        if self.entity_effects or self.time_effects or self.other_effects:
            if matrix_rank(x) < x.shape[1]:
                raise AbsorbingEffectError(absorbing_error_msg)

        params = np.linalg.lstsq(x, y)[0]
        nobs = self.dependent.dataframe.shape[0]
        df_model = x.shape[1] + neffects
        df_resid = nobs - df_model
        cov_est, cov_config = self._choose_cov(cov_type, **cov_config)
        if auto_df:
            count_effects = self._determine_df_adjustment(cov_type, **cov_config)
        extra_df = neffects if count_effects else 0

        cov = cov_est(y, x, params, self.dependent.entity_ids, self.dependent.time_ids,
                      debiased=debiased, extra_df=extra_df, **cov_config)
        weps = y - x @ params
        eps = weps
        _y = self.dependent.values2d
        _x = self.exog.values2d
        if weighted:
            eps = (_y - y_effects) - (_x - x_effects) @ params
            if self.has_constant:
                # Correction since y_effecs and x_effects @ params add mean
                w = self.weights.values2d
                eps -= (w * eps).sum() / w.sum()

        eps_effects = _y - _x @ params
        sigma2_tot = float(eps_effects.T @ eps_effects / nobs)
        sigma2_eps = float(eps.T @ eps / nobs)
        sigma2_effects = sigma2_tot - sigma2_eps
        rho = sigma2_effects / sigma2_tot

        resid_ss = float(weps.T @ weps)
        if self.has_constant:
            mu = ybar
        else:
            mu = 0
        total_ss = float((y - mu).T @ (y - mu))
        r2 = 1 - resid_ss / total_ss

        root_w = np.sqrt(self.weights.values2d)
        y_ex = root_w * self.dependent.values2d
        mu_ex = 0
        if self.has_constant or self.entity_effects or self.time_effects or self.other_effects:
            mu_ex = root_w * ((root_w.T @ y_ex) / (root_w.T @ root_w))
        total_ss_ex_effect = float((y_ex - mu_ex).T @ (y_ex - mu_ex))
        r2_ex_effects = 1 - resid_ss / total_ss_ex_effect

        res = self._postestimation(params, cov, debiased, df_resid, weps, y, x, root_w)
        ######################################
        # Pooled f-stat
        ######################################
        if self.entity_effects or self.time_effects or self.other_effects:
            wy, wx = root_w * self.dependent.values2d, root_w * self.exog.values2d
            df_num, df_denom = (df_model - wx.shape[1]), df_resid
            if not self.has_constant:
                # Correction for when models does not have explicit constant
                wy -= root_w * np.linalg.lstsq(root_w, wy)[0]
                wx -= root_w * np.linalg.lstsq(root_w, wx)[0]
                df_num -= 1
            weps_pooled = wy - wx @ np.linalg.lstsq(wx, wy)[0]
            resid_ss_pooled = float(weps_pooled.T @ weps_pooled)
            num = (resid_ss_pooled - resid_ss) / df_num

            denom = resid_ss / df_denom
            stat = num / denom
            f_pooled = WaldTestStatistic(stat, 'Effects are zero',
                                         df_num, df_denom=df_denom,
                                         name='Pooled F-statistic')
            res.update(f_pooled=f_pooled)
            effects = pd.DataFrame(eps_effects - eps, columns=['effects'],
                                   index=self.dependent.index)
        else:
            effects = pd.DataFrame(np.zeros_like(eps), columns=['effects'],
                                   index=self.dependent.index)

        res.update(dict(df_resid=df_resid, df_model=df_model, nobs=y.shape[0],
                        residual_ss=resid_ss, total_ss=total_ss, wresids=weps, resids=eps,
                        r2=r2, entity_effects=self.entity_effects, time_effects=self.time_effects,
                        other_effects=self.other_effects, sigma2_eps=sigma2_eps,
                        sigma2_effects=sigma2_effects, rho=rho, r2_ex_effects=r2_ex_effects,
                        effects=effects))

        return PanelEffectsResults(res)


class BetweenOLS(PooledOLS):
    r"""
    Between estimator for panel data

    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array-like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        \bar{y}_{i}=  \beta^{\prime}\bar{x}_{i}+\bar{\epsilon}_{i}

    where :math:`\bar{z}` is the time-average.
    """

    def __init__(self, dependent, exog, *, weights=None):
        super(BetweenOLS, self).__init__(dependent, exog, weights=weights)
        self._cov_estimators = CovarianceManager(self.__class__.__name__, HomoskedasticCovariance,
                                                 HeteroskedasticCovariance, ClusteredCovariance)

    def _choose_cov(self, cov_type, **cov_config):
        """Return covariance estimator reformat clusters"""
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

    def fit(self, *, reweight=False, cov_type='unadjusted', debiased=True, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        reweight : bool
            Flag indicating to reweight observations if the input data is
            unbalanced using a WLS estimator.  If weights are provided, these
            are accounted for when reweighting. Has no effect on balanced data.
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        results :  PanelResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import BetweenOLS
        >>> mod = BetweenOLS(y, x)
        >>> res = mod.fit(cov_type='robust')

        Notes
        -----
        Three covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables

        When using a clustered covariance estimator, all cluster ids must be
        identical within an entity.
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
        cov = cov_est(wy, wx, params, self.dependent.entity_ids, self.dependent.time_ids,
                      debiased=debiased, **cov_config)
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
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array-like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array-like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        model : BetweenOLS
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import BetweenOLS
        >>> mod = BetweenOLS.from_formula('y ~ 1 + x1', data)
        >>> res = mod.fit()
        """
        return super(BetweenOLS, cls).from_formula(formula, data, weights=weights)


class FirstDifferenceOLS(PooledOLS):
    r"""
    First difference model for panel data

    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array-like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

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
        """Return covariance estimator and reformat clusters"""
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
                                  index=self.dependent.index,
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
        clusters = clusters.loc[self.dependent.first_difference().index]
        clusters = clusters.astype(np.int64)

        cov_config_upd['clusters'] = clusters.values if clusters is not None else clusters

        return cov_est, cov_config_upd

    def fit(self, *, cov_type='unadjusted', debiased=True, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        results : PanelResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import FirstDifferenceOLS
        >>> mod = FirstDifferenceOLS(y, x)
        >>> res = mod.fit(cov_type='robust')
        >>> res = mod.fit(cov_type='cluster', cluster_entity=True)

        Notes
        -----
        Three covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters

        * 'kernel' - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.

        When using a clustered covariance estimator, all cluster ids must be
        identical within a first difference.  In most scenarios, this requires
        ids to be identical within an entity.
        """
        y = self.dependent.first_difference()
        time_ids = y.time_ids
        entity_ids = y.entity_ids
        index = y.index
        y = y.values2d
        x = self.exog.first_difference().values2d

        if np.all(self.weights.values2d == 1.0):
            w = root_w = np.ones_like(y)
        else:
            w = 1.0 / self.weights.values3d
            w = w[:, :-1] + w[:, 1:]
            w = 1.0 / w
            w = pd.Panel(w, items=self.weights.panel.items,
                         major_axis=self.weights.panel.major_axis[1:],
                         minor_axis=self.weights.panel.minor_axis)
            w = w.swapaxes(1, 2).to_frame(filter_observations=False)
            w = w.reindex(self.weights.index).dropna(how='any')
            index = w.index
            w = w.values

            w /= w.mean()
            root_w = np.sqrt(w)

        wx = root_w * x
        wy = root_w * y
        params = lstsq(wx, wy)[0]
        df_resid = y.shape[0] - x.shape[1]
        cov_est, cov_config = self._choose_cov(cov_type, **cov_config)
        cov = cov_est(wy, wx, params, entity_ids, time_ids, debiased=debiased, **cov_config)

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
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array-like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array-like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        model : FirstDifferenceOLS
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import FirstDifferenceOLS
        >>> mod = FirstDifferenceOLS.from_formula('y ~ 1 + x1', data)
        >>> res = mod.fit()
        """
        return super(FirstDifferenceOLS, cls).from_formula(formula, data, weights=weights)


class RandomEffects(PooledOLS):
    r"""
    One-way Random Effects model for panel data

    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array-like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        y_{it} = \beta^{\prime}x_{it} + u_i + \epsilon_{it}

    where :math:`u_i` is a shock that is independent of :math:`x_{it}` but
    common to all entities i.
    """

    @classmethod
    def from_formula(cls, formula, data, *, weights=None):
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array-like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array-like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        model : RandomEffects
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import RandomEffects
        >>> mod = RandomEffects.from_formula('y ~ 1 + x1', data)
        >>> res = mod.fit()
        """
        return super(RandomEffects, cls).from_formula(formula, data, weights=weights)

    def fit(self, *, small_sample=False, cov_type='unadjusted', debiased=True, **cov_config):
        w = self.weights.values2d
        root_w = np.sqrt(w)
        y = self.dependent.demean('entity', weights=self.weights).values2d
        x = self.exog.demean('entity', weights=self.weights).values2d
        if self.has_constant:
            w_sum = w.sum()
            y_gm = (w * self.dependent.values2d).sum(0) / w_sum
            x_gm = (w * self.exog.values2d).sum(0) / w_sum
            y += root_w * y_gm
            x += root_w * x_gm
        params = np.linalg.lstsq(x, y)[0]
        weps = y - x @ params

        wybar = self.dependent.mean('entity', weights=self.weights)
        wxbar = self.exog.mean('entity', weights=self.weights)
        params = np.linalg.lstsq(wxbar, wybar)[0]
        wu = wybar.values - wxbar.values @ params

        nobs = weps.shape[0]
        neffects = wu.shape[0]
        nvar = x.shape[1]
        sigma2_e = float(weps.T @ weps) / (nobs - nvar - neffects + 1)
        ssr = float(wu.T @ wu)
        t = self.dependent.count('entity').values
        unbalanced = np.ptp(t) != 0
        if small_sample and unbalanced:
            ssr = float((t * wu).T @ wu)
            wx = root_w * self.exog.dataframe
            means = wx.groupby(level=0).transform('mean').values
            denom = means.T @ means
            sums = wx.groupby(level=0).sum().values
            num = sums.T @ sums
            tr = np.trace(np.linalg.inv(denom) @ num)
            sigma2_u = max(0, (ssr - (neffects - nvar) * sigma2_e) / (nobs - tr))
        else:
            t_bar = neffects / ((1.0 / t).sum())
            sigma2_u = max(0, ssr / (neffects - nvar) - sigma2_e / t_bar)
        rho = sigma2_u / (sigma2_u + sigma2_e)

        theta = 1 - np.sqrt(sigma2_e / (t * sigma2_u + sigma2_e))
        theta_out = pd.DataFrame(theta, columns=['theta'], index=wybar.index)
        wy = root_w * self.dependent.values2d
        wx = root_w * self.exog.values2d
        index = self.dependent.index
        reindex = index.levels[0][index.labels[0]]
        wybar = (theta * wybar).loc[reindex]
        wxbar = (theta * wxbar).loc[reindex]
        wy -= wybar.values
        wx -= wxbar.values
        params = np.linalg.lstsq(wx, wy)[0]

        df_resid = wy.shape[0] - wx.shape[1]
        cov_est, cov_config = self._choose_cov(cov_type, **cov_config)
        cov = cov_est(wy, wx, params, self.dependent.entity_ids, self.dependent.time_ids,
                      debiased=debiased, **cov_config)

        weps = wy - wx @ params
        eps = weps / root_w
        residual_ss = float(weps.T @ weps)
        wmu = 0
        if self.has_constant:
            wmu = root_w * lstsq(root_w, wy)[0]
        wy_demeaned = wy - wmu
        total_ss = float(wy_demeaned.T @ wy_demeaned)
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(params, cov, debiased, df_resid, weps, wy, wx, root_w)
        res.update(dict(df_resid=df_resid, df_model=x.shape[1], nobs=y.shape[0],
                        residual_ss=residual_ss, total_ss=total_ss, r2=r2,
                        resids=eps, wresids=weps, index=index, sigma2_eps=sigma2_e,
                        sigma2_effects=sigma2_u, rho=rho, theta=theta_out))

        return RandomEffectsResults(res)


class FamaMacBeth(PooledOLS):
    r"""
    Pooled coefficient estimator for panel data

    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity)
    exog : array-like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array-like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\beta^{\prime}x_{it}+\epsilon_{it}

    The Fama-MacBeth estimator is computed by performing T regressions, one
    for each time period using all availabl entity observations.  Denote the
    estimate of the model parameters as :math:`\hat{\beta}_t`.  The reported
    estimator is then

    .. math::

        \hat{\beta} = T^{-1}\sum_{t=1}^T \hat{\beta}_t

    While the model does not explicitly include time-effects, the
    implementation based on regressing all observation in a single
    time period is "as-if" time effects are included.

    Parameter inference is made using the set T parameter estimates using
    either the standard covariance estiamtor or a kernel-based covariance,
    depending on ``cov_type``.
    """

    def __init__(self, dependent, exog, *, weights=None):
        super(FamaMacBeth, self).__init__(dependent, exog, weights=weights)

    def fit(self, cov_type='unadjusted', debiased=True, **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        results :  PanelResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import FamaMacBeth
        >>> mod = FamaMacBeth(y, x)
        >>> res = mod.fit(cov_type='kernel', kernel='Parzen')

        Notes
        -----
        Four covariance estimators are supported:

        * 'unadjusted', 'homoskedastic', 'robust', 'heteroskedastic' - Use the
          standard covariance estimator of the T parameter estimates.
        * 'kernel' - HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is implements the the
            Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """
        y = self.dependent.dataframe
        x = self.exog.dataframe
        yx = pd.DataFrame(np.c_[y.values, x.values],
                          columns=list(y.columns) + list(x.columns),
                          index=y.index)

        def single(z: pd.DataFrame):
            x = z.iloc[:, 1:].values
            if x.shape[0] < x.shape[1]:
                return pd.Series([np.nan] * len(z.columns), index=z.columns)
            y = z.iloc[:, :1].values
            params = np.linalg.lstsq(x, y)[0]
            return pd.Series(np.r_[np.nan, params.ravel()], index=z.columns)

        all_params = yx.groupby(level=1).apply(single)
        all_params = all_params.iloc[:, 1:]
        params = all_params.mean(0).values[:, None]
        all_params = all_params.values

        # df_resid = params.shape[1]
        wy = self.dependent.values2d
        wx = self.exog.values2d
        weps = eps = self.dependent.values2d - self.exog.values2d @ params
        w = self.weights.values2d
        root_w = np.sqrt(w)
        #
        residual_ss = float(weps.T @ weps)
        e = y
        if self.has_constant:
            e = y - (w * y).sum() / w.sum()
        total_ss = float(w.T @ (e ** 2))
        r2 = 1 - residual_ss / total_ss

        if cov_type in ('robust', 'unadjusted', 'homoskedastic', 'heteroskedastic'):
            cov_est = FamaMacBethCovariance
        elif cov_type == 'kernel':
            cov_est = FamaMacBethKernelCovariance
        else:
            raise ValueError('Unknown cov_type')

        cov = cov_est(wy, wx, params, all_params, debiased=debiased, **cov_config)
        df_resid = wy.shape[0] - params.shape[0]
        res = self._postestimation(params, cov, debiased, df_resid, weps, wy, wx, root_w)
        index = self.dependent.index
        res.update(dict(df_resid=df_resid, df_model=x.shape[1], nobs=y.shape[0],
                        residual_ss=residual_ss, total_ss=total_ss,
                        r2=r2, resids=eps, wresids=weps, index=index))
        return PanelResults(res)
