import datetime as dt

import numpy as np
from numpy import diag, sqrt
from pandas import DataFrame, Series, concat
from scipy import stats
from statsmodels.iolib.summary import SimpleTable, fmt_2cols, fmt_params

from linearmodels.compat.statsmodels import Summary
from linearmodels.iv.results import default_txt_fmt, stub_concat, table_concat
from linearmodels.utility import (_ModelComparison, _str, _SummaryStr,
                                  cached_property, pval_format)

__all__ = ['PanelResults', 'PanelEffectsResults', 'RandomEffectsResults']


class PanelResults(_SummaryStr):
    """
    Results container for panel data models that do not include effects
    """

    def __init__(self, res):
        self._params = res.params.squeeze()
        self._deferred_cov = res.deferred_cov
        self._debiased = res.debiased
        self._df_resid = res.df_resid
        self._df_model = res.df_model
        self._nobs = res.nobs
        self._name = res.name
        self._var_names = res.var_names
        self._residual_ss = res.residual_ss
        self._total_ss = res.total_ss
        self._r2 = res.r2
        self._r2w = res.r2w
        self._r2b = res.r2b
        self._r2o = res.r2o
        self._s2 = res.s2
        self._entity_info = res.entity_info
        self._time_info = res.time_info
        self.model = res.model
        self._cov_type = res.cov_type
        self._datetime = dt.datetime.now()
        self._resids = res.resids
        self._wresids = res.wresids
        self._index = res.index
        self._deferred_f = res.deferred_f
        self._f_stat = res.f_stat
        self._loglik = res.loglik

    @property
    def params(self):
        """Estimated parameters"""
        return Series(self._params, index=self._var_names, name='parameter')

    @cached_property
    def cov(self):
        """Estimated covariance of parameters"""
        return DataFrame(self._deferred_cov(),
                         columns=self._var_names,
                         index=self._var_names)

    @property
    def std_errors(self):
        """Estimated parameter standard errors"""
        return Series(sqrt(diag(self.cov)), self._var_names, name='std_error')

    @property
    def tstats(self):
        """Parameter t-statistics"""
        return Series(self._params / self.std_errors, name='tstat')

    @cached_property
    def pvalues(self):
        """
        Parameter p-vals. Uses t(df_resid) if ``debiased`` is True, else normal
        """
        abs_tstats = np.abs(self.tstats)
        if self._debiased:
            pv = 2 * (1 - stats.t.cdf(abs_tstats, self.df_resid))
        else:
            pv = 2 * (1 - stats.norm.cdf(abs_tstats))
        return Series(pv, index=self._var_names, name='pvalue')

    @property
    def df_resid(self):
        """
        Residual degree of freedom

        Notes
        -----
        Defined as nobs minus nvar minus the number of included effects, if any.
        """
        return self._df_resid

    @property
    def df_model(self):
        """
        Model degree of freedom

        Notes
        -----
        Defined as nvar plus the number of included effects, if any.
        """
        return self._df_model

    @property
    def nobs(self):
        """Number of observations used to estimate the model"""
        return self._nobs

    @property
    def name(self):
        """Model name"""
        return self._name

    @property
    def total_ss(self):
        """Total sum of squares"""
        return self._total_ss

    @property
    def model_ss(self):
        """Residual sum of squares"""
        return self._total_ss - self._residual_ss

    @property
    def resid_ss(self):
        """Residual sum of squares"""
        return self._residual_ss

    @property
    def rsquared(self):
        """Model Coefficient of determination"""
        return self._r2

    @property
    def rsquared_between(self):
        """Between Coefficient of determination

        Returns
        -------
        rsquared : float
            Between coefficient of determination

        Notes
        -----
        The between rsquared measures the fit of the time-averaged dependent
        variable on the time averaged dependent variables.
        """
        return self._r2b

    @property
    def rsquared_within(self):
        """Within coefficient of determination

        Returns
        -------
        rsquared : float
            Within coefficient of determination

        Notes
        -----
        The within rsquared measures the fit of the dependent purged of entity
        effects on the exogenous purged of entity effects.
        """
        return self._r2w

    @property
    def rsquared_overall(self):
        """Overall coefficient of determination

        Returns
        -------
        rsquared : float
            Between coefficient of determination

        Notes
        -----
        The overall rsquared measures the fit of the dependent
        variable on the dependent variables ignoring any included effects.
        """

        return self._r2o

    @property
    def s2(self):
        """Residual variance estimator"""
        return self._s2

    @property
    def entity_info(self):
        """Statistics on observations per entity"""
        return self._entity_info

    @property
    def time_info(self):
        """Statistics on observations per time interval"""
        return self._time_info

    def conf_int(self, level=0.95):
        """
        Confidence interval construction

        Parameters
        ----------
        level : float
            Confidence level for interval

        Returns
        -------
        ci : DataFrame
            Confidence interval of the form [lower, upper] for each parameters

        Notes
        -----
        Uses a t(df_resid) if ``debiased`` is True, else normal.
        """
        ci_quantiles = [(1 - level) / 2, 1 - (1 - level) / 2]
        if self._debiased:
            q = stats.t.ppf(ci_quantiles, self.df_resid)
        else:
            q = stats.norm.ppf(ci_quantiles)
        q = q[None, :]
        ci = self.params[:, None] + self.std_errors[:, None] * q
        return DataFrame(ci, index=self._var_names, columns=['lower', 'upper'])

    @property
    def summary(self):
        """Summary table of model estimation results"""

        title = self.name + ' Estimation Summary'
        mod = self.model

        top_left = [('Dep. Variable:', mod.dependent.vars[0]),
                    ('Estimator:', self.name),
                    ('No. Observations:', self.nobs),
                    ('Date:', self._datetime.strftime('%a, %b %d %Y')),
                    ('Time:', self._datetime.strftime('%H:%M:%S')),
                    ('Cov. Estimator:', self._cov_type),
                    ('', ''),
                    ('Entities:', str(int(self.entity_info['total']))),
                    ('Avg Obs:', _str(self.entity_info['mean'])),
                    ('Min Obs:', _str(self.entity_info['min'])),
                    ('Max Obs:', _str(self.entity_info['max'])),
                    ('', ''),
                    ('Time periods:', str(int(self.time_info['total']))),
                    ('Avg Obs:', _str(self.time_info['mean'])),
                    ('Min Obs:', _str(self.time_info['min'])),
                    ('Max Obs:', _str(self.time_info['max'])),
                    ('', '')]

        is_invalid = np.isfinite(self.f_statistic.stat)
        f_stat = _str(self.f_statistic.stat) if is_invalid else '--'
        f_pval = pval_format(self.f_statistic.pval) if is_invalid else '--'
        f_dist = self.f_statistic.dist_name if is_invalid else '--'

        f_robust = _str(self.f_statistic_robust.stat) if is_invalid else '--'
        f_robust_pval = pval_format(self.f_statistic_robust.pval) if is_invalid else '--'
        f_robust_name = self.f_statistic_robust.dist_name if is_invalid else '--'

        top_right = [('R-squared:', _str(self.rsquared)),
                     ('R-squared (Between):', _str(self.rsquared_between)),
                     ('R-squared (Within):', _str(self.rsquared_within)),
                     ('R-squared (Overall):', _str(self.rsquared_overall)),
                     ('Log-likelihood', _str(self._loglik)),
                     ('', ''),
                     ('F-statistic:', f_stat),
                     ('P-value', f_pval),
                     ('Distribution:', f_dist),
                     ('', ''),
                     ('F-statistic (robust):', f_robust),
                     ('P-value', f_robust_pval),
                     ('Distribution:', f_robust_name),
                     ('', ''),
                     ('', ''),
                     ('', ''),
                     ('', ''),
                     ]

        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # create summary table instance
        smry = Summary()
        # Top Table
        # Parameter table
        fmt = fmt_2cols
        fmt['data_fmts'][1] = '%18s'

        top_right = [('%-21s' % ('  ' + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        param_data = np.c_[self.params.values[:, None],
                           self.std_errors.values[:, None],
                           self.tstats.values[:, None],
                           self.pvalues.values[:, None],
                           self.conf_int()]
        data = []
        for row in param_data:
            txt_row = []
            for i, v in enumerate(row):
                f = _str
                if i == 3:
                    f = pval_format
                txt_row.append(f(v))
            data.append(txt_row)
        title = 'Parameter Estimates'
        table_stubs = list(self.params.index)
        header = ['Parameter', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']
        table = SimpleTable(data,
                            stubs=table_stubs,
                            txt_fmt=fmt_params,
                            headers=header,
                            title=title)
        smry.tables.append(table)

        return smry

    @property
    def resids(self):
        """Model residuals"""
        return Series(self._resids.squeeze(), index=self._index, name='residual')

    @property
    def wresids(self):
        """Weighted model residuals"""
        return Series(self._wresids.squeeze(), index=self._index, name='weighted residual')

    @property
    def f_statistic_robust(self):
        r"""
        Joint test of significance for non-constant regressors

        Returns
        -------
        f_stat : WaldTestStatistic
            Statistic value, distribution and p-value

        Notes
        -----
        Implemented as a Wald test using the estimated parameter covariance,
        and so inherits any robustness that the choice of covariance estimator
        provides.

        .. math::

           W = \hat{\beta}_{-}' \hat{\Sigma}_{-}^{-1} \hat{\beta}_{-}

        where :math:`\hat{\beta}_{-}` does not include the model constant and
        :math:`\hat{\Sigma}_{-}` is tht estimated covariance of the
        parameters, also excluding the constant.  The test statistic is
        distributed as :math:`\chi^2_{k}` where k is the number of non-
        constant parameters.

        If ``debiased`` is True, then the Wald statistic is divided by the
        number of restrictions and inference is made using an :math:`F_{k,df}`
        distribution where df is the residual degree of freedom from the model.
        """
        return self._deferred_f()

    @property
    def f_statistic(self):
        r"""
        Joint test of significance for non-constant regressors

        Returns
        -------
        f_stat : WaldTestStatistic
            Statistic value, distribution and p-value

        Notes
        -----
        Classical F-stat that is only correct under an assumption of
        homoskedasticity.  The test statistic is defined as

        .. math::

          F = \frac{(RSS_R - RSS_U)/ k}{RSS_U / df_U}

        where :math:`RSS_R` is the restricted sum of squares from the model
        where the coefficients on all exog variables is zero, excluding a
        constant if one was included. :math:`RSS_U` is the unrestricted
        residual sum of squares.  k is the number of non-constant regressors
        in the model and :math:`df_U` is the residual degree of freedom in the
        unrestricted model.  The test has an :math:`F_{k,df_U}` distribution.
        """
        return self._f_stat

    @property
    def loglik(self):
        """Log-likelihood of model"""
        return self._loglik


class PanelEffectsResults(PanelResults):
    """
    Results container for panel data models that include effects
    """

    def __init__(self, res):
        super(PanelEffectsResults, self).__init__(res)
        self._other_info = res.other_info
        self._f_pooled = res.f_pooled
        self._entity_effect = res.entity_effects
        self._time_effect = res.time_effects
        self._other_effect = res.other_effects
        self._rho = res.rho
        self._sigma2_eps = res.sigma2_eps
        self._sigma2_effects = res.sigma2_effects
        self._r2_ex_effects = res.r2_ex_effects
        self._effects = res.effects

    @property
    def f_pooled(self):
        r"""
        Test that included effects are jointly zero.

        Returns
        -------
        f_pooled : WaldTestStatistic
            Statistic value, distribution and p-value

        Notes
        -----
        Joint test that all included effects are zero.  Only correct under an
        assumption of homoskedasticity.

        The test statistic is defined as

        .. math::

          F = \frac{(RSS_{pool}-RSS_{effect})/(df_{pool}-df_{effect})}{RSS_{effect}/df_{effect}}

        where :math:`RSS_{pool}` is the residual sum of squares from a no-
        effect (pooled) model. :math:`RSS_{effect}` is the residual sum of
        squares from a model with effects.  :math:`df_{pool}` is the residual
        degree of freedom in the pooled regression and :math:`df_{effect}` is
        the residual degree of freedom from the model with effects. The test
        has an :math:`F_{k,df_{effect}}` distribution where
        :math:`k=df_{pool}-df_{effect}`.
        """
        return self._f_pooled

    @property
    def included_effects(self):
        """List of effects included in the model"""
        entity_effect = self._entity_effect
        time_effect = self._time_effect
        other_effect = self._other_effect
        if entity_effect or time_effect or other_effect:
            effects = []
            if entity_effect:
                effects.append('Entity')
            if time_effect:
                effects.append('Time')
            if other_effect:
                oe = self.model._other_effect_cats.dataframe
                for c in oe:
                    effects.append('Other Effect (' + str(c) + ')')
        else:
            effects = []
        return effects

    @property
    def other_info(self):
        """Statistics on observations per group for other effects"""
        return self._other_info

    @property
    def rsquared_inclusive(self):
        """Model Coefficient of determination including fit of included effects"""
        return self._r2_ex_effects

    @property
    def summary(self):
        """Summary table of model estimation results"""
        smry = super(PanelEffectsResults, self).summary

        is_invalid = np.isfinite(self.f_pooled.stat)
        f_pool = _str(self.f_pooled.stat) if is_invalid else '--'
        f_pool_pval = pval_format(self.f_pooled.pval) if is_invalid else '--'
        f_pool_name = self.f_pooled.dist_name if is_invalid else '--'

        extra_text = []
        if is_invalid:
            extra_text.append('F-test for Poolability: {0}'.format(f_pool))
            extra_text.append('P-value: {0}'.format(f_pool_pval))
            extra_text.append('Distribution: {0}'.format(f_pool_name))
            extra_text.append('')

        if self.included_effects:
            effects = ', '.join(self.included_effects)
            extra_text.append('Included effects: ' + effects)

        if self.other_info is not None:
            ncol = self.other_info.shape[1]
            extra_text.append('Model includes {0} other effects'.format(ncol))
            for c in self.other_info.T:
                col = self.other_info.T[c]
                extra_text.append('Other Effect {0}:'.format(c))
                stats = 'Avg Obs: {0}, Min Obs: {1}, Max Obs: {2}, Groups: {3}'
                stats = stats.format(_str(col['mean']), _str(col['min']), _str(col['max']),
                                     int(col['total']))
                extra_text.append(stats)

        smry.add_extra_txt(extra_text)

        return smry

    @property
    def estimated_effects(self):
        """Estimated effects"""
        return self._effects

    @property
    def variance_decomposition(self):
        """Decomposition of total variance into effects and residuals"""
        vals = [self._sigma2_effects, self._sigma2_eps, self._rho]
        index = ['Effects', 'Residual', 'Percent due to Effects']
        return Series(vals, index=index, name='Variance Decomposition')


class RandomEffectsResults(PanelResults):
    """
    Results container for random effect panel data models
    """

    def __init__(self, res):
        super(RandomEffectsResults, self).__init__(res)
        self._theta = res.theta
        self._sigma2_effects = res.sigma2_effects
        self._sigma2_eps = res.sigma2_eps
        self._rho = res.rho

    @property
    def variance_decomposition(self):
        """Decomposition of total variance into effects and residuals"""
        vals = [self._sigma2_effects, self._sigma2_eps, self._rho]
        index = ['Effects', 'Residual', 'Percent due to Effects']
        return Series(vals, index=index, name='Variance Decomposition')

    @property
    def theta(self):
        """Values used in generalized demeaning"""
        return self._theta


def compare(results):
    """
    Compare the results of multiple models

    Parameters
    ----------
    results : {list, dict, OrderedDict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.  An OrderedDict will preserve the model order the comparisons.

    Returns
    -------
    comparison : PanelModelComparison
    """
    return PanelModelComparison(results)


class PanelModelComparison(_ModelComparison):
    """
    Comparison of multiple models

    Parameters
    ----------
    results : {list, dict, OrderedDict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.  An OrderedDict will preserve the model order the comparisons.
    """
    _supported = (PanelEffectsResults, PanelResults, RandomEffectsResults)

    def __init__(self, results):
        super(PanelModelComparison, self).__init__(results)

    @property
    def rsquared_between(self):
        """Coefficients of determination (R**2)"""
        return self._get_property('rsquared_between')

    @property
    def rsquared_within(self):
        """Coefficients of determination (R**2)"""
        return self._get_property('rsquared_within')

    @property
    def rsquared_overall(self):
        """Coefficients of determination (R**2)"""
        return self._get_property('rsquared_overall')

    @property
    def estimator_method(self):
        """Estimation methods"""
        return self._get_property('name')

    @property
    def cov_estimator(self):
        """Covariance estimator descriptions"""
        return self._get_property('_cov_type')

    @property
    def summary(self):
        """Summary table of model comparison"""
        smry = Summary()
        models = list(self._results.keys())
        title = 'Model Comparison'
        stubs = ['Dep. Variable', 'Estimator', 'No. Observations', 'Cov. Est.', 'R-squared',
                 'R-Squared (Within)', 'R-Squared (Between)', 'R-Squared (Overall)',
                 'F-statistic', 'P-value (F-stat)']
        dep_name = {}
        for key in self._results:
            dep_name[key] = self._results[key].model.dependent.vars[0]
        dep_name = Series(dep_name)

        vals = concat([dep_name, self.estimator_method, self.nobs, self.cov_estimator,
                       self.rsquared, self.rsquared_within, self.rsquared_between,
                       self.rsquared_overall, self.f_statistic], 1)
        vals = [[i for i in v] for v in vals.T.values]
        vals[2] = [str(v) for v in vals[2]]
        for i in range(4, len(vals)):
            f = _str
            if i == 9:
                f = pval_format
            vals[i] = [f(v) for v in vals[i]]

        params = self.params
        tstats = self.tstats
        params_fmt = []
        params_stub = []
        for i in range(len(params)):
            params_fmt.append([_str(v) for v in params.values[i]])
            tstats_fmt = []
            for v in tstats.values[i]:
                v_str = _str(v)
                v_str = '({0})'.format(v_str) if v_str.strip() else v_str
                tstats_fmt.append(v_str)
            params_fmt.append(tstats_fmt)
            params_stub.append(params.index[i])
            params_stub.append(' ')

        vals = table_concat((vals, params_fmt))
        stubs = stub_concat((stubs, params_stub))

        all_effects = []
        for key in self._results:
            res = self._results[key]
            effects = getattr(res, 'included_effects', [])
            all_effects.append(effects)

        neffect = max(map(lambda l: len(l), all_effects))
        effects = []
        effects_stub = ['Effects']
        for i in range(neffect):
            if i > 0:
                effects_stub.append('')
            row = []
            for j in range(len(self._results)):
                effect = all_effects[j]
                if len(effect) > i:
                    row.append(effect[i])
                else:
                    row.append('')
            effects.append(row)
        if effects:
            vals = table_concat((vals, effects))
            stubs = stub_concat((stubs, effects_stub))

        txt_fmt = default_txt_fmt.copy()
        txt_fmt['data_aligns'] = 'r'
        txt_fmt['header_align'] = 'r'
        table = SimpleTable(vals, headers=models, title=title, stubs=stubs, txt_fmt=txt_fmt)
        smry.tables.append(table)
        smry.add_extra_txt(['T-stats reported in parentheses'])
        return smry
