import scipy.stats as stats
from numpy import c_, diag, log, ones, sqrt
from numpy.linalg import inv, pinv
from pandas import DataFrame, Series

from panel.utility import (InvalidTestStatistic, WaldTestStatistic,
                           _annihilate, _proj, cached_property)


class OLSResults(object):
    def __init__(self, results, model):
        self._resid = results['eps']
        self._params = results['params']
        self._cov = results['cov']
        self._model = model
        self._r2 = results['r2']
        self._cov_type = results['cov_type']
        self._rss = results['residual_ss']
        self._tss = results['total_ss']
        self._s2 = results['s2']
        self._debiased = results['debiased']
        self._f_statistic = results['fstat']
        self._vars = results['vars']
        self._cov_config = results['cov_config']
        self._method = results['method']
        self._kappa = results.get('kappa', None)
        self._liml_kappa = results.get('liml_kappa', None)

    @property
    def cov_config(self):
        """Parameter values from covariance estimator"""
        return self._cov_config

    @property
    def cov_estimator(self):
        """Type of covariance estimator used to compute covariance"""
        return self._cov_type

    @property
    def cov(self):
        """Estimated covariance of parameters"""
        return self._cov

    @property
    def params(self):
        """Estimated parameters"""
        return self._params

    @property
    def resids(self):
        """Estimated residuals"""
        return self._resid

    @property
    def nobs(self):
        """Number of observations"""
        return self._model.endog.shape[0]

    @property
    def df_resid(self):
        """Residual degree of freedom"""
        return self.nobs - self._model.exog.shape[1]

    @property
    def df_model(self):
        """Model degree of freedom"""
        return self._model._x.shape[1]

    @property
    def has_constant(self):
        """Flag indicating the model includes a constant or equivalent"""
        return self._model.has_constant

    @property
    def kappa(self):
        """k-class estimator value"""
        return self._kappa

    @property
    def rsquared(self):
        """Coefficient of determination (R**2)"""
        return self._r2

    @property
    def rsquared_adj(self):
        """Sample-size adjusted coefficient of determination (R**2)"""
        n, k, c = self.nobs, self.df_model, int(self.has_constant)
        return 1 - ((n - c) / (n - k)) * (1 - self._r2)

    @property
    def cov_type(self):
        """Covariance estimator used"""
        return self._cov_type

    @property
    def std_errors(self):
        """Estimated parameter standard errors"""
        std_errors = sqrt(diag(self.cov))
        return Series(std_errors, index=self._vars, name='stderr')

    @property
    def tstats(self):
        """Parameter t-statistics"""
        return self.params / self.std_errors

    @cached_property
    def pvalues(self):
        """
        Parameter p-vals. Uses t(df_resid) if debiased is True, other normal.
        """
        if self.debiased:
            pvals = 2 - 2 * stats.t.cdf(abs(self.tstats), self.df_resid)
        else:
            pvals = 2 - 2 * stats.norm.cdf(abs(self.tstats))

        return Series(pvals, index=self._vars, name='pvalue')

    @property
    def total_ss(self):
        """Total sum of squares"""
        return self._tss

    @property
    def model_ss(self):
        """Residual sum of squares"""
        return self._tss - self._rss

    @property
    def resid_ss(self):
        """Residual sum of squares"""
        return self._rss

    @property
    def s2(self):
        """Residual variance estimator"""
        return self._s2

    @property
    def debiased(self):
        """Flag indicating whether covariance uses a small-sample adjustment"""
        return self._debiased

    @property
    def f_statistic(self):
        """Joint test of significance for non-constant regressors"""
        return self._f_statistic

    @property
    def method(self):
        """Method used to estimate model parameters"""
        return self._method

    def conf_int(self, level=0.95):
        """
        Confidence interval construction

        Parameters
        ----------
        level : float
            Confidence level for interval

        Returns
        -------
        ci : ndarray
            Confidence interval of the form [lower, upper] for each parameters
        """
        q = stats.norm.ppf([(1 - level) / 2, 1 - (1 - level) / 2])
        q = q[None, :]
        ci = self.params[:, None] + self.std_errors[:, None] * q
        return DataFrame(ci, index=self._vars, columns=['lower', 'upper'])


class _CommonIVResults(OLSResults):
    """
    Results from IV estimation

    Notes
    -----
    .. todo::

        * Hypothesis testing
        * First stage diagnostics
        * Model diagnostics
    """

    def __init__(self, results, model):
        super(_CommonIVResults, self).__init__(results, model)
        self._kappa = results.get('kappa', 1)

    @property
    def first_stage(self):
        """
        First stage regression results

        Returns
        -------
        first : FirstStageResults
            Object containing results for diagnosing instrument relevance issues.
        """
        return FirstStageResults(self._model.dependent, self._model.exog,
                                 self._model.endog, self._model.instruments,
                                 self._cov_type, self._cov_config)


class IVResults(_CommonIVResults):
    """
    Results from IV estimation

    Notes
    -----
    .. todo::

        * Hypothesis testing
        * First stage diagnostics
        * Model diagnostics
    """

    def __init__(self, results, model):
        super(IVResults, self).__init__(results, model)
        self._kappa = results.get('kappa', 1)

    @cached_property
    def sargan(self):
        """
        Sargan test of overidentifying restrictions
        """
        z = self._model.instruments.ndarray
        nobs, ninstr = z.shape
        nendog = self._model.endog.shape[1]
        name = 'Sargan\'s test of overidentification'
        if ninstr - nendog == 0:
            return InvalidTestStatistic('Test requires more instruments than '
                                        'endogenous variables.', name=name)

        eps = self.resids.values[:, None]
        u = eps - z @ (pinv(z) @ eps)
        stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
        null = 'The model is not overidentified.'

        return WaldTestStatistic(stat, null, ninstr - nendog, name=name)

    @cached_property
    def basmann(self):
        """
        Basmann's test of overidentifying restrictions
        """
        mod = self._model
        nobs, ninstr = mod.instruments.shape
        nendog = mod.endog.shape[1]
        nvar = mod.exog.shape[1] + nendog
        name = 'Basmann\'s test of overidentification'
        if ninstr - nendog == 0:
            return InvalidTestStatistic('Test requires more instruments than '
                                        'endogenous variables.', name=name)
        sargan_test = self.sargan
        s = sargan_test.stat
        stat = s * (nobs - ninstr) / (nobs - nvar)
        return WaldTestStatistic(stat, sargan_test.null, sargan_test.df, name=name)

    @cached_property
    def durbin(self):
        """
        Durbin's test of exogeneity
        """
        exog_endog = c_[self._model.exog.ndarray, self._model.endog.ndarray]
        nendog = self._model.endog.shape[1]
        e_ols = _annihilate(self._model.dependent.ndarray, exog_endog)
        nobs = e_ols.shape[0]
        e_2sls = self.resids.values
        e_ols_pz = _proj(e_ols, self._model.instruments.ndarray)
        e_2sls_pz = _proj(e_2sls, self._model.instruments.ndarray)
        stat = e_ols_pz.T @ e_ols_pz - e_2sls_pz.T @ e_2sls_pz
        stat /= (e_ols.T @ e_ols) / nobs
        null = 'Endogenous variables are exogenous'
        name = 'Durbin test of exogeneity'
        return WaldTestStatistic(stat.squeeze(), null, nendog, name=name)

    @cached_property
    def wu_hausman(self):
        """
        Wu-Hausman test of exogeneity
        """
        durb = self.durbin
        exog_endog = c_[self._model.exog.ndarray, self._model.endog.ndarray]
        e_ols = _annihilate(self._model.dependent.ndarray, exog_endog)
        nobs = e_ols.shape[0]
        nendog, nexog = self._model.endog.shape[1], self._model.exog.shape[1]
        rss_ols = (e_ols ** 2).sum()
        delta = durb.stat * (rss_ols / nobs)
        df = nexog
        wh_num = delta / df
        df_denom = nobs - nexog - nendog - nendog
        wh_denom = (rss_ols - delta) / df_denom
        stat = wh_num / wh_denom
        name = 'Wu-Hausman test of exogeneity'
        return WaldTestStatistic(stat, durb.null, df, df_denom, name=name)

    @cached_property
    def wooldridge_score(self):
        """
        Wooldridge's score test of exogeneity 
        """
        from panel.iv.model import IV2SLS
        e = _annihilate(self._model.dependent.ndarray, self._model._x)
        r = _annihilate(self._model.endog.ndarray, self._model._z)
        res = IV2SLS(e, r, None, None).fit('unadjusted')
        stat = res.nobs * res.rsquared
        df = self._model.endog.shape[1]
        null = 'Endogenous variables are exogenous'
        name = 'Wooldridge\'s score test of exogeneity'
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def wooldridge_regression(self):
        """
        Wooldridge's regression test of exogeneity 
        """
        from panel.iv.model import IV2SLS
        r = _annihilate(self._model.endog.ndarray, self._model._z)
        augx = c_[self._model._x, r]
        mod = IV2SLS(self._model.dependent, augx, None, None)
        res = mod.fit(self.cov_type, **self.cov_config)
        norig = self._model._x.shape[1]
        test_params = res.params.values[norig:]
        test_cov = res.cov.values[norig:, norig:]
        stat = test_params.T @ inv(test_cov) @ test_params
        df = len(test_params)
        null = 'Endogenous variables are exogenous'
        name = 'Wooldridge\'s regression test of exogeneity'
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def wooldridge_overid(self):
        """
        Wooldridge's score test of overidentification 
        """
        from panel.iv.model import IV2SLS
        endog, instruments = self._model.endog, self._model.instruments
        proj_reg = _proj(self._model._z, self._model._z)
        nobs, nendog = endog.shape
        ninstr = instruments.shape[1]
        if ninstr - nendog == 0:
            import warnings
            warnings.warn('Test requires more instruments than '
                          'endogenous variables',
                          UserWarning)
            return WaldTestStatistic(0, 'Test is not feasible.', 1, name='Infeasible test.')

        q = instruments.ndarray[:, :(ninstr - nendog)]
        q_proj = _proj(q, proj_reg)
        resids = self.resids.values
        test_functions = q_proj * resids[:, None]
        mod = IV2SLS(ones((nobs, 1)), test_functions, None, None)
        res = mod.fit('unadjusted')
        stat = res.nobs * res.rsquared
        df = q.shape[1]
        null = 'Model is not overidentified.'
        name = 'Wooldridge\'s score test of overidentification'
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def anderson_rubin(self):
        """Anderson-Rubin test of overidentifying restrictions"""
        nobs, ninstr = self._model.instruments.shape
        nendog = self._model.endog.shape[1]
        name = 'Anderson-Rubin test of overidentification'
        if ninstr - nendog == 0:
            return InvalidTestStatistic('Test requires more instruments than '
                                        'endogenous variables.', name=name)
        stat = nobs * log(self._liml_kappa)
        df = ninstr - nendog
        null = 'The model is not overidentified.'
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def basmann_f(self):
        """Basmann's F test of overidentifying restrictions"""
        nobs, ninstr = self._model.instruments.shape
        nendog = self._model.endog.shape[1]
        name = 'Basmann\' F  test of overidentification'
        if ninstr - nendog == 0:
            return InvalidTestStatistic('Test requires more instruments than '
                                        'endogenous variables.', name=name)
        stat = (self._liml_kappa - 1) * (nobs - ninstr) / (ninstr - nendog)
        df = ninstr - nendog
        df_denom = nobs - ninstr
        null = 'The model is not overidentified.'
        return WaldTestStatistic(stat, null, df, df_denom=df_denom, name=name)


class IVGMMResults(_CommonIVResults):
    """
    Results from GMM estimation of IV models

    Notes
    -----
    .. todo::

        * Hypothesis testing
        * First stage diagnostics
        * Model diagnostics
    """

    def __init__(self, results, model):
        super(IVGMMResults, self).__init__(results, model)
        self._weight_mat = results['weight_mat']
        self._weight_type = results['weight_type']
        self._weight_config = results['weight_config']
        self._iterations = results['iterations']
        self._j_stat = results['j_stat']

    @property
    def weight_matrix(self):
        """Weight matrix used in the final-step GMM estimation"""
        return self._weight_mat

    @property
    def iterations(self):
        """Iterations used in GMM estimation"""
        return self._iterations

    @property
    def weight_type(self):
        """Weighting matrix method used in estimation"""
        return self._weight_type

    @property
    def weight_config(self):
        """Weighting matrix parameters used in estimation"""
        return self._weight_config

    @property
    def j_stat(self):
        """J-test of overidentifying restrictions"""
        return self._j_stat

    def c_stat(self):
        """C-test of endogeneity"""
        # TODO
        pass


class FirstStageResults(object):
    """
    First stage estimation results and diagnostics
    
    .. todo ::

      * Summary
    """

    def __init__(self, dep, exog, endog, instr, cov_type, cov_config):
        self.dep = dep
        self.exog = exog
        self.endog = endog
        self.instr = instr
        reg = c_[self.exog.ndarray, self.endog.ndarray]
        self._reg = DataFrame(reg, columns=self.exog.cols + self.endog.cols)
        self._cov_type = cov_type
        self._cov_config = cov_config
        self._fitted = {}

    @cached_property
    def diagnostics(self):
        """
        Post estimation diagnostics of first-stage fit

        Returns
        -------
        res : DataFrame
            DataFrame where each endogenous variable appears as a row and
            the columns contain alternative measures.  The columns are:

            * rsquared - Rsquared from regression of endogenous on exogenous
              and instruments
            * partial.rsquared - Rsquared from regression of the exogenous
              variable on instruments where both the exogenous variable and
              the instrument have been orthogonalized to the exogenous
              regressors in the model.   
            * f.stat - Test that all coefficients are zero in the model
              used to estimate the partial rsquared. Uses a standard F-test
              when the covariance estimtor is unadjusted - otherwise uses a
              Wald test statistic with a chi2 distribution.
            * f.pval - P-value of the test that all coefficients are zero
              in the model used to estimate the partial rsquared
            * shea.rsquared - Shea's r-squared which measures the correlation
              between the projected and orthogonalized instrument on the
              orthogonoalized endogenous regressor where the orthogonalization
              is with respect to the other included variables in the model.
        """
        from panel.iv.model import IV2SLS
        endog, exog, instr = self.endog, self.exog, self.instr
        z = instr.ndarray
        x = exog.ndarray
        px = x @ pinv(x)
        ez = z - px @ z
        out = {}
        individal_results = self.individual
        for col in endog.pandas:
            inner = {}
            inner['rsquared'] = individal_results[col].rsquared
            y = endog.pandas[[col]].values
            ey = y - px @ y
            mod = IV2SLS(ey, ez, None, None)
            res = mod.fit(self._cov_type, **self._cov_config)
            inner['partial.rsquared'] = res.rsquared
            params = res.params.values
            params = params[:, None]
            stat = params.T @ inv(res.cov) @ params
            stat = stat.squeeze()
            w = WaldTestStatistic(stat, null='', df=params.shape[0])
            inner['f.stat'] = w.stat
            inner['f.pval'] = w.pval
            out[col] = Series(inner)
        out = DataFrame(out).T

        dep = self.dep
        r2sls = IV2SLS(dep, exog, endog, instr).fit('unadjusted')
        rols = IV2SLS(dep, self._reg, None, None).fit('unadjusted')
        shea = (rols.std_errors / r2sls.std_errors) ** 2
        shea *= (1 - r2sls.rsquared) / (1 - rols.rsquared)
        out['shea.rsquared'] = shea[out.index]
        cols = ['rsquared', 'partial.rsquared', 'shea.rsquared', 'f.stat', 'f.pval']
        out = out[cols]
        return out

    @cached_property
    def individual(self):
        """
        Individual model results from first-stage regressions

        Returns
        -------
        res : dict
            Dictionary containing first stage estimation results. Keys are
            the variable names of the endogenous regressors.
        """
        from panel.iv.model import IV2SLS
        exog_instr = c_[self.exog.ndarray, self.instr.ndarray]
        res = {}
        for col in self.endog.pandas:
            mod = IV2SLS(self.endog.pandas[col], exog_instr, None, None)
            res[col] = mod.fit(self._cov_type, **self._cov_config)

        return res
