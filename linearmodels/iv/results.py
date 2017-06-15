"""
Results containers and post-estimation diagnostics for IV models
"""
import datetime as dt
from collections import OrderedDict

import scipy.stats as stats
from numpy import array, asarray, c_, diag, empty, log, ones, sqrt, zeros
from numpy.linalg import inv, pinv
from pandas import DataFrame, Series, concat, to_numeric
from statsmodels.iolib.summary import SimpleTable, fmt_2cols, fmt_params
from statsmodels.iolib.table import default_txt_fmt

from linearmodels.compat.statsmodels import Summary
from linearmodels.iv._utility import annihilate, proj
from linearmodels.utility import (InvalidTestStatistic, WaldTestStatistic,
                                  _ModelComparison, _str, _SummaryStr,
                                  cached_property, pval_format)


def stub_concat(lists, sep='='):
    col_size = max([max(map(lambda s: len(s), l)) for l in lists])
    out = []
    for l in lists:
        out.extend(l)
        out.append(sep * (col_size + 2))
    return out[:-1]


def table_concat(lists, sep='='):
    col_sizes = []
    for l in lists:
        size = list(map(lambda r: list(map(lambda v: len(v), r)), l))
        col_sizes.append(list(array(size).max(0)))
    col_size = array(col_sizes).max(axis=0)
    sep_cols = [sep * (cs + 2) for cs in col_size]
    out = []
    for l in lists:
        out.extend(l)
        out.append(sep_cols)
    return out[:-1]


class OLSResults(_SummaryStr):
    def __init__(self, results, model):
        self._resid = results['eps']
        self._wresid = results['weps']
        self._params = results['params']
        self._cov = results['cov']
        self.model = model
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
        self._datetime = dt.datetime.now()
        self._cov_estimator = results['cov_estimator']

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
    def wresids(self):
        """Weighted estimated residuals"""
        return self._wresid

    @property
    def nobs(self):
        """Number of observations"""
        return self.model.endog.shape[0]

    @property
    def df_resid(self):
        """Residual degree of freedom"""
        return self.nobs - self.model.exog.shape[1]

    @property
    def df_model(self):
        """Model degree of freedom"""
        return self.model._x.shape[1]

    @property
    def has_constant(self):
        """Flag indicating the model includes a constant or equivalent"""
        return self.model.has_constant

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
        return Series(self._params / self.std_errors, name='tstat')

    @cached_property
    def pvalues(self):
        """
        Parameter p-vals. Uses t(df_resid) if ``debiased`` is True, else normal
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
        """
        Model F-statistic

        Returns
        -------
        f : WaldTestStatistic
            Test statistic for null all coefficients excluding constant terms
            are zero.

        Notes
        -----
        Despite name, always implemented using a quadratic-form test based on
        estimated parameter covariance. Default is to use a chi2 distribution
        to compute p-values. If ``debiased`` is True, divides statistic by
        number of parameters tested and uses an F-distribution.

        This version of the F-statistic directly uses the model covariance
        estimator and so is robust against the same specification issues.
        """
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
        return DataFrame(ci, index=self._vars, columns=['lower', 'upper'])

    @property
    def summary(self):
        """Summary table of model estimation results"""

        title = self._method + ' Estimation Summary'
        mod = self.model
        top_left = [('Dep. Variable:', mod.dependent.cols[0]),
                    ('Estimator:', self._method),
                    ('No. Observations:', self.nobs),
                    ('Date:', self._datetime.strftime('%a, %b %d %Y')),
                    ('Time:', self._datetime.strftime('%H:%M:%S')),
                    ('Cov. Estimator:', self._cov_type),
                    ('', '')]

        top_right = [('R-squared:', _str(self.rsquared)),
                     ('Adj. R-squared:', _str(self.rsquared_adj)),
                     ('F-statistic:', _str(self.f_statistic.stat)),
                     ('P-value (F-stat)', pval_format(self.f_statistic.pval)),
                     ('Distribution:', str(self.f_statistic.dist_name)),
                     ('', ''),
                     ('', '')]

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

        param_data = c_[self.params.values[:, None],
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

        instruments = self.model.instruments
        if instruments.shape[1] > 0:
            extra_text = []
            endog = self.model.endog
            extra_text.append('Endogenous: ' + ', '.join(endog.cols))
            extra_text.append('Instruments: ' + ', '.join(instruments.cols))
            cov_descr = str(self._cov_estimator)
            for line in cov_descr.split('\n'):
                extra_text.append(line)
            smry.add_extra_txt(extra_text)

        return smry

    def test_linear_constraint(self, restriction, value):
        """
        Test linear equality constraints using a Wald test

        Parameters
        ----------
        restriction : {ndarray, DataFrame}
            q by nvar array containing linear weights to apply to parameters
            when formin the restrictions.
        value : {ndarray, Series}
            q element array containing the restircted values

        Returns
        -------
        t: WaldTestStatistic
            Test statistic for null that restrictions are valid.

        Notes
        -----
        Hypothesis test examines whether :math:`H_0:C\theta=v` where the
        matrix C is ``restriction`` and v is ``value``. The test statistic
        has a :math:`\chi^2_q` distribution where q is the number of rows in C.
        """
        restriction = asarray(restriction)
        value = asarray(value)[:, None]
        diff = restriction @ self.params.values[:, None] - value
        rcov = restriction @ self.cov @ restriction.T
        stat = float(diff.T @ inv(rcov) @ diff)
        df = restriction.shape[0]
        null = 'Linear equality constraint is valid'
        name = 'Linear Equality Hypothesis Test'
        return WaldTestStatistic(stat, null, df, name=name)


class _CommonIVResults(OLSResults):
    """
    Results from IV estimation
    """

    def __init__(self, results, model):
        super(_CommonIVResults, self).__init__(results, model)
        self._liml_kappa = results.get('liml_kappa', None)

    @property
    def first_stage(self):
        """
        First stage regression results

        Returns
        -------
        first : FirstStageResults
            Object containing results for diagnosing instrument relevance issues.
        """
        return FirstStageResults(self.model.dependent, self.model.exog,
                                 self.model.endog, self.model.instruments,
                                 self.model.weights, self._cov_type,
                                 self._cov_config)


class IVResults(_CommonIVResults):
    """
    Results from IV estimation
    """

    def __init__(self, results, model):
        super(IVResults, self).__init__(results, model)
        self._kappa = results.get('kappa', 1)

    @cached_property
    def sargan(self):
        """
        Sargan test of overidentifying restrictions

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Requires more instruments than endogenous variables

        Tests the ratio of re-projected IV regression residual variance to
        variance of the IV residuals.

        .. math ::

          n(1-\hat{\epsilon}^{\prime}M_{Z}\hat{\epsilon}/
          \hat{\epsilon}^{\prime}\hat{\epsilon})\sim\chi_{v}^{2}

        where :math:`M_{z}` is the annihilator matrix where z is the set of
        instruments and :math:`\hat{\epsilon}` are the residuals from the IV
        estimator.  The degree of freedom is the difference between the number
        of instruments and the number of endogenous regressors.

        .. math ::

          v = n_{instr} - n_{exog}
        """
        z = self.model.instruments.ndarray
        nobs, ninstr = z.shape
        nendog = self.model.endog.shape[1]
        name = 'Sargan\'s test of overidentification'
        if ninstr - nendog == 0:
            return InvalidTestStatistic('Test requires more instruments than '
                                        'endogenous variables.', name=name)

        eps = self.resids.values[:, None]
        u = annihilate(eps, self.model._z)
        stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
        null = 'The model is not overidentified.'

        return WaldTestStatistic(stat, null, ninstr - nendog, name=name)

    @cached_property
    def basmann(self):
        """
        Basmann's test of overidentifying restrictions

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Requires more instruments than endogenous variables

        Tests is a small-sample version of Sargan's test that has the same
        distribution.

        .. math ::

          s (n - n_{instr}) / (n - s) \sim \chi^2_{v}

        where :math:`n_{instr}` is the number of instruments, :math:`n_{exog}`
        is the number of exogenous regressors and :math:`n_{endog}` is the
        number of endogenous regressors.  The degree of freedom is the
        difference between the number of instruments and the number of
        endogenous regressors.

        .. math ::

          v = n_{instr} - n_{exog}
        """
        mod = self.model
        ninstr = mod.instruments.shape[1]
        nobs, nendog = mod.endog.shape
        nz = mod._z.shape[1]
        name = 'Basmann\'s test of overidentification'
        if ninstr - nendog == 0:
            return InvalidTestStatistic('Test requires more instruments than '
                                        'endogenous variables.', name=name)
        sargan_test = self.sargan
        s = sargan_test.stat
        stat = s * (nobs - nz) / (nobs - s)
        return WaldTestStatistic(stat, sargan_test.null, sargan_test.df, name=name)

    def _endogeneity_setup(self, vars=None):
        """Setup function for some endogeneity iv"""
        if vars is not None and not isinstance(vars, list):
            vars = [vars]
        nobs = self.model.dependent.shape[0]
        e2 = self.resids.values
        nendog, nexog = self.model.endog.shape[1], self.model.exog.shape[1]
        if vars is None:
            assumed_exog = self.model.endog.ndarray
            aug_exog = c_[self.model.exog.ndarray, assumed_exog]
            still_endog = empty((nobs, 0))
        else:
            assumed_exog = self.model.endog.pandas[vars].values
            ex = [c for c in self.model.endog.cols if c not in vars]
            still_endog = self.model.endog.pandas[ex].values
            aug_exog = c_[self.model.exog.ndarray, assumed_exog]
        ntested = assumed_exog.shape[1]

        from linearmodels.iv import IV2SLS
        mod = IV2SLS(self.model.dependent, aug_exog, still_endog,
                     self.model.instruments)
        e0 = mod.fit().resids.values[:, None]

        z2 = c_[self.model.exog.ndarray, self.model.instruments.ndarray]
        z1 = c_[z2, assumed_exog]

        e1 = proj(e0, z1)
        e2 = proj(e2, self.model.instruments.ndarray)
        return e0, e1, e2, nobs, nexog, nendog, ntested

    def durbin(self, vars=None):
        r"""
        Durbin's test of exogeneity

        Parameters
        ----------
        vars : list(str), optional
            List of variables to test for exogeneity.  If None, all variables
            are jointly tested.

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Test statistic is difference between sum of squared OLS and sum of
        squared IV residuals where each set of residuals has been projected
        onto the set of instruments in teh IV model.

        Start by defining

        .. math ::

          \delta & = \hat{\epsilon}'_e P_{[z,w]} \hat{\epsilon}_e -
                     \hat{\epsilon}'_c P_{z} \hat{\epsilon}_c

        where :math:`\hat{\epsilon}_e` are the regression residuals from a
        model where ``vars`` are treated as exogenous,
        :math:`\hat{\epsilon}_c` are the regression residuals from the model
        leaving ``vars`` as endogenous, :math:`P_{[z,w]}` is a projection
        matrix onto the exogenous variables and instruments (`z`) as well as
        ``vars``, and :math:`P_{z}` is a projection matrix only onto `z`.

        The test statistic is then

        .. math ::

            \delta / (\hat{\epsilon}'_e\hat{\epsilon}_e) / n \sim \chi^2_{q}

        where :math:`q` is the number of variables tested.
        """
        null = 'All endogenous variables are exogenous'
        if vars is not None:
            null = 'Variables {0} are exogenous'.format(', '.join(vars))

        e0, e1, e2, nobs, nexog, nendog, ntested = self._endogeneity_setup(vars)
        stat = e1.T @ e1 - e2.T @ e2
        stat /= (e0.T @ e0) / nobs

        name = 'Durbin test of exogeneity'
        df = ntested
        return WaldTestStatistic(float(stat), null, df, name=name)

    def wu_hausman(self, vars=None):
        r"""
        Wu-Hausman test of exogeneity

        Parameters
        ----------
        vars : list(str), optional
            List of variables to test for exogeneity.  If None, all variables
            are jointly tested.

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Test statistic is difference between sum of squared OLS and sum of
        squared IV residuals where each set of residuals has been projected
        onto the set of instruments in teh IV model.

        Start by defining

        .. math ::

          \delta & = \hat{\epsilon}'_e P_{[z,w]} \hat{\epsilon}_e -
                     \hat{\epsilon}'_c P_{z} \hat{\epsilon}_c

        where :math:`\hat{\epsilon}_e` are the regression residuals from a
        model where ``vars`` are treated as exogenous,
        :math:`\hat{\epsilon}_c` are the regression residuals from the model
        leaving ``vars`` as endogenous, :math:`P_{[z,w]}` is a projection
        matrix onto the exogenous variables and instruments (`z`) as well as
        ``vars``, and :math:`P_{z}` is a projection matrix only onto `z`.

        The test statistic is then

        .. math ::

            \frac{\delta / q}{(\hat{\epsilon}'_e\hat{\epsilon}_e - \delta) / v}

        where :math:`q` is the number of variables iv,
        :math:`v = n - n_{endog} - n_{exog} - q`. The test statistic has a
        :math:`F_{q, v}` distribution.
        """
        null = 'All endogenous variables are exogenous'
        if vars is not None:
            null = 'Variables {0} are exogenous'.format(', '.join(vars))

        e0, e1, e2, nobs, nexog, nendog, ntested = self._endogeneity_setup(vars)

        df = ntested
        df_denom = nobs - nexog - nendog - ntested
        delta = (e1.T @ e1 - e2.T @ e2)
        stat = delta / df
        stat /= (e0.T @ e0 - delta) / df_denom
        stat = float(stat)

        name = 'Wu-Hausman test of exogeneity'
        return WaldTestStatistic(stat, null, df, df_denom, name=name)

    @cached_property
    def wooldridge_score(self):
        r"""
        Wooldridge's score test of exogeneity

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Wooldridge's test examines whether there is correlation between the
        errors produced when the endogenous variable are treated as
        exogenous so that the model can be fit by OLS, and the component of
        the endogenous variables that cannot be explained by the instruments.

        The test is implemented using a regression,

        .. math ::

          1 = \gamma_1 \hat{\epsilon}_1 \hat{v}_{1,i} + \ldots
            + \gamma_p \hat{\epsilon}_1 \hat{v}_{p,i} + \eta_i

        where :math:`\hat{v}_{j,i}` is the residual from regressing endogenous
        variable :math:`x_j` on the exogenous variables and instruments.

        The test is a :math:`n\times R^2 \sim \chi^2_{p}`.

        Implemented using the expression in Wooldridge (2002), Eq. 6.19
        """
        from linearmodels.iv.model import _OLS

        e = annihilate(self.model.dependent.ndarray, self.model._x)
        r = annihilate(self.model.endog.ndarray, self.model._z)
        nobs = e.shape[0]
        r = annihilate(r, self.model._x)
        res = _OLS(ones((nobs, 1)), r * e).fit(cov_type='unadjusted')
        stat = res.nobs - res.resid_ss
        df = self.model.endog.shape[1]
        null = 'Endogenous variables are exogenous'
        name = 'Wooldridge\'s score test of exogeneity'
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def wooldridge_regression(self):
        r"""
        Wooldridge's regression test of exogeneity

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Wooldridge's test examines whether there is correlation between the
        components of the endogenous variables that cannot be explained by
        the instruments and the OLS regression residuals.

        The test is implemented as an OLS where

        .. math ::

          y_i = x_{1i}\beta_i + x_{2i}\beta_2 + \hat{e}_i\gamma + \epsilon_i

        where :math:`x_{1i}` are the exogenous regressors, :math:`x_{2i}` are
        the  endogenous regressors and :math:`\hat{e}_{i}` are the residuals
        from regressing the endogenous variables on the exogenous variables
        and instruments. The null is :math:`\gamma=0` and is implemented
        using a Wald test.  The covariance estimator used in the test is
        identical to the covariance estimator used with ``fit``.
        """
        from linearmodels.iv.model import _OLS
        r = annihilate(self.model.endog.ndarray, self.model._z)
        augx = c_[self.model._x, r]
        mod = _OLS(self.model.dependent, augx)
        res = mod.fit(cov_type=self.cov_type, **self.cov_config)
        norig = self.model._x.shape[1]
        test_params = res.params.values[norig:]
        test_cov = res.cov.values[norig:, norig:]
        stat = test_params.T @ inv(test_cov) @ test_params
        df = len(test_params)
        null = 'Endogenous variables are exogenous'
        name = 'Wooldridge\'s regression test of exogeneity'
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def wooldridge_overid(self):
        r"""
        Wooldridge's score test of overidentification

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Wooldridge's test examines whether there is correlation between the
        model residuals and the component of the instruments that is
        orthogonal to the endogenous variables. Define :math:`\tilde{z}`
        to be the residuals of the instruments regressed on the exogenous
        variables and the first-stage fitted values of the endogenous
        variables.  The test is computed as a regression

        .. math ::

          1 = \gamma_1 \hat{\epsilon}_i \tilde{z}_{i,1} + \ldots +
              \gamma_q \hat{\epsilon}_i \tilde{z}_{i,q}

        where :math:`q = n_{instr} - n_{endog}`.  The test is a
        :math:`n\times R^2 \sim \chi^2_{q}`.

        The order of the instruments does not affect this test.
        """
        from linearmodels.iv.model import _OLS
        exog, endog = self.model.exog, self.model.endog
        instruments = self.model.instruments
        nobs, nendog = endog.shape
        ninstr = instruments.shape[1]
        if ninstr - nendog == 0:
            import warnings
            warnings.warn('Test requires more instruments than '
                          'endogenous variables',
                          UserWarning)
            return WaldTestStatistic(0, 'Test is not feasible.', 1, name='Infeasible test.')

        endog_hat = proj(endog.ndarray, c_[exog.ndarray, instruments.ndarray])
        q = instruments.ndarray[:, :(ninstr - nendog)]
        q_res = annihilate(q, c_[self.model.exog.ndarray, endog_hat])
        test_functions = q_res * self.resids.values[:, None]
        res = _OLS(ones((nobs, 1)), test_functions).fit(cov_type='unadjusted')

        stat = res.nobs * res.rsquared
        df = ninstr - nendog
        null = 'Model is not overidentified.'
        name = 'Wooldridge\'s score test of overidentification'
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def anderson_rubin(self):
        """
        Anderson-Rubin test of overidentifying restrictions

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        The Anderson-Rubin test examines whether the value of :math:`\kappa`
        computed for the LIML estimator is sufficiently close to one to
        indicate the model is not overidentified. The test statistic is

        .. math ::

          n \ln(\hat{\kappa}) \sim \chi^2_{q}

        where :math:`q = n_{instr} - n_{endog}`.
        """
        nobs, ninstr = self.model.instruments.shape
        nendog = self.model.endog.shape[1]
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
        """
        Basmann's F test of overidentifying restrictions

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Banmann's F test examines whether the value of :math:`\kappa`
        computed for the LIML estimator is sufficiently close to one to
        indicate the model is not overidentified. The test statistic is

        .. math ::

          \hat{\kappa} (n -n_{instr})/q \sim F_{q, n - n_{instr}}

        where :math:`q = n_{instr} - n_{endog}`.
        """
        nobs, ninstr = self.model.instruments.shape
        nendog, nexog = self.model.endog.shape[1], self.model.exog.shape[1]
        name = 'Basmann\' F  test of overidentification'
        if ninstr - nendog == 0:
            return InvalidTestStatistic('Test requires more instruments than '
                                        'endogenous variables.', name=name)
        df = ninstr - nendog
        df_denom = nobs - (nexog + ninstr)
        stat = (self._liml_kappa - 1) * df_denom / df
        null = 'The model is not overidentified.'
        return WaldTestStatistic(stat, null, df, df_denom=df_denom, name=name)


class IVGMMResults(_CommonIVResults):
    """
    Results from GMM estimation of IV models
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
        """Weighting matrix configuration used in estimation"""
        return self._weight_config

    @property
    def j_stat(self):
        r"""
        J-test of overidentifying restrictions

        Returns
        -------
        j : WaldTestStatistic
            J statistic  test of overidentifying restrictions

        Notes
        -----
        The J statistic  iv whether the moment conditions are sufficiently
        close to zero to indicate that the model is not overidentified. The
        statistic is defined as

        .. math ::

          n \bar{g}'W^{-1}\bar{g} \sim \chi^2_q

        where :math:`\bar{g} = n^{-1}\sum \hat{\epsilon}_i z_i` where
        :math:`z_i` includes both the exogenous variables and instruments and
        :math:`\hat{\epsilon}_i` are the model residuals. :math:`W` is a consistent
        estimator of the variance of :math:`\sqrt{n}\bar{g}`. The degree of
        freedom is :math:`q = n_{instr} - n_{endog}`.
        """
        return self._j_stat

    def c_stat(self, vars=None):
        r"""
        C-test of endogeneity

        Parameters
        ----------
        vars : list(str), optional
            List of variables to test for exogeneity.  If None, all variables
            are jointly tested.

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        The C statistic iv the difference between the model estimated by
        assuming one or more of the endogenous variables is actually
        exogenous.  The test is implemented as the difference between the
        J statistic s of two GMM estimations where both use the same weighting
        matrix.  The use of a common weighting matrix is required for the C
        statistic to be positive.

        The first model is a estimated uses GMM estimation where one or more
        of the endogenous variables are assumed to be endogenous.  The model
        would be relatively efficient if the assumption were true, and two
        quantities are computed, the J statistic, :math:`J_e`, and the
        moment weighting matrix, :math:`W_e`.

        WLOG assume the q variables tested are in the final q positions so that
        the first :math:`n_{exog} + n_{instr}` rows and columns correspond to
        the moment conditions in the original model. The second J statistic is
        computed using parameters estimated using the original moment
        conditions along with the upper left block of :math:`W_e`.  Denote this
        values as :math:`J_c` where the c is used to indicate consistent.

        The test statistic is then

        .. math ::

          J_e - J_c \sim \chi^2_{m}

        where :math:`m` is the number of variables whose exogeneity is being
        tested.
        """
        dependent, instruments = self.model.dependent, self.model.instruments
        exog, endog = self.model.exog, self.model.endog
        if vars is None:
            exog_e = c_[exog.ndarray, endog.ndarray]
            nobs = exog_e.shape[0]
            endog_e = empty((nobs, 0))
            null = 'All endogenous variables are exogenous'
        else:
            if not isinstance(vars, list):
                vars = [vars]
            exog_e = c_[exog.ndarray, endog.pandas[vars].values]
            ex = [c for c in endog.pandas if c not in vars]
            endog_e = endog.pandas[ex].values
            null = 'Variables {0} are exogenous'.format(', '.join(vars))
        from linearmodels.iv import IVGMM
        mod = IVGMM(dependent, exog_e, endog_e, instruments)
        res_e = mod.fit(cov_type=self.cov_type, **self.cov_config)
        j_e = res_e.j_stat.stat

        x = self.model._x
        y = self.model._y
        z = self.model._z
        nz = z.shape[1]
        weight_mat_c = res_e.weight_matrix.values[:nz, :nz]
        params_c = mod.estimate_parameters(x, y, z, weight_mat_c)
        j_c = self.model._j_statistic(params_c, weight_mat_c).stat

        stat = j_e - j_c
        df = exog_e.shape[1] - exog.shape[1]
        return WaldTestStatistic(stat, null, df, name='C-statistic')


class FirstStageResults(_SummaryStr):
    """
    First stage estimation results and diagnostics
    """

    def __init__(self, dep, exog, endog, instr, weights, cov_type, cov_config):
        self.dep = dep
        self.exog = exog
        self.endog = endog
        self.instr = instr
        self.weights = weights
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

            * rsquared - R-squared from regression of endogenous on exogenous
              and instruments
            * partial.rsquared - R-squared from regression of the exogenous
              variable on instruments where both the exogenous variable and
              the instrument have been orthogonalized to the exogenous
              regressors in the model.
            * f.stat - Test that all coefficients are zero in the model
              used to estimate the partial R-squared. Uses a standard F-test
              when the covariance estimator is unadjusted - otherwise uses a
              Wald test statistic with a chi2 distribution.
            * f.pval - P-value of the test that all coefficients are zero
              in the model used to estimate the partial R-squared
            * f.dist - Distribution of f.stat
            * shea.rsquared - Shea's r-squared which measures the correlation
              between the projected and orthogonalized instrument on the
              orthogonalized endogenous regressor where the orthogonalization
              is with respect to the other included variables in the model.
        """
        from linearmodels.iv.model import _OLS, IV2SLS
        endog, exog, instr, weights = self.endog, self.exog, self.instr, self.weights
        w = sqrt(weights.ndarray)
        z = w * instr.ndarray
        x = w * exog.ndarray
        nobs = endog.shape[0]
        if x.shape[1] == 0:
            # No exogenous regressors
            px = zeros((nobs, nobs))
        else:
            px = x @ pinv(x)
        ez = z - px @ z
        out = OrderedDict()
        individual_results = self.individual
        for col in endog.pandas:
            inner = {}
            inner['rsquared'] = individual_results[col].rsquared
            y = w * endog.pandas[[col]].values
            ey = y - px @ y
            mod = _OLS(ey, ez)
            res = mod.fit(cov_type=self._cov_type, **self._cov_config)
            inner['partial.rsquared'] = res.rsquared
            params = res.params.values
            params = params[:, None]
            stat = params.T @ inv(res.cov) @ params
            stat = float(stat.squeeze())
            w_test = WaldTestStatistic(stat, null='', df=params.shape[0])
            inner['f.stat'] = w_test.stat
            inner['f.pval'] = w_test.pval
            inner['f.dist'] = w_test.dist_name
            out[col] = Series(inner)
        out = DataFrame(out).T

        dep = self.dep
        r2sls = IV2SLS(dep, exog, endog, instr, weights=weights).fit(cov_type='unadjusted')
        rols = _OLS(dep, self._reg, weights=weights).fit(cov_type='unadjusted')
        shea = (rols.std_errors / r2sls.std_errors) ** 2
        shea *= (1 - r2sls.rsquared) / (1 - rols.rsquared)
        out['shea.rsquared'] = shea[out.index]
        cols = ['rsquared', 'partial.rsquared', 'shea.rsquared', 'f.stat', 'f.pval', 'f.dist']
        out = out[cols]
        for c in out:
            out[c] = to_numeric(out[c], errors='ignore')

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
        from linearmodels.iv.model import _OLS
        w = sqrt(self.weights.ndarray)
        exog_instr = w * c_[self.exog.ndarray, self.instr.ndarray]
        exog_instr = DataFrame(exog_instr, columns=self.exog.cols + self.instr.cols)
        res = OrderedDict()
        for col in self.endog.pandas:
            dep = w.squeeze() * self.endog.pandas[col]
            mod = _OLS(dep, exog_instr)
            res[col] = mod.fit(cov_type=self._cov_type, **self._cov_config)

        return res

    @property
    def summary(self):
        """Summary table of first-stage estimation results"""
        stubs_lookup = {'rsquared': 'R-squared',
                        'partial.rsquared': 'Partial R-squared',
                        'shea.rsquared': 'Shea\'s R-squared',
                        'f.stat': 'Partial F-statistic',
                        'f.pval': 'P-value (Partial F-stat)',
                        'f.dist': 'Partial F-stat Distn'}
        smry = Summary()
        diagnostics = self.diagnostics
        vals = []
        for c in diagnostics:
            if c != 'f.dist':
                vals.append([_str(v) for v in diagnostics[c]])
            else:
                vals.append([v for v in diagnostics[c]])
        stubs = [stubs_lookup[s] for s in list(diagnostics.columns)]
        header = list(diagnostics.index)

        params = []
        for var in header:
            res = self.individual[var]
            v = c_[res.params.values, res.tstats.values]
            params.append(v.ravel())
        params = array(params)
        params_fmt = [[_str(val) for val in row] for row in params.T]
        for i in range(1, len(params_fmt), 2):
            for j in range(len(params_fmt[i])):
                params_fmt[i][j] = '({0})'.format(params_fmt[i][j])

        params_stub = []
        for var in res.params.index:
            params_stub.extend([var, ''])

        title = 'First Stage Estimation Results'

        vals = table_concat((vals, params_fmt))
        stubs = stub_concat((stubs, params_stub))

        txt_fmt = default_txt_fmt.copy()
        txt_fmt['data_aligns'] = 'r'
        txt_fmt['header_align'] = 'r'
        table = SimpleTable(vals, headers=header, title=title, stubs=stubs, txt_fmt=txt_fmt)
        smry.tables.append(table)
        extra_txt = ['T-stats reported in parentheses',
                     'T-stats use same covariance type as original model']
        smry.add_extra_txt(extra_txt)
        return smry


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
    comparison : IVModelComparison
    """
    return IVModelComparison(results)


class IVModelComparison(_ModelComparison):
    """
    Comparison of multiple models

    Parameters
    ----------
    results : {list, dict, OrderedDict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.  An OrderedDict will preserve the model order the comparisons.
    """
    _supported = (IVResults, IVGMMResults, OLSResults)

    def __init__(self, results):
        super(IVModelComparison, self).__init__(results)

    @property
    def rsquared_adj(self):
        """Sample-size adjusted coefficients of determination (R**2)"""
        return self._get_property('rsquared_adj')

    @property
    def estimator_method(self):
        """Estimation methods"""
        return self._get_property('_method')

    @property
    def cov_estimator(self):
        """Covariance estimator descriptions"""
        return self._get_property('cov_estimator')

    @property
    def summary(self):
        """Summary table of model comparison"""
        smry = Summary()
        models = list(self._results.keys())
        title = 'Model Comparison'
        stubs = ['Dep. Variable', 'Estimator', 'No. Observations', 'Cov. Est.', 'R-squared',
                 'Adj. R-squared', 'F-statistic', 'P-value (F-stat)']
        dep_name = OrderedDict()
        for key in self._results:
            dep_name[key] = self._results[key].model.dependent.cols[0]
        dep_name = Series(dep_name)

        vals = concat([dep_name, self.estimator_method, self.nobs, self.cov_estimator,
                       self.rsquared, self.rsquared_adj, self.f_statistic], 1)
        vals = [[i for i in v] for v in vals.T.values]
        vals[2] = [str(v) for v in vals[2]]
        for i in range(4, len(vals)):
            vals[i] = [_str(v) for v in vals[i]]

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

        all_instr = []
        for key in self._results:
            res = self._results[key]
            all_instr.append(res.model.instruments.cols)
        ninstr = max(map(lambda l: len(l), all_instr))
        instruments = []
        instrument_stub = ['Instruments']
        for i in range(ninstr):
            if i > 0:
                instrument_stub.append('')
            row = []
            for j in range(len(self._results)):
                instr = all_instr[j]
                if len(instr) > i:
                    row.append(instr[i])
                else:
                    row.append('')
            instruments.append(row)
        if instruments:
            vals = table_concat((vals, instruments))
            stubs = stub_concat((stubs, instrument_stub))

        txt_fmt = default_txt_fmt.copy()
        txt_fmt['data_aligns'] = 'r'
        txt_fmt['header_align'] = 'r'
        table = SimpleTable(vals, headers=models, title=title, stubs=stubs, txt_fmt=txt_fmt)
        smry.tables.append(table)
        smry.add_extra_txt(['T-stats reported in parentheses'])
        return smry
