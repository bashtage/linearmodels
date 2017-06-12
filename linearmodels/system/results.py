import datetime as dt
import numpy as np
from numpy import diag, sqrt
from pandas import DataFrame, Series
from scipy import stats
from statsmodels.iolib.summary import SimpleTable, fmt_2cols

from linearmodels.compat.statsmodels import Summary
from linearmodels.utility import (AttrDict, _SummaryStr, cached_property,
                                  _str, param_table, pval_format)

__all__ = ['SURResults', 'SUREquationResult']


class _CommonResults(_SummaryStr):
    def __init__(self, results):
        self._method = results.method
        self._params = results.params
        self._cov = results.cov
        self._param_names = results.param_names
        self._debiased = results.debiased
        self._r2 = results.r2
        self._resid = results.resid
        self._wresid = results.wresid
        self._nobs = results.nobs
        self._df_model = results.df_model
        self._df_resid = self._nobs - self._df_model
        self._index = results.index
        self._iter = results.iter
        self._cov_type = results.cov_type
        self._tss = results.total_ss
        self._rss = results.resid_ss
        self._datetime = dt.datetime.now()

    @property
    def method(self):
        """Estimation method"""
        return self._method

    @property
    def cov(self):
        """Estimated covariance of parameters"""
        return DataFrame(self._cov, index=self._param_names,
                         columns=self._param_names)

    @property
    def cov_estimator(self):
        """Type of covariance estimator used to compute covariance"""
        return self._cov_type

    @property
    def iterations(self):
        """Number of iterations of the GLS executed"""
        return self._iter

    @property
    def debiased(self):
        """Flag indicating whether covariance uses a small-sample adjustment"""
        return self._debiased

    @property
    def params(self):
        """Estimated parameters"""
        return Series(self._params.squeeze(), index=self._param_names, name='params')

    @property
    def std_errors(self):
        """Estimated parameter standard errors"""
        std_errors = sqrt(diag(self.cov))
        return Series(std_errors, index=self._param_names, name='stderr')

    @property
    def tstats(self):
        """Parameter t-statistics"""
        return Series(self.params / self.std_errors, name='tstat')

    @cached_property
    def pvalues(self):
        """
        Parameter p-vals. Uses t(df_resid) if ``debiased`` is True, else normal
        """
        if self.debiased:
            pvals = 2 - 2 * stats.t.cdf(np.abs(self.tstats), self._df_resid)
        else:
            pvals = 2 - 2 * stats.norm.cdf(np.abs(self.tstats))

        return Series(pvals, index=self._param_names, name='pvalue')

    @property
    def rsquared(self):
        """Coefficient of determination (R**2)"""
        return self._r2

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
    def nobs(self):
        """Number of observations"""
        return self._nobs

    @property
    def df_resid(self):
        """Residual degree of freedom"""
        return self._df_resid

    @property
    def df_model(self):
        """Model degree of freedom"""
        return self._df_model

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
        return DataFrame(ci, index=self._param_names, columns=['lower', 'upper'])


class SURResults(_CommonResults):
    """
    Results from Seemingly Unrelated Regression Estimators

    Parameters
    ----------
    results : AttrDict
        Dictionary of model estimation results
    """

    def __init__(self, results):
        super(SURResults, self).__init__(results)
        self._individual = AttrDict()
        for key in results.individual:
            self._individual[key] = SUREquationResult(results.individual[key])
        self._sigma = results.sigma
        self._model = results.model
        self._constraints = results.constraints
        self._num_constraints = 'None'
        if results.constraints is not None:
            self._num_constraints = str(results.constraints.r.shape[0])

    @property
    def model(self):
        """Model used in estimation"""
        return self._model

    @property
    def equations(self):
        """Individual equation results"""
        return self._individual

    @property
    def equation_labels(self):
        """Individual equation labels"""
        return list(self._individual.keys())

    @property
    def resids(self):
        """Estimated residuals"""
        return DataFrame(self._resid, index=self._index, columns=self.equation_labels)

    @property
    def wresids(self):
        """Weighted estimated residuals"""
        return DataFrame(self._wresid, index=self._index, columns=self.equation_labels)

    @property
    def sigma(self):
        """Estimated residual covariance"""
        return self._sigma

    @property
    def summary(self):
        """Model summary"""
        title = 'System ' + self._method + ' Estimation Summary'

        top_left = [('Estimator:', self._method),
                    ('No. Equations.:', str(len(self.equation_labels))),
                    ('No. Observations:', str(self.resids.shape[0])),
                    ('Date:', self._datetime.strftime('%a, %b %d %Y')),
                    ('Time:', self._datetime.strftime('%H:%M:%S')),
                    ('', ''),
                    ('', '')]

        top_right = [('Overall R-squared:', _str(self.rsquared)),
                     ('Cov. Estimator:', self._cov_type),
                     ('Num. Constraints: ', self._num_constraints),
                     ('', ''),
                     ('', ''),
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
        fmt['data_fmts'][1] = '%10s'

        top_right = [('%-21s' % ('  ' + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        for eqlabel in self.equation_labels:
            results = self.equations[eqlabel]
            dep_name = results.dependent
            title = 'Equation: {0}, Dependent Variable: {1}'.format(eqlabel, dep_name)
            smry.tables.append(param_table(results, title, pad_bottom=True))

        return smry


class SUREquationResult(_CommonResults):
    """
    Results from a single equation of a Seemingly Unrelated Regression

    Parameters
    ----------
    results : AttrDict
        Dictionary of model estimation results
    """

    def __init__(self, results):
        super(SUREquationResult, self).__init__(results)
        self._eq_label = results.eq_label
        self._dependent = results.dependent
        self._f_statistic = results.f_stat
        self._r2a = results.r2a

    @property
    def equation_label(self):
        """Equation label"""
        return self._eq_label

    @property
    def dependent(self):
        """Name of dependent variable"""
        return self._dependent

    @property
    def summary(self):
        """Equation summary"""
        title = self._method + ' Estimation Summary'

        top_left = [('Eq. Label:', self.equation_label),
                    ('Dep. Variable:', self.dependent),
                    ('Estimator:', self._method),
                    ('No. Observations:', self.nobs),
                    ('Date:', self._datetime.strftime('%a, %b %d %Y')),
                    ('Time:', self._datetime.strftime('%H:%M:%S')),

                    ('', '')]

        top_right = [('R-squared:', _str(self.rsquared)),
                     ('Adj. R-squared:', _str(self.rsquared_adj)),
                     ('Cov. Estimator:', self._cov_type),
                     ('F-statistic:', _str(self.f_statistic.stat)),
                     ('P-value (F-stat)', pval_format(self.f_statistic.pval)),
                     ('Distribution:', str(self.f_statistic.dist_name)),
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
        fmt['data_fmts'][1] = '%10s'

        top_right = [('%-21s' % ('  ' + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)
        smry.tables.append(param_table(self, 'Parameter Estimates', pad_bottom=True))

        return smry

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
    def resids(self):
        """Estimated residuals"""
        return Series(self._resid.squeeze(), index=self._index, name='resid')

    @property
    def wresids(self):
        """Weighted estimated residuals"""
        return Series(self._wresid.squeeze(), index=self._index, name='wresid')

    @property
    def rsquared_adj(self):
        """Sample-size adjusted coefficient of determination (R**2)"""
        return self._r2a
