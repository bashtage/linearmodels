from numpy import diag, sqrt
from pandas import DataFrame, Series
from scipy import stats
from statsmodels.iolib.summary import SimpleTable

from linearmodels.compat.statsmodels import Summary
from linearmodels.utility import AttrDict, _SummaryStr, cached_property

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

    @property
    def cov(self):
        """Estimated covariance of parameters"""
        return DataFrame(self._cov, index=self._param_names,
                         columns=self._param_names)

    @property
    def cov_estimator(self):
        """Type of covariance estimator used to compute covariance"""
        return self._cov_type

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
            pvals = 2 - 2 * stats.t.cdf(abs(self.tstats), self._df_resid)
        else:
            pvals = 2 - 2 * stats.norm.cdf(abs(self.tstats))

        return Series(pvals, index=self._param_names, name='pvalue')

    @property
    def rsquared(self):
        """Coefficient of determination (R**2)"""
        return self._r2


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

    @property
    def summary(self):
        """Model summary"""
        smry = Summary()
        table = SimpleTable(['No summary yet'])
        smry.tables.append(table)

        return smry

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


class SUREquationResult(_CommonResults):
    def __init__(self, results):
        super(SUREquationResult, self).__init__(results)
        self._eq_label = results.eq_label
        self._dependent = results.dependent
        self._f_statistic = results.f_stat

    @property
    def equation_label(self):
        """Equation label"""
        return self._eq_label

    @property
    def dependent(self):
        """Name of dependent variable"""
        return self._eq_label

    @property
    def summary(self):
        """Equation summary"""
        smry = Summary()
        table = SimpleTable(['No summary yet'])
        smry.tables.append(table)

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
