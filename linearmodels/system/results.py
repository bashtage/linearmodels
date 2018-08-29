import datetime as dt

import numpy as np
from pandas import DataFrame, Series, concat
from scipy import stats
from statsmodels.iolib.summary import SimpleTable, fmt_2cols

from linearmodels.compat.statsmodels import Summary
from linearmodels.utility import (AttrDict, _SummaryStr, _str, cached_property, format_wide,
                                  param_table, pval_format)

__all__ = ['SystemResults', 'SystemEquationResult', 'GMMSystemResults']


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
        self._fitted = results.fitted
        self._nobs = results.nobs
        self._df_model = results.df_model
        self._df_resid = self._nobs - self._df_model
        self._index = results.index
        self._iter = results.iter
        self._cov_type = results.cov_type
        self._tss = results.total_ss
        self._rss = results.resid_ss
        self._datetime = dt.datetime.now()
        self._cov_estimator = results.cov_estimator
        self._cov_config = results.cov_config
        self._original_index = results.original_index

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
    def cov_config(self):
        """Configuration of covariance estimator used to compute covariance"""
        return self._cov_config

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
        std_errors = np.sqrt(np.diag(self.cov))
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


class SystemResults(_CommonResults):
    """
    Results from Seemingly Unrelated Regression Estimators

    Parameters
    ----------
    results : AttrDict
        Dictionary of model estimation results
    """

    def __init__(self, results):
        super(SystemResults, self).__init__(results)
        self._individual = AttrDict()
        for key in results.individual:
            self._individual[key] = SystemEquationResult(results.individual[key])
        self._sigma = results.sigma
        self._model = results.model
        self._constraints = results.constraints
        self._num_constraints = 'None'
        if results.constraints is not None:
            self._num_constraints = str(results.constraints.r.shape[0])
        self._weight_estimtor = results.get('weight_estimator', None)

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
    def fitted_values(self):
        """Fitted values"""
        return DataFrame(self._fitted, index=self._index, columns=self.equation_labels)

    def _out_of_sample(self, equations, data, missing, dataframe):
        if equations is not None and data is not None:
            raise ValueError('Predictions can only be constructed using one '
                             'of eqns or data, but not both.')
        pred = self.model.predict(self.params, equations=equations, data=data)  # type: DataFrame
        if not dataframe:
            pred = {col: pred[[col]] for col in pred}
            if not missing:
                for key in pred:
                    pred[key] = pred[key].dropna()
        else:
            pred = pred.dropna(how='all', axis=1)

        return pred

    def predict(self, equations=None, *, data=None, fitted=True,
                idiosyncratic=False, missing=False, dataframe=False):
        """
        In- and out-of-sample predictions

        Parameters
        ----------
        equations : dict
            Dictionary-like structure containing exogenous and endogenous
            variables.  Each key is an equations label and must
            match the labels used to fir the model. Each value must be either a tuple
            of the form (exog, endog) or a dictionary with keys 'exog' and 'endog'.
            If predictions are not required for one of more of the model equations,
            these keys can be omitted.
        data : DataFrame
            DataFrame to use for out-of-sample predictions when model was
            constructed using a formula.
        fitted : bool, optional
            Flag indicating whether to include the fitted values
        idiosyncratic : bool, optional
            Flag indicating whether to include the estimated idiosyncratic shock
        missing : bool, optional
            Flag indicating to adjust for dropped observations.  if True, the
            values returns will have the same size as the original input data
            before filtering missing values
        dataframe : bool, optional
            Flag indicating to return output as a dataframe. If False, a
            dictionary is returned using the equation labels as keys.

        Returns
        -------
        predictions : DataFrame, dict
            DataFrame or dictionary containing selected outputs

        Notes
        -----
        If `equations` and `data` are both `None`, in-sample predictions
        (fitted values) will be returned.

        If `data` is not none, then `equations` must be none.
        Predictions from models constructed using formulas can
        be computed using either `equations`, which will treat these are
        arrays of values corresponding to the formula-process data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.

        When using `exog` and `endog`, the regressor array for a particular
        equation is assembled as
        `[equations[eqn]['exog'], equations[eqn]['endog']]` where `eqn` is
        an equation label. These must correspond to the columns in the
        estimated model.
        """
        if equations is not None or data is not None:
            return self._out_of_sample(equations, data, missing, dataframe)
        if not (fitted or idiosyncratic):
            raise ValueError('At least one output must be selected')
        if dataframe:
            if fitted and not idiosyncratic:
                out = self.fitted_values
            elif idiosyncratic and not fitted:
                out = self.resids
            else:
                out = {'fitted_values': self.fitted_values, 'idiosyncratic': self.resids}
        else:
            out = {}
            for key in self.equation_labels:
                vals = []
                if fitted:
                    vals.append(self.fitted_values[[key]])
                if idiosyncratic:
                    vals.append(self.resids[[key]])
                out[key] = concat(vals, 1)
        if missing:
            if isinstance(out, DataFrame):
                out = out.reindex(self._original_index)
            else:
                for key in out:
                    out[key] = out[key].reindex(self._original_index)

        return out

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
        """:obj:`statsmodels.iolib.summary.Summary` : Summary table of model estimation results

        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """

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

        for i, eqlabel in enumerate(self.equation_labels):
            last_row = i == (len(self.equation_labels) - 1)
            results = self.equations[eqlabel]
            dep_name = results.dependent
            title = 'Equation: {0}, Dependent Variable: {1}'.format(eqlabel, dep_name)
            pad_bottom = results.instruments is not None and not last_row
            smry.tables.append(param_table(results, title, pad_bottom=pad_bottom))
            if results.instruments:
                formatted = format_wide(results.instruments, 80)
                if not last_row:
                    formatted.append([' '])
                smry.tables.append(SimpleTable(formatted, headers=['Instruments']))
        extra_text = ['Covariance Estimator:']
        for line in str(self._cov_estimator).split('\n'):
            extra_text.append(line)
        if self._weight_estimtor:
            extra_text.append('Weight Estimator:')
            for line in str(self._weight_estimtor).split('\n'):
                extra_text.append(line)
        smry.add_extra_txt(extra_text)

        return smry


class SystemEquationResult(_CommonResults):
    """
    Results from a single equation of a Seemingly Unrelated Regression

    Parameters
    ----------
    results : AttrDict
        Dictionary of model estimation results
    """

    def __init__(self, results):
        super(SystemEquationResult, self).__init__(results)
        self._eq_label = results.eq_label
        self._dependent = results.dependent
        self._f_statistic = results.f_stat
        self._r2a = results.r2a
        self._instruments = results.instruments
        self._endog = results.endog
        self._weight_estimator = results.get('weight_estimator', None)

    @property
    def equation_label(self):
        """Equation label"""
        return self._eq_label

    @property
    def dependent(self):
        """Name of dependent variable"""
        return self._dependent

    @property
    def instruments(self):
        """Instruments used in estimation.  None if all variables assumed exogenous."""
        return self._instruments

    @property
    def summary(self):
        """:obj:`statsmodels.iolib.summary.Summary` : Summary table of model estimation results

        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """

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

        extra_text = []
        instruments = self._instruments
        if instruments:
            endog = self._endog
            extra_text = []
            extra_text.append('Endogenous: ' + ', '.join(endog))
            extra_text.append('Instruments: ' + ', '.join(instruments))

        extra_text.append('Covariance Estimator:')
        for line in str(self._cov_estimator).split('\n'):
            extra_text.append(line)
        if self._weight_estimator:
            extra_text.append('Weight Estimator:')
            for line in str(self._weight_estimator).split('\n'):
                extra_text.append(line)
        smry.add_extra_txt(extra_text)

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
    def fitted_values(self):
        """Fitted values"""
        return Series(self._fitted.squeeze(), index=self._index, name='fitted_values')

    @property
    def rsquared_adj(self):
        """Sample-size adjusted coefficient of determination (R**2)"""
        return self._r2a


class GMMSystemResults(SystemResults):
    def __init__(self, results):
        super(GMMSystemResults, self).__init__(results)
        self._wmat = results.wmat
        self._weight_type = results.weight_type
        self._weight_config = results.weight_config
        self._j_stat = results.j_stat

    @property
    def w(self):
        """GMM weight matrix used in estimation"""
        return self._wmat

    @property
    def weight_type(self):
        """Type of weighting used in GMM estimation"""
        return self._weight_type

    @property
    def weight_config(self):
        """Weight configuration options used in GMM estimation"""
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
        The J statistic tests whether the moment conditions are sufficiently
        close to zero to indicate that the model is not overidentified. The
        statistic is defined as

        .. math ::

          n \bar{g}'W^{-1}\bar{g} \sim \chi^2_q

        where :math:`\bar{g}` is the average of the moment conditional and
        :math:`W` is a consistent estimator of the variance of
        :math:`\sqrt{n}\bar{g}`. The degree of freedom is
        :math:`q = n_{instr} - n_{endog}`.
        """
        return self._j_stat
