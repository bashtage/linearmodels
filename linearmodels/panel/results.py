import datetime as dt

import numpy as np
from numpy import sqrt, diag
from pandas import Series, DataFrame
from scipy import stats
from statsmodels.iolib.summary import SimpleTable, Summary, fmt_2cols, \
    fmt_params

from linearmodels.utility import cached_property, pval_format, _str, _SummaryStr


class PanelResults(_SummaryStr):
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

    @property
    def params(self):
        return Series(self._params, index=self._var_names)

    @cached_property
    def cov(self):
        return DataFrame(self._deferred_cov(), columns=self._var_names, index=self._var_names)

    @property
    def std_errors(self):
        return Series(sqrt(diag(self.cov)), self._var_names)

    @property
    def tstats(self):
        return self._params / self.std_errors

    @cached_property
    def pvalues(self):
        abs_tstats = np.abs(self.tstats)
        if self._debiased:
            pv = 2 * (1 - stats.t.cdf(abs_tstats, self.df_resid))
        else:
            pv = 2 * (1 - stats.norm.cdf(abs_tstats))
        return Series(pv, index=self._var_names)

    @property
    def df_resid(self):
        return self._df_resid

    @property
    def df_model(self):
        return self._df_model

    @property
    def nobs(self):
        return self._nobs

    @property
    def name(self):
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
        return self._r2

    @property
    def rsquared_between(self):
        return self._r2b

    @property
    def rsquared_within(self):
        return self._r2w

    @property
    def rsquared_overall(self):
        return self._r2o

    @property
    def s2(self):
        return self._s2

    @property
    def entity_info(self):
        return self._entity_info

    @property
    def time_info(self):
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
                    ('', ''),
                    ('Entities:', str(int(self.entity_info['total']))),
                    ('Avg Obs:', _str(self.entity_info['mean'])),
                    ('Min Obs:', _str(self.entity_info['min'])),
                    ('Max Obs:', _str(self.entity_info['max'])),
                    ('', '')]

        top_right = [('R-squared:', _str(self.rsquared)),
                     ('R-squared (Between):', _str(self.rsquared_between)),
                     ('R-squared (Within):', _str(self.rsquared_within)),
                     ('R-squared (Overall):', _str(self.rsquared_overall)),
                     ('F-statistic:', '---'),  # TODO
                     ('P-value (F-stat)', '---'),  # TODO
                     ('Distribution:', '---'),  # TODO
                     ('', ''),
                     ('Time periods:', str(int(self.time_info['total']))),
                     ('Avg Obs:', _str(self.time_info['mean'])),
                     ('Min Obs:', _str(self.time_info['min'])),
                     ('Max Obs:', _str(self.time_info['max'])),
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
        header = ['Parameters', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']
        table = SimpleTable(data,
                            stubs=table_stubs,
                            txt_fmt=fmt_params,
                            headers=header,
                            title=title)
        smry.tables.append(table)

        return smry
