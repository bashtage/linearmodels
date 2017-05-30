"""
Results for linear factor models
"""
import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import SimpleTable, fmt_2cols, fmt_params

from linearmodels.compat.statsmodels import Summary
from linearmodels.utility import (_str, _SummaryStr, cached_property,
                                  pval_format)


class LinearFactorModelResults(_SummaryStr):
    def __init__(self, res):
        self._jstat = res.jstat
        self._params = res.params
        self._param_names = res.param_names
        self._factor_names = res.factor_names
        self._portfolio_names = res.portfolio_names
        self._rp = res.rp
        self._cov = res.cov
        self._rp_cov = res.rp_cov
        self._rsquared = res.rsquared
        self._total_ss = res.total_ss
        self._residual_ss = res.residual_ss
        self._name = res.name
        self._cov_type = res.cov_type
        self.model = res.model
        self._nobs = res.nobs
        self._datetime = dt.datetime.now()
        self._cols = ['alpha'] + ['{0}'.format(f) for f in self._factor_names]
        self._rp_names = res.rp_names
        self._alpha_vcv = res.alpha_vcv
        self._cov_est = res.cov_est

    @property
    def summary(self):
        """Summary table of model estimation results"""

        title = self.name + ' Estimation Summary'

        top_left = [('No. Test Portfolios:', len(self._portfolio_names)),
                    ('No. Factors:', len(self._factor_names)),
                    ('No. Observations:', self.nobs),
                    ('Date:', self._datetime.strftime('%a, %b %d %Y')),
                    ('Time:', self._datetime.strftime('%H:%M:%S')),
                    ('Cov. Estimator:', self._cov_type),
                    ('', '')]

        j_stat = _str(self.j_statistic.stat)
        j_pval = pval_format(self.j_statistic.pval)
        j_dist = self.j_statistic.dist_name

        top_right = [('R-squared:', _str(self.rsquared)),
                     ('J-statistic:', j_stat),
                     ('P-value', j_pval),
                     ('Distribution:', j_dist),
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
        fmt['data_fmts'][1] = '%18s'

        top_right = [('%-21s' % ('  ' + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        rp = self.risk_premia.values[:, None]
        se = self.risk_premia_se.values[:, None]
        tstats = (self.risk_premia / self.risk_premia_se).values
        pvalues = 2 - 2 * stats.norm.cdf(np.abs(tstats))
        ci = rp + se * stats.norm.ppf([[0.025, 0.975]])
        param_data = np.c_[rp,
                           se,
                           tstats[:, None],
                           pvalues[:, None],
                           ci]
        data = []
        for row in param_data:
            txt_row = []
            for i, v in enumerate(row):
                f = _str
                if i == 3:
                    f = pval_format
                txt_row.append(f(v))
            data.append(txt_row)
        title = 'Risk Premia Estimates'
        table_stubs = list(self.risk_premia.index)
        header = ['Parameter', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']
        table = SimpleTable(data,
                            stubs=table_stubs,
                            txt_fmt=fmt_params,
                            headers=header,
                            title=title)
        smry.tables.append(table)
        smry.add_extra_txt(['Covariance estimator:',
                            str(self._cov_est),
                            'See full_summary for complete results'])

        return smry

    @staticmethod
    def _single_table(params, se, name, param_names, first=False):
        tstats = (params / se)
        pvalues = 2 - 2 * stats.norm.cdf(tstats)
        ci = params + se * stats.norm.ppf([[0.025, 0.975]])
        param_data = np.c_[params, se, tstats, pvalues, ci]

        data = []
        for row in param_data:
            txt_row = []
            for i, v in enumerate(row):
                f = _str
                if i == 3:
                    f = pval_format
                txt_row.append(f(v))
            data.append(txt_row)
        title = '{0} Coefficients'.format(name)
        table_stubs = param_names
        if first:
            header = ['Parameter', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']
        else:
            header = None
        table = SimpleTable(data, stubs=table_stubs, txt_fmt=fmt_params, headers=header,
                            title=title)

        return table

    @property
    def full_summary(self):
        """Complete summary including factor loadings and mispricing measures"""
        smry = self.summary
        params = self.params
        se = self.std_errors
        param_names = list(params.columns)
        first = True
        for row in params.index:
            smry.tables.append(SimpleTable(['']))
            smry.tables.append(self._single_table(params.loc[row].values[:, None],
                                                  se.loc[row].values[:, None],
                                                  row, param_names, first))
            first = False

        return smry

    @property
    def nobs(self):
        """Number of observations"""
        return self._nobs

    @property
    def name(self):
        """Model type"""
        return self._name

    @property
    def alphas(self):
        """Mispricing estimates"""
        return self.params.iloc[:, 0]

    @property
    def betas(self):
        """Estimated factor loadings"""
        return self.params.iloc[:, 1:]

    @property
    def params(self):
        """Estimated parameters"""
        return pd.DataFrame(self._params, columns=self._cols, index=self._portfolio_names)

    @property
    def std_errors(self):
        """Estimated parameter standard errors"""
        se = np.sqrt(np.diag(self._cov))
        nportfolio, nfactor = self._params.shape
        nloadings = nportfolio * nfactor
        se = se[:nloadings]
        se = se.reshape((nportfolio, nfactor))
        return pd.DataFrame(se, columns=self._cols, index=self._portfolio_names)

    @cached_property
    def tstats(self):
        """Parameter t-statistics"""
        return self.params / self.std_errors

    @property
    def cov_estimator(self):
        """Type of covariance estimator used to compute covariance"""
        return str(self._cov_est)

    @property
    def cov(self):
        """Estimated covariance of parameters"""
        return pd.DataFrame(self._cov, columns=self._param_names, index=self._param_names)

    @property
    def j_statistic(self):
        r"""
        Model J statistic

        Returns
        -------
        j : WaldTestStatistic
            Test statistic for null that model prices test portfolios

        Notes
        -----
        Joint test that all estimated :math:`\hat{\alpha}_i` are zero.
        Implemented using a Wald test using the estimated parameter
        covariance.
        """

        return self._jstat

    @property
    def risk_premia(self):
        """Estimated factor risk premia (lambda)"""
        return pd.Series(self._rp.squeeze(), index=self._rp_names)

    @property
    def risk_premia_se(self):
        """Estimated factor risk premia standard errors"""
        se = np.sqrt(np.diag(self._rp_cov))
        return pd.Series(se, index=self._rp_names)

    @property
    def risk_premia_tstats(self):
        """Risk premia t-statistics"""
        return self.risk_premia / self.risk_premia_se

    @property
    def rsquared(self):
        """Coefficient of determination (R**2)"""
        return self._rsquared

    @property
    def total_ss(self):
        """Total sum of squares"""
        return self._total_ss

    @property
    def residual_ss(self):
        """Residual sum of squares"""
        return self._residual_ss


class GMMFactorModelResults(LinearFactorModelResults):
    def __init__(self, res):
        super(GMMFactorModelResults, self).__init__(res)
        self._iter = res.iter

    @property
    def std_errors(self):
        """Estimated parameter standard errors"""
        se = np.sqrt(np.diag(self._cov))
        ase = np.sqrt(np.diag(self._alpha_vcv))
        nportfolio, nfactor = self._params.shape
        nloadings = nportfolio * (nfactor - 1)
        se = np.r_[ase, se[:nloadings]]
        se = se.reshape((nportfolio, nfactor))
        return pd.DataFrame(se, columns=self._cols, index=self._portfolio_names)

    def iterations(self):
        """Number of steps in GMM estimation"""
        return self._iter
