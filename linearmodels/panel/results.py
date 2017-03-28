import numpy as np
from numpy import sqrt, diag
from pandas import Series, DataFrame
from scipy import stats

from linearmodels.utility import cached_property


class PanelResults(object):
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
        self._r2= res.r2
        self._r2w= res.r2w
        self._r2b= res.r2b
        self._s2 = res.s2

    @property
    def params(self):
        return Series(self._params, index=self._var_names)

    @cached_property
    def cov(self):
        return DataFrame(self._deferred_cov(), columns=self._var_names, index=self._var_names)

    @property
    def stderr(self):
        return Series(sqrt(diag(self.cov)), self._var_names)

    @property
    def tstats(self):
        return self._params / self.stderr

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
    def s2(self):
        return self._s2