import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import pinv
from numpy.testing import assert_allclose, assert_equal
from pandas.util.testing import assert_series_equal
from statsmodels.api import add_constant

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from linearmodels.iv.model import _OLS
from linearmodels.iv.results import compare
from linearmodels.utility import AttrDict


@pytest.fixture(scope='module')
def data():
    n, q, k, p = 1000, 2, 5, 3
    np.random.seed(12345)
    clusters = np.random.randint(0, 10, n)
    rho = 0.5
    r = np.zeros((k + p + 1, k + p + 1))
    r.fill(rho)
    r[-1, 2:] = 0
    r[2:, -1] = 0
    r[-1, -1] = 0.5
    r += np.eye(9) * 0.5
    v = np.random.multivariate_normal(np.zeros(r.shape[0]), r, n)
    x = v[:, :k]
    z = v[:, k:k + p]
    e = v[:, [-1]]
    params = np.arange(1, k + 1) / k
    params = params[:, None]
    y = x @ params + e
    xhat = z @ np.linalg.pinv(z) @ x
    nobs, nvar = x.shape
    s2 = e.T @ e / nobs
    s2_debiased = e.T @ e / (nobs - nvar)
    v = xhat.T @ xhat / nobs
    vinv = np.linalg.inv(v)
    kappa = 0.99
    vk = (x.T @ x * (1 - kappa) + kappa * xhat.T @ xhat) / nobs
    return AttrDict(nobs=nobs, e=e, x=x, y=y, z=z, xhat=xhat,
                    params=params, s2=s2, s2_debiased=s2_debiased,
                    clusters=clusters, nvar=nvar, v=v, vinv=vinv, vk=vk,
                    kappa=kappa, dep=y, exog=x[:, q:], endog=x[:, :q],
                    instr=z)


def get_all(v):
    attr = [d for d in dir(v) if not d.startswith('_')]
    for a in attr:
        val = getattr(v, a)
        if a in ('conf_int', 'durbin', 'wu_hausman', 'c_stat'):
            val()


class TestErrors(object):
    def test_rank_deficient_exog(self, data):
        exog = data.exog.copy()
        exog[:, :2] = 1
        with pytest.raises(ValueError):
            IV2SLS(data.dep, exog, data.endog, data.instr)

    def test_rank_deficient_endog(self, data):
        endog = data.endog.copy()
        endog[:, :2] = 1
        with pytest.raises(ValueError):
            IV2SLS(data.dep, data.exog, endog, data.instr)
        with pytest.raises(ValueError):
            IV2SLS(data.dep, data.exog, data.exog, data.instr)

    def test_invalid_weights(self, data):
        weights = np.zeros_like(data.dep)
        with pytest.raises(ValueError):
            IV2SLS(data.dep, data.exog, data.endog, data.instr, weights=weights)

    def test_rank_deficient_instr(self, data):
        instr = data.instr.copy()
        instr[:, :2] = 1
        with pytest.raises(ValueError):
            IV2SLS(data.dep, data.exog, data.endog, instr)
        with pytest.raises(ValueError):
            IV2SLS(data.dep, data.exog, data.endog, data.exog)

    def test_kappa_error(self, data):
        with pytest.raises(ValueError):
            IVLIML(data.dep, data.exog, data.endog, data.instr, kappa=np.array([1]))

    def test_fuller_error(self, data):
        with pytest.raises(ValueError):
            IVLIML(data.dep, data.exog, data.endog, data.instr, fuller=np.array([1]))

    def test_kappa_fuller_warning(self, data):
        with warnings.catch_warnings(record=True) as w:
            IVLIML(data.dep, data.exog, data.endog, data.instr, kappa=0.99, fuller=1)
        assert len(w) == 1

    def test_string_cat(self, data):
        instr = data.instr.copy()
        n = data.instr.shape[0]
        cat = pd.Series(['a'] * (n // 2) + ['b'] * (n // 2))
        instr = pd.DataFrame(instr)
        instr['cat'] = cat
        res = IV2SLS(data.dep, data.exog, data.endog, instr).fit(cov_type='unadjusted')
        instr['cat'] = cat.astype('category')
        res_cat = IV2SLS(data.dep, data.exog, data.endog, instr).fit(cov_type='unadjusted')
        assert_series_equal(res.params, res_cat.params)

    def test_no_regressors(self, data):
        with pytest.raises(ValueError):
            IV2SLS(data.dep, None, None, None)

    def test_too_few_instruments(self, data):
        with pytest.raises(ValueError):
            IV2SLS(data.dep, data.exog, data.endog, None)


def test_2sls_direct(data):
    mod = IV2SLS(data.dep, add_constant(data.exog), data.endog, data.instr)
    res = mod.fit()
    x = np.c_[add_constant(data.exog), data.endog]
    z = np.c_[add_constant(data.exog), data.instr]
    y = data.y
    xhat = z @ pinv(z) @ x
    params = pinv(xhat) @ y
    assert_allclose(res.params, params.ravel())
    # This is just a quick smoke check of results
    get_all(res)


def test_2sls_direct_small(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit()
    res2 = mod.fit(debiased=True)
    assert np.all(res.tstats != res2.tstats)
    get_all(res2)
    fs = res.first_stage
    fs.diagnostics
    # Fetch again to test cache
    get_all(res2)


def test_liml_direct(data):
    mod = IVLIML(data.dep, data.exog, data.endog, data.instr)
    nobs = data.dep.shape[0]
    ninstr = data.exog.shape[1] + data.instr.shape[1]
    res = mod.fit()
    get_all(res)
    mod2 = IVLIML(data.dep, data.exog, data.endog, data.instr, kappa=res.kappa)
    res2 = mod2.fit()
    assert_allclose(res.params, res2.params)
    mod3 = IVLIML(data.dep, data.exog, data.endog, data.instr, fuller=1)
    res3 = mod3.fit()
    assert_allclose(res3.kappa, res.kappa - 1 / (nobs - ninstr))


def test_2sls_ols_equiv(data):
    mod = IV2SLS(data.dep, data.exog, None, None)
    res = mod.fit()
    params = pinv(data.exog) @ data.dep
    assert_allclose(res.params, params.ravel())


def test_gmm_iter(data):
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit(iter_limit=100)
    assert res.iterations > 2
    # This is just a quick smoke check of results
    get_all(res)


def test_gmm_cue(data):
    mod = IVGMMCUE(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit(display=False)
    assert res.iterations > 2
    mod2 = IVGMM(data.dep, data.exog, data.endog, data.instr)
    res2 = mod2.fit()
    assert res.j_stat.stat <= res2.j_stat.stat

    mod = IVGMMCUE(data.dep, data.exog, data.endog, data.instr, center=False)
    res = mod.fit(display=False)
    mod2 = IVGMM(data.dep, data.exog, data.endog, data.instr, center=False)
    res2 = mod2.fit()
    assert res.j_stat.stat <= res2.j_stat.stat


def test_gmm_cue_starting_vals(data):
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    sv = mod.fit().params
    mod = IVGMMCUE(data.dep, data.exog, data.endog, data.instr)
    mod.fit(starting=sv, display=False)

    with pytest.raises(ValueError):
        mod.fit(starting=sv[:-1], display=True)


def test_2sls_just_identified(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr[:, :2])
    res = mod.fit()
    get_all(res)
    fs = res.first_stage
    fs.diagnostics
    # Fetch again to test cache
    get_all(fs)
    get_all(res)

    mod = IV2SLS(data.dep, data.exog, data.endog[:, :1], data.instr[:, :1])
    res = mod.fit()
    get_all(res)
    fs = res.first_stage
    fs.diagnostics
    get_all(fs)

    mod = IV2SLS(data.dep, None, data.endog[:, :1], data.instr[:, :1])
    res = mod.fit()
    get_all(res)
    fs = res.first_stage
    fs.diagnostics
    get_all(fs)


def test_durbin_smoke(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit()
    res.durbin()
    res.durbin([mod.endog.cols[1]])


def test_wu_hausman_smoke(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit()
    res.wu_hausman()
    res.wu_hausman([mod.endog.cols[1]])


def test_wooldridge_smoke(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit()
    res.wooldridge_regression
    res.wooldridge_score


def test_model_summary_smoke(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit()
    res.__repr__()
    res.__str__()
    res._repr_html_()
    res.summary

    res = _OLS(data.dep, data.exog).fit()
    res.__repr__()
    res.__str__()
    res._repr_html_()
    res.summary


def test_model_missing(data):
    import copy
    data2 = AttrDict()
    for key in data:
        data2[key] = copy.deepcopy(data[key])
    data = data2
    data.dep[::7, :] = np.nan
    data.exog[::13, :] = np.nan
    data.endog[::23, :] = np.nan
    data.instr[::29, :] = np.nan
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit()

    vars = [data.dep, data.exog, data.endog, data.instr]
    missing = list(map(lambda x: np.any(np.isnan(x), 1), vars))
    missing = np.any(np.c_[missing], 0)
    not_missing = missing.shape[0] - missing.sum()
    assert res.nobs == not_missing
    assert_equal(mod.isnull, missing)
    assert_equal(mod.notnull, ~missing)


def test_compare(data):
    res1 = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit()
    res2 = IV2SLS(data.dep, data.exog, data.endog, data.instr[:, :-1]).fit()
    res3 = IVGMM(data.dep, data.exog[:, :2], data.endog, data.instr).fit()
    res4 = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit()
    c = compare([res1, res2, res3, res4])
    assert len(c.rsquared) == 4
    c.summary
    c = compare({'Model A': res1,
                 'Model B': res2,
                 'Model C': res3,
                 'Model D': res4})
    c.summary
    res = OrderedDict()
    res['Model A'] = res1
    res['Model B'] = res2
    res['Model C'] = res3
    res['Model D'] = res4
    c = compare(res)
    c.summary
    c.pvalues

    res1 = IV2SLS(data.dep, data.exog[:, :1], None, None).fit()
    res2 = IV2SLS(data.dep, data.exog[:, :2], None, None).fit()
    c = compare({'Model A': res1,
                 'Model B': res2})
    c.summary


def test_compare_single(data):
    res1 = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit()
    c = compare([res1])
    assert len(c.rsquared) == 1
    c.summary
    c = compare({'Model A': res1})
    c.summary
    res = OrderedDict()
    res['Model A'] = res1
    c = compare(res)
    c.summary
    c.pvalues


def test_compare_single_single_parameter(data):
    res1 = IV2SLS(data.dep, data.exog[:, :1], None, None).fit()
    c = compare([res1])
    assert len(c.rsquared) == 1
    c.summary


def test_first_stage_summary(data):
    res1 = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit()
    res1.first_stage.summary


def test_gmm_str(data):
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    str(mod.fit(cov_type='unadjusted'))
    str(mod.fit(cov_type='robust'))
    str(mod.fit(cov_type='clustered', clusters=data.clusters))
    str(mod.fit(cov_type='kernel'))
