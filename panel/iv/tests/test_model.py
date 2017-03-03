import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import pinv
from numpy.testing import assert_allclose

from panel.iv import IV2SLS, IVLIML, IVGMM, IVGMMCUE
from panel.utility import AttrDict


@pytest.fixture(scope='module')
def data():
    n, k, p = 1000, 5, 3
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
    z = v[:, 2:k + p]
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
                    kappa=kappa, dep=y, exog=x[:, 2:], endog=x[:, :2],
                    instr=z[:, 3:])


def get_all(v):
    attr = [d for d in dir(v) if not d.startswith('_')]
    for a in attr:
        val = getattr(v, a)
        if a == 'conf_int':
            val = val()


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

    def test_invalid_cat(self, data):
        instr = data.instr.copy()
        n = data.instr.shape[0]
        cat = pd.Series(['a'] * (n // 2) + ['b'] * (n // 2))
        instr = pd.DataFrame(instr)
        instr['cat'] = cat
        with pytest.raises(ValueError):
            IV2SLS(data.dep, data.exog, data.endog, instr)

    def test_no_regressors(self, data):
        with pytest.raises(ValueError):
            IV2SLS(data.dep, None, None, None)

    def test_too_few_instruments(self, data):
        with pytest.raises(ValueError):
            IV2SLS(data.dep, data.exog, data.endog, None)


def test_2sls_direct(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit()
    x = np.c_[data.exog, data.endog]
    z = np.c_[data.exog, data.instr]
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
    stats = fs.rsquared
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
    res = mod.fit()
    assert res.iterations > 2
    mod2 = IVGMM(data.dep, data.exog, data.endog, data.instr)
    res2 = mod2.fit()
    assert res.j_stat.stat <= res2.j_stat.stat

    mod = IVGMMCUE(data.dep, data.exog, data.endog, data.instr, center=False)
    res = mod.fit()
    mod2 = IVGMM(data.dep, data.exog, data.endog, data.instr, center=False)
    res2 = mod2.fit()
    assert res.j_stat.stat <= res2.j_stat.stat


def test_2sls_just_identified(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr[:, :2])
    res = mod.fit()
    get_all(res)
    fs = res.first_stage
    stats = fs.rsquared
    # Fetch again to test cache
    get_all(res)
