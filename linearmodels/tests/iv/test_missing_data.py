import numpy as np
import pytest
from pandas.util.testing import assert_series_equal

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from linearmodels.utility import AttrDict


@pytest.fixture(scope='module', params=[IV2SLS, IVLIML, IVGMM, IVGMMCUE])
def model(request):
    return request.param


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
    v.flat[::93] = np.nan
    x = v[:, :k]
    z = v[:, k:k + p]
    e = v[:, [-1]]
    params = np.arange(1, k + 1) / k
    params = params[:, None]
    y = x @ params + e

    dep = y
    exog = x[:, q:]
    endog = x[:, :q]
    instr = z

    not_missing = ~np.any(np.isnan(v), 1)
    y_clean = y[not_missing]
    x_clean = x[not_missing]
    z_clean = z[not_missing]
    dep_clean = y_clean
    exog_clean = x_clean[:, q:]
    endog_clean = x_clean[:, :q]
    instr_clean = z_clean
    clusters_clean = clusters[not_missing]
    return AttrDict(dep=dep, exog=exog, endog=endog, instr=instr,
                    dep_clean=dep_clean, exog_clean=exog_clean,
                    endog_clean=endog_clean, instr_clean=instr_clean,
                    clusters=clusters, clusters_clean=clusters_clean)


def get_all(v):
    attr = [d for d in dir(v) if not d.startswith('_')]
    for a in attr:
        val = getattr(v, a)
        if a in ('conf_int', 'durbin', 'wu_hausman', 'c_stat'):
            val()


def test_missing(data, model):
    mod = model(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit()
    mod = model(data.dep_clean, data.exog_clean,
                data.endog_clean, data.instr_clean)
    res2 = mod.fit()
    assert res.nobs == res2.nobs
    assert_series_equal(res.params, res2.params)
    get_all(res)


def test_missing_clustered(data, model):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr)
    with pytest.raises(ValueError):
        res = mod.fit(cov_type='clustered', clusters=data.clusters)
    res = mod.fit(cov_type='clustered', clusters=data.clusters_clean)
    mod = IV2SLS(data.dep_clean, data.exog_clean,
                 data.endog_clean, data.instr_clean)
    res2 = mod.fit(cov_type='clustered', clusters=data.clusters_clean)
    assert res.nobs == res2.nobs
    assert_series_equal(res.params, res2.params)
    get_all(res)


def test_all_missing(data, model):
    with pytest.raises(ValueError):
        model(data.dep * np.nan, data.exog, data.endog, data.instr)
