from copy import deepcopy

import numpy as np
import os
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from panel.iv import IV2SLS, IVLIML, IVGMM, IVGMMCUE
from panel.iv.tests.results.read_stata_results import process_results

CWD = os.path.split(os.path.abspath(__file__))[0]

HOUSING_DATA = pd.read_csv(os.path.join(CWD, 'housing.csv'), index_col=0)
HOUSING_DATA.region = HOUSING_DATA.region.astype('category')
HOUSING_DATA.state = HOUSING_DATA.state.astype('category')
HOUSING_DATA.division = HOUSING_DATA.division.astype('category')

SIMULATED_DATA = pd.read_stata(os.path.join(CWD, 'results', 'simulated-data.dta'))

filepath = os.path.join(CWD, 'results', 'stata-iv-housing-results.txt')
HOUSING_RESULTS = process_results(filepath)
filepath = os.path.join(CWD, 'results', 'stata-iv-simulated-results.txt')
SIMULATED_RESULTS = process_results(filepath)

MODELS = {'2sls': IV2SLS, 'gmm': IVGMM, 'liml': IVLIML}
COV_OPTIONS = {'cluster': {'cov_type': 'clustered', 'clusters': HOUSING_DATA.division},
               'robust': {'cov_type': 'robust'},
               'unadjusted': {'cov_type': 'unadjusted'},
               'bartlett_12': {'cov_type': 'kernel', 'kernel': 'bartlett', 'bandwidth': 12}}


@pytest.fixture(params=list(HOUSING_RESULTS.keys()))
def housing(request):
    result = HOUSING_RESULTS[request.param]
    keys = request.param.split('-')
    mod = MODELS[keys[0]]

    data = HOUSING_DATA
    endog = data.rent
    exog = sm.add_constant(data.pcturban)
    instd = data.hsngval
    instr = data[['faminc', 'region']]
    cov_opts = deepcopy(COV_OPTIONS[keys[1]])
    cov_opts['debiased'] = keys[2] == 'small'
    if keys[0] == 'gmm':
        weight_opts = deepcopy(COV_OPTIONS[keys[1]])
        weight_opts['weight_type'] = weight_opts['cov_type']
        del weight_opts['cov_type']
    else:
        weight_opts = {}

    model_result = mod(endog, exog, instd, instr, **weight_opts).fit(**cov_opts)
    return model_result, result


def get_all(v):
    attr = [d for d in dir(v) if not d.startswith('_')]
    for a in attr:
        val = getattr(v, a)
        if a == 'conf_int':
            val = val()
        print(a)
        print(val)


class TestHousingResults(object):
    def test_rsquared(self, housing):
        res, stata = housing
        assert_allclose(res.rsquared, stata.rsquared)

    def test_rsquared_adj(self, housing):
        res, stata = housing
        assert_allclose(res.rsquared_adj, stata.rsquared_adj)

    def test_model_ss(self, housing):
        res, stata = housing
        assert_allclose(res.model_ss, stata.model_ss)

    def test_residual_ss(self, housing):
        res, stata = housing
        assert_allclose(res.resid_ss, stata.resid_ss)

    def test_fstat(self, housing):
        res, stata = housing
        assert_allclose(res.f_statistic.stat, stata.f_statistic)

    def test_params(self, housing):
        res, stata = housing
        stata_params = stata.params.reindex_like(res.params)
        assert_allclose(res.params, stata_params)

    def test_tstats(self, housing):
        res, stata = housing
        stata_tstats = stata.tstats.reindex_like(res.params)
        assert_allclose(res.tstats, stata_tstats)

    def test_cov(self, housing):
        res, stata = housing
        stata_cov = stata.cov.reindex_like(res.cov)
        stata_cov = stata_cov[res.cov.columns]
        assert_allclose(res.cov, stata_cov, rtol=1e-4)


SIMULATED_COV_OPTIONS = {
    'cluster': {'cov_type': 'clustered', 'clusters': SIMULATED_DATA.cluster_id},
    'robust': {'cov_type': 'robust'},
    'unadjusted': {'cov_type': 'unadjusted'},
    'bartlett_12': {'cov_type': 'kernel', 'kernel': 'bartlett', 'bandwidth': 12}}

keys = ['gmm-bartlett_12-asymptotic']


@pytest.fixture(params=keys)  # list(SIMULATED_RESULTS.keys()))
def simulated(request):
    result = SIMULATED_RESULTS[request.param]
    print(request.param)
    keys = request.param.split('-')
    mod = MODELS[keys[0]]

    data = SIMULATED_DATA
    deps = {'unadjusted': data.y_unadjusted,
            'robust': data.y_robust,
            'cluster': data.y_clustered,
            'bartlett_12': data.y_kernel}
    dep = deps[keys[1]]
    exog = sm.add_constant(data[['x2', 'x3', 'x4']])
    instd = data.x1
    instr = data.z

    cov_opts = deepcopy(SIMULATED_COV_OPTIONS[keys[1]])
    cov_opts['debiased'] = keys[2] == 'small'
    if keys[0] == 'gmm':
        weight_opts = deepcopy(SIMULATED_COV_OPTIONS[keys[1]])
        weight_opts['weight_type'] = weight_opts['cov_type']
        del weight_opts['cov_type']
    else:
        weight_opts = {}

    model_result = mod(dep, exog, instd, instr, **weight_opts).fit(**cov_opts)
    return model_result, result


class TestSimulatedResults(object):
    def test_rsquared(self, simulated):
        res, stata = simulated
        assert_allclose(res.rsquared, stata.rsquared)

    def test_rsquared_adj(self, simulated):
        res, stata = simulated
        assert_allclose(res.rsquared_adj, stata.rsquared_adj)

    def test_model_ss(self, simulated):
        res, stata = simulated
        assert_allclose(res.model_ss, stata.model_ss)

    def test_residual_ss(self, simulated):
        res, stata = simulated
        assert_allclose(res.resid_ss, stata.resid_ss)

    def test_fstat(self, simulated):
        res, stata = simulated
        a = res.f_statistic
        b = stata.f_statistic
        assert_allclose(res.f_statistic.stat, stata.f_statistic)

    def test_params(self, simulated):
        res, stata = simulated
        stata_params = stata.params.reindex_like(res.params)
        assert_allclose(res.params, stata_params)

    def test_tstats(self, simulated):
        res, stata = simulated
        stata_tstats = stata.tstats.reindex_like(res.params)
        assert_allclose(res.tstats, stata_tstats)

    def test_cov(self, simulated):
        res, stata = simulated
        stata_cov = stata.cov.reindex_like(res.cov)
        stata_cov = stata_cov[res.cov.columns]
        assert_allclose(res.cov, stata_cov, rtol=1e-4)


class TestIV(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        t, k, m = 5000, 3, 3
        beta = np.arange(1, k + 1)[:, None]
        cls.x = np.random.standard_normal((t, k))
        cls.e = np.random.standard_normal((t, 1))
        cls.z = np.random.standard_normal((t, m))
        cls.x[:, 0] = cls.x[:, 0] + cls.e[:, 0] + cls.z.sum(axis=1)
        cls.x_endog = cls.x[:, [0]]
        cls.x_exog = cls.x[:, 1:]
        cls.y = cls.x @ beta + cls.e

    #
    def test_iv2sls_smoke(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit()

    #
    def test_fake_ols_smoke(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit()
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.x_endog)
        mod.fit()

    #
    def test_iv2sls_smoke_homoskedastic(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='unadjusted')

    #
    def test_iv2sls_smoke_cov_config(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='unadjusted', debiased=True)

    #
    def test_iv2sls_smoke_nw(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='kernel', kernel='newey-west')
        mod.fit(cov_type='kernel', kernel='bartlett')
        mod.fit(cov_type='kernel', kernel='parzen')
        mod.fit(cov_type='kernel', kernel='qs')

    #
    def test_iv2sls_smoke_cluster(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        #
        clusters = np.tile(np.arange(5), (self.y.shape[0] // 5,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)
        #
        clusters = np.tile(np.arange(100), (self.y.shape[0] // 100,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)
        #
        clusters = np.tile(np.arange(500), (self.y.shape[0] // 500,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)
        #
        clusters = np.tile(np.arange(1000), (self.y.shape[0] // 1000,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)
        #
        clusters = np.tile(np.arange(2500), (self.y.shape[0] // 2500,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)
        #
        res = mod.fit(cov_type='one-way')
        get_all(res)

    #
    def test_ivgmm_smoke(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit()

    #
    def test_ivgmm_smoke_iter(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(iter_limit=100)

    #
    def test_ivgmm_smoke_weights(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='unadjusted')
        mod.fit()
        #
        with pytest.raises(TypeError):
            IVGMM(self.y, self.x_exog, self.x_endog, self.z, bw=20)
            #

    def test_ivgmm_kernel_smoke(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='kernel')
        mod.fit()
        #
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='kernel',
                    kernel='parzen')
        mod.fit()
        #
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='kernel', kernel='qs')
        mod.fit()

    #
    def test_ivgmm_cluster_smoke(self):
        k = 500
        clusters = np.tile(np.arange(k), (self.y.shape[0] // k, 1)).ravel()
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='clustered',
                    clusters=clusters)
        res = mod.fit()

    #
    def test_ivgmm_cluster_size_1(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='clustered',
                    clusters=np.arange(self.y.shape[0]))
        mod.fit()
        #
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        res = mod.fit()
        get_all(res)

    #
    def test_ivliml_smoke(self):
        mod = IVLIML(self.y, self.x_exog, self.x_endog, self.z)
        res = mod.fit()
        get_all(res)

    #
    def test_ivgmmcue_smoke(self):
        mod = IVGMMCUE(self.y, self.x_exog, self.x_endog, self.z)
        res = mod.fit()
        get_all(res)
        print(res.j_stat)

    #
    def test_alt_dims_smoke(self):
        mod = IV2SLS(self.y.squeeze(), self.x_exog.squeeze(),
                     self.x_endog.squeeze(), self.z.squeeze())
        mod.fit()

    #
    def test_pandas_smoke(self):
        mod = IV2SLS(pd.Series(self.y.squeeze()), pd.DataFrame(self.x_exog.squeeze()),
                     pd.Series(self.x_endog.squeeze()), pd.DataFrame(self.z.squeeze()))
        mod.fit()

    #
    def test_real(self):
        path = os.path.join(CWD, 'housing.csv')
        data = pd.read_csv(path, index_col=0)
        endog = data.rent
        exog = sm.add_constant(data.pcturban)
        instd = data.hsngval
        instr = pd.concat([data.faminc, pd.get_dummies(data.region, drop_first=True)], axis=1)
        #
        mod = IV2SLS(endog, exog, instd, instr)
        mod.fit(cov_type='unadjusted')

    #
    def test_real_cat(self):
        path = os.path.join(CWD, 'housing.csv')
        data = pd.read_csv(path, index_col=0)
        data.region = data.region.astype('category')
        data.state = data.state.astype('category')
        data.division = data.division.astype('category')
        endog = data.rent
        exog = sm.add_constant(data.pcturban)
        instd = data.hsngval
        instr = data[['faminc', 'region']]
        #
        mod = IV2SLS(endog, exog, instd, instr)
        res = mod.fit(cov_type='unadjusted')
        get_all(res)

    #
    def test_invalid_cat(self):
        path = os.path.join(CWD, 'housing.csv')
        data = pd.read_csv(path, index_col=0)
        endog = data.rent
        exog = sm.add_constant(data.pcturban)
        instd = data.hsngval
        instr = data[['faminc', 'region']]
        #
        with pytest.raises(ValueError):
            IV2SLS(endog, exog, instd, instr)
            #

    def test_gmm_homo(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='unadjusted')

    #
    def test_gmm_hetero(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='robust')

    #
    def test_gmm_clustered(self):
        clusters = np.tile(np.arange(500), (self.y.shape[0] // 500,)).ravel()
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='clustered', clusters=clusters)

    #
    def test_gmm_kernel(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='kernel')
        #
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='kernel', kernel='qs', bandwidth=100)
