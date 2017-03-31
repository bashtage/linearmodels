import os
from copy import deepcopy

import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from linearmodels.iv import IV2SLS, IVGMM, IVLIML
from linearmodels.tests.iv.results.read_stata_results import process_results

CWD = os.path.split(os.path.abspath(__file__))[0]

HOUSING_DATA = pd.read_csv(os.path.join(CWD, 'results', 'housing.csv'), index_col=0)
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


@pytest.fixture(params=list(HOUSING_RESULTS.keys()), scope='module')
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
    'vce(cluster cluster_id)': {'cov_type': 'clustered', 'clusters': SIMULATED_DATA.cluster_id},
    'vce(robust)': {'cov_type': 'robust'},
    'vce(unadjusted)': {'cov_type': 'unadjusted'},
    'vce(hac bartlett 12)': {'cov_type': 'kernel', 'kernel': 'bartlett', 'bandwidth': 12}}


def construct_model(key):
    model, nendog, nexog, ninstr, weighted, var, other = key.split('-')
    var = var.replace('wmatrix', 'vce')
    mod = MODELS[model]
    data = SIMULATED_DATA
    endog = data[['x1', 'x2']] if '2' in nendog else data.x1
    exog = data[['x3', 'x4', 'x5']]
    instr = data[['z1', 'z2']] if '2' in ninstr else data.z1
    deps = {'vce(unadjusted)': data.y_unadjusted,
            'vce(robust)': data.y_robust,
            'vce(cluster cluster_id)': data.y_clustered,
            'vce(hac bartlett 12)': data.y_kernel}
    dep = deps[var]
    if 'noconstant' not in other:
        exog = sm.add_constant(data[['x3', 'x4', 'x5']])

    cov_opts = deepcopy(SIMULATED_COV_OPTIONS[var])
    cov_opts['debiased'] = 'small' in other
    mod_options = {}
    if 'True' in weighted:
        mod_options['weights'] = data.weights
    if model == 'gmm':
        mod_options.update(deepcopy(SIMULATED_COV_OPTIONS[var]))
        mod_options['weight_type'] = mod_options['cov_type']
        del mod_options['cov_type']
        mod_options['center'] = 'center' in other

    model_result = mod(dep, exog, endog, instr, **mod_options).fit(**cov_opts)
    if model == 'gmm' and 'True' in weighted:
        pytest.skip('Weighted GMM differs slightly')
    return model_result


@pytest.fixture(params=list(SIMULATED_RESULTS.keys()),
                scope='module')
def simulated(request):
    result = SIMULATED_RESULTS[request.param]
    model_result = construct_model(request.param)
    return model_result, result


class TestSimulatedResults(object):
    def test_rsquared(self, simulated):
        res, stata = simulated
        if stata.rsquared is None:
            return
        assert_allclose(res.rsquared, stata.rsquared)

    def test_rsquared_adj(self, simulated):
        res, stata = simulated
        if stata.rsquared_adj is None:
            return
        assert_allclose(res.rsquared_adj, stata.rsquared_adj)

    def test_model_ss(self, simulated):
        res, stata = simulated
        assert_allclose(res.model_ss, stata.model_ss)

    def test_residual_ss(self, simulated):
        res, stata = simulated
        assert_allclose(res.resid_ss, stata.resid_ss)

    def test_fstat(self, simulated):
        res, stata = simulated
        if stata.f_statistic is None:
            pytest.skip('Comparison result not available')
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

    def test_weight_mat(self, simulated):
        res, stata = simulated
        if not hasattr(stata, 'weight_mat') or not isinstance(stata.weight_mat, pd.DataFrame):
            return
        stata_weight_mat = stata.weight_mat.reindex_like(res.weight_matrix)
        stata_weight_mat = stata_weight_mat[res.weight_matrix.columns]
        assert_allclose(res.weight_matrix, stata_weight_mat, rtol=1e-4)

    def test_j_stat(self, simulated):
        res, stata = simulated
        if not hasattr(stata, 'J') or stata.J is None:
            return
        assert_allclose(res.j_stat.stat, stata.J, atol=1e-6, rtol=1e-4)

    def test_kappa(self, simulated):
        res, stata = simulated
        if not hasattr(stata, 'kappa') or stata.kappa is None:
            return
        assert_allclose(res.kappa, stata.kappa, rtol=1e-4)
