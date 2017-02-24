import os

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from panel.iv import IV2SLS, IVGMM, IVLIML, IVGMMCUE

CWD = os.path.split(os.path.abspath(__file__))[0]

HOUSING_DATA = pd.read_csv(os.path.join(CWD, 'housing.csv'), index_col=0)
HOUSING_DATA.region = HOUSING_DATA.region.astype('category')
HOUSING_DATA.state = HOUSING_DATA.state.astype('category')
HOUSING_DATA.division = HOUSING_DATA.division.astype('category')


def get_all(v):
    attr = [d for d in dir(v) if not d.startswith('_')]
    for a in attr:
        val = getattr(v, a)
        if a == 'conf_int':
            val = val()
        print(a)
        print(val)


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

    def test_iv2sls_smoke(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit()

    def test_fake_ols_smoke(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit()
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.x_endog)
        mod.fit()

    def test_iv2sls_smoke_homoskedastic(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='unadjusted')

    def test_iv2sls_smoke_cov_config(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='unadjusted', debiased=True)

    def test_iv2sls_smoke_nw(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='kernel', kernel='newey-west')
        mod.fit(cov_type='kernel', kernel='bartlett')
        mod.fit(cov_type='kernel', kernel='parzen')
        mod.fit(cov_type='kernel', kernel='qs')

    def test_iv2sls_smoke_cluster(self):
        mod = IV2SLS(self.y, self.x_exog, self.x_endog, self.z)

        clusters = np.tile(np.arange(5), (self.y.shape[0] // 5,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)

        clusters = np.tile(np.arange(100), (self.y.shape[0] // 100,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)

        clusters = np.tile(np.arange(500), (self.y.shape[0] // 500,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)

        clusters = np.tile(np.arange(1000), (self.y.shape[0] // 1000,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)

        clusters = np.tile(np.arange(2500), (self.y.shape[0] // 2500,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)

        res = mod.fit(cov_type='one-way')
        get_all(res)

    def test_ivgmm_smoke(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit()

    def test_ivgmm_smoke_iter(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(iter_limit=100)

    def test_ivgmm_smoke_weights(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='unadjusted')
        mod.fit()

        with pytest.raises(TypeError):
            IVGMM(self.y, self.x_exog, self.x_endog, self.z, bw=20)

    def test_ivgmm_kernel_smoke(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='kernel')
        mod.fit()

        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='kernel',
                    kernel='parzen')
        mod.fit()

        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='kernel', kernel='qs')
        mod.fit()

    def test_ivgmm_cluster_smoke(self):
        k = 500
        clusters = np.tile(np.arange(k), (self.y.shape[0] // k, 1)).ravel()
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='clustered',
                    clusters=clusters)
        res = mod.fit()

    def test_ivgmm_cluster_size_1(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z, weight_type='clustered',
                    clusters=np.arange(self.y.shape[0]))
        mod.fit()

        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        res = mod.fit()
        get_all(res)

    def test_ivliml_smoke(self):
        mod = IVLIML(self.y, self.x_exog, self.x_endog, self.z)
        res = mod.fit()
        get_all(res)

    def test_ivgmmcue_smoke(self):
        mod = IVGMMCUE(self.y, self.x_exog, self.x_endog, self.z)
        res = mod.fit()
        get_all(res)
        print(res.j_stat)

    def test_alt_dims_smoke(self):
        mod = IV2SLS(self.y.squeeze(), self.x_exog.squeeze(),
                     self.x_endog.squeeze(), self.z.squeeze())
        mod.fit()

    def test_pandas_smoke(self):
        mod = IV2SLS(pd.Series(self.y.squeeze()), pd.DataFrame(self.x_exog.squeeze()),
                     pd.Series(self.x_endog.squeeze()), pd.DataFrame(self.z.squeeze()))
        mod.fit()

    def test_real(self):
        path = os.path.join(CWD, 'housing.csv')
        data = pd.read_csv(path, index_col=0)
        endog = data.rent
        exog = sm.add_constant(data.pcturban)
        instd = data.hsngval
        instr = pd.concat([data.faminc, pd.get_dummies(data.region, drop_first=True)], axis=1)

        mod = IV2SLS(endog, exog, instd, instr)
        mod.fit(cov_type='unadjusted')

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

        mod = IV2SLS(endog, exog, instd, instr)
        res = mod.fit(cov_type='unadjusted')
        get_all(res)

    def test_invalid_cat(self):
        path = os.path.join(CWD, 'housing.csv')
        data = pd.read_csv(path, index_col=0)
        endog = data.rent
        exog = sm.add_constant(data.pcturban)
        instd = data.hsngval
        instr = data[['faminc', 'region']]

        with pytest.raises(ValueError):
            IV2SLS(endog, exog, instd, instr)

    def test_gmm_homo(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='unadjusted')

    def test_gmm_hetero(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='robust')

    def test_gmm_clustered(self):
        clusters = np.tile(np.arange(500), (self.y.shape[0] // 500,)).ravel()
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='clustered', clusters=clusters)

    def test_gmm_kernel(self):
        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='kernel')

        mod = IVGMM(self.y, self.x_exog, self.x_endog, self.z)
        mod.fit(cov_type='kernel', kernel='qs', bandwidth=100)


class CheckIV2SLSAgainstStata(object):
    @classmethod
    def setup_class(cls):
        from panel.iv.tests.results.read_stata_results import read_result

        data = HOUSING_DATA
        endog = data.rent
        exog = sm.add_constant(data.pcturban)
        instd = data.hsngval
        instr = data[['faminc', 'region']]

        mod = IV2SLS(endog, exog, instd, instr)
        cls.res = mod.fit(**cls.fit_opts)
        filepath = os.path.join(CWD, 'results', cls.file)
        cls.stata = read_result(filepath)
        print(cls.stata)

    def test_rsquared(self):
        assert_allclose(self.res.rsquared, self.stata.rsquared)

    def test_rsquared_adj(self):
        assert_allclose(self.res.rsquared_adj, self.stata.rsquared_adj)

    def test_total_ss(self):
        assert_allclose(self.res.model_ss, self.stata.model_ss)

    def test_residual_ss(self):
        assert_allclose(self.res.resid_ss, self.stata.resid_ss)

    def test_fstat(self):
        assert_allclose(self.res.f_statistic.stat, self.stata.f_statistic)

    def test_params(self):
        for name in self.res.params.index:
            assert_allclose(self.res.params[name], self.stata.params[name])

    def test_tstats(self):
        for name in self.res.tstats.index:
            assert_allclose(self.res.tstats[name], self.stata.tstats[name])

    def test_cov(self):
        names = self.res.tstats.index
        for row in names:
            for col in names:
                assert_allclose(self.res.cov[col][row], self.stata.cov[col][row],
                                rtol=1e-4)


class CheckIVGMMAgainstStata(object):
    @classmethod
    def setup_class(cls):
        from panel.iv.tests.results.read_stata_results import read_result

        data = HOUSING_DATA
        endog = data.rent
        exog = sm.add_constant(data.pcturban)
        instd = data.hsngval
        instr = data[['faminc', 'region']]

        mod = IVGMM(endog, exog, instd, instr, **cls.mod_opts)
        cls.res = mod.fit(**cls.fit_opts)
        filepath = os.path.join(CWD, 'results', cls.file)
        cls.stata = read_result(filepath)
        print(cls.stata)


class TestIV2SLSStataUnadjusted(CheckIV2SLSAgainstStata):
    @classmethod
    def setup_class(cls):
        cls.file = 'stata-iv2sls-unadjusted.txt'
        cls.fit_opts = {'cov_type': 'unadjusted'}
        super(TestIV2SLSStataUnadjusted, cls).setup_class()


class TestIV2SLSStataRobust(CheckIV2SLSAgainstStata):
    @classmethod
    def setup_class(cls):
        cls.file = 'stata-iv2sls-robust.txt'
        cls.fit_opts = {'cov_type': 'robust'}
        super(TestIV2SLSStataRobust, cls).setup_class()


class TestIVGMMStataRobust(CheckIVGMMAgainstStata):
    @classmethod
    def setup_class(cls):
        cls.file = 'stata-ivgmm-robust.txt'
        cls.fit_opts = {'cov_type': 'robust'}
        cls.mod_opts = {'weight_type': 'robust'}
        super(TestIVGMMStataRobust, cls).setup_class()


class TestIVGMMStataUnadjusted(CheckIVGMMAgainstStata):
    @classmethod
    def setup_class(cls):
        cls.file = 'stata-ivgmm-unadjusted.txt'
        cls.fit_opts = {'cov_type': 'unadjusted'}
        cls.mod_opts = {'weight_type': 'unadjusted'}
        super(TestIVGMMStataUnadjusted, cls).setup_class()
