import warnings

import numpy as np
import pandas as pd
import pytest
from panel.iv import IV2SLS, IVLIML, IVGMM, IVGMMCUE
from panel.utility import AttrDict


@pytest.fixture(scope='module')
def data():
    n, k, p = 1000, 5, 3
    np.random.seed(12345)
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

    return AttrDict(dep=y, exog=x[:, 2:], endog=x[:, :2], instr=z[:, 3:])


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

        mod = IVLIML(self.y, self.x_exog, self.x_endog, self.z, kappa=0.99)
        res = mod.fit()
        get_all(res)

        mod = IVLIML(self.y, self.x_exog, self.x_endog, self.z, fuller=1)
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
