import numpy as np
import pytest

from panel.iv import IV2SLS, IVGMM, IVLIML


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
        cls.z = np.c_[cls.x[:, 1:], cls.z]
        cls.y = cls.x @ beta + cls.e

    def test_iv2sls_smoke(self):
        mod = IV2SLS(self.y, self.x, self.z)
        mod.fit()

    def test_iv2sls_smoke_homoskedastic(self):
        mod = IV2SLS(self.y, self.x, self.z)
        mod.fit(cov_type='unadjusted')

    def test_iv2sls_smoke_cov_config(self):
        mod = IV2SLS(self.y, self.x, self.z)
        mod.fit(cov_type='unadjusted', debiased=True)

    def test_invalid_cov_config(self):
        mod = IV2SLS(self.y, self.x, self.z)
        with pytest.raises(ValueError):
            mod.fit(cov_type='unadjusted', large_sample=True)

    def test_iv2sls_smoke_nw(self):
        mod = IV2SLS(self.y, self.x, self.z)
        mod.fit(cov_type='newey-west')
        mod.fit(cov_type='bartlett')
        mod.fit(cov_type='bartlett', bw=0)

    def test_iv2sls_smoke_cluster(self):
        mod = IV2SLS(self.y, self.x, self.z)

        clusters = np.tile(np.arange(2), (self.y.shape[0] // 2,)).ravel()
        mod.fit(cov_type='one-way', clusters=clusters)

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

        mod.fit(cov_type='one-way')

    def test_ivgmm_smoke(self):
        mod = IVGMM(self.y, self.x, self.z)
        mod.fit()

    def test_ivgmm_smoke_iter(self):
        mod = IVGMM(self.y, self.x, self.z)
        res = mod.fit(iter_limit=100)
        print(res.iterations)

    def test_ivgmm_smoke_weights(self):
        mod = IVGMM(self.y, self.x, self.z, weight_type='unadjusted')
        mod.fit()

        with pytest.raises(ValueError):
            IVGMM(self.y, self.x, self.z, bw=20)

    def test_ivgmm_kernel_smoke(self):
        mod = IVGMM(self.y, self.x, self.z, weight_type='kernel')
        mod.fit()

    def test_ivgmm_cluster_smoke(self):
        k = 500
        clusters = np.tile(np.arange(k), (self.y.shape[0] // k, 1)).ravel()
        mod = IVGMM(self.y, self.x, self.z, weight_type='clustered',
                    clusters=clusters)
        mod.fit()

    def test_ivgmm_cluster_is(self):
        mod = IVGMM(self.y, self.x, self.z, weight_type='clustered',
                    clusters=np.arange(self.y.shape[0]))
        mod.fit()

        mod = IVGMM(self.y, self.x, self.z)
        mod.fit()

    def test_ivliml_smoke(self):
        mod = IVLIML(self.y, self.x, self.z)
        res = mod.fit()
        print(res.params)
