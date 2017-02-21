import numpy as np
import pytest

from panel.iv import IV2SLS, IVGMM


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
        res = mod.fit()
        # print('White')
        # print(res.params)
        # print(res.tstats)

    def test_iv2sls_smoke_homoskedastic(self):
        mod = IV2SLS(self.y, self.x, self.z)
        res = mod.fit(cov_type='unadjusted')
        # print(res.params)
        # print(res.tstats)

    def test_iv2sls_smoke_cov_config(self):
        mod = IV2SLS(self.y, self.x, self.z)
        res = mod.fit(cov_type='unadjusted', debiased=True)
        # print(res.params)
        # print(res.tstats)

    def test_invalid_cov_config(self):
        mod = IV2SLS(self.y, self.x, self.z)
        with pytest.raises(ValueError):
            mod.fit(cov_type='unadjusted', large_sample=True)

    def test_iv2sls_smoke_nw(self):
        mod = IV2SLS(self.y, self.x, self.z)
        res = mod.fit(cov_type='newey-west')
        # print('NW ')
        # print(res.params)
        # print(res.tstats)
        res = mod.fit(cov_type='bartlett')
        # print('Bartlett')
        # print(res.params)
        # print(res.tstats)
        res = mod.fit(cov_type='bartlett', bw=0)
        # print('Bartlett 0 ')
        # print(res.params)
        # print(res.tstats)

    def test_iv2sls_smoke_cluster(self):
        mod = IV2SLS(self.y, self.x, self.z)

        clusters = np.tile(np.arange(2), (self.y.shape[0] // 2,)).ravel()
        res = mod.fit(cov_type='one-way', clusters=clusters)
        # print('Clustered-Two')
        # print(res.params)
        # print(res.tstats)

        clusters = np.tile(np.arange(5), (self.y.shape[0] // 5,)).ravel()
        res = mod.fit(cov_type='one-way', clusters=clusters)
        # print('Clustered-Few')
        # print(res.params)
        # print(res.tstats)


        clusters = np.tile(np.arange(100),(self.y.shape[0] // 100,)).ravel()
        res = mod.fit(cov_type='one-way',clusters=clusters)
        # print('Clustered')
        # print(res.params)
        # print(res.tstats)

        clusters = np.tile(np.arange(500),(self.y.shape[0] // 500,)).ravel()
        res = mod.fit(cov_type='one-way',clusters=clusters)
        # print('Clustered - mid')
        # print(res.params)
        # print(res.tstats)

        clusters = np.tile(np.arange(1000),(self.y.shape[0] // 1000,)).ravel()
        res = mod.fit(cov_type='one-way',clusters=clusters)
        # print('Clustered - mid')
        # print(res.params)
        # print(res.tstats)

        clusters = np.tile(np.arange(2500),(self.y.shape[0] // 2500,)).ravel()
        res = mod.fit(cov_type='one-way',clusters=clusters)
        # print('Clustered - 2x')
        # print(res.params)
        # print(res.tstats)


        res = mod.fit(cov_type='one-way')
        # print('Clustered - indiv')
        # print(res.params)
        # print(res.tstats)

    def test_ivgmm_smoke(self):
        mod = IVGMM(self.y, self.x, self.z)
        print(mod.fit())

        mod = IVGMM(self.y, self.x, self.z)
        res = mod.fit(iter=100)
        print(res.params)
        print(res.tstats)
