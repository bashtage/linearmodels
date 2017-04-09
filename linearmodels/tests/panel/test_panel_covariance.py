import numpy as np
import pytest
from numpy.testing import assert_allclose

from linearmodels.panel.covariance import ClusteredCovariance, HeteroskedasticCovariance, HomoskedasticCovariance


class TestCovariance(object):
    @classmethod
    def setup_class(cls):
        cls.n, cls.t, cls.k = 1000, 4, 10
        cls.x = np.random.random_sample((cls.n * cls.t, cls.k))
        cls.epsilon = np.random.random_sample((cls.n * cls.t, 1))
        cls.params = np.arange(1, cls.k + 1)[:, None] / cls.k
        cls.df_resid = cls.n * cls.t
        cls.y = cls.x @ cls.params + cls.epsilon
        cls.cluster1 = np.tile(np.arange(1000)[:, None], (4, 1))
        cls.cluster2 = np.tile(np.arange(4)[:, None], (1000, 1))
        cls.cluster3 = np.random.random_integers(0, 10, (cls.n * cls.t, 1))
        cls.cluster4 = np.random.randint(0, 10, (cls.n * cls.t, 2))
        cls.cluster5 = np.random.randint(0, 10, (cls.n * cls.t, 3))

    def test_heteroskedastic_smoke(self):
        cov = HeteroskedasticCovariance(self.y, self.x, self.params, extra_df=0).cov
        assert cov.shape == (self.k, self.k)
        cov = HeteroskedasticCovariance(self.y, self.x, self.params, extra_df=0).cov
        assert cov.shape == (self.k, self.k)

    def test_homoskedastic_smoke(self):
        cov = HomoskedasticCovariance(self.y, self.x, self.params, extra_df=0).cov
        assert cov.shape == (self.k, self.k)
        cov = HomoskedasticCovariance(self.y, self.x, self.params, extra_df=0).cov
        assert cov.shape == (self.k, self.k)

    def test_clustered_covariance_smoke(self):
        cov = ClusteredCovariance(self.y, self.x, self.params, extra_df=0).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, extra_df=0,
                                  clusters=self.cluster1).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, extra_df=0,
                                  clusters=self.cluster2, group_debias=True).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, extra_df=0,
                                  clusters=self.cluster3).cov
        assert cov.shape == (self.k, self.k)
        cov = ClusteredCovariance(self.y, self.x, self.params, extra_df=0,
                                  clusters=self.cluster3, group_debias=True).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, extra_df=0,
                                  clusters=self.cluster4).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, extra_df=0,
                                  clusters=self.cluster4, group_debias=True).cov
        assert cov.shape == (self.k, self.k)

        with pytest.raises(ValueError):
            ClusteredCovariance(self.y, self.x, self.params, extra_df=0,
                                clusters=self.cluster5)
