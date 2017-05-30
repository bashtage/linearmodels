import numpy as np
import pytest

from linearmodels.panel.covariance import (ACCovariance, ClusteredCovariance,
                                           CovarianceManager, DriscollKraay,
                                           HeteroskedasticCovariance,
                                           HomoskedasticCovariance)


class TestCovariance(object):
    @classmethod
    def setup_class(cls):
        cls.n, cls.t, cls.k = 100, 50, 10
        cls.x = np.random.random_sample((cls.n * cls.t, cls.k))
        cls.epsilon = np.random.random_sample((cls.n * cls.t, 1))
        cls.params = np.arange(1, cls.k + 1)[:, None] / cls.k
        cls.df_resid = cls.n * cls.t
        cls.y = cls.x @ cls.params + cls.epsilon
        cls.time_ids = np.tile(np.arange(cls.t)[:, None], (cls.n, 1))
        cls.cluster1 = np.tile(np.arange(cls.n)[:, None], (cls.t, 1))
        cls.entity_ids = cls.cluster1
        cls.cluster2 = np.tile(np.arange(cls.t)[:, None], (cls.n, 1))
        cls.cluster3 = np.random.random_integers(0, 10, (cls.n * cls.t, 1))
        cls.cluster4 = np.random.randint(0, 10, (cls.n * cls.t, 2))
        cls.cluster5 = np.random.randint(0, 10, (cls.n * cls.t, 3))

    def test_heteroskedastic_smoke(self):
        cov = HeteroskedasticCovariance(self.y, self.x, self.params, self.entity_ids,
                                        self.time_ids, extra_df=0).cov
        assert cov.shape == (self.k, self.k)
        cov = HeteroskedasticCovariance(self.y, self.x, self.params, self.entity_ids,
                                        self.time_ids, extra_df=0).cov
        assert cov.shape == (self.k, self.k)

    def test_homoskedastic_smoke(self):
        cov = HomoskedasticCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                      extra_df=0).cov
        assert cov.shape == (self.k, self.k)
        cov = HomoskedasticCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                      extra_df=0).cov
        assert cov.shape == (self.k, self.k)

    def test_clustered_covariance_smoke(self):
        cov = ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                  extra_df=0).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                  extra_df=0,
                                  clusters=self.cluster1).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                  extra_df=0,
                                  clusters=self.cluster2, group_debias=True).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                  extra_df=0,
                                  clusters=self.cluster3).cov
        assert cov.shape == (self.k, self.k)
        cov = ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                  extra_df=0,
                                  clusters=self.cluster3, group_debias=True).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                  extra_df=0,
                                  clusters=self.cluster4).cov
        assert cov.shape == (self.k, self.k)

        cov = ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                  extra_df=0,
                                  clusters=self.cluster4, group_debias=True).cov
        assert cov.shape == (self.k, self.k)

    def test_clustered_covariance_error(self):
        with pytest.raises(ValueError):
            ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                extra_df=0,
                                clusters=self.cluster5)

        with pytest.raises(ValueError):
            ClusteredCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                                extra_df=0,
                                clusters=self.cluster4[::2])

    def test_driscoll_kraay_smoke(self):
        cov = DriscollKraay(self.y, self.x, self.params, self.entity_ids, self.time_ids).cov
        assert cov.shape == (self.k, self.k)
        cov = DriscollKraay(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                            kernel='parzen').cov
        assert cov.shape == (self.k, self.k)
        cov = DriscollKraay(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                            bandwidth=12).cov
        assert cov.shape == (self.k, self.k)

    def test_ac_covariance_smoke(self):
        cov = ACCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids).cov
        assert cov.shape == (self.k, self.k)
        cov = ACCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                           kernel='parzen').cov
        assert cov.shape == (self.k, self.k)
        cov = ACCovariance(self.y, self.x, self.params, self.entity_ids, self.time_ids,
                           bandwidth=12).cov
        assert cov.shape == (self.k, self.k)


def test_covariance_manager():
    cm = CovarianceManager('made-up-class', HomoskedasticCovariance, HeteroskedasticCovariance)
    with pytest.raises(ValueError):
        cm['clustered']

    with pytest.raises(KeyError):
        cm['unknown']

    assert cm['unadjusted'] is HomoskedasticCovariance
    assert cm['homoskedastic'] is HomoskedasticCovariance
    assert cm['robust'] is HeteroskedasticCovariance
    assert cm['heteroskedastic'] is HeteroskedasticCovariance
