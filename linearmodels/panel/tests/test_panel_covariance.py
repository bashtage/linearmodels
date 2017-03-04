import numpy as np

from linearmodels.panel.covariance import (oneway_clustered_covariance,
                                           heteroskedastic_covariance, homoskedastic_covariance)


class TestCovariance(object):
    @classmethod
    def setup_class(cls):
        cls.n, cls.t, cls.k = 1000, 4, 10
        cls.x = np.random.random_sample((cls.n * cls.t, cls.k))
        cls.epsilon = np.random.random_sample((cls.n * cls.t, 1))
        cls.cluster1 = np.tile(np.arange(1000)[:, None], (4, 1))
        cls.cluster2 = np.tile(np.arange(4)[:, None], (1000, 1))
        cls.cluster3 = np.random.random_integers(0, 10, (cls.n * cls.t, 1))

    def test_clustered_covariance(self):
        oneway_clustered_covariance(self.x, self.epsilon, self.cluster1)
        oneway_clustered_covariance(self.x, self.epsilon, self.cluster2)
        oneway_clustered_covariance(self.x, self.epsilon, self.cluster3)

    def test_heteroskedastic(self):
        heteroskedastic_covariance(self.x, self.epsilon)
        heteroskedastic_covariance(self.x, self.epsilon, debiased=True)

    def test_homoskedastic(self):
        homoskedastic_covariance(self.x, self.epsilon)
        homoskedastic_covariance(self.x, self.epsilon, debiased=True)
