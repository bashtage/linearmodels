from __future__ import print_function, absolute_import, division

from numpy import ceil, where, argsort, r_, unique, zeros, arange, pi, sin, cos
from numpy.linalg import pinv, inv

class IVCovariance(object):
    def __init__(self, x, z, eps, **config):
        self.x = x
        self.z = z
        self.eps = eps
        self.config = self._check_config(**config)
        self._pinvz = None

    def _check_config(self, **config):
        if len(config) == 0:
            return config

        valid_keys = list(self.defaults.keys())
        invalid = []
        for key in config:
            if key not in valid_keys:
                invalid.append(key)
        if invalid:
            keys = ', '.join(config.keys())
            raise ValueError('Unexpected keywords in config: {0}'.format(keys))

        return config

    @property
    def defaults(self):
        return {}

    @property
    def cov(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape

        scale = nobs / (nobs - nvar) if self.config.get('debiased', False) else 1
        self._pinvz = pinvz = pinv(z) if self._pinvz is None else self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        vinv = inv(v)

        return scale * vinv @ self.s @ vinv / nobs


class HomoskedasticCovariance(IVCovariance):
    def __init__(self, x, z, eps, **config):
        super(HomoskedasticCovariance, self).__init__(x, z, eps, **config)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        s2 = eps.T @ eps / nobs
        v = (x.T @ z) @ (pinv(z) @ x) / nobs

        return s2 * v

    @property
    def defaults(self):
        return {'debiased': False}


class KernelCovariance(HomoskedasticCovariance):
    def __init__(self, x, z, eps, **config):
        super(KernelCovariance, self).__init__(x, z, eps, **config)
        self._kernels = {'bartlett': self._weight_bartlett,
                         'newey-west': self._weight_bartlett,
                         'quadratic-spectral': self._weight_quadratic_spectral,
                         'andrews': self._weight_quadratic_spectral,
                         'gallant': self._weight_parzen,
                         'parzen': self._weight_parzen}

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape

        kernel = self.config.get('kernel', 'bartlett')
        # TODO: Bandwidth selection method
        bw = self.config.get('bw', ceil(20 * (nobs / 100) ** (2 / 9)))
        if kernel in ('andrews', 'quadratic-spectral'):
            bw = ceil(20 * ((nobs / 100) ** (2 / 25)))
        elif kernel in ('parzen', 'gallant'):
            bw = ceil(20 * ((nobs / 100) ** (4 / 25)))
        bw = int(bw)
        w = self._kernels[kernel](bw)

        self._pinvz = pinvz = pinv(z) if self._pinvz is None else self._pinvz
        xhat_e = z @ (pinvz @ x) * eps
        s = xhat_e.T @ xhat_e

        for i in range(bw):
            s += 2 * w[i + 1] * xhat_e[i + 1:].T @ xhat_e[:-(i + 1)]
        s /= nobs

        return s

    @property
    def defaults(self):
        """
        Default values

        Returns
        -------
        defaults : dict
            Dictionary containing valid options and their default value

        Notes
        -----
        When ``bw`` is None, automatic bandwidth selection is used.
        """
        return {'bw': None,
                'kernel': 'bartlett',
                'debiased': False}

    @staticmethod
    def _weight_bartlett(max_lag):
        return 1 - arange(max_lag + 1) / (max_lag + 1)

    @staticmethod
    def _weight_quadratic_spectral(max_lag):
        w = 6 * pi * arange(max_lag + 1) / 5
        w[0] = 1
        w[1:] = 3 * (sin(w[1:]) / w[1:] - cos(w[1:])) / w[1:] ** 2
        return w

    @staticmethod
    def _weight_parzen(max_lag):
        z = arange(max_lag + 1) / (max_lag + 1)
        w = 1 - 6 * z ** 2 + 6 * z ** 3
        w[z > 0.5] = 2 * (1 - z[z > 0.5]) ** 3
        return w


class HeteroskedasticCovariance(HomoskedasticCovariance):
    def __init__(self, x, z, eps, **config):
        super(HeteroskedasticCovariance, self).__init__(x, z, eps, **config)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        self._pinvz = pinvz = pinv(z) if self._pinvz is None else self._pinvz
        xhat_e = z @ (pinvz @ x) * eps
        s = xhat_e.T @ xhat_e / nobs
        return s


class OneWayClusteredCovariance(HomoskedasticCovariance):
    def __init__(self, x, z, eps, **config):
        super(OneWayClusteredCovariance, self).__init__(x, z, eps, **config)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        self._pinvz = pinvz = pinv(z) if self._pinvz is None else self._pinvz
        xhat_e = z @ (pinvz @ x) * eps

        nobs, nvar = x.shape
        clusters = self.config.get('clusters', arange(nobs))
        num_clusters = len(unique(clusters))

        clusters = clusters.squeeze()
        if num_clusters > 1:
            sort_args = argsort(clusters)
        else:
            sort_args = list(range(nobs))

        clusters = clusters[sort_args]
        locs = where(r_[True, clusters[:-1] != clusters[1:], True])[0]
        xhat_e = xhat_e[sort_args]

        s = zeros((nvar, nvar))
        for i in range(num_clusters):
            st, en = locs[i], locs[i + 1]
            xhat_e_bar = xhat_e[st:en].sum(axis=0)[:, None]
            s += xhat_e_bar @ xhat_e_bar.T

        s *= num_clusters / (num_clusters - 1) / nobs

        return s

    @property
    def defaults(self):
        return {'debiased': False,
                'clusters': None}
