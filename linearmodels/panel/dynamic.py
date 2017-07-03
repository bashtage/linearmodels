"""
Dynamic models for Panel Data

Module provides

* Blundell-Bond/Arrelano-Bover Estimator
* Arellano-Bond estimator as a special case
* PanelAR as a special case
"""
import numpy as np
import scipy.linalg

from linearmodels.panel.data import PanelData


class PanelAR(object):
    def __init__(self, dependent, lags):
        self._dependent = PanelData(dependent)
        try:
            lags = int(lags)
            self._max_lag = lags
            self._lags = list(range(1, self._max_lag + 1))
        except TypeError:
            self._lags = list(lags)
            self._max_lag = max(self._lags)

        t = self._dependent.shape[1]
        if self._max_lag != 1:
            raise ValueError('One one lags supported')
        if t < 3:
            raise ValueError('dependent must have at least 3 time-series '
                             'observations')

    def _generate_instruments(self):
        t, n = self._dependent.shape[1:]
        z = np.zeros((n, t - 2, (t - 2) * (t - 1) // 2))
        y = self._dependent.panel.values.squeeze()
        loc = 0
        for i in range(t - 2):
            z[:, i, loc:loc + i + 1] = y[:(i + 1)].T
            loc += (i + 1)
        y = np.diff(y, 1, 0)
        x = y[:-1]
        y = y[1:]
        return x, y, z

    def fit(self):
        x, y, z = self._generate_instruments()
        t, n = self._dependent.shape[1:]
        t2 = (t - 2) * (t - 1) // 2
        qxz = np.zeros((1, (t - 2) * (t - 1) // 2))
        qzy = np.zeros((t2, 1))
        if t == 3:
            h = 1
        else:
            h = scipy.linalg.toeplitz([1.0, -0.5] + [0] * (t - 4))
        a = np.zeros((t2, t2))
        for i in range(n):
            qxz += x[:, i:i + 1].T @ z[i]
            qzy += z[i].T @ y[:, i:i + 1]
            a += z[i].T @ h @ z[i]
        w = qxz @ a @ qxz.T
        v = qxz @ a @ qzy
        params = np.linalg.solve(w, v).ravel()
        return params
