"""
Covariance estimators for linear factor models
"""
from numpy.linalg import inv

from linearmodels.iv.covariance import (KERNEL_LOOKUP, _cov_kernel,
                                        kernel_optimal_bandwidth)


class HeteroskedasticCovariance(object):
    """
    Heteroskedasticity robust covariance estimator

    Parameters
    ----------
    xe : ndarray
        Scores/moment conditionas
    jacobian : ndarray, optional
        Jacobian.  One and only one of jacobian and inv_jacobian must
        be provided
    inv_jacobian : ndarray, optional
        Inverse jacobian.  One and only one of jacobian and inv_jacobian must
        be provided
    center : bool, optional
        Falg indicating to center the scores when computing the covariance
    debiased : bool, optional
        Flag indicating to use a debiased estimator
    df : int, optional
        Degree of freedom value ot use if debiasing
    """

    def __init__(self, xe, *, jacobian=None, inv_jacobian=None,
                 center=True, debiased=False, df=0):

        self._xe = xe
        self._jac = jacobian
        self._inv_jac = inv_jacobian
        self._center = center
        if (jacobian is None and inv_jacobian is None) \
                or (jacobian is not None and inv_jacobian is not None):
            raise ValueError('One and only one of jacobian or inv_jacobian must be provided.')
        self._debiased = debiased
        self._df = df
        if jacobian is not None:
            self._square = jacobian.shape[0] == jacobian.shape[1]
        else:
            self._square = inv_jacobian.shape[0] == inv_jacobian.shape[1]

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__() + ', id: {0}'.format(hex(id(self)))

    @property
    def config(self):
        return {'type': self.__class__.__name__}

    @property
    def s(self):
        """
        Score/moment condition covariance

        Returns
        -------
        s : ndarray
            Covariance of the scores or moment conditions
        """
        xe = self._xe
        nobs = xe.shape[0]

        if self._center:
            xe = xe - xe.mean(0)[None, :]
        out = xe.T @ xe / nobs
        return (out + out.T) / 2

    @property
    def jacobian(self):
        """The Jacobian"""
        if self._jac is None:
            self._jac = inv(self._inv_jac)
        return self._jac

    @property
    def inv_jacobian(self):
        """Inverse Jacobian"""
        if self._inv_jac is None:
            self._inv_jac = inv(self._jac)
        return self._inv_jac

    @property
    def square(self):
        """Flag indicating if jacobian is square"""
        return self._square

    @property
    def cov(self):
        """
        Compute parameter covariance

        Returns
        -------
        c : ndarray
            Parameter covariance
        """
        s = self.s
        nobs = self._xe.shape[0]
        scale = 1 / (nobs - int(self._debiased) * self._df)
        if self.square:
            ji = self.inv_jacobian
            out = ji @ s @ ji.T
        else:
            j = self.jacobian
            out = inv(j.T @ inv(s) @ j)
        out = (scale / 2) * (out + out.T)
        return out


class KernelCovariance(HeteroskedasticCovariance):
    """
    Heteroskedasticity-autocorrelation (HAC) robust covariance estimator

    Parameters
    ----------
    xe : ndarray
        Scores/moment conditionas
    jacobian : ndarray, optional
        Jacobian.  One and only one of jacobian and inv_jacobian must
        be provided
    inv_jacobian : ndarray, optional
        Inverse jacobian.  One and only one of jacobian and inv_jacobian must
        be provided
    kernel : str, optional
        Kernel name. See notes for available kernels.
    bandwidth : int, optional
        Non-negative integer bandwidth
    center : bool, optional
        Flag indicating to center the scores when computing the covariance
    debiased : bool, optional
        Flag indicating to use a debiased estimator
    df : int, optional
        Degree of freedom value ot use if debiasing
    """

    def __init__(self, xe, *, jacobian=None, inv_jacobian=None,
                 kernel='bartlett', bandwidth=None, center=True,
                 debiased=False, df=0):
        super(KernelCovariance, self).__init__(xe, jacobian=jacobian,
                                               inv_jacobian=inv_jacobian,
                                               center=center,
                                               debiased=debiased, df=df)
        self._kernel = kernel.lower()
        if self._kernel not in KERNEL_LOOKUP:
            raise ValueError('Unknown kernel')
        self._bandwidth = bandwidth
        if bandwidth is not None:
            if bandwidth < 0:
                raise ValueError('bandwidth must be non-negative.')

    def __str__(self):
        descr = ', Kernel: {0}, Bandwidth: {1}'.format(self._kernel,
                                                       self.bandwidth)
        return self.__class__.__name__ + descr

    @property
    def config(self):
        out = super(KernelCovariance, self).config
        out['kernel'] = self._kernel
        out['bandwidth'] = self.bandwidth
        return out

    @property
    def kernel(self):
        """Kernel used in estimation"""
        return self._kernel

    @property
    def bandwidth(self):
        """Bandwidth used in estimation"""
        if self._bandwidth is None:
            xe = self._xe
            x = xe / xe.std(0)[None, :]
            x = x.sum(1)
            bw = kernel_optimal_bandwidth(x, kernel=self.kernel)
            self._bandwidth = int(bw)

        return self._bandwidth

    @property
    def s(self):
        """
        Score/moment condition covariance

        Returns
        -------
        s : ndarray
            Covariance of the scores or moment conditions
        """
        xe = self._xe
        nobs = xe.shape[0]
        bw = self.bandwidth
        kernel = self._kernel
        kernel = KERNEL_LOOKUP[kernel]
        weights = kernel(bw, nobs - 1)
        out = _cov_kernel(xe, weights)

        return (out + out.T) / 2


class HeteroskedasticWeight(object):
    """
    GMM weighing matrix estimation

    Parameters
    ----------
    moments : ndarray
        Moment conditions (nobs by nmoments)
    center : bool, optional
        Flag indicating to center the moments when computing the weights
    """

    def __init__(self, moments, center=True):
        self._moments = moments
        self._center = center

    def w(self, moments):
        """
        Score/moment condition weighting matrix

        Parameters
        ----------
        moments : ndarray
            Moment conditions (nobs by nmoments)

        Returns
        -------
        w : ndarray
            Weighting matrix computed from moment conditions
        """
        if self._center:
            moments = moments - moments.mean(0)[None, :]
        nobs = moments.shape[0]
        out = moments.T @ moments / nobs

        return inv((out + out.T) / 2.0)


class KernelWeight(HeteroskedasticWeight):
    """
    HAC GMM weighing matrix estimation

    Parameters
    ----------
    moments : ndarray
        Moment conditions (nobs by nmoments)
    center : bool, optional
        Flag indicating to center the moments when computing the weights
    kernel : str, optional
        Kernel name. See notes for available kernels.
    bandwidth : int, optional
        Non-negative integer bandwidth

    """

    def __init__(self, moments, center=True, kernel='bartlett', bandwidth=None):
        super(KernelWeight, self).__init__(moments, center=center)
        self._kernel = kernel.lower()
        if self._kernel not in KERNEL_LOOKUP:
            raise ValueError('Unknown kernel')
        self._bandwidth = bandwidth
        if bandwidth is not None:
            if bandwidth < 0:
                raise ValueError('bandwidth must be non-negative.')

    @property
    def kernel(self):
        """Kernel used in estimation"""
        return self._kernel

    @property
    def bandwidth(self):
        """Bandwidth used in estimation"""
        if self._bandwidth is None:
            moments = self._moments
            m = moments / moments.std(0)[None, :]
            m = m.sum(1)
            bw = kernel_optimal_bandwidth(m, kernel=self.kernel)
            self._bandwidth = int(bw)

        return self._bandwidth

    def w(self, moments):
        """
        Score/moment condition weighting matrix

        Parameters
        ----------
        moments : ndarray
            Moment conditions (nobs by nmoments)

        Returns
        -------
        w : ndarray
            Weighting matrix computed from moment conditions
        """
        if self._center:
            moments = moments - moments.mean(0)[None, :]
        nobs = moments.shape[0]
        bw = self.bandwidth
        kernel = self._kernel
        kernel = KERNEL_LOOKUP[kernel]
        weights = kernel(bw, nobs - 1)
        out = _cov_kernel(moments, weights)

        return inv((out + out.T) / 2.0)
