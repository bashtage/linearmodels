"""
Covariance estimators for linear factor models
"""

from __future__ import annotations

import numpy
from numpy import empty
from numpy.linalg import inv

from linearmodels.iv.covariance import (
    KERNEL_LOOKUP,
    cov_kernel,
    kernel_optimal_bandwidth,
)
import linearmodels.typing.data


class _HACMixin:
    def __init__(self, kernel: str, bandwidth: float | None) -> None:
        self._kernel: str | None = None
        self._bandwidth: float | None = None  # pragma: no cover
        self._moments: numpy.ndarray = empty((0,))  # pragma: no cover
        self._check_kernel(kernel)
        self._check_bandwidth(bandwidth)

    @property
    def kernel(self) -> str:
        """Kernel used in estimation"""
        assert self._kernel is not None
        return self._kernel

    @property
    def bandwidth(self) -> float:
        """Bandwidth used in estimation"""
        if self._bandwidth is None:
            assert self._moments is not None
            moments = self._moments
            m = moments / moments.std(0)[None, :]
            m = m.sum(1)
            bw = kernel_optimal_bandwidth(m, kernel=self.kernel)
            self._bandwidth = bw

        return self._bandwidth

    def _check_kernel(self, kernel: str) -> None:
        if not isinstance(kernel, str):
            raise TypeError("kernel must be the name of a kernel")
        self._kernel = kernel.lower()
        if self._kernel not in KERNEL_LOOKUP:
            raise ValueError("Unknown kernel")

    def _check_bandwidth(self, bandwidth: float | None) -> None:
        self._bandwidth = bandwidth
        if bandwidth is not None:
            try:
                assert bandwidth is not None
                bandwidth = float(bandwidth)
            except (TypeError, ValueError):
                raise TypeError("bandwidth must be either None or a float")
            if bandwidth < 0:
                raise ValueError("bandwidth must be non-negative.")

    def _kernel_cov(
        self, z: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        nobs = z.shape[0]
        bw = self.bandwidth
        kernel = self._kernel
        assert kernel is not None
        kernel_estimator = KERNEL_LOOKUP[kernel]
        weights = kernel_estimator(bw, nobs - 1)
        out = cov_kernel(z, weights)
        return (out + out.T) / 2


class HeteroskedasticCovariance:
    """
    Heteroskedasticity robust covariance estimator

    Parameters
    ----------
    xe : ndarray
        Scores/moment conditions
    jacobian : ndarray
        Jacobian.  One and only one of jacobian and inv_jacobian must
        be provided
    inv_jacobian : {ndarray, None}
        Inverse jacobian.  One and only one of jacobian and inv_jacobian must
        be provided. Default is None.
    center : bool
        Flag indicating to center the scores when computing the covariance. Default
        is true.
    debiased : bool
        Flag indicating to use a debiased estimator. Default is False.
    df : int
        Degree of freedom value ot use if debiasing. Default is 0.
    """

    def __init__(
        self,
        xe: linearmodels.typing.data.Float64Array,
        *,
        jacobian: numpy.ndarray | None = None,
        inv_jacobian: numpy.ndarray | None = None,
        center: bool = True,
        debiased: bool = False,
        df: int = 0,
    ) -> None:
        self._moments = self._xe = xe
        self._jac = jacobian
        self._inv_jac = inv_jacobian
        self._center = center
        if (jacobian is not None) == (inv_jacobian is not None):
            raise ValueError(
                "One and only one of jacobian or inv_jacobian must be provided."
            )
        self._debiased = debiased
        self._df = df
        if jacobian is not None:
            self._square = jacobian.shape[0] == jacobian.shape[1]
        else:
            assert inv_jacobian is not None
            self._square = inv_jacobian.shape[0] == inv_jacobian.shape[1]

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__() + f", id: {hex(id(self))}"

    @property
    def config(self) -> dict[str, str | float]:
        return {"type": self.__class__.__name__}

    @property
    def s(self) -> linearmodels.typing.data.Float64Array:
        """
        Score/moment condition covariance

        Returns
        -------
        ndarray
            Covariance of the scores or moment conditions
        """
        xe = self._xe
        nobs = xe.shape[0]

        if self._center:
            xe = xe - xe.mean(0)[None, :]
        out = xe.T @ xe / nobs
        return (out + out.T) / 2

    @property
    def jacobian(self) -> linearmodels.typing.data.Float64Array:
        """The Jacobian"""
        if self._jac is None:
            assert self._inv_jac is not None
            self._jac = inv(self._inv_jac)
        assert self._jac is not None
        return self._jac

    @property
    def inv_jacobian(self) -> linearmodels.typing.data.Float64Array:
        """Inverse Jacobian"""
        if self._inv_jac is None:
            assert self._jac is not None
            self._inv_jac = inv(self._jac)
        assert self._inv_jac is not None
        return self._inv_jac

    @property
    def square(self) -> bool:
        """Flag indicating if jacobian is square"""
        return self._square

    @property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """
        Compute parameter covariance

        Returns
        -------
        ndarray
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


class KernelCovariance(HeteroskedasticCovariance, _HACMixin):
    """
    Heteroskedasticity-autocorrelation (HAC) robust covariance estimator

    Parameters
    ----------
    xe : ndarray
        The scores (moment) conditions.
    jacobian : ndarray
        Jacobian.  One and only one of jacobian and inv_jacobian must
        be provided.
    inv_jacobian : ndarray
        Inverse jacobian.  One and only one of jacobian and inv_jacobian must
        be provided.
    kernel : str
        Kernel name. See notes for available kernels. The default is "bartlett".
    bandwidth : int
        Non-negative integer bandwidth. If None, the optimal bandwidth is
        estimated.
    center : bool
        Flag indicating to center the scores when computing the covariance.
    debiased : bool
        Flag indicating to use a debiased estimator.
    df : int
        Degree of freedom value ot use if debiasing.

    See Also
    --------
    linearmodels.iv.covariance.kernel_weight_bartlett,
    linearmodels.iv.covariance.kernel_weight_parzen,
    linearmodels.iv.covariance.kernel_weight_quadratic_spectral
    """

    def __init__(
        self,
        xe: linearmodels.typing.data.Float64Array,
        *,
        jacobian: numpy.ndarray | None = None,
        inv_jacobian: numpy.ndarray | None = None,
        kernel: str | None = None,
        bandwidth: float | None = None,
        center: bool = True,
        debiased: bool = False,
        df: int = 0,
    ) -> None:
        kernel = "bartlett" if kernel is None else kernel
        _HACMixin.__init__(self, kernel, bandwidth)
        super().__init__(
            xe,
            jacobian=jacobian,
            inv_jacobian=inv_jacobian,
            center=center,
            debiased=debiased,
            df=df,
        )

    def __str__(self) -> str:
        descr = f", Kernel: {self._kernel}, Bandwidth: {self.bandwidth}"
        return self.__class__.__name__ + descr

    @property
    def config(self) -> dict[str, str | float]:
        out = super().config
        out["kernel"] = self.kernel
        out["bandwidth"] = self.bandwidth
        return out

    @property
    def s(self) -> linearmodels.typing.data.Float64Array:
        """
        Score/moment condition covariance

        Returns
        -------
        ndarray
            Covariance of the scores or moment conditions
        """
        xe = self._xe
        out = self._kernel_cov(xe)

        return (out + out.T) / 2


class HeteroskedasticWeight:
    """
    GMM weighing matrix estimation

    Parameters
    ----------
    moments : ndarray
        Moment conditions (nobs by nmoments)
    center : bool
        Flag indicating to center the moments when computing the weights
    """

    def __init__(
        self, moments: linearmodels.typing.data.Float64Array, center: bool = True
    ) -> None:
        self._moments = moments
        self._center = center

    def w(
        self, moments: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """
        Score/moment condition weighting matrix

        Parameters
        ----------
        moments : ndarray
            Moment conditions (nobs by nmoments)

        Returns
        -------
        ndarray
            Weighting matrix computed from moment conditions
        """
        if self._center:
            moments = moments - moments.mean(0)[None, :]
        nobs = moments.shape[0]
        out = moments.T @ moments / nobs

        return inv((out + out.T) / 2.0)


class KernelWeight(HeteroskedasticWeight, _HACMixin):
    """
    HAC GMM weighing matrix estimation

    Parameters
    ----------
    moments : ndarray
        Moment conditions (nobs by nmoments)
    center : bool
        Flag indicating to center the moments when computing the weights
    kernel : str
        Kernel name. See notes for available kernels. If None, the kernel
        is set to "bartlett".
    bandwidth : int.
        Non-negative integer bandwidth. If None, the optimal bandwidth is
        estimated.
    """

    def __init__(
        self,
        moments: linearmodels.typing.data.Float64Array,
        center: bool = True,
        kernel: str | None = None,
        bandwidth: float | None = None,
    ):
        kernel = "bartlett" if kernel is None else kernel
        _HACMixin.__init__(self, kernel, bandwidth)
        super().__init__(moments, center=center)

    def w(
        self, moments: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """
        Score/moment condition weighting matrix

        Parameters
        ----------
        moments : ndarray
            Moment conditions (nobs by nmoments)

        Returns
        -------
        ndarray
            Weighting matrix computed from moment conditions
        """
        if self._center:
            moments = moments - moments.mean(0)[None, :]
        out = self._kernel_cov(moments)

        return inv(out)
