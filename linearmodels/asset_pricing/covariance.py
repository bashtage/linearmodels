"""
Covariance estimators for linear factor models.
"""

from __future__ import annotations

from numpy import empty, errstate, ndarray, zeros_like
from numpy.linalg import inv

from linearmodels.iv.covariance import (
    KERNEL_LOOKUP,
    cov_kernel,
    kernel_optimal_bandwidth,
)
from linearmodels.shared.exceptions import unknown_option_error
import linearmodels.typing.data


class _HACMixin:
    def __init__(self, kernel: str, bandwidth: float | None) -> None:
        self._kernel: str | None = None
        self._bandwidth: float | None = None  # pragma: no cover
        self._moments: ndarray = empty((0,))  # pragma: no cover
        self._check_kernel(kernel)
        self._check_bandwidth(bandwidth)

    @property
    def kernel(self) -> str:
        """Kernel used in estimation."""
        assert self._kernel is not None
        return self._kernel

    @property
    def bandwidth(self) -> float:
        """Bandwidth used in estimation."""
        if self._bandwidth is None:
            assert self._moments is not None
            moments = self._moments
            # A moment that does not vary has a standard deviation of 0, and the
            # series is NaN. kernel_optimal_bandwidth raises, so do not warn
            with errstate(divide="ignore", invalid="ignore"):
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
            raise unknown_option_error("kernel", kernel, KERNEL_LOOKUP)

    def _check_bandwidth(self, bandwidth: float | None) -> None:
        self._bandwidth = bandwidth
        if bandwidth is not None:
            try:
                assert bandwidth is not None
                bandwidth = float(bandwidth)
            except (TypeError, ValueError) as exc:
                raise TypeError("bandwidth must be either None or a float") from exc
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
    Heteroskedasticity robust covariance estimator.

    Parameters
    ----------
    xe : ndarray
        Scores/moment conditions, with shape (nobs, nmoments).
    jacobian : ndarray, optional
        Jacobian, with shape (nmoments, nparams). One and only one of jacobian and
        inv_jacobian must be provided.
    inv_jacobian : ndarray, optional
        Inverse jacobian. One and only one of jacobian and inv_jacobian must
        be provided.
    center : bool, optional
        Flag indicating to center the scores when computing the covariance. The
        default is True.
    debiased : bool, optional
        Flag indicating to use a debiased estimator. The default is False.
    df : int, optional
        Degree of freedom value to use if debiasing. The default is 0.

    Raises
    ------
    ValueError
        If both or neither of ``jacobian`` and ``inv_jacobian`` are provided.
    """

    def __init__(
        self,
        xe: linearmodels.typing.data.Float64Array,
        *,
        jacobian: ndarray | None = None,
        inv_jacobian: ndarray | None = None,
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
        """Configuration of the covariance estimator."""
        return {"type": self.__class__.__name__}

    @property
    def s(self) -> linearmodels.typing.data.Float64Array:
        """
        Score/moment condition covariance.

        Returns
        -------
        ndarray
            Covariance of the scores or moment conditions.
        """
        xe = self._xe
        nobs = xe.shape[0]

        if self._center:
            xe = xe - xe.mean(0)[None, :]
        out = xe.T @ xe / nobs
        return (out + out.T) / 2

    @property
    def jacobian(self) -> linearmodels.typing.data.Float64Array:
        """The Jacobian."""
        if self._jac is None:
            assert self._inv_jac is not None
            self._jac = inv(self._inv_jac)
        assert self._jac is not None
        return self._jac

    @property
    def inv_jacobian(self) -> linearmodels.typing.data.Float64Array:
        """Inverse Jacobian."""
        if self._inv_jac is None:
            assert self._jac is not None
            self._inv_jac = inv(self._jac)
        assert self._inv_jac is not None
        return self._inv_jac

    @property
    def square(self) -> bool:
        """Flag indicating if jacobian is square."""
        return self._square

    @property
    def _scale(self) -> float:
        """The inverse of the number of observations, less df if debiased"""
        nobs = self._xe.shape[0]
        return 1 / (nobs - int(self._debiased) * self._df)

    @property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """
        Compute parameter covariance.

        Returns
        -------
        ndarray
            Parameter covariance.
        """
        s = self.s
        scale = self._scale
        if self.square:
            ji = self.inv_jacobian
            out = ji @ s @ ji.T
        else:
            j = self.jacobian
            out = inv(j.T @ inv(s) @ j).astype(float, copy=False)
        out = (scale / 2) * (out + out.T)
        return out

    @property
    def _moment_cov(self) -> linearmodels.typing.data.Float64Array:
        r"""
        Covariance of the average moment conditions at the estimates.

        Returns
        -------
        ndarray
            Covariance of the average of the moment conditions, evaluated at
            the estimated parameters.

        Notes
        -----
        The moment conditions are not zero at the estimates of an over-identified
        model. If :math:`S` is the covariance of the moment conditions and
        :math:`G` is the Jacobian, the efficient GMM estimates have

        .. math::

           (S - G (G^{\prime} S^{-1} G)^{-1} G^{\prime}) / (T - df)

        as the covariance of the average moment conditions, where :math:`df` is
        zero unless the covariance is debiased. This uses the same :math:`S`
        and the same scale as :attr:`cov`, which is
        :math:`(G^{\prime} S^{-1} G)^{-1} / (T - df)`. The covariance is
        singular, with rank equal to the number of moment conditions less the
        number of parameters. It is zero if the Jacobian is square, since the
        moment conditions are then zero at the estimates.

        This is used for the standard errors of the alphas of
        ``LinearFactorModelGMM``. It is private, and not part of the API.
        """
        s = self.s
        if self.square:
            return zeros_like(s)
        j = self.jacobian
        out = self._scale * s - j @ self.cov @ j.T
        return (out + out.T) / 2


class KernelCovariance(HeteroskedasticCovariance, _HACMixin):
    """
    Heteroskedasticity-autocorrelation (HAC) robust covariance estimator.

    Parameters
    ----------
    xe : ndarray
        The scores (moment) conditions, with shape (nobs, nmoments).
    jacobian : ndarray, optional
        Jacobian, with shape (nmoments, nparams). One and only one of jacobian and
        inv_jacobian must be provided.
    inv_jacobian : ndarray, optional
        Inverse jacobian. One and only one of jacobian and inv_jacobian must
        be provided.
    kernel : str, optional
        Name of the kernel. The default is "bartlett". The names are not case
        sensitive. Supported kernels are:

        * "bartlett", "newey-west" - Bartlett's kernel
        * "parzen", "gallant" - Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

    bandwidth : float, optional
        Non-negative bandwidth. If None, the optimal bandwidth is
        estimated. It needs at least 4 observations with the Bartlett kernel, 5
        with the Parzen kernel and 6 with the quadratic spectral kernel, and a
        ``ValueError`` is raised for fewer when the bandwidth is first used. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.
    center : bool, optional
        Flag indicating to center the scores when computing the covariance. The
        default is True.
    debiased : bool, optional
        Flag indicating to use a debiased estimator. The default is False.
    df : int, optional
        Degree of freedom value to use if debiasing. The default is 0.

    Raises
    ------
    ValueError
        If both or neither of ``jacobian`` and ``inv_jacobian`` are provided, or
        if ``bandwidth`` is negative.
    TypeError
        If ``kernel`` is not a str, or ``bandwidth`` is not a number.
    UnknownOptionError
        If ``kernel`` is not supported. It is both a ``KeyError`` and a
        ``ValueError``.

    See Also
    --------
    linearmodels.iv.covariance.kernel_weight_bartlett
        Weights of the Bartlett kernel.
    linearmodels.iv.covariance.kernel_weight_parzen
        Weights of the Parzen kernel.
    linearmodels.iv.covariance.kernel_weight_quadratic_spectral
        Weights of the quadratic spectral kernel.

    Notes
    -----
    .. versionchanged:: 7.1

       The optimal bandwidth raises a ``ValueError`` if there are fewer
       observations than the kernel needs, and if it is not finite. Earlier
       versions raised "cannot convert float infinity to integer" (an
       ``OverflowError``) or "cannot convert float NaN to integer" for some
       data, and used a bandwidth of nobs - 1, or another bandwidth, for other
       data.
    """

    def __init__(
        self,
        xe: linearmodels.typing.data.Float64Array,
        *,
        jacobian: ndarray | None = None,
        inv_jacobian: ndarray | None = None,
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
        """Configuration of the covariance estimator."""
        out = super().config
        out["kernel"] = self.kernel
        out["bandwidth"] = self.bandwidth
        return out

    @property
    def s(self) -> linearmodels.typing.data.Float64Array:
        """
        Score/moment condition covariance.

        Returns
        -------
        ndarray
            Covariance of the scores or moment conditions.
        """
        xe = self._xe
        out = self._kernel_cov(xe)

        return (out + out.T) / 2


class HeteroskedasticWeight:
    """
    GMM weighing matrix estimation.

    Parameters
    ----------
    moments : ndarray
        Moment conditions (nobs by nmoments).
    center : bool, optional
        Flag indicating to center the moments when computing the weights. The
        default is True.
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
        Score/moment condition weighting matrix.

        Parameters
        ----------
        moments : ndarray
            Moment conditions (nobs by nmoments).

        Returns
        -------
        ndarray
            Weighting matrix computed from moment conditions.
        """
        if self._center:
            moments = moments - moments.mean(0)[None, :]
        nobs = moments.shape[0]
        out = moments.T @ moments / nobs

        return inv((out + out.T) / 2.0)


class KernelWeight(HeteroskedasticWeight, _HACMixin):
    """
    HAC GMM weighing matrix estimation.

    Parameters
    ----------
    moments : ndarray
        Moment conditions (nobs by nmoments).
    center : bool, optional
        Flag indicating to center the moments when computing the weights. The
        default is True.
    kernel : str, optional
        Name of the kernel. The default is "bartlett". The names are not case
        sensitive. Supported kernels are:

        * "bartlett", "newey-west" - Bartlett's kernel
        * "parzen", "gallant" - Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

    bandwidth : float, optional
        Non-negative bandwidth. If None, the optimal bandwidth is
        estimated. It needs at least 4 observations with the Bartlett kernel, 5
        with the Parzen kernel and 6 with the quadratic spectral kernel, and a
        ``ValueError`` is raised for fewer when the bandwidth is first used. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.

    Notes
    -----
    .. versionchanged:: 7.1

       The optimal bandwidth raises a ``ValueError`` if there are fewer
       observations than the kernel needs, and if it is not finite. Earlier
       versions raised "cannot convert float infinity to integer" (an
       ``OverflowError``) or "cannot convert float NaN to integer" for some
       data, and used a bandwidth of nobs - 1, or another bandwidth, for other
       data.
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
        Score/moment condition weighting matrix.

        Parameters
        ----------
        moments : ndarray
            Moment conditions (nobs by nmoments).

        Returns
        -------
        ndarray
            Weighting matrix computed from moment conditions.
        """
        if self._center:
            moments = moments - moments.mean(0)[None, :]
        out = self._kernel_cov(moments)

        return inv(out).astype(float, copy=False)
