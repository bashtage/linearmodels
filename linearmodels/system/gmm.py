"""
Covariance and weight estimation for GMM IV estimators
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy
from numpy import array, empty, ndarray, repeat, sqrt, zeros_like

from linearmodels.asset_pricing.covariance import _HACMixin
from linearmodels.iv.covariance import kernel_optimal_bandwidth
from linearmodels.shared.utility import AttrDict
from linearmodels.system._utility import blocked_inner_prod
import linearmodels.typing.data


class HomoskedasticWeightMatrix:
    r"""
    Homoskedastic (unadjusted) weight estimation

    Parameters
    ----------
    center : bool
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix.
    debiased : bool
        Flag indicating whether to use small-sample adjustments

    Notes
    -----
    The weight matrix estimator is

    .. math::

      Z'(\Sigma \otimes I_N)Z

    where :math:`Z` is a block diagonal matrix containing both the exogenous
    regressors and instruments and :math:`\Sigma` is the covariance of the
    model residuals.

    ``center`` has no effect on this estimator since it is always centered.
    """

    def __init__(self, center: bool = False, debiased: bool = False) -> None:
        self._center = center
        self._debiased = debiased
        self._bandwidth: float | None = 0
        self._name = "Homoskedastic (Unadjusted) Weighting"
        self._config = AttrDict(center=center, debiased=debiased)

    def __str__(self) -> str:
        out = self._name
        extra = []
        for key in self._str_extra:
            extra.append(": ".join([str(key), str(self._str_extra[key])]))
        if extra:
            out += " (" + ", ".join(extra) + ")"
        return out

    def __repr__(self) -> str:
        return self.__str__() + f", id: {hex(id(self))}"

    @property
    def _str_extra(self) -> AttrDict:
        return AttrDict(Debiased=self._debiased, Center=self._center)

    def sigma(
        self,
        eps: linearmodels.typing.data.Float64Array,
        x: Sequence[linearmodels.typing.data.Float64Array],
    ) -> linearmodels.typing.data.Float64Array:
        """
        Estimate residual covariance.

        Parameters
        ----------
        eps : ndarray
            The residuals from the system of equations.
        x : list[ndarray]
            A list of the regressor matrices for each equation in the system.

        Returns
        -------
        ndarray
            The estimated covariance matrix of the residuals.
        """
        nobs = eps.shape[0]
        eps = eps - eps.mean(0)
        sigma = eps.T @ eps / nobs
        scale = 1.0
        if self._debiased:
            k = array([a.shape[1] for a in x])[:, None]
            k = sqrt(k)
            scale = nobs / (nobs - k @ k.T)
        sigma *= scale

        return sigma

    def weight_matrix(
        self,
        x: Sequence[linearmodels.typing.data.Float64Array],
        z: Sequence[linearmodels.typing.data.Float64Array],
        eps: linearmodels.typing.data.Float64Array,
        *,
        sigma: numpy.ndarray,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Construct a GMM weight matrix for a model.

        Parameters
        ----------
        x : list[ndarray]
            List of containing model regressors for each equation in the system
        z : list[ndarray]
            List of containing instruments for each equation in the system
        eps : ndarray
            Model errors (nobs by neqn)
        sigma : ndarray
            Fixed covariance of model errors. If None, estimated from eps.

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs = z[0].shape[0]
        w = cast(ndarray, blocked_inner_prod(z, sigma) / nobs)
        return w

    @property
    def config(self) -> AttrDict:
        """
        Weight estimator configuration

        Returns
        -------
        AttrDict
            Dictionary containing weight estimator configuration information
        """
        return self._config


class HeteroskedasticWeightMatrix(HomoskedasticWeightMatrix):
    r"""
    Heteroskedasticity robust weight estimation

    Parameters
    ----------
    center : bool
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix.
    debiased : bool
        Flag indicating whether to use small-sample adjustments

    Notes
    -----
    The weight matrix estimator is

    .. math::

      W   & = n^{-1}\sum_{i=1}^{n}g'_ig_i \\
      g_i & = (z_{1i}\epsilon_{1i},z_{2i}\epsilon_{2i},\ldots,z_{ki}\epsilon_{ki})

    where :math:`g_i` is the vector of scores across all equations for
    observation i.  :math:`z_{ji}` is the vector of instruments for equation
    j and :math:`\epsilon_{ji}` is the error for equation j for observation
    i.  This form allows for heteroskedasticity and arbitrary cross-sectional
    dependence between the moment conditions.
    """

    def __init__(self, center: bool = False, debiased: bool = False) -> None:
        super().__init__(center, debiased)
        self._name = "Heteroskedastic (Robust) Weighting"

    def weight_matrix(
        self,
        x: Sequence[linearmodels.typing.data.Float64Array],
        z: Sequence[linearmodels.typing.data.Float64Array],
        eps: linearmodels.typing.data.Float64Array,
        *,
        sigma: numpy.ndarray | None = None,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Construct a GMM weight matrix for a model.

        Parameters
        ----------
        x : list[ndarray]
            Model regressors (exog and endog), (nobs by nvar)
        z : list[ndarray]
            Model instruments (exog and instruments), (nobs by ninstr)
        eps : ndarray
            Model errors (nobs by 1)
        sigma : ndarray
            Fixed covariance of model errors. If None, estimated from eps.

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs = x[0].shape[0]
        k = len(x)
        k_total = sum(map(lambda a: a.shape[1], z))
        ze = empty((nobs, k_total))
        loc = 0
        for i in range(k):
            e = eps[:, [i]]
            zk = z[i].shape[1]
            ze[:, loc : loc + zk] = z[i] * e
            loc += zk
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu
        w = ze.T @ ze / nobs
        scale = self._debias_scale(nobs, x, z)
        w *= scale

        return w

    def _debias_scale(
        self,
        nobs: int,
        x: Sequence[linearmodels.typing.data.Float64Array],
        z: Sequence[linearmodels.typing.data.Float64Array],
    ) -> linearmodels.typing.data.Float64Array:
        nvar = array([a.shape[1] for a in x])
        ninstr = array([a.shape[1] for a in z])
        nvar = repeat(nvar, ninstr)
        if not self._debiased:
            nvar = zeros_like(nvar)
        nvar = cast(linearmodels.typing.data.Float64Array, sqrt(nvar))[:, None]
        scale = nobs / (nobs - nvar @ nvar.T)
        return scale


class KernelWeightMatrix(HeteroskedasticWeightMatrix, _HACMixin):
    r"""
    Heteroskedasticity robust weight estimation

    Parameters
    ----------
    center : bool
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix.
    debiased : bool
        Flag indicating whether to use small-sample adjustments
    kernel : str
        Name of kernel to use.  Supported kernels include:

        * "bartlett", "newey-west" : Bartlett's kernel
        * "parzen", "gallant" : Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" : Quadratic spectral kernel

    bandwidth : float
        Bandwidth to use for the kernel.  If not provided the optimal
        bandwidth will be estimated.
    optimal_bw : bool
        Flag indicating whether to estimate the optimal bandwidth, when
        bandwidth is None.  If False, nobs - 2 is used


    Notes
    -----
    The weight matrix estimator is

    .. math::

      W & = \hat{\Gamma}_0+\sum_{i=1}^{n-1} w_i (\hat{\Gamma}_i+\hat{\Gamma}_i^\prime)
      \hat{\Gamma}_j & = n^{-1}\sum_{i=1}^{n-j} g'_ig_{i+j} \\
      g_i & = (z_{1i}\epsilon_{1i},z_{2i}\epsilon_{2i},\ldots,z_{ki}\epsilon_{ki})

    where :math:`g_i` is the vector of scores across all equations for
    observation i and :math:`w_j` are the kernel weights which depend on the
    selected kernel and bandwidth.  :math:`z_{ji}` is the vector of instruments
    for equation j and :math:`\epsilon_{ji}` is the error for equation j for
    observation i.  This form allows for heteroskedasticity and autocorrelation
    between the moment conditions.
    """

    def __init__(
        self,
        center: bool = False,
        debiased: bool = False,
        kernel: str = "bartlett",
        bandwidth: float | None = None,
        optimal_bw: bool = False,
    ) -> None:
        _HACMixin.__init__(self, kernel, bandwidth)
        super().__init__(center, debiased)
        self._name = "Kernel (HAC) Weighting"
        self._check_kernel(kernel)
        self._check_bandwidth(bandwidth)
        self._predefined_bw = self._bandwidth
        self._optimal_bw = optimal_bw

    def weight_matrix(
        self,
        x: Sequence[linearmodels.typing.data.Float64Array],
        z: Sequence[linearmodels.typing.data.Float64Array],
        eps: linearmodels.typing.data.Float64Array,
        *,
        sigma: numpy.ndarray | None = None,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Construct a GMM weight matrix for a model.

        Parameters
        ----------
        x : list[ndarray]
            Model regressors (exog and endog)
        z : list[ndarray]
            Model instruments (exog and instruments)
        eps : ndarray
            Model errors (nobs by nequation)
        sigma : ndarray
            Fixed covariance of model errors. If None, estimated from eps.

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs = x[0].shape[0]
        k = len(x)
        k_total = sum(map(lambda a: a.shape[1], z))
        ze = empty((nobs, k_total))
        loc = 0
        for i in range(k):
            e = eps[:, [i]]
            zk = z[i].shape[1]
            ze[:, loc : loc + zk] = z[i] * e
            loc += zk
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu
        self._optimal_bandwidth(ze)
        w = self._kernel_cov(ze)
        scale = self._debias_scale(nobs, x, z)
        w *= scale

        return w

    def _optimal_bandwidth(
        self, moments: linearmodels.typing.data.Float64Array
    ) -> float:
        """Compute optimal bandwidth used in estimation if needed"""
        if self._predefined_bw is not None:
            return self._predefined_bw
        elif not self._optimal_bw:
            self._bandwidth = moments.shape[0] - 2
        else:
            m = moments / moments.std(0)[None, :]
            m = m.sum(1)
            self._bandwidth = kernel_optimal_bandwidth(m, kernel=self.kernel)
        assert self._bandwidth is not None
        return self._bandwidth

    @property
    def bandwidth(self) -> float:
        """Bandwidth used to estimate covariance of moment conditions"""
        assert self._bandwidth is not None
        return self._bandwidth

    @property
    def config(self) -> AttrDict:
        """
        Weight estimator configuration

        Returns
        -------
        AttrDict
            Dictionary containing weight estimator configuration information
        """
        out = AttrDict([(k, v) for k, v in self._config.items()])
        out["bandwidth"] = self.bandwidth
        return out
