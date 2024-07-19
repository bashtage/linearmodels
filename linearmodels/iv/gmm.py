"""
Covariance and weight estimation for GMM IV estimators
"""

from __future__ import annotations

from typing import Any

import numpy
from numpy import asarray, unique
from numpy.linalg import inv

from linearmodels.iv.covariance import (
    KERNEL_LOOKUP,
    HomoskedasticCovariance,
    kernel_optimal_bandwidth,
)
from linearmodels.shared.covariance import cov_cluster, cov_kernel
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

      s^{2} & =n^{-1}\sum_{i=1}^{n}(\epsilon_i-\bar{\epsilon})^2 \\
      W & =n^{-1}s^{2}\sum_{i=1}^{n}z_i'z_i

    where :math:`z_i` contains both the exogenous regressors and instruments.

    ``center`` has no effect on this estimator since it is always centered.
    """

    def __init__(self, center: bool = False, debiased: bool = False) -> None:
        self._center = center
        self._debiased = debiased
        self._bandwidth: int | None = 0

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar)
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr)
        eps : ndarray
            Model errors (nobs by 1)

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape
        mu = eps.mean(0)
        s2 = (eps - mu).T @ (eps - mu) / nobs
        w = s2 * z.T @ z / nobs
        w *= 1 if not self._debiased else nobs / (nobs - nvar)
        return w

    @property
    def config(self) -> dict[str, str | bool | numpy.ndarray | int | None]:
        """
        Weight estimator configuration

        Returns
        -------
        dict
            Dictionary containing weight estimator configuration information
        """
        return {"center": self._center, "debiased": self._debiased}


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

      g_i & =z_i\epsilon_i\\
      W & =n^{-1}\sum_{i=1}^{n}g'_ig_i

    where :math:`z_i` contains both the exogenous regressors and instruments.
    """

    def __init__(self, center: bool = False, debiased: bool = False) -> None:
        super().__init__(center, debiased)

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar)
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr)
        eps : ndarray
            Model errors (nobs by 1)

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape
        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        w = ze.T @ ze / nobs
        w *= 1 if not self._debiased else nobs / (nobs - nvar)
        return w


class KernelWeightMatrix(HomoskedasticWeightMatrix):
    r"""
    Heteroskedasticity, autocorrelation robust weight estimation

    Parameters
    ----------
    kernel : str
        Name of kernel weighting function to use
    bandwidth : {int, None}
        Bandwidth to use when computing kernel weights
    center : bool
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix.
    debiased : bool
        Flag indicating whether to use small-sample adjustments
    optimal_bw : bool
        Flag indicating whether to estimate the optimal bandwidth, when
        bandwidth is None.  If False, nobs - 2 is used

    Notes
    -----
    Supported kernels:

    * "bartlett", "newey-west" - Bartlett's kernel
    * "parzen", "gallant" - Parzen's kernel
    * "qs", "quadratic-spectral", "andrews" - The quadratic spectral kernel

    .. math::

      g_i & =z_i \epsilon_i \\
      W & =n^{-1}(\Gamma_0+\sum_{j=1}^{n-1}k(j)(\Gamma_j+\Gamma_j')) \\
      \Gamma_j & =\sum_{i=j+1}^n g'_i g_{j-j}

    where :math:`k(j)` is the kernel weight for lag j and :math:`z_i`
    contains both the exogenous regressors and instruments..

    See Also
    --------
    linearmodels.iv.covariance.kernel_weight_bartlett,
    linearmodels.iv.covariance.kernel_weight_parzen,
    linearmodels.iv.covariance.kernel_weight_quadratic_spectral
    """

    def __init__(
        self,
        kernel: str = "bartlett",
        bandwidth: int | None = None,
        center: bool = False,
        debiased: bool = False,
        optimal_bw: bool = False,
    ) -> None:
        super().__init__(center, debiased)
        self._bandwidth = bandwidth
        self._orig_bandwidth = bandwidth
        self._kernel = kernel
        self._kernels = KERNEL_LOOKUP
        self._optimal_bw = optimal_bw

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar)
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr)
        eps : ndarray
            Model errors (nobs by 1)

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape
        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        if self._orig_bandwidth is None and self._optimal_bw:
            g = ze / ze.std(0)[None, :]
            g = g.sum(1)
            self._bandwidth = kernel_optimal_bandwidth(g, self._kernel)
        elif self._orig_bandwidth is None:
            self._bandwidth = nobs - 2
        bw = self._bandwidth
        assert bw is not None
        w = self._kernels[self._kernel](bw, nobs - 1)

        s = cov_kernel(ze, w)
        s *= 1 if not self._debiased else nobs / (nobs - nvar)

        return s

    @property
    def config(self) -> dict[str, str | bool | numpy.ndarray | int | None]:
        """
        Weight estimator configuration

        Returns
        -------
        dict
            Dictionary containing weight estimator configuration information
        """
        return {
            "center": self._center,
            "bandwidth": self._bandwidth,
            "kernel": self._kernel,
            "debiased": self._debiased,
        }

    @property
    def bandwidth(self) -> int | None:
        """Actual bandwidth used in estimating the weight matrix"""
        return self._bandwidth


class OneWayClusteredWeightMatrix(HomoskedasticWeightMatrix):
    """
    Clustered (one-way) weight estimation

    Parameters
    ----------
    clusters : ndarray
        Array indicating cluster membership
    center : bool
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix.
    debiased : bool
        Flag indicating whether to use small-sample adjustments
    """

    def __init__(
        self,
        clusters: linearmodels.typing.data.AnyArray,
        center: bool = False,
        debiased: bool = False,
    ) -> None:
        super().__init__(center, debiased)
        self._clusters = clusters

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar)
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr)
        eps : ndarray
            Model errors (nobs by 1)

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape

        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        clusters = self._clusters
        if clusters.shape[0] != nobs:
            raise ValueError(
                "clusters has the wrong nobs. Expected {}, "
                "got {}".format(nobs, clusters.shape[0])
            )
        clusters = asarray(clusters).copy().squeeze()

        s = cov_cluster(ze, clusters)

        if self._debiased:
            num_clusters = len(unique(clusters))
            scale = (nobs - 1) / (nobs - nvar) * num_clusters / (num_clusters - 1)
            s *= scale

        return s

    @property
    def config(self) -> dict[str, str | bool | numpy.ndarray | int | None]:
        """
        Weight estimator configuration

        Returns
        -------
        dict
            Dictionary containing weight estimator configuration information
        """
        return {
            "center": self._center,
            "clusters": self._clusters,
            "debiased": self._debiased,
        }


class IVGMMCovariance(HomoskedasticCovariance):
    """
    Covariance estimation for GMM models

    Parameters
    ----------
    x : ndarray
        Model regressors (nobs by nvar)
    y : ndarray
        Series ,modeled (nobs by 1)
    z : ndarray
        Instruments used for endogenous regressors (nobs by ninstr)
    params : ndarray
        Estimated model parameters (nvar by 1)
    w : ndarray
        Weighting matrix used in GMM estimation
    cov_type : str
        Covariance estimator to use  Valid choices are

        * "unadjusted", "homoskedastic" - Assumes moment conditions are
          homoskedastic
        * "robust", "heteroskedastic" - Allows for heteroskedasticity by not
          autocorrelation
        * "kernel" - Allows for heteroskedasticity and autocorrelation
        * "cluster" - Allows for one-way cluster dependence

    debiased : bool
        Flag indicating whether to debias the covariance estimator
    cov_config
        Optional keyword arguments that are specific to a particular cov_type

    Notes
    -----
    Optional keyword argument for specific covariance estimators:

    **kernel**

    * ``kernel``: Name of kernel to use.  See
      :class:`~linearmodels.iv.covariance.KernelCovariance` for details on
      available kernels
    * ``bandwidth``: Bandwidth to use when computing the weight.  If not
      provided, nobs - 2 is used.

    **cluster**

    * ``clusters``: Array containing the cluster indices.  See
      :class:`~linearmodels.iv.covariance.ClusteredCovariance`

    See Also
    --------
    linearmodels.iv.covariance.HomoskedasticCovariance,
    linearmodels.iv.covariance.HeteroskedasticCovariance,
    linearmodels.iv.covariance.KernelCovariance,
    linearmodels.iv.covariance.ClusteredCovariance
    """

    # TODO: 2-way clustering
    def __init__(
        self,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        w: linearmodels.typing.data.Float64Array,
        cov_type: str = "robust",
        debiased: bool = False,
        **cov_config: str | bool,
    ) -> None:
        super().__init__(x, y, z, params, debiased)
        self._cov_type = cov_type
        self._cov_config = cov_config
        self.w = w
        self._bandwidth = cov_config.get("bandwidth", None)
        self._kernel = cov_config.get("kernel", "")
        self._name = "GMM Covariance"
        if cov_type in ("robust", "heteroskedastic"):
            score_cov_estimator: Any = HeteroskedasticWeightMatrix
        elif cov_type in ("unadjusted", "homoskedastic"):
            score_cov_estimator = HomoskedasticWeightMatrix
        elif cov_type == "clustered":
            score_cov_estimator = OneWayClusteredWeightMatrix
        elif cov_type == "kernel":
            score_cov_estimator = KernelWeightMatrix
        else:
            raise ValueError("Unknown cov_type")
        self._score_cov_estimator = score_cov_estimator

    def __str__(self) -> str:
        out = super().__str__()
        cov_type = self._cov_type
        if cov_type in ("robust", "heteroskedastic"):
            out += "\nRobust (Heteroskedastic)"
        elif cov_type in ("unadjusted", "homoskedastic"):
            out += "\nUnadjusted (Homoskedastic)"
        elif cov_type == "clustered":
            out += "\nClustered (One-way)"
            clusters = self._cov_config.get("clusters", None)
            if clusters is not None:
                nclusters = len(unique(asarray(clusters)))
                out += f"\nNum Clusters: {nclusters}"
        else:  # kernel
            out += "\nKernel (HAC)"
            if self._cov_config.get("kernel", False):
                out += "\nKernel: {}".format(self._cov_config["kernel"])
            if self._cov_config.get("bandwidth", False):
                out += "\nBandwidth: {}".format(self._cov_config["bandwidth"])
        return out

    @property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        x, z, eps, w = self.x, self.z, self.eps, self.w
        nobs = x.shape[0]
        xpz = x.T @ z / nobs
        xpzw = xpz @ w
        xpzwzpx_inv = inv(xpzw @ xpz.T)

        score_cov = self._score_cov_estimator(
            debiased=self.debiased, **self._cov_config
        )
        s = score_cov.weight_matrix(x, z, eps)
        self._cov_config = score_cov.config

        c = xpzwzpx_inv @ (xpzw @ s @ xpzw.T) @ xpzwzpx_inv / nobs
        return (c + c.T) / 2

    @property
    def config(self) -> dict[str, str | bool | numpy.ndarray | int | None]:
        conf: dict[str, str | bool | numpy.ndarray | int | None] = {
            "debiased": self.debiased
        }
        conf.update(self._cov_config)
        return conf
