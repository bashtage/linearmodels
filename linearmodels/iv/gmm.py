"""
Covariance and weight estimation for GMM IV estimators
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from numpy import (
    asarray,
    diagflat,
    dot,
    kron,
    ndarray,
    ones,
    sum as npsum,
    unique,
    zeros)
from numpy.linalg import inv

from linearmodels.iv.covariance import (
    KERNEL_LOOKUP,
    HomoskedasticCovariance,
    kernel_optimal_bandwidth,
    MisspecificationCovariance,
    OneStepMisspecificationCovariance,
)
from linearmodels.shared.covariance import cov_cluster, cov_kernel
from linearmodels.typing import AnyArray, Float64Array


class HomoskedasticWeightMatrix(object):
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
        self._bandwidth: Optional[int] = 0

    def weight_matrix(
            self, x: Float64Array, z: Float64Array, eps: Float64Array
    ) -> Float64Array:
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
    def config(self) -> Dict[str, Union[str, bool, ndarray, Optional[int]]]:
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
        super(HeteroskedasticWeightMatrix, self).__init__(center, debiased)

    def weight_matrix(
            self, x: Float64Array, z: Float64Array, eps: Float64Array
    ) -> Float64Array:
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

    * 'bartlett', 'newey-west' - Bartlett's kernel
    * 'parzen', 'gallant' - Parzen's kernel
    * 'qs', 'quadratic-spectral', 'andrews' - The quadratic spectral kernel

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
            bandwidth: Optional[int] = None,
            center: bool = False,
            debiased: bool = False,
            optimal_bw: bool = False,
    ) -> None:
        super(KernelWeightMatrix, self).__init__(center, debiased)
        self._bandwidth = bandwidth
        self._orig_bandwidth = bandwidth
        self._kernel = kernel
        self._kernels = KERNEL_LOOKUP
        self._optimal_bw = optimal_bw

    def weight_matrix(
            self, x: Float64Array, z: Float64Array, eps: Float64Array
    ) -> Float64Array:
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
    def config(self) -> Dict[str, Union[str, bool, ndarray, Optional[int]]]:
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
    def bandwidth(self) -> Optional[int]:
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
            self, clusters: AnyArray, center: bool = False, debiased: bool = False
    ) -> None:
        super(OneWayClusteredWeightMatrix, self).__init__(center, debiased)
        self._clusters = clusters

    def weight_matrix(
            self, x: Float64Array, z: Float64Array, eps: Float64Array
    ) -> Float64Array:
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
                "clusters has the wrong nobs. Expected {0}, "
                "got {1}".format(nobs, clusters.shape[0])
            )
        clusters = asarray(clusters).copy().squeeze()

        s = cov_cluster(ze, clusters)

        if self._debiased:
            num_clusters = len(unique(clusters))
            scale = (nobs - 1) / (nobs - nvar) * num_clusters / (num_clusters - 1)
            s *= scale

        return s

    @property
    def config(self) -> Dict[str, Union[str, bool, ndarray, Optional[int]]]:
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


def conutnum(x: Float64Array, binranges: Float64Array) -> Float64Array:
    """
    Bin counting for histograms, equal to MATLAB fun:histc
    Returns
    -------
    ndarray
        Ndarray of bin counting for histograms
    """
    temp = zeros((len(binranges), 2))
    temp[:, 0] = binranges
    for i, element in enumerate(binranges):
        if i == 0:
            temp[i, 1] = npsum((x <= binranges[0]))
        elif i == len(binranges) - 1:
            temp[i, 1] = npsum((x >= element))
        else:
            temp[i, 1] = npsum((x > binranges[i - 1]) & (x <= element))
    return temp


class OneStepMisspecificationWeightMatrix(HomoskedasticWeightMatrix):
    """
    One-Step Misspecification  (one-way) weight estimation

    Parameters
    ----------
    clusters : ndarray
        Array indicating cluster membership
    center : bool, optional
        None
    debiased : bool, optional
        None
    """

    def __init__(
            self, clusters: AnyArray, center: bool = False, debiased: bool = False
    ) -> None:
        super(OneStepMisspecificationWeightMatrix, self).__init__(center, debiased)
        self._clusters = clusters

    def weight_matrix(
            self, x: Float64Array, z: Float64Array, eps: Float64Array
    ) -> Float64Array:
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
        nobs = x.shape[0]
        z_num = z.shape[1]
        clusters = self._clusters
        if clusters.shape[0] != nobs:
            raise ValueError(
                "clusters has the wrong nobs. Expected {0}, "
                "got {1}".format(nobs, clusters.shape[0])
            )
        clusters = asarray(clusters).copy().squeeze()
        cc = unique(clusters)
        G = int(len(cc))
        g_ng = self.conutnum(clusters[:], cc)
        ng = g_ng[:, 1]
        wmat = zeros((z_num, z_num))
        W0i = zeros((z_num, z_num, G))
        for i in range(G):
            Zi = z[int(npsum(ng[0: i + 1]) - ng[i]): int(npsum(ng[0: i + 1])), :]

            h0 = 2 * ones((int(ng[i]), 1))
            h1 = -1 * ones((int(ng[i]) - 1, 1))
            Hm = diagflat(h0) + diagflat(h1, 1) + diagflat(h1, -1)

            W0i[:, :, i] = Zi.T @ Hm @ Zi
            wmat = wmat + W0i[:, :, i]
        wmat = wmat / nobs
        return wmat

    @property
    def config(self) -> Dict[str, Union[str, bool, ndarray, Optional[int]]]:
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


def repmat(a: Float64Array, b: int, c: int) -> Float64Array:
    """
    Parameters
    ----------
    a : ndarray
    b : int
    c : int

    Returns
    -------
    ndarray
        repmat for MATLAB
    """
    return kron(ones((c, b)), a)


class MisspecificationWeightMatrix(HomoskedasticWeightMatrix):
    """
    Misspecification (one-way)  Iter GMM weight estimation

    Parameters
    ----------
    clusters : ndarray
        Array indicating cluster membership
    center : bool, optional
        none
    debiased : bool, optional
        none
    """

    def __init__(
            self, clusters: AnyArray, center: bool = False, debiased: bool = False
    ) -> None:
        super(MisspecificationWeightMatrix, self).__init__(center, debiased)
        self._clusters = clusters

    def weight_matrix(
            self, x: Float64Array, z: Float64Array, eps: Float64Array
    ) -> Float64Array:
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
        nobs = x.shape[0]

        z_num = z.shape[1]
        clusters = self._clusters
        if clusters.shape[0] != nobs:
            raise ValueError(
                "clusters has the wrong nobs. Expected {0}, "
                "got {1}".format(nobs, clusters.shape[0])
            )
        clusters = asarray(clusters).copy().squeeze()

        wmat = zeros((z_num, z_num))
        cc = unique(clusters)
        G = int(len(cc))
        idx = self.repmat(clusters, 1, G).T == kron(
            ones((nobs, 1)), unique(clusters).reshape(1, -1)
        )
        if nobs == G:
            ze = dot(z, repmat(eps, 1, z_num).T)
            wmat = ze.T @ ze
        else:
            for g in range(G):
                zg = z[idx[:, g], :]
                eg = eps[idx[:, g]]
                zeg = zg.T @ eg
                wmat = wmat + zeg @ zeg.T
        wmat = wmat / nobs
        return wmat

    @property
    def config(self) -> Dict[str, Union[str, bool, ndarray, Optional[int]]]:
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

        * 'unadjusted', 'homoskedastic' - Assumes moment conditions are
          homoskedastic
        * 'robust', 'heteroskedastic' - Allows for heteroskedasticity by not
          autocorrelation
        * 'kernel' - Allows for heteroskedasticity and autocorrelation
        * 'cluster' - Allows for one-way cluster dependence

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
            x: Float64Array,
            y: Float64Array,
            z: Float64Array,
            params: Float64Array,
            w: Float64Array,
            cov_type: str = "robust",
            debiased: bool = False,
            **cov_config: Union[str, bool],
    ) -> None:
        super(IVGMMCovariance, self).__init__(x, y, z, params, debiased)
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
        elif cov_type == "Misspecification":
            score_cov_estimator = MisspecificationWeightMatrix
        elif cov_type == "OneStepMisspecification":
            score_cov_estimator = OneStepMisspecificationWeightMatrix
        else:
            raise ValueError("Unknown cov_type")
        self._score_cov_estimator = score_cov_estimator

    def __str__(self) -> str:
        out = super(IVGMMCovariance, self).__str__()
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
                out += "\nNum Clusters: {0}".format(nclusters)
        elif cov_type == "Misspecification":
            out += "\nmisspecification (One-way)"
            clusters = self._cov_config.get("clusters", None)
            if clusters is not None:
                nclusters = len(unique(asarray(clusters)))
                out += "\nNum Clusters: {0}".format(nclusters)
        elif cov_type == "OneStepMisspecification":
            out += "\nOneStepMisspecification (One-way)"
            clusters = self._cov_config.get("clusters", None)
            if clusters is not None:
                nclusters = len(unique(asarray(clusters)))
                out += "\nNum Clusters: {0}".format(nclusters)
        else:  # kernel
            out += "\nKernel (HAC)"
            if self._cov_config.get("kernel", False):
                out += "\nKernel: {0}".format(self._cov_config["kernel"])
            if self._cov_config.get("bandwidth", False):
                out += "\nBandwidth: {0}".format(self._cov_config["bandwidth"])
        return out

    @property
    def cov(self) -> Float64Array:
        x, z, eps, w = self.x, self.z, self.eps, self.w
        score_cov = self._score_cov_estimator(
            debiased=self.debiased, **self._cov_config
        )
        self._cov_config = score_cov.config
        if self._cov_type == "OneStepMisspecification":
            c = OneStepMisspecificationCovariance(
                self.x,
                self.y,
                self.z,
                self.params,
                self.config["clusters"],
                self.config["debiased"],
            ).s
        elif self._cov_type == "Misspecification":
            c = MisspecificationCovariance(
                self.x,
                self.y,
                self.z,
                self.params,
                self.config["clusters"],
                self.config["debiased"],
                w=self.w,
            ).s
        else:

            nobs = x.shape[0]
            xpz = x.T @ z / nobs
            xpzw = xpz @ w
            xpzwzpx_inv = inv(xpzw @ xpz.T)
            s = score_cov.weight_matrix(x, z, eps)
            c = xpzwzpx_inv @ (xpzw @ s @ xpzw.T) @ xpzwzpx_inv / nobs
            c = (c + c.T) / 2

        return c

    @property
    def config(self) -> Dict[str, Union[str, bool, ndarray, Optional[int]]]:
        conf: Dict[str, Union[str, bool, ndarray, Optional[int]]] = {
            "debiased": self.debiased
        }
        conf.update(self._cov_config)
        return conf
