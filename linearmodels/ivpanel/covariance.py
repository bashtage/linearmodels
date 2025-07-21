from __future__ import annotations

from functools import cached_property
from linearmodels.panel.covariance import HomoskedasticCovariance as PHC
from linearmodels.panel.covariance import setup_covariance_estimator
from linearmodels.iv.covariance import HomoskedasticCovariance as IVHC
from linearmodels.iv.covariance import CLUSTER_ERR, cov_cluster
from linearmodels.shared.covariance import cluster_union, group_debias_coefficient
from linearmodels.typing import AnyArray, ArrayLike, Float64Array, IntArray, Numeric
import numpy as np
from numpy import arange, asarray, int64, ndarray, unique
from numpy.linalg import inv, pinv
from typing import Any, Union, cast

from linearmodels.shared.typed_getters import (
    get_array_like,
    get_bool,
    get_float,
    get_string,
)


class HomoskedasticCovariance(PHC, IVHC):
    def __init__(
            self,
            y: Float64Array,
            x: Float64Array,
            z: Float64Array,
            params: Float64Array,
            entity_ids: IntArray | None,
            time_ids: IntArray | None,
            *,
            debiased: bool = False,
            extra_df: int = 0,
            kappa: Numeric = 1,
    ):
        self._y = y
        self._x = x
        self._z = z
        self._params = params
        self._entity_ids = entity_ids
        self._time_ids = time_ids
        self._debiased = debiased
        self._extra_df = extra_df
        self._nobs, self._nvar = x.shape
        self._nobs_eff = self._nobs - extra_df
        if debiased:
            self._nobs_eff -= self._nvar
        self._scale = self._nobs / self._nobs_eff
        self._name = "Unadjusted"

        if not (x.shape[0] == y.shape[0] == z.shape[0]):
            raise ValueError("x, y and z must have the same number of rows")
        if not x.shape[1] == len(params):
            raise ValueError("x and params must have compatible dimensions")

        self.x = x
        self.y = y
        self.z = z
        self.params = params
        self._debiased = debiased
        # self.eps = y - x @ params
        self._kappa = kappa
        self._pinvz = pinv(z)
        nobs, nvar = x.shape
        self._scale: float = nobs / (nobs - nvar) if self._debiased else 1.0
        self._name = "Unadjusted Covariance (Homoskedastic)"

    @property
    def name(self) -> str:
        """Covariance estimator name"""
        return self._name

    @property
    def eps(self) -> Float64Array:
        """Model residuals"""
        return self._y - self._x @ self._params

    @property
    def s2(self) -> float:
        """Error variance"""
        eps = self.eps
        return self._scale * float(np.squeeze(eps.T @ eps)) / self._nobs

    # @cached_property
    # def cov(self) -> Float64Array:
    #     """Estimated covariance"""
    #     x = self._x
    #     out = self.s2 * inv(x.T @ x)
    #     return (out + out.T) / 2

    @property
    def cov(self) -> Float64Array:
        """Covariance of estimated parameters"""

        x, z = self.x, self.z
        nobs = x.shape[0]

        pinvz = self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        if self._kappa != 1:
            kappa = self._kappa
            xpx = x.T @ x / nobs
            v = (1 - kappa) * xpx + kappa * v

        vinv = inv(v)
        c = vinv @ self.s @ vinv / nobs
        return (c + c.T) / 2

    def deferred_cov(self) -> Float64Array:
        """Covariance calculation deferred until executed"""
        return self.cov

    def __str__(self) -> str:
        out = self._name
        out += f"\nDebiased: {self._debiased}"
        if self._kappa != 1:
            out += f"\nKappa: {self._kappa:0.3f}"
        return out

    def __repr__(self) -> str:
        return (
                self.__str__() + "\n" + self.__class__.__name__ + f", id: {hex(id(self))}"
        )

    @property
    def s(self) -> Float64Array:
        """Score covariance estimate"""
        x, z, eps = self.x, self.z, self.eps
        nobs = x.shape[0]
        s2 = eps.T @ eps / nobs
        pinvz = self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        if self._kappa != 1:
            kappa = self._kappa
            xpx = x.T @ x / nobs
            v = (1 - kappa) * xpx + kappa * v

        return self._scale * s2 * v


    # @property
    # def s2(self) -> Float64Array:
    #     """
    #     Estimated variance of residuals. Small-sample adjusted if debiased.
    #     """
    #     nobs = self.x.shape[0]
    #     eps = self.eps
    #
    #     return self._scale * eps.T @ eps / nobs

    @property
    def debiased(self) -> bool:
        """Flag indicating if covariance is debiased"""
        return self._debiased

    @property
    def config(self) -> dict[str, Any]:
        return {"debiased": self.debiased, "kappa": self._kappa}


class ClusteredCovariance(HomoskedasticCovariance):
    ALLOWED_KWARGS = ("clusters", "group_debias", "cluster_entity", "cluster_time")

    def __init__(
            self,
            y: Float64Array,
            x: Float64Array,
            z: Float64Array,
            params: Float64Array,
            entity_ids: IntArray,
            time_ids: IntArray,
            *,
            debiased: bool = False,
            extra_df: int = 0,
            cluster_entity: bool = False,
            cluster_time: bool = False,
            clusters: ArrayLike | None = None,
            group_debias: bool = False,
            kappa: Numeric = 1,
    ) -> None:
        super().__init__(
            y, x, z, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df, kappa=kappa
        )
        if clusters is None:
            clusters = np.arange(self._x.shape[0])
        clusters = np.asarray(clusters).squeeze()
        assert clusters is not None
        self._group_debias = bool(group_debias)
        dim1 = 1 if clusters.ndim == 1 else clusters.shape[1]
        if clusters.ndim > 2 or dim1 > 2:
            raise ValueError("Only 1 or 2-way clustering supported.")
        nobs = y.shape[0]
        if clusters.shape[0] != nobs:
            raise ValueError(CLUSTER_ERR.format(nobs, clusters.shape[0]))
        self._cluster_entity = cluster_entity
        self._cluster_time = cluster_time
        self._clusters = clusters
        self._name = "Clustered"






        # nobs = x.shape[0]
        # clusters = arange(nobs) if clusters is None else clusters
        # clusters = cast(AnyArray, asarray(clusters).squeeze())
        # if clusters.shape[0] != nobs:
        #     raise ValueError(CLUSTER_ERR.format(nobs, clusters.shape[0]))
        # self._clusters = clusters
        # if clusters.ndim == 1:
        #     self._num_clusters = [len(unique(clusters))]
        #     self._num_clusters_str = str(self._num_clusters[0])
        # else:
        #     self._num_clusters = [
        #         len(unique(clusters[:, 0])),
        #         len(unique(clusters[:, 1])),
        #     ]
        #     self._num_clusters_str = ", ".join(map(str, self._num_clusters))
        # if clusters is not None and clusters.shape[0] != nobs:
        #     raise ValueError(CLUSTER_ERR.format(nobs, clusters.shape[0]))




    @cached_property
    def cov(self) -> Float64Array:
        """Estimated covariance"""
        x, z = self._x, self._z
        nobs = x.shape[0]

        pinvz = self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        vinv = inv(v)

        eps = self.eps
        xhat_e = asarray(z @ (pinvz @ x) * eps, dtype=float)
        if self._clusters.ndim == 1:
            xeex = cov_cluster(xhat_e, self._clusters)
            if self._group_debias:
                xeex *= group_debias_coefficient(self._clusters)

        else:
            clusters0 = self._clusters[:, 0]
            clusters1 = self._clusters[:, 1]
            xeex0 = cov_cluster(xhat_e, clusters0)
            xeex1 = cov_cluster(xhat_e, clusters1)

            clusters01 = cluster_union(self._clusters)
            xeex01 = cov_cluster(xhat_e, clusters01)

            if self._group_debias:
                xeex0 *= group_debias_coefficient(clusters0)
                xeex1 *= group_debias_coefficient(clusters1)
                xeex01 *= group_debias_coefficient(clusters01)

            xeex = xeex0 + xeex1 - xeex01

        xeex *= self._scale
        out = (vinv @ xeex @ vinv) / nobs
        return (out + out.T) / 2

    @property
    def config(self) -> dict[str, Any]:
        return {"debiased": self.debiased, "kappa": self._kappa,
                "cluster_entity": self._cluster_entity,
                "cluster_time": self._cluster_time}


CovarianceEstimatorType = Union[
    type[HomoskedasticCovariance],
    # type[HeteroskedasticCovariance],
    type[ClusteredCovariance],
    # type[DriscollKraay],
    # type[ACCovariance],
]


class CovarianceManager:
    COVARIANCE_ESTIMATORS: dict[str, CovarianceEstimatorType] = {
        "unadjusted": HomoskedasticCovariance,
        "conventional": HomoskedasticCovariance,
        "homoskedastic": HomoskedasticCovariance,
        # "robust": HeteroskedasticCovariance,
        # "heteroskedastic": HeteroskedasticCovariance,
        "cluster": ClusteredCovariance,
        "clustered": ClusteredCovariance,
        # "driscoll-kraay": DriscollKraay,
        # "dk": DriscollKraay,
        # "kernel": DriscollKraay,
        # "ac": ACCovariance,
        # "autocorrelated": ACCovariance,
    }

    def __init__(
        self, estimator: str, *cov_estimators: CovarianceEstimatorType
    ) -> None:
        self._estimator = estimator
        self._supported = cov_estimators

    def __getitem__(self, item: str) -> CovarianceEstimatorType:
        if item not in self.COVARIANCE_ESTIMATORS:
            raise KeyError("Unknown covariance estimator type.")
        cov_est = self.COVARIANCE_ESTIMATORS[item]
        if cov_est not in self._supported:
            raise ValueError(
                "Requested covariance estimator is not supported "
                "for the {}.".format(self._estimator)
            )
        return cov_est


def setup_covariance_estimator(
    cov_estimators: CovarianceManager,
    cov_type: str,
    y: Float64Array,
    x: Float64Array,
    z: Float64Array,
    params: Float64Array,
    entity_ids: IntArray,
    time_ids: IntArray,
    *,
    debiased: bool = False,
    extra_df: int = 0,
    **cov_config: Any,
) -> HomoskedasticCovariance:
    estimator = cov_estimators[cov_type]
    unknown_kwargs = [
        str(key) for key in cov_config if str(key) not in estimator.ALLOWED_KWARGS
    ]
    if unknown_kwargs:
        if estimator.ALLOWED_KWARGS:
            allowed = ", ".join(estimator.ALLOWED_KWARGS)
            kwarg_err = f"only supports the keyword arguments: {allowed}"
        else:
            kwarg_err = "does not support any keyword arguments"
        msg = (
            f"Covariance estimator {estimator.__name__} {kwarg_err}. Unknown keyword "
            f"arguments were passed to the estimator. The unknown keyword argument(s) "
            f"are: {', '.join(unknown_kwargs)} "
        )
        raise ValueError(msg)
    # kernel = get_string(cov_config, "kernel")
    # bandwidth = get_float(cov_config, "bandwidth")
    group_debias = get_bool(cov_config, "group_debias")
    cluster_entity = get_bool(cov_config, "cluster_entity")
    cluster_time = get_bool(cov_config, "cluster_time")
    clusters = get_array_like(cov_config, "clusters")

    if estimator is HomoskedasticCovariance:
        return HomoskedasticCovariance(
            y, x, z, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df
        )
    # elif estimator is HeteroskedasticCovariance:
    #     return HeteroskedasticCovariance(
    #         y, x, z, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df
    #     )
    elif estimator is ClusteredCovariance:
        return ClusteredCovariance(
            y,
            x,
            z,
            params,
            entity_ids,
            time_ids,
            debiased=debiased,
            extra_df=extra_df,
            cluster_entity=cluster_entity,
            cluster_time=cluster_time,
            clusters=clusters,
            group_debias=group_debias,
        )
    # elif estimator is DriscollKraay:
    #     return DriscollKraay(
    #         y,
    #         x,
    #         z,
    #         params,
    #         entity_ids,
    #         time_ids,
    #         debiased=debiased,
    #         extra_df=extra_df,
    #         kernel=kernel,
    #         bandwidth=bandwidth,
    #     )
    # else:  # ACCovariance:
    #     return ACCovariance(
    #         y,
    #         x,
    #         z,
    #         params,
    #         entity_ids,
    #         time_ids,
    #         debiased=debiased,
    #         extra_df=extra_df,
    #         kernel=kernel,
    #         bandwidth=bandwidth,
    #     )
    #
