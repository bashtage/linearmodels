"""
Instrumental variable estimators.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar, Union, cast
import warnings

from numpy import (
    all as npall,
    any as npany,
    array,
    asarray,
    atleast_2d,
    average,
    c_,
    column_stack,
    diag,
    dtype,
    eye,
    finfo,
    float64,
    isfinite,
    isscalar,
    logical_not,
    nan,
    ndarray,
    ones,
    outer,
    sqrt,
    squeeze,
    unique,
)
from numpy.linalg import LinAlgError, eigvalsh, inv, matrix_rank, norm, pinv, qr, solve
from pandas import DataFrame, Series, concat
from scipy.optimize import minimize

from linearmodels.iv._utility import IVFormulaParser
from linearmodels.iv.common import f_statistic, find_constant
from linearmodels.iv.covariance import (
    KERNEL_LOOKUP,
    ClusteredCovariance,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
    KernelCovariance,
)
from linearmodels.iv.data import IVData, IVDataLike
from linearmodels.iv.gmm import (
    HeteroskedasticWeightMatrix,
    HomoskedasticWeightMatrix,
    IVGMMCovariance,
    KernelWeightMatrix,
    OneWayClusteredWeightMatrix,
    _uses_default_bandwidth,
    _warn_default_bandwidth,
)
from linearmodels.iv.results import IVGMMResults, IVResults, OLSResults
from linearmodels.shared.alignment import (
    check_alignment,
    reference_labels,
    row_labels,
)
from linearmodels.shared.exceptions import (
    IndexWarning,
    check_cov_config,
    lookup_option,
    lower_option,
    missing_warning,
    unknown_option_error,
)
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.shared.linalg import has_constant, inv_sqrth
from linearmodels.shared.rsquared import r_squared
from linearmodels.shared.utility import DataFrameWrapper, SeriesWrapper
from linearmodels.shared.weights import check_weights, normalize_weights
import linearmodels.typing
import linearmodels.typing.data

IVResultType = type[Union[IVResults, IVGMMResults, OLSResults]]

__all__ = [
    "COVARIANCE_ESTIMATORS",
    "IV2SLS",
    "IVGMM",
    "IVGMMCUE",
    "IVJIVE",
    "IVLIML",
    "WEIGHT_MATRICES",
    "_OLS",
    "IVResultType",
]

_CENTER = ("center",)
# kappa is overwritten by fit. It is accepted so that the cov_config of results
# can be passed to fit. extra_df is the number of degrees of freedom that the model
# uses beyond the regressors, which the covariance estimators document
_LS_KEYS = (*_CENTER, "kappa", "extra_df")
_COV_KEYS = {
    HomoskedasticCovariance: _LS_KEYS,
    HeteroskedasticCovariance: _LS_KEYS,
    KernelCovariance: (*_LS_KEYS, "kernel", "bandwidth"),
    ClusteredCovariance: (*_LS_KEYS, "clusters"),
}
# The keys of the covariance estimators of GMM models, which also select the
# bandwidth from the data
_GMM_COV_KEYS = {
    "unadjusted": _CENTER,
    "homoskedastic": _CENTER,
    "robust": _CENTER,
    "heteroskedastic": _CENTER,
    "clustered": (*_CENTER, "clusters"),
    "kernel": (*_CENTER, "kernel", "bandwidth", "optimal_bw"),
}

COVARIANCE_ESTIMATORS = {
    "homoskedastic": HomoskedasticCovariance,
    "unadjusted": HomoskedasticCovariance,
    "HomoskedasticCovariance": HomoskedasticCovariance,
    "homo": HomoskedasticCovariance,
    "robust": HeteroskedasticCovariance,
    "heteroskedastic": HeteroskedasticCovariance,
    "HeteroskedasticCovariance": HeteroskedasticCovariance,
    "hccm": HeteroskedasticCovariance,
    "kernel": KernelCovariance,
    "KernelCovariance": KernelCovariance,
    "one-way": ClusteredCovariance,
    "clustered": ClusteredCovariance,
    "OneWayClusteredCovariance": ClusteredCovariance,
}
_LS_COV_KEYS = {
    name.lower(): _COV_KEYS[estimator]
    for name, estimator in COVARIANCE_ESTIMATORS.items()
}

CovarianceEstimator = TypeVar(
    "CovarianceEstimator",
    HomoskedasticCovariance,
    HeteroskedasticCovariance,
    KernelCovariance,
    ClusteredCovariance,
)

WEIGHT_MATRICES = {
    "unadjusted": HomoskedasticWeightMatrix,
    "homoskedastic": HomoskedasticWeightMatrix,
    "robust": HeteroskedasticWeightMatrix,
    "heteroskedastic": HeteroskedasticWeightMatrix,
    "kernel": KernelWeightMatrix,
    "clustered": OneWayClusteredWeightMatrix,
    "one-way": OneWayClusteredWeightMatrix,
}

# The residuals of the dependent variable and the endogenous regressors on the exogenous
# regressors and instruments are linearly dependent if the smallest singular value of
# the residuals, scaled to have columns of unit length, is no larger than this. It is the
# square root of the machine precision, about 1.5e-8. The cross product of the scaled
# residuals has the squares of the singular values as its eigenvalues, so that at this
# value its smallest eigenvalue is at its rounding error, about 2.2e-16 times the largest
# eigenvalue, and has no digits left.
_KAPPA_TOLERANCE = float(sqrt(finfo(float64).eps))


class _IVModelBase:
    r"""
    Limited information ML and k-class estimation of IV models.

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like or None
        Exogenous regressors (nobs by nexog). None if the model has no exogenous
        regressors.
    endog : array_like or None
        Endogenous regressors (nobs by nendog). None if the model has no endogenous
        regressors.
    instruments : array_like or None
        Instrumental variables (nobs by ninstr). None if the model has no instruments.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.
    fuller : float, optional
        Fuller's alpha to modify the LIML estimator. The default is 0, which returns
        the unmodified LIML estimator.
    kappa : float, optional
        Parameter value for k-class estimation. If not provided, it is computed to
        produce the LIML parameter estimate.

    Raises
    ------
    ValueError
        If the model has no regressors, if there are fewer instruments than
        endogenous regressors, if ``[exog, endog]`` or ``[exog, instruments]``
        does not have full column rank, if all observations have a missing value,
        if the inputs have different numbers of observations or labels that do
        not match (see Notes), if ``weights`` are not strictly positive and
        finite, or if ``kappa`` or ``fuller`` is not a scalar.
    TypeError
        If an input is complex or cannot be converted to float64.

    See Also
    --------
    IV2SLS : Estimation of IV models using two-stage least squares.
    IVGMM : Estimation of IV models using the generalized method of moments.
    IVGMMCUE : Estimation of IV models using continuously updating GMM.

    Notes
    -----
    ``kappa`` and ``fuller`` should not be used simultaneously since Fuller's
    alpha applies an adjustment to ``kappa``, and so the same result can be
    computed using only ``kappa``. Fuller's alpha is used to adjust the
    LIML estimate of :math:`\kappa`, which is computed whenever ``kappa``
    is not provided.

    The LIML estimator is defined as

    .. math::

      \hat{\beta}_{\kappa} & =(X(I-\kappa M_{z})X)^{-1}X(I-\kappa M_{z})Y\\
      M_{z} & =I-P_{z}\\
      P_{z} & =Z(Z'Z)^{-1}Z'

    where :math:`Z` contains both the exogenous regressors and the instruments.
    :math:`\kappa` is estimated as part of the LIML estimator.

    When using Fuller's :math:`\alpha`, the value used is modified to

    .. math::

      \kappa-\alpha/(n-n_{instr})

    .. todo::

      * VCV: bootstrap
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike | None,
        endog: IVDataLike | None,
        instruments: IVDataLike | None,
        *,
        weights: IVDataLike | None = None,
        fuller: linearmodels.typing.Numeric = 0,
        kappa: linearmodels.typing.OptionalNumeric = None,
    ):
        self.dependent = IVData(dependent, var_name="dependent")
        nobs: int = self.dependent.shape[0]
        self.exog = IVData(exog, var_name="exog", nobs=nobs)
        self.endog = IVData(endog, var_name="endog", nobs=nobs)
        self.instruments = IVData(instruments, var_name="instruments", nobs=nobs)
        self._original_index = self.dependent.pandas.index
        if weights is None:
            weights = ones(self.dependent.shape)
        weights_data = IVData(weights, var_name="weights")
        inputs = {
            "exog": self.exog,
            "endog": self.endog,
            "instruments": self.instruments,
            "weights": weights_data,
        }
        # The inputs with labels use the labels of dependent. If dependent has no
        # labels, they use the labels of the first of them that has labels
        self._labels, self._labels_name = reference_labels(
            row_labels(self.dependent), inputs
        )
        check_alignment(self._labels, inputs, reference=self._labels_name)
        weights = weights_data.ndarray
        check_weights(weights)
        self.weights = IVData(weights, var_name="weights", nobs=nobs)

        self._drop_locs = self._drop_missing()
        # The weights have mean 1 in the sample that is used to estimate the model,
        # which does not include the observations that were dropped
        kept = self.weights.ndarray
        self.weights = IVData(normalize_weights(kept), var_name="weights")
        # Weighted data is data scaled by the square root of the weights
        self._sqrt_weights = w = sqrt(self.weights.ndarray)
        # dependent variable
        self._y = self.dependent.ndarray
        self._wy = self._y * w
        # model regressors
        self._x = c_[self.exog.ndarray, self.endog.ndarray]
        self._wx = self._x * w
        # first-stage regressors
        self._z = c_[self.exog.ndarray, self.instruments.ndarray]
        self._wz = self._z * w

        self._has_constant = False
        self._regressor_is_exog = array(
            [True] * self.exog.shape[1] + [False] * self.endog.shape[1]
        )
        self._columns = self.exog.cols + self.endog.cols
        self._instr_columns = self.exog.cols + self.instruments.cols
        self._index = self.dependent.rows

        self._validate_inputs()
        if not hasattr(self, "_method"):
            self._method = "IV-LIML"
            additional = []
            if fuller != 0:
                additional.append(f"fuller(alpha={fuller})")
            if kappa is not None:
                additional.append(f"kappa={kappa}")
            if additional:
                self._method += "(" + ", ".join(additional) + ")"

        self._kappa = kappa
        self._fuller = fuller
        if kappa is not None and not isscalar(kappa):
            raise ValueError("kappa must be None or a scalar")
        if not isscalar(fuller):
            raise ValueError("fuller must be None or a scalar")
        if kappa is not None and fuller != 0:
            warnings.warn(
                "kappa and fuller should not normally be used "
                "simultaneously.  Identical results can be computed "
                "using kappa only",
                UserWarning,
                stacklevel=2,
            )
        if endog is None and instruments is None:
            self._method = "OLS"
        self._formula = ""

    def predict(
        self,
        params: linearmodels.typing.data.ArrayLike,
        *,
        exog: IVDataLike | None = None,
        endog: IVDataLike | None = None,
        data: DataFrame | None = None,
        eval_env: int = 2,
    ) -> DataFrame:
        """
        Predict values for additional data.

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1).
        exog : array_like, optional
            Exogenous regressors (nobs by nexog).
        endog : array_like, optional
            Endogenous regressors (nobs by nendog).
        data : DataFrame, optional
            Values to use when making predictions from a model constructed
            from a formula.
        eval_env : int, optional
            Depth of the stack frame whose variables are used when evaluating the
            formula, counted from the formula parser. The default is 2, which is
            the frame that calls predict.

        Returns
        -------
        DataFrame
            Fitted values from supplied data and parameters.

        Raises
        ------
        ValueError
            If ``data`` is used and the model was not created using a formula,
            if ``data`` is used with ``exog`` or ``endog``, if none of ``exog``,
            ``endog`` and ``data`` is provided, or if ``exog`` and ``endog`` have
            different numbers of rows.

        Warns
        -----
        IndexWarning
            If the indices of ``exog`` and ``endog`` do not match. The index of
            ``exog`` is used.

        Notes
        -----
        The number of parameters must satisfy nvar = nexog + nendog.

        When using `exog` and `endog`, regressor matrix is constructed as
        `[exog, endog]` and so parameters must be aligned to this structure.
        This is the same structure used in model estimation.

        If `data` is not none, then `exog` and `endog` must be none.
        Predictions from models constructed using formulas can
        be computed using either `exog` and `endog`, which will treat these are
        arrays of values corresponding to the formula-processed data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.
        """
        if data is not None and not self.formula:
            raise ValueError(
                "Unable to use data when the model was not created using a formula."
            )
        if data is not None and (exog is not None or endog is not None):
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog/endog or data, but not both."
            )
        if exog is not None:
            exog = IVData(exog, var_name="exog").pandas
        if endog is not None:
            endog = IVData(endog, var_name="endog").pandas
        if data is not None:
            parser = IVFormulaParser(self.formula, data, eval_env=eval_env)
            exog = parser.exog
            endog = parser.endog
        if all(a is None for a in (exog, endog, data)):
            raise ValueError("At least one of exog, endog, or data must be provided.")
        if exog is None:
            assert endog is not None
            exog = IVData(exog, nobs=endog.shape[0]).pandas
            exog.index = endog.index
        if endog is None:
            assert exog is not None
            endog = IVData(endog, nobs=exog.shape[0]).pandas
            endog.index = exog.index
        assert exog is not None
        assert endog is not None
        if exog.shape[0] != endog.shape[0]:
            raise ValueError("exog and endog must have the same number of rows.")
        if (exog.index != endog.index).any():
            warnings.warn(
                "The indices of exog and endog do not match.  Predictions created "
                "using the index of exog.",
                IndexWarning,
                stacklevel=2,
            )
        exog_endog = concat([exog, endog], axis=1)
        x = asarray(exog_endog)
        params = atleast_2d(asarray(params))
        if params.shape[0] == 1:
            params = params.T
        pred = DataFrame(x @ params, index=exog_endog.index, columns=["predictions"])

        return pred

    @property
    def formula(self) -> str:
        """Formula used to create the model."""
        return self._formula

    @formula.setter
    def formula(self, value: str) -> None:
        """
        Set the formula used to create the model.

        Parameters
        ----------
        value : str
            The formula.
        """
        self._formula = value

    def _validate_inputs(self) -> None:
        x, z = self._x, self._z
        if x.shape[1] == 0:
            raise ValueError("Model must contain at least one regressor.")
        if self.instruments.shape[1] < self.endog.shape[1]:
            raise ValueError(
                "The number of instruments ({}) must be at least "
                "as large as the number of endogenous regressors"
                " ({}).".format(self.instruments.shape[1], self.endog.shape[1])
            )
        if matrix_rank(x) < x.shape[1]:
            raise ValueError("regressors [exog endog] do not have full column rank")
        if matrix_rank(z) < z.shape[1]:
            raise ValueError(
                "instruments [exog instruments]  do not have full column rank"
            )
        self._has_constant, self._const_loc = has_constant(x)

    def _drop_missing(self) -> linearmodels.typing.data.BoolArray:
        data = (self.dependent, self.exog, self.endog, self.instruments, self.weights)
        missing = cast(
            "linearmodels.typing.data.BoolArray",
            npany(column_stack([dh.isnull for dh in data]), axis=1),
        )
        if npany(missing):
            if npall(missing):
                raise ValueError(
                    "All observations contain missing data. "
                    "Model cannot be estimated."
                )
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self.endog.drop(missing)
            self.instruments.drop(missing)
            self.weights.drop(missing)

        missing_warning(missing, stacklevel=4)
        return missing

    def _check_clusters(self, config: Mapping[str, Any]) -> None:
        """Check that clusters with labels use the labels of the other inputs"""
        # Clusters are for the observations that are used to estimate the model
        labels = None if self._labels is None else self._labels[self.notnull]
        check_alignment(
            labels, {"clusters": config.get("clusters")}, reference=self._labels_name
        )

    def wresids(
        self, params: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute weighted model residuals.

        Parameters
        ----------
        params : ndarray
            Model parameters (nvar by 1).

        Returns
        -------
        ndarray
            Weighted model residuals.

        Notes
        -----
        Uses weighted versions of data instead of raw data.  Identical to
        resids if all weights are unity.
        """
        return self._wy - self._wx @ params

    def resids(
        self, params: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute model residuals.

        Parameters
        ----------
        params : ndarray
            Model parameters (nvar by 1).

        Returns
        -------
        ndarray
            Model residuals.
        """
        return self._y - self._x @ params

    @property
    def has_constant(self) -> bool:
        """Flag indicating the model includes a constant or equivalent."""
        return self._has_constant

    @property
    def isnull(self) -> linearmodels.typing.data.BoolArray:
        """Locations of observations with missing values."""
        return self._drop_locs

    @property
    def notnull(self) -> linearmodels.typing.data.BoolArray:
        """Locations of observations included in estimation."""
        return cast("linearmodels.typing.data.BoolArray", logical_not(self._drop_locs))

    def _f_statistic(
        self,
        params: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        debiased: bool,
    ) -> WaldTestStatistic | InvalidTestStatistic:
        const_loc = find_constant(self._x)
        nobs, nvar = self._x.shape
        return f_statistic(params, cov, debiased, nobs - nvar, const_loc)

    def _estimate_kappa(self) -> float:
        """
        The value of kappa in LIML, which is the same for any estimator

        Returns
        -------
        float
            The smallest eigenvalue of the cross product of the residuals on the
            exogenous regressors, relative to the cross product of the residuals on
            the exogenous regressors and instruments. The residuals are those of the
            dependent variable and the endogenous regressors.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the residuals on the exogenous regressors and instruments are linearly
            dependent, so that their cross product, which is inverted, is singular.
            This occurs if there are fewer than 1 + the number of endogenous
            regressors observations more than the number of exogenous regressors and
            instruments, and if the columns of the residuals are collinear, to
            rounding error, or a column is 0, for another reason.

        Notes
        -----
        Whether the residuals are linearly dependent does not depend on the units of
        the variables: the count of observations does not, and the rank is tested on
        residuals that are scaled to have columns of unit length. Kappa is computed
        from the residuals as they are. Two cases are not covered, see issue 823. The
        accuracy of kappa depends on the units of the variables if there are 2 or more
        endogenous regressors, and a regressor that is exactly a linear combination of
        the exogenous regressors and instruments, which has residuals that are only
        rounding error, is not detected.
        """
        y, x, z = self._wy, self._wx, self._wz
        is_exog = self._regressor_is_exog
        e = c_[y, x[:, ~is_exog]]
        x1 = x[:, is_exog]

        nobs, nz = z.shape
        ncol = e.shape[1]
        # z has full column rank, so the residuals on z have a rank of at most nobs - nz.
        # The count does not depend on the rounding error of the projections, which is
        # large if z is nearly collinear, and which can hide the rank of the residuals
        if nobs - nz < ncol:
            raise LinAlgError(
                f"Kappa cannot be estimated using {nobs} observations. It needs at "
                f"least {nz + ncol} observations: the {nz} exogenous regressors and "
                f"instruments plus {ncol}, which is 1 plus the number of endogenous "
                f"regressors ({ncol - 1}), so that the residuals of the dependent "
                "variable and the endogenous regressors on the exogenous regressors "
                "and instruments are linearly independent."
            )

        ez = e - z @ (pinv(z) @ e)
        if x1.shape[1] == 0:  # No exogenous regressors
            ex1 = e
        else:
            ex1 = e - x1 @ (pinv(x1) @ e)

        # The rank is tested on a copy of the residuals with columns of unit length, so
        # that it does not depend on the units of the variables. A column that is exactly
        # 0 cannot be scaled, and is linearly dependent. Kappa is computed from the
        # residuals as they are, see issue 823
        scale = norm(ez, axis=0)
        full_rank = False
        if npall(scale > 0):
            full_rank = bool(matrix_rank(ez / scale, tol=_KAPPA_TOLERANCE) == ncol)
        if not full_rank:
            raise LinAlgError(
                "The residuals of the dependent variable and the endogenous "
                "regressors on the exogenous regressors and instruments are linearly "
                "dependent, so that the cross product of the residuals, which is "
                "inverted to estimate kappa, is singular."
            )

        vpmzv_sqinv = inv_sqrth(ez.T @ ez)
        q = vpmzv_sqinv @ (ex1.T @ ex1) @ vpmzv_sqinv
        return min(eigvalsh(q))

    def _post_estimation(
        self,
        params: linearmodels.typing.data.Float64Array,
        cov_estimator: CovarianceEstimator,
        cov_type: str,
    ) -> dict[str, Any]:
        columns = self._columns
        index = self._index
        eps = self.resids(params)
        y = self.dependent.pandas
        fitted = DataFrameWrapper(
            asarray(y) - eps, index=y.index, columns=["fitted_values"]
        )
        weps = self.wresids(params)
        cov = cov_estimator.cov
        debiased = cov_estimator.debiased

        residual_ss = squeeze(weps.T @ weps)

        w = self.weights.ndarray
        e = self._wy
        if self.has_constant:
            e = e - self._sqrt_weights * average(self._y, weights=w)

        total_ss = float(squeeze(e.T @ e))
        y_ss = float(squeeze(self._wy.T @ self._wy))
        r2 = r_squared(float(residual_ss), total_ss, y_ss, self._y.shape[0])

        fstat = self._f_statistic(params, cov, debiased)
        out = {
            "params": Series(params.squeeze(), columns, name="parameter"),
            "eps": SeriesWrapper(eps.squeeze(), index=index, name="residual"),
            "weps": SeriesWrapper(
                weps.squeeze(), index=index, name="weighted residual"
            ),
            "cov": DataFrame(cov, columns=columns, index=columns),
            "s2": float(squeeze(cov_estimator.s2)),
            "debiased": debiased,
            "residual_ss": float(residual_ss),
            "total_ss": float(total_ss),
            "r2": r2,
            "fstat": fstat,
            "vars": columns,
            "instruments": self._instr_columns,
            "cov_config": cov_estimator.config,
            "cov_type": cov_type,
            "method": self._method,
            "cov_estimator": cov_estimator,
            "fitted": fitted,
            "original_index": self._original_index,
        }

        return out


class _IVLSModelBase(_IVModelBase):
    r"""
    Limited information ML and k-class estimation of IV models.

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like or None
        Exogenous regressors (nobs by nexog). None if the model has no exogenous
        regressors.
    endog : array_like or None
        Endogenous regressors (nobs by nendog). None if the model has no endogenous
        regressors.
    instruments : array_like or None
        Instrumental variables (nobs by ninstr). None if the model has no instruments.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.
    fuller : float, optional
        Fuller's alpha to modify the LIML estimator. The default is 0, which returns
        the unmodified LIML estimator.
    kappa : float, optional
        Parameter value for k-class estimation. If not provided, it is computed to
        produce the LIML parameter estimate.

    Raises
    ------
    ValueError
        If the model has no regressors, if there are fewer instruments than
        endogenous regressors, if ``[exog, endog]`` or ``[exog, instruments]``
        does not have full column rank, if all observations have a missing value,
        if the inputs have different numbers of observations or labels that do
        not match (see Notes), if ``weights`` are not strictly positive and
        finite, or if ``kappa`` or ``fuller`` is not a scalar.
    TypeError
        If an input is complex or cannot be converted to float64.

    See Also
    --------
    IV2SLS : Estimation of IV models using two-stage least squares.
    IVGMM : Estimation of IV models using the generalized method of moments.
    IVGMMCUE : Estimation of IV models using continuously updating GMM.

    Notes
    -----
    ``kappa`` and ``fuller`` should not be used simultaneously since Fuller's
    alpha applies an adjustment to ``kappa``, and so the same result can be
    computed using only ``kappa``. Fuller's alpha is used to adjust the
    LIML estimate of :math:`\kappa`, which is computed whenever ``kappa``
    is not provided.

    The LIML estimator is defined as

    .. math::

      \hat{\beta}_{\kappa} & =(X(I-\kappa M_{z})X)^{-1}X(I-\kappa M_{z})Y\\
      M_{z} & =I-P_{z}\\
      P_{z} & =Z(Z'Z)^{-1}Z'

    where :math:`Z` contains both the exogenous regressors and the instruments.
    :math:`\kappa` is estimated as part of the LIML estimator.

    When using Fuller's :math:`\alpha`, the value used is modified to

    .. math::

      \kappa-\alpha/(n-n_{instr})

    .. todo::

      * VCV: bootstrap
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike | None,
        endog: IVDataLike | None,
        instruments: IVDataLike | None,
        *,
        weights: IVDataLike | None = None,
        fuller: linearmodels.typing.Numeric = 0,
        kappa: linearmodels.typing.OptionalNumeric = None,
    ):
        super().__init__(
            dependent,
            exog,
            endog,
            instruments,
            weights=weights,
            fuller=fuller,
            kappa=kappa,
        )

    @staticmethod
    def estimate_parameters(
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        kappa: linearmodels.typing.Numeric,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Parameter estimation without error checking.

        Parameters
        ----------
        x : ndarray
            Regressor matrix (nobs by nvar).
        y : ndarray
            Regressand matrix (nobs by 1).
        z : ndarray
            Instrument matrix (nobs by ninstr).
        kappa : float
            Parameter value for k-class estimator.

        Returns
        -------
        ndarray
            Estimated parameters (nvar by 1).

        Notes
        -----
        Exposed as a static method to facilitate estimation with other data,
        e.g., bootstrapped samples.  Performs no error checking.
        """
        pinvz = pinv(z)
        p1 = (x.T @ x) * (1 - kappa) + kappa * ((x.T @ z) @ (pinvz @ x))
        p2 = (x.T @ y) * (1 - kappa) + kappa * ((x.T @ z) @ (pinvz @ y))
        return inv(p1) @ p2

    def fit(
        self, *, cov_type: str = "robust", debiased: bool = False, **cov_config: Any
    ) -> OLSResults | IVResults:
        r"""
        Estimate model parameters.

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator to use. The default is "robust". The names
            are not case sensitive. Supported covariance estimators are:

            * "unadjusted", "homoskedastic", "homo", "homoskedasticcovariance" -
              Classic homoskedastic inference
            * "robust", "heteroskedastic", "hccm", "heteroskedasticcovariance" -
              Heteroskedasticity robust inference
            * "kernel", "kernelcovariance" - Heteroskedasticity and autocorrelation
              robust inference
            * "clustered", "one-way", "onewayclusteredcovariance" - Cluster
              dependent inference, which is heteroskedasticity robust. It is
              one-way unless ``clusters`` has two columns.

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is False.
        **cov_config : dict
            Additional parameters to pass to the covariance estimator. The keys that
            are used depend on ``cov_type``. "unadjusted" and "robust" do not use any.
            "kernel" uses ``kernel`` (str), the name of the kernel, which is "bartlett"
            if it is not provided, and ``bandwidth`` (int), which is selected from the
            data if it is not provided. The names of the kernels are not case
            sensitive. "clustered" uses ``clusters`` (array_like), the cluster
            assignment of each observation, with one or two columns. Every
            covariance estimator accepts ``extra_df`` (int), the number of degrees of
            freedom that the model uses in addition to its regressors. It is 0 if it is
            not provided, and it is only used if ``debiased`` is True, when the
            covariance is scaled by n / (n - k - ``extra_df``), where k is the number
            of regressors. The keys ``center`` and ``kappa`` are accepted and ignored:
            ``kappa`` is set by the model. See
            :class:`~linearmodels.iv.covariance.HomoskedasticCovariance`,
            :class:`~linearmodels.iv.covariance.HeteroskedasticCovariance`,
            :class:`~linearmodels.iv.covariance.KernelCovariance` and
            :class:`~linearmodels.iv.covariance.ClusteredCovariance` for the details.

        Returns
        -------
        IVResults or OLSResults
            Results container. It is an OLSResults if the model has no endogenous
            regressors and no instruments.

        Raises
        ------
        ValueError
            If ``kappa`` is None, so that it is estimated, as in
            :class:`~linearmodels.iv.model.IVLIML` by default, and the estimate of
            :math:`\kappa` is not defined. That occurs if there are fewer than 1 +
            the number of endogenous regressors observations more than the number
            of exogenous regressors and instruments, or the residuals of the
            dependent variable and the endogenous regressors on the exogenous
            regressors and instruments are collinear. See the notes of
            :class:`~linearmodels.iv.model.IVLIML`. It is also raised if the
            labels of ``clusters`` do not match the labels of the other inputs.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 raised the ``TypeError`` of the constructor of the covariance
            estimator.
        UnknownOptionError
            If ``cov_type`` or ``kernel`` is not supported. It is both a
            ``KeyError`` and a ``ValueError``.

        See Also
        --------
        linearmodels.iv.covariance.HomoskedasticCovariance : Homoskedastic covariance.
        linearmodels.iv.covariance.HeteroskedasticCovariance : Heteroskedasticity
            robust covariance.
        linearmodels.iv.covariance.KernelCovariance : Kernel (HAC) covariance.
        linearmodels.iv.covariance.ClusteredCovariance : Clustered covariance.

        Notes
        -----
        Additional covariance parameters depend on specific covariance used.
        See the docstring of the specific covariance estimator for a list of
        supported options. Defaults are used if no covariance configuration
        is provided.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _LS_COV_KEYS, KERNEL_LOOKUP)
        self._check_clusters(cov_config)
        wy, wx, wz = self._wy, self._wx, self._wz
        kappa = self._kappa

        try:
            liml_kappa: float = self._estimate_kappa()
        except Exception as exc:
            liml_kappa = nan
            if kappa is None:
                raise ValueError(
                    "Unable to estimate kappa. This most likely occurs if the "
                    f"instrument matrix is rank deficient. The error raised when "
                    f"computing kappa was:\n\n{exc}"
                ) from exc
        if kappa is not None:
            est_kappa = kappa
        else:
            est_kappa = liml_kappa

        if self._fuller != 0:
            nobs, ninstr = wz.shape
            est_kappa -= self._fuller / (nobs - ninstr)

        params = self.estimate_parameters(wx, wy, wz, est_kappa)

        cov_estimator = lookup_option(COVARIANCE_ESTIMATORS, cov_type, "cov_type")
        cov_config["debiased"] = debiased
        cov_config["kappa"] = est_kappa
        cov_config_copy = dict(cov_config.items())
        cov_config_copy.pop("center", None)
        cov_estimator_inst = cov_estimator(wx, wy, wz, params, **cov_config_copy)

        results = {"kappa": est_kappa, "liml_kappa": liml_kappa}
        pe = self._post_estimation(params, cov_estimator_inst, cov_type)
        results.update(pe)

        if self.endog.shape[1] == 0 and self.instruments.shape[1] == 0:
            return OLSResults(results, self)
        else:
            return IVResults(results, self)


class IVLIML(_IVLSModelBase):
    r"""
    Limited information ML and k-class estimation of IV models.

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like or None
        Exogenous regressors (nobs by nexog). None if the model has no exogenous
        regressors.
    endog : array_like or None
        Endogenous regressors (nobs by nendog). None if the model has no endogenous
        regressors.
    instruments : array_like or None
        Instrumental variables (nobs by ninstr). None if the model has no instruments.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.
    fuller : float, optional
        Fuller's alpha to modify the LIML estimator. The default is 0, which returns
        the unmodified LIML estimator.
    kappa : float, optional
        Parameter value for k-class estimation. If not provided, it is computed to
        produce the LIML parameter estimate.

    Raises
    ------
    ValueError
        If the model has no regressors, if there are fewer instruments than
        endogenous regressors, if ``[exog, endog]`` or ``[exog, instruments]``
        does not have full column rank, if all observations have a missing value,
        if the inputs have different numbers of observations or labels that do
        not match (see Notes), if ``weights`` are not strictly positive and
        finite, or if ``kappa`` or ``fuller`` is not a scalar.
    TypeError
        If an input is complex or cannot be converted to float64.

    See Also
    --------
    IV2SLS : Estimation of IV models using two-stage least squares.
    IVGMM : Estimation of IV models using the generalized method of moments.
    IVGMMCUE : Estimation of IV models using continuously updating GMM.

    Notes
    -----
    ``kappa`` and ``fuller`` should not be used simultaneously since Fuller's
    alpha applies an adjustment to ``kappa``, and so the same result can be
    computed using only ``kappa``. Fuller's alpha is used to adjust the
    LIML estimate of :math:`\kappa`, which is computed whenever ``kappa``
    is not provided.

    The LIML estimator is defined as

    .. math::

      \hat{\beta}_{\kappa} & =(X(I-\kappa M_{z})X)^{-1}X(I-\kappa M_{z})Y\\
      M_{z} & =I-P_{z}\\
      P_{z} & =Z(Z'Z)^{-1}Z'

    where :math:`Z` contains both the exogenous regressors and the instruments.
    :math:`\kappa` is estimated as part of the LIML estimator.

    When using Fuller's :math:`\alpha`, the value used is modified to

    .. math::

      \kappa-\alpha/(n-n_{instr})

    :math:`\kappa` is the smallest eigenvalue of the cross product of the
    residuals of the dependent variable and the endogenous regressors on the
    exogenous regressors, relative to the cross product of their residuals on the
    exogenous regressors and the instruments, which is inverted. It is not defined
    if the residuals on the exogenous regressors and the instruments are linearly
    dependent. That occurs if there are fewer than :math:`1 + n_{endog}`
    observations more than the number of exogenous regressors and instruments (for
    example, fewer than 4 observations for a constant, one endogenous regressor and
    one instrument), and if the residuals are collinear for another reason. Whether
    the residuals are collinear does not depend on the units of the variables.
    ``fit`` raises a ``ValueError`` if ``kappa`` is not provided and
    :math:`\kappa` is not defined. The model is estimated if ``kappa`` is provided,
    and then the Anderson-Rubin and Basmann F tests of overidentification of the
    results are an ``InvalidTestStatistic``, since they use the estimate of
    :math:`\kappa`.

    Two cases are not covered, see :issue:`823`. An endogenous regressor that is
    exactly a linear combination of the exogenous regressors and instruments has
    residuals that are only rounding error, and so does :math:`\kappa`. With 2 or
    more endogenous regressors, :math:`\kappa` loses accuracy, and the estimate can
    fail, if the variables are on very different scales.

    .. versionchanged:: 7.1

       A ``ValueError`` is raised if :math:`\kappa` is not defined. Earlier versions
       used the rounding error of the singular cross product, for example 0.8125
       instead of 1 in a just-identified model with 3 observations, a constant, one
       endogenous regressor and one instrument. It was NaN on some platforms, with
       the estimates and their covariance, and a ``RuntimeWarning`` was emitted.

    Rows are paired by position. An input that carries labels (a pandas
    Series or DataFrame with an index other than the default 0, 1, ..., n-1,
    or an xarray DataArray with coordinates) must use the same index, in the
    same order, as ``dependent``, so that a value cannot be attached to the
    wrong observation. A ``ValueError`` is raised if the index of ``exog``,
    ``endog``, ``instruments`` or ``weights`` differs from the index of
    ``dependent``. Either reindex the input to the index of ``dependent``, or
    call ``to_numpy()`` to pair its values by position. NumPy arrays and
    pandas objects with a default index do not carry labels and are always
    paired by position. If ``dependent`` does not carry labels, the inputs
    that do must use the same index as the first of ``exog``, ``endog``,
    ``instruments`` and ``weights`` that does. The same rule applies to
    ``clusters`` passed to ``fit``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.

    .. todo::

      * VCV: bootstrap
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike | None,
        endog: IVDataLike | None,
        instruments: IVDataLike | None,
        *,
        weights: IVDataLike | None = None,
        fuller: linearmodels.typing.Numeric = 0,
        kappa: linearmodels.typing.OptionalNumeric = None,
    ):
        super().__init__(
            dependent,
            exog,
            endog,
            instruments,
            weights=weights,
            fuller=fuller,
            kappa=kappa,
        )

    @staticmethod
    def from_formula(
        formula: str,
        data: DataFrame,
        *,
        weights: IVDataLike | None = None,
        fuller: float = 0,
        kappa: linearmodels.typing.OptionalNumeric = None,
    ) -> IVLIML:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula modified for the IV syntax described in the notes
            section.
        data : DataFrame
            DataFrame containing the variables used in the formula.
        weights : array_like, optional
            Observation weights used in estimation. The weights must be strictly
            positive and finite. They are rescaled to have a mean of 1. A NaN weight is
            missing, and its observation is dropped. The default is to weight all
            observations equally.
        fuller : float, optional
            Fuller's alpha to modify the LIML estimator. The default is 0, which
            returns the unmodified LIML estimator.
        kappa : float, optional
            Parameter value for k-class estimation. If not provided, it is computed
            to produce the LIML parameter estimate.

        Returns
        -------
        IVLIML
            Model instance.

        Raises
        ------
        ValueError
            If the formula is not an IV formula, or if the model that it describes is
            not valid. See the class for the errors that the model raises.

        Notes
        -----
        The IV formula modifies the standard formula syntax to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure is `dependent ~ exog [endog ~ instruments]` and it must
        be the case that the formula expressions constructed from blocks
        `dependent ~ exog endog` and `dependent ~ exog instruments` are both
        valid formulas.

        A constant must be explicitly included using "1 +" if required. Use
        `I(exper ** 2)` for the square of a variable, since `exper ** 2` is
        the power of a set of terms in a formula, which is `exper`.

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IVLIML
        >>> data = wage.load()
        >>> formula = "np.log(wage) ~ 1 + exper + I(exper ** 2) + brthord + [educ ~ sibs]"
        >>> mod = IVLIML.from_formula(formula, data)
        """
        parser = IVFormulaParser(formula, data)
        dep, exog, endog, instr = parser.data
        mod: IVLIML = IVLIML(
            dep, exog, endog, instr, weights=weights, fuller=fuller, kappa=kappa
        )
        mod.formula = formula
        return mod


class IV2SLS(_IVLSModelBase):
    r"""
    Estimation of IV models using two-stage least squares.

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like or None
        Exogenous regressors (nobs by nexog). None if the model has no exogenous
        regressors.
    endog : array_like or None
        Endogenous regressors (nobs by nendog). None if the model has no endogenous
        regressors.
    instruments : array_like or None
        Instrumental variables (nobs by ninstr). None if the model has no instruments.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.

    Raises
    ------
    ValueError
        If the model has no regressors, if there are fewer instruments than
        endogenous regressors, if ``[exog, endog]`` or ``[exog, instruments]``
        does not have full column rank, if all observations have a missing value,
        if the inputs have different numbers of observations or labels that do
        not match (see Notes), or if ``weights`` are not strictly positive and
        finite.
    TypeError
        If an input is complex or cannot be converted to float64.

    See Also
    --------
    IVLIML : Limited information ML and k-class estimation of IV models.
    IVGMM : Estimation of IV models using the generalized method of moments.
    IVGMMCUE : Estimation of IV models using continuously updating GMM.

    Notes
    -----
    The 2SLS estimator is defined

    .. math::

      \hat{\beta}_{2SLS} & =(X'Z(Z'Z)^{-1}Z'X)^{-1}X'Z(Z'Z)^{-1}Z'Y\\
                         & =(\hat{X}'\hat{X})^{-1}\hat{X}'Y\\
                 \hat{X} & =Z(Z'Z)^{-1}Z'X

    The 2SLS estimator is a special case of a k-class estimator with
    :math:`\kappa=1`,

    The estimates do not use the estimate of :math:`\kappa` of LIML, but the
    Anderson-Rubin and Basmann F tests of overidentification of the results do.
    The estimate of :math:`\kappa` is not defined if there are fewer than
    :math:`1 + n_{endog}` observations more than the number of exogenous
    regressors and instruments, or if the residuals of the dependent variable and
    the endogenous regressors on the exogenous regressors and instruments are
    collinear (see :class:`~linearmodels.iv.model.IVLIML`). The model is
    estimated then, and the tests are an ``InvalidTestStatistic``.

    .. versionchanged:: 7.1

       The Anderson-Rubin and Basmann F tests are an ``InvalidTestStatistic`` if
       :math:`\kappa` is not defined. Earlier versions computed them from the
       rounding error of a singular matrix, or the statistics were NaN.

    Rows are paired by position. An input that carries labels (a pandas
    Series or DataFrame with an index other than the default 0, 1, ..., n-1,
    or an xarray DataArray with coordinates) must use the same index, in the
    same order, as ``dependent``, so that a value cannot be attached to the
    wrong observation. A ``ValueError`` is raised if the index of ``exog``,
    ``endog``, ``instruments`` or ``weights`` differs from the index of
    ``dependent``. Either reindex the input to the index of ``dependent``, or
    call ``to_numpy()`` to pair its values by position. NumPy arrays and
    pandas objects with a default index do not carry labels and are always
    paired by position. If ``dependent`` does not carry labels, the inputs
    that do must use the same index as the first of ``exog``, ``endog``,
    ``instruments`` and ``weights`` that does. The same rule applies to
    ``clusters`` passed to ``fit``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.

    .. todo::

      * VCV: bootstrap
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike | None,
        endog: IVDataLike | None,
        instruments: IVDataLike | None,
        *,
        weights: IVDataLike | None = None,
    ):
        self._method = "IV-2SLS"
        super().__init__(
            dependent, exog, endog, instruments, weights=weights, fuller=0, kappa=1
        )

    @staticmethod
    def from_formula(
        formula: str, data: DataFrame, *, weights: IVDataLike | None = None
    ) -> IV2SLS:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula modified for the IV syntax described in the notes
            section.
        data : DataFrame
            DataFrame containing the variables used in the formula.
        weights : array_like, optional
            Observation weights used in estimation. The weights must be strictly
            positive and finite. They are rescaled to have a mean of 1. A NaN weight is
            missing, and its observation is dropped. The default is to weight all
            observations equally.

        Returns
        -------
        IV2SLS
            Model instance.

        Raises
        ------
        ValueError
            If the formula is not an IV formula, or if the model that it describes is
            not valid. See the class for the errors that the model raises.

        Notes
        -----
        The IV formula modifies the standard formula syntax to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure is `dependent ~ exog [endog ~ instruments]` and it must
        be the case that the formula expressions constructed from blocks
        `dependent ~ exog endog` and `dependent ~ exog instruments` are both
        valid formulas.

        A constant must be explicitly included using "1 +" if required. Use
        `I(exper ** 2)` for the square of a variable, since `exper ** 2` is
        the power of a set of terms in a formula, which is `exper`.

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IV2SLS
        >>> data = wage.load()
        >>> formula = 'np.log(wage) ~ 1 + exper + I(exper ** 2) + brthord + [educ ~ sibs]'
        >>> mod = IV2SLS.from_formula(formula, data)
        """
        parser = IVFormulaParser(formula, data)
        dep, exog, endog, instr = parser.data
        mod = IV2SLS(dep, exog, endog, instr, weights=weights)
        mod.formula = formula
        return mod


def _is_singular(cov: linearmodels.typing.data.Float64Array) -> bool:
    """
    Test whether the covariance of the moment conditions is singular

    Parameters
    ----------
    cov : ndarray
        Estimate of the covariance of the moment conditions (ninstr by ninstr)

    Returns
    -------
    bool
        True if the covariance is not finite, has a variance that is not
        positive or its correlation matrix does not have full rank.

    Notes
    -----
    The rank is that of the correlation matrix of the moment conditions, so
    that it does not depend on the units in which the instruments are
    measured. The tolerance of the rank test is relative to the largest
    eigenvalue, and so it treats the covariance as singular if the variances
    of the moment conditions differ by a factor of about 1e16.
    """
    variances = diag(cov)
    if not (npall(isfinite(cov)) and npall(variances > 0)):
        return True
    std = sqrt(variances)
    return bool(matrix_rank(cov / outer(std, std), hermitian=True) < cov.shape[0])


class _IVGMMBase(_IVModelBase):
    r"""
    Estimation of IV models using the generalized method of moments (GMM).

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like or None
        Exogenous regressors (nobs by nexog). None if the model has no exogenous
        regressors.
    endog : array_like or None
        Endogenous regressors (nobs by nendog). None if the model has no endogenous
        regressors.
    instruments : array_like or None
        Instrumental variables (nobs by ninstr). None if the model has no instruments.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.
    weight_type : str, optional
        Name of moment condition weight function to use in the GMM estimation. The
        default is "robust". The names are not case sensitive. The supported weight
        functions are:

        * "unadjusted", "homoskedastic"
        * "robust", "heteroskedastic"
        * "kernel"
        * "clustered", "one-way"

        See Notes for their assumptions.
    **weight_config : dict
        Additional keyword arguments to pass to the moment condition weight
        function. The keys are ``center`` (bool) and ``debiased`` (bool) for every
        weight function, ``kernel`` (str), ``bandwidth`` (int) and ``optimal_bw``
        (bool) for "kernel", and ``clusters`` (array_like) for "clustered".
        The names of the kernels are not case sensitive. ``clusters`` is required
        and has one element for each observation.
        See :class:`~linearmodels.iv.gmm.HomoskedasticWeightMatrix`,
        :class:`~linearmodels.iv.gmm.HeteroskedasticWeightMatrix`,
        :class:`~linearmodels.iv.gmm.KernelWeightMatrix` and
        :class:`~linearmodels.iv.gmm.OneWayClusteredWeightMatrix` for the details.

    Raises
    ------
    ValueError
        If the model has no regressors, if there are fewer instruments than
        endogenous regressors, if ``[exog, endog]`` or ``[exog, instruments]``
        does not have full column rank, if all observations have a missing value,
        if the inputs have different numbers of observations or labels that do
        not match (see Notes), or if ``weights`` are not strictly positive and
        finite.
    UnknownOptionError
        If ``weight_type`` is not supported, or if it is "kernel" and ``kernel``
        is not a supported kernel. It is both a ``KeyError`` and a
        ``ValueError``. Before 7.1, an unsupported ``kernel`` was reported by
        ``fit``.
    TypeError
        If an input is complex or cannot be converted to float64.

    See Also
    --------
    IV2SLS : Estimation of IV models using two-stage least squares.
    IVLIML : Limited information ML and k-class estimation of IV models.
    IVGMMCUE : Estimation of IV models using continuously updating GMM.

    Notes
    -----
    Available GMM weight functions are:

    * "unadjusted", "homoskedastic" - Assumes moment conditions are
      homoskedastic
    * "robust", "heteroskedastic" - Allows for heteroskedasticity but not
      autocorrelation
    * "kernel" - Allows for heteroskedasticity and autocorrelation
    * "clustered", "one-way" - Allows for one-way cluster dependence

    The estimator is defined as

    .. math::

      \hat{\beta}_{gmm}=(X'ZW^{-1}Z'X)^{-1}X'ZW^{-1}Z'Y

    where :math:`W` is a positive definite weight matrix and :math:`Z`
    contains both the exogenous regressors and the instruments.

    .. todo::

      * VCV: bootstrap
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike | None,
        endog: IVDataLike | None,
        instruments: IVDataLike | None,
        *,
        weights: IVDataLike | None = None,
        weight_type: str = "robust",
        **weight_config: Any,
    ):
        super().__init__(dependent, exog, endog, instruments, weights=weights)
        self._method = "IV-GMM"

        weight_type = lower_option(weight_type)
        self._check_clusters(weight_config)
        weight_matrix_estimator = lookup_option(
            WEIGHT_MATRICES, weight_type, "weight_type"
        )
        # Kernel weights without a bandwidth select it from the data. fit warns
        # about the change of this default, and makes the choice explicit so that
        # the weight estimators and models that are created from the arguments
        # do not warn as well
        self._default_weight_bandwidth = (
            weight_matrix_estimator is KernelWeightMatrix
            and _uses_default_bandwidth(weight_config)
        )
        if self._default_weight_bandwidth:
            weight_config = {**weight_config, "optimal_bw": True}
        self._weight = weight_matrix_estimator(**weight_config)
        self._weight_type = weight_type
        self._weight_config = self._weight.config
        # The arguments needed to create another weight estimator of this
        # type. config does not contain all of them, e.g., optimal_bw
        self._weight_kwargs = dict(weight_config)

    def _check_default_bandwidth(
        self, cov_type: str, cov_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Warn once if a kernel estimator in a fit uses the default bandwidth

        The default of the bandwidth of the kernel weight matrix and of the
        kernel covariance is to estimate it from the data since 7.1. It was
        nobs - 2 before.

        Returns
        -------
        dict
            The covariance configuration. If the covariance uses the default
            bandwidth, ``optimal_bw`` is set so that the covariance estimator
            that is created from the configuration does not warn again.
        """
        covariance = cov_type == "kernel" and _uses_default_bandwidth(cov_config)
        weights = self._default_weight_bandwidth
        if weights or covariance:
            # Warn at the call to fit, which is above this method and fit. The
            # example in the message is for the class of the model
            _warn_default_bandwidth(
                weights=weights,
                covariance=covariance,
                stacklevel=4,
                model=type(self).__name__,
            )
        if covariance:
            cov_config = {**cov_config, "optimal_bw": True}
        return cov_config

    def _gmm_post_estimation(
        self,
        params: linearmodels.typing.data.Float64Array,
        weight_mat: linearmodels.typing.data.Float64Array,
        iters: int,
        moment_cov: linearmodels.typing.data.Float64Array | None = None,
    ) -> dict[str, Any]:
        """GMM-specific post-estimation results"""
        instr = self._instr_columns
        gmm_specific = {
            "weight_mat": DataFrame(weight_mat, columns=instr, index=instr),
            "weight_type": self._weight_type,
            "weight_config": self._weight.config,
            "iterations": iters,
            "j_stat": self._j_statistic(params, weight_mat, moment_cov),
            "centered_j_stat": self._centered_j_statistic(params),
        }

        return gmm_specific

    def _centered_j_statistic(
        self, params: linearmodels.typing.data.Float64Array
    ) -> WaldTestStatistic | InvalidTestStatistic:
        """
        J statistic that uses a centered estimate of the moment covariance

        The covariance of the moment conditions is estimated using the same
        estimator as the model, i.e., the one selected by ``weight_type`` and
        ``weight_config``, except that the mean of the moment conditions is
        always subtracted, and it is evaluated at ``params``.
        """
        y, x, z = self._wy, self._wx, self._wz
        nobs, nvar, ninstr = y.shape[0], x.shape[1], z.shape[1]
        name = "Centered J-test of overidentifying restrictions"
        eps = y - x @ params
        g_bar = (z * eps).mean(0)

        weight_matrix_estimator = WEIGHT_MATRICES[self._weight_type]
        kwargs = {**self._weight_kwargs, "center": True}
        weight = self._weight
        if (
            isinstance(weight, KernelWeightMatrix)
            and weight._fixed_bandwidth is not None
        ):
            # A bandwidth that the model estimated once is not estimated again
            kwargs["bandwidth"] = weight._fixed_bandwidth
        centered = weight_matrix_estimator(**kwargs)
        s_c = centered.weight_matrix(x, z, eps)
        if _is_singular(s_c):
            # The centered cluster sums add to zero, so a clustered estimator
            # has rank at most one less than the number of clusters
            reason = (
                "The centered covariance of the moment conditions is singular, "
                "so the statistic is not defined. This occurs, for example, "
                "when there are too few clusters."
            )
            return InvalidTestStatistic(reason, name=name)
        stat = float(nobs * g_bar @ solve(s_c, g_bar))
        null = "Expected moment conditions are equal to 0"
        return WaldTestStatistic(stat, null, ninstr - nvar, name=name)

    def _j_statistic(
        self,
        params: linearmodels.typing.data.Float64Array,
        weight_mat: linearmodels.typing.data.Float64Array,
        moment_cov: linearmodels.typing.data.Float64Array | None = None,
    ) -> WaldTestStatistic | InvalidTestStatistic:
        """
        J stat and test

        moment_cov is the estimate of the covariance of the moment conditions
        that was inverted to compute weight_mat, if there is one. The statistic
        is not defined if it is singular, since weight_mat is then numerical
        noise. The test is applied to the covariance and not to weight_mat,
        because the inverse of a singular covariance is not always found to be
        singular.
        """
        y, x, z = self._wy, self._wx, self._wz
        nobs, nvar, ninstr = y.shape[0], x.shape[1], z.shape[1]
        if moment_cov is not None and _is_singular(moment_cov):
            reason = (
                "The covariance of the moment conditions that is inverted to "
                "compute the weight matrix is singular, so the statistic is "
                "not defined. This occurs, for example, when there are too "
                "few clusters."
            )
            name = "J-test of overidentifying restrictions"
            return InvalidTestStatistic(reason, name=name)
        eps = y - x @ params
        g_bar = (z * eps).mean(0)
        stat = float(nobs * g_bar.T @ weight_mat @ g_bar.T)
        null = "Expected moment conditions are equal to 0"
        return WaldTestStatistic(stat, null, ninstr - nvar)

    def _raise_if_singular(
        self,
        moment_cov: linearmodels.typing.data.Float64Array,
        consequence: str,
    ) -> None:
        """
        Raise a ValueError if the covariance of the moment conditions is singular

        Parameters
        ----------
        moment_cov : ndarray
            Estimate of the covariance of the moment conditions (ninstr by
            ninstr).
        consequence : str
            Sentence that says why the estimator needs the inverse of the
            covariance. It ends with the period.

        Raises
        ------
        ValueError
            If the covariance is singular. The message says that there are too
            few clusters or observations and how many are needed.
        """
        if not _is_singular(moment_cov):
            return
        nobs, nmoment = self._wz.shape
        if isinstance(self._weight, OneWayClusteredWeightMatrix):
            nclusters = len(unique(asarray(self._weight_config["clusters"])))
            centered = self._weight_config["center"]
            rank = nclusters - 1 if centered else nclusters
            rank_reason = (
                "one less than the number of clusters, since the cluster sums of "
                "centered moment conditions add to zero"
                if centered
                else "the number of clusters"
            )
            msg = (
                f"The clustered covariance of the {nmoment} moment conditions is "
                f"singular because there are too few clusters ({nclusters} "
                f"clusters). Its rank is at most {rank}, {rank_reason}. "
                f"{consequence} Use more clusters than moment conditions, or a "
                "weight_type that does not use clusters."
            )
        else:
            msg = (
                f"The covariance of the {nmoment} moment conditions is singular, "
                f"which occurs if there are too few observations ({nobs}) for the "
                f"number of moment conditions. {consequence} Use more "
                "observations or fewer moment conditions."
            )
        raise ValueError(msg)


class IVGMM(_IVGMMBase):
    r"""
    Estimation of IV models using the generalized method of moments (GMM).

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like or None
        Exogenous regressors (nobs by nexog). None if the model has no exogenous
        regressors.
    endog : array_like or None
        Endogenous regressors (nobs by nendog). None if the model has no endogenous
        regressors.
    instruments : array_like or None
        Instrumental variables (nobs by ninstr). None if the model has no instruments.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.
    weight_type : str, optional
        Name of moment condition weight function to use in the GMM estimation. The
        default is "robust". The names are not case sensitive. The supported weight
        functions are:

        * "unadjusted", "homoskedastic"
        * "robust", "heteroskedastic"
        * "kernel"
        * "clustered", "one-way"

        See Notes for their assumptions.
    **weight_config : dict
        Additional keyword arguments to pass to the moment condition weight
        function. The keys are ``center`` (bool) and ``debiased`` (bool) for every
        weight function, ``kernel`` (str), ``bandwidth`` (int) and ``optimal_bw``
        (bool) for "kernel", and ``clusters`` (array_like) for "clustered".
        The names of the kernels are not case sensitive. ``clusters`` is required
        and has one element for each observation.
        See :class:`~linearmodels.iv.gmm.HomoskedasticWeightMatrix`,
        :class:`~linearmodels.iv.gmm.HeteroskedasticWeightMatrix`,
        :class:`~linearmodels.iv.gmm.KernelWeightMatrix` and
        :class:`~linearmodels.iv.gmm.OneWayClusteredWeightMatrix` for the details.

    Raises
    ------
    ValueError
        If the model has no regressors, if there are fewer instruments than
        endogenous regressors, if ``[exog, endog]`` or ``[exog, instruments]``
        does not have full column rank, if all observations have a missing value,
        if the inputs have different numbers of observations or labels that do
        not match (see Notes), or if ``weights`` are not strictly positive and
        finite.
    UnknownOptionError
        If ``weight_type`` is not supported, or if it is "kernel" and ``kernel``
        is not a supported kernel. It is both a ``KeyError`` and a
        ``ValueError``. Before 7.1, an unsupported ``kernel`` was reported by
        ``fit``.
    TypeError
        If an input is complex or cannot be converted to float64.

    See Also
    --------
    IV2SLS : Estimation of IV models using two-stage least squares.
    IVLIML : Limited information ML and k-class estimation of IV models.
    IVGMMCUE : Estimation of IV models using continuously updating GMM.

    Notes
    -----
    Available GMM weight functions are:

    * "unadjusted", "homoskedastic" - Assumes moment conditions are
      homoskedastic
    * "robust", "heteroskedastic" - Allows for heteroskedasticity but not
      autocorrelation
    * "kernel" - Allows for heteroskedasticity and autocorrelation
    * "clustered", "one-way" - Allows for one-way cluster dependence

    The estimator is defined as

    .. math::

      \hat{\beta}_{gmm}=(X'ZW^{-1}Z'X)^{-1}X'ZW^{-1}Z'Y

    where :math:`W` is a positive definite weight matrix and :math:`Z`
    contains both the exogenous regressors and the instruments.

    The bandwidth of a kernel weight matrix (``weight_type="kernel"``) is
    ``bandwidth`` if it is given in ``weight_config``. If neither ``bandwidth``
    nor ``optimal_bw`` is given, it is estimated from the data, as the
    bandwidth of the kernel covariance of :class:`~linearmodels.iv.model.IV2SLS`
    and :class:`~linearmodels.iv.model.IVLIML` is. In 7.0 and earlier it was
    ``nobs - 2``, which is not a consistent estimator, so the estimates,
    standard errors and J statistic can be different. ``fit`` issues a
    ``DefaultBandwidthWarning`` (a ``UserWarning``) when it uses the default.
    To restore the previous behavior, pass ``optimal_bw=False`` or
    ``bandwidth=nobs - 2`` in ``weight_config``, for example
    ``IVGMM(dependent, exog, endog, instruments, weight_type="kernel",
    optimal_bw=False)``. Passing ``bandwidth`` or ``optimal_bw`` silences the
    warning, and ``optimal_bw=True`` selects the bandwidth from the data
    without it. The same holds for ``cov_type="kernel"`` in ``fit``. See
    :class:`~linearmodels.iv.gmm.KernelWeightMatrix` and
    :class:`~linearmodels.iv.gmm.IVGMMCovariance`.

    The bandwidth that is estimated from the data needs at least 4 observations
    with the Bartlett kernel, 5 with the Parzen kernel and 6 with the quadratic
    spectral kernel. ``fit`` raises a ``ValueError`` for fewer observations, and
    if the estimated bandwidth is not finite. Pass ``bandwidth`` or
    ``optimal_bw=False`` to use the kernel with fewer observations. See
    :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.

    Rows are paired by position. An input that carries labels (a pandas
    Series or DataFrame with an index other than the default 0, 1, ..., n-1,
    or an xarray DataArray with coordinates) must use the same index, in the
    same order, as ``dependent``, so that a value cannot be attached to the
    wrong observation. A ``ValueError`` is raised if the index of ``exog``,
    ``endog``, ``instruments`` or ``weights`` differs from the index of
    ``dependent``. Either reindex the input to the index of ``dependent``, or
    call ``to_numpy()`` to pair its values by position. NumPy arrays and
    pandas objects with a default index do not carry labels and are always
    paired by position. If ``dependent`` does not carry labels, the inputs
    that do must use the same index as the first of ``exog``, ``endog``,
    ``instruments`` and ``weights`` that does. The same rule applies to
    ``clusters`` passed to ``fit`` or in ``weight_config``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.

    The weight matrix is the inverse of an estimate of the covariance of the
    moment conditions, so that the estimate has to be nonsingular. A clustered
    estimate from G clusters has rank at most G, and at most G - 1 if the
    moment conditions are centered (``center`` is ``True``), since the
    centered cluster sums add to zero. There have to be at least as many
    clusters as moment conditions, and more if they are centered. The first
    step moment conditions of an exactly identified model add to zero as well,
    so that it needs more clusters than moment conditions with every value of
    ``center``. The rank of a kernel or robust estimate is bounded in the same
    way by the number of observations. ``fit`` raises a ``ValueError`` that
    says so if the estimate is singular, since the estimates would be
    numerical noise. The estimate is not used, and so it is not tested, if
    ``iter_limit`` is 1.

    .. todo::

      * VCV: bootstrap
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike | None,
        endog: IVDataLike | None,
        instruments: IVDataLike | None,
        *,
        weights: IVDataLike | None = None,
        weight_type: str = "robust",
        **weight_config: Any,
    ):
        super().__init__(
            dependent,
            exog,
            endog,
            instruments,
            weights=weights,
            weight_type=weight_type,
            **weight_config,
        )

    @staticmethod
    def from_formula(
        formula: str,
        data: DataFrame,
        *,
        weights: IVDataLike | None = None,
        weight_type: str = "robust",
        **weight_config: Any,
    ) -> IVGMM:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula modified for the IV syntax described in the notes
            section.
        data : DataFrame
            DataFrame containing the variables used in the formula.
        weights : array_like, optional
            Observation weights used in estimation. The weights must be strictly
            positive and finite. They are rescaled to have a mean of 1. A NaN weight is
            missing, and its observation is dropped. The default is to weight all
            observations equally.
        weight_type : str, optional
            Name of moment condition weight function to use in the GMM estimation.
            The default is "robust". The names are not case sensitive. See
            :class:`~linearmodels.iv.model.IVGMM` for the supported weight functions.
        **weight_config : dict
            Additional keyword arguments to pass to the moment condition weight
            function. See :class:`~linearmodels.iv.model.IVGMM` for the keys.

        Returns
        -------
        IVGMM
            Model instance.

        Raises
        ------
        ValueError
            If the formula is not an IV formula, or if the model that it describes is
            not valid. See the class for the errors that the model raises.

        Notes
        -----
        The IV formula modifies the standard formula syntax to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure is `dependent ~ exog [endog ~ instruments]` and it must
        be the case that the formula expressions constructed from blocks
        `dependent ~ exog endog` and `dependent ~ exog instruments` are both
        valid formulas.

        A constant must be explicitly included using "1 +" if required. Use
        `I(exper ** 2)` for the square of a variable, since `exper ** 2` is
        the power of a set of terms in a formula, which is `exper`.

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IVGMM
        >>> data = wage.load()
        >>> formula = "np.log(wage) ~ 1 + exper + I(exper ** 2) + brthord + [educ ~ sibs]"
        >>> mod = IVGMM.from_formula(formula, data)
        """
        mod = _gmm_model_from_formula(
            IVGMM, formula, data, weights, weight_type, **weight_config
        )
        assert isinstance(mod, IVGMM)
        return mod

    @staticmethod
    def estimate_parameters(
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        w: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Parameter estimation without error checking.

        Parameters
        ----------
        x : ndarray
            Regressor matrix (nobs by nvar).
        y : ndarray
            Regressand matrix (nobs by 1).
        z : ndarray
            Instrument matrix (nobs by ninstr).
        w : ndarray
            GMM weight matrix (ninstr by ninstr).

        Returns
        -------
        ndarray
            Estimated parameters (nvar by 1).

        Notes
        -----
        Exposed as a static method to facilitate estimation with other data,
        e.g., bootstrapped samples.  Performs no error checking.
        """
        xpz = x.T @ z
        zpy = z.T @ y
        return inv(xpz @ w @ xpz.T) @ (xpz @ w @ zpy)

    def fit(
        self,
        *,
        iter_limit: int = 2,
        tol: float = 1e-4,
        initial_weight: linearmodels.typing.data.Float64Array | None = None,
        cov_type: str = "robust",
        debiased: bool = False,
        **cov_config: Any,
    ) -> OLSResults | IVGMMResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        iter_limit : int, optional
            Maximum number of iterations.  Default is 2, which produces
            two-step efficient GMM estimates.  Larger values can be used
            to iterate between parameter estimation and optimal weight
            matrix estimation until convergence.
        tol : float, optional
            Convergence criteria.  Measured as covariance normalized change in
            parameters across iterations where the covariance estimator is
            based on the first step parameter estimates. The default is 1e-4.
        initial_weight : array_like, optional
            Initial weighting matrix to use in the first step, with shape
            (nexog + ninstr, nexog + ninstr).  If not specified, uses the inverse of
            the average outer-product of the set containing the exogenous variables
            and instruments.
        cov_type : str, optional
            Name of covariance estimator to use. The default is "robust". The names
            are not case sensitive. Supported covariance estimators are:

            * "unadjusted", "homoskedastic" - Assumes moment conditions are
              homoskedastic
            * "robust", "heteroskedastic" - Allows for heteroskedasticity but
              not autocorrelation
            * "kernel" - Allows for heteroskedasticity and autocorrelation
            * "clustered" - Allows for one-way cluster dependence

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is False.
        **cov_config : dict
            Additional parameters to pass to the covariance estimator. The keys are
            ``center`` (bool) for every covariance estimator, ``kernel`` (str),
            ``bandwidth`` (int) and ``optimal_bw`` (bool) for "kernel", and
            ``clusters`` (array_like) for "clustered". The names of the kernels are
            not case sensitive. See
            :class:`~linearmodels.iv.gmm.IVGMMCovariance` for details on the
            available options. Defaults are used if no covariance configuration is
            provided.

        Returns
        -------
        IVGMMResults
            Results container.

        Raises
        ------
        ValueError
            If the estimate of the covariance of the moment conditions is
            singular, which occurs if there are too few clusters or
            observations. A clustered estimate needs at least as many clusters
            as moment conditions, and more if the moment conditions are
            centered or the model is exactly identified. The estimate is not
            used if ``iter_limit`` is 1. The error is also raised if the
            bandwidth of a kernel weight matrix or of the kernel covariance is
            estimated from the data and there are fewer observations than the
            kernel needs, or the estimated bandwidth is not finite, if
            ``initial_weight`` does not have the right shape, and if the labels
            of ``clusters`` do not match the labels of the other inputs.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 raised the ``TypeError`` of the constructor of the covariance
            estimator.
        UnknownOptionError
            If ``cov_type`` is not supported. It is both a ``KeyError`` and a
            ``ValueError``.

        See Also
        --------
        linearmodels.iv.gmm.IVGMMCovariance : Covariance estimation for GMM models.

        Notes
        -----
        If ``cov_type`` is "kernel" and neither ``bandwidth`` nor
        ``optimal_bw`` is in ``cov_config``, the bandwidth of the covariance
        is estimated from the data. In 7.0 and earlier it was ``nobs - 2``.
        ``fit`` issues a ``DefaultBandwidthWarning`` (a ``UserWarning``) once
        for each call when the covariance, or the kernel weight matrix of the
        model, uses the default bandwidth. To restore the previous covariance,
        pass ``optimal_bw=False`` or ``bandwidth=nobs - 2`` in ``cov_config``,
        for example ``fit(cov_type="kernel", optimal_bw=False)``. Passing
        ``bandwidth`` or ``optimal_bw`` silences the warning, and
        ``optimal_bw=True`` selects the bandwidth from the data without it.

        The bandwidth that is estimated from the data needs at least 4
        observations with the Bartlett kernel, 5 with the Parzen kernel and 6
        with the quadratic spectral kernel. Pass ``bandwidth`` or
        ``optimal_bw=False`` to use the kernel with fewer observations. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _GMM_COV_KEYS, KERNEL_LOOKUP)
        self._check_clusters(cov_config)
        cov_config = self._check_default_bandwidth(cov_type, cov_config)
        wy, wx, wz = self._wy, self._wx, self._wz
        nobs = wy.shape[0]
        weight_matrix = self._weight.weight_matrix

        k_wz = wz.shape[1]
        if initial_weight is not None:
            initial_weight = asarray(initial_weight)
            if initial_weight.ndim != 2 or initial_weight.shape != (k_wz, k_wz):
                raise ValueError(f"initial_weight must be a {k_wz} by {k_wz} array")
        wmat = inv(wz.T @ wz / nobs) if initial_weight is None else initial_weight
        _params = params = self.estimate_parameters(wx, wy, wz, wmat)

        iters, norm = 1, 10 * tol + 1
        vinv = eye(params.shape[0])
        moment_cov: linearmodels.typing.data.Float64Array | None = None
        while iters < iter_limit and norm > tol:
            eps = wy - wx @ params
            moment_cov = weight_matrix(wx, wz, eps)
            self._raise_if_singular(
                moment_cov,
                "The weight matrix of the GMM estimator is the inverse of the "
                "covariance, so it is not defined.",
            )
            wmat = inv(moment_cov)
            params = self.estimate_parameters(wx, wy, wz, wmat)
            delta = params - _params
            if iters == 1:
                xpz = wx.T @ wz / nobs
                v = (xpz @ wmat @ xpz.T) / nobs
                vinv = inv(v)
            _params = params
            norm = float(squeeze(delta.T @ vinv @ delta))
            iters += 1

        cov_config["debiased"] = debiased
        cov_estimator = IVGMMCovariance(
            wx, wy, wz, params, wmat, cov_type, **cov_config
        )

        results = self._post_estimation(params, cov_estimator, cov_type)
        gmm_pe = self._gmm_post_estimation(params, wmat, iters, moment_cov)

        results.update(gmm_pe)

        return IVGMMResults(results, self)

    def _gmm_post_estimation(
        self,
        params: linearmodels.typing.data.Float64Array,
        weight_mat: linearmodels.typing.data.Float64Array,
        iters: int,
        moment_cov: linearmodels.typing.data.Float64Array | None = None,
    ) -> dict[str, Any]:
        """GMM-specific post-estimation results"""
        instr = self._instr_columns
        gmm_specific = {
            "weight_mat": DataFrame(weight_mat, columns=instr, index=instr),
            "weight_type": self._weight_type,
            "weight_config": self._weight.config,
            "iterations": iters,
            "j_stat": self._j_statistic(params, weight_mat, moment_cov),
            "centered_j_stat": self._centered_j_statistic(params),
        }

        return gmm_specific


class IVGMMCUE(_IVGMMBase):
    r"""
    Estimation of IV models using continuously updating GMM.

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like or None
        Exogenous regressors (nobs by nexog). None if the model has no exogenous
        regressors.
    endog : array_like or None
        Endogenous regressors (nobs by nendog). None if the model has no endogenous
        regressors.
    instruments : array_like or None
        Instrumental variables (nobs by ninstr). None if the model has no instruments.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.
    weight_type : str, optional
        Name of moment condition weight function to use in the GMM estimation. The
        default is "robust". The names are not case sensitive. The supported weight
        functions are:

        * "unadjusted", "homoskedastic"
        * "robust", "heteroskedastic"
        * "kernel"
        * "clustered", "one-way"

        See Notes for their assumptions.
    **weight_config : dict
        Additional keyword arguments to pass to the moment condition weight
        function. The keys are ``center`` (bool) and ``debiased`` (bool) for every
        weight function, ``kernel`` (str), ``bandwidth`` (int) and ``optimal_bw``
        (bool) for "kernel", and ``clusters`` (array_like) for "clustered".
        The names of the kernels are not case sensitive. ``clusters`` is required
        and has one element for each observation.
        See :class:`~linearmodels.iv.gmm.HomoskedasticWeightMatrix`,
        :class:`~linearmodels.iv.gmm.HeteroskedasticWeightMatrix`,
        :class:`~linearmodels.iv.gmm.KernelWeightMatrix` and
        :class:`~linearmodels.iv.gmm.OneWayClusteredWeightMatrix` for the details.

    Raises
    ------
    ValueError
        If the model has no regressors, if there are fewer instruments than
        endogenous regressors, if ``[exog, endog]`` or ``[exog, instruments]``
        does not have full column rank, if all observations have a missing value,
        if the inputs have different numbers of observations or labels that do
        not match (see Notes), or if ``weights`` are not strictly positive and
        finite.
    UnknownOptionError
        If ``weight_type`` is not supported, or if it is "kernel" and ``kernel``
        is not a supported kernel. It is both a ``KeyError`` and a
        ``ValueError``. Before 7.1, an unsupported ``kernel`` was reported by
        ``fit``.
    TypeError
        If an input is complex or cannot be converted to float64.

    See Also
    --------
    IV2SLS : Estimation of IV models using two-stage least squares.
    IVLIML : Limited information ML and k-class estimation of IV models.
    IVGMM : Estimation of IV models using the generalized method of moments.

    Notes
    -----
    Available weight functions are:

    * "unadjusted", "homoskedastic" - Assumes moment conditions are
      homoskedastic
    * "robust", "heteroskedastic" - Allows for heteroskedasticity but not
      autocorrelation
    * "kernel" - Allows for heteroskedasticity and autocorrelation
    * "clustered", "one-way" - Allows for one-way cluster dependence

    In most circumstances, the ``center`` weight option should be ``True`` to
    avoid starting value dependence. It is ``True`` unless ``center`` is
    passed in ``weight_config``.

    The covariance of the moment conditions has to be nonsingular. A clustered
    covariance that is estimated from G clusters has rank at most G, and at
    most G - 1 if the moment conditions are centered, so that there have to be
    more clusters than moment conditions if ``center`` is ``True``. If
    ``center`` is ``False`` and there are as many clusters as moment
    conditions, the covariance is nonsingular but the objective function is
    the same for every value of the parameters, so that there is nothing to
    optimize. There have to be more clusters than moment conditions whether or
    not the moment conditions are centered.

    .. math::

      \hat{\beta}_{cue} & =\min_{\beta}\bar{g}(\beta)'W(\beta)^{-1}g(\beta)\\
      g(\beta) & =n^{-1}\sum_{i=1}^{n}z_{i}(y_{i}-x_{i}\beta)

    where :math:`W(\beta)` is a weight matrix that depends on :math:`\beta`
    through :math:`\epsilon_i = y_i - x_i\beta`.

    The bandwidth of a kernel weight matrix (``weight_type="kernel"``) is
    ``bandwidth`` if it is given in ``weight_config``, and ``nobs - 2`` if
    ``optimal_bw`` is ``False``. Otherwise, which is the default and what
    ``optimal_bw=True`` does, it is estimated from the data, as the bandwidth
    of the kernel covariance of :class:`~linearmodels.iv.model.IV2SLS` and
    :class:`~linearmodels.iv.model.IVLIML` is. It is estimated once, from the
    residuals of the first step estimates (the two-stage least squares
    estimates, which are also the first step of :class:`~linearmodels.iv.model.IVGMM`),
    and the same bandwidth is used for the starting values, for every
    evaluation of the objective function, and for the weight matrix, J
    statistic and centered J statistic of the estimates. The objective function
    is smooth in the parameters, as the optimizer needs it to be. A bandwidth
    that is estimated again at every evaluation changes by whole lags with the
    parameters, which makes the objective a step function, and the optimizer
    stopped before the minimum in about one fit in ten in a simulation with
    autocorrelated errors. The bandwidth that was used is in
    :attr:`~linearmodels.iv.results.IVGMMResults.weight_config`.

    In 7.0 and earlier the default bandwidth was ``nobs - 2``, which is not a
    consistent estimator, so the estimates, standard errors and J statistic can
    be different. ``fit`` issues a ``DefaultBandwidthWarning`` (a
    ``UserWarning``) when it uses the default. To restore the previous
    behavior, pass ``optimal_bw=False`` or ``bandwidth=nobs - 2`` in
    ``weight_config``, for example ``IVGMMCUE(dependent, exog, endog,
    instruments, weight_type="kernel", optimal_bw=False)``. Passing
    ``bandwidth`` or ``optimal_bw`` silences the warning. The same holds for
    ``cov_type="kernel"`` in ``fit``, whose bandwidth is estimated from the
    residuals of the final estimates. Before 7.1, ``optimal_bw=True`` estimated
    the bandwidth of the objective function at every evaluation and used
    ``nobs - 2`` for the starting values, so its estimates change slightly. See
    :class:`~linearmodels.iv.gmm.KernelWeightMatrix` and
    :class:`~linearmodels.iv.gmm.IVGMMCovariance`.

    The bandwidth that is estimated from the data needs at least 4 observations
    with the Bartlett kernel, 5 with the Parzen kernel and 6 with the quadratic
    spectral kernel. ``fit`` raises a ``ValueError`` for fewer observations, and
    if the estimated bandwidth is not finite. Pass ``bandwidth`` or
    ``optimal_bw=False`` to use the kernel with fewer observations. See
    :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.

    Rows are paired by position. An input that carries labels (a pandas
    Series or DataFrame with an index other than the default 0, 1, ..., n-1,
    or an xarray DataArray with coordinates) must use the same index, in the
    same order, as ``dependent``, so that a value cannot be attached to the
    wrong observation. A ``ValueError`` is raised if the index of ``exog``,
    ``endog``, ``instruments`` or ``weights`` differs from the index of
    ``dependent``. Either reindex the input to the index of ``dependent``, or
    call ``to_numpy()`` to pair its values by position. NumPy arrays and
    pandas objects with a default index do not carry labels and are always
    paired by position. If ``dependent`` does not carry labels, the inputs
    that do must use the same index as the first of ``exog``, ``endog``,
    ``instruments`` and ``weights`` that does. The same rule applies to
    ``clusters`` passed to ``fit`` or in ``weight_config``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike | None,
        endog: IVDataLike | None,
        instruments: IVDataLike | None,
        *,
        weights: IVDataLike | None = None,
        weight_type: str = "robust",
        **weight_config: Any,
    ) -> None:
        # The default has to be set before the weight estimator is created
        if "center" not in weight_config:
            weight_config["center"] = True
        super().__init__(
            dependent,
            exog,
            endog,
            instruments,
            weights=weights,
            weight_type=weight_type,
            **weight_config,
        )
        # Set after the base class, which sets the method to IV-GMM
        self._method = "IV-GMM-CUE"

    @staticmethod
    def from_formula(
        formula: str,
        data: DataFrame,
        *,
        weights: IVDataLike | None = None,
        weight_type: str = "robust",
        **weight_config: Any,
    ) -> IVGMMCUE:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula modified for the IV syntax described in the notes
            section.
        data : DataFrame
            DataFrame containing the variables used in the formula.
        weights : array_like, optional
            Observation weights used in estimation. The weights must be strictly
            positive and finite. They are rescaled to have a mean of 1. A NaN weight is
            missing, and its observation is dropped. The default is to weight all
            observations equally.
        weight_type : str, optional
            Name of moment condition weight function to use in the GMM estimation.
            The default is "robust". The names are not case sensitive. See
            :class:`~linearmodels.iv.model.IVGMMCUE` for the supported weight
            functions.
        **weight_config : dict
            Additional keyword arguments to pass to the moment condition weight
            function. See :class:`~linearmodels.iv.model.IVGMMCUE` for the keys.

        Returns
        -------
        IVGMMCUE
            Model instance.

        Raises
        ------
        ValueError
            If the formula is not an IV formula, or if the model that it describes is
            not valid. See the class for the errors that the model raises.

        Notes
        -----
        The IV formula modifies the standard formula syntax to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure is `dependent ~ exog [endog ~ instruments]` and it must
        be the case that the formula expressions constructed from blocks
        `dependent ~ exog endog` and `dependent ~ exog instruments` are both
        valid formulas.

        A constant must be explicitly included using "1 +" if required. Use
        `I(exper ** 2)` for the square of a variable, since `exper ** 2` is
        the power of a set of terms in a formula, which is `exper`.

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IVGMMCUE
        >>> data = wage.load()
        >>> formula = "np.log(wage) ~ 1 + exper + I(exper ** 2) + brthord + [educ ~ sibs]"
        >>> mod = IVGMMCUE.from_formula(formula, data)
        """
        mod = _gmm_model_from_formula(
            IVGMMCUE, formula, data, weights, weight_type, **weight_config
        )
        assert isinstance(mod, IVGMMCUE)
        return mod

    def j(
        self,
        params: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
    ) -> float:
        r"""
        Optimization target.

        Parameters
        ----------
        params : ndarray
            Parameter vector (nvar).
        x : ndarray
            Regressor matrix (nobs by nvar).
        y : ndarray
            Regressand matrix (nobs by 1).
        z : ndarray
            Instrument matrix (nobs by ninstr).

        Returns
        -------
        float
            GMM objective function, also known as the J statistic.

        Notes
        -----
        The GMM objective function is defined as

        .. math::

          J(\beta) = \bar{g}(\beta)'W(\beta)^{-1}\bar{g}(\beta)

        where :math:`\bar{g}(\beta)` is the average of the moment
        conditions, :math:`z_i \hat{\epsilon}_i`, where
        :math:`\hat{\epsilon}_i = y_i - x_i\beta`.  The weighting matrix
        is some estimator of the long-run variance of the moment conditions.

        Unlike tradition GMM, the weighting matrix is simultaneously computed
        with the moment conditions, and so has explicit dependence on
        :math:`\beta`.

        If the bandwidth of a kernel weight matrix is estimated from the data,
        it is estimated once, from the residuals of the first step estimates
        of the data of the model, the first time that it is needed, and the
        same bandwidth is used each time that the function is evaluated. The
        function is then smooth in the parameters.
        """
        self._select_bandwidth()
        nobs = y.shape[0]
        weight_matrix = self._weight.weight_matrix
        eps = y - x @ params[:, None]
        w = inv(weight_matrix(x, z, eps))
        g_bar = (z * eps).mean(0)
        return nobs * float(g_bar.T @ w @ g_bar.T)

    def estimate_parameters(
        self,
        starting: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        display: int = 0,
        opt_options: dict[str, Any] | None = None,
    ) -> tuple[linearmodels.typing.data.Float64Array, int]:
        r"""
        Estimate the parameters by minimizing the objective function.

        Parameters
        ----------
        starting : array_like
            Starting values for the optimization.
        x : ndarray
            Regressor matrix (nobs by nvar).
        y : ndarray
            Regressand matrix (nobs by 1).
        z : ndarray
            Instrument matrix (nobs by ninstr).
        display : int, optional
            Number of iterations between displaying. Set to 0 to suppress output.
            The default is 0.
        opt_options : dict, optional
            Dictionary containing additional keyword arguments to pass to
            scipy.optimize.minimize. The default is to pass none.

        Returns
        -------
        params : ndarray
            Estimated parameters (nvar by 1).
        iterations : int
            Number of iterations used by the optimizer.

        See Also
        --------
        scipy.optimize.minimize : Minimization of a scalar function.

        Notes
        -----
        Exposed to facilitate estimation with other data, e.g., bootstrapped
        samples.  Performs no error checking. The bandwidth of a kernel weight
        matrix that is estimated from the data is the one that is estimated
        from the data of the model, see
        :meth:`~linearmodels.iv.model.IVGMMCUE.j`, and not from ``x``, ``y``
        and ``z``.
        """
        args = (x, y, z)
        if opt_options is None:
            opt_options = {}
        assert opt_options is not None
        options = {}
        if "options" in opt_options:
            opt_options = opt_options.copy()
            options.update(opt_options.pop("options"))

        def callback_factory(
            _disp: int,
        ) -> Callable[[ndarray[tuple[int], dtype[float64]]], None]:
            d = {"iter": 0}

            def _callback(params: ndarray[tuple[int], dtype[float64]]) -> None:
                fval = self.j(params, *args)
                if _disp > 0 and (d["iter"] % _disp == 0):
                    print("Iteration: {}, Objective: {}".format(d["iter"], fval))
                d["iter"] += 1

            return _callback

        callback = callback_factory(display)

        res = minimize(
            self.j,
            starting,
            args=args,
            options=options,
            callback=callback,
            **opt_options,
        )

        return res.x[:, None], res.nit

    def _first_step_residuals(self) -> linearmodels.typing.data.Float64Array:
        """
        Residuals of the first step estimates of two-step GMM

        The first step uses the weight matrix (Z'Z / n)^{-1}, which gives the
        two-stage least squares estimates. They are also the first step of the
        IVGMM estimates that are the starting values.
        """
        wy, wx, wz = self._wy, self._wx, self._wz
        nobs = wz.shape[0]
        params = IVGMM.estimate_parameters(wx, wy, wz, inv(wz.T @ wz / nobs))
        return wy - wx @ params

    def _select_bandwidth(self) -> None:
        """
        Estimate the bandwidth of a kernel weight matrix once

        If the bandwidth of the weight matrix is estimated from the data, it is
        estimated from the first step residuals the first time that this is
        called, and the weight matrix uses it from then on. The model has the
        same bandwidth in the starting values, in the objective function, which
        is then smooth in the parameters, in the weight matrix of the estimates
        and in the centered J statistic. A bandwidth that is given, or that is
        nobs - 2 if optimal_bw is False, is not changed.
        """
        weight = self._weight
        if (
            isinstance(weight, KernelWeightMatrix)
            and weight._estimates_bandwidth
            and weight._fixed_bandwidth is None
        ):
            weight._fix_bandwidth(self._wz, self._first_step_residuals())

    def _check_moment_covariance(self) -> None:
        """
        Raise a ValueError if the objective function is not defined or constant

        The objective function uses the inverse of the covariance of the moment
        conditions, so it is not defined if the covariance is singular. The rank
        of the covariance does not depend on the parameters, other than for
        special values such as residuals that are all 0, and so it is tested at
        the first step estimates of two-step GMM.

        If the covariance is clustered, the moment conditions are not centered
        and there are as many clusters as moment conditions, the covariance is
        not singular but the objective function is the same for all values of
        the parameters. The cluster sums of the moment conditions are the rows
        of a square matrix, Xi, the average of the moment conditions is
        Xi'1 / n, and the covariance is Xi'Xi / n, so that the objective function
        is 1'Xi(Xi'Xi)^{-1}Xi'1, which is the number of clusters whenever Xi is
        nonsingular (it is divided by the small sample adjustment if debiased).
        """
        wx, wz = self._wx, self._wz
        nmoment = wz.shape[1]
        self._raise_if_singular(
            self._weight.weight_matrix(wx, wz, self._first_step_residuals()),
            "The objective function of the continuously updating GMM estimator "
            "uses the inverse of the covariance, so it is not defined.",
        )
        if not isinstance(self._weight, OneWayClusteredWeightMatrix):
            return
        nclusters = len(unique(asarray(self._weight_config["clusters"])))
        # Fewer clusters, or centered moment conditions, give a singular covariance
        if nclusters > nmoment:
            return
        msg = (
            f"The clustered covariance of the {nmoment} moment conditions is "
            f"estimated from as many clusters ({nclusters} clusters) as there "
            "are moment conditions, and the moment conditions are not centered. "
            "The objective function of the continuously updating GMM estimator "
            "is then the same for every value of the parameters, so that there is "
            "nothing to optimize and the estimates would be the starting values. "
            "Use more clusters than moment conditions, or a weight_type that "
            "does not use clusters."
        )
        raise ValueError(msg)

    def fit(
        self,
        *,
        starting: linearmodels.typing.data.Float64Array | Series | None = None,
        display: bool = False,
        cov_type: str = "robust",
        debiased: bool = False,
        opt_options: dict[str, Any] | None = None,
        **cov_config: Any,
    ) -> OLSResults | IVGMMResults:
        r"""
        Estimate model parameters.

        Parameters
        ----------
        starting : array_like, optional
            Starting values to use in optimization, with one value for each
            regressor. If not provided, the estimates of IVGMM are used. See Notes.
        display : bool, optional
            Flag indicating whether to display optimization output. The default is
            False.
        cov_type : str, optional
            Name of covariance estimator to use. The default is "robust". The names
            are not case sensitive. Supported covariance estimators are:

            * "unadjusted", "homoskedastic" - Assumes moment conditions are
              homoskedastic
            * "robust", "heteroskedastic" - Allows for heteroskedasticity but
              not autocorrelation
            * "kernel" - Allows for heteroskedasticity and autocorrelation
            * "clustered" - Allows for one-way cluster dependence

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is False.
        opt_options : dict, optional
            Additional options to pass to scipy.optimize.minimize when
            optimizing the objective function. If not provided, defers to
            scipy to choose an appropriate optimizer. All minimize inputs
            except ``fun``, ``x0``, and ``args`` can be overridden.
        **cov_config : dict
            Additional parameters to pass to the covariance estimator. The keys are
            ``center`` (bool) for every covariance estimator, ``kernel`` (str),
            ``bandwidth`` (int) and ``optimal_bw`` (bool) for "kernel", and
            ``clusters`` (array_like) for "clustered". The names of the kernels are
            not case sensitive. See
            :class:`~linearmodels.iv.gmm.IVGMMCovariance` for details on the
            available options. Defaults are used if no covariance configuration is
            provided.

        Returns
        -------
        IVGMMResults
            Results container.

        Raises
        ------
        ValueError
            If the covariance of the moment conditions is singular, which
            occurs if there are too few clusters or observations, or if the
            moment conditions are not centered and there are as many clusters
            as moment conditions, so that the objective function is constant.
            The number of clusters has to be larger than the number of moment
            conditions. The error is also raised if the bandwidth of a kernel
            weight matrix or of the kernel covariance is estimated from the data
            and there are fewer observations than the kernel needs, or the
            estimated bandwidth is not finite, if ``starting`` does not have one
            value for each regressor, and if the labels of ``clusters`` do not match
            the labels of the other inputs.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 raised the ``TypeError`` of the constructor of the covariance
            estimator.
        UnknownOptionError
            If ``cov_type`` is not supported. It is both a ``KeyError`` and a
            ``ValueError``.

        See Also
        --------
        linearmodels.iv.gmm.IVGMMCovariance : Covariance estimation for GMM models.

        Notes
        -----
        Starting values are computed by IVGMM with the weight type and
        configuration of the model. A kernel weight matrix has the bandwidth
        of the objective function: ``bandwidth`` if it was given and
        ``nobs - 2`` if ``optimal_bw`` is False. Otherwise the bandwidth is
        estimated from the data, once, from the residuals of the first step
        estimates, which are the first step of IVGMM too. The same bandwidth
        is used for the starting values, for every evaluation of the objective
        function and for the weight matrix of the estimates, so that the
        objective function is smooth in the parameters. In 7.0 and earlier
        the starting values used ``nobs - 2`` whenever ``bandwidth`` was not
        given, and the bandwidth that was estimated from the data was estimated
        again at every evaluation of the objective function.

        If ``cov_type`` is "kernel" and neither ``bandwidth`` nor
        ``optimal_bw`` is in ``cov_config``, the bandwidth of the covariance
        is estimated from the data, from the residuals of the final
        estimates. In 7.0 and earlier it was ``nobs - 2``.
        ``fit`` issues a ``DefaultBandwidthWarning`` (a ``UserWarning``) once
        for each call when the covariance, or the kernel weight matrix of the
        model, uses the default bandwidth. To restore the previous covariance,
        pass ``optimal_bw=False`` or ``bandwidth=nobs - 2`` in ``cov_config``,
        for example ``fit(cov_type="kernel", optimal_bw=False)``. Passing
        ``bandwidth`` or ``optimal_bw`` silences the warning, and
        ``optimal_bw=True`` selects the bandwidth from the data without it.

        The bandwidth that is estimated from the data needs at least 4
        observations with the Bartlett kernel, 5 with the Parzen kernel and 6
        with the quadratic spectral kernel. Pass ``bandwidth`` or
        ``optimal_bw=False`` to use the kernel with fewer observations. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _GMM_COV_KEYS, KERNEL_LOOKUP)
        self._check_clusters(cov_config)
        cov_config = self._check_default_bandwidth(cov_type, cov_config)
        self._select_bandwidth()
        wy, wx, wz = self._wy, self._wx, self._wz
        weight_matrix = self._weight.weight_matrix
        self._check_moment_covariance()
        if starting is None:
            exog = None if self.exog.shape[1] == 0 else self.exog
            endog = None if self.endog.shape[1] == 0 else self.endog
            instr = None if self.instruments.shape[1] == 0 else self.instruments

            weight_config = dict(self._weight_config)
            weight = self._weight
            if isinstance(weight, KernelWeightMatrix):
                # The starting values use the bandwidth of the objective function.
                # Being explicit keeps IVGMM from selecting a bandwidth by itself,
                # and from warning about the default
                if weight._fixed_bandwidth is None:
                    # The bandwidth that was given, or nobs - 2
                    weight_config["optimal_bw"] = False
                else:
                    weight_config["bandwidth"] = weight._fixed_bandwidth
            res = IVGMM(
                self.dependent,
                exog,
                endog,
                instr,
                weights=self.weights,
                weight_type=self._weight_type,
                **weight_config,
            ).fit()
            starting = asarray(res.params)
        else:
            starting = asarray(starting)
            if len(starting) != self.exog.shape[1] + self.endog.shape[1]:
                raise ValueError("starting does not have the correct number of values")
        params, iters = self.estimate_parameters(
            starting, wx, wy, wz, display, opt_options=opt_options
        )
        eps = wy - wx @ params
        moment_cov = weight_matrix(wx, wz, eps)
        wmat = inv(moment_cov)

        cov_config["debiased"] = debiased
        cov_estimator = IVGMMCovariance(
            wx, wy, wz, params, wmat, cov_type, **cov_config
        )
        results = self._post_estimation(params, cov_estimator, cov_type)
        gmm_pe = self._gmm_post_estimation(params, wmat, iters, moment_cov)
        results.update(gmm_pe)

        return IVGMMResults(results, self)


class _OLS(IVLIML):
    """
    Compute OLS estimates when required.

    Private class used when model reduces to OLS. Should use the statsmodels
    version when needing a supported public API.

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like
        Exogenous regressors (nobs by nexog).
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.

    See Also
    --------
    statsmodels.regression.linear_model.OLS : Ordinary least squares.
    statsmodels.regression.linear_model.GLS : Generalized least squares.

    Notes
    -----
    Uses IV2SLS internally by setting endog and instruments to None.
    Uses IVLIML with kappa=0 to estimate OLS models.

    Rows are paired by position. An input that carries labels (a pandas
    Series or DataFrame with an index other than the default 0, 1, ..., n-1,
    or an xarray DataArray with coordinates) must use the same index, in the
    same order, as ``dependent``, so that a value cannot be attached to the
    wrong observation. A ``ValueError`` is raised if the index of ``exog`` or
    ``weights`` differs from the index of ``dependent``. Either reindex the
    input to the index of ``dependent``, or call ``to_numpy()`` to pair its
    values by position. NumPy arrays and pandas objects with a default index
    do not carry labels and are always paired by position. If ``dependent``
    does not carry labels, the inputs that do must use the same index as the
    first of ``exog`` and ``weights`` that does. The same rule applies to
    ``clusters`` passed to ``fit``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike,
        *,
        weights: IVDataLike | None = None,
    ):
        super().__init__(dependent, exog, None, None, weights=weights, kappa=0.0)
        self._result_container = OLSResults


def _gmm_model_from_formula(
    cls: type[IVGMM | IVGMMCUE],
    formula: str,
    data: DataFrame,
    weights: IVDataLike | None,
    weight_type: str,
    **weight_config: Any,
) -> IVGMM | IVGMMCUE:
    """
    Create a GMM model from a formula.

    Parameters
    ----------
    cls : type
        The class of the model, IVGMM or IVGMMCUE.
    formula : str
        Formula modified for the IV syntax described in the notes
        section.
    data : DataFrame
        DataFrame containing the variables used in the formula.
    weights : array_like or None
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped.
    weight_type : str
        Name of moment condition weight function to use in the GMM estimation.
    **weight_config : dict
        Additional keyword arguments to pass to the moment condition weight
        function.

    Returns
    -------
    IVGMM or IVGMMCUE
        Model instance.
    """
    parser = IVFormulaParser(formula, data, eval_env=3)
    dep, exog, endog, instr = parser.data
    mod = cls(
        dep,
        exog,
        endog,
        instr,
        weights=weights,
        weight_type=weight_type,
        **weight_config,
    )
    mod.formula = formula
    return mod


class _UncenteredHomoskedasticWeightMatrix(HomoskedasticWeightMatrix):
    r"""
    Homoskedastic covariance of the moment conditions, without centering

    The homoskedastic covariance estimators of the IV models use the uncentered
    variance :math:`\hat{\epsilon}'\hat{\epsilon}/n` of the residuals.
    HomoskedasticWeightMatrix always subtracts the mean of the residuals, which
    only has no effect if the mean is zero. The mean of the weighted residuals
    is not zero in a weighted model that has a constant.
    """

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute the covariance of the moment conditions.

        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar).
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr).
        eps : ndarray
            Model errors (nobs by 1).

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape
        s2 = float(squeeze(eps.T @ eps)) / nobs
        w = s2 * z.T @ z / nobs
        w *= 1 if not self._debiased else nobs / (nobs - nvar)
        return w


class _JIVECovariance(IVGMMCovariance):
    r"""
    Covariance of the JIVE estimator

    JIVE is the IV estimator that uses the leave-one-out first-stage
    predictions :math:`\tilde{X}` as the instruments, and
    :math:`\hat\beta=(\tilde{X}'X)^{-1}\tilde{X}'y`. This is a just identified
    GMM estimator with moment conditions :math:`\tilde{x}_i\epsilon_i`, so the
    covariance is

    .. math::

        n^{-1}(\tilde{X}'X/n)^{-1} S (X'\tilde{X}/n)^{-1}

    where :math:`S` is an estimate of the covariance of
    :math:`\tilde{x}_i\hat\epsilon_i`, and :math:`\hat\epsilon_i` is the
    residual computed with :math:`X` and not :math:`\tilde{X}`. The choices of
    :math:`S` are the same as in the other IV models.

    Parameters
    ----------
    x : ndarray
        Weighted regressors (nobs by nvar)
    y : ndarray
        Weighted dependent variable (nobs by 1)
    x_loo : ndarray
        Weighted leave-one-out first-stage predictions (nobs by nvar)
    params : ndarray
        Estimated model parameters (nvar by 1)
    cov_type : str
        Covariance estimator to use. See IVJIVE.fit for the choices.
    debiased : bool
        Flag indicating whether to debias the covariance estimator
    cov_config
        Optional keyword arguments that are specific to a particular cov_type
    """

    def __init__(
        self,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        x_loo: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        cov_type: str = "robust",
        debiased: bool = False,
        **cov_config: str | bool,
    ) -> None:
        # The weight matrix has no effect in a just identified model
        w = eye(x.shape[1])
        if cov_type == "kernel":
            kernel = lower_option(cov_config.get("kernel", "bartlett"))
            if kernel not in KERNEL_LOOKUP:
                raise unknown_option_error("kernel", kernel, KERNEL_LOOKUP)
            # Select the bandwidth using the data unless it is given, as in
            # IV2SLS and IVLIML. This is the default of the weight matrix
            # estimator since 7.1, and it is explicit here so that it does not
            # warn about the change. The default was nobs - 2 before, which is
            # not a consistent covariance estimator
            cov_config = {"optimal_bw": True, **cov_config}
        elif cov_type == "clustered":
            clusters = cov_config.get("clusters")
            if clusters is not None and asarray(clusters).squeeze().ndim > 1:
                raise ValueError(
                    "IVJIVE supports one-way clustering only. clusters must be "
                    "a 1-d array, or a 2-d array with a single column."
                )
        super().__init__(x, y, x_loo, params, w, cov_type, debiased, **cov_config)
        self._name = "JIVE Covariance"
        if cov_type in ("unadjusted", "homoskedastic"):
            self._score_cov_estimator = _UncenteredHomoskedasticWeightMatrix

    @property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Return the covariance of the estimated parameters."""
        # Not the GMM covariance with an identity weight matrix, which forms
        # the inverse of (X'X~)(X~'X) and squares the condition number. That is
        # not accurate when the regressors have different scales
        x, x_loo, eps = self.x, self.z, self.eps
        nobs = x.shape[0]
        score_cov = self._score_cov_estimator(
            debiased=self.debiased, **self._cov_config
        )
        s = score_cov.weight_matrix(x, x_loo, eps)
        self._cov_config = score_cov.config
        bread = inv(x_loo.T @ x / nobs)
        c = bread @ s @ bread.T / nobs
        return (c + c.T) / 2


class IVJIVE(_IVModelBase):
    r"""
    Jackknife instrumental variables estimator (JIVE).

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like or None
        Exogenous regressors (nobs by nexog). None if the model has no exogenous
        regressors.
    endog : array_like or None
        Endogenous regressors (nobs by nendog). None if the model has no endogenous
        regressors.
    instruments : array_like or None
        Instrumental variables (nobs by ninstr). None if the model has no instruments.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.

    Raises
    ------
    ValueError
        If the model has no regressors, if there are fewer instruments than
        endogenous regressors, if ``[exog, endog]`` or ``[exog, instruments]``
        does not have full column rank, if all observations have a missing value,
        if the inputs have different numbers of observations or labels that do
        not match (see Notes), or if ``weights`` are not strictly positive and
        finite.
    TypeError
        If an input is complex or cannot be converted to float64.

    See Also
    --------
    IV2SLS : Estimation of IV models using two-stage least squares.
    IVLIML : Limited information ML and k-class estimation of IV models.
    IVGMM : Estimation of IV models using the generalized method of moments.
    IVGMMCUE : Estimation of IV models using continuously updating GMM.

    Notes
    -----
    JIVE replaces the 2SLS first-stage fitted values :math:`\hat{X}=P_ZX` with
    leave-one-out predictions. The prediction for observation :math:`i` is the
    fitted value from a first stage regression that does not use observation
    :math:`i`

    .. math::

        \tilde{x}_i = \frac{\hat{x}_i - h_i x_i}{1 - h_i},
        \qquad h_i = z_i'(Z'Z)^{-1}z_i,

    where :math:`Z` contains the exogenous regressors and the instruments and
    :math:`h_i` is the leverage of observation :math:`i`. The estimator is

    .. math::

        \hat{\beta}_{JIVE} = (\tilde{X}'X)^{-1} \tilde{X}'y.

    This is the estimator that is called JIVE1 by Angrist, Imbens and Krueger
    (1999). Exogenous regressors are in :math:`Z`, so they are their own
    leave-one-out predictions. In a weighted model the estimator is applied to
    the data that has been multiplied by the square root of the weights.

    The leverages are used to compute :math:`\tilde{X}` without estimating
    :math:`n` regressions, using a QR decomposition of :math:`Z`. The estimator
    cannot be computed if an observation has a leverage of 1, which happens if
    it is the only observation that has a non-zero value of one of the
    instruments, e.g., a dummy variable that is 1 for a single observation.

    The covariance estimators are the same as in the other IV models, and are
    computed using :math:`\tilde{X}` as the instruments. See
    :meth:`~linearmodels.iv.model.IVJIVE.fit`. These are valid when the number
    of instruments is small relative to the strength of the first stage. They
    can understate the uncertainty when there are many instruments (see
    Chao, Swanson, Hausman, Newey and Woutersen, 2012).

    JIVE reduces the many instruments bias of 2SLS, but it can have a larger
    variance, particularly when the instruments are weak (see Davidson and
    MacKinnon, 2006).

    The estimates do not use the estimate of :math:`\kappa` of LIML, but the
    Anderson-Rubin and Basmann F tests of overidentification of the results do.
    The estimate of :math:`\kappa` is not defined if there are fewer than
    :math:`1 + n_{endog}` observations more than the number of exogenous
    regressors and instruments, or if the residuals of the dependent variable and
    the endogenous regressors on the exogenous regressors and instruments are
    collinear (see :class:`~linearmodels.iv.model.IVLIML`). The model is
    estimated then, and the tests are an ``InvalidTestStatistic``.

    .. versionchanged:: 7.1

       The Anderson-Rubin and Basmann F tests are an ``InvalidTestStatistic`` if
       :math:`\kappa` is not defined. Earlier versions computed them from the
       rounding error of a singular matrix, or the statistics were NaN.

    Rows are paired by position. An input that carries labels (a pandas
    Series or DataFrame with an index other than the default 0, 1, ..., n-1,
    or an xarray DataArray with coordinates) must use the same index, in the
    same order, as ``dependent``, so that a value cannot be attached to the
    wrong observation. A ``ValueError`` is raised if the index of ``exog``,
    ``endog``, ``instruments`` or ``weights`` differs from the index of
    ``dependent``. Either reindex the input to the index of ``dependent``, or
    call ``to_numpy()`` to pair its values by position. NumPy arrays and
    pandas objects with a default index do not carry labels and are always
    paired by position. If ``dependent`` does not carry labels, the inputs
    that do must use the same index as the first of ``exog``, ``endog``,
    ``instruments`` and ``weights`` that does. The same rule applies to
    ``clusters`` passed to ``fit``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.

    References
    ----------
    Angrist, J. D., Imbens, G. W., & Krueger, A. B. (1999). Jackknife
    instrumental variables estimation. *Journal of Applied Econometrics*,
    14(1), 57-67.

    Chao, J. C., Swanson, N. R., Hausman, J. A., Newey, W. K., & Woutersen, T.
    (2012). Asymptotic distribution of JIVE in a heteroskedastic IV regression
    with many instruments. *Econometric Theory*, 28(1), 42-86.

    Davidson, R., & MacKinnon, J. G. (2006). The case against JIVE. *Journal of
    Applied Econometrics*, 21(6), 827-833.

    Examples
    --------
    >>> import numpy as np
    >>> from linearmodels.iv import IVJIVE
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> z = rng.standard_normal((n, 3))
    >>> x = z[:, 0] + rng.standard_normal(n)
    >>> exog = np.ones((n, 1))
    >>> y = x + rng.standard_normal(n)
    >>> mod = IVJIVE(y, exog, x[:, None], z)
    >>> res = mod.fit()
    """

    def __init__(
        self,
        dependent: IVDataLike,
        exog: IVDataLike | None,
        endog: IVDataLike | None,
        instruments: IVDataLike | None,
        *,
        weights: IVDataLike | None = None,
    ) -> None:
        self._method = "IV-JIVE"
        super().__init__(dependent, exog, endog, instruments, weights=weights)

    @staticmethod
    def from_formula(
        formula: str, data: DataFrame, *, weights: IVDataLike | None = None
    ) -> IVJIVE:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula modified for the IV syntax described in the notes section.
        data : DataFrame
            DataFrame containing the variables used in the formula.
        weights : array_like, optional
            Observation weights used in estimation. The weights must be strictly
            positive and finite. They are rescaled to have a mean of 1. A NaN weight is
            missing, and its observation is dropped. The default is to weight all
            observations equally.

        Returns
        -------
        IVJIVE
            Model instance.

        Raises
        ------
        ValueError
            If the formula is not an IV formula, or if the model that it describes is
            not valid. See the class for the errors that the model raises.

        Notes
        -----
        The IV formula modifies the standard formula syntax to include a
        block of the form [endog ~ instruments] which is used to indicate
        the list of endogenous variables and instruments.  The general
        structure of an IV formula is `dependent ~ exog [endog ~ instruments]`
        and it must be the case that the formula expressions constructed from
        blocks `dependent ~ exog endog` and `dependent ~ exog instruments` are
        both valid formulas.

        A constant must be explicitly included using "1 +" if required. Use
        `I(exper ** 2)` for the square of a variable, since `exper ** 2` is
        the power of a set of terms in a formula, which is `exper`.

        Examples
        --------
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IVJIVE
        >>> data = wage.load()
        >>> formula = "np.log(wage) ~ 1 + exper + I(exper ** 2) + brthord + [educ ~ sibs]"
        >>> mod = IVJIVE.from_formula(formula, data)
        """
        parser = IVFormulaParser(formula, data)
        dep, exog, endog, instr = parser.data
        mod = IVJIVE(dep, exog, endog, instr, weights=weights)
        mod.formula = formula
        return mod

    def fit(
        self, *, cov_type: str = "robust", debiased: bool = False, **cov_config: Any
    ) -> IVResults:
        r"""
        Estimate model parameters.

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator to use. The default is "robust". The names
            are not case sensitive. Supported covariance estimators are:

            * "unadjusted", "homoskedastic" - Classic homoskedastic inference
            * "robust", "heteroskedastic" - Heteroskedasticity robust inference
            * "kernel" - Heteroskedasticity and autocorrelation robust
              inference
            * "clustered" - One-way cluster dependent inference.
              Heteroskedasticity robust

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is False.
        **cov_config : dict
            Additional parameters to pass to the covariance estimator. The keys are
            ``center`` (bool) for every covariance estimator, ``kernel`` (str),
            ``bandwidth`` (int) and ``optimal_bw`` (bool) for "kernel", and
            ``clusters`` (array_like, one column) for "clustered". The names of the
            kernels are not case sensitive. See the Notes.

        Returns
        -------
        IVResults
            Results container.

        Raises
        ------
        ValueError
            If JIVE cannot be computed because an observation has a leverage of 1,
            if the bandwidth of the kernel is estimated from the data and there are
            fewer observations than the kernel needs, if ``clusters`` has more than
            one column, or if the labels of ``clusters`` do not match the labels of
            the other inputs.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 raised the ``TypeError`` of the constructor of the covariance
            estimator.
        UnknownOptionError
            If ``cov_type`` or ``kernel`` is not supported. It is both a
            ``KeyError`` and a ``ValueError``.

        See Also
        --------
        linearmodels.iv.gmm.IVGMMCovariance : Covariance estimation for GMM models.

        Notes
        -----
        The covariance estimators are the same as in the other IV models with
        the leave-one-out first-stage predictions as the instruments. The
        additional parameters depend on the estimator: ``kernel`` (the default
        is "bartlett") and ``bandwidth`` for "kernel" and ``clusters`` for
        "clustered". If ``bandwidth`` is not provided it is selected using the
        data, as in :class:`~linearmodels.iv.model.IV2SLS`, and the bandwidth
        that was used is in ``cov_config`` of the results. The selection needs
        at least 4 observations with the Bartlett kernel, 5 with the Parzen
        kernel and 6 with the quadratic spectral kernel, and a ``ValueError`` is
        raised for fewer. "clustered" is one-way clustering, and ``clusters``
        has one element for each observation.

        Since JIVE is not a k-class estimator, ``kappa`` is None in the
        results. The estimate of :math:`\kappa` of LIML is only used by the
        Anderson-Rubin and Basmann F tests of overidentification, which are an
        ``InvalidTestStatistic`` if it is not defined. The model is estimated
        then. See the notes of the class.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _GMM_COV_KEYS, KERNEL_LOOKUP)
        self._check_clusters(cov_config)
        wy, wx, wz = self._wy, self._wx, self._wz

        # The leverages and the first-stage predictions from an orthonormal
        # basis of Z. This is much more accurate than forming (Z'Z)^{-1} when
        # the instruments are highly correlated, which is common with many
        # instruments
        q, _ = qr(wz)
        leverage = (q * q).sum(axis=1)
        if npany(leverage >= 1.0 - 1e-10):
            nbad = int((leverage >= 1.0 - 1e-10).sum())
            raise ValueError(
                f"JIVE cannot be computed since {nbad} observation(s) have a "
                "leverage of 1. This happens when an observation is the only "
                "one with a non-zero value of one of the instruments, e.g., a "
                "dummy variable that is 1 for a single observation."
            )

        xhat = q @ (q.T @ wx)
        x_loo = (xhat - leverage[:, None] * wx) / (1.0 - leverage)[:, None]
        # Exogenous regressors are in Z and so are their own predictions
        is_exog = self._regressor_is_exog
        x_loo[:, is_exog] = wx[:, is_exog]

        params = solve(x_loo.T @ wx, x_loo.T @ wy)

        cov_estimator = _JIVECovariance(
            wx, wy, x_loo, params, cov_type, debiased, **cov_config
        )
        results = self._post_estimation(params, cov_estimator, cov_type)
        try:
            liml_kappa = self._estimate_kappa()
        except LinAlgError:
            # Only needed for two tests of overidentification
            liml_kappa = nan
        results["liml_kappa"] = liml_kappa
        # JIVE is not a k-class estimator
        results["kappa"] = None
        return IVResults(results, self)
