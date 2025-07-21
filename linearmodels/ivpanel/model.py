from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple, Union, cast

from formulaic.formula import Formula
from formulaic.model_spec import NAAction
from formulaic.parser.algos.tokenize import tokenize
from formulaic.utils.context import capture_context
import numpy as np
from numpy import all as npall, any as npany
from numpy import array, c_, isscalar, ones, nanmean, sqrt
import pandas as pd
from pandas import Categorical, DataFrame, Index, MultiIndex, Series, get_dummies
from scipy.linalg import lstsq as sp_lstsq
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import lsmr

from .data import IVPanelData, IVPanelDataLike
from .covariance import HomoskedasticCovariance, ClusteredCovariance, CovarianceManager, setup_covariance_estimator
from .results import (
    # FamaMacBethResults,
    IVPanelEffectsResults,
    # PanelResults,
    # RandomEffectsResults,
)

from linearmodels.panel.model import _PanelModelBase, _lstsq, panel_structure_stats, CovarianceEstimator, CovarianceEstimatorType
from linearmodels.iv.model import _IVModelBase
from linearmodels.iv._utility import IVFormulaParser
from linearmodels.iv.data import IVData

from linearmodels.panel.utility import (
    AbsorbingEffectWarning,
    absorbing_warn_msg,
    check_absorbed,
    dummy_matrix,
    in_2core_graph,
    not_absorbed,
)
from linearmodels.shared.exceptions import (
    IndexWarning,
    InferenceUnavailableWarning,
    MemoryWarning,
    MissingValueWarning,
    SingletonWarning,
    missing_warning,
)
from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    InvalidTestStatistic,
    WaldTestStatistic,
)
from linearmodels.shared.linalg import has_constant, inv_sqrth
from linearmodels.shared.typed_getters import get_iv_panel_data_like
from linearmodels.shared.utility import AttrDict, DataFrameWrapper, SeriesWrapper, ensure_unique_column, panel_to_frame
from linearmodels.typing import (
    ArrayLike,
    BoolArray,
    Float64Array,
    IntArray,
    Numeric,
    NumericArray,
    OptionalNumeric,
)
import warnings


from linearmodels.panel.covariance import (
    ACCovariance,
    # ClusteredCovariance,
    # CovarianceManager,
    DriscollKraay,
    FamaMacBethCovariance,
    # HeteroskedasticCovariance,
    # HomoskedasticCovariance,
    # setup_covariance_estimator,
)



from linearmodels.iv.covariance import (
    # ClusteredCovariance,
    # HeteroskedasticCovariance,
    # HomoskedasticCovariance,
    KernelCovariance,
)


class _IVPanelModelBase(_PanelModelBase, _IVModelBase):
    def __init__(
            self,
            dependent: IVPanelDataLike,
            exog: IVPanelDataLike,
            endog: IVPanelDataLike,
            instruments: IVPanelDataLike,
            *,
            weights: IVPanelDataLike | None = None,
            fuller: Numeric = 0,
            kappa: OptionalNumeric = None,
            check_rank: bool = True,
    ) -> None:
        self.dependent = IVPanelData(dependent, "Dep")
        self.exog = IVPanelData(exog, "Exog")
        self.endog = IVPanelData(endog, var_name="endog")
        self.instruments = IVPanelData(instruments, var_name="instruments")
        self.X = IVPanelData(pd.concat([self.exog.dataframe, self.endog.dataframe], axis=1))
        self.Z = IVPanelData(pd.concat([self.exog.dataframe, self.instruments.dataframe], axis=1))

        self._original_shape = self.dependent.shape
        self._has_constant = False
        self._formula: str | None = None
        self._is_weighted = True
        self._name = self.__class__.__name__
        self.weights = self._adapt_weights(weights)
        self._not_null = np.ones(self.dependent.values2d.shape[0], dtype=bool)
        self._cov_estimators = CovarianceManager(
            self.__class__.__name__,
            HomoskedasticCovariance,
            # HeteroskedasticCovariance,
            ClusteredCovariance,
            DriscollKraay,
            ACCovariance,
        )
        self._original_index = self.dependent.index.copy()
        self._constant_index: int | None = None
        self._check_rank = bool(check_rank)
        self._validate_data()
        self._singleton_index: BoolArray | None = None






        self._original_index = self.dependent.pandas.index
        if weights is None:
            weights = pd.DataFrame(ones(dependent.shape), index=dependent.index)
        # weights = IVPanelData(weights).ndarray
        if npany(weights <= 0):
            raise ValueError("weights must be strictly positive.")
        weights = weights / nanmean(weights)
        self.weights = IVPanelData(weights, var_name="weights")

        # self._drop_locs = self._drop_missing()
        # dependent variable
        w = sqrt(self.weights.ndarray)
        self._y = self.dependent.ndarray
        self._wy = self._y * w
        # model regressors
        self._x = self.X.ndarray
        self._wx = self._x * w
        # first-stage regressors
        self._z = self.Z.ndarray
        self._wz = self._z * w

        self._regressor_is_exog = array(
            [True] * len(self.exog.cols) + [False] * len(self.endog.cols)
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

    @property
    def formula(self) -> str | None:
        """Formula used to construct the model"""
        return self._formula

    @formula.setter
    def formula(self, value: str | None) -> None:
        self._formula = value

    @property
    def has_constant(self) -> bool:
        """Flag indicating the model a constant or implicit constant"""
        return self._has_constant

    def _prepare_between(self) -> tuple[Float64Array, Float64Array, Float64Array]:
        """Prepare values for between estimation of R2"""
        weights = self.weights if self._is_weighted else None
        y = np.asarray(self.dependent.mean("entity", weights=weights))
        x = np.asarray(self.X.mean("entity", weights=weights))
        # Weight transformation
        wcount, wmean = self.weights.count("entity"), self.weights.mean("entity")
        wsum = wcount * wmean
        w = np.asarray(wsum)
        w = w / w.mean()

        return y, x, w

    def _rsquared_corr(self, params: Float64Array) -> tuple[float, float, float]:
        """Correlation-based measures of R2"""
        # Overall
        y = self.dependent.values2d
        x = self.X.values2d
        xb = x @ params
        r2o = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2o = np.corrcoef(y.T, (x @ params).T)[0, 1]
        # Between
        y = np.asarray(self.dependent.mean("entity"))
        x = np.asarray(self.X.mean("entity"))
        xb = x @ params
        r2b = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2b = np.corrcoef(y.T, (x @ params).T)[0, 1]

        # Within
        y = self.dependent.demean("entity", return_panel=False)
        x = self.X.demean("entity", return_panel=False)
        xb = x @ params
        r2w = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2w = np.corrcoef(y.T, xb.T)[0, 1]

        return r2o**2, r2w**2, r2b**2

    def _rsquared(
        self, params: Float64Array, reweight: bool = False
    ) -> tuple[float, float, float]:
        """Compute alternative measures of R2"""
        if self.has_constant and self.X.nvar == 1:
            # Constant only fast track
            return 0.0, 0.0, 0.0

        #############################################
        # R2 - Between
        #############################################
        y, x, w = self._prepare_between()
        if np.all(self.weights.values2d == 1.0) and not reweight:
            w = root_w = np.ones_like(w)
        else:
            root_w = cast(Float64Array, np.sqrt(w))
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        e = y
        if self.has_constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(np.squeeze(w.T @ (e**2)))
        r2b = 1 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        #############################################
        # R2 - Overall
        #############################################
        y = self.dependent.values2d
        x = self.X.values2d
        w = self.weights.values2d
        root_w = cast(Float64Array, np.sqrt(w))
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        mu = (w * y).sum() / w.sum() if self.has_constant else 0
        we = wy - root_w * mu
        total_ss = float(np.squeeze(we.T @ we))
        r2o = 1 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        #############################################
        # R2 - Within
        #############################################
        weights = self.weights if self._is_weighted else None
        wy = cast(
            Float64Array,
            self.dependent.demean("entity", weights=weights, return_panel=False),
        )
        wx = cast(
            Float64Array,
            self.X.demean("entity", weights=weights, return_panel=False),
        )
        assert isinstance(wy, np.ndarray)
        assert isinstance(wx, np.ndarray)
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        total_ss = float(np.squeeze(wy.T @ wy))
        if self.dependent.nobs == 1 or (self.X.nvar == 1 and self.has_constant):
            r2w = 0.0
        else:
            r2w = 1.0 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        return r2o, r2w, r2b


    def _postestimation(
        self,
        params: Float64Array,
        cov: CovarianceEstimator,
        debiased: bool,
        df_resid: int,
        weps: Float64Array,
        y: Float64Array,
        x: Float64Array,
        root_w: Float64Array,
    ) -> AttrDict:
        """Common post-estimation values"""
        columns = self._columns
        index = self._index
        eps = self.resids(params)
        fitted = DataFrameWrapper(
            np.asarray(y) - eps, index=index, columns=["fitted_values"]
        )
        cov_old = cov.cov
        residual_ss = np.squeeze(weps.T @ weps)
        w = self.weights.ndarray
        e = self._wy
        if self.has_constant:
            e = e - sqrt(self.weights.ndarray) * np.average(self._y, weights=w)

        total_ss = float(np.squeeze(e.T @ e))

        f_info = self._f_statistic_robust(params)
        f_stat = self._f_statistic(weps, y, x, root_w, df_resid)
        r2o, r2w, r2b = self._rsquared(params)
        c2o, c2w, c2b = self._rsquared_corr(params)
        f_pooled = InapplicableTestStatistic(
            reason="Model has no effects", name="Pooled F-stat"
        )
        entity_info, time_info, other_info = self._info()
        nobs = weps.shape[0]
        sigma2 = float(np.squeeze(weps.T @ weps) / nobs)
        if sigma2 > 0.0:
            loglik = -0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        else:
            loglik = np.nan

        res = AttrDict(
            params=Series(params.squeeze(), columns, name="parameter"),
            eps=SeriesWrapper(eps.squeeze(), index=index, name="residual"),
            weps=SeriesWrapper(
                weps.squeeze(), index=index, name="weighted residual"
            ),
            cov=cov_old,
            cov_estimator=cov,
            deferred_cov=cov.deferred_cov,
            f_info=f_info,
            f_stat=f_stat,
            fstat=f_stat,
            debiased=debiased,
            name=self._name,
            var_names=self._columns,
            r2w=r2w,
            r2b=r2b,
            r2=r2w,
            r2o=r2o,
            c2o=c2o,
            c2b=c2b,
            c2w=c2w,
            s2=cov.s2,
            residual_ss=residual_ss,
            total_ss=total_ss,
            vars=columns,
            instruments=self._instr_columns,
            cov_config=cov.config,
            model=self,
            cov_type=cov.name,
            index=self.dependent.index,
            entity_info=entity_info,
            time_info=time_info,
            other_info=other_info,
            f_pooled=f_pooled,
            loglik=loglik,
            not_null=self._not_null,
            original_index=self._original_index,
            method=self._method,
            fitted=fitted,
        )
        return res

    def _f_statistic(
        *args, **kwargs
    ) -> WaldTestStatistic | InvalidTestStatistic:
        return _PanelModelBase._f_statistic(*args, **kwargs)

    def reformat_clusters(self, clusters: IntArray | IVPanelDataLike) -> IVPanelData:
        """
        Reformat cluster variables

        Parameters
        ----------
        clusters : array_like
            Values to use for variance clustering

        Returns
        -------
        PanelData
            Original data with matching axis and observation dropped where
            missing in the model data.

        Notes
        -----
        This is exposed for testing and is not normally needed for estimation
        """
        clusters_pd = IVPanelData(clusters, var_name="cov.cluster", convert_dummies=False)
        if clusters_pd.shape[1:] != self._original_shape[1:]:
            raise ValueError(
                "clusters must have the same number of entities "
                "and time periods as the model data."
            )
        clusters_pd.drop(~self.not_null)
        return clusters_pd.copy()

    def _setup_clusters(
        self,
        cov_config: Mapping[str, bool | float | str | IntArray | DataFrame | IVPanelData],
    ) -> dict[str, bool | float | str | IntArray | DataFrame | IVPanelData]:
        cov_config_upd = dict(cov_config)
        cluster_types = ("clusters", "cluster_entity", "cluster_time")
        common = set(cov_config.keys()).intersection(cluster_types)
        if not common:
            return cov_config_upd

        cov_config_upd = {k: v for k, v in cov_config.items()}

        clusters = get_iv_panel_data_like(cov_config, "clusters")
        clusters_frame: DataFrame | None = None
        if clusters is not None:
            formatted_clusters = self.reformat_clusters(clusters)
            for col in formatted_clusters.dataframe:
                cat = Categorical(formatted_clusters.dataframe[col])
                # TODO: Bug in pandas-stubs
                #  https://github.com/pandas-dev/pandas-stubs/issues/111
                formatted_clusters.dataframe[col] = cat.codes.astype(
                    np.int64
                )  # type: ignore
            clusters_frame = formatted_clusters.dataframe

        cluster_entity = cov_config_upd["cluster_entity"] if 'cluster_entity' in cov_config_upd else False
        if cluster_entity:
            group_ids_arr = self.dependent.entity_ids.squeeze()
            name = "cov.cluster.entity"
            group_ids = Series(group_ids_arr, index=self.dependent.index, name=name)
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)

        cluster_time = cov_config_upd["cluster_time"] if 'cluster_time' in cov_config_upd else False
        if cluster_time:
            group_ids_arr = self.dependent.time_ids.squeeze()
            name = "cov.cluster.time"
            group_ids = Series(group_ids_arr, index=self.dependent.index, name=name)
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)
        if self._singleton_index is not None and clusters_frame is not None:
            clusters_frame = clusters_frame.loc[~self._singleton_index]

        if clusters_frame is not None:
            cov_config_upd["clusters"] = np.asarray(clusters_frame)

        return cov_config_upd

    def predict(
        self,
        params: ArrayLike,
        *,
        exog: IVPanelDataLike | None = None,
        endog: IVPanelDataLike | None = None,
        data: DataFrame | None = None,
        eval_env: int = 4,  # In the panel version, this is 1. Why?
        context: Mapping[str, Any] | None = None,
    ) -> DataFrame:
        """
        Predict values for additional data

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1)
        exog : array_like
            Exogenous regressors (nobs by nexog)
        endog : array_like
            Endogenous regressors (nobs by nendog)
        data : DataFrame
            Values to use when making predictions from a model constructed
            from a formula
        eval_env : int
            Depth of use when evaluating formulas.

        Returns
        -------
        DataFrame
            Fitted values from supplied data and parameters

        Notes
        -----
        The number of parameters must satisfy nvar = nexog + nendog.

        When using `exog` and `endog`, regressor matrix is constructed as
        `[exog, endog]` and so parameters must be aligned to this structure.
        The the the same structure used in model estimation.

        If `data` is not none, then `exog` and `endog` must be none.
        Predictions from models constructed using formulas can
        be computed using either `exog` and `endog`, which will treat these are
        arrays of values corresponding to the formula-processed data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.
        """
        if data is not None and not self.formula:
            raise ValueError(
                "Unable to use data when the model was not " "created using a formula."
            )
        if data is not None and (exog is not None or endog is not None):
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog/endog or data, but not both."
            )
        if exog is not None or endog is not None:
            exog = IVPanelData(exog).pandas
            endog = IVPanelData(endog).pandas
        elif data is not None:
            if context is None:
                context = capture_context(eval_env)
            parser = IVPanelFormulaParser(self.formula, data, context=context)
            exog = parser.exog
            endog = parser.endog
        else:
            raise ValueError("exog and endog or data must be provided.")
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
        exog_endog = pd.concat([exog, endog], axis=1)
        x = np.asarray(exog_endog)
        params = np.atleast_2d(np.asarray(params))
        if params.shape[0] == 1:
            params = params.T
        pred = DataFrame(x @ params, index=exog_endog.index, columns=["predictions"])

        return pred

    @staticmethod
    def estimate_parameters(
            x: Float64Array, y: Float64Array, z: Float64Array, kappa: Numeric
    ) -> Float64Array:
        """
        Parameter estimation without error checking

        Parameters
        ----------
        x : ndarray
            Regressor matrix (nobs by nvar)
        y : ndarray
            Regressand matrix (nobs by 1)
        z : ndarray
            Instrument matrix (nobs by ninstr)
        kappa : scalar
            Parameter value for k-class estimator

        Returns
        -------
        ndarray
            Estimated parameters (nvar by 1)

        Notes
        -----
        Exposed as a static method to facilitate estimation with other data,
        e.g., bootstrapped samples.  Performs no error checking.
        """
        pinvz = np.linalg.pinv(z)
        p1 = (x.T @ x) * (1 - kappa) + kappa * ((x.T @ z) @ (pinvz @ x))
        p2 = (x.T @ y) * (1 - kappa) + kappa * ((x.T @ z) @ (pinvz @ y))
        return np.linalg.inv(p1) @ p2

    def _estimate_kappa(self) -> float:
        y, x, z = self._wy, self._wx, self._wz
        is_exog = self._regressor_is_exog
        e = c_[y, x[:, ~is_exog]]
        x1 = x[:, is_exog]

        ez = e - z @ (np.linalg.pinv(z) @ e)
        if x1.shape[1] == 0:  # No exogenous regressors
            ex1 = e
        else:
            ex1 = e - x1 @ (np.linalg.pinv(x1) @ e)

        vpmzv_sqinv = inv_sqrth(ez.T @ ez)
        q = vpmzv_sqinv @ (ex1.T @ ex1) @ vpmzv_sqinv
        return min(np.linalg.eigvalsh(q))


class IVPanel2SLS(_IVPanelModelBase):
    r"""
        One- and two-way fixed effects estimator for panel data

        Parameters
        ----------
        dependent : array_like
            Dependent (left-hand-side) variable (time by entity).
        exog : array_like
            Exogenous or right-hand-side variables (variable by time by entity).
        weights : array_like
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual time
            the weight should be homoskedastic.
        entity_effects : bool
            Flag whether to include entity (fixed) effects in the model
        time_effects : bool
            Flag whether to include time effects in the model
        other_effects : array_like
            Category codes to use for any effects that are not entity or time
            effects. Each variable is treated as an effect.
        singletons : bool
            Flag indicating whether to drop singleton observation
        drop_absorbed : bool
            Flag indicating whether to drop absorbed variables
        check_rank : bool
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model specification.
            Results may be numerically unstable if this check is skipped and
            the matrix is not full rank.

        Notes
        -----
        Many models can be estimated. The most common included entity effects and
        can be described

        .. math::

            y_{it} = \alpha_i + \beta^{\prime}x_{it} + \epsilon_{it}

        where :math:`\alpha_i` is included if ``entity_effects=True``.

        Time effect are also supported, which leads to a model of the form

        .. math::

            y_{it}= \gamma_t + \beta^{\prime}x_{it} + \epsilon_{it}

        where :math:`\gamma_i` is included if ``time_effects=True``.

        Both effects can be simultaneously used,

        .. math::

            y_{it}=\alpha_i + \gamma_t + \beta^{\prime}x_{it} + \epsilon_{it}

        Additionally , arbitrary effects can be specified using categorical variables.

        If both ``entity_effect`` and ``time_effects`` are ``False``, and no other
        effects are included, the model reduces to :class:`PooledOLS`.

        Model supports at most 2 effects.  These can be entity-time, entity-other,
        time-other or 2 other.
        """

    def __init__(
            self,
            dependent: IVPanelDataLike,
            exog: IVPanelDataLike,
            endog: IVPanelDataLike,
            instruments: IVPanelDataLike,
            *,
            weights: IVPanelDataLike | None = None,
            entity_effects: bool = False,
            time_effects: bool = False,
            other_effects: IVPanelDataLike | None = None,
            singletons: bool = True,
            drop_absorbed: bool = False,
            fuller: Numeric = 0,
            kappa: OptionalNumeric = None,
            check_rank: bool = True,
    ) -> None:
        super().__init__(dependent, exog, endog, instruments, weights=weights, fuller=fuller, kappa=kappa, check_rank=check_rank)

        self._entity_effects = entity_effects
        self._time_effects = time_effects
        self._other_effect_cats: IVPanelData | None = None
        self._singletons = singletons
        self._other_effects = self._validate_effects(other_effects)
        self._has_effect = entity_effects or time_effects or self.other_effects
        self._drop_absorbed = drop_absorbed
        self._singleton_index = None
        self._drop_singletons()

    def _collect_effects(self) -> NumericArray:
        if not self._has_effect:
            return np.empty((self.dependent.shape[0], 0))
        effects = []
        if self.entity_effects:
            effects.append(np.asarray(self.dependent.entity_ids).squeeze())
        if self.time_effects:
            effects.append(np.asarray(self.dependent.time_ids).squeeze())
        if self.other_effects:
            assert self._other_effect_cats is not None
            other = self._other_effect_cats.dataframe
            for col in other:
                effects.append(np.asarray(other[col]).squeeze())
        return np.column_stack(effects)

    def _drop_singletons(self) -> None:
        if self._singletons or not self._has_effect:
            return
        effects = self._collect_effects()
        retain = in_2core_graph(effects)

        if np.all(retain):
            return

        import warnings as warn

        nobs = retain.shape[0]
        ndropped = nobs - retain.sum()
        warn.warn(
            f"{ndropped} singleton observations dropped",
            SingletonWarning,
            stacklevel=3,
        )
        drop = ~retain
        self._singleton_index = cast(BoolArray, drop)
        self.dependent.drop(drop)
        self._xw.drop(drop)
        self._xz.drop(drop)
        self.weights.drop(drop)
        if self.other_effects:
            assert self._other_effect_cats is not None
            self._other_effect_cats.drop(drop)
        # Reverify exog matrix
        self._check_exog_rank()

    def __str__(self) -> str:
        out = super().__str__()
        additional = (
            "\nEntity Effects: {ee}, Time Effects: {te}, Num Other Effects: {oe}"
        )
        oe = 0
        if self.other_effects:
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.nvar
        additional = additional.format(
            ee=self.entity_effects, te=self.time_effects, oe=oe
        )
        out += additional
        return out

    def _validate_effects(self, effects: IVPanelDataLike | None) -> bool:
        """Check model effects"""
        if effects is None:
            return False
        effects = IVPanelData(effects, var_name="OtherEffect", convert_dummies=False)

        if effects.shape[1:] != self._original_shape[1:]:
            raise ValueError(
                "other_effects must have the same number of "
                "entities and time periods as dependent."
            )

        num_effects = effects.nvar
        if num_effects + self.entity_effects + self.time_effects > 2:
            raise ValueError("At most two effects supported.")
        cats = {}
        effects_frame = effects.dataframe
        for col in effects_frame:
            cat = Categorical(effects_frame[col])
            # TODO: Bug in pandas-stube
            #  https://github.com/pandas-dev/pandas-stubs/issues/111
            cats[col] = cat.codes.astype(np.int64)  # type: ignore
        cats_df = DataFrame(cats, index=effects_frame.index)
        cats_df = cats_df[effects_frame.columns]
        other_effects = IVPanelData(cats_df)
        other_effects.drop(~self.not_null)
        self._other_effect_cats = other_effects
        cats_array = other_effects.values2d
        nested = False
        nesting_effect = ""
        if cats_array.shape[1] == 2:
            nested = self._is_effect_nested(cats_array[:, [0]], cats_array[:, [1]])
            nested |= self._is_effect_nested(cats_array[:, [1]], cats_array[:, [0]])
            nesting_effect = "other effects"
        elif self.entity_effects:
            nested = self._is_effect_nested(
                cats_array[:, [0]], self.dependent.entity_ids
            )
            nested |= self._is_effect_nested(
                self.dependent.entity_ids, cats_array[:, [0]]
            )
            nesting_effect = "entity effects"
        elif self.time_effects:
            nested = self._is_effect_nested(cats_array[:, [0]], self.dependent.time_ids)
            nested |= self._is_effect_nested(
                self.dependent.time_ids, cats_array[:, [0]]
            )
            nesting_effect = "time effects"
        if nested:
            raise ValueError(
                "Included other effects nest or are nested "
                "by {effect}".format(effect=nesting_effect)
            )

        return True

    @property
    def entity_effects(self) -> bool:
        """Flag indicating whether entity effects are included"""
        return self._entity_effects

    @property
    def time_effects(self) -> bool:
        """Flag indicating whether time effects are included"""
        return self._time_effects

    @property
    def other_effects(self) -> bool:
        """Flag indicating whether other (generic) effects are included"""
        return self._other_effects

    @classmethod
    def from_formula(
            cls,
            formula: str,
            data: IVPanelDataLike,
            *,
            weights: IVPanelDataLike | None = None,
            other_effects: IVPanelDataLike | None = None,
            singletons: bool = True,
            drop_absorbed: bool = False,
            check_rank: bool = True,
    ) -> IVPanel2SLS:
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to formulaic formula
            rules with two special variable names, EntityEffects and
            TimeEffects which can be used to specify that the model should
            contain an entity effect or a time effect, respectively. See
            Examples.
        data : array_like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array_like
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual time
            the weight should be homoskedastic.
        other_effects : array_like
            Category codes to use for any effects that are not entity or time
            effects. Each variable is treated as an effect.
        singletons : bool
            Flag indicating whether to drop singleton observation
        drop_absorbed : bool
            Flag indicating whether to drop absorbed variables
        check_rank : bool
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model
            specification. Results may be numerically unstable if this check
            is skipped and the matrix is not full rank.

        Returns
        -------
        PanelOLS
            Model specified using the formula

        Examples
        --------
        >>> from linearmodels import PanelOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = PanelOLS.from_formula("y ~ 1 + x1 + EntityEffects", panel_data.data)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)
        """
        parser = IVPanelFormulaParser(formula, data, context=capture_context(1))
        entity_effect = parser.entity_effect
        time_effect = parser.time_effect
        dependent, exog, endog, instr = parser.data
        mod = cls(
            dependent,
            exog,
            endog,
            instr,
            weights=weights,
            entity_effects=entity_effect,
            time_effects=time_effect,
            other_effects=other_effects,
            singletons=singletons,
            drop_absorbed=drop_absorbed,
            check_rank=check_rank,
        )
        mod.formula = formula
        return mod

    def _lsmr_path(
            self,
    ) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array, Float64Array]:
        """Sparse implementation, works for all scenarios"""
        y = cast(Float64Array, self.dependent.values2d)
        x = cast(Float64Array, self.X.values2d)
        z = cast(Float64Array, self.Z.values2d)
        w = cast(Float64Array, self.weights.values2d)
        root_w = np.sqrt(w)
        wybar = root_w * (w.T @ y / w.sum())
        wy = root_w * y
        wx = root_w * x
        if not self._has_effect:
            y_effect, x_effect = np.zeros_like(wy), np.zeros_like(wx)
            return wy, wx, wybar, y_effect, x_effect

        wy_gm = wybar
        wx_gm = root_w * (w.T @ x / w.sum())
        root_w_sparse = csc_matrix(root_w)

        cats_l: list[IntArray | Float64Array] = []
        if self.entity_effects:
            cats_l.append(self.dependent.entity_ids)
        if self.time_effects:
            cats_l.append(self.dependent.time_ids)
        if self.other_effects:
            assert self._other_effect_cats is not None
            cats_l.append(self._other_effect_cats.values2d)
        cats = np.concatenate(cats_l, 1)

        wd, cond = dummy_matrix(cats, precondition=True)
        assert isinstance(wd, csc_matrix)
        if self._is_weighted:
            wd = wd.multiply(root_w_sparse)

        wx_mean_l = []
        for i in range(x.shape[1]):
            cond_mean = lsmr(wd, wx[:, i], atol=1e-8, btol=1e-8)[0]
            cond_mean /= cond
            wx_mean_l.append(cond_mean)
        wx_mean = np.column_stack(wx_mean_l)
        wy_mean = lsmr(wd, wy, atol=1e-8, btol=1e-8)[0]
        wy_mean /= cond
        wy_mean = wy_mean[:, None]

        wx_mean = csc_matrix(wx_mean)
        wy_mean = csc_matrix(wy_mean)

        # Purge fitted, weighted values
        sp_cond = diags(cond, format="csc")
        wx = wx - (wd @ sp_cond @ wx_mean).A
        wy = wy - (wd @ sp_cond @ wy_mean).A

        if self.has_constant:
            wy += wy_gm
            wx += wx_gm
        else:
            wybar = 0

        y_effects = y - wy / root_w
        x_effects = x - wx / root_w

        return wy, wx, wybar, y_effects, x_effects

    def _slow_path(
            self,
    ) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array, Float64Array]:
        """Frisch-Waugh-Lovell implementation, works for all scenarios"""
        w = cast(Float64Array, self.weights.values2d)
        root_w = np.sqrt(w)

        y = root_w * cast(Float64Array, self.dependent.values2d)
        x = root_w * cast(Float64Array, self.X.values2d)
        z = root_w * cast(Float64Array, self.Z.values2d)
        if not self._has_effect:
            ybar = root_w @ _lstsq(root_w, y, rcond=None)[0]
            y_effect, x_effect, z_effect = np.zeros_like(y), np.zeros_like(x), np.zeros_like(z)
            return y, x, z, ybar, y_effect, x_effect, z_effect

        drop_first = self._has_constant
        d_l = []
        if self.entity_effects:
            d_l.append(self.dependent.dummies("entity", drop_first=drop_first).values)
            drop_first = True
        if self.time_effects:
            d_l.append(self.dependent.dummies("time", drop_first=drop_first).values)
            drop_first = True
        if self.other_effects:
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                dummies = get_dummies(oe[c], drop_first=drop_first).astype(np.float64)
                d_l.append(dummies.values)
                drop_first = True

        d = np.column_stack(d_l)
        wd = root_w * d
        if self.has_constant:
            wd -= root_w * (w.T @ d / w.sum())
            z = np.ones_like(root_w)
            d -= z * (z.T @ d / z.sum())

        x_mean = _lstsq(wd, x, rcond=None)[0]
        y_mean = _lstsq(wd, y, rcond=None)[0]
        z_mean = _lstsq(wd, z, rcond=None)[0]

        # Save fitted unweighted effects to use in eps calculation
        x_effects = d @ x_mean
        y_effects = d @ y_mean
        z_effects = d @ z_mean

        # Purge fitted, weighted values
        x = x - wd @ x_mean
        y = y - wd @ y_mean
        z = z - wd @ z_mean

        ybar = root_w @ _lstsq(root_w, y, rcond=None)[0]
        return y, x, z, ybar, y_effects, x_effects, z_effects

    def _choose_twoway_algo(self) -> bool:
        if not (self.entity_effects and self.time_effects):
            return False
        nentity, nobs = self.dependent.nentity, self.dependent.nobs
        nreg = min(nentity, nobs)
        if nreg < self._wx.shape[1]:
            return False
        # MiB
        reg_size = 8 * nentity * nobs * nreg // 2 ** 20
        low_memory = reg_size > 2 ** 10
        if low_memory:
            import warnings

            warnings.warn(
                "Using low-memory algorithm to estimate two-way model. Explicitly set "
                "low_memory=True to silence this message.  Set low_memory=False to use "
                "the standard algorithm that creates dummy variables for the smaller "
                "of the number of entities or number of time periods.",
                MemoryWarning,
                stacklevel=3,
            )
        return low_memory

    def _fast_path(
            self, low_memory: bool
    ) -> tuple[Float64Array, Float64Array, Float64Array]:
        """Dummy-variable free estimation without weights"""
        _y = self.dependent.values2d
        _x = self.X.values2d
        _z = self.Z.values2d
        ybar = np.asarray(_y.mean(0))

        if not self._has_effect:
            return _y, _x, _z, ybar

        y_gm = ybar
        x_gm = _x.mean(0)
        z_gm = _z.mean(0)

        y = self.dependent
        x = self.X
        z = self.Z

        if self.other_effects:
            assert self._other_effect_cats is not None
            groups = self._other_effect_cats
            if self.entity_effects or self.time_effects:
                groups = groups.copy()
                if self.entity_effects:
                    effect = self.dependent.entity_ids
                else:
                    effect = self.dependent.time_ids
                col = ensure_unique_column("additional.effect", groups.dataframe)
                groups.dataframe[col] = effect
            y = cast(IVPanelData, y.general_demean(groups))
            x = cast(IVPanelData, x.general_demean(groups))
            z = cast(IVPanelData, z.general_demean(groups))
        elif self.entity_effects and self.time_effects:
            y = cast(IVPanelData, y.demean("both", low_memory=low_memory))
            x = cast(IVPanelData, x.demean("both", low_memory=low_memory))
            z = cast(IVPanelData, z.demean("both", low_memory=low_memory))
        elif self.entity_effects:
            y = cast(IVPanelData, y.demean("entity"))
            x = cast(IVPanelData, x.demean("entity"))
            z = cast(IVPanelData, z.demean("entity"))
        else:  # self.time_effects
            y = cast(IVPanelData, y.demean("time"))
            x = cast(IVPanelData, x.demean("time"))
            z = cast(IVPanelData, z.demean("time"))

        y_arr = y.values2d
        x_arr = x.values2d
        z_arr = z.values2d

        if self.has_constant:
            y_arr = y_arr + y_gm
            x_arr = x_arr + x_gm
            z_arr = z_arr + z_gm
        else:
            ybar = np.asarray(0.0)

        return y_arr, x_arr, z_arr, ybar

    def _weighted_fast_path(
            self, low_memory: bool
    ) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array, Float64Array]:
        """Dummy-variable free estimation with weights"""
        y_arr = self.dependent.values2d
        x_arr = self._xw.values2d
        w = self.weights.values2d
        root_w = cast(Float64Array, np.sqrt(w))
        wybar = root_w * (w.T @ y_arr / w.sum())

        if not self._has_effect:
            wy_arr = root_w * self.dependent.values2d
            wx_arr = root_w * self._xw.values2d
            y_effect, x_effect = np.zeros_like(wy_arr), np.zeros_like(wx_arr)
            return wy_arr, wx_arr, wybar, y_effect, x_effect

        wy_gm = wybar
        wx_gm = root_w * (w.T @ x_arr / w.sum())

        y = self.dependent
        x = self._xw

        if self.other_effects:
            assert self._other_effect_cats is not None
            groups = self._other_effect_cats
            if self.entity_effects or self.time_effects:
                groups = groups.copy()
                if self.entity_effects:
                    effect = self.dependent.entity_ids
                else:
                    effect = self.dependent.time_ids
                col = ensure_unique_column("additional.effect", groups.dataframe)
                groups.dataframe[col] = effect
            wy = y.general_demean(groups, weights=self.weights)
            wx = x.general_demean(groups, weights=self.weights)
        elif self.entity_effects and self.time_effects:
            wy = cast(
                IVPanelData, y.demean("both", weights=self.weights, low_memory=low_memory)
            )
            wx = cast(
                IVPanelData, x.demean("both", weights=self.weights, low_memory=low_memory)
            )
        elif self.entity_effects:
            wy = cast(IVPanelData, y.demean("entity", weights=self.weights))
            wx = cast(IVPanelData, x.demean("entity", weights=self.weights))
        else:  # self.time_effects
            wy = cast(IVPanelData, y.demean("time", weights=self.weights))
            wx = cast(IVPanelData, x.demean("time", weights=self.weights))

        wy_arr = wy.values2d
        wx_arr = wx.values2d

        if self.has_constant:
            wy_arr += wy_gm
            wx_arr += wx_gm
        else:
            wybar = 0

        wy_effects = y.values2d - wy_arr / root_w
        wx_effects = x.values2d - wx_arr / root_w

        return wy_arr, wx_arr, wybar, wy_effects, wx_effects

    def _info(self) -> tuple[Series, Series, DataFrame | None]:
        """Information about model effects and panel structure"""

        entity_info, time_info, other_info = super()._info()

        if self.other_effects:
            other_info_values: list[Series] = []
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                name = "Observations per group (" + str(c) + ")"
                other_info_values.append(
                    panel_structure_stats(oe[c].values.astype(np.int32), name)
                )
            other_info = DataFrame(other_info_values)

        return entity_info, time_info, other_info

    @staticmethod
    def _is_effect_nested(effects: NumericArray, clusters: NumericArray) -> bool:
        """Determine whether an effect is nested by the covariance clusters"""
        is_nested = np.zeros(effects.shape[1], dtype=bool)
        for i, e in enumerate(effects.T):
            e = (e - e.min()).astype(np.int64)
            e_count = len(np.unique(e))
            for c in clusters.T:
                c = (c - c.min()).astype(np.int64)
                cmax = c.max()
                ec = e * (cmax + 1) + c
                is_nested[i] = len(np.unique(ec)) == e_count
        return bool(np.all(is_nested))

    def _determine_df_adjustment(
            self,
            cov_type: str,
            **cov_config: bool | float | str | IntArray | DataFrame | IVPanelData,
    ) -> bool:
        if cov_type != "clustered" or not self._has_effect:
            return True
        num_effects = self.entity_effects + self.time_effects
        if self.other_effects:
            assert self._other_effect_cats is not None
            num_effects += self._other_effect_cats.shape[1]

        clusters = cov_config.get("clusters", None)
        if clusters is None:  # No clusters
            return True

        effects = self._collect_effects()
        if num_effects == 1:
            return not self._is_effect_nested(effects, cast(IntArray, clusters))
        return True  # Default case for 2-way -- not completely clear

    def fit(
            self,
            *,
            use_lsdv: bool = False,
            use_lsmr: bool = False,
            low_memory: bool | None = None,
            cov_type: str = "unadjusted",
            debiased: bool = False,
            auto_df: bool = True,
            count_effects: bool = True,
            **cov_config: bool | float | str | IntArray | DataFrame | IVPanelData,
    ) -> IVPanelEffectsResults:
        """
        Estimate model parameters

        Parameters
        ----------
        use_lsdv : bool
            Flag indicating to use the Least Squares Dummy Variable estimator
            to eliminate effects.  The default value uses only means and does
            note require constructing dummy variables for each effect.
        use_lsmr : bool
            Flag indicating to use LSDV with the Sparse Equations and Least
            Squares estimator to eliminate the fixed effects.
        low_memory : {bool, None}
            Flag indicating whether to use a low-memory algorithm when a model
            contains two-way fixed effects. If `None`, the choice is taken
            automatically, and the low memory algorithm is used if the
            required dummy variable array is both larger than then array of
            regressors in the model and requires more than 1 GiB .
        cov_type : str
            Name of covariance estimator. See Notes.
        debiased : bool
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        auto_df : bool
            Flag indicating that the treatment of estimated effects in degree
            of freedom adjustment is automatically handled. This is useful
            since clustered standard errors that are clustered using the same
            variable as an effect do not require degree of freedom correction
            while other estimators such as the unadjusted covariance do.
        count_effects : bool
            Flag indicating that the covariance estimator should be adjusted
            to account for the estimation of effects in the model. Only used
            if ``auto_df=False``.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        PanelEffectsResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import PanelOLS
        >>> mod = PanelOLS(y, x, entity_effects=True)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)

        Notes
        -----
        Three covariance estimators are supported:

        * "unadjusted", "homoskedastic" - Assume residual are homoskedastic
        * "robust", "heteroskedastic" - Control for heteroskedasticity using
          White's estimator
        * "clustered` - One- or two-way clustering.  Configuration options are:

          * ``clusters`` - Input containing 1 or 2 variables.
            Clusters should be integer valued, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters
          * ``cluster_time`` - Boolean indicating to use time clusters

        * "kernel" - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """






        # cov_estimator = COVARIANCE_ESTIMATORS[cov_type]
        # cov_config["debiased"] = debiased
        # cov_config["kappa"] = est_kappa
        # cov_config_copy = {k: v for k, v in cov_config.items()}
        # if "center" in cov_config_copy:
        #     del cov_config_copy["center"]
        # cov_estimator_inst = cov_estimator(wx, wy, wz, params, **cov_config_copy)
        #
        # results = {"kappa": est_kappa, "liml_kappa": liml_kappa}
        # pe = self._post_estimation(params, cov_estimator_inst, cov_type)
        # results.update(pe)
        #
        # if self.endog.shape[1] == 0 and self.instruments.shape[1] == 0:
        #     return OLSResults(results, self)
        # else:
        #     return IVResults(results, self)


        # TODO: I DON'T KNOW HOW TO COMBINE THE TWO FIT METHODS (PANELOLS AND IV2SLS)!





        weighted = np.any(self.weights.values2d != 1.0)

        if use_lsmr:
            y, x, ybar, y_effects, x_effects = self._lsmr_path()
        elif use_lsdv:
            y, x, ybar, y_effects, x_effects = self._slow_path()
        else:
            low_memory = (
                self._choose_twoway_algo() if low_memory is None else low_memory
            )
            if not weighted:
                y, x, z, ybar = self._fast_path(low_memory=low_memory)
                y_effects = np.array([0.0])
                x_effects = np.zeros(x.shape[1])
                z_effects = np.zeros(z.shape[1])
            else:
                y, x, z, ybar, y_effects, x_effects, z_effects = self._weighted_fast_path(
                    low_memory=low_memory
                )

        neffects = 0
        drop_first = self.has_constant
        if self.entity_effects:
            neffects += self.dependent.nentity - drop_first
            drop_first = True
        if self.time_effects:
            neffects += self.dependent.nobs - drop_first
            drop_first = True
        if self.other_effects:
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                neffects += oe[c].nunique() - drop_first
                drop_first = True

        if self.entity_effects or self.time_effects or self.other_effects:
            if not self._drop_absorbed:
                check_absorbed(x, [str(var) for var in self.X.vars])
                check_absorbed(z, [str(var) for var in self.Z.vars])
            else:
                # TODO: Need to special case the constant here when determining which
                #  to retain since we always want to retain the constant if present
                retain = not_absorbed(x, self._has_constant, self._constant_index)
                if not retain:
                    raise ValueError(
                        "All columns in exog have been fully absorbed by the included"
                        " effects. This model cannot be estimated."
                    )
                if len(retain) != x.shape[1]:
                    drop = set(range(x.shape[1])).difference(retain)
                    dropped = ", ".join([str(self.X.vars[i]) for i in drop])
                    import warnings

                    warnings.warn(
                        absorbing_warn_msg.format(absorbed_variables=dropped),
                        AbsorbingEffectWarning,
                        stacklevel=2,
                    )
                    x = x[:, retain]
                    # Update constant index loc
                    if self._has_constant:
                        assert isinstance(self._constant_index, int)
                        self._constant_index = int(
                            np.argwhere(np.array(retain) == self._constant_index)
                        )

                    # Adjust exog
                    self.X = IVPanelData(self.X.dataframe.iloc[:, retain])
                    x_effects = x_effects[retain]
        #

        # TODO: DO I NEED TO ELIMINATE EFFECTS FROM Z IN THE PATHS ABOVE???

        wy, wx, wz = y, x, self._wz
        kappa = self._kappa

        try:
            liml_kappa: float = self._estimate_kappa()
        except Exception as exc:
            liml_kappa = np.nan
            if kappa is None:
                raise ValueError(
                    "Unable to estimate kappa. This is most likely occurs if the "
                    f"instrument matrix is rank deficient. The error raised when "
                    f"computing kappa was:\n\n{exc}"
                )
        if kappa is not None:
            est_kappa = kappa
        else:
            est_kappa = liml_kappa

        if self._fuller != 0:
            nobs, ninstr = wz.shape
            est_kappa -= self._fuller / (nobs - ninstr)

        params = self.estimate_parameters(wx, wy, wz, est_kappa)

        #


        # params = _lstsq(x, y, rcond=None)[0]
        nobs = self.dependent.dataframe.shape[0]
        df_model = x.shape[1] + neffects
        df_resid = nobs - df_model
        # Check clusters if singletons were removed
        cov_config = self._setup_clusters(cov_config)
        if auto_df:
            count_effects = self._determine_df_adjustment(cov_type, **cov_config)
        extra_df = neffects if count_effects else 0
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            y,
            x,
            z,
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )

        weps = y - x @ params
        eps = weps
        _y = self.dependent.values2d
        _x = self.X.values2d
        _z = self.Z.values2d
        if weighted:
            eps = (_y - y_effects) - (_x - x_effects) @ params
            if self.has_constant:
                # Correction since y_effects and x_effects @ params add mean
                w = self.weights.values2d
                eps -= (w * eps).sum() / w.sum()
        index = self.dependent.index
        fitted = DataFrame(_x @ params, index, ["fitted_values"])
        idiosyncratic = DataFrame(eps, index, ["idiosyncratic"])
        eps_effects = _y - fitted.values

        sigma2_tot = float(np.squeeze(eps_effects.T @ eps_effects) / nobs)
        sigma2_eps = float(np.squeeze(eps.T @ eps) / nobs)
        sigma2_effects = sigma2_tot - sigma2_eps
        rho = sigma2_effects / sigma2_tot if sigma2_tot > 0.0 else 0.0

        resid_ss = float(np.squeeze(weps.T @ weps))
        if self.has_constant:
            mu = ybar
        else:
            mu = np.array([0.0])
        total_ss = float(np.squeeze((y - mu).T @ (y - mu)))
        r2 = 1 - resid_ss / total_ss if total_ss > 0.0 else 0.0

        root_w = cast(Float64Array, np.sqrt(self.weights.values2d))
        y_ex = root_w * self.dependent.values2d
        mu_ex = 0
        if (
                self.has_constant
                or self.entity_effects
                or self.time_effects
                or self.other_effects
        ):
            mu_ex = root_w * ((root_w.T @ y_ex) / (root_w.T @ root_w))
        total_ss_ex_effect = float(np.squeeze((y_ex - mu_ex).T @ (y_ex - mu_ex)))
        r2_ex_effects = (
            1 - resid_ss / total_ss_ex_effect if total_ss_ex_effect > 0.0 else 0.0
        )

        res = self._postestimation(params, cov, debiased, df_resid, weps, y, x, root_w)
        ######################################
        # Pooled f-stat
        ######################################
        if self.entity_effects or self.time_effects or self.other_effects:
            wy, wx = root_w * self.dependent.values2d, root_w * self.X.values2d
            df_num, df_denom = (df_model - wx.shape[1]), df_resid
            if not self.has_constant:
                # Correction for when models does not have explicit constant
                wy -= root_w * _lstsq(root_w, wy, rcond=None)[0]
                wx -= root_w * _lstsq(root_w, wx, rcond=None)[0]
                df_num -= 1
            weps_pooled = wy - wx @ _lstsq(wx, wy, rcond=None)[0]
            resid_ss_pooled = float(np.squeeze(weps_pooled.T @ weps_pooled))
            num = (resid_ss_pooled - resid_ss) / df_num

            denom = resid_ss / df_denom
            stat = num / denom
            f_pooled = WaldTestStatistic(
                stat,
                "Effects are zero",
                df_num,
                df_denom=df_denom,
                name="Pooled F-statistic",
            )
            res.update(f_pooled=f_pooled)
            effects = DataFrame(
                eps_effects - eps,
                columns=["estimated_effects"],
                index=self.dependent.index,
            )
        else:
            effects = DataFrame(
                np.zeros_like(eps),
                columns=["estimated_effects"],
                index=self.dependent.index,
            )

        res.update(
            dict(
                df_resid=df_resid,
                df_model=df_model,
                nobs=y.shape[0],
                residual_ss=resid_ss,
                total_ss=total_ss,
                wresids=weps,
                resids=eps,
                r2=r2,
                entity_effects=self.entity_effects,
                time_effects=self.time_effects,
                other_effects=self.other_effects,
                sigma2_eps=sigma2_eps,
                sigma2_effects=sigma2_effects,
                rho=rho,
                r2_ex_effects=r2_ex_effects,
                effects=effects,
                fitted=fitted,
                idiosyncratic=idiosyncratic,
            )
        )

        return IVPanelEffectsResults(res, self)


class IVPanelFormulaParser(IVFormulaParser):
    def __init__(
        self,
        formula: str,
        data: DataFrame,
        eval_env: int = 2,
        context: Mapping[str, Any] | None = None,
    ):
        self._formula = formula
        self._data = data
        self._eval_env = eval_env
        if not context:
            self._context = capture_context(context=self._eval_env)
        else:
            self._context = context
        self._components: dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        parts = self._formula.split("~")
        parts[1] = " 0 + " + parts[1]
        cln_formula = "~".join(parts)
        rm_list = []
        effects = {"EntityEffects": False, "FixedEffects": False, "TimeEffects": False}
        tokens = tokenize(cln_formula)
        for token in tokens:
            _token = str(token)
            if _token in effects:
                effects[_token] = True
                rm_list.append(_token)
        if effects["EntityEffects"] and effects["FixedEffects"]:
            raise ValueError("Cannot use both FixedEffects and EntityEffects")
        self._entity_effect = effects["EntityEffects"] or effects["FixedEffects"]
        self._time_effect = effects["TimeEffects"]
        for effect in effects:
            if effects[effect]:
                loc = cln_formula.find(effect)
                start = cln_formula.rfind("+", 0, loc)
                end = loc + len(effect)
                cln_formula = cln_formula[:start] + cln_formula[end:]
        self._formula = cln_formula
        
        super()._parse()

    @property
    def entity_effect(self) -> bool:
        """Formula contains entity effect"""
        return self._entity_effect

    @property
    def time_effect(self) -> bool:
        """Formula contains time effect"""
        return self._time_effect

