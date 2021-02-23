from typing import Dict, NamedTuple, Optional, Tuple, Type, Union, cast

import numpy as np
from numpy.linalg import lstsq, matrix_rank
from pandas import Categorical, DataFrame, MultiIndex, Series, get_dummies
from patsy.highlevel import ModelDesc, dmatrix
from patsy.missing import NAAction
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import lsmr

from linearmodels.panel.covariance import (
    ACCovariance,
    ClusteredCovariance,
    CovarianceManager,
    DriscollKraay,
    FamaMacBethCovariance,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
    setup_covariance_estimator,
)
from linearmodels.panel.data import PanelData, PanelDataLike
from linearmodels.panel.results import (
    FamaMacBethResults,
    PanelEffectsResults,
    PanelResults,
    RandomEffectsResults,
)
from linearmodels.panel.utility import (
    AbsorbingEffectWarning,
    absorbing_warn_msg,
    check_absorbed,
    dummy_matrix,
    in_2core_graph,
    not_absorbed,
)
from linearmodels.shared.exceptions import (
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
from linearmodels.shared.linalg import has_constant
from linearmodels.shared.typed_getters import get_panel_data_like
from linearmodels.shared.utility import AttrDict, ensure_unique_column, panel_to_frame
from linearmodels.typing import ArrayLike, NDArray

CovarianceEstimator = Union[
    ACCovariance,
    ClusteredCovariance,
    DriscollKraay,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
]

CovarianceEstimatorType = Union[
    Type[ACCovariance],
    Type[ClusteredCovariance],
    Type[DriscollKraay],
    Type[HeteroskedasticCovariance],
    Type[HomoskedasticCovariance],
]


def panel_structure_stats(ids: NDArray, name: str) -> Series:
    bc = np.bincount(ids)
    index = ["mean", "median", "max", "min", "total"]
    out = [bc.mean(), np.median(bc), bc.max(), bc.min(), bc.shape[0]]
    return Series(out, index=index, name=name)


class FInfo(NamedTuple):
    sel: NDArray
    name: str
    invalid_test_stat: Optional[InvalidTestStatistic]
    is_invalid: bool


def _deferred_f(
    params: Series, cov: DataFrame, debiased: bool, df_resid: int, f_info: FInfo
) -> Union[InvalidTestStatistic, WaldTestStatistic]:
    if f_info.is_invalid:
        assert f_info.invalid_test_stat is not None
        return f_info.invalid_test_stat
    sel = f_info.sel
    name = f_info.name
    test_params = np.asarray(params)[sel]
    test_cov = np.asarray(cov)[sel][:, sel]
    test_stat = test_params.T @ np.linalg.inv(test_cov) @ test_params
    test_stat = float(test_stat)
    df = int(sel.sum())
    null = "All parameters ex. constant not zero"

    if debiased:
        wald = WaldTestStatistic(test_stat / df, null, df, df_resid, name=name)
    else:
        wald = WaldTestStatistic(test_stat, null, df, name=name)
    return wald


class PanelFormulaParser(object):
    """
    Parse formulas for OLS and IV models

    Parameters
    ----------
    formula : str
        String formula object.
    data : DataFrame
        Frame containing values for variables used in formula
    eval_env : int
        Stack depth to use when evaluating Patsy formulas

    Notes
    -----
    The general structure of a formula is `dep ~ exog`
    """

    def __init__(self, formula: str, data: PanelDataLike, eval_env: int = 2) -> None:
        self._formula = formula
        self._data = PanelData(data, convert_dummies=False, copy=False)
        self._na_action = NAAction(on_NA="raise", NA_types=[])
        self._eval_env = eval_env
        self._dependent = self._exog = None
        self._parse()

    def _parse(self) -> None:
        parts = self._formula.split("~")
        parts[1] = " 0 + " + parts[1]
        cln_formula = "~".join(parts)

        mod_descr = ModelDesc.from_formula(cln_formula)
        rm_list = []
        effects = {"EntityEffects": False, "FixedEffects": False, "TimeEffects": False}
        for term in mod_descr.rhs_termlist:
            if term.name() in effects:
                effects[term.name()] = True
                rm_list.append(term)
        for term in rm_list:
            mod_descr.rhs_termlist.remove(term)

        if effects["EntityEffects"] and effects["FixedEffects"]:
            raise ValueError("Cannot use both FixedEffects and EntityEffects")
        self._entity_effect = effects["EntityEffects"] or effects["FixedEffects"]
        self._time_effect = effects["TimeEffects"]
        cln_formula = mod_descr.describe()
        self._lhs, self._rhs = map(lambda s: s.strip(), cln_formula.split("~"))
        self._lhs = "0 + " + self._lhs

    @property
    def entity_effect(self) -> bool:
        """Formula contains entity effect"""
        return self._entity_effect

    @property
    def time_effect(self) -> bool:
        """Formula contains time effect"""
        return self._time_effect

    @property
    def eval_env(self) -> int:
        """Set or get the eval env depth"""
        return self._eval_env

    @eval_env.setter
    def eval_env(self, value: int) -> None:
        self._eval_env = value

    @property
    def data(self) -> Tuple[DataFrame, DataFrame]:
        """Returns a tuple containing the dependent, exog, endog"""
        self._eval_env += 1
        out = self.dependent, self.exog
        self._eval_env -= 1
        return out

    @property
    def dependent(self) -> DataFrame:
        """DataFrame containing the dependent variable"""
        return dmatrix(
            self._lhs,
            self._data.dataframe,
            eval_env=self._eval_env,
            return_type="dataframe",
            NA_action=self._na_action,
        )

    @property
    def exog(self) -> DataFrame:
        """DataFrame containing the exogenous variables"""
        out = dmatrix(
            self._rhs,
            self._data.dataframe,
            eval_env=self._eval_env,
            return_type="dataframe",
            NA_action=self._na_action,
        )
        return out


class AmbiguityError(Exception):
    pass


__all__ = [
    "PanelOLS",
    "PooledOLS",
    "RandomEffects",
    "FirstDifferenceOLS",
    "BetweenOLS",
    "AmbiguityError",
    "FamaMacBeth",
]


# Likely
# TODO: Formal test of other outputs
# Future
# TODO: Bootstrap covariance
# TODO: Possibly add AIC/BIC
# TODO: ML Estimation of RE model
# TODO: Defer estimation of 3 R2 values -- slow

EXOG_PREDICT_MSG = """\
exog does not have the correct number of columns. Saw {x_shape}, expected
{params_shape}. This can happen since exog is converted to a PanelData object
before computing the fitted value. The best practice is to pass a DataFrame
with a 2-level MultiIndex containing the entity- and time-ids."""


class _PanelModelBase(object):
    r"""
    Base class for all panel models

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity)
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> None:
        self.dependent = PanelData(dependent, "Dep")
        self.exog = PanelData(exog, "Exog")
        self._original_shape = self.dependent.shape
        self._constant = False
        self._formula: Optional[str] = None
        self._is_weighted = True
        self._name = self.__class__.__name__
        self.weights = self._adapt_weights(weights)
        self._not_null = np.ones(self.dependent.values2d.shape[0], dtype=bool)
        self._cov_estimators = CovarianceManager(
            self.__class__.__name__,
            HomoskedasticCovariance,
            HeteroskedasticCovariance,
            ClusteredCovariance,
            DriscollKraay,
            ACCovariance,
        )
        self._original_index = self.dependent.index.copy()
        self._constant_index: Optional[int] = None
        self._validate_data()
        self._singleton_index: Optional[NDArray] = None

    def __str__(self) -> str:
        out = "{name} \nNum exog: {num_exog}, Constant: {has_constant}"
        return out.format(
            name=self.__class__.__name__,
            num_exog=self.exog.dataframe.shape[1],
            has_constant=self.has_constant,
        )

    def __repr__(self) -> str:
        return self.__str__() + "\nid: " + str(hex(id(self)))

    def reformat_clusters(self, clusters: PanelDataLike) -> PanelData:
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
        clusters = PanelData(clusters, var_name="cov.cluster", convert_dummies=False)
        if clusters.shape[1:] != self._original_shape[1:]:
            raise ValueError(
                "clusters must have the same number of entities "
                "and time periods as the model data."
            )
        clusters.drop(~self.not_null)
        return clusters.copy()

    def _info(self) -> Tuple[Series, Series, None]:
        """Information about panel structure"""

        entity_info = panel_structure_stats(
            self.dependent.entity_ids.squeeze(), "Observations per entity"
        )
        time_info = panel_structure_stats(
            self.dependent.time_ids.squeeze(), "Observations per time period"
        )
        other_info = None

        return entity_info, time_info, other_info

    def _adapt_weights(self, weights: Optional[PanelDataLike]) -> PanelData:
        """Check and transform weights depending on size"""
        if weights is None:
            self._is_weighted = False
            frame = self.dependent.dataframe.copy()
            frame.iloc[:, :] = 1
            frame.columns = ["weight"]
            return PanelData(frame)

        frame = DataFrame(columns=self.dependent.entities, index=self.dependent.time)
        nobs, nentity = self.exog.nobs, self.exog.nentity

        if weights.ndim == 3 or weights.shape == (nobs, nentity):
            return PanelData(weights)

        if isinstance(weights, np.ndarray):
            weights = cast(NDArray, np.squeeze(weights))
        if weights.shape[0] == nobs and nobs == nentity:
            raise AmbiguityError(
                "Unable to distinguish nobs form nentity since they are "
                "equal. You must use an 2-d array to avoid ambiguity."
            )
        if (
            isinstance(weights, (Series, DataFrame))
            and isinstance(weights.index, MultiIndex)
            and weights.shape[0] == self.dependent.dataframe.shape[0]
        ):
            frame = weights
        elif weights.shape[0] == nobs:
            weights = np.asarray(weights)[:, None]
            weights = weights @ np.ones((1, nentity))
            frame.iloc[:, :] = weights
        elif weights.shape[0] == nentity:
            weights = np.asarray(weights)[None, :]
            weights = np.ones((nobs, 1)) @ weights
            frame.iloc[:, :] = weights
        elif weights.shape[0] == nentity * nobs:
            frame = self.dependent.dataframe.copy()
            frame.iloc[:, :] = np.asarray(weights)[:, None]
        else:
            raise ValueError("Weights do not have a supported shape.")
        return PanelData(frame)

    def _check_exog_rank(self) -> int:
        x = self.exog.values2d
        rank_of_x = matrix_rank(x)
        if rank_of_x < x.shape[1]:
            raise ValueError("exog does not have full column rank.")
        return rank_of_x

    def _validate_data(self) -> None:
        """Check input shape and remove missing"""
        y = self._y = self.dependent.values2d
        x = self._x = self.exog.values2d
        w = self._w = self.weights.values2d
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                "dependent and exog must have the same number of " "observations."
            )
        if y.shape[0] != w.shape[0]:
            raise ValueError(
                "weights must have the same number of " "observations as dependent."
            )

        all_missing = np.any(np.isnan(y), axis=1) & np.all(np.isnan(x), axis=1)
        missing = (
            np.any(np.isnan(y), axis=1)
            | np.any(np.isnan(x), axis=1)
            | np.any(np.isnan(w), axis=1)
        )

        missing_warning(np.asarray(all_missing ^ missing))
        if np.any(missing):
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self.weights.drop(missing)

            x = self.exog.values2d
            self._not_null = np.asarray(~missing)

        w = self.weights.dataframe
        if np.any(np.asarray(w) <= 0):
            raise ValueError("weights must be strictly positive.")
        w = w / w.mean()
        self.weights = PanelData(w)
        rank_of_x = self._check_exog_rank()
        self._constant, self._constant_index = has_constant(x, rank_of_x)

    @property
    def formula(self) -> Optional[str]:
        """Formula used to construct the model"""
        return self._formula

    @formula.setter
    def formula(self, value: Optional[str]) -> None:
        self._formula = value

    @property
    def has_constant(self) -> bool:
        """Flag indicating the model a constant or implicit constant"""
        return self._constant

    def _f_statistic(
        self, weps: NDArray, y: NDArray, x: NDArray, root_w: NDArray, df_resid: int
    ) -> Union[WaldTestStatistic, InvalidTestStatistic]:
        """Compute model F-statistic"""
        weps_const = y
        num_df = x.shape[1]
        name = "Model F-statistic (homoskedastic)"
        if self.has_constant:
            if num_df == 1:
                return InvalidTestStatistic("Model contains only a constant", name=name)

            num_df -= 1
            weps_const = cast(NDArray, y - float((root_w.T @ y) / (root_w.T @ root_w)))

        resid_ss = weps.T @ weps
        num = float(weps_const.T @ weps_const - resid_ss)
        denom = resid_ss
        denom_df = df_resid
        stat = float((num / num_df) / (denom / denom_df)) if denom > 0.0 else 0.0
        return WaldTestStatistic(
            stat,
            null="All parameters ex. constant not zero",
            df=num_df,
            df_denom=denom_df,
            name=name,
        )

    def _f_statistic_robust(
        self,
        params: NDArray,
    ) -> FInfo:
        """Compute Wald test that all parameters are 0, ex. constant"""
        sel = np.ones(params.shape[0], dtype=bool)
        name = "Model F-statistic (robust)"

        if self.has_constant:
            if len(sel) == 1:
                return FInfo(
                    sel,
                    name,
                    InvalidTestStatistic("Model contains only a constant", name=name),
                    True,
                )
            assert isinstance(self._constant_index, int)
            sel[self._constant_index] = False

        return FInfo(sel, name, None, False)

    def _prepare_between(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Prepare values for between estimation of R2"""
        weights = self.weights if self._is_weighted else None
        y = np.asarray(self.dependent.mean("entity", weights=weights))
        x = np.asarray(self.exog.mean("entity", weights=weights))
        # Weight transformation
        wcount, wmean = self.weights.count("entity"), self.weights.mean("entity")
        wsum = wcount * wmean
        w = np.asarray(wsum)
        w = w / w.mean()

        return y, x, w

    def _rsquared_corr(self, params: NDArray) -> Tuple[float, float, float]:
        """Correlation-based measures of R2"""
        # Overall
        y = self.dependent.values2d
        x = self.exog.values2d
        xb = x @ params
        r2o = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2o = np.corrcoef(y.T, (x @ params).T)[0, 1]
        # Between
        y = np.asarray(self.dependent.mean("entity"))
        x = np.asarray(self.exog.mean("entity"))
        xb = x @ params
        r2b = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2b = np.corrcoef(y.T, (x @ params).T)[0, 1]

        # Within
        y = cast(NDArray, self.dependent.demean("entity", return_panel=False))
        x = cast(NDArray, self.exog.demean("entity", return_panel=False))
        xb = x @ params
        r2w = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2w = np.corrcoef(y.T, xb.T)[0, 1]

        return r2o, r2w, r2b

    def _rsquared(
        self, params: NDArray, reweight: bool = False
    ) -> Tuple[float, float, float]:
        """Compute alternative measures of R2"""
        if self.has_constant and self.exog.nvar == 1:
            # Constant only fast track
            return 0.0, 0.0, 0.0

        #############################################
        # R2 - Between
        #############################################
        y, x, w = self._prepare_between()
        if np.all(self.weights.values2d == 1.0) and not reweight:
            w = root_w = np.ones_like(w)
        else:
            root_w = cast(NDArray, np.sqrt(w))
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(weps.T @ weps)
        e = y
        if self.has_constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(w.T @ (e ** 2))
        r2b = 1 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        #############################################
        # R2 - Overall
        #############################################
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
        root_w = cast(NDArray, np.sqrt(w))
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(weps.T @ weps)
        mu = (w * y).sum() / w.sum() if self.has_constant else 0
        we = wy - root_w * mu
        total_ss = float(we.T @ we)
        r2o = 1 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        #############################################
        # R2 - Within
        #############################################
        weights = self.weights if self._is_weighted else None
        wy = self.dependent.demean("entity", weights=weights, return_panel=False)
        wx = self.exog.demean("entity", weights=weights, return_panel=False)
        assert isinstance(wy, np.ndarray)
        assert isinstance(wx, np.ndarray)
        weps = wy - wx @ params
        residual_ss = float(weps.T @ weps)
        total_ss = float(wy.T @ wy)
        if self.dependent.nobs == 1 or (self.exog.nvar == 1 and self.has_constant):
            r2w = 0.0
        else:
            r2w = 1.0 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        return r2o, r2w, r2b

    def _postestimation(
        self,
        params: NDArray,
        cov: CovarianceEstimator,
        debiased: bool,
        df_resid: int,
        weps: NDArray,
        y: NDArray,
        x: NDArray,
        root_w: NDArray,
    ) -> AttrDict:
        """Common post-estimation values"""
        f_info = self._f_statistic_robust(params)
        f_stat = self._f_statistic(weps, y, x, root_w, df_resid)
        r2o, r2w, r2b = self._rsquared(params)
        c2o, c2w, c2b = self._rsquared_corr(params)
        f_pooled = InapplicableTestStatistic(
            reason="Model has no effects", name="Pooled F-stat"
        )
        entity_info, time_info, other_info = self._info()
        nobs = weps.shape[0]
        sigma2 = float(weps.T @ weps / nobs)
        if sigma2 > 0.0:
            loglik = -0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        else:
            loglik = np.nan

        res = AttrDict(
            params=params,
            deferred_cov=cov.deferred_cov,
            f_info=f_info,
            f_stat=f_stat,
            debiased=debiased,
            name=self._name,
            var_names=self.exog.vars,
            r2w=r2w,
            r2b=r2b,
            r2=r2w,
            r2o=r2o,
            c2o=c2o,
            c2b=c2b,
            c2w=c2w,
            s2=cov.s2,
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
        )
        return res

    @property
    def not_null(self) -> NDArray:
        """Locations of non-missing observations"""
        return self._not_null

    def _setup_clusters(
        self, cov_config: Dict[str, Union[bool, float, str, PanelDataLike]]
    ) -> Dict[str, Union[bool, float, str, NDArray, DataFrame, PanelData]]:

        cov_config_upd = cov_config.copy()
        cluster_types = ("clusters", "cluster_entity", "cluster_time")
        common = set(cov_config.keys()).intersection(cluster_types)
        if not common:
            return cov_config_upd

        cov_config_upd = {k: v for k, v in cov_config.items()}

        clusters = get_panel_data_like(cov_config, "clusters")
        clusters_frame: Optional[DataFrame] = None
        if clusters is not None:
            formatted_clusters = self.reformat_clusters(clusters)
            for col in formatted_clusters.dataframe:
                cat = Categorical(formatted_clusters.dataframe[col])
                formatted_clusters.dataframe[col] = cat.codes.astype(np.int64)
            clusters_frame = formatted_clusters.dataframe

        cluster_entity = bool(cov_config_upd.pop("cluster_entity", False))
        if cluster_entity:
            group_ids = self.dependent.entity_ids.squeeze()
            name = "cov.cluster.entity"
            group_ids = Series(group_ids, index=self.dependent.index, name=name)
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)

        cluster_time = bool(cov_config_upd.pop("cluster_time", False))
        if cluster_time:
            group_ids = self.dependent.time_ids.squeeze()
            name = "cov.cluster.time"
            group_ids = Series(group_ids, index=self.dependent.index, name=name)
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)
        if self._singleton_index is not None and clusters_frame is not None:
            clusters_frame = clusters_frame.loc[~self._singleton_index]

        cov_config_upd["clusters"] = (
            np.asarray(clusters_frame) if clusters_frame is not None else clusters_frame
        )

        return cov_config_upd

    def predict(
        self,
        params: ArrayLike,
        *,
        exog: Optional[PanelDataLike] = None,
        data: Optional[PanelDataLike] = None,
        eval_env: int = 4,
    ) -> DataFrame:
        """
        Predict values for additional data

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1)
        exog : array_like
            Exogenous regressors (nobs by nvar)
        data : DataFrame
            Values to use when making predictions from a model constructed
            from a formula
        eval_env : int
            Depth of use when evaluating formulas using Patsy.

        Returns
        -------
        DataFrame
            Fitted values from supplied data and parameters

        Notes
        -----
        If `data` is not None, then `exog` must be None.
        Predictions from models constructed using formulas can
        be computed using either `exog`, which will treat these are
        arrays of values corresponding to the formula-processed data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.
        """
        if data is not None and self.formula is None:
            raise ValueError(
                "Unable to use data when the model was not " "created using a formula."
            )
        if data is not None and exog is not None:
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog or data, but not both."
            )
        if exog is not None:
            exog = PanelData(exog).dataframe
        else:
            assert self._formula is not None
            assert data is not None
            parser = PanelFormulaParser(self._formula, data, eval_env=eval_env)
            exog = parser.exog
        x = exog.values
        params = np.atleast_2d(np.asarray(params))
        if params.shape[0] == 1:
            params = params.T
        if x.shape[1] != params.shape[0]:
            raise ValueError(
                EXOG_PREDICT_MSG.format(
                    x_shape=x.shape[1], params_shape=params.shape[0]
                )
            )

        pred = DataFrame(x @ params, index=exog.index, columns=["predictions"])

        return pred


class PooledOLS(_PanelModelBase):
    r"""
    Pooled coefficient estimator for panel data

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity)
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\beta^{\prime}x_{it}+\epsilon_{it}
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> None:
        super().__init__(dependent, exog, weights=weights)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> "PooledOLS":
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array_like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array_like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        PooledOLS
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import PooledOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = PooledOLS.from_formula('y ~ 1 + x1', panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data)
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights)
        mod.formula = formula
        return mod

    def fit(
        self,
        *,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: Union[bool, float, str, NDArray, DataFrame, PanelData],
    ) -> PanelResults:
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        PanelResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import PooledOLS
        >>> mod = PooledOLS(y, x)
        >>> res = mod.fit(cov_type='clustered', cluster_entity=True)

        Notes
        -----
        Four covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters
          * ``cluster_time`` - Boolean indicating to use time clusters

        * 'kernel' - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
        root_w = cast(NDArray, np.sqrt(w))
        wx = root_w * x
        wy = root_w * y

        params = lstsq(wx, wy, rcond=None)[0]

        nobs = y.shape[0]
        df_model = x.shape[1]
        df_resid = nobs - df_model
        cov_config = self._setup_clusters(cov_config)
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.get("extra_df", 0)
            assert isinstance(_extra_df, (int, str))
            extra_df = int(_extra_df)
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            wx,
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )
        weps = wy - wx @ params
        index = self.dependent.index
        fitted = DataFrame(x @ params, index, ["fitted_values"])
        effects = DataFrame(
            np.full_like(np.asarray(fitted), np.nan), index, ["estimated_effects"]
        )
        eps = y - fitted.values
        idiosyncratic = DataFrame(eps, index, ["idiosyncratic"])
        residual_ss = float(weps.T @ weps)
        e = y
        if self._constant:
            e = e - (w * y).sum() / w.sum()

        total_ss = float(w.T @ (e ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, root_w
        )
        res.update(
            dict(
                df_resid=df_resid,
                df_model=df_model,
                nobs=y.shape[0],
                residual_ss=residual_ss,
                total_ss=total_ss,
                r2=r2,
                wresids=weps,
                resids=eps,
                index=self.dependent.index,
                fitted=fitted,
                effects=effects,
                idiosyncratic=idiosyncratic,
            )
        )

        return PanelResults(res)

    def predict(
        self,
        params: ArrayLike,
        *,
        exog: Optional[PanelDataLike] = None,
        data: Optional[DataFrame] = None,
        eval_env: int = 4,
    ) -> DataFrame:
        """
        Predict values for additional data

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1)
        exog : array_like
            Exogenous regressors (nobs by nvar)
        data : DataFrame
            Values to use when making predictions from a model constructed
            from a formula
        eval_env : int
            Depth of use when evaluating formulas using Patsy.

        Returns
        -------
        DataFrame
            Fitted values from supplied data and parameters

        Notes
        -----
        If `data` is not None, then `exog` must be None.
        Predictions from models constructed using formulas can
        be computed using either `exog`, which will treat these are
        arrays of values corresponding to the formula-processed data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.
        """
        if data is not None and self.formula is None:
            raise ValueError(
                "Unable to use data when the model was not " "created using a formula."
            )
        if data is not None and exog is not None:
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog or data, but not both."
            )
        if exog is not None:
            exog = PanelData(exog).dataframe
        else:
            assert self._formula is not None
            assert data is not None
            parser = PanelFormulaParser(self._formula, data, eval_env=eval_env)
            exog = parser.exog
        x = exog.values
        params = np.atleast_2d(np.asarray(params))
        if params.shape[0] == 1:
            params = params.T
        if x.shape[1] != params.shape[0]:
            raise ValueError(
                EXOG_PREDICT_MSG.format(
                    x_shape=x.shape[1], params_shape=params.shape[0]
                )
            )
        pred = DataFrame(x @ params, index=exog.index, columns=["predictions"])

        return pred


class PanelOLS(_PanelModelBase):
    r"""
    One- and two-way fixed effects estimator for panel data

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity).
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.
    entity_effects : bool, optional
        Flag whether to include entity (fixed) effects in the model
    time_effects : bool, optional
        Flag whether to include time effects in the model
    other_effects : array_like, optional
        Category codes to use for any effects that are not entity or time
        effects. Each variable is treated as an effect.
    singletons : bool, optional
        Flag indicating whether to drop singleton observation
    drop_absorbed : bool, optional
        Flag indicating whether to drop absorbed variables

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

    If both ``entity_effect``  and``time_effects`` are ``False``, and no other
    effects are included, the model reduces to :class:`PooledOLS`.

    Model supports at most 2 effects.  These can be entity-time, entity-other, time-other or
    2 other.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
        entity_effects: bool = False,
        time_effects: bool = False,
        other_effects: Optional[PanelDataLike] = None,
        singletons: bool = True,
        drop_absorbed: bool = False,
    ) -> None:
        super(PanelOLS, self).__init__(dependent, exog, weights=weights)

        self._entity_effects = entity_effects
        self._time_effects = time_effects
        self._other_effect_cats: Optional[PanelData] = None
        self._singletons = singletons
        self._other_effects = self._validate_effects(other_effects)
        self._has_effect = entity_effects or time_effects or self.other_effects
        self._drop_absorbed = drop_absorbed
        self._singleton_index = None
        self._drop_singletons()

    def _collect_effects(self) -> NDArray:
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
            "{0} singleton observations dropped".format(ndropped), SingletonWarning
        )
        drop = ~retain
        self._singleton_index = cast(NDArray, drop)
        self.dependent.drop(drop)
        self.exog.drop(drop)
        self.weights.drop(drop)
        if self.other_effects:
            assert self._other_effect_cats is not None
            self._other_effect_cats.drop(drop)
        # Reverify exog matrix
        self._check_exog_rank()

    def __str__(self) -> str:
        out = super(PanelOLS, self).__str__()
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

    def _validate_effects(self, effects: Optional[PanelDataLike]) -> bool:
        """Check model effects"""
        if effects is None:
            return False
        effects = PanelData(effects, var_name="OtherEffect", convert_dummies=False)

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
            cats[col] = cat.codes.astype(np.int64)
        cats = DataFrame(cats, index=effects_frame.index)
        cats = cats[effects_frame.columns]
        other_effects = PanelData(cats)
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
        data: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
        other_effects: Optional[PanelDataLike] = None,
        singletons: bool = True,
        drop_absorbed: bool = False,
    ) -> "PanelOLS":
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules
            with two special variable names, EntityEffects and TimeEffects
            which can be used to specify that the model should contain an
            entity effect or a time effect, respectively. See Examples.
        data : array_like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array_like
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual time
            the weight should be homoskedastic.
        other_effects : array_like, optional
            Category codes to use for any effects that are not entity or time
            effects. Each variable is treated as an effect.
        singletons : bool, optional
            Flag indicating whether to drop singleton observation
        drop_absorbed : bool, optional
            Flag indicating whether to drop absorbed variables



        Returns
        -------
        PanelOLS
            Model specified using the formula

        Examples
        --------
        >>> from linearmodels import PanelOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = PanelOLS.from_formula('y ~ 1 + x1 + EntityEffects', panel_data.data)
        >>> res = mod.fit(cov_type='clustered', cluster_entity=True)
        """
        parser = PanelFormulaParser(formula, data)
        entity_effect = parser.entity_effect
        time_effect = parser.time_effect
        dependent, exog = parser.data
        mod = cls(
            dependent,
            exog,
            entity_effects=entity_effect,
            time_effects=time_effect,
            weights=weights,
            other_effects=other_effects,
            singletons=singletons,
            drop_absorbed=drop_absorbed,
        )
        mod.formula = formula
        return mod

    def _lsmr_path(self) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Sparse implementation, works for all scenarios"""
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
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

        cats = []
        if self.entity_effects:
            cats.append(self.dependent.entity_ids)
        if self.time_effects:
            cats.append(self.dependent.time_ids)
        if self.other_effects:
            assert self._other_effect_cats is not None
            cats.append(self._other_effect_cats.values2d)
        cats = np.concatenate(cats, 1)

        wd, cond = dummy_matrix(cats, precondition=True)
        assert isinstance(wd, csc_matrix)
        if self._is_weighted:
            wd = wd.multiply(root_w_sparse)

        wx_mean = []
        for i in range(x.shape[1]):
            cond_mean = lsmr(wd, wx[:, i], atol=1e-8, btol=1e-8)[0]
            cond_mean /= cond
            wx_mean.append(cond_mean)
        wx_mean = np.column_stack(wx_mean)
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

    def _slow_path(self) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Frisch-Waugh-Lovell implementation, works for all scenarios"""
        w = self.weights.values2d
        root_w = np.sqrt(w)

        y = root_w * self.dependent.values2d
        x = root_w * self.exog.values2d
        if not self._has_effect:
            ybar = root_w @ lstsq(root_w, y, rcond=None)[0]
            y_effect, x_effect = np.zeros_like(y), np.zeros_like(x)
            return y, x, ybar, y_effect, x_effect

        drop_first = self._constant
        d = []
        if self.entity_effects:
            d.append(self.dependent.dummies("entity", drop_first=drop_first).values)
            drop_first = True
        if self.time_effects:
            d.append(self.dependent.dummies("time", drop_first=drop_first).values)
            drop_first = True
        if self.other_effects:
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                dummies = get_dummies(oe[c], drop_first=drop_first).astype(np.float64)
                d.append(dummies.values)
                drop_first = True

        d = np.column_stack(d)
        wd = root_w * d
        if self.has_constant:
            wd -= root_w * (w.T @ d / w.sum())
            z = np.ones_like(root_w)
            d -= z * (z.T @ d / z.sum())

        x_mean = lstsq(wd, x, rcond=None)[0]
        y_mean = lstsq(wd, y, rcond=None)[0]

        # Save fitted unweighted effects to use in eps calculation
        x_effects = d @ x_mean
        y_effects = d @ y_mean

        # Purge fitted, weighted values
        x = x - wd @ x_mean
        y = y - wd @ y_mean

        ybar = root_w @ lstsq(root_w, y, rcond=None)[0]
        return y, x, ybar, y_effects, x_effects

    def _choose_twoway_algo(self) -> bool:
        if not (self.entity_effects and self.time_effects):
            return False
        nentity, nobs = self.dependent.nentity, self.dependent.nobs
        nreg = min(nentity, nobs)
        if nreg < self.exog.shape[1]:
            return False
        # MiB
        reg_size = 8 * nentity * nobs * nreg // 2 ** 20
        low_memory = reg_size > 2 ** 10
        if low_memory:
            import warnings

            warnings.warn(
                "Using low-memory algorithm to estimate two-way model. Explicitly set "
                "low_memory=True to silence this message.  Set low_memory=False to use "
                "the standard algorithm that creates dummy variables for the smaller of "
                "the number of entities or number of time periods.",
                MemoryWarning,
            )
        return low_memory

    def _fast_path(self, low_memory: bool) -> Tuple[NDArray, NDArray, NDArray]:
        """Dummy-variable free estimation without weights"""
        _y = self.dependent.values2d
        _x = self.exog.values2d
        ybar = np.asarray(_y.mean(0))

        if not self._has_effect:
            return _y, _x, ybar

        y_gm = ybar
        x_gm = _x.mean(0)

        y = self.dependent
        x = self.exog

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
            y = cast(PanelData, y.general_demean(groups))
            x = cast(PanelData, x.general_demean(groups))
        elif self.entity_effects and self.time_effects:
            y = cast(PanelData, y.demean("both", low_memory=low_memory))
            x = cast(PanelData, x.demean("both", low_memory=low_memory))
        elif self.entity_effects:
            y = cast(PanelData, y.demean("entity"))
            x = cast(PanelData, x.demean("entity"))
        else:  # self.time_effects
            y = cast(PanelData, y.demean("time"))
            x = cast(PanelData, x.demean("time"))

        y_arr = y.values2d
        x_arr = x.values2d

        if self.has_constant:
            y_arr = y_arr + y_gm
            x_arr = x_arr + x_gm
        else:
            ybar = np.asarray(0.0)

        return y_arr, x_arr, ybar

    def _weighted_fast_path(
        self, low_memory: bool
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Dummy-variable free estimation with weights"""
        y_arr = self.dependent.values2d
        x_arr = self.exog.values2d
        w = self.weights.values2d
        root_w = cast(NDArray, np.sqrt(w))
        wybar = root_w * (w.T @ y_arr / w.sum())

        if not self._has_effect:
            wy = root_w * self.dependent.values2d
            wx = root_w * self.exog.values2d
            y_effect, x_effect = np.zeros_like(wy), np.zeros_like(wx)
            return wy, wx, wybar, y_effect, x_effect

        wy_gm = wybar
        wx_gm = root_w * (w.T @ x_arr / w.sum())

        y = self.dependent
        x = self.exog

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
            wy = y.demean("both", weights=self.weights, low_memory=low_memory)
            wx = x.demean("both", weights=self.weights, low_memory=low_memory)
        elif self.entity_effects:
            wy = y.demean("entity", weights=self.weights)
            wx = x.demean("entity", weights=self.weights)
        else:  # self.time_effects
            wy = y.demean("time", weights=self.weights)
            wx = x.demean("time", weights=self.weights)

        wy = wy.values2d
        wx = wx.values2d

        if self.has_constant:
            wy += wy_gm
            wx += wx_gm
        else:
            wybar = 0

        wy_effects = y.values2d - wy / root_w
        wx_effects = x.values2d - wx / root_w

        return wy, wx, wybar, wy_effects, wx_effects

    def _info(self) -> Tuple[Series, Series, DataFrame]:
        """Information about model effects and panel structure"""

        entity_info, time_info, other_info = super(PanelOLS, self)._info()

        if self.other_effects:
            other_info = []
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                name = "Observations per group (" + str(c) + ")"
                other_info.append(
                    panel_structure_stats(oe[c].values.astype(np.int32), name)
                )
            other_info = DataFrame(other_info)

        return entity_info, time_info, other_info

    @staticmethod
    def _is_effect_nested(effects: NDArray, clusters: NDArray) -> bool:
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
        **cov_config: Union[bool, float, str, NDArray, DataFrame, PanelData],
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
            return not self._is_effect_nested(effects, cast(NDArray, clusters))
        return True  # Default case for 2-way -- not completely clear

    def fit(
        self,
        *,
        use_lsdv: bool = False,
        use_lsmr: bool = False,
        low_memory: Optional[bool] = None,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        auto_df: bool = True,
        count_effects: bool = True,
        **cov_config: Union[bool, float, str, NDArray, DataFrame, PanelData],
    ) -> PanelEffectsResults:
        """
        Estimate model parameters

        Parameters
        ----------
        use_lsdv : bool, optional
            Flag indicating to use the Least Squares Dummy Variable estimator
            to eliminate effects.  The default value uses only means and does
            note require constructing dummy variables for each effect.
        use_lsmr : bool, optional
            Flag indicating to use LSDV with the Sparse Equations and Least
            Squares estimator to eliminate the fixed effects.
        low_memory : {bool, None}
            Flag indicating whether to use a low-memory algorithm when a model
            contains two-way fixed effects. If `None`, the choice is taken
            automatically, and the low memory algorithm is used if the
            required dummy variable array is both larger than then array of
            regressors in the model and requires more than 1 GiB .
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        auto_df : bool, optional
            Flag indicating that the treatment of estimated effects in degree
            of freedom adjustment is automatically handled. This is useful
            since clustered standard errors that are clustered using the same
            variable as an effect do not require degree of freedom correction
            while other estimators such as the unadjusted covariance do.
        count_effects : bool, optional
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
        >>> res = mod.fit(cov_type='clustered', cluster_entity=True)

        Notes
        -----
        Three covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer valued, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters
          * ``cluster_time`` - Boolean indicating to use time clusters

        * 'kernel' - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """

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
                y, x, ybar = self._fast_path(low_memory=low_memory)
                y_effects = np.array([0.0])
                x_effects = np.zeros(x.shape[1])
            else:
                y, x, ybar, y_effects, x_effects = self._weighted_fast_path(
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
                check_absorbed(x, [str(var) for var in self.exog.vars])
            else:
                # TODO: Need to special case the constant here when determining which to retain
                # since we always want to retain the constant if present
                retain = not_absorbed(x, self._constant, self._constant_index)
                if not retain:
                    raise ValueError(
                        "All columns in exog have been fully absorbed by the included"
                        " effects. This model cannot be estimated."
                    )
                if len(retain) != x.shape[1]:
                    drop = set(range(x.shape[1])).difference(retain)
                    dropped = ", ".join([str(self.exog.vars[i]) for i in drop])
                    import warnings

                    warnings.warn(
                        absorbing_warn_msg.format(absorbed_variables=dropped),
                        AbsorbingEffectWarning,
                    )
                    x = x[:, retain]
                    # Update constant index loc
                    if self._constant:
                        assert isinstance(self._constant_index, int)
                        self._constant_index = int(
                            np.argwhere(np.array(retain) == self._constant_index)
                        )

                    # Adjust exog
                    self.exog = PanelData(self.exog.dataframe.iloc[:, retain])
                    x_effects = x_effects[retain]

        params = lstsq(x, y, rcond=None)[0]
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
        _x = self.exog.values2d
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

        sigma2_tot = float(eps_effects.T @ eps_effects / nobs)
        sigma2_eps = float(eps.T @ eps / nobs)
        sigma2_effects = sigma2_tot - sigma2_eps
        rho = sigma2_effects / sigma2_tot if sigma2_tot > 0.0 else 0.0

        resid_ss = float(weps.T @ weps)
        if self.has_constant:
            mu = ybar
        else:
            mu = np.array([0.0])
        total_ss = float((y - mu).T @ (y - mu))
        r2 = 1 - resid_ss / total_ss if total_ss > 0.0 else 0.0

        root_w = cast(NDArray, np.sqrt(self.weights.values2d))
        y_ex = root_w * self.dependent.values2d
        mu_ex = 0
        if (
            self.has_constant
            or self.entity_effects
            or self.time_effects
            or self.other_effects
        ):
            mu_ex = root_w * ((root_w.T @ y_ex) / (root_w.T @ root_w))
        total_ss_ex_effect = float((y_ex - mu_ex).T @ (y_ex - mu_ex))
        r2_ex_effects = (
            1 - resid_ss / total_ss_ex_effect if total_ss_ex_effect > 0.0 else 0.0
        )

        res = self._postestimation(params, cov, debiased, df_resid, weps, y, x, root_w)
        ######################################
        # Pooled f-stat
        ######################################
        if self.entity_effects or self.time_effects or self.other_effects:
            wy, wx = root_w * self.dependent.values2d, root_w * self.exog.values2d
            df_num, df_denom = (df_model - wx.shape[1]), df_resid
            if not self.has_constant:
                # Correction for when models does not have explicit constant
                wy -= root_w * lstsq(root_w, wy, rcond=None)[0]
                wx -= root_w * lstsq(root_w, wx, rcond=None)[0]
                df_num -= 1
            weps_pooled = wy - wx @ lstsq(wx, wy, rcond=None)[0]
            resid_ss_pooled = float(weps_pooled.T @ weps_pooled)
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

        return PanelEffectsResults(res)


class BetweenOLS(_PanelModelBase):
    r"""
    Between estimator for panel data

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity)
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        \bar{y}_{i}=  \beta^{\prime}\bar{x}_{i}+\bar{\epsilon}_{i}

    where :math:`\bar{z}` is the time-average.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> None:
        super(BetweenOLS, self).__init__(dependent, exog, weights=weights)
        self._cov_estimators = CovarianceManager(
            self.__class__.__name__,
            HomoskedasticCovariance,
            HeteroskedasticCovariance,
            ClusteredCovariance,
        )

    def _setup_clusters(
        self, cov_config: Dict[str, Union[bool, float, str, PanelDataLike]]
    ) -> Dict[str, Union[bool, float, str, NDArray, DataFrame, PanelData]]:
        """Return covariance estimator reformat clusters"""
        cov_config_upd = cov_config.copy()
        if "clusters" not in cov_config:
            return cov_config_upd

        clusters = cov_config.get("clusters", None)
        if clusters is not None:
            clusters_panel = self.reformat_clusters(clusters)
            cluster_max = np.nanmax(clusters_panel.values3d, axis=1)
            delta = cluster_max - np.nanmin(clusters_panel.values3d, axis=1)
            if np.any(delta != 0):
                raise ValueError("clusters must not vary within an entity")

            index = clusters_panel.panel.minor_axis
            reindex = clusters_panel.entities
            clusters_frame = DataFrame(
                cluster_max.T, index=index, columns=clusters_panel.vars
            )
            clusters_frame = clusters_frame.loc[reindex].astype(np.int64)
            cov_config_upd["clusters"] = clusters_frame

        return cov_config_upd

    def fit(
        self,
        *,
        reweight: bool = False,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: Union[bool, float, str, NDArray, DataFrame, PanelData],
    ) -> PanelResults:
        """
        Estimate model parameters

        Parameters
        ----------
        reweight : bool
            Flag indicating to reweight observations if the input data is
            unbalanced using a WLS estimator.  If weights are provided, these
            are accounted for when reweighting. Has no effect on balanced data.
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        PanelResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import BetweenOLS
        >>> mod = BetweenOLS(y, x)
        >>> res = mod.fit(cov_type='robust')

        Notes
        -----
        Three covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables

        When using a clustered covariance estimator, all cluster ids must be
        identical within an entity.
        """
        y, x, w = self._prepare_between()
        if np.all(self.weights.values2d == 1.0) and not reweight:
            w = root_w = np.ones_like(y)
        else:
            root_w = cast(NDArray, np.sqrt(w))

        wx = root_w * x
        wy = root_w * y
        params = lstsq(wx, wy, rcond=None)[0]

        df_resid = y.shape[0] - x.shape[1]
        df_model = (x.shape[1],)
        nobs = y.shape[0]
        cov_config = self._setup_clusters(cov_config)
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.get("extra_df", 0)
            assert isinstance(_extra_df, (int, str))
            extra_df = int(_extra_df)
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            wx,
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )
        weps = wy - wx @ params
        index = self.dependent.index
        fitted = DataFrame(self.exog.values2d @ params, index, ["fitted_values"])
        eps = y - x @ params
        effects = DataFrame(eps, self.dependent.entities, ["estimated_effects"])
        entities = fitted.index.levels[0][fitted.index.codes[0]]
        effects = effects.loc[entities]
        effects.index = fitted.index
        dep = self.dependent.dataframe
        fitted = fitted.reindex(dep.index)
        effects = effects.reindex(dep.index)
        idiosyncratic = DataFrame(
            np.asarray(dep) - np.asarray(fitted) - np.asarray(effects),
            dep.index,
            ["idiosyncratic"],
        )

        residual_ss = float(weps.T @ weps)
        e = y
        if self._constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(w.T @ (e ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, root_w
        )
        res.update(
            dict(
                df_resid=df_resid,
                df_model=df_model,
                nobs=nobs,
                residual_ss=residual_ss,
                total_ss=total_ss,
                r2=r2,
                wresids=weps,
                resids=eps,
                index=self.dependent.entities,
                fitted=fitted,
                effects=effects,
                idiosyncratic=idiosyncratic,
            )
        )

        return PanelResults(res)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> "BetweenOLS":
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array_like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array_like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        BetweenOLS
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import BetweenOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = BetweenOLS.from_formula('y ~ 1 + x1', panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data)
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights)
        mod.formula = formula
        return mod


class FirstDifferenceOLS(_PanelModelBase):
    r"""
    First difference model for panel data

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity)
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        \Delta y_{it}=\beta^{\prime}\Delta x_{it}+\Delta\epsilon_{it}
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ):
        super(FirstDifferenceOLS, self).__init__(dependent, exog, weights=weights)
        if self._constant:
            raise ValueError(
                "Constants are not allowed in first difference regressions."
            )
        if self.dependent.nobs < 2:
            raise ValueError("Panel must have at least 2 time periods")

    def _setup_clusters(
        self, cov_config: Dict[str, Union[bool, float, str, PanelDataLike]]
    ) -> Dict[str, Union[bool, float, str, DataFrame]]:
        cov_config_upd = cov_config.copy()
        cluster_types = ("clusters", "cluster_entity")
        common = set(cov_config.keys()).intersection(cluster_types)
        if not common:
            return cov_config_upd

        clusters = cov_config.get("clusters", None)
        clusters_frame: Optional[DataFrame] = None
        if clusters is not None:
            clusters_panel = self.reformat_clusters(clusters)
            fd = clusters_panel.first_difference()
            fd_array = fd.values2d
            if np.any(fd_array.flat[np.isfinite(fd_array.flat)] != 0):
                raise ValueError(
                    "clusters must be identical for values used "
                    "to compute the first difference"
                )
            clusters_frame = clusters_panel.dataframe.copy()

        cluster_entity = cov_config_upd.pop("cluster_entity", False)
        if cluster_entity:
            group_ids = self.dependent.entity_ids.squeeze()
            name = "cov.cluster.entity"
            group_ids = Series(group_ids, index=self.dependent.index, name=name)
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)
        cluster_data = PanelData(clusters_frame)
        values = cluster_data.values3d[:, 1:]
        cluster_frame = panel_to_frame(
            values,
            cluster_data.panel.items,
            cluster_data.panel.major_axis[1:],
            cluster_data.panel.minor_axis,
            True,
        )
        cluster_frame = PanelData(cluster_frame).dataframe
        cluster_frame = cluster_frame.loc[self.dependent.first_difference().index]
        cluster_frame = cluster_frame.astype(np.int64)

        cov_config_upd["clusters"] = (
            cluster_frame.values if cluster_frame is not None else None
        )

        return cov_config_upd

    def fit(
        self,
        *,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: Union[bool, float, str, NDArray, DataFrame, PanelData],
    ) -> PanelResults:
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator. See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        PanelResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import FirstDifferenceOLS
        >>> mod = FirstDifferenceOLS(y, x)
        >>> robust = mod.fit(cov_type='robust')
        >>> clustered = mod.fit(cov_type='clustered', cluster_entity=True)

        Notes
        -----
        Three covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters

        * 'kernel' - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.

        When using a clustered covariance estimator, all cluster ids must be
        identical within a first difference.  In most scenarios, this requires
        ids to be identical within an entity.
        """
        y_fd = self.dependent.first_difference()
        time_ids = y_fd.time_ids
        entity_ids = y_fd.entity_ids
        index = y_fd.index
        y = y_fd.values2d
        x = self.exog.first_difference().values2d

        if np.all(self.weights.values2d == 1.0):
            w = root_w = np.ones_like(y)
        else:
            w = cast(NDArray, 1.0 / self.weights.values3d)
            w = w[:, :-1] + w[:, 1:]
            w = cast(NDArray, 1.0 / w)
            w_frame = panel_to_frame(
                w,
                self.weights.panel.items,
                self.weights.panel.major_axis[1:],
                self.weights.panel.minor_axis,
                True,
            )
            w_frame = w_frame.reindex(self.weights.index).dropna(how="any")
            index = w_frame.index
            w = w_frame.to_numpy()

            w /= w.mean()
            root_w = cast(NDArray, np.sqrt(w))

        wx = root_w * x
        wy = root_w * y
        params = lstsq(wx, wy, rcond=None)[0]
        df_resid = y.shape[0] - x.shape[1]
        cov_config = self._setup_clusters(cov_config)
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.get("extra_df", 0)
            assert isinstance(_extra_df, (int, str))
            extra_df = int(_extra_df)

        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            wx,
            params,
            entity_ids,
            time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )

        weps = wy - wx @ params
        fitted = DataFrame(
            self.exog.values2d @ params, self.dependent.index, ["fitted_values"]
        )
        idiosyncratic = DataFrame(
            self.dependent.values2d - fitted.values,
            self.dependent.index,
            ["idiosyncratic"],
        )
        effects = DataFrame(
            np.full_like(np.asarray(fitted), np.nan),
            self.dependent.index,
            ["estimated_effects"],
        )
        eps = y - x @ params

        residual_ss = float(weps.T @ weps)
        total_ss = float(w.T @ (y ** 2))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, root_w
        )
        res.update(
            dict(
                df_resid=df_resid,
                df_model=x.shape[1],
                nobs=y.shape[0],
                residual_ss=residual_ss,
                total_ss=total_ss,
                r2=r2,
                resids=eps,
                wresids=weps,
                index=index,
                fitted=fitted,
                effects=effects,
                idiosyncratic=idiosyncratic,
            )
        )

        return PanelResults(res)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> "FirstDifferenceOLS":
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array_like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array_like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        FirstDifferenceOLS
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import FirstDifferenceOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = FirstDifferenceOLS.from_formula('y ~ x1', panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data)
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights)
        mod.formula = formula
        return mod


class RandomEffects(_PanelModelBase):
    r"""
    One-way Random Effects model for panel data

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity)
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        y_{it} = \beta^{\prime}x_{it} + u_i + \epsilon_{it}

    where :math:`u_i` is a shock that is independent of :math:`x_{it}` but
    common to all entities i.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> None:
        super().__init__(dependent, exog, weights=weights)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> "RandomEffects":
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array_like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array_like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        RandomEffects
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import RandomEffects
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = RandomEffects.from_formula('y ~ 1 + x1', panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data)
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights)
        mod.formula = formula
        return mod

    def fit(
        self,
        *,
        small_sample: bool = False,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: Union[bool, float, str, NDArray, DataFrame, PanelData],
    ) -> RandomEffectsResults:
        """
        Estimate model parameters

        Parameters
        ----------
        small_sample : bool, default False
            Apply a small-sample correction to the estimate of the variance of
            the random effect.
        cov_type : str, default "unadjusted"
            Name of covariance estimator. See Notes.
        debiased : bool, default True
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        RandomEffectsResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import RandomEffects
        >>> mod = RandomEffects(y, x)
        >>> res = mod.fit(cov_type='clustered', cluster_entity=True)

        Notes
        -----
        Four covariance estimators are supported:

        * 'unadjusted', 'homoskedastic' - Assume residual are homoskedastic
        * 'robust', 'heteroskedastic' - Control for heteroskedasticity using
          White's estimator
        * 'clustered` - One or two way clustering.  Configuration options are:

          * ``clusters`` - Input containing containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters
          * ``cluster_time`` - Boolean indicating to use time clusters

        * 'kernel' - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """
        w = self.weights.values2d
        root_w = cast(NDArray, np.sqrt(w))
        demeaned_dep = self.dependent.demean("entity", weights=self.weights)
        demeaned_exog = self.exog.demean("entity", weights=self.weights)
        assert isinstance(demeaned_dep, PanelData)
        assert isinstance(demeaned_exog, PanelData)
        y = demeaned_dep.values2d
        x = demeaned_exog.values2d
        if self.has_constant:
            w_sum = w.sum()
            y_gm = (w * self.dependent.values2d).sum(0) / w_sum
            x_gm = (w * self.exog.values2d).sum(0) / w_sum
            y += root_w * y_gm
            x += root_w * x_gm
        params = lstsq(x, y, rcond=None)[0]
        weps = y - x @ params

        wybar = self.dependent.mean("entity", weights=self.weights)
        wxbar = self.exog.mean("entity", weights=self.weights)
        params = lstsq(np.asarray(wxbar), np.asarray(wybar), rcond=None)[0]
        wu = np.asarray(wybar) - np.asarray(wxbar) @ params

        nobs = weps.shape[0]
        neffects = wu.shape[0]
        nvar = x.shape[1]
        sigma2_e = float(weps.T @ weps) / (nobs - nvar - neffects + 1)
        ssr = float(wu.T @ wu)
        t = np.asarray(self.dependent.count("entity"))
        unbalanced = np.ptp(t) != 0
        if small_sample and unbalanced:
            ssr = float((t * wu).T @ wu)
            wx = root_w * self.exog.dataframe
            means = wx.groupby(level=0).transform("mean").values
            denom = means.T @ means
            sums = wx.groupby(level=0).sum().values
            num = sums.T @ sums
            tr = np.trace(np.linalg.inv(denom) @ num)
            sigma2_u = max(0, (ssr - (neffects - nvar) * sigma2_e) / (nobs - tr))
        else:
            t_bar = neffects / ((1.0 / t).sum())
            sigma2_u = max(0, ssr / (neffects - nvar) - sigma2_e / t_bar)
        rho = sigma2_u / (sigma2_u + sigma2_e)

        theta = 1.0 - np.sqrt(sigma2_e / (t * sigma2_u + sigma2_e))
        theta_out = DataFrame(theta, columns=["theta"], index=wybar.index)
        wy = root_w * self.dependent.values2d
        wx = root_w * self.exog.values2d
        index = self.dependent.index
        reindex = index.levels[0][index.codes[0]]
        wybar = (theta * wybar).loc[reindex]
        wxbar = (theta * wxbar).loc[reindex]
        wy -= wybar.values
        wx -= wxbar.values
        params = lstsq(wx, wy, rcond=None)[0]

        df_resid = wy.shape[0] - wx.shape[1]
        cov_config = self._setup_clusters(cov_config)
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.get("extra_df", 0)
            assert isinstance(_extra_df, (int, str))
            extra_df = int(_extra_df)

        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            wx,
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )

        weps = wy - wx @ params
        eps = weps / root_w
        index = self.dependent.index
        fitted = DataFrame(self.exog.values2d @ params, index, ["fitted_values"])
        effects = DataFrame(
            self.dependent.values2d - np.asarray(fitted) - eps,
            index,
            ["estimated_effects"],
        )
        idiosyncratic = DataFrame(eps, index, ["idiosyncratic"])
        residual_ss = float(weps.T @ weps)
        wmu = 0
        if self.has_constant:
            wmu = root_w * lstsq(root_w, wy, rcond=None)[0]
        wy_demeaned = wy - wmu
        total_ss = float(wy_demeaned.T @ wy_demeaned)
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, root_w
        )
        res.update(
            dict(
                df_resid=df_resid,
                df_model=x.shape[1],
                nobs=y.shape[0],
                residual_ss=residual_ss,
                total_ss=total_ss,
                r2=r2,
                resids=eps,
                wresids=weps,
                index=index,
                sigma2_eps=sigma2_e,
                sigma2_effects=sigma2_u,
                rho=rho,
                theta=theta_out,
                fitted=fitted,
                effects=effects,
                idiosyncratic=idiosyncratic,
            )
        )

        return RandomEffectsResults(res)


class FamaMacBeth(_PanelModelBase):
    r"""
    Pooled coefficient estimator for panel data

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity)
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like, optional
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\beta^{\prime}x_{it}+\epsilon_{it}

    The Fama-MacBeth estimator is computed by performing T regressions, one
    for each time period using all available entity observations.  Denote the
    estimate of the model parameters as :math:`\hat{\beta}_t`.  The reported
    estimator is then

    .. math::

        \hat{\beta} = T^{-1}\sum_{t=1}^T \hat{\beta}_t

    While the model does not explicitly include time-effects, the
    implementation based on regressing all observation in a single
    time period is "as-if" time effects are included.

    Parameter inference is made using the set of T parameter estimates with
    either the standard covariance estimator or a kernel-based covariance,
    depending on ``cov_type``.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ):
        super(FamaMacBeth, self).__init__(dependent, exog, weights=weights)
        self._validate_blocks()

    def _validate_blocks(self) -> None:
        x = self._x
        root_w = np.sqrt(self._w)
        wx = root_w * x

        exog = self.exog.dataframe
        wx = DataFrame(
            wx[self._not_null], index=exog.notnull().index, columns=exog.columns
        )

        def validate_block(ex: NDArray) -> bool:
            return ex.shape[0] >= ex.shape[1] and matrix_rank(ex) == ex.shape[1]

        valid_blocks = wx.groupby(level=1).apply(validate_block)
        if not valid_blocks.any():
            err = (
                "Model cannot be estimated. All blocks of time-series observations are rank\n"
                "deficient, and so it is not possible to estimate any cross-sectional "
                "regressions."
            )
            raise ValueError(err)
        if valid_blocks.sum() < exog.shape[1]:
            import warnings

            warnings.warn(
                "The number of time-series observation available to estimate "
                "cross-sectional\nregressions, {0}, is less than the number of "
                "parameters in the model. Parameter\ninference is not "
                "available.".format(valid_blocks.sum()),
                InferenceUnavailableWarning,
            )
        elif valid_blocks.sum() < valid_blocks.shape[0]:
            import warnings

            warnings.warn(
                "{0} of the time-series regressions cannot be estimated due to "
                "deficient rank.".format(valid_blocks.shape[0] - valid_blocks.sum()),
                MissingValueWarning,
            )

    def fit(
        self,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        bandwidth: Optional[float] = None,
        kernel: Optional[str] = None,
    ) -> FamaMacBethResults:
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, default "unadjusted"
            Name of covariance estimator. See Notes.
        debiased : bool, default True
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        bandwidth : float, default None
            The bandwidth to use when cov_type is "kernel". If None, it is
            automatically computed.
        kernel : str, default None
            The kernel to use.  None chooses the default kernel.

        Returns
        -------
        PanelResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import FamaMacBeth
        >>> mod = FamaMacBeth(y, x)
        >>> res = mod.fit(cov_type='kernel', kernel='Parzen')

        Notes
        -----
        Two covariance estimators are supported:

        * 'unadjusted', 'homoskedastic', 'robust', 'heteroskedastic' use the
          standard covariance estimator of the T parameter estimates.
        * 'kernel' is a HAC estimator. Configurations options are:
        """
        y = self._y
        x = self._x
        root_w = np.sqrt(self._w)
        wy = root_w * y
        wx = root_w * x

        dep = self.dependent.dataframe
        exog = self.exog.dataframe
        index = self.dependent.index
        wy = DataFrame(wy[self._not_null], index=index, columns=dep.columns)
        wx = DataFrame(
            wx[self._not_null], index=exog.notnull().index, columns=exog.columns
        )

        yx = DataFrame(
            np.c_[wy.values, wx.values],
            columns=list(wy.columns) + list(wx.columns),
            index=wy.index,
        )

        def single(z: DataFrame) -> Series:
            exog = z.iloc[:, 1:].values
            if exog.shape[0] < exog.shape[1] or matrix_rank(exog) != exog.shape[1]:
                return Series([np.nan] * len(z.columns), index=z.columns)
            dep = z.iloc[:, :1].values
            params = lstsq(exog, dep, rcond=None)[0]
            return Series(np.r_[np.nan, params.ravel()], index=z.columns)

        all_params = yx.groupby(level=1).apply(single)
        all_params = all_params.iloc[:, 1:]
        params = all_params.mean(0).values[:, None]

        wy = np.asarray(wy)
        wx = np.asarray(wx)
        index = self.dependent.index
        fitted = DataFrame(self.exog.values2d @ params, index, ["fitted_values"])
        effects = DataFrame(np.full(fitted.shape, np.nan), index, ["estimated_effects"])
        idiosyncratic = DataFrame(
            self.dependent.values2d - fitted.values, index, ["idiosyncratic"]
        )

        eps = self.dependent.values2d - fitted.values
        weps = wy - wx @ params
        w = self.weights.values2d
        root_w = cast(NDArray, np.sqrt(w))
        #
        residual_ss = float(weps.T @ weps)
        y = e = self.dependent.values2d
        if self.has_constant:
            e = y - (w * y).sum() / w.sum()
        total_ss = float(w.T @ (e ** 2))
        r2 = 1 - residual_ss / total_ss

        if cov_type not in (
            "robust",
            "unadjusted",
            "homoskedastic",
            "heteroskedastic",
            "kernel",
        ):
            raise ValueError("Unknown cov_type")

        bandwidth = 0.0 if cov_type != "kernel" else bandwidth
        cov = FamaMacBethCovariance(
            wy,
            wx,
            params,
            all_params,
            debiased=debiased,
            kernel=kernel,
            bandwidth=bandwidth,
        )

        df_resid = wy.shape[0] - params.shape[0]
        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, root_w
        )
        index = self.dependent.index
        res.update(
            dict(
                df_resid=df_resid,
                df_model=x.shape[1],
                nobs=y.shape[0],
                residual_ss=residual_ss,
                total_ss=total_ss,
                r2=r2,
                resids=eps,
                wresids=weps,
                index=index,
                fitted=fitted,
                effects=effects,
                idiosyncratic=idiosyncratic,
                all_params=all_params,
            )
        )
        return FamaMacBethResults(res)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: Optional[PanelDataLike] = None,
    ) -> "FamaMacBeth":
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to patsy formula rules.
        data : array_like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array_like, optional
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.

        Returns
        -------
        FamaMacBeth
            Model specified using the formula

        Notes
        -----
        Unlike standard patsy, it is necessary to explicitly include a
        constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import BetweenOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = FamaMacBeth.from_formula('y ~ 1 + x1', panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data)
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights)
        mod.formula = formula
        return mod
