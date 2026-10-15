"""Results containers for panel models, and tools to compare them."""

from __future__ import annotations

from linearmodels.compat.statsmodels import Summary

from collections.abc import Mapping
import datetime as dt
from functools import cached_property
from typing import Any, NamedTuple, Union
import warnings

from formulaic.utils.context import capture_context
import numpy as np
from pandas import DataFrame, Series, concat
from scipy import stats
from statsmodels.iolib.summary import SimpleTable, fmt_2cols, fmt_params

from linearmodels.iv.results import default_txt_fmt, stub_concat, table_concat
from linearmodels.panel.covariance import HomoskedasticCovariance
from linearmodels.panel.data import PanelDataLike
from linearmodels.shared.alignment import same_labels
from linearmodels.shared.base import _ModelComparison, _SummaryStr
from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    InvalidTestStatistic,
    NormalTestStatistic,
    WaldTestStatistic,
    quadratic_form_test,
)
from linearmodels.shared.io import _str, add_star, pval_format
from linearmodels.shared.utility import AttrDict
import linearmodels.typing.data

__all__ = [
    "FamaMacBethResults",
    "HausmanTestResults",
    "PanelEffectsResults",
    "PanelResults",
    "RandomEffectsResults",
    "compare",
]


class PanelResults(_SummaryStr):
    """
    Results container for panel data models that do not include effects.

    Parameters
    ----------
    res : AttrDict
        The results of the estimation, which the ``fit`` method of the model
        constructs.
    """

    def __init__(self, res: AttrDict):
        self._params = res.params.squeeze()
        self._deferred_cov = res.deferred_cov
        self._debiased = res.debiased
        self._df_resid = res.df_resid
        self._df_model = res.df_model
        self._nobs = res.nobs
        self._name = res.name
        self._var_names = res.var_names
        self._residual_ss = res.residual_ss
        self._total_ss = res.total_ss
        self._r2 = res.r2
        self._r2w = res.r2w
        self._r2b = res.r2b
        self._r2o = res.r2o
        self._c2w = res.c2w
        self._c2b = res.c2b
        self._c2o = res.c2o
        self._s2 = res.s2
        self._entity_info = res.entity_info
        self._time_info = res.time_info
        self.model = res.model
        self._cov_type = res.cov_type
        self._datetime = dt.datetime.now()
        self._resids = res.resids
        self._wresids = res.wresids
        self._index = res.index
        self._f_info = res.f_info
        self._f_stat = res.f_stat
        self._loglik = res.loglik
        self._fitted = res.fitted
        self._effects = res.effects
        self._idiosyncratic = res.idiosyncratic
        self._original_index = res.original_index
        self._not_null = res.not_null

    @property
    def params(self) -> Series:
        """Estimated parameters."""
        return Series(self._params, index=self._var_names, name="parameter")

    @cached_property
    def cov(self) -> DataFrame:
        """Estimated covariance of parameters."""
        return DataFrame(
            self._deferred_cov(), columns=self._var_names, index=self._var_names
        )

    @property
    def std_errors(self) -> Series:
        """Estimated parameter standard errors."""
        return Series(np.sqrt(np.diag(self.cov)), self._var_names, name="std_error")

    @property
    def tstats(self) -> Series:
        """Parameter t-statistics."""
        return Series(self._params / self.std_errors, name="tstat")

    @cached_property
    def pvalues(self) -> Series:
        """
        Parameter p-vals. Uses t(df_resid) if ``debiased`` is True, else normal.
        """
        abs_tstats = np.abs(self.tstats)
        if self._debiased:
            pv = 2 * stats.t.sf(abs_tstats, self.df_resid)
        else:
            pv = 2 * stats.norm.sf(abs_tstats)
        return Series(pv, index=self._var_names, name="pvalue")

    @property
    def df_resid(self) -> int:
        """
        Residual degree of freedom.

        Notes
        -----
        Defined as nobs minus nvar minus the number of included effects, if any.
        """
        return self._df_resid

    @property
    def df_model(self) -> int:
        """
        Model degree of freedom.

        Notes
        -----
        Defined as nvar plus the number of included effects, if any.
        """
        return self._df_model

    @property
    def nobs(self) -> int:
        """Number of observations used to estimate the model."""
        return self._nobs

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def total_ss(self) -> float:
        """Total sum of squares."""
        return self._total_ss

    @property
    def model_ss(self) -> float:
        """Model sum of squares."""
        return self._total_ss - self._residual_ss

    @property
    def resid_ss(self) -> float:
        """Residual sum of squares."""
        return self._residual_ss

    @property
    def rsquared(self) -> float:
        """
        Model coefficient of determination.

        Returns
        -------
        float
            The coefficient of determination of the model.

        Notes
        -----
        The coefficient of determination is NaN if its total sum of squares is zero,
        up to rounding error, since it is not defined then. This happens if the
        dependent variable does not vary, or if nothing varies once the means or the
        effects of the model are removed. No error or warning is raised.

        The limit is the rounding error of the computation. If the effects are removed
        by an iterative algorithm (``use_lsmr=True`` in ``fit``), its error is larger,
        by default about 1e-8 relative to the size of the dependent variable, so that a
        dependent variable that the effects explain exactly can have a total sum of
        squares that is not recognized as zero.
        """
        return self._r2

    @property
    def rsquared_between(self) -> float:
        """
        Between coefficient of determination.

        Returns
        -------
        float
            Between coefficient of determination.

        Notes
        -----
        The between rsquared measures the fit of the time-averaged dependent
        variable on the time averaged dependent variables. It accounts for the
        weights used in the estimation of the model.

        See the mathematical reference in the documentation for the formal
        definition of this measure.

        It is NaN if its total sum of squares is zero, up to rounding error, i.e., if
        the time-averaged dependent variable is the same for every entity in a model
        with a constant, or is 0 for every entity in a model without one. No error or
        warning is raised.
        """
        return self._r2b

    @property
    def rsquared_within(self) -> float:
        """
        Within coefficient of determination.

        Returns
        -------
        float
            Within coefficient of determination.

        Notes
        -----
        The within rsquared measures the fit of the dependent purged of entity
        effects on the exogenous purged of entity effects. It accounts for the
        weights used in the estimation of the model.

        See the mathematical reference in the documentation for the formal
        definition of this measure.

        It is NaN if its total sum of squares is zero, up to rounding error, i.e., if
        the dependent variable does not vary within entities, which includes a panel
        with a single time period. No error or warning is raised.
        """
        return self._r2w

    @property
    def rsquared_overall(self) -> float:
        """
        Overall coefficient of determination.

        Returns
        -------
        float
            Overall coefficient of determination.

        Notes
        -----
        The overall rsquared measures the fit of the dependent
        variable on the dependent variables ignoring any included effects.
        It accounts for the weights used in the estimation of the model.

        See the mathematical reference in the documentation for the formal
        definition of this measure.

        It is NaN if its total sum of squares is zero, up to rounding error, i.e., if
        the dependent variable does not vary in a model with a constant, or is 0 in a
        model without one. No error or warning is raised.
        """
        return self._r2o

    @property
    def corr_squared_between(self) -> float:
        r"""
        Between coefficient of determination using squared correlation.

        Returns
        -------
        float
            Between coefficient of determination.

        Notes
        -----
        The between rsquared measures the fit of the time-averaged dependent
        variable on the time averaged dependent variables.

        This measure is based on the squared correlation between the
        entity-wise averaged dependent variables and their average
        predictions.

        .. math::

           Corr[\bar{y}_i, \bar{x}_i\hat{\beta}]

        This measure **does not** account for weights.

        It is NaN if the entity-wise averaged dependent variable is the same for every
        entity, up to rounding error, since the correlation is not defined, and it is 0
        if the predictions are the same for every entity. No error or warning is
        raised.
        """
        return self._c2b

    @property
    def corr_squared_within(self) -> float:
        r"""
        Within coefficient of determination using squared correlation.

        Returns
        -------
        float
            Within coefficient of determination.

        Notes
        -----
        The within rsquared measures the fit of the dependent purged of entity
        effects on the exogenous purged of entity effects.

        This measure is based on the squared correlation between the
        entity-wise demeaned dependent variables and their demeaned
        predictions.

        .. math::

           Corr[y_{it}-\bar{y}_i, (x_{it}-\bar{x}_i)\hat{\beta}]

        This measure **does not** account for weights.

        It is NaN if the dependent variable does not vary within entities, up to
        rounding error, since the correlation is not defined, and it is 0 if the
        demeaned predictions do not vary. No error or warning is raised.
        """
        return self._c2w

    @property
    def corr_squared_overall(self) -> float:
        r"""
        Overall coefficient of determination using squared correlation.

        Returns
        -------
        float
            Overall coefficient of determination.

        Notes
        -----
        The overall rsquared measures the fit of the dependent
        variable on the dependent variables ignoring any included effects.

        This measure is based on the squared correlation between the
        dependent variables and their predictions.

        .. math::

           Corr[y_{it}, x_{it}\hat{\beta}]

        This measure **does not** account for weights.

        It is NaN if the dependent variable does not vary, up to rounding error, since
        the correlation is not defined, and it is 0 if the predictions do not vary. No
        error or warning is raised.
        """
        return self._c2o

    @property
    def s2(self) -> float:
        """Residual variance estimator."""
        return self._s2

    @property
    def entity_info(self) -> Series:
        """Statistics on observations per entity."""
        return self._entity_info

    @property
    def time_info(self) -> Series:
        """Statistics on observations per time interval."""
        return self._time_info

    def conf_int(self, level: float = 0.95) -> DataFrame:
        """
        Confidence interval construction.

        Parameters
        ----------
        level : float, optional
            Confidence level for interval. The default is 0.95.

        Returns
        -------
        DataFrame
            Confidence interval of the form [lower, upper] for each parameter.

        Notes
        -----
        Uses a t(df_resid) if ``debiased`` is True, else normal.
        """
        ci_quantiles = np.array([(1 - level) / 2, 1 - (1 - level) / 2])
        if self._debiased:
            q: np.ndarray = stats.t.ppf(ci_quantiles, self.df_resid)
        else:
            q = stats.norm.ppf(ci_quantiles)
        q = q[None, :]
        params = np.asarray(self.params)[:, None]
        ci = params + np.asarray(self.std_errors)[:, None] * q
        return DataFrame(ci, index=self._var_names, columns=["lower", "upper"])

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results.

        Notes
        -----
        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """

        title = self.name + " Estimation Summary"
        mod = self.model

        top_left = [
            ("Dep. Variable:", mod.dependent.vars[0]),
            ("Estimator:", self.name),
            ("No. Observations:", self.nobs),
            ("Date:", self._datetime.strftime("%a, %b %d %Y")),
            ("Time:", self._datetime.strftime("%H:%M:%S")),
            ("Cov. Estimator:", self._cov_type),
            ("", ""),
            ("Entities:", str(int(self.entity_info["total"]))),
            ("Avg Obs:", _str(self.entity_info["mean"])),
            ("Min Obs:", _str(self.entity_info["min"])),
            ("Max Obs:", _str(self.entity_info["max"])),
            ("", ""),
            ("Time periods:", str(int(self.time_info["total"]))),
            ("Avg Obs:", _str(self.time_info["mean"])),
            ("Min Obs:", _str(self.time_info["min"])),
            ("Max Obs:", _str(self.time_info["max"])),
            ("", ""),
        ]

        is_invalid = np.isfinite(self.f_statistic.stat)
        f_stat = _str(self.f_statistic.stat) if is_invalid else "--"
        f_pval = pval_format(self.f_statistic.pval) if is_invalid else "--"
        f_dist = self.f_statistic.dist_name if is_invalid else "--"

        f_robust = _str(self.f_statistic_robust.stat) if is_invalid else "--"
        f_robust_pval = (
            pval_format(self.f_statistic_robust.pval) if is_invalid else "--"
        )
        f_robust_name = self.f_statistic_robust.dist_name if is_invalid else "--"

        top_right = [
            ("R-squared:", _str(self.rsquared)),
            ("R-squared (Between):", _str(self.rsquared_between)),
            ("R-squared (Within):", _str(self.rsquared_within)),
            ("R-squared (Overall):", _str(self.rsquared_overall)),
            ("Log-likelihood", _str(self._loglik)),
            ("", ""),
            ("F-statistic:", f_stat),
            ("P-value", f_pval),
            ("Distribution:", f_dist),
            ("", ""),
            ("F-statistic (robust):", f_robust),
            ("P-value", f_robust_pval),
            ("Distribution:", f_robust_name),
            ("", ""),
            ("", ""),
            ("", ""),
            ("", ""),
        ]

        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # create summary table instance
        smry = Summary()
        # Top Table
        # Parameter table
        fmt = fmt_2cols
        fmt["data_fmts"][1] = "%18s"

        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)
        if self.params.shape[0] == 0:
            # A model that only has effects has no table of parameters
            return smry

        param_data = np.c_[
            self.params.to_numpy()[:, None],
            self.std_errors.to_numpy()[:, None],
            self.tstats.to_numpy()[:, None],
            self.pvalues.to_numpy()[:, None],
            self.conf_int().to_numpy(),
        ]
        data = []
        for row in param_data:
            txt_row = []
            for i, v in enumerate(row):
                f = _str
                if i == 3:
                    f = pval_format
                txt_row.append(f(v))
            data.append(txt_row)
        title = "Parameter Estimates"
        table_stubs = list(self.params.index)
        header = ["Parameter", "Std. Err.", "T-stat", "P-value", "Lower CI", "Upper CI"]
        table = SimpleTable(
            data, stubs=table_stubs, txt_fmt=fmt_params, headers=header, title=title
        )
        smry.tables.append(table)

        return smry

    @property
    def resids(self) -> Series:
        """
        Model residuals.

        Notes
        -----
        These residuals are from the estimated model. They will not have the
        same shape as the original data whenever the model is estimated on
        transformed data which has a different shape.
        """
        return Series(self._resids.squeeze(), index=self._index, name="residual")

    def _out_of_sample(
        self,
        exog: PanelDataLike | None,
        data: DataFrame | None,
        missing: bool,
        context: Mapping[str, Any] | None = None,
    ) -> DataFrame:
        """Interface between model predict and predict for OOS fits"""
        if exog is not None and data is not None:
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog or data, but not both."
            )
        pred = self.model.predict(self.params, exog=exog, data=data, context=context)
        if not missing:
            pred = pred.loc[pred.notnull().all(axis=1)]
        return pred

    def predict(
        self,
        exog: PanelDataLike | None = None,
        *,
        data: DataFrame | None = None,
        fitted: bool = True,
        effects: bool = False,
        idiosyncratic: bool = False,
        missing: bool = False,
    ) -> DataFrame:
        """
        In- and out-of-sample predictions.

        Parameters
        ----------
        exog : PanelData, DataFrame, Series, ndarray or DataArray, optional
            Exogenous values to use in out-of-sample prediction (nobs by nexog).
            The containers are those of ``exog`` in the model. A list or a tuple
            raises a ``TypeError`` that names ``exog``. If neither `exog` nor
            `data` is provided, the in-sample values are returned.
        data : DataFrame, optional
            DataFrame to use for out-of-sample predictions when model was
            constructed using a formula.
        fitted : bool, optional
            Flag indicating whether to include the fitted values. The default
            is True. Only used for in-sample values.
        effects : bool, optional
            Flag indicating whether to include estimated effects. The default
            is False. Only used for in-sample values.
        idiosyncratic : bool, optional
            Flag indicating whether to include the estimated idiosyncratic
            shock. The default is False. Only used for in-sample values.
        missing : bool, optional
            Flag indicating to adjust for dropped observations. If True, the
            values returned will have the same size as the original input data
            before filtering missing values. The default is False.

        Returns
        -------
        DataFrame
            DataFrame containing columns for all selected output.

        Raises
        ------
        ValueError
            If both `exog` and `data` are provided, or if no in-sample output
            is selected.

        Notes
        -----
        `data` can only be used when the model was created using the formula
        interface.  `exog` can be used for both a model created using a formula
        or a model specified with dependent and exog arrays.

        When using `exog` to generate out-of-sample predictions, the variable
        order must match the variables in the original model.

        Idiosyncratic errors and effects are not available for out-of-sample
        predictions.
        """
        if not (exog is None and data is None):
            context = capture_context(1)
            return self._out_of_sample(exog, data, missing, context=context)
        out = []
        if fitted:
            out.append(self.fitted_values)
        if effects:
            out.append(self.estimated_effects)
        if idiosyncratic:
            out.append(self.idiosyncratic)
        if len(out) == 0:
            raise ValueError("At least one output must be selected")
        out_df: DataFrame = concat(out, axis=1)
        if missing:
            index = self._original_index
            out_df = out_df.reindex(index)
        return out_df

    @property
    def fitted_values(self) -> DataFrame:
        """Fitted values."""
        return self._fitted

    @property
    def estimated_effects(self) -> DataFrame:
        """
        Estimated effects.

        Notes
        -----
        The combined effect of all of the effects in the model, in one column.
        It is the sum of the entity, time and other effects of each
        observation. The effects are not available separately.
        See :ref:`panel-estimated-effects` for how to recover the entity and
        time effects of a model that has both.

        The estimated effects have no standard errors or any other inference.
        They are not estimated consistently when the number of time periods is
        fixed. The effect of an entity is estimated from its observations, so
        the error does not shrink when the number of entities grows. To make
        inference about the effects of a small number of entities, include
        them as dummy variables in the regression.

        The values are NaN for models that do not have effects, which are
        PooledOLS, FirstDifferenceOLS and FamaMacBeth. They are 0 for a
        PanelOLS model without effects.

        The effects of RandomEffects are theta times the entity mean of the
        difference between the dependent variable and the fitted values, where
        theta is in the results. They are all 0 if the estimated variance of
        the random effect is negative and so it is set to 0, since theta is
        then 0 and the model is pooled OLS. ``variance_decomposition`` shows
        whether this is the case.
        """
        return self._effects

    @property
    def idiosyncratic(self) -> DataFrame:
        r"""
        Idiosyncratic error.

        Notes
        -----
        Differs from resids since this is the estimated idiosyncratic shock
        from the data. It has the same dimension as the dependent data.
        The shape and nature of resids depends on the model estimated. These
        estimates only depend on the model estimated through the estimation
        of parameters and inclusion of effects, if any.

        For ``FirstDifferenceOLS`` the values are the dependent variable less
        the fitted values, :math:`y_{it} - x_{it}\hat{\beta}`, in levels. They
        are not the shocks of the differenced equation, which are in ``resids``.
        """
        return self._idiosyncratic

    @property
    def wresids(self) -> Series:
        """Weighted model residuals."""
        return Series(
            self._wresids.squeeze(), index=self._index, name="weighted residual"
        )

    def _pesaran_cd_inapplicable(self) -> str | None:
        """Reason why the Pesaran CD test is not valid for the model, if any"""
        from linearmodels.panel.model import BetweenOLS  # noqa: PLC0415

        if isinstance(self.model, BetweenOLS):
            return (
                "The Pesaran CD test is not applicable to BetweenOLS. The model "
                "is fitted to the entity means, so it has no time-varying "
                "residuals, and the series implied by its coefficients is not "
                "the model's error."
            )
        return None

    @cached_property
    def pesaran_cd(self) -> InapplicableTestStatistic | NormalTestStatistic:
        r"""
        Pesaran CD test of residual cross-sectional dependence.

        Returns
        -------
        NormalTestStatistic or InapplicableTestStatistic
            Statistic value, distribution and p-value. If the test is not
            valid for the model, an InapplicableTestStatistic that explains
            why in its reason.

        Notes
        -----
        Tests the null hypothesis that the model's idiosyncratic shocks are
        cross-sectionally independent [1]_. Let :math:`\hat{\rho}_{ij}` denote
        the sample correlation of the estimated idiosyncratic shocks for
        entities :math:`i` and :math:`j`, computed using their common time
        observations, and let :math:`T_{ij}` denote the number of overlapping
        observations. The statistic is

        .. math::

           CD = \frac{1}{\sqrt{M}}\sum_{i<j}\sqrt{T_{ij}}\hat{\rho}_{ij},

        where :math:`M` is the number of entity pairs with at least two
        overlapping observations and finite pairwise correlation estimates.
        Under the null, the statistic is asymptotically standard normal.

        The shocks are the model's ``idiosyncratic`` values, so the weights of
        a weighted model are not applied to them. The exception is
        ``FirstDifferenceOLS``, which uses ``resids``, the residuals of the
        differenced equation, as ``plm::pcdtest`` does for a pooled regression
        of the differenced data. Its ``idiosyncratic`` values are in levels,
        and are persistent when the errors are, for example a random walk,
        which is when first differences are used. Since the correlations are
        computed over time, the test assumes that the shocks are not serially
        correlated, and it over-rejects if they are persistent. Conversely, if
        the errors in levels are serially uncorrelated, the residuals of
        ``FirstDifferenceOLS`` are negatively autocorrelated (an MA(1) with
        autocorrelation -0.5) and the test over-rejects, with a size of about
        11% at the nominal 5% level in large panels. The test is then better
        applied to a ``PanelOLS`` model with entity effects.

        The test is not valid for every model and an InapplicableTestStatistic
        is returned for the following.

        * ``BetweenOLS``. The model is fitted to the entity means, so it has
          no time-varying residuals, and the series implied by its
          coefficients is not the model's error.
        * ``FamaMacBeth``. The model's errors are the residuals of the
          period-by-period cross-sectional regressions. The series implied by
          the averaged coefficients contains the common, time-varying term
          :math:`(\beta_t - \bar{\beta})^{\prime}x_{it}`, so it is
          cross-sectionally dependent by construction. Fama-MacBeth inference
          is also already robust to cross-sectional dependence.
        * Models with time effects, and models with other effects that
          change over time for some entity and have a group of several
          entities that is missing from some periods, such as period,
          industry-by-period or decade effects. These effects demean the
          residuals across entities that are observed in the same periods,
          which makes the residuals negatively correlated across entities
          and biases the statistic downwards. For example, when the model
          has time effects and the errors are independent, the statistic is
          centered at about :math:`-\sqrt{T/2}` and its standard deviation
          is far below 1, so it is not standard normal [2]_. Other effects
          that do not change over time for an entity, or that are present
          in every period, are not affected.

        Time dummy variables (for example ``C(year)``) among the regressors have
        the same effect on the residuals as time effects, and make the test
        invalid in the same way. The code cannot reliably detect
        such regressors, so that the test is returned without a warning and
        should not be used for a model that has them.

        The statistic is computed from dense N by N matrices of pairwise
        correlations and numbers of overlapping observations, where N is the
        number of entities. Computing them takes about 450 MB of memory when
        N is 4,000, and the memory use is proportional to the square of N.

        References
        ----------
        .. [1] Pesaran, M. H. (2021). General diagnostic tests for
           cross-sectional dependence in panels. Empirical Economics, 60(1),
           13-50.
        .. [2] Juodis, A. and Reese, S. (2022). The incidental parameters
           problem in testing for remaining cross-section correlation.
           Journal of Business & Economic Statistics, 40(3), 1191-1203.
        """
        from linearmodels.panel.model import FirstDifferenceOLS  # noqa: PLC0415

        reason = self._pesaran_cd_inapplicable()
        if reason is not None:
            return InapplicableTestStatistic(reason=reason, name="Pesaran CD test")
        if isinstance(self.model, FirstDifferenceOLS):
            eps = self.resids
        else:
            eps = self.idiosyncratic.iloc[:, 0]
        wide = eps.unstack(level=0)
        wide = wide.loc[:, wide.notnull().any(axis=0)]
        nentity = wide.shape[1]
        if nentity < 2:
            return InapplicableTestStatistic(
                reason="Pesaran CD test requires at least two entities.",
                name="Pesaran CD test",
            )

        corr = wide.corr(min_periods=2).to_numpy(dtype=float)
        counts = wide.notnull().to_numpy(dtype=np.int64).T @ wide.notnull().to_numpy(
            dtype=np.int64
        )
        selector = np.tril(np.ones_like(counts, dtype=bool), k=-1)
        selector &= counts >= 2
        selector &= np.isfinite(corr)
        npairs = int(selector.sum())
        if npairs == 0:
            return InapplicableTestStatistic(
                reason=(
                    "Pesaran CD test requires at least one pair of entities "
                    "with two or more overlapping observations."
                ),
                name="Pesaran CD test",
            )

        stat = float(
            np.sum(np.sqrt(counts[selector].astype(float)) * corr[selector])
            / np.sqrt(npairs)
        )
        null = "Idiosyncratic shocks are cross-sectionally independent"
        return NormalTestStatistic(stat, null, name="Pesaran CD test")

    @property
    def f_statistic_robust(self) -> WaldTestStatistic:
        r"""
        Joint test of significance for non-constant regressors.

        Returns
        -------
        WaldTestStatistic or InvalidTestStatistic
            Statistic value, distribution and p-value. An ``InvalidTestStatistic``
            is returned if the test cannot be computed, as described in Notes.

        Notes
        -----
        Implemented as a Wald test using the estimated parameter covariance,
        and so inherits any robustness that the choice of covariance estimator
        provides.

        .. math::

           W = \hat{\beta}_{-}' \hat{\Sigma}_{-}^{-1} \hat{\beta}_{-}

        where :math:`\hat{\beta}_{-}` does not include the model constant and
        :math:`\hat{\Sigma}_{-}` is the estimated covariance of the
        parameters, also excluding the constant.  The test statistic is
        distributed as :math:`\chi^2_{k}` where k is the number of non-
        constant parameters.

        If ``debiased`` is True, then the Wald statistic is divided by the
        number of restrictions and inference is made using an :math:`F_{k,df}`
        distribution where df is the residual degree of freedom from the model.

        An ``InvalidTestStatistic`` is returned if the model only has a constant,
        or if the covariance of the parameters is singular, e.g., the dependent
        variable is all zero, so that the model fits exactly.
        """
        from linearmodels.panel.model import _deferred_f  # noqa: PLC0415

        return _deferred_f(
            self.params, self.cov, self._debiased, self.df_resid, self._f_info
        )

    @property
    def f_statistic(self) -> WaldTestStatistic:
        r"""
        Joint test of significance for non-constant regressors.

        Returns
        -------
        WaldTestStatistic or InvalidTestStatistic
            Statistic value, distribution and p-value. An ``InvalidTestStatistic``
            is returned if the model has no regressors, only a constant, or no
            residual degrees of freedom.

        Notes
        -----
        Classical F-stat that is only correct under an assumption of
        homoskedasticity.  The test statistic is defined as

        .. math::

          F = \frac{(RSS_R - RSS_U)/ k}{RSS_U / df_U}

        where :math:`RSS_R` is the restricted sum of squares from the model
        where the coefficients on all exog variables is zero, excluding a
        constant if one was included. :math:`RSS_U` is the unrestricted
        residual sum of squares.  k is the number of non-constant regressors
        in the model and :math:`df_U` is the residual degree of freedom in the
        unrestricted model.  The test has an :math:`F_{k,df_U}` distribution.
        """
        return self._f_stat

    @property
    def loglik(self) -> float:
        r"""
        Log-likelihood of model.

        Notes
        -----
        The Gaussian log-likelihood of the regression that the model estimates, at
        the estimates and the maximum likelihood estimate of the error variance,

        .. math::

           -\frac{n}{2}\left(\ln 2\pi + \ln\hat{\sigma}^2 + 1\right)
           + \frac{1}{2}\sum_{i=1}^{n}\ln w_i

        where :math:`\hat{\sigma}^2 = \hat{\epsilon}'\hat{\epsilon}/n` is the
        residual sum of squares of the regression divided by its number of
        observations :math:`n`, and :math:`\hat{\epsilon}_i` are the weighted
        residuals, :math:`\sqrt{w_i}` times the residuals. The last term is the
        Jacobian of the weights, which is 0 if the model is not weighted. The
        variance of the error of observation :math:`i` is
        :math:`\sigma^2/w_i`, and so the log-likelihood is that of the weighted least
        squares regression of the original data, for example of ``WLS(...).llf`` in
        statsmodels for ``PooledOLS``. It does not depend on the scale of the weights,
        and the log-likelihoods of the models with different weights are comparable
        if they have the same observations, for example in an information
        criterion.

        The regression and its observations depend on the model.

        * ``PooledOLS`` - The regression of the observations. The weights are the
          weights of the model.
        * ``PanelOLS`` - The regression on the dummy variables of the effects, with
          the effects as parameters that are estimated. It is computed from the
          regression of the observations with the effects removed. The weights are
          the weights of the model.
        * ``BetweenOLS`` - The regression of the means of the entities. The
          observations are the entities, and the weights are the sums of the
          weights of the entities, scaled to a mean of one, if the model is weighted
          or ``reweight`` is True. Otherwise all of the weights are 1.
        * ``FirstDifferenceOLS`` - The regression of the first differences. The
          weights are those of the differences, :math:`1/(1/w_{it}+1/w_{it-1})`.
        * ``RandomEffects`` - The regression of the data that are weighted and
          quasi-demeaned. It is not the likelihood of the random effects model, which
          also depends on the variances of the effects.
        * ``FamaMacBeth`` - The pooled regression at the average of the estimates
          of the periods.

        .. versionchanged:: 7.1

           The log-likelihood of a weighted model includes the Jacobian of the
           weights. Earlier versions omitted it, so that the log-likelihood was that
           of the regression of the weighted data, which is not the likelihood of the
           original data, and the log-likelihoods of models with different weights
           were not comparable. For the pooled regression of the log wages in the wage
           panel on a constant, ``educ``, ``married`` and ``union`` with the hours as
           weights it was -3065.75, where ``WLS`` of statsmodels has -3153.16. The
           log-likelihoods of models without weights do not change.
        """
        return self._loglik

    def wald_test(
        self,
        restriction: linearmodels.typing.data.Float64Array | DataFrame | None = None,
        value: linearmodels.typing.data.Float64Array | Series | None = None,
        *,
        formula: str | list[str] | None = None,
    ) -> WaldTestStatistic:
        r"""
        Test linear equality constraints using a Wald test.

        Parameters
        ----------
        restriction : array_like, optional
            Array with q rows and nvar columns containing linear weights to
            apply to parameters when forming the restrictions. It is not
            possible to use both restriction and formula.
        value : array_like, optional
            Array with q elements containing the restricted values. If not
            provided, the restricted values are all 0.
        formula : str, list of str or dict of str to float, optional
            Formulaic linear constraints. The simplest formats are one of:

            * A single comma-separated string such as "x1=0, x2+x3=1"
            * A list of strings where each element is a single constraint
              such as ["x1=0", "x2+x3=1"]
            * A single string without commas to test simple constraints such
              as "x1=x2=x3=0"
            * A dictionary where each key is a parameter restriction and
              the corresponding value is the restriction value, e.g.,
              {"x1": 0, "x2+x3": 1}.

            It is not possible to use both ``restriction`` and ``formula``.
            The names in the restrictions are the entries of the index of
            ``params``. A name that is not valid Python, for example one that
            contains ``:`` or ``[`` such as ``C(year)[T.1987]:educ``, must be
            enclosed in back-ticks, e.g.,
            ``"`C(year)[T.1987]:educ` = 0"``.

        Returns
        -------
        WaldTestStatistic
            Test statistic for null that restrictions are valid.

        Raises
        ------
        ValueError
            If the model has no parameters, e.g., a PanelOLS model that only has
            effects, or if both `restriction` and `formula` are provided.

        Notes
        -----
        Hypothesis test examines whether :math:`H_0:C\theta=v` where the
        matrix C is ``restriction`` and v is ``value``. The test statistic
        has a :math:`\chi^2_q` distribution where q is the number of rows in C.

        Examples
        --------
        >>> from linearmodels.datasets import wage_panel
        >>> import statsmodels.api as sm
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = wage_panel.load()
        >>> year = pd.Categorical(data.year)
        >>> data = data.set_index(["nr", "year"])
        >>> data["year"] = year
        >>> from linearmodels.panel import PanelOLS
        >>> exog_vars = ["expersq", "union", "married", "year"]
        >>> exog = sm.add_constant(data[exog_vars])

        >>> mod = PanelOLS(data.lwage, exog, entity_effects=True)
        >>> fe_res = mod.fit()

        Test the restriction that union and married have 0 coefficients

        >>> restriction = np.zeros((2, 11))
        >>> restriction[0, 2] = 1
        >>> restriction[1, 3] = 1
        >>> value = np.array([0, 0])
        >>> wald_res = fe_res.wald_test(restriction, value)

        The same test using formulas

        >>> formula = "union = married = 0"
        >>> wald_res = fe_res.wald_test(formula=formula)

        A name that is not valid Python, such as the interaction of a
        categorical variable with a regressor, is enclosed in back-ticks

        >>> from linearmodels import PooledOLS
        >>> mod = PooledOLS.from_formula("lwage ~ 1 + union + C(year) * educ", data)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)
        >>> null = "`C(year)[T.1987]:educ` = 0, `C(year)[T.1986]:educ` = 0"
        >>> wald_res = res.wald_test(formula=null)
        """
        if self.params.shape[0] == 0:
            raise ValueError(
                "The model has no parameters, so that there is nothing to test. It "
                "only has effects, which have no inference."
            )
        return quadratic_form_test(
            self.params, self.cov, restriction=restriction, value=value, formula=formula
        )


def _is_concentrated_in_time(groups: Any, entity_ids: Any, time_ids: Any) -> bool:
    """
    Test whether groups of entities change over time and are missing in periods

    Removing the mean of a group that is the same in every period for an
    entity does not change the correlations over time. Removing the mean of a
    group that changes over time and is not observed in every period removes a
    component that varies over time and is shared by the entities in the group.
    A group with a single entity is not shared.
    """
    frame = DataFrame(
        {
            "group": np.asarray(groups).ravel(),
            "entity": np.asarray(entity_ids).ravel(),
            "time": np.asarray(time_ids).ravel(),
        }
    )
    if frame.groupby("entity")["group"].nunique().max() <= 1:
        return False
    by_group = frame.groupby("group")
    shared = by_group["entity"].nunique() > 1
    nperiod = frame["time"].nunique()
    return bool((by_group["time"].nunique()[shared] < nperiod).any())


class PanelEffectsResults(PanelResults):
    """
    Results container for panel data models that include effects.

    Parameters
    ----------
    res : AttrDict
        The results of the estimation, which the ``fit`` method of the model
        constructs.
    """

    def __init__(self, res: AttrDict) -> None:
        super().__init__(res)
        self._other_info = res.other_info
        self._f_pooled = res.f_pooled
        self._entity_effect = res.entity_effects
        self._time_effect = res.time_effects
        self._other_effect = res.other_effects
        self._rho = res.rho
        self._sigma2_eps = res.sigma2_eps
        self._sigma2_effects = res.sigma2_effects
        self._r2_ex_effects = res.r2_ex_effects
        self._effects = res.effects

    def _pesaran_cd_inapplicable(self) -> str | None:
        reason = (
            "The Pesaran CD test is not applicable to models with {effect}. "
            "These effects demean the residuals across entities that are "
            "observed in the same periods, which makes the residuals "
            "negatively correlated across entities and biases the statistic "
            "downwards, away from its null distribution (Juodis and Reese, "
            "2022)."
        )
        if self._time_effect:
            return reason.format(effect="time effects")
        if self._other_effect:
            other = self.model._other_effect_cats.dataframe
            entity_ids = self.model.dependent.entity_ids
            time_ids = self.model.dependent.time_ids
            for col in other:
                if _is_concentrated_in_time(other[col], entity_ids, time_ids):
                    return reason.format(
                        effect="other effects that are concentrated in some periods"
                    )
        return super()._pesaran_cd_inapplicable()

    @property
    def f_pooled(self) -> WaldTestStatistic:
        r"""
        Test that included effects are jointly zero.

        Returns
        -------
        WaldTestStatistic, InapplicableTestStatistic or InvalidTestStatistic
            Statistic value, distribution and p-value. It is an
            ``InapplicableTestStatistic`` if the effects do not use any degrees
            of freedom, e.g., a single entity with a constant, and an
            ``InvalidTestStatistic`` if the model has no residual degrees of
            freedom or its residual sum of squares is 0.

        Notes
        -----
        Joint test that all included effects are zero.  Only correct under an
        assumption of homoskedasticity.

        The test statistic is defined as

        .. math::

          F = \frac{(RSS_{pool}-RSS_{effect})/(df_{pool}-df_{effect})}
              {RSS_{effect}/df_{effect}}

        where :math:`RSS_{pool}` is the residual sum of squares from a no-
        effect (pooled) model. :math:`RSS_{effect}` is the residual sum of
        squares from a model with effects.  :math:`df_{pool}` is the residual
        degree of freedom in the pooled regression and :math:`df_{effect}` is
        the residual degree of freedom from the model with effects. The test
        has an :math:`F_{k,df_{effect}}` distribution where
        :math:`k=df_{pool}-df_{effect}`.
        """
        return self._f_pooled

    @property
    def included_effects(self) -> list[str]:
        """List of effects included in the model."""
        entity_effect = self._entity_effect
        time_effect = self._time_effect
        other_effect = self._other_effect
        effects = []
        if entity_effect or time_effect or other_effect:
            if entity_effect:
                effects.append("Entity")
            if time_effect:
                effects.append("Time")
            if other_effect:
                oe = self.model._other_effect_cats.dataframe
                effects.extend(["Other Effect (" + str(c) + ")" for c in oe])
        return effects

    @property
    def other_info(self) -> DataFrame | None:
        """Statistics on observations per group for other effects."""
        return self._other_info

    @property
    def rsquared_inclusive(self) -> float:
        """
        Model Coefficient of determination including fit of included effects.

        Returns
        -------
        float
            The coefficient of determination of the model and the effects.

        Notes
        -----
        It is NaN if its total sum of squares is zero, up to rounding error, i.e., if
        the dependent variable does not vary. No error or warning is raised.
        """
        return self._r2_ex_effects

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results.

        Notes
        -----
        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """

        smry = super().summary

        is_invalid = np.isfinite(self.f_pooled.stat)
        f_pool = _str(self.f_pooled.stat) if is_invalid else "--"
        f_pool_pval = pval_format(self.f_pooled.pval) if is_invalid else "--"
        f_pool_name = self.f_pooled.dist_name if is_invalid else "--"

        extra_text = []
        if is_invalid:
            extra_text.append(f"F-test for Poolability: {f_pool}")
            extra_text.append(f"P-value: {f_pool_pval}")
            extra_text.append(f"Distribution: {f_pool_name}")
            extra_text.append("")

        if self.included_effects:
            effects = ", ".join(self.included_effects)
            extra_text.append("Included effects: " + effects)

        if self.other_info is not None:
            nrow = self.other_info.shape[0]
            plural = "s" if nrow > 1 else ""
            extra_text.append(f"Model includes {nrow} other effect{plural}")
            for c in self.other_info.T:
                col = self.other_info.T[c]
                extra_text.append(f"Other Effect {c}:")
                stats = "Avg Obs: {0}, Min Obs: {1}, Max Obs: {2}, Groups: {3}"
                stats = stats.format(
                    _str(col["mean"]),
                    _str(col["min"]),
                    _str(col["max"]),
                    int(col["total"]),
                )
                extra_text.append(stats)

        smry.add_extra_txt(extra_text)

        return smry

    @property
    def variance_decomposition(self) -> Series:
        """Decomposition of total variance into effects and residuals."""
        vals = [self._sigma2_effects, self._sigma2_eps, self._rho]
        index = ["Effects", "Residual", "Percent due to Effects"]
        return Series(vals, index=index, name="Variance Decomposition")


def _has_unadjusted_cov(res: PanelResults) -> bool:
    """Test whether results use the unadjusted covariance estimator

    The robust, clustered, kernel and autocorrelation-robust estimators all
    derive from the unadjusted estimator, so the exact type must be checked.
    """
    estimator = getattr(res._deferred_cov, "__self__", None)
    return type(estimator) is HomoskedasticCovariance


def _check_same_sample(res: PanelResults, other: PanelResults) -> None:
    """Check that two sets of results use the same data and weights"""
    if res.nobs != other.nobs:
        raise ValueError(
            "The models must be estimated using the same observations. "
            f"This model uses {res.nobs} and other uses {other.nobs}."
        )
    model, other_model = res.model, other.model
    if not same_labels(model.dependent.index, other_model.dependent.index):
        raise ValueError(
            "The models must be estimated using the same observations. The "
            "entities and time periods of the observations differ."
        )
    if not np.allclose(
        model.dependent.values2d, other_model.dependent.values2d, rtol=1e-10, atol=0.0
    ):
        raise ValueError(
            "The models must have the same dependent variable, but the "
            "dependent variables of the models differ."
        )
    # Weights are rescaled to have mean 1, so only the relative weights matter
    if not np.allclose(
        model.weights.values2d, other_model.weights.values2d, rtol=1e-10, atol=0.0
    ):
        raise ValueError(
            "The models must be estimated using the same weights, but the "
            "weights of the models differ."
        )


class HausmanTestResults(NamedTuple):  # numpydoc ignore=PR02
    """
    Results of a Hausman specification test.

    Parameters
    ----------
    statistic : WaldTestStatistic or InvalidTestStatistic
        The test statistic with its p-value, distribution and null hypothesis.
        An InvalidTestStatistic is used if the statistic is negative or not
        finite, which means that the models do not meet the assumptions of the
        test.
    estimates : DataFrame
        The coefficients included in the test. The columns are ``b0`` (from
        the model that is consistent under both hypotheses, ``other``), ``b1``
        (from the model that is efficient under the null), their difference
        ``b0-b1`` and ``Std. Err.``, the standard error of the difference.
        The standard error is NaN if the variance of the difference is
        negative.

    See Also
    --------
    RandomEffectsResults.hausman
        Compute the Hausman specification test.

    Notes
    -----
    The results are a named tuple, so the statistic and the estimates can be
    unpacked as in ``statistic, estimates = results`` or read using the
    ``statistic`` and ``estimates`` fields. Printing the results shows the
    statistic followed by the table of estimates.
    """

    statistic: WaldTestStatistic | InvalidTestStatistic
    estimates: DataFrame

    def __str__(self) -> str:
        legend = "b0: consistent under H0 and H1, b1: efficient under H0"
        return f"{self.statistic}\n\nCoefficients ({legend})\n{self.estimates}"

    def __repr__(self) -> str:
        return self.__str__()


class RandomEffectsResults(PanelResults):
    """
    Results container for random effect panel data models.

    Parameters
    ----------
    res : AttrDict
        The results of the estimation, which the ``fit`` method of the model
        constructs.
    """

    def __init__(self, res: AttrDict) -> None:
        super().__init__(res)
        self._theta = res.theta
        self._sigma2_effects = res.sigma2_effects
        self._sigma2_eps = res.sigma2_eps
        self._rho = res.rho

    @property
    def variance_decomposition(self) -> Series:
        """Decomposition of total variance into effects and residuals."""
        vals = [self._sigma2_effects, self._sigma2_eps, self._rho]
        index = ["Effects", "Residual", "Percent due to Effects"]
        return Series(vals, index=index, name="Variance Decomposition")

    @property
    def theta(self) -> DataFrame:
        """Values used in generalized demeaning."""
        return self._theta

    def hausman(
        self,
        other: PanelResults,
        include_constant: bool = False,
        sigmamore: bool = False,
        sigmaless: bool = False,
    ) -> HausmanTestResults:
        r"""
        Hausman specification test against results from a consistent model.

        Parameters
        ----------
        other : PanelResults
            Results from a model that is consistent whether or not the null
            hypothesis is true, typically a fixed effects (entity effects)
            model estimated using the same observations, dependent variable,
            regressors and weights as this model.
        include_constant : bool, optional
            Flag indicating whether to include the constant in the comparison.
            The default is False.
        sigmamore : bool, optional
            Flag indicating whether to base both covariance estimates on the
            error variance estimated by this model, which is efficient under
            the null. Cannot be used with ``sigmaless``. The default is False.
        sigmaless : bool, optional
            Flag indicating whether to base both covariance estimates on the
            error variance estimated by ``other``. Cannot be used with
            ``sigmamore``. The default is False.

        Returns
        -------
        HausmanTestResults
            A named tuple with the fields ``statistic``, which contains the
            test statistic, p-value, distribution and null as a
            WaldTestStatistic, and ``estimates``, a DataFrame with the
            coefficients included in the test. An InvalidTestStatistic is
            returned as the statistic when the statistic is negative or not
            finite. The columns of ``estimates`` are ``b0`` (from ``other``),
            ``b1`` (from this model), their difference ``b0-b1`` and
            ``Std. Err.``, the standard error of the difference. The standard
            error is NaN if the variance of the difference is negative.

        Raises
        ------
        TypeError
            If ``other`` is not a panel results object, or if either model
            was not estimated using the unadjusted covariance estimator.
        ValueError
            If ``sigmamore`` and ``sigmaless`` are both True, if the models
            were not estimated using the same observations, dependent variable
            and weights, or if the models do not have any coefficients in
            common.

        Notes
        -----
        The test statistic is

        .. math::

            H = (b_0 - b_1)^{\prime} (V_0 - V_1)^{-1} (b_0 - b_1)

        where :math:`b_0` and :math:`V_0` are the coefficients and covariance
        from ``other``, which is consistent under both the null and the
        alternative, and :math:`b_1` and :math:`V_1` are from this model,
        which is efficient under the null. The null hypothesis is that there
        is no systematic difference in the coefficients, which for a fixed
        effects model compared with a random effects model means that the
        effects are uncorrelated with the regressors. The statistic is
        asymptotically :math:`\chi^2_k`, where :math:`k` is the number of
        coefficients compared.

        Only coefficients found in both models are compared. The constant is
        excluded unless ``include_constant`` is True.

        The result is only meaningful when :math:`V_0 - V_1` is positive
        definite, which is guaranteed in large samples if the null is true. If
        it is not, a warning is issued and a generalized inverse is used. If
        the resulting statistic is negative or not finite, the data do not
        satisfy the assumptions of the test and an InvalidTestStatistic is
        returned. This can happen, for example, if ``other`` is a pooled
        model, since the unadjusted covariance of a pooled model ignores the
        entity effects.

        Both models must use the default unadjusted covariance estimator. The
        statistic also requires that the models are estimated using the same
        observations, the same dependent variable and the same weights, which
        are checked. The regressors are not checked: only coefficients with
        the same name in both models are compared.

        Effects that are included in ``other`` but not in this model are not
        tested separately. For example, if ``other`` includes time effects
        and this model does not, the null hypothesis is that the regressors
        are uncorrelated with the entity effects and with the time effects,
        and the test can reject because of the time effects. To test only the
        entity effects, include the time dummies as regressors in this model.
        Only the coefficients found in both models are compared, so the
        coefficients on the time dummies are not part of the test.

        References
        ----------
        Hausman, J. A. (1978). Specification tests in econometrics.
        Econometrica, 46(6), 1251-1271.

        Examples
        --------
        >>> from linearmodels import PanelOLS, RandomEffects
        >>> from linearmodels.datasets import wage_panel
        >>> data = wage_panel.load().set_index(["nr", "year"])
        >>> y = data["lwage"]
        >>> x = data[["hours"]]
        >>> fe_res = PanelOLS(y, x, entity_effects=True).fit()
        >>> re_res = RandomEffects(y, x).fit()
        >>> results = re_res.hausman(fe_res)
        >>> print(results)  # doctest: +ELLIPSIS
        Hausman specification test
        H0: No systematic difference in coefficients between models
        Statistic: ...
        P-value: ...
        Distributed: chi2(1)
        ...

        The statistic and the estimates can be unpacked.

        >>> test, estimates = results
        """
        if not isinstance(other, PanelResults):
            raise TypeError("other must be the results of a panel model.")
        if sigmamore and sigmaless:
            raise ValueError("sigmamore and sigmaless cannot both be True.")
        if not (_has_unadjusted_cov(self) and _has_unadjusted_cov(other)):
            raise TypeError(
                "The Hausman test can only be used with models estimated using "
                "the unadjusted covariance estimator."
            )
        _check_same_sample(self, other)

        in_other = set(other.params.index)
        common = [name for name in self.params.index if name in in_other]
        if not include_constant:
            constants: set[Any] = set()
            for res in (self, other):
                if res.model.has_constant:
                    loc = res.model._constant_index
                    assert isinstance(loc, int)
                    constants.add(res.model.exog.vars[loc])
            common = [name for name in common if name not in constants]
        if not common:
            raise ValueError("The models do not have any coefficients in common.")

        # The unadjusted covariance is s2 * inv(x'x), so using the error
        # variance of the other model only requires rescaling.
        cov0 = other.cov
        cov1 = self.cov
        if sigmamore:
            cov0 = cov0 * (self.s2 / other.s2)
        elif sigmaless:
            cov1 = cov1 * (other.s2 / self.s2)

        b0 = other.params[common]
        b1 = self.params[common]
        b_diff = b0 - b1
        var_diff = cov0.loc[common, common] - cov1.loc[common, common]
        diag = np.diag(var_diff)
        std_errors = Series(np.sqrt(np.where(diag >= 0, diag, np.nan)), index=common)
        estimates = DataFrame(
            {"b0": b0, "b1": b1, "b0-b1": b_diff, "Std. Err.": std_errors}
        )

        diff = var_diff.to_numpy()
        delta = b_diff.to_numpy()
        name = "Hausman specification test"
        if not (np.all(np.isfinite(diff)) and np.all(np.isfinite(delta))):
            invalid = InvalidTestStatistic(
                "The coefficients or covariances of the models are not finite, "
                "so the statistic cannot be computed.",
                name=name,
            )
            return HausmanTestResults(invalid, estimates)
        try:
            np.linalg.cholesky(diff)
            positive_definite = True
        except np.linalg.LinAlgError:
            positive_definite = False
        if positive_definite:
            stat = float(delta @ np.linalg.solve(diff, delta))
        else:
            warnings.warn(
                "Var(b0) - Var(b1) is not positive definite. A generalized "
                "inverse is used to compute the statistic.",
                UserWarning,
                stacklevel=2,
            )
            stat = float(delta @ np.linalg.pinv(diff, hermitian=True) @ delta)

        if not np.isfinite(stat) or stat < 0:
            problem = "not finite" if not np.isfinite(stat) else "negative"
            invalid = InvalidTestStatistic(
                f"The statistic is {problem}, so the models do not meet the "
                "assumptions of the Hausman test.",
                name=name,
            )
            return HausmanTestResults(invalid, estimates)
        null = "No systematic difference in coefficients between models"
        wald = WaldTestStatistic(stat, null, len(common), name=name)
        return HausmanTestResults(wald, estimates)


PanelModelResults = Union[PanelEffectsResults, PanelResults, RandomEffectsResults]


class FamaMacBethResults(PanelResults):
    """
    Results container for Fama MacBeth panel data models.

    Parameters
    ----------
    res : AttrDict
        The results of the estimation, which the ``fit`` method of the model
        constructs.
    """

    def __init__(self, res: AttrDict):
        super().__init__(res)
        self._all_params = res.all_params
        self._avg_r2 = res.avg_r2
        self._avg_adj_r2 = res.avg_adj_r2

    @property
    def df_resid(self) -> int:
        """
        Degrees of freedom of the t distribution used for inference.

        Returns
        -------
        int
            T - 1, where T is the number of cross-sectional regressions that
            are averaged.

        Notes
        -----
        The parameters are the average of T cross-sectional estimates, and
        their covariance is estimated from the same T estimates. If
        ``debiased`` is True, p-values and confidence intervals use a t
        distribution with T - 1 degrees of freedom, as in Fama and MacBeth
        (1973). The periods whose cross-section cannot be estimated are not
        counted. This is not nobs minus nvar, which is the residual degree of
        freedom of the pooled regression and the one that ``f_statistic``, the
        classical F-statistic of the pooled regression, uses. The joint test
        ``f_statistic_robust`` uses T - k denominator degrees of freedom, where
        k is the number of parameters that it tests.
        """
        return self._df_resid

    @property
    def f_statistic_robust(self) -> WaldTestStatistic:
        r"""
        Joint test of significance for non-constant regressors.

        Returns
        -------
        WaldTestStatistic or InvalidTestStatistic
            Statistic value, distribution and p-value. It is an
            ``InvalidTestStatistic`` if the model contains only a constant, or
            if the covariance of the parameters is singular, which it is if the
            number of periods that are averaged is not larger than the number
            of tested parameters, or if the dependent variable is all zero.

        Notes
        -----
        Implemented as a Wald test using the estimated parameter covariance,
        which is the covariance of the average of the T cross-sectional
        estimates that is selected by ``cov_type``.

        .. math::

           W = \hat{\beta}_{-}' \hat{\Sigma}_{-}^{-1} \hat{\beta}_{-}

        where :math:`\hat{\beta}_{-}` does not include the model constant and
        :math:`\hat{\Sigma}_{-}` is the estimated covariance of the
        parameters, also excluding the constant. Let :math:`k` be the number of
        non-constant parameters and :math:`T` the number of periods whose
        cross-sectional regression is estimated, which is ``df_resid`` plus 1.

        If ``debiased`` is True, which is the default, the statistic is

        .. math::

           F = \frac{T-k}{k(T-1)} W

        and it is compared with an :math:`F_{k,T-k}` distribution. This is
        the Hotelling T-squared test [1]_ of the hypothesis that the mean of
        the :math:`T` estimates is zero. It is exact when the :math:`T`
        estimates are independent draws from the same normal distribution and
        the covariance is the standard covariance of their average
        (``cov_type`` equal to "unadjusted", "homoskedastic", "robust" or
        "heteroskedastic"), and the test then has the size that it claims.
        With one tested parameter, :math:`F` is the square of the t-statistic
        and the distribution is the square of the :math:`t_{T-1}`
        distribution, so that the p-value is that of the parameter.

        With the kernel covariance ("kernel") the test is only an
        approximation. A kernel estimate of the long-run covariance from only
        :math:`T` estimates is much less precise than the sample covariance, so
        that the test can reject a true null hypothesis far too often even if
        the estimates are independent, as can the t-test of a single
        parameter, which is the case :math:`k=1`. In simulations of
        independent normal estimates with the default bandwidth, the 5% test
        rejected the null hypothesis in 21% of the samples with :math:`k=1` and
        :math:`T=10`, in 37% with :math:`k=3` and :math:`T=10`, and in 23% with
        :math:`k=3` and :math:`T=40`, where the test with the standard
        covariance rejected in 5%.

        The covariance is estimated from only :math:`T` estimates, which makes
        :math:`W/k` stochastically larger than an :math:`F_{k,df}` variable
        with :math:`df=T-1`, or with :math:`df=nobs-nvar` as in 7.0 and
        earlier, when there are several tested parameters. A test that uses
        those distributions rejects a true null hypothesis too often, and more
        with more tested parameters and fewer periods. The factor and the
        :math:`F_{k,T-k}` distribution correct this. In simulations of 20,000
        samples of 40 entities in which the null hypothesis is true, the
        tests of nominal size 5% with :math:`k=3` and :math:`T=10`, with
        :math:`k=5` and :math:`T=10`, and with :math:`k=5` and :math:`T=20`
        rejected the null hypothesis in 10%, 24% and 11% of the samples with the
        :math:`F_{k,T-1}` distribution, in 20%, 41% and 18% with the
        :math:`F_{k,nobs-nvar}` distribution of 7.0, and in 5% with the test
        described here.

        If ``debiased`` is False, the statistic is :math:`W` and the
        distribution is :math:`\chi^2_k`. The general
        :meth:`~linearmodels.panel.results.PanelResults.wald_test` of the
        results also compares its statistic with a chi-squared distribution,
        and does not apply this correction.

        References
        ----------
        .. [1] Hotelling, H. (1931). The generalization of Student's ratio.
           The Annals of Mathematical Statistics, 2(3), 360-378.
        """
        from linearmodels.panel.model import _hotelling_f  # noqa: PLC0415

        # The periods whose cross-section is estimated are df_resid + 1
        return _hotelling_f(
            self.params, self.cov, self._debiased, self._df_resid + 1, self._f_info
        )

    def _pesaran_cd_inapplicable(self) -> str | None:
        return (
            "The Pesaran CD test is not applicable to FamaMacBeth. The errors "
            "of the model are the residuals of the period-by-period "
            "cross-sectional regressions. The series implied by the averaged "
            "coefficients contains the common, time-varying term "
            "(beta_t - beta_bar)'x_it, so it is cross-sectionally dependent by "
            "construction. Fama-MacBeth inference is already robust to "
            "cross-sectional dependence."
        )

    @property
    def all_params(self) -> DataFrame:
        """
        The set of parameters estimated for each of the time periods.

        Returns
        -------
        DataFrame
            The parameters of each of the time periods, with shape (nobs,
            nparam).
        """
        return self._all_params

    @property
    def avg_rsquared(self) -> float:
        """
        The average coefficient of determination.

        This value contains the average of the individual rsquared values
        across the cross-sectional regressions, one for each time period. An
        R-squared value is only included in the average if a cross-section has
        full rank and if the dependent variable of the cross-section varies,
        since the coefficient of determination is NaN, up to rounding error, if
        its total sum of squares is zero.
        """
        return self._avg_r2

    @property
    def avg_adj_rsquared(self) -> float:
        """
        The average coefficient of determination, adjusted for sample size.

        This value contains the average of the individual adjusted rsquared
        values across the cross-sectional regressions, one for each time
        period. An R-squared value is only included in the average if a
        cross-section has full rank, if the number of observations in a
        cross-section is larger than the number of regressors, and if the
        dependent variable of the cross-section varies, since the coefficient
        of determination is NaN, up to rounding error, if its total sum of
        squares is zero.
        """
        return self._avg_adj_r2


class PanelModelComparison(_ModelComparison):
    """
    Comparison of multiple models.

    Parameters
    ----------
    results : list of results or dict of str to results
        Set of results to compare. The results are those of the panel models
        in the package. If a dict, the keys will be used as model names.
    precision : {"tstats", "std_errors", "std-errors", "pvalues"}, optional
        Estimator precision estimator to include in the comparison output. The
        default is "tstats". The names are not case sensitive.
    stars : bool, optional
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively. The
        default is False.

    Raises
    ------
    TypeError
        If a result is not the result of a panel model.
    ValueError
        If `precision` is not one of the supported values.
    """

    _supported = (
        PanelEffectsResults,
        PanelResults,
        RandomEffectsResults,
        FamaMacBethResults,
    )

    def __init__(
        self,
        results: list[PanelModelResults] | dict[str, PanelModelResults],
        *,
        precision: str = "tstats",
        stars: bool = False,
    ) -> None:
        super().__init__(results, precision=precision, stars=stars)

    @property
    def rsquared_between(self) -> Series:
        """
        Between coefficients of determination (R**2).

        Returns
        -------
        Series
            The between coefficient of determination of each model.

        Notes
        -----
        The value of a model is NaN if its between coefficient of determination is not
        defined, i.e., if the total sum of squares is zero, up to rounding error. See
        ``rsquared_between`` of the results of the models.
        """
        return self._get_property("rsquared_between")

    @property
    def rsquared_within(self) -> Series:
        """
        Within coefficients of determination (R**2).

        Returns
        -------
        Series
            The within coefficient of determination of each model.

        Notes
        -----
        The value of a model is NaN if its within coefficient of determination is not
        defined, i.e., if the total sum of squares is zero, up to rounding error. See
        ``rsquared_within`` of the results of the models.
        """
        return self._get_property("rsquared_within")

    @property
    def rsquared_overall(self) -> Series:
        """
        Overall coefficients of determination (R**2).

        Returns
        -------
        Series
            The overall coefficient of determination of each model.

        Notes
        -----
        The value of a model is NaN if its overall coefficient of determination is not
        defined, i.e., if the total sum of squares is zero, up to rounding error. See
        ``rsquared_overall`` of the results of the models.
        """
        return self._get_property("rsquared_overall")

    @property
    def estimator_method(self) -> Series:
        """Estimation methods."""
        return self._get_property("name")

    @property
    def cov_estimator(self) -> Series:
        """Covariance estimator descriptions."""
        return self._get_property("_cov_type")

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results.

        Notes
        -----
        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """

        smry = Summary()
        models = list(self._results.keys())
        title = "Model Comparison"
        stubs = [
            "Dep. Variable",
            "Estimator",
            "No. Observations",
            "Cov. Est.",
            "R-squared",
            "R-Squared (Within)",
            "R-Squared (Between)",
            "R-Squared (Overall)",
            "F-statistic",
            "P-value (F-stat)",
        ]
        dep_name = {}
        for key in self._results:
            dep_name[key] = self._results[key].model.dependent.vars[0]

        vals = concat(
            [
                Series(dep_name),
                self.estimator_method,
                self.nobs,
                self.cov_estimator,
                self.rsquared,
                self.rsquared_within,
                self.rsquared_between,
                self.rsquared_overall,
                self.f_statistic,
            ],
            axis=1,
        )
        vals_lst = [list(v) for v in vals.T.values]
        vals_lst[2] = [str(v) for v in vals_lst[2]]
        for i in range(4, len(vals_lst)):
            f = _str
            if i == 9:
                f = pval_format
            vals_lst[i] = [f(v) for v in vals_lst[i]]

        params = self.params
        precision = getattr(self, self._precision)
        pvalues = np.asarray(self.pvalues)
        params_fmt = []
        params_stub: list[str] = []
        for i in range(len(params)):
            formatted_and_starred = []
            for v, pv in zip(params.values[i], pvalues[i], strict=False):
                formatted_and_starred.append(add_star(_str(v), pv, self._stars))
            params_fmt.append(formatted_and_starred)

            precision_fmt = []
            for v in precision.values[i]:
                v_str = _str(v)
                v_str = f"({v_str})" if v_str.strip() else v_str
                precision_fmt.append(v_str)
            params_fmt.append(precision_fmt)
            params_stub.append(str(params.index[i]))
            params_stub.append(" ")

        if params_fmt:
            # None of the models has parameters if they only have effects
            vals_lst = table_concat((vals_lst, params_fmt))
            stubs = stub_concat((stubs, params_stub))

        all_effects = []
        for key in self._results:
            res = self._results[key]
            effects = getattr(res, "included_effects", [])
            all_effects.append(effects)

        neffect = max(map(len, all_effects))
        effects = []
        effects_stub = ["Effects"]
        for i in range(neffect):
            if i > 0:
                effects_stub.append("")
            row = []
            for j in range(len(self._results)):
                effect = all_effects[j]
                if len(effect) > i:
                    row.append(effect[i])
                else:
                    row.append("")
            effects.append(row)
        if effects:
            vals_lst = table_concat((vals_lst, effects))
            stubs = stub_concat((stubs, effects_stub))

        txt_fmt = default_txt_fmt.copy()
        txt_fmt["data_aligns"] = "r"
        txt_fmt["header_align"] = "r"
        table = SimpleTable(
            vals_lst, headers=models, title=title, stubs=stubs, txt_fmt=txt_fmt
        )
        smry.tables.append(table)
        prec_type = self._PRECISION_TYPES[self._precision]
        smry.add_extra_txt([f"{prec_type} reported in parentheses"])
        return smry


def compare(
    results: list[PanelModelResults] | dict[str, PanelModelResults],
    *,
    precision: str = "tstats",
    stars: bool = False,
) -> PanelModelComparison:
    """
    Compare the results of multiple models.

    Parameters
    ----------
    results : list of results or dict of str to results
        Set of results to compare. The results are those of the panel models
        in the package. If a dict, the keys will be used as model names.
    precision : {"tstats", "std_errors", "std-errors", "pvalues"}, optional
        Estimator precision estimator to include in the comparison output. The
        default is "tstats". The names are not case sensitive.
    stars : bool, optional
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively. The
        default is False.

    Returns
    -------
    PanelModelComparison
        The model comparison object.

    Raises
    ------
    TypeError
        If a result is not the result of a panel model.
    ValueError
        If `precision` is not one of the supported values.
    """
    return PanelModelComparison(results, precision=precision, stars=stars)
