from __future__ import annotations

from linearmodels.compat.statsmodels import Summary

from collections.abc import Mapping
import datetime as dt
from functools import cached_property
from typing import Any, Union

from formulaic.utils.context import capture_context
import numpy as np
import pandas
from pandas import DataFrame, Series, concat
from scipy import stats
from statsmodels.iolib.summary import SimpleTable, fmt_2cols, fmt_params

from linearmodels.iv.results import default_txt_fmt, stub_concat, table_concat
from linearmodels.shared.base import _ModelComparison, _SummaryStr
from linearmodels.shared.hypotheses import WaldTestStatistic, quadratic_form_test
from linearmodels.shared.io import _str, add_star, pval_format
from linearmodels.shared.utility import AttrDict
import linearmodels.typing.data

__all__ = [
    "PanelResults",
    "PanelEffectsResults",
    "RandomEffectsResults",
    "FamaMacBethResults",
    "compare",
]


class PanelResults(_SummaryStr):
    """
    Results container for panel data models that do not include effects
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
        """Estimated parameters"""
        return Series(self._params, index=self._var_names, name="parameter")

    @cached_property
    def cov(self) -> DataFrame:
        """Estimated covariance of parameters"""
        return DataFrame(
            self._deferred_cov(), columns=self._var_names, index=self._var_names
        )

    @property
    def std_errors(self) -> Series:
        """Estimated parameter standard errors"""
        return Series(np.sqrt(np.diag(self.cov)), self._var_names, name="std_error")

    @property
    def tstats(self) -> Series:
        """Parameter t-statistics"""
        return Series(self._params / self.std_errors, name="tstat")

    @cached_property
    def pvalues(self) -> Series:
        """
        Parameter p-vals. Uses t(df_resid) if ``debiased`` is True, else normal
        """
        abs_tstats = np.abs(self.tstats)
        if self._debiased:
            pv = 2 * (1 - stats.t.cdf(abs_tstats, self.df_resid))
        else:
            pv = 2 * (1 - stats.norm.cdf(abs_tstats))
        return Series(pv, index=self._var_names, name="pvalue")

    @property
    def df_resid(self) -> int:
        """
        Residual degree of freedom

        Notes
        -----
        Defined as nobs minus nvar minus the number of included effects, if any.
        """
        return self._df_resid

    @property
    def df_model(self) -> int:
        """
        Model degree of freedom

        Notes
        -----
        Defined as nvar plus the number of included effects, if any.
        """
        return self._df_model

    @property
    def nobs(self) -> int:
        """Number of observations used to estimate the model"""
        return self._nobs

    @property
    def name(self) -> str:
        """Model name"""
        return self._name

    @property
    def total_ss(self) -> float:
        """Total sum of squares"""
        return self._total_ss

    @property
    def model_ss(self) -> float:
        """Residual sum of squares"""
        return self._total_ss - self._residual_ss

    @property
    def resid_ss(self) -> float:
        """Residual sum of squares"""
        return self._residual_ss

    @property
    def rsquared(self) -> float:
        """Model Coefficient of determination"""
        return self._r2

    @property
    def rsquared_between(self) -> float:
        """
        Between Coefficient of determination

        Returns
        -------
        float
            Between coefficient of determination

        Notes
        -----
        The between rsquared measures the fit of the time-averaged dependent
        variable on the time averaged dependent variables. It accounts for the
        weights used in the estimation of the model.

        See the mathematical reference in the documentation for the formal
        definition of this measure.
        """
        return self._r2b

    @property
    def rsquared_within(self) -> float:
        """
        Within coefficient of determination

        Returns
        -------
        float
            Within coefficient of determination

        Notes
        -----
        The within rsquared measures the fit of the dependent purged of entity
        effects on the exogenous purged of entity effects. It accounts for the
        weights used in the estimation of the model.

        See the mathematical reference in the documentation for the formal
        definition of this measure.
        """
        return self._r2w

    @property
    def rsquared_overall(self) -> float:
        """
        Overall coefficient of determination

        Returns
        -------
        float
            Between coefficient of determination

        Notes
        -----
        The overall rsquared measures the fit of the dependent
        variable on the dependent variables ignoring any included effects.
        It accounts for the weights used in the estimation of the model.

        See the mathematical reference in the documentation for the formal
        definition of this measure.
        """
        return self._r2o

    @property
    def corr_squared_between(self) -> float:
        r"""
        Between Coefficient of determination using squared correlation

        Returns
        -------
        float
            Between coefficient of determination

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
        """
        return self._c2b

    @property
    def corr_squared_within(self) -> float:
        r"""
        Within coefficient of determination using squared correlation

        Returns
        -------
        float
            Within coefficient of determination

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
        """
        return self._c2w

    @property
    def corr_squared_overall(self) -> float:
        r"""
        Overall coefficient of determination using squared correlation

        Returns
        -------
        float
            Between coefficient of determination

        Notes
        -----
        The overall rsquared measures the fit of the dependent
        variable on the dependent variables ignoring any included effects.

        This measure is based on the squared correlation between the
        dependent variables and their predictions.

        .. math::

           Corr[y_{it}, x_{it}\hat{\beta}]

        This measure **does not** account for weights.
        """
        return self._c2o

    @property
    def s2(self) -> float:
        """Residual variance estimator"""
        return self._s2

    @property
    def entity_info(self) -> Series:
        """Statistics on observations per entity"""
        return self._entity_info

    @property
    def time_info(self) -> Series:
        """Statistics on observations per time interval"""
        return self._time_info

    def conf_int(self, level: float = 0.95) -> DataFrame:
        """
        Confidence interval construction

        Parameters
        ----------
        level : float
            Confidence level for interval

        Returns
        -------
        DataFrame
            Confidence interval of the form [lower, upper] for each parameters

        Notes
        -----
        Uses a t(df_resid) if ``debiased`` is True, else normal.
        """
        ci_quantiles = [(1 - level) / 2, 1 - (1 - level) / 2]
        if self._debiased:
            q = stats.t.ppf(ci_quantiles, self.df_resid)
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
            Summary table of model estimation results

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

        param_data = np.c_[
            self.params.values[:, None],
            self.std_errors.values[:, None],
            self.tstats.values[:, None],
            self.pvalues.values[:, None],
            self.conf_int(),
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
        Model residuals

        Notes
        -----
        These residuals are from the estimated model. They will not have the
        same shape as the original data whenever the model is estimated on
        transformed data which has a different shape."""
        return Series(self._resids.squeeze(), index=self._index, name="residual")

    def _out_of_sample(
        self,
        exog: linearmodels.typing.data.ArrayLike | None,
        data: pandas.DataFrame | None,
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
        exog: linearmodels.typing.data.ArrayLike | None = None,
        *,
        data: pandas.DataFrame | None = None,
        fitted: bool = True,
        effects: bool = False,
        idiosyncratic: bool = False,
        missing: bool = False,
    ) -> DataFrame:
        """
        In- and out-of-sample predictions

        Parameters
        ----------
        exog : array_like
            Exogenous values to use in out-of-sample prediction (nobs by nexog)
        data : DataFrame
            DataFrame to use for out-of-sample predictions when model was
            constructed using a formula.
        fitted : bool
            Flag indicating whether to include the fitted values
        effects : bool
            Flag indicating whether to include estimated effects
        idiosyncratic : bool
            Flag indicating whether to include the estimated idiosyncratic shock
        missing : bool
            Flag indicating to adjust for dropped observations.  if True, the
            values returns will have the same size as the original input data
            before filtering missing values

        Returns
        -------
        DataFrame
            DataFrame containing columns for all selected output

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
        out_df: pandas.DataFrame = concat(out, axis=1)
        if missing:
            index = self._original_index
            out_df = out_df.reindex(index)
        return out_df

    @property
    def fitted_values(self) -> Series:
        """Fitted values"""
        return self._fitted

    @property
    def estimated_effects(self) -> Series:
        """
        Estimated effects

        Notes
        -----
        NaN filled when models do not include effects.
        """
        return self._effects

    @property
    def idiosyncratic(self) -> Series:
        """
        Idiosyncratic error

        Notes
        -----
        Differs from resids since this is the estimated idiosyncratic shock
        from the data. It has the same dimension as the dependent data.
        The shape and nature of resids depends on the model estimated. These
        estimates only depend on the model estimated through the estimation
        of parameters and inclusion of effects, if any.
        """
        return self._idiosyncratic

    @property
    def wresids(self) -> Series:
        """Weighted model residuals"""
        return Series(
            self._wresids.squeeze(), index=self._index, name="weighted residual"
        )

    @property
    def f_statistic_robust(self) -> WaldTestStatistic:
        r"""
        Joint test of significance for non-constant regressors

        Returns
        -------
        WaldTestStatistic
            Statistic value, distribution and p-value

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
        """
        from linearmodels.panel.model import _deferred_f

        return _deferred_f(
            self.params, self.cov, self._debiased, self.df_resid, self._f_info
        )

    @property
    def f_statistic(self) -> WaldTestStatistic:
        r"""
        Joint test of significance for non-constant regressors

        Returns
        -------
        WaldTestStatistic
            Statistic value, distribution and p-value

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
        """Log-likelihood of model"""
        return self._loglik

    def wald_test(
        self,
        restriction: (
            linearmodels.typing.data.Float64Array | pandas.DataFrame | None
        ) = None,
        value: linearmodels.typing.data.Float64Array | pandas.Series | None = None,
        *,
        formula: str | list[str] | None = None,
    ) -> WaldTestStatistic:
        r"""
        Test linear equality constraints using a Wald test

        Parameters
        ----------
        restriction : {ndarray, DataFrame}
            q by nvar array containing linear weights to apply to parameters
            when forming the restrictions. It is not possible to use both
            restriction and formula.
        value : {ndarray, Series}
            q element array containing the restricted values.
        formula : Union[str, list[str]]
            formulaic linear constraints. The simplest formats are one of:

            * A single comma-separated string such as "x1=0, x2+x3=1"
            * A list of strings where each element is a single constraint
              such as ["x1=0", "x2+x3=1"]
            * A single string without commas to test simple constraints such
              as "x1=x2=x3=0"
            * A dictionary where each key is a parameter restriction and
              the corresponding value is the restriction value, e.g.,
              {"x1": 0, "x2+x3": 1}.

            It is not possible to use both ``restriction`` and ``formula``.

        Returns
        -------
        WaldTestStatistic
            Test statistic for null that restrictions are valid.

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
        """
        return quadratic_form_test(
            self.params, self.cov, restriction=restriction, value=value, formula=formula
        )


class PanelEffectsResults(PanelResults):
    """
    Results container for panel data models that include effects
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

    @property
    def f_pooled(self) -> WaldTestStatistic:
        r"""
        Test that included effects are jointly zero.

        Returns
        -------
        WaldTestStatistic
            Statistic value, distribution and p-value

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
        """List of effects included in the model"""
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
                for c in oe:
                    effects.append("Other Effect (" + str(c) + ")")
        return effects

    @property
    def other_info(self) -> pandas.DataFrame | None:
        """Statistics on observations per group for other effects"""
        return self._other_info

    @property
    def rsquared_inclusive(self) -> float:
        """Model Coefficient of determination including fit of included effects"""
        return self._r2_ex_effects

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results

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
        """Decomposition of total variance into effects and residuals"""
        vals = [self._sigma2_effects, self._sigma2_eps, self._rho]
        index = ["Effects", "Residual", "Percent due to Effects"]
        return Series(vals, index=index, name="Variance Decomposition")


class RandomEffectsResults(PanelResults):
    """
    Results container for random effect panel data models
    """

    def __init__(self, res: AttrDict) -> None:
        super().__init__(res)
        self._theta = res.theta
        self._sigma2_effects = res.sigma2_effects
        self._sigma2_eps = res.sigma2_eps
        self._rho = res.rho

    @property
    def variance_decomposition(self) -> Series:
        """Decomposition of total variance into effects and residuals"""
        vals = [self._sigma2_effects, self._sigma2_eps, self._rho]
        index = ["Effects", "Residual", "Percent due to Effects"]
        return Series(vals, index=index, name="Variance Decomposition")

    @property
    def theta(self) -> DataFrame:
        """Values used in generalized demeaning"""
        return self._theta


PanelModelResults = Union[PanelEffectsResults, PanelResults, RandomEffectsResults]


class FamaMacBethResults(PanelResults):
    """
    Results container for Fama MacBeth panel data models
    """

    def __init__(self, res: AttrDict):
        super().__init__(res)
        self._all_params = res.all_params
        self._avg_r2 = res.avg_r2
        self._avg_adj_r2 = res.avg_adj_r2

    @property
    def all_params(self) -> DataFrame:
        """
        The set of parameters estimated for each of the time periods

        Returns
        -------
        DataFrame
            The parameters (nobs, nparam).
        """
        return self._all_params

    @property
    def avg_rsquared(self) -> float:
        """
        The average coefficient of determination

        This value contains the average of the individual adjusted rsquared
        values across the nobs cross-sectional regressions. An rsquare value
        is only included in the average if a cross-section has full rank.
        """
        return self._avg_r2

    @property
    def avg_adj_rsquared(self) -> float:
        """
        The average coefficient of determination, adjusted for sample size.

        This value contains the average of the individual adjusted rsquared
        values across the nobs cross-sectional regressions. An rsquare value
        is only included in the average if a cross-section has full rank and
        if the number of dependent variables in a cross-section is larger than
        the number of regressors.
        """
        return self._avg_adj_r2


class PanelModelComparison(_ModelComparison):
    """
    Comparison of multiple models

    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {"tstats","std_errors", "std-errors", "pvalues"}
        Estimator precision estimator to include in the comparison output.
        Default is "tstats".
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.
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
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_between")

    @property
    def rsquared_within(self) -> Series:
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_within")

    @property
    def rsquared_overall(self) -> Series:
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_overall")

    @property
    def estimator_method(self) -> Series:
        """Estimation methods"""
        return self._get_property("name")

    @property
    def cov_estimator(self) -> Series:
        """Covariance estimator descriptions"""
        return self._get_property("_cov_type")

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results

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
        vals_lst = [[i for i in v] for v in vals.T.values]
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
            for v, pv in zip(params.values[i], pvalues[i]):
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
    Compare the results of multiple models

    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {"tstats","std_errors", "std-errors", "pvalues"}
        Estimator precision estimator to include in the comparison output.
        Default is "tstats".
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.

    Returns
    -------
    PanelModelComparison
        The model comparison object.
    """
    return PanelModelComparison(results, precision=precision, stars=stars)
