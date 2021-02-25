"""
Results containers and post-estimation diagnostics for IV models
"""
from linearmodels.compat.statsmodels import Summary

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from numpy import array, asarray, c_, diag, empty, isnan, log, ndarray, ones, sqrt
from numpy.linalg import inv
from pandas import DataFrame, Series, concat, to_numeric
from property_cached import cached_property
import scipy.stats as stats
from statsmodels.iolib.summary import SimpleTable, fmt_2cols, fmt_params
from statsmodels.iolib.table import default_txt_fmt

import linearmodels
from linearmodels.iv._utility import annihilate, proj
from linearmodels.iv.data import IVData
from linearmodels.shared.base import _ModelComparison, _SummaryStr
from linearmodels.shared.hypotheses import (
    InvalidTestStatistic,
    WaldTestStatistic,
    quadratic_form_test,
)
from linearmodels.shared.io import _str, add_star, pval_format
from linearmodels.typing import ArrayLike, NDArray, OptionalArrayLike


def stub_concat(lists: Sequence[Sequence[str]], sep: str = "=") -> List[str]:
    col_size = max([max(map(len, stubs)) for stubs in lists])
    out: List[str] = []
    for stubs in lists:
        out.extend(stubs)
        out.append(sep * (col_size + 2))
    return out[:-1]


def table_concat(lists: Sequence[List[List[str]]], sep: str = "=") -> List[List[str]]:
    col_sizes = []
    for table in lists:
        size = [[len(item) for item in row] for row in table]
        size_arr = array(size)
        col_sizes.append(list(asarray(size_arr.max(0))))
    col_size = asarray(array(col_sizes).max(axis=0))
    sep_cols: List[str] = [sep * (cs + 2) for cs in col_size]
    out: List[List[str]] = []
    for table in lists:
        out.extend(table)
        out.append(sep_cols)
    return out[:-1]


class _LSModelResultsBase(_SummaryStr):
    """
    Results from OLS model estimation

    Parameters
    ----------
    results : dict[str, any]
        A dictionary of results from the model estimation.
    model : _OLS
        The model used to estimate parameters.
    """

    def __init__(self, results: Dict[str, Any], model: Any) -> None:
        self._resid = results["eps"]
        self._wresid = results["weps"]
        self._params = results["params"]
        self._cov = results["cov"]
        self.model = model
        self._r2 = results["r2"]
        self._cov_type = results["cov_type"]
        self._rss = results["residual_ss"]
        self._tss = results["total_ss"]
        self._s2 = results["s2"]
        self._debiased = results["debiased"]
        self._f_statistic = results["fstat"]
        self._vars = results["vars"]
        self._cov_config = results["cov_config"]
        self._method = results["method"]
        self._kappa = results.get("kappa", None)
        self._datetime = dt.datetime.now()
        self._cov_estimator = results["cov_estimator"]
        self._original_index = results["original_index"]
        self._fitted = results["fitted"]
        self._df_model = results.get("df_model", self._params.shape[0])

    @property
    def cov_config(self) -> Dict[str, Any]:
        """Parameter values from covariance estimator"""
        return self._cov_config

    @property
    def cov_estimator(self) -> str:
        """Type of covariance estimator used to compute covariance"""
        return self._cov_type

    @property
    def cov(self) -> DataFrame:
        """Estimated covariance of parameters"""
        return self._cov

    @property
    def params(self) -> Series:
        """Estimated parameters"""
        return self._params

    @cached_property
    def resids(self) -> Series:
        """Estimated residuals"""
        return self._resid()

    @cached_property
    def fitted_values(self) -> Series:
        """Fitted values"""
        return self._fitted()

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
        return self.resids

    @cached_property
    def wresids(self) -> Series:
        """Weighted estimated residuals"""
        return self._wresid()

    @property
    def nobs(self) -> int:
        """Number of observations"""
        return self.model.dependent.shape[0]

    @property
    def df_resid(self) -> int:
        """Residual degree of freedom"""
        return self.nobs - self.df_model

    @property
    def df_model(self) -> int:
        """Model degree of freedom"""
        return int(self._df_model)

    @property
    def has_constant(self) -> bool:
        """Flag indicating the model includes a constant or equivalent"""
        return self.model.has_constant

    @property
    def rsquared(self) -> float:
        """Coefficient of determination (R**2)"""
        return self._r2

    @property
    def rsquared_adj(self) -> float:
        """Sample-size adjusted coefficient of determination (R**2)"""
        n, k, c = self.nobs, self.df_model, int(self.has_constant)
        return 1 - ((n - c) / (n - k)) * (1 - self._r2)

    @property
    def cov_type(self) -> str:
        """Covariance estimator used"""
        return self._cov_type

    @cached_property
    def std_errors(self) -> Series:
        """Estimated parameter standard errors"""
        std_errors = sqrt(diag(self.cov))
        return Series(std_errors, index=self._vars, name="stderr")

    @cached_property
    def tstats(self) -> Series:
        """Parameter t-statistics"""
        return Series(self._params / self.std_errors, name="tstat")

    @cached_property
    def pvalues(self) -> Series:
        """
        Parameter p-vals. Uses t(df_resid) if ``debiased`` is True, else normal
        """
        if self.debiased:
            pvals = 2 - 2 * stats.t.cdf(abs(self.tstats), self.df_resid)
        else:
            pvals = 2 - 2 * stats.norm.cdf(abs(self.tstats))

        return Series(pvals, index=self._vars, name="pvalue")

    @property
    def total_ss(self) -> float:
        """Total sum of squares"""
        return self._tss

    @property
    def model_ss(self) -> float:
        """Residual sum of squares"""
        return self._tss - self._rss

    @property
    def resid_ss(self) -> float:
        """Residual sum of squares"""
        return self._rss

    @property
    def s2(self) -> float:
        """Residual variance estimator"""
        return self._s2

    @property
    def debiased(self) -> bool:
        """Flag indicating whether covariance uses a small-sample adjustment"""
        return self._debiased

    @property
    def f_statistic(self) -> WaldTestStatistic:
        """
        Model F-statistic

        Returns
        -------
        WaldTestStatistic
            Test statistic for null all coefficients excluding constant terms
            are zero.

        Notes
        -----
        Despite name, always implemented using a quadratic-form test based on
        estimated parameter covariance. Default is to use a chi2 distribution
        to compute p-values. If ``debiased`` is True, divides statistic by
        number of parameters tested and uses an F-distribution.

        This version of the F-statistic directly uses the model covariance
        estimator and so is robust against the same specification issues.
        """
        return self._f_statistic

    @property
    def method(self) -> str:
        """Method used to estimate model parameters"""
        return self._method

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
        ci = asarray(self.params)[:, None] + asarray(self.std_errors)[:, None] * q
        return DataFrame(ci, index=self._vars, columns=["lower", "upper"])

    def _top_right(self) -> List[Tuple[str, str]]:
        f_stat = _str(self.f_statistic.stat)
        if isnan(self.f_statistic.stat):
            f_stat = "      N/A"

        return [
            ("R-squared:", _str(self.rsquared)),
            ("Adj. R-squared:", _str(self.rsquared_adj)),
            ("F-statistic:", f_stat),
            ("P-value (F-stat)", pval_format(self.f_statistic.pval)),
            ("Distribution:", str(self.f_statistic.dist_name)),
            ("", ""),
            ("", ""),
        ]

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results

        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """

        title = self._method + " Estimation Summary"
        mod = self.model
        top_left = [
            ("Dep. Variable:", mod.dependent.cols[0]),
            ("Estimator:", self._method),
            ("No. Observations:", self.nobs),
            ("Date:", self._datetime.strftime("%a, %b %d %Y")),
            ("Time:", self._datetime.strftime("%H:%M:%S")),
            ("Cov. Estimator:", self._cov_type),
            ("", ""),
        ]

        top_right = self._top_right()

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

        param_data = c_[
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
        extra_text = []
        if table_stubs:
            header = [
                "Parameter",
                "Std. Err.",
                "T-stat",
                "P-value",
                "Lower CI",
                "Upper CI",
            ]
            table = SimpleTable(
                data, stubs=table_stubs, txt_fmt=fmt_params, headers=header, title=title
            )
            smry.tables.append(table)
        else:
            extra_text.append("Model contains no parameters")

        extra_text = self._update_extra_text(extra_text)
        if extra_text:
            smry.add_extra_txt(extra_text)

        return smry

    def _update_extra_text(self, extra_text: List[str]) -> List[str]:
        return extra_text

    def wald_test(
        self,
        restriction: Optional[Union[DataFrame, ndarray]] = None,
        value: Optional[Union[Series, ndarray]] = None,
        *,
        formula: Optional[Union[str, List[str]]] = None,
    ) -> WaldTestStatistic:
        r"""
        Test linear equality constraints using a Wald test

        Parameters
        ----------
        restriction : {ndarray, DataFrame}, optional
            q by nvar array containing linear weights to apply to parameters
            when forming the restrictions. It is not possible to use both
            restriction and formula.
        value : {ndarray, Series}, optional
            q element array containing the restricted values.
        formula : Union[str, list[str]], optional
            patsy linear constraints. The simplest formats are one of:

            * A single comma-separated string such as 'x1=0, x2+x3=1'
            * A list of strings where each element is a single constraint
              such as ['x1=0', 'x2+x3=1']
            * A single string without commas to test simple constraints such
              as 'x1=x2=x3=0'

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
        >>> import numpy as np
        >>> from linearmodels.datasets import wage
        >>> from linearmodels.iv import IV2SLS
        >>> data = wage.load()
        >>> formula = 'np.log(wage) ~ 1 + exper + I(exper**2) + brthord + [educ ~ sibs]'
        >>> res = IV2SLS.from_formula(formula, data).fit()

        Testing the experience is not needed in the model

        >>> restriction = np.array([[0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0]])
        >>> value = np.array([0, 0])
        >>> res.wald_test(restriction, value)

        Using the formula interface to test the same restrictions

        >>> formula = 'exper = I(exper ** 2) = 0'
        >>> res.wald_test(formula=formula)

        Using the formula interface with a list

        >>> res.wald_test(formula=['exper = 0', 'I(exper ** 2) = 0'])
        """
        return quadratic_form_test(
            self._params,
            self.cov,
            restriction=restriction,
            value=value,
            formula=formula,
        )


class OLSResults(_LSModelResultsBase):
    """
    Results from OLS model estimation

    Parameters
    ----------
    results : dict[str, any]
        A dictionary of results from the model estimation.
    model : _OLS
        The model used to estimate parameters.
    """

    def __init__(
        self,
        results: Dict[str, Any],
        model: "linearmodels.iv.model._IVModelBase",
    ) -> None:
        super().__init__(results, model)

    def _out_of_sample(
        self,
        exog: ArrayLike,
        endog: ArrayLike,
        data: ArrayLike,
        missing: Optional[bool],
    ) -> DataFrame:
        """Interface between model predict and predict for OOS fits"""
        if not (exog is None and endog is None) and data is not None:
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog/endog or data, but not both."
            )
        pred = self.model.predict(self.params, exog=exog, endog=endog, data=data)
        if not missing:
            pred = pred.loc[pred.notnull().all(1)]
        return pred

    def predict(
        self,
        exog: OptionalArrayLike = None,
        endog: OptionalArrayLike = None,
        *,
        data: Optional[DataFrame] = None,
        fitted: bool = True,
        idiosyncratic: bool = False,
        missing: bool = False,
    ) -> DataFrame:
        """
        In- and out-of-sample predictions

        Parameters
        ----------
        exog : array_like
            Exogenous values to use in out-of-sample prediction (nobs by nexog)
        endog : array_like
            Endogenous values to use in out-of-sample prediction (nobs by nendog)
        data : DataFrame, optional
            DataFrame to use for out-of-sample predictions when model was
            constructed using a formula.
        fitted : bool, optional
            Flag indicating whether to include the fitted values
        idiosyncratic : bool, optional
            Flag indicating whether to include the estimated idiosyncratic shock
        missing : bool, optional
            Flag indicating to adjust for dropped observations.  If True, the
            values returned will have the same size as the original input data
            before filtering missing values.  If False, then missing
            observations will not be returned.

        Returns
        -------
        DataFrame
            DataFrame containing columns for all selected outputs

        Notes
        -----
        If `exog`, `endog` and `data` are all `None`, in-sample predictions
        (fitted values) will be returned.

        If `data` is not none, then `exog` and `endog` must be none.
        Predictions from models constructed using formulas can
        be computed using either `exog` and `endog`, which will treat these are
        arrays of values corresponding to the formula-process data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.
        """
        if not (exog is None and endog is None and data is None):
            return self._out_of_sample(exog, endog, data, missing)
        out = []
        if fitted:
            out.append(self.fitted_values)
        if idiosyncratic:
            out.append(self.idiosyncratic)
        if len(out) == 0:
            raise ValueError("At least one output must be selected")
        out_df: DataFrame = concat(out, 1)
        if missing:
            index = self._original_index
            out_df = out_df.reindex(index)
        return out_df

    @property
    def kappa(self) -> float:
        """k-class estimator value"""
        return self._kappa

    def _update_extra_text(self, extra_text: List[str]) -> List[str]:
        instruments = self.model.instruments
        if instruments.shape[1] > 0:

            endog = self.model.endog
            extra_text.append("Endogenous: " + ", ".join(endog.cols))
            extra_text.append("Instruments: " + ", ".join(instruments.cols))
            cov_descr = str(self._cov_estimator)
            for line in cov_descr.split("\n"):
                extra_text.append(line)
        return extra_text


class AbsorbingLSResults(_LSModelResultsBase):
    """
    Results from IV estimation

    Parameters
    ----------
    results : dict[str, any]
        A dictionary of results from the model estimation.
    model : AbsorbingLS
        The model used to estimate parameters.
    """

    def __init__(
        self, results: Dict[str, Any], model: "linearmodels.iv.absorbing.AbsorbingLS"
    ) -> None:
        super(AbsorbingLSResults, self).__init__(results, model)
        self._absorbed_rsquared = results["absorbed_r2"]
        self._absorbed_effects = results["absorbed_effects"]

    def _top_right(self) -> List[Tuple[str, str]]:
        f_stat = _str(self.f_statistic.stat)
        if isnan(self.f_statistic.stat):
            f_stat = "      N/A"

        return [
            ("R-squared:", _str(self.rsquared)),
            ("Adj. R-squared:", _str(self.rsquared_adj)),
            ("F-statistic:", f_stat),
            ("P-value (F-stat):", pval_format(self.f_statistic.pval)),
            ("Distribution:", str(self.f_statistic.dist_name)),
            ("R-squared (No Effects):", _str(round(self.absorbed_rsquared, 5))),
            ("Varaibles Absorbed:", _str(self.df_absorbed)),
        ]

    @property
    def absorbed_rsquared(self) -> float:
        """Coefficient of determination (R**2), ignoring absorbed variables"""
        return self._absorbed_rsquared

    @cached_property
    def absorbed_effects(self) -> DataFrame:
        """Fitted values from only absorbed terms"""
        return self._absorbed_effects()

    @property
    def df_absorbed(self) -> int:
        """Number of variables absorbed"""
        return self.df_model - self.params.shape[0]


class FirstStageResults(_SummaryStr):
    """
    First stage estimation results and diagnostics
    """

    def __init__(
        self,
        dep: IVData,
        exog: IVData,
        endog: IVData,
        instr: IVData,
        weights: IVData,
        cov_type: str,
        cov_config: Dict[str, Any],
    ) -> None:
        self.dep = dep
        self.exog = exog
        self.endog = endog
        self.instr = instr
        self.weights = weights
        reg = c_[self.exog.ndarray, self.endog.ndarray]
        self._reg = DataFrame(reg, columns=self.exog.cols + self.endog.cols)
        self._cov_type = cov_type
        self._cov_config = cov_config

    @cached_property
    def diagnostics(self) -> DataFrame:
        """
        Post estimation diagnostics of first-stage fit

        Returns
        -------
        DataFrame
            DataFrame where each endogenous variable appears as a row and
            the columns contain alternative measures.  The columns are:

            * rsquared - R-squared from regression of endogenous on exogenous
              and instruments
            * partial.rsquared - R-squared from regression of the exogenous
              variable on instruments where both the exogenous variable and
              the instrument have been orthogonalized to the exogenous
              regressors in the model.
            * f.stat - Test that all coefficients are zero in the model
              used to estimate the partial R-squared. Uses a standard F-test
              when the covariance estimator is unadjusted - otherwise uses a
              Wald test statistic with a chi2 distribution.
            * f.pval - P-value of the test that all coefficients are zero
              in the model used to estimate the partial R-squared
            * f.dist - Distribution of f.stat
            * shea.rsquared - Shea's r-squared which measures the correlation
              between the projected and orthogonalized instrument on the
              orthogonalized endogenous regressor where the orthogonalization
              is with respect to the other included variables in the model.
        """
        from linearmodels.iv.model import _OLS, IV2SLS

        endog, exog, instr, weights = self.endog, self.exog, self.instr, self.weights
        w = sqrt(weights.ndarray)
        z = w * instr.ndarray
        nz = z.shape[1]
        x = w * exog.ndarray
        ez = annihilate(z, x)
        individual_results = self.individual
        out_df = DataFrame(
            index=["rsquared", "partial.rsquared", "f.stat", "f.pval", "f.dist"],
            columns=[],
        )
        for col in endog.pandas:
            y = w * endog.pandas[[col]].values
            ey = annihilate(y, x)
            partial = _OLS(ey, ez).fit(cov_type=self._cov_type, **self._cov_config)
            full = individual_results[col]
            params = full.params.values[-nz:]
            params = params[:, None]
            c = asarray(full.cov)[-nz:, -nz:]
            stat = params.T @ inv(c) @ params
            stat = float(stat.squeeze())
            if full.cov_type in ("homoskedastic", "unadjusted"):
                df_denom = full.df_resid
                stat /= params.shape[0]
            else:
                df_denom = None
            w_test = WaldTestStatistic(
                stat, null="", df=params.shape[0], df_denom=df_denom
            )
            inner = {
                "rsquared": full.rsquared,
                "partial.rsquared": partial.rsquared,
                "f.stat": w_test.stat,
                "f.pval": w_test.pval,
                "f.dist": w_test.dist_name,
            }
            out_df[col] = Series(inner)
        out_df = out_df.T

        dep = self.dep
        r2sls = IV2SLS(dep, exog, endog, instr, weights=weights).fit(
            cov_type="unadjusted"
        )
        rols = _OLS(dep, self._reg, weights=weights).fit(cov_type="unadjusted")
        shea = (rols.std_errors / r2sls.std_errors) ** 2
        shea *= (1 - r2sls.rsquared) / (1 - rols.rsquared)
        out_df["shea.rsquared"] = shea[out_df.index]
        cols = [
            "rsquared",
            "partial.rsquared",
            "shea.rsquared",
            "f.stat",
            "f.pval",
            "f.dist",
        ]
        out_df = out_df[cols]
        for c in out_df:
            out_df[c] = to_numeric(out_df[c], errors="ignore")

        return out_df

    @cached_property
    def individual(self) -> Dict[str, OLSResults]:
        """
        Individual model results from first-stage regressions

        Returns
        -------
        dict
            Dictionary containing first stage estimation results. Keys are
            the variable names of the endogenous regressors.
        """
        from linearmodels.iv.model import _OLS

        exog_instr = DataFrame(
            c_[self.exog.ndarray, self.instr.ndarray],
            columns=self.exog.cols + self.instr.cols,
        )
        res: Dict[str, OLSResults] = {}
        for col in self.endog.pandas:
            dep = self.endog.pandas[col]
            mod = _OLS(dep, exog_instr, weights=self.weights.ndarray)
            res[col] = mod.fit(cov_type=self._cov_type, **self._cov_config)

        return res

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results

        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """
        smry = Summary()
        if not self.individual:
            table = SimpleTable([[]])
            smry.tables.append(table)
            smry.add_extra_txt(
                ["Model contains no endogenous variables. No first stage results."]
            )
            return smry
        stubs_lookup = {
            "rsquared": "R-squared",
            "partial.rsquared": "Partial R-squared",
            "shea.rsquared": "Shea's R-squared",
            "f.stat": "Partial F-statistic",
            "f.pval": "P-value (Partial F-stat)",
            "f.dist": "Partial F-stat Distn",
        }

        diagnostics = self.diagnostics
        vals = []
        for c in diagnostics:
            if c != "f.dist":
                vals.append([_str(v) for v in diagnostics[c]])
            else:
                vals.append([v for v in diagnostics[c]])
        stubs = [stubs_lookup[s] for s in list(diagnostics.columns)]
        header = list(diagnostics.index)

        params = []
        for var in header:
            res = self.individual[var]
            v = c_[res.params.values, res.tstats.values]
            params.append(v.ravel())
        params_arr = array(params)
        params_fmt = [[_str(val) for val in row] for row in params_arr.T]
        for i in range(1, len(params_fmt), 2):
            for j in range(len(params_fmt[i])):
                params_fmt[i][j] = "({0})".format(params_fmt[i][j])

        params_stub = []
        for var in res.params.index:
            params_stub.extend([var, ""])

        title = "First Stage Estimation Results"

        vals = table_concat((vals, params_fmt))
        stubs = stub_concat((stubs, params_stub))

        txt_fmt = default_txt_fmt.copy()
        txt_fmt["data_aligns"] = "r"
        txt_fmt["header_align"] = "r"
        table = SimpleTable(
            vals, headers=header, title=title, stubs=stubs, txt_fmt=txt_fmt
        )
        smry.tables.append(table)
        extra_txt = [
            "T-stats reported in parentheses",
            "T-stats use same covariance type as original model",
        ]
        smry.add_extra_txt(extra_txt)
        return smry


class _CommonIVResults(OLSResults):
    """
    Results from IV estimation
    """

    def __init__(
        self,
        results: Dict[str, Any],
        model: "linearmodels.iv.model._IVModelBase",
    ) -> None:
        super().__init__(results, model)
        self._liml_kappa = results.get("liml_kappa", None)

    @property
    def first_stage(self) -> FirstStageResults:
        """
        First stage regression results

        Returns
        -------
        FirstStageResults
            Object containing results for diagnosing instrument relevance issues.
        """
        return FirstStageResults(
            self.model.dependent,
            self.model.exog,
            self.model.endog,
            self.model.instruments,
            self.model.weights,
            self._cov_type,
            self._cov_config,
        )


class IVResults(_CommonIVResults):
    """
    Results from IV estimation

    Parameters
    ----------
    results : dict[str, any]
        A dictionary of results from the model estimation.
    model : {IV2SLS, IVLIML}
        The model used to estimate parameters.
    """

    def __init__(
        self, results: Dict[str, Any], model: "linearmodels.iv.model._IVLSModelBase"
    ) -> None:
        super(IVResults, self).__init__(results, model)
        self._kappa = results.get("kappa", 1)

    @cached_property
    def sargan(self) -> Union[InvalidTestStatistic, WaldTestStatistic]:
        r"""
        Sargan test of overidentifying restrictions

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Requires more instruments than endogenous variables

        Tests the ratio of re-projected IV regression residual variance to
        variance of the IV residuals.

        .. math ::

          n(1-\hat{\epsilon}^{\prime}M_{Z}\hat{\epsilon}/
          \hat{\epsilon}^{\prime}\hat{\epsilon})\sim\chi_{v}^{2}

        where :math:`M_{z}` is the annihilator matrix where z is the set of
        instruments and :math:`\hat{\epsilon}` are the residuals from the IV
        estimator.  The degree of freedom is the difference between the number
        of instruments and the number of endogenous regressors.

        .. math ::

          v = n_{instr} - n_{exog}
        """
        z = self.model.instruments.ndarray
        nobs, ninstr = z.shape
        nendog = self.model.endog.shape[1]
        name = "Sargan's test of overidentification"
        if ninstr - nendog == 0:
            return InvalidTestStatistic(
                "Test requires more instruments than " "endogenous variables.",
                name=name,
            )

        eps = self.resids.values[:, None]
        u = annihilate(eps, self.model._z)
        stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
        null = "The model is not overidentified."

        return WaldTestStatistic(stat, null, ninstr - nendog, name=name)

    @cached_property
    def basmann(self) -> Union[InvalidTestStatistic, WaldTestStatistic]:
        r"""
        Basmann's test of overidentifying restrictions

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Requires more instruments than endogenous variables

        Tests is a small-sample version of Sargan's test that has the same
        distribution.

        .. math ::

          s (n - n_{instr}) / (n - s) \sim \chi^2_{v}

        where :math:`n_{instr}` is the number of instruments, :math:`n_{exog}`
        is the number of exogenous regressors and :math:`n_{endog}` is the
        number of endogenous regressors.  The degree of freedom is the
        difference between the number of instruments and the number of
        endogenous regressors.

        .. math ::

          v = n_{instr} - n_{exog}
        """
        mod = self.model
        ninstr = mod.instruments.shape[1]
        nobs, nendog = mod.endog.shape
        nz = mod._z.shape[1]
        name = "Basmann's test of overidentification"
        if ninstr - nendog == 0:
            return InvalidTestStatistic(
                "Test requires more instruments than " "endogenous variables.",
                name=name,
            )
        sargan_test = self.sargan
        s = sargan_test.stat
        stat = s * (nobs - nz) / (nobs - s)
        return WaldTestStatistic(stat, sargan_test.null, sargan_test.df, name=name)

    def _endogeneity_setup(
        self, variables: Optional[Union[str, List[str]]] = None
    ) -> Tuple[ndarray, ndarray, ndarray, int, int, int, int]:
        """Setup function for some endogeneity iv"""
        if isinstance(variables, str):
            variables = [variables]
        elif variables is not None and not isinstance(variables, list):
            raise TypeError("variables must be a str or a list of str.")

        nobs = self.model.dependent.shape[0]
        e2 = self.resids.values
        nendog, nexog = self.model.endog.shape[1], self.model.exog.shape[1]
        if variables is None:
            assumed_exog = self.model.endog.ndarray
            aug_exog = c_[self.model.exog.ndarray, assumed_exog]
            still_endog = empty((nobs, 0))
        else:
            assert isinstance(variables, list)
            assumed_exog = self.model.endog.pandas[variables].values
            ex = [c for c in self.model.endog.cols if c not in variables]
            still_endog = self.model.endog.pandas[ex].values
            aug_exog = c_[self.model.exog.ndarray, assumed_exog]
        ntested = assumed_exog.shape[1]

        from linearmodels.iv import IV2SLS

        mod = IV2SLS(
            self.model.dependent, aug_exog, still_endog, self.model.instruments
        )
        e0 = mod.fit().resids.values[:, None]

        z2 = c_[self.model.exog.ndarray, self.model.instruments.ndarray]
        z1 = c_[z2, assumed_exog]

        e1 = proj(e0, z1)
        e2 = proj(e2, self.model.instruments.ndarray)
        return e0, e1, e2, nobs, nexog, nendog, ntested

    def durbin(
        self, variables: Optional[Union[str, List[str]]] = None
    ) -> WaldTestStatistic:
        r"""
        Durbin's test of exogeneity

        Parameters
        ----------
        variables : {str, List[str]}, default None
            List of variables to test for exogeneity.  If None, all variables
            are jointly tested.

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Test statistic is difference between sum of squared OLS and sum of
        squared IV residuals where each set of residuals has been projected
        onto the set of instruments in the IV model.

        Start by defining

        .. math ::

          \delta & = \hat{\epsilon}'_e P_{[z,w]} \hat{\epsilon}_e -
                     \hat{\epsilon}'_c P_{z} \hat{\epsilon}_c

        where :math:`\hat{\epsilon}_e` are the regression residuals from a
        model where ``vars`` are treated as exogenous,
        :math:`\hat{\epsilon}_c` are the regression residuals from the model
        leaving ``vars`` as endogenous, :math:`P_{[z,w]}` is a projection
        matrix onto the exogenous variables and instruments (`z`) as well as
        ``vars``, and :math:`P_{z}` is a projection matrix only onto `z`.

        The test statistic is then

        .. math ::

            \delta / (\hat{\epsilon}'_e\hat{\epsilon}_e) / n \sim \chi^2_{q}

        where :math:`q` is the number of variables tested.
        """
        null = "All endogenous variables are exogenous"
        if variables is not None:
            null = "Variables {0} are exogenous".format(", ".join(variables))

        e0, e1, e2, nobs, _, _, ntested = self._endogeneity_setup(variables)
        stat = e1.T @ e1 - e2.T @ e2
        stat /= (e0.T @ e0) / nobs

        name = "Durbin test of exogeneity"
        df = ntested
        return WaldTestStatistic(float(stat), null, df, name=name)

    def wu_hausman(
        self, variables: Optional[Union[str, List[str]]] = None
    ) -> WaldTestStatistic:
        r"""
        Wu-Hausman test of exogeneity

        Parameters
        ----------
        variables : {str, List[str]}, default None
            List of variables to test for exogeneity.  If None, all variables
            are jointly tested.

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Test statistic is difference between sum of squared OLS and sum of
        squared IV residuals where each set of residuals has been projected
        onto the set of instruments in the IV model.

        Start by defining

        .. math ::

          \delta & = \hat{\epsilon}'_e P_{[z,w]} \hat{\epsilon}_e -
                     \hat{\epsilon}'_c P_{z} \hat{\epsilon}_c

        where :math:`\hat{\epsilon}_e` are the regression residuals from a
        model where ``vars`` are treated as exogenous,
        :math:`\hat{\epsilon}_c` are the regression residuals from the model
        leaving ``vars`` as endogenous, :math:`P_{[z,w]}` is a projection
        matrix onto the exogenous variables and instruments (`z`) as well as
        ``vars``, and :math:`P_{z}` is a projection matrix only onto `z`.

        The test statistic is then

        .. math ::

            \frac{\delta / q}{(\hat{\epsilon}'_e\hat{\epsilon}_e - \delta) / v}

        where :math:`q` is the number of variables iv,
        :math:`v = n - n_{endog} - n_{exog} - q`. The test statistic has a
        :math:`F_{q, v}` distribution.
        """
        null = "All endogenous variables are exogenous"
        if variables is not None:
            null = "Variables {0} are exogenous".format(", ".join(variables))

        e0, e1, e2, nobs, nexog, nendog, ntested = self._endogeneity_setup(variables)

        df = ntested
        df_denom = nobs - nexog - nendog - ntested
        delta = e1.T @ e1 - e2.T @ e2
        stat = delta / df
        stat /= (e0.T @ e0 - delta) / df_denom
        stat = float(stat)

        name = "Wu-Hausman test of exogeneity"
        return WaldTestStatistic(stat, null, df, df_denom, name=name)

    @cached_property
    def wooldridge_score(self) -> WaldTestStatistic:
        r"""
        Wooldridge's score test of exogeneity

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Wooldridge's test examines whether there is correlation between the
        errors produced when the endogenous variable are treated as
        exogenous so that the model can be fit by OLS, and the component of
        the endogenous variables that cannot be explained by the instruments.

        The test is implemented using a regression,

        .. math ::

          1 = \gamma_1 \hat{\epsilon}_1 \hat{v}_{1,i} + \ldots
            + \gamma_p \hat{\epsilon}_1 \hat{v}_{p,i} + \eta_i

        where :math:`\hat{v}_{j,i}` is the residual from regressing endogenous
        variable :math:`x_j` on the exogenous variables and instruments.

        The test is a :math:`n\times R^2 \sim \chi^2_{p}`.

        Implemented using the expression in Wooldridge (2002), Eq. 6.19
        """
        from linearmodels.iv.model import _OLS

        e = annihilate(self.model.dependent.ndarray, self.model._x)
        r = annihilate(self.model.endog.ndarray, self.model._z)
        nobs = e.shape[0]
        r = annihilate(r, self.model._x)
        res = _OLS(ones((nobs, 1)), r * e).fit(cov_type="unadjusted")
        stat = res.nobs - res.resid_ss
        df = self.model.endog.shape[1]
        null = "Endogenous variables are exogenous"
        name = "Wooldridge's score test of exogeneity"
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def wooldridge_regression(self) -> WaldTestStatistic:
        r"""
        Wooldridge's regression test of exogeneity

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Wooldridge's test examines whether there is correlation between the
        components of the endogenous variables that cannot be explained by
        the instruments and the OLS regression residuals.

        The test is implemented as an OLS where

        .. math ::

          y_i = x_{1i}\beta_i + x_{2i}\beta_2 + \hat{e}_i\gamma + \epsilon_i

        where :math:`x_{1i}` are the exogenous regressors, :math:`x_{2i}` are
        the  endogenous regressors and :math:`\hat{e}_{i}` are the residuals
        from regressing the endogenous variables on the exogenous variables
        and instruments. The null is :math:`\gamma=0` and is implemented
        using a Wald test.  The covariance estimator used in the test is
        identical to the covariance estimator used with ``fit``.
        """
        from linearmodels.iv.model import _OLS

        r = annihilate(self.model.endog.ndarray, self.model._z)
        augx = c_[self.model._x, r]
        mod = _OLS(self.model.dependent, augx)
        res = mod.fit(cov_type=self.cov_type, **self.cov_config)
        norig = self.model._x.shape[1]
        test_params = res.params.values[norig:]
        test_cov = res.cov.values[norig:, norig:]
        stat = test_params.T @ inv(test_cov) @ test_params
        df = len(test_params)
        null = "Endogenous variables are exogenous"
        name = "Wooldridge's regression test of exogeneity"
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def wooldridge_overid(self) -> Union[InvalidTestStatistic, WaldTestStatistic]:
        r"""
        Wooldridge's score test of overidentification

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Wooldridge's test examines whether there is correlation between the
        model residuals and the component of the instruments that is
        orthogonal to the endogenous variables. Define :math:`\tilde{z}`
        to be the residuals of the instruments regressed on the exogenous
        variables and the first-stage fitted values of the endogenous
        variables.  The test is computed as a regression

        .. math ::

          1 = \gamma_1 \hat{\epsilon}_i \tilde{z}_{i,1} + \ldots +
              \gamma_q \hat{\epsilon}_i \tilde{z}_{i,q}

        where :math:`q = n_{instr} - n_{endog}`.  The test is a
        :math:`n\times R^2 \sim \chi^2_{q}`.

        The order of the instruments does not affect this test.
        """
        from linearmodels.iv.model import _OLS

        exog, endog = self.model.exog, self.model.endog
        instruments = self.model.instruments
        nobs, nendog = endog.shape
        ninstr = instruments.shape[1]
        name = "Wooldridge's score test of overidentification"
        if ninstr - nendog == 0:
            return InvalidTestStatistic(
                "Test requires more instruments than " "endogenous variables.",
                name=name,
            )

        endog_hat = proj(endog.ndarray, c_[exog.ndarray, instruments.ndarray])
        q = instruments.ndarray[:, : (ninstr - nendog)]
        q_res = annihilate(q, c_[self.model.exog.ndarray, endog_hat])
        test_functions = q_res * self.resids.values[:, None]
        res = _OLS(ones((nobs, 1)), test_functions).fit(cov_type="unadjusted")

        stat = res.nobs * res.rsquared
        df = ninstr - nendog
        null = "Model is not overidentified."
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def anderson_rubin(self) -> Union[InvalidTestStatistic, WaldTestStatistic]:
        r"""
        Anderson-Rubin test of overidentifying restrictions

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        The Anderson-Rubin test examines whether the value of :math:`\kappa`
        computed for the LIML estimator is sufficiently close to one to
        indicate the model is not overidentified. The test statistic is

        .. math ::

          n \ln(\hat{\kappa}) \sim \chi^2_{q}

        where :math:`q = n_{instr} - n_{endog}`.
        """
        nobs, ninstr = self.model.instruments.shape
        nendog = self.model.endog.shape[1]
        name = "Anderson-Rubin test of overidentification"
        if ninstr - nendog == 0:
            return InvalidTestStatistic(
                "Test requires more instruments than " "endogenous variables.",
                name=name,
            )
        stat = nobs * log(self._liml_kappa)
        df = ninstr - nendog
        null = "The model is not overidentified."
        return WaldTestStatistic(stat, null, df, name=name)

    @cached_property
    def basmann_f(self) -> Union[InvalidTestStatistic, WaldTestStatistic]:
        r"""
        Basmann's F test of overidentifying restrictions

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Basmann's F test examines whether the value of :math:`\kappa`
        computed for the LIML estimator is sufficiently close to one to
        indicate the model is not overidentified. The test statistic is

        .. math ::

          \hat{\kappa} (n -n_{instr})/q \sim F_{q, n - n_{instr}}

        where :math:`q = n_{instr} - n_{endog}`.
        """
        nobs, ninstr = self.model.instruments.shape
        nendog, nexog = self.model.endog.shape[1], self.model.exog.shape[1]
        name = "Basmann' F  test of overidentification"
        if ninstr - nendog == 0:
            return InvalidTestStatistic(
                "Test requires more instruments than " "endogenous variables.",
                name=name,
            )
        df = ninstr - nendog
        df_denom = nobs - (nexog + ninstr)
        stat = (self._liml_kappa - 1) * df_denom / df
        null = "The model is not overidentified."
        return WaldTestStatistic(stat, null, df, df_denom=df_denom, name=name)


class IVGMMResults(_CommonIVResults):
    """
    Results from GMM estimation of IV models

    Parameters
    ----------
    results : dict[str, any]
        A dictionary of results from the model estimation.
    model : {IVGMM, IVGMMCUE}
        The model used to estimate parameters.
    """

    def __init__(
        self, results: Dict[str, Any], model: "linearmodels.iv.model._IVGMMBase"
    ):
        super(IVGMMResults, self).__init__(results, model)
        self._weight_mat = results["weight_mat"]
        self._weight_type = results["weight_type"]
        self._weight_config = results["weight_config"]
        self._iterations = results["iterations"]
        self._j_stat = results["j_stat"]

    @property
    def weight_matrix(self) -> NDArray:
        """Weight matrix used in the final-step GMM estimation"""
        return self._weight_mat

    @property
    def iterations(self) -> int:
        """Iterations used in GMM estimation"""
        return self._iterations

    @property
    def weight_type(self) -> str:
        """Weighting matrix method used in estimation"""
        return self._weight_type

    @property
    def weight_config(self) -> Dict[str, Any]:
        """Weighting matrix configuration used in estimation"""
        return self._weight_config

    @property
    def j_stat(self) -> Union[InvalidTestStatistic, WaldTestStatistic]:
        r"""
        J-test of overidentifying restrictions

        Returns
        -------
        WaldTestStatistic
            J statistic  test of overidentifying restrictions

        Notes
        -----
        The J statistic tests whether the moment conditions are sufficiently
        close to zero to indicate that the model is not overidentified. The
        statistic is defined as

        .. math ::

          n \bar{g}'W^{-1}\bar{g} \sim \chi^2_q

        where :math:`\bar{g} = n^{-1}\sum \hat{\epsilon}_i z_i` where
        :math:`z_i` includes both the exogenous variables and instruments and
        :math:`\hat{\epsilon}_i` are the model residuals. :math:`W` is a consistent
        estimator of the variance of :math:`\sqrt{n}\bar{g}`. The degree of
        freedom is :math:`q = n_{instr} - n_{endog}`.
        """
        return self._j_stat

    def c_stat(
        self, variables: Optional[Union[List[str], str]] = None
    ) -> WaldTestStatistic:
        r"""
        C-test of endogeneity

        Parameters
        ----------
        variables : {str, List[str]}, default None
            List of variables to test for exogeneity.  If None, all variables
            are jointly tested.

        Returns
        -------
        WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        The C statistic iv the difference between the model estimated by
        assuming one or more of the endogenous variables is actually
        exogenous.  The test is implemented as the difference between the
        J statistic s of two GMM estimations where both use the same weighting
        matrix.  The use of a common weighting matrix is required for the C
        statistic to be positive.

        The first model is a estimated uses GMM estimation where one or more
        of the endogenous variables are assumed to be endogenous.  The model
        would be relatively efficient if the assumption were true, and two
        quantities are computed, the J statistic, :math:`J_e`, and the
        moment weighting matrix, :math:`W_e`.

        WLOG assume the q variables tested are in the final q positions so that
        the first :math:`n_{exog} + n_{instr}` rows and columns correspond to
        the moment conditions in the original model. The second J statistic is
        computed using parameters estimated using the original moment
        conditions along with the upper left block of :math:`W_e`.  Denote this
        values as :math:`J_c` where the c is used to indicate consistent.

        The test statistic is then

        .. math ::

          J_e - J_c \sim \chi^2_{m}

        where :math:`m` is the number of variables whose exogeneity is being
        tested.
        """
        dependent, instruments = self.model.dependent, self.model.instruments
        exog, endog = self.model.exog, self.model.endog
        if variables is None:
            exog_e = c_[exog.ndarray, endog.ndarray]
            nobs = exog_e.shape[0]
            endog_e = empty((nobs, 0))
            null = "All endogenous variables are exogenous"
        else:
            if isinstance(variables, list):
                variable_lst = variables
            elif isinstance(variables, str):
                variable_lst = [variables]
            else:
                raise TypeError("variables must be a str of a list of str.")
            exog_e = c_[exog.ndarray, endog.pandas[variable_lst].values]
            ex = [c for c in endog.pandas if c not in variable_lst]
            endog_e = endog.pandas[ex].values
            null = "Variables {0} are exogenous".format(", ".join(variable_lst))
        from linearmodels.iv import IVGMM

        mod = IVGMM(dependent, exog_e, endog_e, instruments)
        res_e = mod.fit(cov_type=self.cov_type, **self.cov_config)
        assert isinstance(res_e, IVGMMResults)
        j_e = res_e.j_stat.stat

        x = self.model._x
        y = self.model._y
        z = self.model._z
        nz = z.shape[1]
        weight_mat_c = asarray(res_e.weight_matrix)[:nz, :nz]
        params_c = mod.estimate_parameters(x, y, z, weight_mat_c)
        from linearmodels.iv.model import IVGMM, IVGMMCUE

        assert isinstance(self.model, (IVGMM, IVGMMCUE))
        j_c = self.model._j_statistic(params_c, weight_mat_c).stat

        stat = j_e - j_c
        df = exog_e.shape[1] - exog.shape[1]
        return WaldTestStatistic(stat, null, df, name="C-statistic")


AnyResult = Union[IVResults, IVGMMResults, OLSResults]


class IVModelComparison(_ModelComparison):
    """
    Comparison of multiple models

    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {'tstats','std_errors', 'std-errors', 'pvalues'}
        Estimator precision estimator to include in the comparison output.
        Default is 'tstats'.
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.
    """

    _supported = (IVResults, IVGMMResults, OLSResults)

    def __init__(
        self,
        results: Union[Sequence[AnyResult], Dict[str, AnyResult]],
        *,
        precision: str = "tstats",
        stars: bool = False,
    ):
        super(IVModelComparison, self).__init__(
            results, precision=precision, stars=stars
        )

    @property
    def rsquared_adj(self) -> Series:
        """Sample-size adjusted coefficients of determination (R**2)"""
        return self._get_property("rsquared_adj")

    @property
    def estimator_method(self) -> Series:
        """Estimation methods"""
        return self._get_property("_method")

    @property
    def cov_estimator(self) -> Series:
        """Covariance estimator descriptions"""
        return self._get_property("cov_estimator")

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results

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
            "Adj. R-squared",
            "F-statistic",
            "P-value (F-stat)",
        ]
        dep_name: Dict[str, str] = {}
        for key in self._results:
            dep_name[key] = str(self._results[key].model.dependent.cols[0])
        dep_names = Series(dep_name)

        vals = concat(
            [
                dep_names,
                self.estimator_method,
                self.nobs,
                self.cov_estimator,
                self.rsquared,
                self.rsquared_adj,
                self.f_statistic,
            ],
            1,
        )
        vals = [[i for i in v] for v in vals.T.values]
        vals[2] = [str(v) for v in vals[2]]
        for i in range(4, len(vals)):
            vals[i] = [_str(v) for v in vals[i]]

        params = self.params
        precision = getattr(self, self._precision)
        pvalues = asarray(self.pvalues)
        params_fmt = []
        params_stub = []

        for i in range(len(params)):
            formatted_and_starred = []
            for v, pv in zip(params.values[i], pvalues[i]):
                formatted_and_starred.append(add_star(_str(v), pv, self._stars))
            params_fmt.append(formatted_and_starred)
            precision_fmt = []
            for v in precision.values[i]:
                v_str = _str(v)
                v_str = "({0})".format(v_str) if v_str.strip() else v_str
                precision_fmt.append(v_str)
            params_fmt.append(precision_fmt)
            params_stub.append(params.index[i])
            params_stub.append(" ")

        vals = table_concat((vals, params_fmt))
        stubs = stub_concat((stubs, params_stub))

        all_instr = []
        for key in self._results:
            res = self._results[key]
            all_instr.append(res.model.instruments.cols)
        ninstr = max(map(len, all_instr))
        instruments = []
        instrument_stub = ["Instruments"]
        for i in range(ninstr):
            if i > 0:
                instrument_stub.append("")
            row = []
            for j in range(len(self._results)):
                instr = all_instr[j]
                if len(instr) > i:
                    row.append(instr[i])
                else:
                    row.append("")
            instruments.append(row)
        if instruments:
            vals = table_concat((vals, instruments))
            stubs = stub_concat((stubs, instrument_stub))

        txt_fmt = default_txt_fmt.copy()
        txt_fmt["data_aligns"] = "r"
        txt_fmt["header_align"] = "r"
        table = SimpleTable(
            vals, headers=models, title=title, stubs=stubs, txt_fmt=txt_fmt
        )
        smry.tables.append(table)
        prec_type = self._PRECISION_TYPES[self._precision]
        smry.add_extra_txt(["{0} reported in parentheses".format(prec_type)])
        return smry


def compare(
    results: Union[Dict[str, AnyResult], Sequence[AnyResult]],
    *,
    precision: str = "tstats",
    stars: bool = False,
) -> IVModelComparison:
    """
    Compare the results of multiple models

    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {'tstats','std_errors', 'std-errors', 'pvalues'}
        Estimator precision estimator to include in the comparison output.
        Default is 'tstats'.
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.

    Returns
    -------
    IVModelComparison
        The model comparison object.
    """
    return IVModelComparison(results, precision=precision, stars=stars)
