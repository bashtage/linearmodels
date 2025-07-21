from __future__ import annotations

from linearmodels.iv.results import IVResults
from linearmodels.panel.results import PanelResults, PanelEffectsResults

# from .model import _IVPanelModelBase

from linearmodels.compat.statsmodels import Summary

from collections.abc import Sequence
import datetime as dt
from functools import cached_property
from typing import Any, Union

from numpy import (
    array,
    asarray,
    c_,
    diag,
    empty,
    isnan,
    log,
    ndarray,
    ones,
    sqrt,
    squeeze,
)
from numpy.linalg import inv
from pandas import DataFrame, Series, concat, to_numeric
import scipy.stats as stats
from statsmodels.iolib.summary import SimpleTable, fmt_2cols, fmt_params
from statsmodels.iolib.table import default_txt_fmt

import linearmodels
from linearmodels.iv._utility import annihilate, proj
from linearmodels.ivpanel.data import IVPanelData
from linearmodels.shared.base import _ModelComparison, _SummaryStr
from linearmodels.shared.hypotheses import (
    InvalidTestStatistic,
    WaldTestStatistic,
    quadratic_form_test,
)
from linearmodels.shared.io import _str, add_star, pval_format
from linearmodels.typing import ArrayLike, Float64Array

class IVPanelEffectsResults(IVResults):
    def __init__(
            self, results: dict[str, Any], model    #: _IVPanelModelBase
    ) -> None:
        super().__init__(results, model)
        self._method = "Panel IV"

    @property
    def nobs(self) -> int:
        """Number of observations"""
        return self.model.dependent.pandas.shape[0]

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
            self.model._entity_effects,
            self.model._time_effects,
            self.model._other_effects,
            self.model.weights,
            self._cov_type,
            self._cov_config,
        )


class FirstStageResults(_SummaryStr):
    """
    First stage estimation results and diagnostics
    """

    def __init__(
        self,
        dep: IVPanelData,
        exog: IVPanelData,
        endog: IVPanelData,
        instr: IVPanelData,
        entity_effect: bool,
            time_effect: bool,
            other_effect: IVPanelData,
        weights: IVPanelData,
        cov_type: str,
        cov_config: dict[str, Any],
    ) -> None:
        self.dep = dep
        self.exog = exog
        self.endog = endog
        self.instr = instr
        self.entity_effect = entity_effect
        self.time_effect = time_effect
        self.other_effect = other_effect if other_effect else None
        self.weights = weights
        reg = c_[self.exog.ndarray, self.endog.ndarray]
        self._reg = DataFrame(reg,
                              columns=self.exog.cols + self.endog.cols,
                              index=self.exog.pandas.index)
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
        from linearmodels.panel.model import PanelOLS
        from linearmodels.ivpanel.model import IVPanel2SLS

        endog, exog, instr, weights = self.endog, self.exog, self.instr, self.weights
        w = sqrt(weights.ndarray)
        z = w * instr.ndarray
        nz = z.shape[1]
        x = w * exog.ndarray
        ez = annihilate(z, x)
        ez = DataFrame(ez, index=instr.pandas.index)
        individual_results = self.individual
        out_df = DataFrame(
            index=["rsquared", "partial.rsquared", "f.stat", "f.pval", "f.dist"],
            columns=[],
        )
        for col in endog.pandas:
            # TODO: BUG in pandas-stubs
            #  https://github.com/pandas-dev/pandas-stubs/issues/97
            y = w * endog.pandas[[col]].values
            ey = annihilate(y, x)
            ey = DataFrame(ey, index=endog.pandas.index)
            config = self._cov_config.copy()
            del config['kappa']
            partial = PanelOLS(ey,
                               ez,
                               entity_effects=self.entity_effect,
                               time_effects=self.time_effect,
                               other_effects=self.other_effect,
                               ).fit(cov_type=self._cov_type.lower(), **config)
            full = individual_results[str(col)]
            params = full.params.values[-nz:]
            params = params[:, None]
            c = asarray(full.cov)[-nz:, -nz:]
            stat = params.T @ inv(c) @ params
            stat = float(stat.squeeze())
            if full._cov_type.lower() in ("homoskedastic", "unadjusted"):
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
        r2sls = IVPanel2SLS(dep.pandas,
                            exog.pandas,
                            endog.pandas,
                            instr.pandas,
                            weights=weights.pandas,
                            entity_effects=self.entity_effect,
                            time_effects=self.time_effect,
                            other_effects=self.other_effect,
                            ).fit(
            cov_type="unadjusted"
        )
        rols = PanelOLS(dep.pandas,
                        self._reg,
                        weights=weights.pandas,
                        entity_effects=self.entity_effect,
                        time_effects=self.time_effect,
                        other_effects=self.other_effect,
                        ).fit(cov_type="unadjusted")
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
        for col in out_df:
            out_df[col] = to_numeric(out_df[col], errors="ignore")

        return out_df

    @cached_property
    def individual(self) -> dict[str, PanelEffectsResults]:
        """
        Individual model results from first-stage regressions

        Returns
        -------
        dict
            Dictionary containing first stage estimation results. Keys are
            the variable names of the endogenous regressors.
        """
        from linearmodels.panel.model import PanelOLS

        exog_instr = DataFrame(
            c_[self.exog.ndarray, self.instr.ndarray],
            columns=self.exog.cols + self.instr.cols,
            index=self.exog.index,
        )
        res: dict[str, PanelEffectsResults] = {}
        for col in self.endog.pandas:
            dep = self.endog.pandas[col]
            mod = PanelOLS(dep,
                           exog_instr,
                           weights=self.weights.pandas,
                           entity_effects=self.entity_effect,
                           time_effects=self.time_effect,
                           other_effects=self.other_effect,
                           )
            config = self._cov_config.copy()
            del config['kappa']
            res[str(col)] = mod.fit(cov_type=self._cov_type.lower(), **config)

        return res

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
                params_fmt[i][j] = f"({params_fmt[i][j]})"

        params_stub = []
        for var in res.params.index:
            params_stub.extend([var, ""])

        title = "First Stage Estimation Results"

        from linearmodels.iv.results import stub_concat, table_concat

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

