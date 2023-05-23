from __future__ import annotations

from linearmodels.compat.statsmodels import Summary

from collections.abc import Sequence
from typing import Any

import numpy as np
from pandas import DataFrame, Index, Series, concat


class _SummaryStr:
    """
    Mixin class for results classes to automatically show the summary.
    """

    @property
    def summary(self) -> Summary:
        return Summary()

    def __str__(self) -> str:
        return self.summary.as_text()

    def __repr__(self) -> str:
        return (
            self.__str__() + "\n" + self.__class__.__name__ + f", id: {hex(id(self))}"
        )

    def _repr_html_(self) -> str:
        return self.summary.as_html() + f"<br/>id: {hex(id(self))}"


class _ModelComparison(_SummaryStr):
    """
    Base class for model comparisons
    """

    _supported: tuple[type, ...] = tuple()
    _PRECISION_TYPES = {
        "tstats": "T-stats",
        "pvalues": "P-values",
        "std_errors": "Std. Errors",
    }

    # TODO: Replace Any with better list of types
    def __init__(
        self,
        results: dict[str, Any] | Sequence[Any],
        *,
        precision: str = "tstats",
        stars: bool = False,
    ) -> None:
        if not isinstance(results, dict):
            _results: dict[str, Any] = {}
            for i, res in enumerate(results):
                _results["Model " + str(i)] = res
        else:
            _results = {}
            _results.update(results)
        self._results = _results

        for key in self._results:
            if not isinstance(self._results[key], self._supported):
                raise TypeError("Results from unknown model")
        precision = precision.lower().replace("-", "_")
        if precision not in ("tstats", "pvalues", "std_errors"):
            raise ValueError(
                "Unknown precision value. Must be one of 'tstats', 'std_errors' "
                "or 'pvalues'."
            )
        self._precision = precision
        self._stars = stars

    def _get_series_property(self, name: str) -> DataFrame:
        out: list[tuple[str, Series]] = [
            (k, getattr(v, name)) for k, v in self._results.items()
        ]
        cols = [v[0] for v in out]
        values = concat([v[1] for v in out], axis=1, sort=False)
        # TODO: Remove once pandas typing fixed
        values.columns = Index(cols)
        return values

    def _get_property(self, name: str) -> Series:
        out = {}
        items = []
        for k, v in self._results.items():
            items.append(k)
            out[k] = getattr(v, name)
        return Series(out, name=name).loc[items]

    @property
    def nobs(self) -> Series:
        """Parameters for all models"""
        return self._get_property("nobs")

    @property
    def params(self) -> DataFrame:
        """Parameters for all models"""
        return self._get_series_property("params")

    @property
    def tstats(self) -> DataFrame:
        """Parameter t-stats for all models"""
        return self._get_series_property("tstats")

    @property
    def std_errors(self) -> DataFrame:
        """Parameter standard errors for all models"""
        return self._get_series_property("std_errors")

    @property
    def pvalues(self) -> DataFrame:
        """Parameter p-vals for all models"""
        return self._get_series_property("pvalues")

    @property
    def rsquared(self) -> Series:
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared")

    @property
    def f_statistic(self) -> DataFrame:
        """F-statistics and P-values"""
        out = self._get_property("f_statistic")
        out_df = DataFrame(
            np.empty((len(out), 2)), columns=["F stat", "P-value"], index=out.index
        )
        for loc in out.index:
            # TODO: Bug in pandas-stubs
            out_df.loc[loc] = out[loc].stat, out[loc].pval  # type: ignore
        return out_df
