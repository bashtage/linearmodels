from __future__ import annotations

from linearmodels.compat.formulaic import monkey_patch_materializers

from collections.abc import Mapping

from formulaic.utils.constraints import LinearConstraints
import numpy as np
from pandas import Series
from scipy.stats import chi2, f

import linearmodels.typing.data

# Monkey patch parsers if needed, remove once formulaic updated
monkey_patch_materializers()


class WaldTestStatistic:
    """
    Test statistic holder for Wald-type tests

    Parameters
    ----------
    stat : float
        The test statistic
    null : str
        A statement of the test's null hypothesis
    df : int
        Degree of freedom.
    df_denom : int
        Numerator degree of freedom.  If provided, uses an
        F(df, df_denom) distribution.
    name : str
        Name of test

    See Also
    --------
    InvalidTestStatistic
    """

    def __init__(
        self,
        stat: float,
        null: str,
        df: int,
        df_denom: int | None = None,
        name: str | None = None,
    ) -> None:
        self._stat = stat
        self._null = null
        self.df = df
        self.df_denom = df_denom
        self._name = name
        if df_denom is None:
            self.dist = chi2(df)
            self.dist_name = f"chi2({df})"
        else:
            self.dist = f(df, df_denom)
            self.dist_name = f"F({df},{df_denom})"

    @property
    def stat(self) -> float:
        """Test statistic"""
        return self._stat

    @property
    def pval(self) -> float:
        """P-value of test statistic"""
        return 1 - self.dist.cdf(self.stat)

    @property
    def critical_values(self) -> dict[str, float] | None:
        """Critical values test for common test sizes"""
        return dict(zip(["10%", "5%", "1%"], self.dist.ppf([0.9, 0.95, 0.99])))

    @property
    def null(self) -> str:
        """Null hypothesis"""
        return self._null

    def __str__(self) -> str:
        name = ""
        if self._name is not None:
            name = self._name + "\n"
        msg = (
            "{name}H0: {null}\nStatistic: {stat:0.4f}\n"
            "P-value: {pval:0.4f}\nDistributed: {dist}"
        )
        return msg.format(
            name=name,
            null=self.null,
            stat=self.stat,
            pval=self.pval,
            dist=self.dist_name,
        )

    def __repr__(self) -> str:
        return (
            self.__str__() + "\n" + self.__class__.__name__ + f", id: {hex(id(self))}"
        )


class InvalidTestWarning(UserWarning):
    pass


class InvalidTestStatistic(WaldTestStatistic):
    """
    Class returned if a requested test is not valid for a model

    Parameters
    ----------
    reason : str
        Explanation why test is invalid
    name : str
        Name of test

    See Also
    --------
    WaldTestStatistic
    """

    def __init__(self, reason: str, *, name: str | None = None) -> None:
        self._reason = reason
        super().__init__(np.nan, "", df=1, df_denom=1, name=name)
        self.dist_name = "None"

    @property
    def pval(self) -> float:
        """Always returns np.nan"""
        return np.nan

    @property
    def critical_values(self) -> None:
        """Always returns None"""
        return None

    def __str__(self) -> str:
        msg = "Invalid test statistic\n{reason}\n{name}"
        name = "" if self._name is None else self._name
        assert name is not None
        return msg.format(name=name, reason=self._reason)


class InapplicableTestStatistic(WaldTestStatistic):
    """
    Class returned if a requested test is not applicable for a specification

    Parameters
    ----------
    reason : str
        Explanation why test is invalid
    name : str
        Name of test

    See Also
    --------
    WaldTestStatistic
    """

    def __init__(self, *, reason: str | None = None, name: str | None = None):
        self._reason = reason
        if reason is None:
            self._reason = "Test is not applicable to model specification"

        super().__init__(np.nan, "", df=1, df_denom=1, name=name)
        self.dist_name = "None"

    @property
    def pval(self) -> float:
        """Always returns np.nan"""
        return np.nan

    @property
    def critical_values(self) -> None:
        """Always returns None"""
        return None

    def __str__(self) -> str:
        msg = "Irrelevant test statistic\n{reason}\n{name}"
        name = "" if self._name is None else self._name
        return msg.format(name=name, reason=self._reason)


_constraint_error = """
The constraint does not appear to have the required syntax.  Constraints
should have the syntax FORMULA = VALUE where FORMULA is a valid formulaic
formula, e.g., x1 or x1+x2+x3, and VALUE should be a single value
convertible to a float. The constraint seen is {cons}.
"""


def _parse_single(constraint: str) -> tuple[str, float]:
    if "=" not in constraint:
        raise ValueError(_constraint_error.format(cons=constraint))
    parts = constraint.split("=")
    try:
        value = float(parts[-1])
    except Exception:
        raise TypeError(_constraint_error.format(cons=constraint))
    expr = "=".join(parts[:-1])
    return expr, value


def _reparse_constraint_formula(
    formula: str | list[str] | dict[str, float]
) -> str | dict[str, float]:
    # TODO: Test against variable names constaining , or =
    if isinstance(formula, Mapping):
        return dict(formula)
    if isinstance(formula, str):
        if formula.count("=") == 1:
            return formula
        if "," not in formula:
            parts = formula.split("=")
            tail = parts[-1]
            formula = [f"{part} = {tail}" for part in parts[:-1]]
        else:
            formula = list(formula.split(","))
    return dict([_parse_single(cons) for cons in formula])


def quadratic_form_test(
    params: linearmodels.typing.data.ArrayLike,
    cov: linearmodels.typing.data.ArrayLike,
    restriction: linearmodels.typing.data.ArrayLike | None = None,
    value: linearmodels.typing.data.ArrayLike | None = None,
    formula: str | list[str] | dict[str, float] | None = None,
) -> WaldTestStatistic:
    if formula is not None and restriction is not None:
        raise ValueError("restriction and formula cannot be used simultaneously.")
    if formula is not None:
        assert isinstance(params, Series)
        param_names = [str(p) for p in params.index]
        rewritten_constraints = _reparse_constraint_formula(formula)
        lc = LinearConstraints.from_spec(rewritten_constraints, param_names)
        restriction, value = lc.constraint_matrix, lc.constraint_values
    restriction = np.asarray(restriction)
    if value is None:
        value = np.zeros(restriction.shape[0])
    value = np.asarray(value).ravel()[:, None]
    diff = restriction @ np.asarray(params)[:, None] - value
    rcov = restriction @ cov @ restriction.T
    stat = float(np.squeeze(diff.T @ np.linalg.inv(rcov) @ diff))
    df = restriction.shape[0]
    null = "Linear equality constraint is valid"
    name = "Linear Equality Hypothesis Test"

    return WaldTestStatistic(stat, null, df, name=name)
