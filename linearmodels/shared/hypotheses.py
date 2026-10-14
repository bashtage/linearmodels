"""
Test statistics and hypothesis tests that the models share.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formulaic.utils.constraints import LinearConstraints
import numpy as np
from pandas import Series
from scipy.stats import chi2, f, norm

import linearmodels.typing.data


class WaldTestStatistic:
    """
    Test statistic holder for Wald-type tests.

    Parameters
    ----------
    stat : float
        The test statistic.
    null : str
        A statement of the test's null hypothesis.
    df : int
        Degree of freedom, which is the numerator degree of freedom if
        ``df_denom`` is given.
    df_denom : int, optional
        Denominator degree of freedom. If provided, the p-value and the critical
        values use an F(df, df_denom) distribution. Otherwise they use a
        chi-squared distribution with ``df`` degrees of freedom.
    name : str, optional
        Name of test.

    See Also
    --------
    InvalidTestStatistic
        Returned if a requested test is not valid for a model.
    InapplicableTestStatistic
        Returned if a requested test is not applicable for a specification.
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
        self.dist: Any
        if df_denom is None:
            self.dist = chi2(df)
            self.dist_name = f"chi2({df})"
        else:
            self.dist = f(df, df_denom)
            self.dist_name = f"F({df},{df_denom})"

    @property
    def stat(self) -> float:
        """Test statistic."""
        return self._stat

    @property
    def pval(self) -> float:
        """P-value of the test statistic."""
        return self.dist.sf(self.stat)

    @property
    def critical_values(self) -> dict[str, float] | None:
        """Critical values for common test sizes."""
        return dict(
            zip(["10%", "5%", "1%"], self.dist.ppf([0.9, 0.95, 0.99]), strict=False)
        )

    @property
    def null(self) -> str:
        """Null hypothesis of the test."""
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


class NormalTestStatistic:
    r"""
    Test statistic holder for tests with an asymptotic standard normal law.

    Parameters
    ----------
    stat : float
        The test statistic.
    null : str
        A statement of the test's null hypothesis.
    name : str, optional
        Name of test.
    two_sided : bool, optional
        Flag indicating whether p-values and critical values are computed
        using a two-sided test. The default is True.

    Notes
    -----
    A two-sided test rejects the null if the absolute value of the statistic
    :math:`s` is large. The p-value is :math:`2\Pr(Z > |s|)`, where :math:`Z`
    has a standard normal distribution, and the critical values for tests of
    size 10%, 5% and 1% are the 0.95, 0.975 and 0.995 quantiles of the standard
    normal.

    A one-sided test rejects the null if the statistic is large. The p-value is
    :math:`\Pr(Z > s)`, which is close to 1 if the statistic is large and
    negative, and the critical values are the 0.9, 0.95 and 0.99 quantiles.
    """

    def __init__(
        self,
        stat: float,
        null: str,
        *,
        name: str | None = None,
        two_sided: bool = True,
    ) -> None:
        self._stat = stat
        self._null = null
        self._name = name
        self._two_sided = two_sided
        self.dist = norm()
        self.dist_name = "N(0,1)"

    @property
    def stat(self) -> float:
        """Test statistic."""
        return self._stat

    @property
    def pval(self) -> float:
        """P-value of the test statistic."""
        if self._two_sided:
            return 2 * self.dist.sf(abs(self.stat))
        return self.dist.sf(self.stat)

    @property
    def critical_values(self) -> dict[str, float]:
        """Critical values for common test sizes."""
        if self._two_sided:
            quantiles = [0.95, 0.975, 0.995]
        else:
            quantiles = [0.9, 0.95, 0.99]
        return dict(zip(["10%", "5%", "1%"], self.dist.ppf(quantiles), strict=False))

    @property
    def null(self) -> str:
        """Null hypothesis of the test."""
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
    """Warning that a test is not valid."""


class InvalidTestStatistic(WaldTestStatistic):
    """
    Class returned if a requested test is not valid for a model.

    Parameters
    ----------
    reason : str
        Explanation why test is invalid.
    name : str, optional
        Name of test.

    See Also
    --------
    WaldTestStatistic
        Test statistic holder for Wald-type tests.
    """

    def __init__(self, reason: str, *, name: str | None = None) -> None:
        self._reason = reason
        super().__init__(np.nan, "", df=1, df_denom=1, name=name)
        self.dist_name = "None"

    @property
    def pval(self) -> float:
        """P-value, which is always NaN."""
        return np.nan

    @property
    def critical_values(self) -> None:
        """Critical values, which are always None."""
        return None

    def __str__(self) -> str:
        msg = "Invalid test statistic\n{reason}\n{name}"
        name = "" if self._name is None else self._name
        assert name is not None
        return msg.format(name=name, reason=self._reason)


class InapplicableTestStatistic(WaldTestStatistic):
    """
    Class returned if a requested test is not applicable for a specification.

    Parameters
    ----------
    reason : str, optional
        Explanation why the test is not applicable. The default says that the
        test is not applicable to the model specification.
    name : str, optional
        Name of test.

    See Also
    --------
    WaldTestStatistic
        Test statistic holder for Wald-type tests.
    """

    def __init__(self, *, reason: str | None = None, name: str | None = None):
        self._reason = reason
        if reason is None:
            self._reason = "Test is not applicable to model specification"

        super().__init__(np.nan, "", df=1, df_denom=1, name=name)
        self.dist_name = "None"

    @property
    def pval(self) -> float:
        """P-value, which is always NaN."""
        return np.nan

    @property
    def critical_values(self) -> None:
        """Critical values, which are always None."""
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
    except Exception as exc:
        raise TypeError(_constraint_error.format(cons=constraint)) from exc
    expr = "=".join(parts[:-1])
    return expr.strip(), value


def _reparse_constraint_formula(
    formula: str | list[str] | dict[str, float],
) -> str | dict[str, float]:
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
    r"""
    Test linear equality constraints using a Wald test.

    Parameters
    ----------
    params : array_like
        Parameter estimates with p elements. It must be a Series, indexed by the
        names of the parameters, if ``formula`` is used.
    cov : array_like
        Covariance of the parameter estimates, with shape (p, p).
    restriction : array_like, optional
        Array with q rows and p columns containing linear weights to apply to
        parameters when forming the restrictions. It is not possible to use both
        ``restriction`` and ``formula``.
    value : array_like, optional
        Array with q elements containing the restricted values. The default is
        zeros.
    formula : str, list of str or dict of str to float, optional
        Linear constraints in the formulaic syntax. The names in the constraints
        are the entries of the index of ``params``. It is one of:

        * A single comma-separated string such as "x1=0, x2+x3=1"
        * A list of strings where each element is a single constraint
          such as ["x1=0", "x2+x3=1"]
        * A single string without commas to test simple constraints such
          as "x1=x2=x3=0"
        * A dictionary where each key is a parameter restriction and
          the corresponding value is the restriction value, e.g.,
          {"x1": 0, "x2+x3": 1}.

    Returns
    -------
    WaldTestStatistic
        Test statistic for the null that the restrictions are valid. It has a
        chi-squared distribution with q degrees of freedom.

    Raises
    ------
    ValueError
        If both ``restriction`` and ``formula`` are provided, or if a string
        constraint does not contain an equal sign.
    TypeError
        If ``formula`` is used and ``params`` is not a Series, or if the value of
        a constraint cannot be converted to a float.

    Notes
    -----
    The null hypothesis is :math:`H_0:C\theta=v`, where the matrix C is
    ``restriction`` and v is ``value``. The statistic is
    :math:`(C\hat\theta-v)'(C\hat\Sigma C')^{-1}(C\hat\theta-v)`, where
    :math:`\hat\Sigma` is ``cov``.
    """
    if formula is not None and restriction is not None:
        raise ValueError("restriction and formula cannot be used simultaneously.")
    if formula is not None:
        if not isinstance(params, Series):
            raise TypeError(
                "params must be a pandas Series when using formula= to specify "
                "linear restrictions (indexed by parameter names)."
            )
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
