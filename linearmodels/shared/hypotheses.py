from typing import Dict, List, Optional, Union

import numpy as np
from pandas.core.series import Series
from patsy.design_info import DesignInfo
from scipy.stats import chi2, f

from linearmodels.typing import ArrayLike, OptionalArrayLike


class WaldTestStatistic(object):
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
    df_denom : int, optional
        Numerator degree of freedom.  If provided, uses an
        F(df, df_denom) distribution.
    name : str, optional
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
        df_denom: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        self._stat = stat
        self._null = null
        self.df = df
        self.df_denom = df_denom
        self._name = name
        if df_denom is None:
            self.dist = chi2(df)
            self.dist_name = "chi2({0})".format(df)
        else:
            self.dist = f(df, df_denom)
            self.dist_name = "F({0},{1})".format(df, df_denom)

    @property
    def stat(self) -> float:
        """Test statistic"""
        return self._stat

    @property
    def pval(self) -> float:
        """P-value of test statistic"""
        return 1 - self.dist.cdf(self.stat)

    @property
    def critical_values(self) -> Optional[Dict[str, float]]:
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
            self.__str__()
            + "\n"
            + self.__class__.__name__
            + ", id: {0}".format(hex(id(self)))
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
    name : str, optional
        Name of test

    See Also
    --------
    WaldTestStatistic
    """

    def __init__(self, reason: str, *, name: Optional[str] = None) -> None:
        self._reason = reason
        super(InvalidTestStatistic, self).__init__(
            np.NaN, "", df=1, df_denom=1, name=name
        )
        self.dist_name = "None"

    @property
    def pval(self) -> float:
        """Always returns np.NaN"""
        return np.NaN

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
    name : str, optional
        Name of test

    See Also
    --------
    WaldTestStatistic
    """

    def __init__(self, *, reason: Optional[str] = None, name: Optional[str] = None):
        self._reason = reason
        if reason is None:
            self._reason = "Test is not applicable to model specification"

        super(InapplicableTestStatistic, self).__init__(
            np.NaN, "", df=1, df_denom=1, name=name
        )
        self.dist_name = "None"

    @property
    def pval(self) -> float:
        """Always returns np.NaN"""
        return np.NaN

    @property
    def critical_values(self) -> None:
        """Always returns None"""
        return None

    def __str__(self) -> str:
        msg = "Irrelevant test statistic\n{reason}\n{name}"
        name = "" if self._name is None else self._name
        return msg.format(name=name, reason=self._reason)


def quadratic_form_test(
    params: ArrayLike,
    cov: ArrayLike,
    restriction: OptionalArrayLike = None,
    value: OptionalArrayLike = None,
    formula: Optional[Union[str, List[str]]] = None,
) -> WaldTestStatistic:
    if formula is not None and restriction is not None:
        raise ValueError("restriction and formula cannot be used" "simultaneously.")
    if formula is not None:
        assert isinstance(params, Series)
        di = DesignInfo(list(params.index))
        lc = di.linear_constraint(formula)
        restriction, value = lc.coefs, lc.constants
    restriction = np.asarray(restriction)
    if value is None:
        value = np.zeros(restriction.shape[0])
    value = np.asarray(value).ravel()[:, None]
    diff = restriction @ np.asarray(params)[:, None] - value
    rcov = restriction @ cov @ restriction.T
    stat = float(diff.T @ np.linalg.inv(rcov) @ diff)
    df = restriction.shape[0]
    null = "Linear equality constraint is valid"
    name = "Linear Equality Hypothesis Test"

    return WaldTestStatistic(stat, null, df, name=name)
