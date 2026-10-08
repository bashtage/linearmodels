"""
P-values must be accurate in the tails

1 - cdf(x) is exactly 0.0 once the tail probability falls below about 1e-16, and
it loses digits well before that, since it can only resolve differences from 1
of about 1e-16. The survival function is accurate far into the tail. Each model
below has parameters with |t| between roughly 10 and 30, where the old
expression gave a p-value that was either 0.0 or wrong in the leading digits.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

from linearmodels.asset_pricing import TradedFactorModel
from linearmodels.iv import IV2SLS
from linearmodels.panel import PooledOLS
from linearmodels.shared.hypotheses import WaldTestStatistic
from linearmodels.system import SUR


def check_two_sided(tstats, pvalues, df=None):
    """
    Compare with the two-sided p-value computed from the survival function

    The statistics must be large enough to be in the region where 1 - cdf fails
    and small enough that the p-value is representable as a float.
    """
    abs_t = np.abs(np.asarray(tstats, dtype=float))
    assert abs_t.max() > 9
    assert abs_t.max() < 35
    dist = stats.norm if df is None else stats.t(df)
    expected = 2 * dist.sf(abs_t)
    assert (np.asarray(pvalues) > 0).all()
    assert_allclose(pvalues, expected, rtol=1e-10)


@pytest.mark.parametrize(
    ("stat", "df", "df_denom", "expected"),
    [
        (30.0, 1, None, 4.320463057827497e-08),
        (45.0, 3, None, 9.252702104537278e-10),
        (100.0, 2, None, 1.928749847963918e-22),
        (150.0, 3, None, 2.634913928488045e-32),
        (90.0, 2, 25, 3.778564545161538e-12),
        (200.0, 4, 60, 5.896396059202165e-34),
    ],
)
def test_wald_statistic_tail(stat, df, df_denom, expected):
    # Reference values are R's pchisq and pf with lower.tail=FALSE
    ts = WaldTestStatistic(stat, "_NULL_", df, df_denom)
    assert_allclose(ts.pval, expected, rtol=1e-10)


@pytest.mark.parametrize("debiased", [True, False])
def test_iv_pvalues(debiased):
    rng = np.random.default_rng(20261008)
    n = 100
    z = rng.standard_normal(n)
    x = z + 0.3 * rng.standard_normal(n)
    y = 1.0 + x + 0.5 * rng.standard_normal(n)
    exog = pd.DataFrame({"const": np.ones(n)})
    res = IV2SLS(y, exog, x, z).fit(cov_type="unadjusted", debiased=debiased)
    df = res.df_resid if debiased else None
    check_two_sided(res.tstats, res.pvalues, df)


@pytest.mark.parametrize("debiased", [True, False])
def test_panel_pvalues(debiased):
    rng = np.random.default_rng(20261009)
    index = pd.MultiIndex.from_product([range(20), range(5)])
    x = pd.Series(rng.standard_normal(100), index=index, name="x")
    y = pd.Series(1.0 + x + 0.5 * rng.standard_normal(100), index=index, name="y")
    exog = pd.DataFrame({"const": 1.0, "x": x})
    res = PooledOLS(y, exog).fit(cov_type="unadjusted", debiased=debiased)
    df = res.df_resid if debiased else None
    check_two_sided(res.tstats, res.pvalues, df)


@pytest.mark.parametrize("debiased", [True, False])
def test_system_pvalues(debiased):
    rng = np.random.default_rng(20261010)
    n = 100
    equations = {}
    for i, name in enumerate(("a", "b")):
        x = pd.DataFrame({"const": np.ones(n), "x": rng.standard_normal(n)})
        y = 1.0 + (0.6 + 0.4 * i) * x["x"] + 0.5 * rng.standard_normal(n)
        equations[name] = {"dependent": y, "exog": x}
    res = SUR(equations).fit(cov_type="unadjusted", debiased=debiased)
    df = res.df_resid if debiased else None
    check_two_sided(res.tstats, res.pvalues, df)


def test_asset_pricing_pvalues():
    rng = np.random.default_rng(20261011)
    n = 100
    factors = pd.DataFrame({"mkt": rng.standard_normal(n)})
    noise = 0.5 * rng.standard_normal((n, 3))
    portfolios = pd.DataFrame(
        factors.to_numpy() @ np.array([[0.8, 1.0, 1.2]]) + noise,
        columns=["p1", "p2", "p3"],
    )
    res = TradedFactorModel(portfolios, factors).fit()
    check_two_sided(res.tstats, res.pvalues)
    # The summary uses the same calculation
    assert "mkt" in str(res.summary)
