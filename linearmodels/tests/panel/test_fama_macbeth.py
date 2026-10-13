from contextlib import nullcontext
from itertools import product
import re
import warnings

import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats
from statsmodels.regression.linear_model import OLS

# Not imported as test_mvmean, which pytest would collect as a test
from statsmodels.stats.multivariate import test_mvmean as hotelling_test

from linearmodels.panel import model as panel_model
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import FamaMacBeth
from linearmodels.shared.exceptions import (
    InferenceUnavailableWarning,
    MissingValueWarning,
)
from linearmodels.shared.hypotheses import InvalidTestStatistic
from linearmodels.shared.io import pval_format
from linearmodels.tests._utility import warns_if_missing
from linearmodels.tests.panel._utility import (
    access_attributes,
    assert_frame_similar,
    datatypes,
    generate_data,
)

missing = [0.0, 0.20]
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = ["-".join(str(param) for param in perm) for perm in perms]


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(
        missing, datatype, const=const, other_effects=1, ntk=(25, 200, 5)
    )


def test_fama_macbeth(data):
    with warns_if_missing(data.y, data.x, match=None):
        res = FamaMacBeth(data.y, data.x).fit(debiased=True)
    y = PanelData(data.y)
    x = PanelData(data.x)
    missing = y.isnull | x.isnull
    y.drop(missing)
    x.drop(missing)
    y = y.dataframe
    x = x.dataframe
    times = y.index.levels[1]
    params = []
    for t in times:
        _y = y.xs(t, level=1)
        _x = x.xs(t, level=1)
        if _x.shape[0] < _x.shape[1]:
            continue
        _x = _x.loc[_y.index]
        params.append(lstsq(_x.values, _y.values, rcond=None)[0])
    params = np.array(params).squeeze()
    all_params = params
    params = params.mean(0)
    assert_allclose(params.squeeze(), res.params)
    assert_allclose(all_params, res.all_params.dropna(how="all"))
    e_params = all_params - params[None, :]
    ntime = e_params.shape[0]
    cov = e_params.T @ e_params / ntime / (ntime - 1)
    assert_allclose(cov, np.asarray(res.cov))
    # Inference uses a t distribution with T - 1 degrees of freedom, where T is
    # the number of cross-sectional regressions that are averaged
    assert res.df_resid == ntime - 1
    tstats = params.squeeze() / np.sqrt(np.diag(cov))
    assert_allclose(res.pvalues, 2 * stats.t(ntime - 1).sf(np.abs(tstats)))
    access_attributes(res)


def test_unknown_cov_type(data):
    with pytest.raises(ValueError, match=r"Unknown cov_type"):
        with warns_if_missing(data.y, data.x, match=None):
            FamaMacBeth(data.y, data.x).fit(cov_type="unknown")


@pytest.mark.smoke
def test_fama_macbeth_kernel_smoke(data):
    with warns_if_missing(data.y, data.x, match=None):
        FamaMacBeth(data.y, data.x).fit(cov_type="kernel")
        FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="bartlett")
        FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="newey-west")
        FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="parzen")
        FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="qs")
        FamaMacBeth(data.y, data.x).fit(cov_type="kernel", bandwidth=3)
        res = FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="andrews")
    access_attributes(res)


def test_fitted_effects_residuals(data):
    with warns_if_missing(data.y, data.x, match=None):
        mod = FamaMacBeth(data.y, data.x)
    res = mod.fit()

    expected = mod.exog.values2d @ res.params.values
    expected = pd.DataFrame(expected, index=mod.exog.index, columns=["fitted_values"])
    assert_allclose(res.fitted_values, expected)
    assert_frame_similar(res.fitted_values, expected)

    idiosyncratic = pd.DataFrame(
        mod.dependent.values2d - expected.values, index=expected.index
    )
    expected.iloc[:, 0] = idiosyncratic.iloc[:, 0]
    expected.columns = ["idiosyncratic"]
    assert_allclose(res.idiosyncratic, expected)
    assert_frame_similar(res.idiosyncratic, expected)

    expected.iloc[:, 0] = np.nan
    expected.columns = ["estimated_effects"]
    assert_allclose(res.estimated_effects, expected)
    assert_frame_similar(res.estimated_effects, expected)


def test_block_size_warnings():
    y = np.arange(12.0)[:, None]
    x = np.ones((12, 3))
    x[:, 1] = np.arange(12.0)
    x[:, 2] = np.arange(12.0) ** 2
    idx = pd.MultiIndex.from_product(
        [["a", "b", "c"], pd.date_range("2000-1-1", periods=4)]
    )
    y = pd.DataFrame(y, index=idx, columns=["y"])
    x = pd.DataFrame(x, index=idx, columns=["x1", "x2", "x3"])
    # The periods are cross-sectional regressions, and not time-series regressions
    with pytest.warns(
        MissingValueWarning, match=r"1 of the cross-sectional regressions"
    ):
        FamaMacBeth(y.iloc[:11], x.iloc[:11])
    with pytest.warns(
        InferenceUnavailableWarning, match=r"cross-sectional regressions that can be"
    ):
        FamaMacBeth(y.iloc[::4], x.iloc[::4])


def test_block_size_error():
    y = np.arange(12.0)[:, None]
    x = np.ones((12, 2))
    x[1::4, 1] = 2
    x[2::4, 1] = 3
    idx = pd.MultiIndex.from_product(
        [["a", "b", "c"], pd.date_range("2000-1-1", periods=4)]
    )
    y = pd.DataFrame(y, index=idx, columns=["y"])
    x = pd.DataFrame(x, index=idx, columns=["x1", "x2"])
    with pytest.raises(ValueError, match=r"Model cannot be estimated"):
        FamaMacBeth(y, x)


def test_limited_redundancy():
    data = generate_data(
        0, datatype="numpy", const=False, other_effects=1, ntk=(25, 200, 5)
    )
    for i in range(0, data.x.shape[1], 7):
        data.x[1, i, :] = data.x[0, i, :]
    with pytest.warns(MissingValueWarning, match="cross-sectional regressions"):
        mod = FamaMacBeth(data.y, data.x)
    res = mod.fit()
    assert np.any(np.isnan(res.all_params))


@pytest.mark.parametrize("just_id", [True, False])
@pytest.mark.parametrize("use_const", [True, False])
def test_avg_r2(use_const, just_id):
    nvar = 10 if not just_id else 3 + int(use_const)
    mi = pd.MultiIndex.from_product([np.arange(nvar), np.arange(10)])
    rg = np.random.default_rng(0)
    n = len(mi)
    e = rg.standard_normal((n, 3))
    cols = ["x1", "x2", "x3"]
    x = pd.DataFrame(e, index=mi, columns=cols)
    if use_const:
        const = pd.Series(np.ones(n), index=mi, name="const")
        x = pd.concat([const, x], axis=1)
    y = pd.Series(rg.standard_normal(n), index=mi, name="y")
    res = FamaMacBeth(y, x).fit()
    z = pd.concat([y, x], axis=1)
    r2 = []
    r2_adj = []
    for _, g in z.groupby(level=1):
        with warnings.catch_warnings():
            # statsmodels divides by the 0 degrees of freedom of the just identified
            # regressions
            warnings.simplefilter("ignore", RuntimeWarning)
            ols_res = OLS(g["y"], g[x.columns]).fit()
            r2.append(ols_res.rsquared)
            r2_adj.append(ols_res.rsquared_adj)
    assert_allclose(np.array(r2).mean(), res.avg_rsquared)
    assert_allclose(np.array(r2_adj).mean(), res.avg_adj_rsquared)


def fama_macbeth_panel(nentity=40, nperiod=10, nreg=1, seed=0):
    """Panel where the slopes change from period to period"""
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    x = pd.DataFrame(
        rg.standard_normal((nentity * nperiod, nreg)),
        index=index,
        columns=[f"x{i}" for i in range(nreg)],
    )
    slopes = 0.3 + 0.5 * rg.standard_normal((nperiod, nreg))
    signal = np.sum(x.to_numpy() * slopes[index.get_level_values(1)], axis=1)
    y = pd.Series(signal + rg.standard_normal(nentity * nperiod), index=index, name="y")
    x.insert(0, "const", 1.0)
    return y, x


def cross_sections(y, x, weights=None):
    """The estimates of the cross-sections that can be estimated, by period"""
    period = x.index.get_level_values(1)
    estimates = {}
    for value in period.unique():
        sel = period == value
        exog, dep = x.to_numpy()[sel], y.to_numpy()[sel]
        if weights is not None:
            root_w = np.sqrt(weights.to_numpy()[sel])
            exog, dep = exog * root_w[:, None], dep * root_w
        if exog.shape[0] < exog.shape[1] or np.linalg.matrix_rank(exog) < exog.shape[1]:
            continue
        estimates[value] = np.linalg.lstsq(exog, dep, rcond=None)[0]
    return pd.DataFrame(estimates, index=x.columns).T


def by_hand(estimates, *, debiased=True, bandwidth=0, level=0.95):
    """
    Parameters, covariance, p-values and the interval from the T estimates

    The Bartlett kernel covariance of the estimates, with weights 1 - j / (m + 1)
    for the lags j = 1, ..., m. The reference distribution is the t distribution
    with T - 1 degrees of freedom if debiased, and the normal distribution if not
    """
    nperiod = estimates.shape[0]
    params = estimates.mean().to_numpy()
    dev = estimates.to_numpy() - params
    long_run = dev.T @ dev
    for lag in range(1, bandwidth + 1):
        gamma = dev[lag:].T @ dev[:-lag]
        long_run += (1 - lag / (bandwidth + 1)) * (gamma + gamma.T)
    cov = long_run / nperiod / (nperiod - int(debiased))
    std_errors = np.sqrt(np.diag(cov))
    dist = stats.t(nperiod - 1) if debiased else stats.norm
    pvalues = 2 * dist.sf(np.abs(params / std_errors))
    crit = dist.ppf(0.5 + level / 2)
    interval = np.c_[params - crit * std_errors, params + crit * std_errors]
    return params, cov, pvalues, interval


def dropped_periods_panel():
    """
    Panel in which 4 of the 25 periods cannot be estimated

    In periods 2, 5 and 9 x1 does not vary across entities, so it is collinear
    with the constant. In period 7 there are 2 entities for 3 regressors.
    """
    y, x = fama_macbeth_panel(nentity=30, nperiod=25, nreg=2, seed=3)
    entity, period = x.index.get_level_values(0), x.index.get_level_values(1)
    x.loc[np.isin(period, [2, 5, 9]), "x1"] = 1.5
    keep = ~((period == 7) & (entity >= 2))
    return y[keep], x[keep]


COV_CONFIGS = [
    ("unadjusted", None),
    ("robust", None),
    ("kernel", 0),
    ("kernel", 1),
    ("kernel", 3),
]


@pytest.mark.parametrize("debiased", [True, False])
@pytest.mark.parametrize(
    ("nperiod", "cov_type", "bandwidth"),
    [
        (nperiod, cov_type, bandwidth)
        for nperiod in (2, 4, 10)
        for cov_type, bandwidth in COV_CONFIGS
        # The bandwidth is smaller than the number of periods
        if bandwidth is None or bandwidth < nperiod
    ],
)
def test_inference_by_hand(nperiod, cov_type, bandwidth, debiased):
    y, x = fama_macbeth_panel(nentity=40, nperiod=nperiod, nreg=1, seed=nperiod)
    res = FamaMacBeth(y, x).fit(
        cov_type=cov_type, kernel="bartlett", bandwidth=bandwidth, debiased=debiased
    )
    params, cov, pvalues, interval = by_hand(
        cross_sections(y, x), debiased=debiased, bandwidth=bandwidth or 0
    )
    # The estimates and their covariance
    assert_allclose(res.params, params, rtol=1e-10)
    assert_allclose(res.cov, cov, rtol=1e-10)
    # The degrees of freedom are those of the T estimates, and not nobs - nvar
    assert res.df_resid == nperiod - 1
    assert_allclose(res.pvalues, pvalues, rtol=1e-10)
    assert_allclose(res.conf_int(), interval, rtol=1e-10)
    interval_90 = by_hand(
        cross_sections(y, x),
        debiased=debiased,
        bandwidth=bandwidth or 0,
        level=0.9,
    )[3]
    assert_allclose(res.conf_int(level=0.9), interval_90, rtol=1e-10)


def test_inference_n40_t10():
    # The example of #258: the same t-statistic, 4.27, has the p-value 0.0021
    # and the interval 0.1207 to 0.3932 with 9 degrees of freedom, and 0.00002
    # and 0.1385 to 0.3754 with the 398 degrees of freedom of nobs - nvar
    rg = np.random.default_rng(1)
    nentity, nperiod = 40, 10
    index = pd.MultiIndex.from_product(
        [range(nentity), range(nperiod)], names=["entity", "time"]
    )
    x = pd.DataFrame(
        {"const": 1.0, "x": rg.standard_normal(nentity * nperiod)}, index=index
    )
    y = pd.Series(
        0.15 * x["x"].to_numpy() + rg.standard_normal(nentity * nperiod),
        index=index,
        name="y",
    )
    res = FamaMacBeth(y, x).fit()
    slopes = cross_sections(y, x)["x"]
    tstat = slopes.mean() / (slopes.std(ddof=1) / np.sqrt(nperiod))
    assert_allclose(res.tstats["x"], tstat)
    assert_allclose(tstat, 4.2662, atol=1e-4)
    assert res.df_resid == nperiod - 1
    assert_allclose(res.pvalues["x"], 2 * stats.t(9).sf(tstat), rtol=1e-10)
    assert_allclose(res.pvalues["x"], 0.00209, atol=5e-6)
    assert_allclose(res.conf_int().loc["x"], [0.1207, 0.3932], atol=1e-4)
    assert res.pvalues["x"] > 50 * 2 * stats.t(398).sf(tstat)


@pytest.mark.parametrize("debiased", [True, False])
@pytest.mark.parametrize("bandwidth", [0, 2])
def test_inference_dropped_periods(bandwidth, debiased):
    # T is the number of periods that are estimated, 21 of the 25 periods
    y, x = dropped_periods_panel()
    with pytest.warns(
        MissingValueWarning, match=r"4 of the cross-sectional regressions"
    ):
        mod = FamaMacBeth(y, x)
    cov_type = "kernel" if bandwidth else "unadjusted"
    res = mod.fit(
        cov_type=cov_type, kernel="bartlett", bandwidth=bandwidth, debiased=debiased
    )
    estimates = cross_sections(y, x)
    assert estimates.shape[0] == 21
    assert res.all_params.shape[0] == 25
    assert res.all_params.isnull().all(axis=1).sum() == 4
    assert res.df_resid == 20

    params, cov, pvalues, interval = by_hand(
        estimates, debiased=debiased, bandwidth=bandwidth
    )
    assert_allclose(res.params, params, rtol=1e-10)
    assert_allclose(res.cov, cov, rtol=1e-10)
    assert_allclose(res.pvalues, pvalues, rtol=1e-10)
    assert_allclose(res.conf_int(), interval, rtol=1e-10)
    if not debiased:
        return
    # Neither the number of periods nor nobs - nvar gives these p-values
    for df in (24, int(res.nobs) - 3):
        wrong = 2 * stats.t(df).sf(np.abs(params / np.sqrt(np.diag(cov))))
        assert not np.allclose(res.pvalues, wrong, rtol=1e-3)


@pytest.mark.parametrize("kernel", [None, "bartlett", "parzen", "qs", "andrews"])
@pytest.mark.parametrize("bandwidth", [None, 3])
def test_kernel_covariances_use_t_distribution(kernel, bandwidth):
    # The kernel covariances also use the T - 1 degrees of freedom, with the
    # automatic bandwidth and a bandwidth that is given
    y, x = dropped_periods_panel()
    with pytest.warns(
        MissingValueWarning, match=r"4 of the cross-sectional regressions"
    ):
        res = FamaMacBeth(y, x).fit(
            cov_type="kernel", kernel=kernel, bandwidth=bandwidth
        )
    nperiod = int(res.all_params.notnull().all(axis=1).sum())
    assert nperiod == 21
    assert res.df_resid == 20
    tstats = res.params / res.std_errors
    assert_allclose(res.pvalues, 2 * stats.t(20).sf(np.abs(tstats)), rtol=1e-10)
    crit = stats.t(20).ppf(0.975)
    lower = res.params - crit * res.std_errors
    upper = res.params + crit * res.std_errors
    assert_allclose(res.conf_int()["lower"], lower, rtol=1e-10)
    assert_allclose(res.conf_int()["upper"], upper, rtol=1e-10)


def test_estimates_and_standard_errors_are_those_of_the_estimates():
    # The degrees of freedom of inference do not change the estimates or the
    # standard errors, which are those of the T estimates
    y, x = fama_macbeth_panel(nentity=40, nperiod=10, nreg=2, seed=11)
    debiased = FamaMacBeth(y, x).fit(debiased=True)
    estimates = cross_sections(y, x)
    assert_allclose(debiased.params, estimates.mean())
    assert_allclose(debiased.std_errors, estimates.std(ddof=1) / np.sqrt(10))
    not_debiased = FamaMacBeth(y, x).fit(debiased=False)
    assert_allclose(not_debiased.params, debiased.params)
    assert_allclose(not_debiased.std_errors, debiased.std_errors * np.sqrt(9 / 10))


def hotelling_by_hand(estimates, *, bandwidth=0, debiased=True):
    """
    Wald statistic of the slopes, Hotelling's F and its p-value, from the T estimates

    The first column of the estimates is the constant, and the tested parameters
    are the others. The Wald statistic W uses the covariance of the average of
    the estimates. With bandwidth 0 it is their sample covariance, with the divisor
    T - 1 if debiased, divided by T, and W is Hotelling's T-squared statistic of
    the slopes. With a bandwidth it is the Bartlett kernel estimate. F =
    (T - k) / (k (T - 1)) W is compared with an F distribution with k and T - k
    degrees of freedom. If not debiased W is compared with a chi-squared
    distribution with k degrees of freedom
    """
    nperiod, nslope = estimates.shape[0], estimates.shape[1] - 1
    params, cov, _, _ = by_hand(estimates, debiased=debiased, bandwidth=bandwidth)
    slopes, slope_cov = params[1:], cov[1:, 1:]
    wald = float(slopes @ np.linalg.solve(slope_cov, slopes))
    if not debiased:
        return wald, wald, stats.chi2(nslope).sf(wald)
    stat = (nperiod - nslope) / (nslope * (nperiod - 1)) * wald
    return wald, stat, stats.f(nslope, nperiod - nslope).sf(stat)


@pytest.mark.parametrize(
    ("cov_type", "bandwidth"),
    [("unadjusted", None), ("robust", None), ("kernel", 0), ("kernel", 2)],
)
@pytest.mark.parametrize("nperiod", [8, 10, 20])
@pytest.mark.parametrize("nreg", [1, 2, 3, 5])
def test_robust_f_statistic_is_hotellings_t_squared(nreg, nperiod, cov_type, bandwidth):
    # The Wald statistic of the slopes is multiplied by (T - k) / (k (T - 1)) and
    # compared with an F distribution with k and T - k degrees of freedom
    y, x = fama_macbeth_panel(
        nentity=40, nperiod=nperiod, nreg=nreg, seed=100 * nreg + nperiod
    )
    res = FamaMacBeth(y, x).fit(
        cov_type=cov_type, kernel="bartlett", bandwidth=bandwidth
    )
    _, stat, pvalue = hotelling_by_hand(cross_sections(y, x), bandwidth=bandwidth or 0)
    robust = res.f_statistic_robust
    assert robust.df == nreg
    assert robust.df_denom == nperiod - nreg
    assert robust.dist_name == f"F({nreg},{nperiod - nreg})"
    assert_allclose(robust.stat, stat, rtol=1e-10)
    assert_allclose(robust.pval, pvalue, rtol=1e-10)
    # The previous reference distributions were wrong when nreg > 1
    wald = robust.stat * nreg * (nperiod - 1) / (nperiod - nreg)
    if nreg > 1:
        assert robust.pval > stats.f(nreg, nperiod - 1).sf(wald / nreg)
        assert robust.pval > stats.f(nreg, int(res.nobs) - nreg - 1).sf(wald / nreg)


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust"])
@pytest.mark.parametrize(("nperiod", "nreg"), [(10, 3), (10, 5), (20, 3), (20, 5)])
def test_robust_f_statistic_matches_statsmodels(nperiod, nreg, cov_type):
    # The standard covariance of the average of the T estimates gives the
    # one-sample Hotelling test of statsmodels for the slopes
    y, x = fama_macbeth_panel(nentity=40, nperiod=nperiod, nreg=nreg, seed=nperiod)
    res = FamaMacBeth(y, x).fit(cov_type=cov_type)
    slopes = cross_sections(y, x).iloc[:, 1:].to_numpy()
    expected = hotelling_test(slopes, mean_null=np.zeros(nreg))
    assert expected.df == (nreg, nperiod - nreg)
    robust = res.f_statistic_robust
    assert_allclose(robust.stat, expected.statistic, rtol=1e-10)
    assert_allclose(robust.pval, expected.pvalue, rtol=1e-10)
    # The Wald statistic is the T-squared statistic of statsmodels
    wald = robust.stat * nreg * (nperiod - 1) / (nperiod - nreg)
    assert_allclose(wald, expected.t2, rtol=1e-10)


@pytest.mark.parametrize("nperiod", [4, 12, 30])
@pytest.mark.parametrize("cov_type", ["unadjusted", "kernel"])
def test_robust_f_statistic_one_regressor_is_the_t_test(nperiod, cov_type):
    # With a single parameter besides the constant F(1, T - 1) is t(T - 1)
    # squared, and the statistic is the square of the t-statistic
    y, x = fama_macbeth_panel(nentity=40, nperiod=nperiod, nreg=1, seed=13 + nperiod)
    bandwidth = 1 if cov_type == "kernel" else None
    res = FamaMacBeth(y, x).fit(cov_type=cov_type, bandwidth=bandwidth)
    robust = res.f_statistic_robust
    assert robust.df == 1
    assert robust.df_denom == nperiod - 1
    assert_allclose(robust.stat, res.tstats["x0"] ** 2, rtol=1e-10)
    assert_allclose(robust.pval, res.pvalues["x0"], rtol=1e-10)
    assert_allclose(
        robust.pval, 2 * stats.t(nperiod - 1).sf(abs(res.tstats["x0"])), rtol=1e-10
    )


def test_robust_f_statistic_degrees_of_freedom():
    # The classical F-statistic of the pooled regression keeps nobs - nvar
    # denominator degrees of freedom, and the robust one uses T - k
    y, x = fama_macbeth_panel(nentity=40, nperiod=15, nreg=2, seed=7)
    res = FamaMacBeth(y, x).fit()
    robust = res.f_statistic_robust
    assert robust.df == 2
    assert robust.df_denom == 13
    assert res.df_resid == 14
    slopes = res.params.iloc[1:].to_numpy()
    cov = res.cov.iloc[1:, 1:].to_numpy()
    wald = float(slopes @ np.linalg.solve(cov, slopes))
    assert_allclose(robust.stat, 13 / (2 * 14) * wald, rtol=1e-10)
    assert_allclose(robust.pval, stats.f(2, 13).sf(13 / 28 * wald), rtol=1e-10)

    # The classical F-statistic is that of the pooled regression that uses the
    # average of the estimates, with nobs - nvar degrees of freedom
    resid = y.to_numpy() - x.to_numpy() @ res.params.to_numpy()
    rss = float(resid @ resid)
    tss = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    classical = ((tss - rss) / 2) / (rss / 597)
    assert res.f_statistic.df_denom == 597
    assert_allclose(res.f_statistic.stat, classical, rtol=1e-10)


@pytest.mark.parametrize("bandwidth", [0, 2])
def test_robust_f_statistic_dropped_periods(bandwidth):
    # T is the number of periods that are estimated, 21 of the 25 periods, and
    # the dropped periods are not part of the statistic or of its degrees of
    # freedom
    y, x = dropped_periods_panel()
    # The count of the dropped periods, and not the words that describe the
    # regressions, which are time-series or cross-sectional in different versions
    with pytest.warns(MissingValueWarning, match=r"4 of the .*regressions"):
        mod = FamaMacBeth(y, x)
    cov_type = "kernel" if bandwidth else "unadjusted"
    res = mod.fit(cov_type=cov_type, kernel="bartlett", bandwidth=bandwidth)
    estimates = cross_sections(y, x)
    assert estimates.shape[0] == 21
    _, stat, pvalue = hotelling_by_hand(estimates, bandwidth=bandwidth)
    robust = res.f_statistic_robust
    assert (robust.df, robust.df_denom) == (2, 19)
    assert_allclose(robust.stat, stat, rtol=1e-10)
    assert_allclose(robust.pval, pvalue, rtol=1e-10)
    # which is not the statistic that counts the 25 periods of the panel
    wald = robust.stat * 2 * 20 / 19
    wrong = (25 - 2) / (2 * (25 - 1)) * wald
    assert not np.isclose(wrong, robust.stat, rtol=1e-3)


def test_robust_f_statistic_weighted():
    y, x = fama_macbeth_panel(nentity=40, nperiod=12, nreg=3, seed=29)
    rg = np.random.default_rng(31)
    weights = pd.Series(rg.uniform(0.5, 2.0, size=y.shape[0]), index=y.index)
    res = FamaMacBeth(y, x, weights=weights).fit()
    _, stat, pvalue = hotelling_by_hand(cross_sections(y, x, weights))
    assert_allclose(res.f_statistic_robust.stat, stat, rtol=1e-10)
    assert_allclose(res.f_statistic_robust.pval, pvalue, rtol=1e-10)


@pytest.mark.parametrize("nreg", [1, 3])
def test_not_debiased_uses_the_chi_squared_distribution(nreg):
    # Without the small sample adjustment the statistic is the Wald statistic
    # and the distribution is chi-squared. This is not Hotelling's reference
    y, x = fama_macbeth_panel(nentity=40, nperiod=10, nreg=nreg, seed=17 + nreg)
    res = FamaMacBeth(y, x).fit(debiased=False)
    wald, _, pvalue = hotelling_by_hand(cross_sections(y, x), debiased=False)
    robust = res.f_statistic_robust
    assert robust.dist_name == f"chi2({nreg})"
    assert robust.df_denom is None
    assert_allclose(robust.stat, wald, rtol=1e-10)
    assert_allclose(robust.pval, pvalue, rtol=1e-10)


def test_not_debiased_uses_normal_distribution():
    y, x = fama_macbeth_panel(nentity=40, nperiod=8, nreg=1, seed=17)
    res = FamaMacBeth(y, x).fit(debiased=False)
    tstats = res.params / res.std_errors
    assert_allclose(res.pvalues, 2 * stats.norm.sf(np.abs(tstats)), rtol=1e-10)
    crit = stats.norm.ppf(0.975)
    assert_allclose(
        res.conf_int()["upper"], res.params + crit * res.std_errors, rtol=1e-10
    )
    # F-test of the slopes is a chi-squared test
    assert res.f_statistic_robust.dist_name == "chi2(1)"


@pytest.mark.parametrize("debiased", [True, False])
@pytest.mark.parametrize(
    ("nperiod", "nreg", "use_constant"),
    [(3, 3, True), (2, 3, True), (3, 3, False), (2, 2, False)],
)
def test_robust_f_statistic_needs_more_periods_than_parameters(
    nperiod, nreg, use_constant, debiased
):
    # With T periods the covariance of the average of the estimates has rank at
    # most T - 1, and the test needs T > k, with and without the small sample
    # adjustment. The covariance is singular otherwise, and the statistic was the
    # inverse of it, which was numerical noise, or a LinAlgError
    y, x = fama_macbeth_panel(nentity=20, nperiod=nperiod, nreg=nreg, seed=3)
    if not use_constant:
        x = x.drop(columns="const")
    # There are fewer periods than parameters in some of the models
    if nperiod < nreg + use_constant:
        warns = pytest.warns(
            InferenceUnavailableWarning, match="cross-sectional regressions"
        )
    else:
        warns = nullcontext()
    with warns:
        res = FamaMacBeth(y, x).fit(debiased=debiased)
    robust = res.f_statistic_robust
    assert isinstance(robust, InvalidTestStatistic)
    assert np.isnan(robust.stat)
    assert np.isnan(robust.pval)
    assert "more periods than tested parameters" in str(robust)
    assert f"The {nreg} tested parameters" in str(robust)
    assert f"of {nperiod} periods" in str(robust)
    assert isinstance(str(res.summary), str)


@pytest.mark.parametrize("debiased", [True, False])
def test_robust_f_statistic_with_one_more_period_than_parameters(debiased):
    y, x = fama_macbeth_panel(nentity=20, nperiod=4, nreg=3, seed=3)
    res = FamaMacBeth(y, x).fit(debiased=debiased)
    robust = res.f_statistic_robust
    assert robust.df == 3
    assert np.isfinite(robust.stat)
    assert np.isfinite(robust.pval)
    # F(3, 1) with the small sample adjustment, and chi2(3) without
    assert robust.dist_name == ("F(3,1)" if debiased else "chi2(3)")


def test_robust_f_statistic_only_constant():
    y, x = fama_macbeth_panel(nentity=40, nperiod=10, nreg=1, seed=5)
    res = FamaMacBeth(y, x[["const"]]).fit()
    assert isinstance(res.f_statistic_robust, InvalidTestStatistic)
    assert "only a constant" in str(res.f_statistic_robust)


def test_robust_f_statistic_without_a_constant():
    # Every parameter is tested, and the constant is not in the model
    y, x = fama_macbeth_panel(nentity=40, nperiod=10, nreg=3, seed=9)
    x = x.drop(columns="const")
    res = FamaMacBeth(y, x).fit()
    estimates = cross_sections(y, x)
    mean = estimates.mean().to_numpy()
    cov = np.cov(estimates.to_numpy().T, ddof=1) / 10
    wald = float(mean @ np.linalg.solve(cov, mean))
    stat = (10 - 3) / (3 * 9) * wald
    robust = res.f_statistic_robust
    assert (robust.df, robust.df_denom) == (3, 7)
    assert_allclose(robust.stat, stat, rtol=1e-10)
    assert_allclose(robust.pval, stats.f(3, 7).sf(stat), rtol=1e-10)


def test_robust_f_statistic_is_in_the_summary():
    y, x = fama_macbeth_panel(nentity=40, nperiod=10, nreg=2, seed=21)
    res = FamaMacBeth(y, x).fit()
    summary = str(res.summary)
    assert "F(2,8)" in summary
    assert pval_format(res.f_statistic_robust.pval) in summary


@pytest.mark.parametrize(
    ("nreg", "nperiod"), [(1, 10), (3, 10), (5, 10), (3, 20), (5, 20)]
)
def test_size_of_the_robust_f_statistic(nreg, nperiod):
    # The T estimates are independent normal vectors with the same distribution,
    # which is the setting of Hotelling's test, and the null hypothesis is true.
    # The test rejects in 5% of the samples at the 5% level. W is then
    # k (T - 1) / (T - k) times an F(k, T - k) variable, so that the share of
    # samples in which the test that compared W / k with F(k, T - 1) rejects is
    # known: 10%, 24% and 11% for k and T of 3 and 10, 5 and 10, and 5 and 20.
    # It is 5% only if there is one tested parameter
    hotelling = stats.f(nreg, nperiod - nreg)
    critical = stats.f(nreg, nperiod - 1).isf(0.05)
    size_t_minus_1 = hotelling.sf((nperiod - nreg) / (nperiod - 1) * critical)
    if nreg > 1:
        assert size_t_minus_1 > 0.07
    rg = np.random.default_rng(1000 * nreg + nperiod)
    reps = 1000
    f_info = panel_model.FInfo(
        np.r_[False, np.ones(nreg, dtype=bool)],
        "Model F-statistic (robust)",
        None,
        False,
    )
    reject, reject_t_minus_1 = 0, 0
    for _ in range(reps):
        estimates = rg.standard_normal((nperiod, nreg + 1))
        # The covariance of the average of the estimates, that FamaMacBeth uses
        cov = np.cov(estimates.T) / nperiod
        test = panel_model._hotelling_f(estimates.mean(0), cov, True, nperiod, f_info)
        wald = test.stat * nreg * (nperiod - 1) / (nperiod - nreg)
        reject += test.pval < 0.05
        reject_t_minus_1 += stats.f(nreg, nperiod - 1).sf(wald / nreg) < 0.05
    # Within five standard errors of the share of 1,000 samples
    for size, share in (
        (0.05, reject / reps),
        (size_t_minus_1, reject_t_minus_1 / reps),
    ):
        assert abs(share - size) < 5 * np.sqrt(size * (1 - size) / reps)


@pytest.mark.parametrize("use_weights", [True, False])
def test_weighted_inference(use_weights):
    y, x = fama_macbeth_panel(nentity=40, nperiod=10, nreg=1, seed=19)
    rg = np.random.default_rng(23)
    weights = pd.Series(rg.uniform(0.5, 2.0, size=y.shape[0]), index=y.index)
    res = FamaMacBeth(y, x, weights=weights if use_weights else None).fit()
    estimates = cross_sections(y, x, weights if use_weights else None)
    params, _, pvalues, interval = by_hand(estimates)
    assert_allclose(res.params, params, rtol=1e-10)
    assert res.df_resid == 9
    assert_allclose(res.pvalues, pvalues, rtol=1e-10)
    assert_allclose(res.conf_int(), interval, rtol=1e-10)


def no_cross_section_data(case):
    """Panels in which none of the cross-sections can be estimated"""
    rg = np.random.default_rng(0)
    nperiod = 24
    if case == "single-entity":
        # One entity, so that every cross-section has a single observation
        index = pd.MultiIndex.from_product([["000001"], range(nperiod)])
        x = pd.DataFrame(rg.standard_normal((nperiod, 3)), index=index)
    elif case == "time-only-regressor":
        # A regressor that is the same for every entity in a period
        index = pd.MultiIndex.from_product([range(20), range(nperiod)])
        market = np.tile(rg.standard_normal(nperiod), 20)
        x = pd.DataFrame({"market": market}, index=index)
    else:
        # Two entities for three regressors
        index = pd.MultiIndex.from_product([range(2), range(nperiod)])
        x = pd.DataFrame(rg.standard_normal((2 * nperiod, 2)), index=index)
    x.insert(0, "const", 1.0)
    y = pd.Series(rg.standard_normal(x.shape[0]), index=index, name="y")
    return y, x


@pytest.mark.parametrize(
    "case", ["single-entity", "time-only-regressor", "fewer-entities"]
)
def test_no_cross_section_can_be_estimated(case):
    # #427: the message had a typo, "anycross-sectional", and spoke of blocks of
    # time-series observations. It has to say what is needed and what to do
    y, x = no_cross_section_data(case)
    with pytest.raises(ValueError, match=r"Model cannot be estimated") as exc:
        FamaMacBeth(y, x)
    message = str(exc.value)
    assert "anycross" not in message
    assert "blocks" not in message
    # Each sentence starts after a space
    assert re.search(r"[a-z]\.[A-Za-z]", message) is None
    assert "one cross-sectional regression for each time period" in message
    assert "at least as many entities as regressors" in message
    assert "regressors must vary across entities within a period" in message
    assert "depends only on time is collinear with the constant" in message
    assert "single entity" in message
    assert "estimate a time-series regression" in message
