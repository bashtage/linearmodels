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

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import FamaMacBeth
from linearmodels.shared.exceptions import (
    InferenceUnavailableWarning,
    MissingValueWarning,
)
from linearmodels.tests.panel._utility import (
    access_attributes,
    assert_frame_similar,
    datatypes,
    generate_data,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
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
        FamaMacBeth(data.y, data.x).fit(cov_type="unknown")


@pytest.mark.smoke
def test_fama_macbeth_kernel_smoke(data):
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="bartlett")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="newey-west")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="parzen")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="qs")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", bandwidth=3)
    res = FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="andrews")
    access_attributes(res)


def test_fitted_effects_residuals(data):
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


@pytest.mark.filterwarnings(
    "always::linearmodels.shared.exceptions.MissingValueWarning"
)
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
    with pytest.warns(MissingValueWarning):
        FamaMacBeth(y.iloc[:11], x.iloc[:11])
    with pytest.warns(InferenceUnavailableWarning):
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
            warnings.simplefilter("ignore")
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
    with pytest.warns(MissingValueWarning, match=r"4 of the time-series regressions"):
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
    res = FamaMacBeth(y, x).fit(cov_type="kernel", kernel=kernel, bandwidth=bandwidth)
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


def test_robust_f_statistic_degrees_of_freedom():
    # The Wald test of the parameters other than the constant is compared with
    # an F distribution with T - 1 denominator degrees of freedom. The classical
    # F-statistic of the pooled regression keeps nobs - nvar
    y, x = fama_macbeth_panel(nentity=40, nperiod=15, nreg=2, seed=7)
    res = FamaMacBeth(y, x).fit()
    robust = res.f_statistic_robust
    assert robust.df == 2
    assert robust.df_denom == 14
    slopes = res.params.iloc[1:].to_numpy()
    cov = res.cov.iloc[1:, 1:].to_numpy()
    wald = float(slopes @ np.linalg.solve(cov, slopes))
    assert_allclose(robust.stat, wald / 2, rtol=1e-10)
    assert_allclose(robust.pval, stats.f(2, 14).sf(wald / 2), rtol=1e-10)

    # The classical F-statistic is that of the pooled regression that uses the
    # average of the estimates, with nobs - nvar degrees of freedom
    resid = y.to_numpy() - x.to_numpy() @ res.params.to_numpy()
    rss = float(resid @ resid)
    tss = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    classical = ((tss - rss) / 2) / (rss / 597)
    assert res.f_statistic.df_denom == 597
    assert_allclose(res.f_statistic.stat, classical, rtol=1e-10)


def test_robust_f_statistic_one_regressor_is_the_t_test():
    # With a single parameter besides the constant F(1, T - 1) is t(T - 1) squared
    y, x = fama_macbeth_panel(nentity=40, nperiod=12, nreg=1, seed=13)
    res = FamaMacBeth(y, x).fit()
    robust = res.f_statistic_robust
    assert robust.df_denom == 11
    assert_allclose(robust.stat, res.tstats["x0"] ** 2, rtol=1e-10)
    assert_allclose(robust.pval, res.pvalues["x0"], rtol=1e-10)


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
        index = pd.MultiIndex.from_product(
            [["000001"], pd.date_range("2010-01-31", periods=nperiod, freq="ME")]
        )
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
